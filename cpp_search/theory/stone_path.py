"""Stone-Royset-Washburn SP1/SPX path optimization by cutting planes.

This module implements the mathematical-programming family in Chapter 4 of
Stone, Royset & Washburn (2016), rather than the Dell H1/H2 heuristics in
``path_constrained``.

``SP1`` is the homogeneous-searcher, single-target model (4.23)-(4.28).
``SPX`` extends it to heterogeneous searchers and targets, cell occupancy
limits, and incompatible moves (4.48)-(4.56).  The implementation uses a
binary, per-searcher disaggregation of Stone's integer arc-flow variables;
the feasible paths and aggregate cell-time hazards are identical.

The nonlinear convex nondetection functions are solved with the cutting-plane
algorithm of Sect. 4.3.1.  Every master problem is a MILP.  Its tangent cuts
give a valid lower bound on nondetection, while an integer path returned by the
master gives an upper bound.  The solver therefore reports an explicit
optimality certificate instead of silently treating a heuristic as exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from math import inf, isfinite
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TypeAlias

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

try:  # Optional persistent backend used by the 240-cell operational solve.
    import highspy
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    highspy = None

try:  # Optional exact-MILP backend for the high-resolution certificate.
    from pyscipopt import Model as ScipModel
except ImportError:  # pragma: no cover - exercised when the optional extra is absent
    ScipModel = None

from cpp_search.theory import transitions as transition_ops
from cpp_search.theory.path_constrained import (
    PathConstrainedProblem,
    exact_best_paths,
    expected_detections,
    path_detection_probability,
)

__all__ = [
    "StoneCuttingPlaneResult",
    "StoneMove",
    "StoneTargetModel",
    "stone_sp1_cutting_plane",
    "stone_spx_cutting_plane",
]


StoneMove: TypeAlias = tuple[int, int, int, int]
"""``(searcher, time, from_state, to_state)`` identifying one move."""


@dataclass(frozen=True, slots=True)
class StoneTargetModel:
    """One SPX target and its target-specific detection effectiveness.

    ``hazard_multiplier`` multiplies each searcher's base arc hazard.  It may
    be a scalar or an array indexed ``[searcher, time, from_state, to_state]``.
    This represents Stone's target-dependent ``alpha[l,k](j',j,t)`` while
    reusing the repository's existing searcher sensor model.
    """

    initial_mass: np.ndarray
    transitions: np.ndarray
    hazard_multiplier: float | np.ndarray = 1.0

    @classmethod
    def from_problem(cls, problem: PathConstrainedProblem) -> "StoneTargetModel":
        return cls(problem.initial_mass, problem.transitions)


@dataclass(frozen=True, slots=True)
class StoneCuttingPlaneResult:
    """Integer path and the SP1/SPX nondetection optimality certificate."""

    method: str
    paths: tuple[tuple[int, ...], ...]
    probability_of_detection: float
    expected_detections: float
    evaluations: int
    lower_bound_nondetection: float
    upper_bound_nondetection: float
    relative_optimality_gap: float
    converged: bool
    iterations: int
    target_nondetection_probabilities: tuple[float, ...]
    fallback_used: bool = False
    fallback_reason: str | None = None
    #: 하드 제약(분리충돌·점유·반대방향 swap)을 어겨서 버린 warm start 후보 수.
    #: H1/H2 는 예약이 모든 첫 수를 막으면 분리를 soft 로 풀고 진행하는데,
    #: SPX 실행가능집합에는 그 경로가 없다. 버리고 master 가 스스로 찾는다.
    discarded_warm_starts: int = 0
    #: 반복별 (iteration, 인증하한, 최선상한, 상대갭). Ch2 의 수렴성 근거이자
    #: "갭이 정체된 것인가, 반복이 모자란 것인가"를 구분하는 유일한 증거다.
    bound_history: tuple[tuple[int, float, float, float], ...] = ()
    #: 반복별 (iteration, 단계, 상태, 소요초). 단계는 "relaxation" 또는
    #: "master". 상태가 ``optimal`` 이면 그 단계에서 시간을 더 줘도 경계가
    #: 안 올라간다 — 절단(반복)이 병목이다. ``limit`` 이면 반대로 시간이
    #: 병목이다. 이 구분을 궤적의 '정지 구간'으로 **추론**하고 있었는데,
    #: 그건 인과가 아니라 정황이었다.
    master_status_history: tuple[tuple[int, str, str, float], ...] = ()
    #: 최종 계획이 어디서 왔는가. ``"warm-start"`` 면 master 가 휴리스틱을
    #: 한 번도 못 이긴 것이다 — 그때 "SPX 가 최적화했다"고 말하면 안 된다.
    incumbent_source: str = "unknown"
    #: warm start 휴리스틱이 낸 최선 비탐지확률. master 개선폭을 재는 기준선.
    warm_start_nondetection: float = float("nan")

    @property
    def relaxation_gap(self) -> float:
        return self.expected_detections - self.probability_of_detection


@dataclass(frozen=True, slots=True)
class _Arc:
    searcher: int
    time: int
    source: int
    destination: int


def stone_sp1_cutting_plane(
    problem: PathConstrainedProblem,
    *,
    relative_tolerance: float = 1e-8,
    max_iterations: int = 100,
    mip_relative_gap: float = 0.0,
    exact_fallback_max_evaluations: int = 200_000,
) -> StoneCuttingPlaneResult:
    """Solve Stone model SP1 for homogeneous searchers and one Markov target.

    The original SP1 uses integer arc counts.  We use one binary flow per
    searcher, an exact disaggregation that also makes extraction of individual
    paths unambiguous.
    """

    _validate_sp1_homogeneity(problem)
    try:
        result = stone_spx_cutting_plane(
            problem,
            targets=(StoneTargetModel.from_problem(problem),),
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
            mip_relative_gap=mip_relative_gap,
        )
    except RuntimeError as error:
        # Some small, sparse SP1 masters trigger a HiGHS incumbent-transform
        # error after several valid outer-approximation iterations.  SP1 is
        # used as a reduced benchmark, so retain a mathematically exact and
        # explicit fallback rather than discarding the feasible experiment.
        exact = exact_best_paths(
            problem,
            max_evaluations=exact_fallback_max_evaluations,
        )
        nondetection = 1.0 - exact.probability_of_detection
        return StoneCuttingPlaneResult(
            method="stone-sp1-cutting-plane+exact-enumeration-fallback",
            paths=exact.paths,
            probability_of_detection=exact.probability_of_detection,
            expected_detections=exact.expected_detections,
            evaluations=exact.evaluations,
            lower_bound_nondetection=nondetection,
            upper_bound_nondetection=nondetection,
            relative_optimality_gap=0.0,
            converged=True,
            iterations=0,
            target_nondetection_probabilities=(nondetection,),
            fallback_used=True,
            fallback_reason=str(error),
        )
    return StoneCuttingPlaneResult(
        method="stone-sp1-cutting-plane",
        paths=result.paths,
        probability_of_detection=result.probability_of_detection,
        expected_detections=result.expected_detections,
        evaluations=result.evaluations,
        lower_bound_nondetection=result.lower_bound_nondetection,
        upper_bound_nondetection=result.upper_bound_nondetection,
        relative_optimality_gap=result.relative_optimality_gap,
        converged=result.converged,
        iterations=result.iterations,
        target_nondetection_probabilities=result.target_nondetection_probabilities,
        fallback_used=result.fallback_used,
        fallback_reason=result.fallback_reason,
    )


def stone_spx_cutting_plane(
    problem: PathConstrainedProblem,
    *,
    targets: tuple[StoneTargetModel, ...] | None = None,
    occupancy_limits: int | np.ndarray | None = None,
    incompatible_moves: tuple[tuple[StoneMove, StoneMove], ...] = (),
    relative_tolerance: float = 1e-8,
    max_iterations: int = 100,
    mip_relative_gap: float = 0.0,
    aggregate_identical_searchers: bool = False,
    destination_conflicts: tuple[tuple[int, int], ...] = (),
    forbid_opposing_edge_swaps: bool = False,
    master_time_limit_s: float | None = None,
    initial_paths: tuple[tuple[int, ...], ...] | None = None,
    initial_path_candidates: tuple[tuple[tuple[int, ...], ...], ...] = (),
    persistent_master: bool = False,
    continuous_relaxation_iterations: int = 0,
    exact_survival_milp: bool = False,
    exact_time_limit_s: float | None = None,
    exact_backend: str = "highs",
    master_backend: str = "highs",
    local_improvement_passes: int = 0,
) -> StoneCuttingPlaneResult:
    """Solve Stone model SPX with the Chapter 4 cutting-plane algorithm.

    The objective is the largest target nondetection probability, as in
    equation (4.49).  ``occupancy_limits`` implements (4.53), and each pair in
    ``incompatible_moves`` implements one instance of (4.54).
    """

    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance cannot be negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if mip_relative_gap < 0.0:
        raise ValueError("mip_relative_gap cannot be negative")
    if master_time_limit_s is not None and master_time_limit_s <= 0.0:
        raise ValueError("master_time_limit_s must be positive when provided")
    if persistent_master and highspy is None:
        raise RuntimeError(
            "persistent_master requires the optional highspy dependency"
        )
    if continuous_relaxation_iterations < 0:
        raise ValueError("continuous_relaxation_iterations cannot be negative")
    if exact_survival_milp and highspy is None:
        raise RuntimeError("exact_survival_milp requires the highspy dependency")
    if exact_time_limit_s is not None and exact_time_limit_s <= 0.0:
        raise ValueError("exact_time_limit_s must be positive when provided")
    if exact_backend not in {"highs", "scip"}:
        raise ValueError("exact_backend must be 'highs' or 'scip'")
    if master_backend not in {"highs", "scip"}:
        raise ValueError("master_backend must be 'highs' or 'scip'")
    if exact_survival_milp and exact_backend == "scip" and ScipModel is None:
        raise RuntimeError("exact_backend='scip' requires pyscipopt")
    if master_backend == "scip" and ScipModel is None:
        raise RuntimeError("master_backend='scip' requires pyscipopt")
    if local_improvement_passes < 0:
        raise ValueError("local_improvement_passes cannot be negative")

    target_models = targets or (StoneTargetModel.from_problem(problem),)
    if not target_models:
        raise ValueError("SPX requires at least one target")
    multipliers = tuple(
        _validated_target(target, problem) for target in target_models
    )
    occupancy = _occupancy_limits(occupancy_limits, problem)

    groups = _searcher_groups(problem, aggregate_identical_searchers)
    for multiplier in multipliers:
        for members in groups:
            reference = multiplier[members[0]]
            if any(
                not np.array_equal(multiplier[member], reference)
                for member in members[1:]
            ):
                raise ValueError(
                    "aggregated searchers must have identical target hazard "
                    "multipliers"
                )
    arcs = _build_group_arcs(problem, groups)
    arc_index = {
        (arc.searcher, arc.time, arc.source, arc.destination): index
        for index, arc in enumerate(arcs)
    }
    target_count = len(target_models)
    state_count = problem.state_count
    time_count = problem.time_count
    arc_count = len(arcs)
    y_count = target_count * time_count * state_count
    eta_index = arc_count + y_count
    variable_count = eta_index + 1

    def y_index(target: int, time: int, state: int) -> int:
        return arc_count + (target * time_count + time) * state_count + state

    lower = np.zeros(variable_count, dtype=float)
    upper = np.full(variable_count, inf, dtype=float)
    for index, arc in enumerate(arcs):
        # A destination occupancy row already implies this bound. Stating it
        # explicitly lets MILP solvers recognize occupancy-one arc flows as
        # binary variables instead of general integers with an inferred cap.
        upper[index] = min(
            len(groups[arc.searcher]),
            int(occupancy[arc.time, arc.destination]),
        )
    upper[eta_index] = 1.0
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:arc_count] = 1
    objective = np.zeros(variable_count, dtype=float)
    objective[eta_index] = 1.0

    base_rows: list[dict[int, float]] = []
    base_lb: list[float] = []
    base_ub: list[float] = []

    def add_row(coefficients: dict[int, float], lb: float, ub: float) -> None:
        base_rows.append(coefficients)
        base_lb.append(lb)
        base_ub.append(ub)

    by_group_time: dict[tuple[int, int], list[int]] = {}
    incoming: dict[tuple[int, int, int], list[int]] = {}
    outgoing: dict[tuple[int, int, int], list[int]] = {}
    by_time_destination: dict[tuple[int, int], list[int]] = {}
    by_time_swept: dict[tuple[int, int], list[int]] = {}
    arc_swept: list[Mapping[int, float]] = []
    by_time_edge: dict[tuple[int, int, int], list[int]] = {}
    for index, arc in enumerate(arcs):
        by_group_time.setdefault((arc.searcher, arc.time), []).append(index)
        incoming.setdefault(
            (arc.searcher, arc.time, arc.destination), []
        ).append(index)
        outgoing.setdefault((arc.searcher, arc.time, arc.source), []).append(index)
        by_time_destination.setdefault((arc.time, arc.destination), []).append(index)
        # 소인은 도착셀이 아니라 지나간 셀들에 쌓인다. 점유·충돌 제약은
        # 여전히 도착셀 기준이므로 두 색인을 나눠 둔다.
        swept = problem.searchers[groups[arc.searcher][0]].swept_cells(
            arc.time, arc.source, arc.destination
        )
        arc_swept.append(swept)
        for cell in swept:
            by_time_swept.setdefault((arc.time, cell), []).append(index)
        if arc.source != arc.destination:
            edge = (min(arc.source, arc.destination), max(arc.source, arc.destination))
            by_time_edge.setdefault((arc.time, *edge), []).append(index)

    # Explicit hazard bounds tighten the continuous relaxation. The occupancy
    # limit times the strongest feasible single-searcher hazard is a safe upper
    # bound even when several homogeneous flows are aggregated in one integer.
    for target_index, multiplier in enumerate(multipliers):
        for time in range(time_count):
            for state in range(state_count):
                # 점유한도로 묶으면 안 된다. 셀에 **머무를** 수 있는 수는
                # 제한되지만 **지나갈** 수 있는 수는 제한이 없다. 점유로
                # 묶었더니 상한이 최적해를 잘라내 하한이 무효가 됐다
                # (완전열거 0.2251 을 0.3141 로 '인증'했다).
                # 그룹마다 (그룹 크기 x 그 그룹이 이 셀에 줄 수 있는 최대)
                # 를 더한 값이 유효한 상한이다.
                per_group = [0.0] * len(groups)
                for index in by_time_swept.get((time, state), ()):
                    arc = arcs[index]
                    members = groups[arc.searcher]
                    per_group[arc.searcher] = max(
                        per_group[arc.searcher],
                        arc_swept[index].get(state, 0.0)
                        * multiplier[
                            members[0], time, arc.source, arc.destination
                        ],
                    )
                bound = sum(
                    len(groups[group]) * value
                    for group, value in enumerate(per_group)
                )
                if all(
                    searcher.swept_hazard is None for searcher in problem.searchers
                ):
                    # 도착셀에만 쌓는 기존 모형에서는 점유한도가 더 강하다.
                    bound = min(
                        bound, float(occupancy[time, state]) * max(per_group)
                    )
                upper[y_index(target_index, time, state)] = bound

    # Exactly group_size move/search actions per homogeneous class and period.
    for searcher, members in enumerate(groups):
        for time in range(time_count):
            add_row(
                {index: 1.0 for index in by_group_time[(searcher, time)]},
                float(len(members)),
                float(len(members)),
            )

    # Time-expanded flow continuity, equivalent to (4.50).
    for searcher in range(len(groups)):
        for time in range(time_count - 1):
            for state in range(state_count):
                coefficients: dict[int, float] = {}
                for index in incoming.get((searcher, time, state), ()):
                    coefficients[index] = 1.0
                for index in outgoing.get((searcher, time + 1, state), ()):
                    coefficients[index] = coefficients.get(index, 0.0) - 1.0
                add_row(coefficients, 0.0, 0.0)

    # Target-specific aggregate cell-time hazard, equation (4.52).
    for target_index in range(target_count):
        multiplier = multipliers[target_index]
        for time in range(time_count):
            for state in range(state_count):
                coefficients = {y_index(target_index, time, state): 1.0}
                for index in by_time_swept.get((time, state), ()):
                    arc = arcs[index]
                    members = groups[arc.searcher]
                    hazard = arc_swept[index].get(state, 0.0) * multiplier[
                        members[0], time, arc.source, arc.destination
                    ]
                    if hazard:
                        coefficients[index] = coefficients.get(index, 0.0) - hazard
                add_row(coefficients, 0.0, 0.0)

    # Cell occupancy deconfliction, equation (4.53).
    for time in range(time_count):
        for state in range(state_count):
            add_row(
                {index: 1.0 for index in by_time_destination.get((time, state), ())},
                -inf,
                float(occupancy[time, state]),
            )

    # Footprint and opposing-edge conflicts are separated lazily below. Their
    # full 240-cell constraint family makes the first MILP much harder even
    # though an incumbent violates only a handful of pairs. Adding every
    # observed violation retains the exact feasible set and certificate.
    destination_conflict_set: set[tuple[int, int]] = set()
    for left_state, right_state in destination_conflicts:
        if not (0 <= left_state < state_count and 0 <= right_state < state_count):
            raise ValueError("destination conflict state is outside the state space")
        if left_state == right_state:
            continue
        destination_conflict_set.add(
            (min(left_state, right_state), max(left_state, right_state))
        )
    separated_destination_rows: set[tuple[int, int, int]] = set()
    separated_edge_rows: set[tuple[int, int, int]] = set()

    # General incompatible-move deconfliction, equation (4.54).
    for left_move, right_move in incompatible_moves:
        if aggregate_identical_searchers:
            raise ValueError(
                "explicit per-searcher incompatible moves cannot be combined with "
                "aggregate_identical_searchers"
            )
        _validate_move_shape(left_move, problem)
        _validate_move_shape(right_move, problem)
        # 두 arc 중 하나라도 그 시점에 **도달 불가**하면 (4.54) 행은 공허하다.
        # 비집계 SPX 에서 arc 변수 상한은 1 이므로 남은 한 변수만으로 합이
        # 이미 1 을 넘지 못한다. 예전에는 이 경우를 KeyError -> ValueError 로
        # 올려서, 모든 셀을 source 로 열거하는 호출자(운용격자 반대방향 edge
        # swap 금지)가 SPX 를 아예 못 돌렸다.
        if left_move not in arc_index or right_move not in arc_index:
            continue
        left_index = arc_index[left_move]
        right_index = arc_index[right_move]
        coefficients = {left_index: 1.0}
        coefficients[right_index] = coefficients.get(right_index, 0.0) + 1.0
        add_row(coefficients, -inf, 1.0)

    cuts: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    zero_hazard = np.zeros((target_count, time_count, state_count), dtype=float)
    for target_index, target in enumerate(target_models):
        value, gradient = _nondetection_value_gradient(target, zero_hazard[target_index])
        cuts.append((target_index, value, gradient, zero_hazard[target_index].copy()))

    # A feasible plan can have exactly PND=1 when all target mass is outside
    # every reachable detection cell.  Start above the probability range so
    # the first feasible incumbent is retained even in that boundary case.
    best_upper = inf
    bound_history: list[tuple[int, float, float, float]] = []
    status_history: list[tuple[int, str, str, float]] = []
    incumbent_source = "none"
    warm_start_nondetection = inf
    best_paths: tuple[tuple[int, ...], ...] | None = None
    best_target_values: tuple[float, ...] = ()
    candidate_paths = list(initial_path_candidates)
    if initial_paths is not None:
        candidate_paths.insert(0, initial_paths)
    seen_candidates: set[tuple[tuple[int, ...], ...]] = set()
    discarded_warm_starts = 0
    for candidate in candidate_paths:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            _validate_initial_paths(
                problem,
                candidate,
                occupancy,
                destination_conflict_set,
                forbid_opposing_edge_swaps,
            )
        except ValueError:
            # 실행 불가 warm start 는 예외가 아니라 정보다. 예전에는 여기서
            # 전체 solve 가 죽어서 20 s 슬라이스 운용격자가 아예 안 돌았다.
            discarded_warm_starts += 1
            continue
        candidate = _locally_improve_paths(
            problem,
            candidate,
            target_models,
            multipliers,
            occupancy,
            destination_conflict_set,
            forbid_opposing_edge_swaps,
            passes=local_improvement_passes,
        )
        initial_values = []
        for target_index, (target, multiplier) in enumerate(
            zip(target_models, multipliers)
        ):
            initial_hazard = _target_path_hazard(
                problem, candidate, multiplier
            )
            value, gradient = _nondetection_value_gradient(
                target, initial_hazard
            )
            cuts.append(
                (target_index, value, gradient, initial_hazard.copy())
            )
            initial_values.append(value)
        candidate_values = tuple(initial_values)
        candidate_upper = max(candidate_values)
        if candidate_upper < best_upper:
            best_paths = candidate
            best_target_values = candidate_values
            best_upper = candidate_upper
            incumbent_source = "warm-start"
    # 휴리스틱이 낸 최선값을 기준선으로 남긴다. master 가 이걸 얼마나
    # 개선했는지가 곧 SPX 를 쓰는 값어치다.
    warm_start_nondetection = best_upper
    if best_paths is not None:
        upper[eta_index] = min(upper[eta_index], best_upper)

    certified_lower = 0.0
    converged = False
    iteration = 0
    fallback_used = False
    fallback_reason: str | None = None

    def cut_row(cut):
        target_index, value, gradient, point = cut
        # f(point) + grad @ (Y - point) <= eta.
        coefficients = {eta_index: -1.0}
        flat_gradient = gradient.ravel()
        for offset, coefficient in enumerate(flat_gradient):
            if coefficient:
                coefficients[
                    arc_count + target_index * time_count * state_count + offset
                ] = float(coefficient)
        approach = float(value - np.dot(flat_gradient, point.ravel()))
        return coefficients, -inf, -approach

    def master_constraints(active_cuts):
        rows = list(base_rows)
        row_lb = list(base_lb)
        row_ub = list(base_ub)
        for cut in active_cuts:
            coefficients, lb, ub = cut_row(cut)
            rows.append(coefficients)
            row_lb.append(lb)
            row_ub.append(ub)
        matrix = _sparse_rows(rows, variable_count)
        return LinearConstraint(matrix, row_lb, row_ub)

    conflict_cliques = _maximal_conflict_cliques(
        state_count, destination_conflict_set
    )
    relaxation_clique_rows: list[dict[int, float]] = []
    relaxation_clique_keys: set[tuple[int, tuple[int, ...]]] = set()

    def relaxation_constraints(active_cuts):
        """Build the LP relaxation with the separated conflict-clique cuts."""

        rows = list(base_rows)
        row_lb = list(base_lb)
        row_ub = list(base_ub)
        rows.extend(relaxation_clique_rows)
        row_lb.extend([-inf] * len(relaxation_clique_rows))
        row_ub.extend([1.0] * len(relaxation_clique_rows))
        for cut in active_cuts:
            coefficients, lb, ub = cut_row(cut)
            rows.append(coefficients)
            row_lb.append(lb)
            row_ub.append(ub)
        matrix = _sparse_rows(rows, variable_count)
        return LinearConstraint(matrix, row_lb, row_ub)

    # Converge the convex continuous relaxation first. Every LP master value is
    # a rigorous lower bound for the integer SPX problem, and its fractional
    # hazard supplies globally valid tangents that strengthen later MILPs.
    for relaxation_round in range(continuous_relaxation_iterations):
        relaxation_started = perf_counter()
        # 이 단계도 시간 한도를 받아야 한다. 예전에는 무한정이었고, 셀 = 탐지폭
        # 격자에서 24 시간을 넘겨도 반환되지 않았다. master 만 묶어 두면 전체
        # 실행시간의 상한이 성립하지 않는다.
        relaxation_options: dict[str, object] = {"presolve": True}
        if master_time_limit_s is not None:
            relaxation_options["time_limit"] = float(master_time_limit_s)
        relaxed = milp(
            objective,
            integrality=np.zeros(variable_count, dtype=np.uint8),
            bounds=Bounds(lower, upper),
            constraints=relaxation_constraints(cuts),
            options=relaxation_options,
        )
        relaxation_status = _milp_status(relaxed)
        if relaxed.x is None or relaxed.fun is None:
            if relaxation_status == "limit":
                # 한도 안에 아무 점도 얻지 못했다. 완화는 포기하고 정수
                # master 로 넘어간다 — 하계는 갱신하지 않는다.
                status_history.append(
                    (
                        -(relaxation_round + 1),
                        "relaxation",
                        relaxation_status,
                        perf_counter() - relaxation_started,
                    )
                )
                break
            raise RuntimeError(
                f"Stone continuous relaxation failed: {relaxed.message}"
            )
        status_history.append(
            (
                -(relaxation_round + 1),
                "relaxation",
                relaxation_status,
                perf_counter() - relaxation_started,
            )
        )
        # **최적으로 풀린 LP 의 목적값만** 하계가 된다. 한도에 걸린 LP 가 낸
        # 값은 실행가능점의 목적값이므로 최소화 문제에서 상계이고, 그것을
        # 하계 자리에 넣으면 L <= OPT 가 깨져 인증 전체가 무효가 된다.
        # 반면 아래에서 만드는 접평면은 평가점이 어디든 유효하므로 그대로 쓴다.
        if relaxation_status == "optimal":
            certified_lower = max(certified_lower, float(relaxed.fun))
        relaxed_arc_values = np.asarray(relaxed.x[:arc_count])
        violations = []
        for time in range(time_count):
            destination_values = np.zeros(state_count, dtype=float)
            for state in range(state_count):
                destination_values[state] = sum(
                    relaxed_arc_values[index]
                    for index in by_time_destination.get((time, state), ())
                )
            for clique in conflict_cliques:
                key = (time, clique)
                if key in relaxation_clique_keys:
                    continue
                excess = float(destination_values[list(clique)].sum() - 1.0)
                if excess > 1e-7:
                    violations.append((excess, key))
        # Add only the strongest violations. Rebuilding thousands of inactive
        # clique rows at every OA round is much slower than this separation.
        for _, key in sorted(violations, reverse=True)[:50]:
            time, clique = key
            indices = [
                index
                for state in clique
                for index in by_time_destination.get((time, state), ())
            ]
            relaxation_clique_rows.append({index: 1.0 for index in indices})
            relaxation_clique_keys.add(key)
        relaxed_hazard = np.asarray(relaxed.x[arc_count:eta_index]).reshape(
            target_count, time_count, state_count
        )
        relaxed_values = []
        for target_index, target in enumerate(target_models):
            value, gradient = _nondetection_value_gradient(
                target, relaxed_hazard[target_index]
            )
            relaxed_values.append(value)
            cuts.append(
                (
                    target_index,
                    value,
                    gradient,
                    relaxed_hazard[target_index].copy(),
                )
            )
        if relaxation_status != "optimal":
            # 한도에 걸렸다면 다음 라운드도 걸린다. 접평면만 챙기고 나간다.
            break
        relaxed_upper = max(relaxed_values)
        relaxed_gap = max(0.0, relaxed_upper - certified_lower) / max(
            abs(relaxed_upper), 1e-15
        )
        if relaxed_gap <= relative_tolerance:
            break

    # The separated clique inequalities are valid for the integer SPX model
    # as well. Carry them into branch-and-bound so the master does not spend
    # time rediscovering the same fractional destination conflicts.
    for row in relaxation_clique_rows:
        add_row(row, -inf, 1.0)

    persistent = (
        _PersistentHighsMaster(objective, lower, upper, integrality)
        if persistent_master or master_backend == "scip"
        else None
    )
    persistent_base_count = 0
    persistent_cut_count = 0

    for iteration in range(1, max_iterations + 1):
        solve_options = {"mip_rel_gap": mip_relative_gap}
        if master_time_limit_s is not None:
            solve_options["time_limit"] = master_time_limit_s
        if persistent is not None:
            if persistent_base_count < len(base_rows):
                persistent.add_rows(
                    base_rows[persistent_base_count:],
                    base_lb[persistent_base_count:],
                    base_ub[persistent_base_count:],
                )
                persistent_base_count = len(base_rows)
            if persistent_cut_count < len(cuts):
                rows, lbs, ubs = [], [], []
                for cut in cuts[persistent_cut_count:]:
                    row, lb, ub = cut_row(cut)
                    rows.append(row)
                    lbs.append(lb)
                    ubs.append(ub)
                persistent.add_rows(rows, lbs, ubs)
                persistent_cut_count = len(cuts)
            warm_start = (
                _master_solution_from_paths(
                    problem,
                    groups,
                    arcs,
                    arc_index,
                    target_models,
                    multipliers,
                    best_paths,
                    variable_count,
                    arc_count,
                    eta_index,
                )
                if best_paths is not None
                else None
            )
            master_started = perf_counter()
            solved = (
                _solve_scip_master(
                    persistent,
                    mip_relative_gap=mip_relative_gap,
                    time_limit_s=master_time_limit_s,
                    warm_start=warm_start,
                )
                if master_backend == "scip"
                else persistent.solve(
                    mip_relative_gap=mip_relative_gap,
                    time_limit_s=master_time_limit_s,
                    warm_start=warm_start,
                )
            )
        else:
            constraints = master_constraints(cuts)
            master_started = perf_counter()
            solved = milp(
                objective,
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=constraints,
                options=solve_options,
            )
        # master 가 ``optimal`` 로 끝났으면 시간을 더 줘도 이 절단 집합에서는
        # 하한이 안 올라간다 (절단이 병목). ``limit`` 이면 시간이 병목이다.
        status_history.append(
            (
                iteration,
                "master",
                _milp_status(solved),
                perf_counter() - master_started,
            )
        )
        usable_incumbent = (
            getattr(solved, "x", None) is not None
            and getattr(solved, "fun", None) is not None
            and np.isfinite(solved.fun)
        )
        if persistent is None and not solved.success and not usable_incumbent:
            # HiGHS can occasionally fail while transforming a presolved
            # integer incumbent after many nearly coincident tangent cuts.
            # The original master remains valid, so retry it without presolve
            # before reporting a real solve failure.
            solved = milp(
                objective,
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=constraints,
                options={**solve_options, "presolve": False},
            )
            usable_incumbent = (
                getattr(solved, "x", None) is not None
                and getattr(solved, "fun", None) is not None
                and np.isfinite(solved.fun)
            )
        if persistent is None and not solved.success and not usable_incumbent:
            # A group of nearly coincident tangent rows can still trigger a
            # HiGHS incumbent-transform error with presolve disabled. Dropping
            # old tangents does not change validity: every retained tangent is
            # a global lower support of the convex PND function. Keep the
            # initial and newest cut for each target, then continue building
            # the outer approximation from that numerically cleaner master.
            compact_cuts = []
            for target_index in range(target_count):
                target_cuts = [cut for cut in cuts if cut[0] == target_index]
                compact_cuts.append(target_cuts[0])
                if len(target_cuts) > 1:
                    compact_cuts.append(target_cuts[-1])
            compact_constraints = master_constraints(compact_cuts)
            solved = milp(
                objective,
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=compact_constraints,
                options={**solve_options, "presolve": False},
            )
            usable_incumbent = (
                getattr(solved, "x", None) is not None
                and getattr(solved, "fun", None) is not None
                and np.isfinite(solved.fun)
            )
            if solved.success:
                cuts = compact_cuts
                fallback_used = True
                fallback_reason = (
                    "HiGHS solve retry with initial/latest tangent cuts at "
                    f"iteration {iteration}"
                )
        elif not solved.success and usable_incumbent:
            fallback_used = True
            fallback_reason = (
                "HiGHS master reached its time limit with a feasible incumbent "
                f"at cutting-plane iteration {iteration}"
            )
        if (not solved.success and not usable_incumbent) or solved.x is None or solved.fun is None:
            if best_paths is None:
                raise RuntimeError(
                    f"Stone cutting-plane master failed at iteration {iteration}: "
                    f"{solved.message}"
                )
            fallback_used = True
            fallback_reason = (
                f"HiGHS failed at iteration {iteration}; returning the best "
                f"certified incumbent: {solved.message}"
            )
            _record_bounds(bound_history, iteration, certified_lower, best_upper)
            break

        # With a nonzero MILP tolerance ``fun`` is only the incumbent master
        # value.  The dual bound is the certificate that remains a valid lower
        # bound on SPX.
        master_lower = getattr(solved, "mip_dual_bound", None)
        if master_lower is None or not np.isfinite(master_lower):
            # dual bound 가 없으면 하한을 **올리지 않는다**. 최적으로 끝났을
            # 때만 incumbent 값이 곧 하한이다. 시간제한에 걸린 master 의
            # ``fun`` 은 incumbent(상한)라서 하한으로 쓰면 인증이 거짓이 된다.
            master_lower = solved.fun if solved.success else -inf
        certified_lower = max(certified_lower, float(master_lower))
        paths = _extract_group_paths(
            problem,
            groups,
            arcs,
            np.asarray(solved.x[:arc_count]),
        )
        new_conflicts = False
        for time in range(time_count):
            destinations = [path[time] for path in paths]
            for left in range(len(destinations)):
                for right in range(left + 1, len(destinations)):
                    pair = (
                        min(destinations[left], destinations[right]),
                        max(destinations[left], destinations[right]),
                    )
                    row_key = (time, *pair)
                    if (
                        pair in destination_conflict_set
                        and row_key not in separated_destination_rows
                    ):
                        indices = tuple(
                            by_time_destination.get((time, pair[0]), ())
                        ) + tuple(by_time_destination.get((time, pair[1]), ()))
                        add_row({index: 1.0 for index in indices}, -inf, 1.0)
                        separated_destination_rows.add(row_key)
                        new_conflicts = True
            if forbid_opposing_edge_swaps:
                sources = [
                    problem.searchers[index].start_state
                    if time == 0
                    else paths[index][time - 1]
                    for index in range(problem.searcher_count)
                ]
                for left in range(len(paths)):
                    for right in range(left + 1, len(paths)):
                        if (
                            sources[left] == destinations[right]
                            and sources[right] == destinations[left]
                            and sources[left] != sources[right]
                        ):
                            edge = (
                                min(sources[left], destinations[left]),
                                max(sources[left], destinations[left]),
                            )
                            row_key = (time, *edge)
                            if row_key not in separated_edge_rows:
                                indices = by_time_edge.get(row_key, ())
                                add_row(
                                    {index: 1.0 for index in indices}, -inf, 1.0
                                )
                                separated_edge_rows.add(row_key)
                                new_conflicts = True
        if new_conflicts:
            # 분리 제약만 추가하고 재해결하는 반복. 상한은 그대로여도 하한은
            # 이미 갱신됐으므로 궤적에 남겨야 한다. 빠뜨리면 수렴 그림이
            # 실제보다 적은 반복으로 닫힌 것처럼 보인다.
            _record_bounds(bound_history, iteration, certified_lower, best_upper)
            continue
        # master 가 돌려준 **그 점**에 접선을 먼저 놓는다. 국소개선이 점을
        # 옮긴 뒤 옮긴 점에만 절단을 놓으면 master 해는 반증되지 않아 같은
        # 해를 계속 내고 하한이 멈춘다 — passes=2 에서 200 회 반복 후에도
        # gap 6% 로 정체했고, 이 한 줄로 7 회에 닫힌다 (완전열거 대비 검증).
        master_hazard = np.stack(
            [
                _target_path_hazard(problem, paths, multiplier)
                for multiplier in multipliers
            ]
        )
        for target_index, target in enumerate(target_models):
            value, gradient = _nondetection_value_gradient(
                target, master_hazard[target_index]
            )
            cuts.append(
                (target_index, value, gradient, master_hazard[target_index].copy())
            )
        paths = _locally_improve_paths(
            problem,
            paths,
            target_models,
            multipliers,
            occupancy,
            destination_conflict_set,
            forbid_opposing_edge_swaps,
            passes=local_improvement_passes,
        )
        hazard = np.stack(
            [
                _target_path_hazard(problem, paths, multiplier)
                for multiplier in multipliers
            ]
        )
        target_values = tuple(
            _nondetection_value_gradient(target, hazard[index])[0]
            for index, target in enumerate(target_models)
        )
        candidate_upper = max(target_values)
        if candidate_upper < best_upper:
            best_upper = candidate_upper
            best_paths = paths
            best_target_values = target_values
            # master 가 휴리스틱을 이긴 반복을 남긴다. 끝까지 "warm-start"
            # 면 SPX 가 계획을 개선하지 못한 것이고, 그때 "SPX 가 최적화한
            # 계획"이라고 부르면 안 된다.
            incumbent_source = f"master-iteration-{iteration}"

        relative_gap = max(0.0, best_upper - certified_lower) / max(
            abs(best_upper), 1e-15
        )
        _record_bounds(bound_history, iteration, certified_lower, best_upper)
        if relative_gap <= relative_tolerance:
            converged = True
            break

        for target_index, target in enumerate(target_models):
            value, gradient = _nondetection_value_gradient(
                target, hazard[target_index]
            )
            cuts.append((target_index, value, gradient, hazard[target_index].copy()))

    if exact_survival_milp:
        if any(searcher.swept_hazard is not None for searcher in problem.searchers):
            raise ValueError(
                "exact_survival_milp still accumulates hazard in the destination "
                "cell only; it cannot be combined with swept hazard"
            )
        if np.any(occupancy > 1):
            raise ValueError(
                "exact_survival_milp currently requires occupancy limits <= 1"
            )
        # Combine the exact survival recursion with every globally valid OA
        # tangent generated above. The former removes approximation error;
        # the latter prevents the binary-product relaxation from collapsing.
        for cut in cuts:
            row, lb, ub = cut_row(cut)
            add_row(row, lb, ub)
        return _solve_exact_survival_milp(
            problem=problem,
            targets=target_models,
            multipliers=multipliers,
            groups=groups,
            arcs=arcs,
            arc_index=arc_index,
            by_time_destination=by_time_destination,
            by_time_edge=by_time_edge,
            base_rows=base_rows,
            base_lb=base_lb,
            base_ub=base_ub,
            lower=lower,
            upper=upper,
            integrality=integrality,
            eta_index=eta_index,
            destination_conflicts=destination_conflict_set,
            forbid_opposing_edge_swaps=forbid_opposing_edge_swaps,
            initial_paths=best_paths,
            relative_tolerance=relative_tolerance,
            mip_relative_gap=mip_relative_gap,
            time_limit_s=(
                exact_time_limit_s
                if exact_time_limit_s is not None
                else master_time_limit_s
            ),
            backend=exact_backend,
        )

    if best_paths is None:
        raise RuntimeError("Stone cutting-plane solver produced no feasible path")

    relative_gap = max(0.0, best_upper - certified_lower) / max(
        abs(best_upper), 1e-15
    )
    # Existing Chapter 3 has one target.  For SPX, this is the worst-target PD.
    probability_of_detection = 1.0 - best_upper
    ed = (
        expected_detections(problem, best_paths)
        if len(target_models) == 1
        else probability_of_detection
    )
    return StoneCuttingPlaneResult(
        method="stone-spx-cutting-plane",
        paths=best_paths,
        probability_of_detection=probability_of_detection,
        expected_detections=ed,
        evaluations=iteration,
        lower_bound_nondetection=certified_lower,
        upper_bound_nondetection=best_upper,
        relative_optimality_gap=relative_gap,
        converged=converged,
        iterations=iteration,
        target_nondetection_probabilities=best_target_values,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        discarded_warm_starts=discarded_warm_starts,
        bound_history=tuple(bound_history),
        master_status_history=tuple(status_history),
        incumbent_source=incumbent_source,
        warm_start_nondetection=(
            float(warm_start_nondetection)
            if isfinite(warm_start_nondetection)
            else float("nan")
        ),
    )


def _milp_status(solved) -> str:
    """scipy/HiGHS 결과를 optimal / limit / failed 로 요약한다."""

    status = getattr(solved, "status", None)
    if status == 0:
        return "optimal"
    if status == 1:
        return "limit"
    if getattr(solved, "success", False):
        return "optimal"
    if getattr(solved, "x", None) is not None:
        return "limit"
    return "failed"


def _record_bounds(
    history: list[tuple[int, float, float, float]],
    iteration: int,
    lower: float,
    upper: float,
) -> None:
    """반복 하나의 (하한, 상한, 상대갭)을 궤적에 남긴다.

    상한이 아직 무한대인 초기 반복은 갭을 1.0 으로 적는다 (정수해가 없어
    아무것도 인증되지 않은 상태). 한 반복에 두 번 불리면 뒤의 값으로 덮는다.
    """

    gap = 1.0
    if isfinite(upper):
        gap = max(0.0, upper - lower) / max(abs(upper), 1e-15)
    row = (iteration, float(lower), float(upper), float(gap))
    if history and history[-1][0] == iteration:
        history[-1] = row
    else:
        history.append(row)


def _solve_exact_survival_milp(
    *,
    problem: PathConstrainedProblem,
    targets: tuple[StoneTargetModel, ...],
    multipliers: tuple[np.ndarray, ...],
    groups: tuple[tuple[int, ...], ...],
    arcs: tuple[_Arc, ...],
    arc_index: dict[tuple[int, int, int, int], int],
    by_time_destination: dict[tuple[int, int], list[int]],
    by_time_edge: dict[tuple[int, int, int], list[int]],
    base_rows: list[dict[int, float]],
    base_lb: list[float],
    base_ub: list[float],
    lower: np.ndarray,
    upper: np.ndarray,
    integrality: np.ndarray,
    eta_index: int,
    destination_conflicts: set[tuple[int, int]],
    forbid_opposing_edge_swaps: bool,
    initial_paths: tuple[tuple[int, ...], ...] | None,
    relative_tolerance: float,
    mip_relative_gap: float,
    time_limit_s: float | None,
    backend: str,
) -> StoneCuttingPlaneResult:
    """Solve occupancy-one SPX exactly as a survival-state MILP.

    With at most one searcher in a cell, each visit indicator is binary. The
    target survival recursion contains only products of a continuous alive
    mass and such a binary visit. Standard four-row binary-product envelopes
    are exact, eliminating the outer approximation gap entirely.
    """

    target_count = len(targets)
    time_count = problem.time_count
    state_count = problem.state_count
    arc_count = len(arcs)
    original_variable_count = len(lower)
    mass_count = target_count * time_count * state_count
    mass_offset = original_variable_count
    post_offset = mass_offset + mass_count
    product_offset = post_offset + mass_count
    product_count = target_count * arc_count
    variable_count = product_offset + product_count

    def mass_index(target: int, time: int, state: int) -> int:
        return mass_offset + (target * time_count + time) * state_count + state

    def post_index(target: int, time: int, state: int) -> int:
        return post_offset + (target * time_count + time) * state_count + state

    def product_index(target: int, arc: int) -> int:
        return product_offset + target * arc_count + arc

    exact_lower = np.concatenate(
        (lower, np.zeros(2 * mass_count + product_count, dtype=float))
    )
    exact_upper = np.concatenate(
        (upper, np.ones(2 * mass_count + product_count, dtype=float))
    )
    exact_integrality = np.concatenate(
        (
            integrality,
            np.zeros(2 * mass_count + product_count, dtype=np.uint8),
        )
    )
    objective = np.zeros(variable_count, dtype=float)
    objective[eta_index] = 1.0
    mass_upper_bounds: list[np.ndarray] = []
    for target_index, target in enumerate(targets):
        no_search_mass = np.zeros((time_count, state_count), dtype=float)
        no_search_mass[0] = np.asarray(target.initial_mass, dtype=float)
        for time in range(time_count - 1):
            no_search_mass[time + 1] = (
                no_search_mass[time]
                @ target.transitions[time]
            )
        mass_upper_bounds.append(no_search_mass)
        start = mass_offset + target_index * time_count * state_count
        exact_upper[start : start + time_count * state_count] = (
            no_search_mass.ravel()
        )
        start = post_offset + target_index * time_count * state_count
        exact_upper[start : start + time_count * state_count] = (
            no_search_mass.ravel()
        )
        for arc_number, arc in enumerate(arcs):
            exact_upper[product_index(target_index, arc_number)] = (
                no_search_mass[arc.time, arc.destination]
            )
    rows = list(base_rows)
    row_lb = list(base_lb)
    row_ub = list(base_ub)

    def add(coefficients: dict[int, float], lb: float, ub: float) -> None:
        rows.append(coefficients)
        row_lb.append(lb)
        row_ub.append(ub)

    # Enforce the full 800 m separation graph through maximal clique rows.
    # These are exactly equivalent to all pairwise conflicts for integer paths.
    for clique in _maximal_conflict_cliques(state_count, destination_conflicts):
        for time in range(time_count):
            indices = [
                index
                for state in clique
                for index in by_time_destination.get((time, state), ())
            ]
            add({index: 1.0 for index in indices}, -inf, 1.0)

    if forbid_opposing_edge_swaps:
        for indices in by_time_edge.values():
            add({index: 1.0 for index in indices}, -inf, 1.0)

    for target_index, (target, multiplier) in enumerate(
        zip(targets, multipliers)
    ):
        initial_mass = np.asarray(target.initial_mass, dtype=float)
        transitions = transition_ops.as_sequence(target.transitions)
        for state in range(state_count):
            add(
                {mass_index(target_index, 0, state): 1.0},
                float(initial_mass[state]),
                float(initial_mass[state]),
            )

        for time in range(time_count):
            for state in range(state_count):
                coefficients = {
                    post_index(target_index, time, state): 1.0,
                    mass_index(target_index, time, state): -1.0,
                }
                for arc_number in by_time_destination.get((time, state), ()):
                    arc = arcs[arc_number]
                    member = groups[arc.searcher][0]
                    hazard = problem.searchers[member].effective_hazard(
                        time, arc.source, arc.destination
                    ) * multiplier[member, time, arc.source, arc.destination]
                    delta = float(np.exp(-hazard) - 1.0)
                    if delta:
                        coefficients[product_index(target_index, arc_number)] = -delta
                add(coefficients, 0.0, 0.0)

        for time in range(time_count - 1):
            # 이 경로는 개별 항목을 하나씩 읽으므로 조밀 배열이 필요하다.
            # 정확 생존 MILP 는 작은 문제에만 쓰이므로(희소화가 필요한
            # 규모에서는 애초에 돌지 않는다) 여기서만 펼친다.
            step = transitions[time]
            transition = (
                step.toarray() if transition_ops.is_sparse(step) else step
            )
            for destination in range(state_count):
                coefficients = {
                    mass_index(target_index, time + 1, destination): 1.0
                }
                for source in np.flatnonzero(transition[:, destination]):
                    coefficients[post_index(target_index, time, int(source))] = (
                        -float(transition[source, destination])
                    )
                add(coefficients, 0.0, 0.0)

        for arc_number, arc in enumerate(arcs):
            product = product_index(target_index, arc_number)
            mass = mass_index(target_index, arc.time, arc.destination)
            big_m = float(
                mass_upper_bounds[target_index][arc.time, arc.destination]
            )
            # product = mass * binary_arc
            add({product: 1.0, mass: -1.0}, -inf, 0.0)
            add({product: 1.0, arc_number: -big_m}, -inf, 0.0)
            add(
                {product: 1.0, mass: -1.0, arc_number: -big_m},
                -big_m,
                inf,
            )

        coefficients = {eta_index: -1.0}
        for state in range(state_count):
            coefficients[post_index(target_index, time_count - 1, state)] = 1.0
        add(coefficients, -inf, 0.0)

    warm_start = None
    if initial_paths is not None:
        warm_start = np.zeros(variable_count, dtype=float)
        original_start = _master_solution_from_paths(
            problem,
            groups,
            arcs,
            arc_index,
            targets,
            multipliers,
            initial_paths,
            original_variable_count,
            arc_count,
            eta_index,
        )
        warm_start[:original_variable_count] = original_start
        for target_index, (target, multiplier) in enumerate(
            zip(targets, multipliers)
        ):
            hazard = _target_path_hazard(problem, initial_paths, multiplier)
            mass_values = np.zeros((time_count, state_count), dtype=float)
            post_values = np.zeros_like(mass_values)
            mass_values[0] = np.asarray(target.initial_mass, dtype=float)
            for time in range(time_count):
                post_values[time] = mass_values[time] * np.exp(-hazard[time])
                if time + 1 < time_count:
                    mass_values[time + 1] = (
                        post_values[time]
                        @ target.transitions[time]
                    )
            start = mass_offset + target_index * time_count * state_count
            warm_start[start : start + time_count * state_count] = (
                mass_values.ravel()
            )
            start = post_offset + target_index * time_count * state_count
            warm_start[start : start + time_count * state_count] = (
                post_values.ravel()
            )
            for arc_number, arc in enumerate(arcs):
                warm_start[product_index(target_index, arc_number)] = (
                    mass_values[arc.time, arc.destination]
                    * original_start[arc_number]
                )

    solver = _PersistentHighsMaster(
        objective, exact_lower, exact_upper, exact_integrality
    )
    solver.add_rows(rows, row_lb, row_ub)
    solved = (
        _solve_scip_master(
            solver,
            mip_relative_gap=mip_relative_gap,
            time_limit_s=time_limit_s,
            warm_start=warm_start,
        )
        if backend == "scip"
        else solver.solve(
            mip_relative_gap=mip_relative_gap,
            time_limit_s=time_limit_s,
            warm_start=warm_start,
        )
    )
    if solved.x is None or solved.fun is None:
        raise RuntimeError(f"Stone exact survival MILP failed: {solved.message}")
    paths = _extract_group_paths(
        problem, groups, arcs, np.asarray(solved.x[:arc_count])
    )
    target_values = tuple(
        _nondetection_value_gradient(
            target, _target_path_hazard(problem, paths, multiplier)
        )[0]
        for target, multiplier in zip(targets, multipliers)
    )
    upper_bound = max(target_values)
    lower_bound = float(solved.mip_dual_bound)
    relative_gap = max(0.0, upper_bound - lower_bound) / max(
        abs(upper_bound), 1e-15
    )
    return StoneCuttingPlaneResult(
        method="stone-spx-exact-survival-milp",
        paths=paths,
        probability_of_detection=1.0 - upper_bound,
        expected_detections=1.0 - upper_bound,
        evaluations=1,
        lower_bound_nondetection=lower_bound,
        upper_bound_nondetection=upper_bound,
        relative_optimality_gap=relative_gap,
        converged=relative_gap <= relative_tolerance,
        iterations=1,
        target_nondetection_probabilities=target_values,
        fallback_used=not solved.success,
        fallback_reason=None if solved.success else str(solved.message),
    )


def _validate_move_shape(
    move: "StoneMove",
    problem: PathConstrainedProblem,
) -> None:
    """구조적으로 말이 안 되는 incompatible move 는 조용히 넘기지 않는다.

    시간 도달가능성은 여기서 보지 않는다 — 도달 불가 arc 는 공허한 제약이라
    호출자 오류가 아니다. 반면 없는 탐색자·범위 밖 상태·인접하지 않은 이동은
    호출자가 문제를 잘못 기술한 것이므로 실패해야 한다.
    """

    searcher, time, source, destination = move
    if not 0 <= searcher < problem.searcher_count:
        raise ValueError(f"incompatible move names an unknown searcher: {move}")
    if not 0 <= time < problem.time_count:
        raise ValueError(f"incompatible move names a time outside the horizon: {move}")
    for state in (source, destination):
        if not 0 <= state < problem.state_count:
            raise ValueError(f"incompatible move names a state outside the grid: {move}")
    if not problem.searchers[searcher].adjacency[source, destination]:
        raise ValueError(f"incompatible move is not an adjacent step: {move}")


def _validate_sp1_homogeneity(problem: PathConstrainedProblem) -> None:
    first = problem.searchers[0]
    if any(searcher.swept_hazard is not None for searcher in problem.searchers):
        raise ValueError(
            "SP1 requires cell-time detection rates independent of the incoming arc; "
            "swept (arc-distributed) hazard needs SPX"
        )
    if first.transit_fraction is not None:
        raise ValueError(
            "SP1 requires cell-time detection rates independent of the incoming arc; "
            "use SPX for transit-dependent rates"
        )
    for searcher in problem.searchers[1:]:
        if searcher.start_state != first.start_state:
            raise ValueError("SP1 requires homogeneous searcher start states")
        if not np.array_equal(searcher.adjacency, first.adjacency):
            raise ValueError("SP1 requires homogeneous searcher movement")
        if not np.allclose(searcher.detection_rate, first.detection_rate):
            raise ValueError("SP1 requires homogeneous detection rates")
        if searcher.transit_fraction is not None:
            raise ValueError(
                "SP1 requires cell-time detection rates independent of the incoming "
                "arc; use SPX for transit-dependent rates"
            )


def _validated_target(
    target: StoneTargetModel,
    problem: PathConstrainedProblem,
) -> np.ndarray:
    mass = np.asarray(target.initial_mass, dtype=float)
    transitions = transition_ops.as_sequence(target.transitions)
    if mass.shape != (problem.state_count,):
        raise ValueError("target initial_mass must match the state space")
    if (mass < 0.0).any() or not np.isclose(mass.sum(), 1.0):
        raise ValueError("target initial_mass must be a probability distribution")
    expected_shape = (
        max(problem.time_count - 1, 0),
        problem.state_count,
        problem.state_count,
    )
    transition_ops.validate(
        transitions,
        states=expected_shape[1],
        steps=expected_shape[0],
        what="target transitions",
    )

    shape = (
        problem.searcher_count,
        problem.time_count,
        problem.state_count,
        problem.state_count,
    )
    multiplier = np.asarray(target.hazard_multiplier, dtype=float)
    try:
        broadcast = np.broadcast_to(multiplier, shape)
    except ValueError as error:
        raise ValueError(
            "hazard_multiplier must be scalar or broadcast to "
            "[searcher,time,from_state,to_state]"
        ) from error
    if (broadcast < 0.0).any() or not np.isfinite(broadcast).all():
        raise ValueError("hazard_multiplier must be finite and nonnegative")
    return np.asarray(broadcast, dtype=float)


def _occupancy_limits(
    limits: int | np.ndarray | None,
    problem: PathConstrainedProblem,
) -> np.ndarray:
    shape = (problem.time_count, problem.state_count)
    if limits is None:
        return np.full(shape, problem.searcher_count, dtype=int)
    array = np.asarray(limits, dtype=int)
    try:
        result = np.broadcast_to(array, shape)
    except ValueError as error:
        raise ValueError("occupancy_limits must broadcast to [time,state]") from error
    if (result < 0).any():
        raise ValueError("occupancy_limits cannot be negative")
    return np.asarray(result, dtype=int)


def _maximal_conflict_cliques(
    state_count: int,
    conflicts: set[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate maximal cliques of the undirected conflict graph."""

    if not conflicts:
        return ()
    neighbors = [set() for _ in range(state_count)]
    for left, right in conflicts:
        neighbors[left].add(right)
        neighbors[right].add(left)

    cliques: list[tuple[int, ...]] = []

    def visit(current: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            if len(current) >= 2:
                cliques.append(tuple(sorted(current)))
            return
        union = candidates | excluded
        pivot = (
            max(union, key=lambda vertex: len(candidates & neighbors[vertex]))
            if union
            else None
        )
        extensions = candidates - (neighbors[pivot] if pivot is not None else set())
        for vertex in tuple(extensions):
            visit(
                current | {vertex},
                candidates & neighbors[vertex],
                excluded & neighbors[vertex],
            )
            candidates.remove(vertex)
            excluded.add(vertex)

    visit(set(), {index for index, item in enumerate(neighbors) if item}, set())
    return tuple(sorted(cliques))


def _searcher_groups(
    problem: PathConstrainedProblem,
    aggregate_identical_searchers: bool,
) -> tuple[tuple[int, ...], ...]:
    """Partition searchers into mathematically interchangeable flow classes."""

    if not aggregate_identical_searchers:
        return tuple((index,) for index in range(problem.searcher_count))
    groups: list[list[int]] = []
    for index, candidate in enumerate(problem.searchers):
        for group in groups:
            reference = problem.searchers[group[0]]
            same = (
                candidate.start_state == reference.start_state
                and np.array_equal(candidate.detection_rate, reference.detection_rate)
                and np.array_equal(candidate.adjacency, reference.adjacency)
                and (
                    candidate.transit_fraction is None
                    and reference.transit_fraction is None
                    or candidate.transit_fraction is not None
                    and reference.transit_fraction is not None
                    and np.array_equal(
                        candidate.transit_fraction,
                        reference.transit_fraction,
                    )
                )
            )
            if same:
                group.append(index)
                break
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _build_group_arcs(
    problem: PathConstrainedProblem,
    groups: tuple[tuple[int, ...], ...],
) -> tuple[_Arc, ...]:
    arcs: list[_Arc] = []
    for group_index, members in enumerate(groups):
        searcher = problem.searchers[members[0]]
        reachable_sources = {searcher.start_state}
        for time in range(problem.time_count):
            next_reachable: set[int] = set()
            for source in sorted(reachable_sources):
                for destination in searcher.successors(source):
                    arcs.append(_Arc(group_index, time, source, destination))
                    next_reachable.add(destination)
            reachable_sources = next_reachable
    return tuple(arcs)


def _extract_group_paths(
    problem: PathConstrainedProblem,
    groups: tuple[tuple[int, ...], ...],
    arcs: tuple[_Arc, ...],
    values: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    paths_by_searcher: list[tuple[int, ...] | None] = [None] * problem.searcher_count
    rounded = np.rint(values).astype(int)
    if not np.allclose(values, rounded, atol=1e-5):
        raise RuntimeError("MILP solution contains a nonintegral arc flow")
    for group_index, members in enumerate(groups):
        paths: list[list[int]] = [[] for _ in members]
        current = [problem.searchers[member].start_state for member in members]
        for time in range(problem.time_count):
            available: dict[int, list[int]] = {}
            for arc, count in zip(arcs, rounded):
                if arc.searcher != group_index or arc.time != time or count <= 0:
                    continue
                available.setdefault(arc.source, []).extend(
                    [arc.destination] * int(count)
                )
            for path_index, source in enumerate(current):
                destinations = available.get(source)
                if not destinations:
                    raise RuntimeError(
                        "aggregate MILP flow cannot be decomposed into paths"
                    )
                destination = destinations.pop()
                paths[path_index].append(destination)
                current[path_index] = destination
        for member, path in zip(members, paths):
            paths_by_searcher[member] = tuple(path)
    if any(path is None for path in paths_by_searcher):
        raise RuntimeError("MILP solution does not define one path per searcher")
    return tuple(path for path in paths_by_searcher if path is not None)


def _target_path_hazard(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    multiplier: np.ndarray,
) -> np.ndarray:
    hazard = np.zeros((problem.time_count, problem.state_count), dtype=float)
    for searcher_index, (searcher, path) in enumerate(
        zip(problem.searchers, paths)
    ):
        source = searcher.start_state
        for time, destination in enumerate(path):
            scale = multiplier[searcher_index, time, source, destination]
            # 이동 중에도 센서는 켜져 있다. hazard 는 도착셀 하나가 아니라
            # **지나간 셀 전부**에 쌓인다 (``swept_cells`` 참조).
            for cell, value in searcher.swept_cells(
                time, source, destination
            ).items():
                hazard[time, cell] += value * scale
            source = destination
    return hazard


def _locally_improve_paths(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    targets: tuple[StoneTargetModel, ...],
    multipliers: tuple[np.ndarray, ...],
    occupancy: np.ndarray,
    destination_conflicts: set[tuple[int, int]],
    forbid_opposing_edge_swaps: bool,
    *,
    passes: int,
) -> tuple[tuple[int, ...], ...]:
    """Best-improvement coordinate search over feasible path waypoints."""

    if passes <= 0:
        return paths
    current = [list(path) for path in paths]

    def score(candidate) -> float:
        candidate_tuple = tuple(tuple(path) for path in candidate)
        return max(
            _nondetection_value_gradient(
                target,
                _target_path_hazard(problem, candidate_tuple, multiplier),
            )[0]
            for target, multiplier in zip(targets, multipliers)
        )

    current_score = score(current)
    for _ in range(passes):
        best_move = None
        best_score = current_score
        for searcher_index, searcher in enumerate(problem.searchers):
            for time in range(problem.time_count):
                source = (
                    searcher.start_state
                    if time == 0
                    else current[searcher_index][time - 1]
                )
                next_state = (
                    None
                    if time + 1 == problem.time_count
                    else current[searcher_index][time + 1]
                )
                for destination in searcher.successors(source):
                    if destination == current[searcher_index][time]:
                        continue
                    if next_state is not None and not searcher.adjacency[
                        destination, next_state
                    ]:
                        continue
                    proposal = [path.copy() for path in current]
                    proposal[searcher_index][time] = destination
                    proposal_tuple = tuple(tuple(path) for path in proposal)
                    try:
                        _validate_initial_paths(
                            problem,
                            proposal_tuple,
                            occupancy,
                            destination_conflicts,
                            forbid_opposing_edge_swaps,
                        )
                    except ValueError:
                        continue
                    proposal_score = score(proposal)
                    if proposal_score < best_score - 1e-12:
                        best_score = proposal_score
                        best_move = proposal
        if best_move is None:
            break
        current = best_move
        current_score = best_score
    return tuple(tuple(path) for path in current)


def _validate_initial_paths(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    occupancy: np.ndarray,
    destination_conflicts: set[tuple[int, int]],
    forbid_opposing_edge_swaps: bool,
) -> None:
    if len(paths) != problem.searcher_count or any(
        len(path) != problem.time_count for path in paths
    ):
        raise ValueError("initial_paths must match searcher count and horizon")
    for searcher, path in zip(problem.searchers, paths):
        source = searcher.start_state
        for destination in path:
            if not 0 <= destination < problem.state_count:
                raise ValueError("initial path state is outside the state space")
            if not searcher.adjacency[source, destination]:
                raise ValueError("initial_paths contain an infeasible move")
            source = destination
    for time in range(problem.time_count):
        destinations = [path[time] for path in paths]
        counts = np.bincount(destinations, minlength=problem.state_count)
        if np.any(counts > occupancy[time]):
            raise ValueError("initial_paths violate occupancy limits")
        for left in range(len(paths)):
            for right in range(left + 1, len(paths)):
                pair = (
                    min(destinations[left], destinations[right]),
                    max(destinations[left], destinations[right]),
                )
                if pair in destination_conflicts:
                    raise ValueError("initial_paths violate a destination conflict")
                if forbid_opposing_edge_swaps:
                    left_source = (
                        problem.searchers[left].start_state
                        if time == 0
                        else paths[left][time - 1]
                    )
                    right_source = (
                        problem.searchers[right].start_state
                        if time == 0
                        else paths[right][time - 1]
                    )
                    if (
                        left_source == destinations[right]
                        and right_source == destinations[left]
                        and left_source != right_source
                    ):
                        raise ValueError("initial_paths contain an opposing edge swap")


def _nondetection_value_gradient(
    target: StoneTargetModel,
    hazard: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Markov forward/backward evaluation of Stone equations (4.57)-(4.58)."""

    hazard = np.asarray(hazard, dtype=float)
    time_count, state_count = hazard.shape
    mass = np.zeros((time_count, state_count), dtype=float)
    mass[0] = np.asarray(target.initial_mass, dtype=float)
    survival = np.exp(-hazard)
    for time in range(time_count - 1):
        mass[time + 1] = (
            mass[time] * survival[time]
        ) @ target.transitions[time]

    backward = np.ones((time_count, state_count), dtype=float)
    for time in range(time_count - 2, -1, -1):
        backward[time] = target.transitions[time] @ (
            survival[time + 1] * backward[time + 1]
        )
    value = float((mass[-1] * survival[-1]).sum())
    gradient = -mass * survival * backward
    return value, gradient


def _master_solution_from_paths(
    problem: PathConstrainedProblem,
    groups: tuple[tuple[int, ...], ...],
    arcs: tuple[_Arc, ...],
    arc_index: dict[tuple[int, int, int, int], int],
    targets: tuple[StoneTargetModel, ...],
    multipliers: tuple[np.ndarray, ...],
    paths: tuple[tuple[int, ...], ...],
    variable_count: int,
    arc_count: int,
    eta_index: int,
) -> np.ndarray:
    """Build a complete feasible MIP start from individual searcher paths."""

    values = np.zeros(variable_count, dtype=float)
    group_of = {
        member: group_index
        for group_index, members in enumerate(groups)
        for member in members
    }
    for searcher_index, path in enumerate(paths):
        source = problem.searchers[searcher_index].start_state
        group = group_of[searcher_index]
        for time, destination in enumerate(path):
            values[arc_index[(group, time, source, destination)]] += 1.0
            source = destination
    target_values = []
    offset = arc_count
    for target, multiplier in zip(targets, multipliers):
        hazard = _target_path_hazard(problem, paths, multiplier)
        values[offset : offset + hazard.size] = hazard.ravel()
        offset += hazard.size
        target_values.append(_nondetection_value_gradient(target, hazard)[0])
    values[eta_index] = max(target_values)
    return values


class _PersistentHighsMaster:
    """Incremental HiGHS MILP whose incumbent survives cutting-plane rounds."""

    def __init__(self, objective, lower, upper, integrality) -> None:
        assert highspy is not None
        self.highs = highspy.Highs()
        self.highs.setOptionValue("output_flag", False)
        self.variable_count = len(objective)
        starts = np.zeros(self.variable_count + 1, dtype=np.int32)
        self.highs.addCols(
            self.variable_count,
            np.asarray(objective, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            0,
            starts,
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=float),
        )
        integer_indices = np.flatnonzero(np.asarray(integrality, dtype=np.uint8))
        if integer_indices.size:
            self.highs.changeColsIntegrality(
                integer_indices.size,
                np.asarray(integer_indices, dtype=np.int32),
                np.full(
                    integer_indices.size,
                    highspy.HighsVarType.kInteger,
                    dtype=np.uint8,
                ),
            )

    def add_rows(self, rows, lower, upper) -> None:
        if not rows:
            return
        starts = [0]
        indices: list[int] = []
        values: list[float] = []
        for row in rows:
            for index, value in sorted(row.items()):
                if value:
                    indices.append(index)
                    values.append(value)
            starts.append(len(indices))
        self.highs.addRows(
            len(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            len(indices),
            np.asarray(starts, dtype=np.int32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(values, dtype=float),
        )

    def solve(
        self,
        *,
        mip_relative_gap: float,
        time_limit_s: float | None,
        warm_start: np.ndarray | None,
    ):
        self.highs.setOptionValue("mip_rel_gap", float(mip_relative_gap))
        self.highs.setOptionValue("mip_abs_gap", 0.0)
        self.highs.setOptionValue(
            "time_limit",
            float(time_limit_s) if time_limit_s is not None else highspy.kHighsInf,
        )
        if warm_start is not None:
            self.highs.setSolution(
                self.variable_count,
                np.arange(self.variable_count, dtype=np.int32),
                np.asarray(warm_start, dtype=float),
            )
        self.highs.run()
        status = self.highs.getModelStatus()
        solution = self.highs.getSolution()
        info = self.highs.getInfo()
        usable = bool(solution.value_valid and info.primal_solution_status)
        return SimpleNamespace(
            success=status == highspy.HighsModelStatus.kOptimal,
            x=(np.asarray(solution.col_value, dtype=float) if usable else None),
            fun=(float(info.objective_function_value) if usable else None),
            mip_dual_bound=float(info.mip_dual_bound),
            message=str(status),
        )


def _solve_scip_master(
    highs_master: _PersistentHighsMaster,
    *,
    mip_relative_gap: float,
    time_limit_s: float | None,
    warm_start: np.ndarray | None,
):
    """Transfer a HiGHS-built sparse MILP to SCIP and return common results."""

    if ScipModel is None:  # pragma: no cover - guarded by public validation
        raise RuntimeError("SCIP backend is unavailable")
    with TemporaryDirectory(prefix="stone-spx-scip-") as directory:
        model_path = Path(directory) / "stone-spx.mps"
        status = highs_master.highs.writeModel(str(model_path))
        if status == highspy.HighsStatus.kError:
            raise RuntimeError(f"failed to export exact SPX model: {status}")
        model = ScipModel()
        model.hideOutput(True)
        model.readProblem(str(model_path))
        model.setRealParam("limits/gap", float(mip_relative_gap))
        if time_limit_s is not None:
            model.setRealParam("limits/time", float(time_limit_s))
        variables = sorted(
            model.getVars(transformed=False), key=lambda variable: variable.getIndex()
        )
        if len(variables) != highs_master.variable_count:
            raise RuntimeError("SCIP changed the exported SPX column count")
        if warm_start is not None:
            solution = model.createSol()
            for variable, value in zip(variables, warm_start):
                if value:
                    model.setSolVal(solution, variable, float(value))
            model.addSol(solution, free=True)
        model.optimize()
        best = model.getBestSol()
        solve_status = str(model.getStatus())
        if best is None:
            return SimpleNamespace(
                success=False,
                x=None,
                fun=None,
                mip_dual_bound=float(model.getDualbound()),
                message=solve_status,
            )
        values = np.asarray(
            [model.getSolVal(best, variable) for variable in variables],
            dtype=float,
        )
        return SimpleNamespace(
            success=solve_status == "optimal",
            x=values,
            fun=float(model.getSolObjVal(best)),
            mip_dual_bound=float(model.getDualbound()),
            message=solve_status,
        )


def _sparse_rows(rows: list[dict[int, float]], variable_count: int):
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    for row_index, coefficients in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            if coefficient:
                row_indices.append(row_index)
                column_indices.append(column_index)
                data.append(coefficient)
    return coo_array(
        (data, (row_indices, column_indices)),
        shape=(len(rows), variable_count),
    ).tocsr()
