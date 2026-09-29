"""Stone SPX 운용 어댑터 — 축소 문제 인스턴스와 그 위의 절단평면 해법.

Stone, Royset & Washburn (2016) 4장의 SPX 모형은 격자 크기에 대해 지수적이다.
그래서 이 모듈은 **선언된 축소 격자** 위에서만 SPX 를 푼다. 하는 일은 셋이고,
세 단계가 서로 분리되어 있는 것이 v0.4 의 설계 핵심이다.

1. ``build_stone_grid_instance``
   지형결합 IMM5 입자 belief -> 시공간 Markov 표적 -> 축소 경로제약 문제
   (:class:`StoneGridInstance`). 계획법을 고르기 **전에** 끝나는 모든 일.

2. ``solve_stone_grid_paths``
   이질 탐색자 · 다중 표적 minimax 를 절단평면으로 푼다. 모든 master 는
   MILP 이고, 접선 절단이 미탐지확률의 유효 하한을, 정수해가 상한을 준다.
   따라서 결과에는 **최적성 인증폭**이 붙는다 — 휴리스틱을 정확해처럼
   보고하지 않는다.

3. ``flown_routes_from_paths``
   셀 경로 -> 시간예산이 맞는 실제 비행경로. 계획법과 무관한 공통 변환.

같은 인스턴스를 ``learning/spx_env`` 의 MAPPO 정책도 받는다. 두 계획법의
결과에 같은 ``common_input_fingerprint`` 가 박히기 때문에, KPI 차이를 문제
설정 차이가 아니라 **셀 경로 선택 차이**로 귀속할 수 있다.

의존
----
* 위: ``theory/{stone_path, path_constrained, markov}``,
  ``planning/{terrain_belief, routes}``, ``cpp_search.{models, probability, profiles}``.
* 아래: ``learning/spx_env``, ``chapters/*``. 이 모듈은 ``learning`` 을
  **절대** import 하지 않는다 (``tests/test_architecture.py`` 가 강제).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import floor
from time import perf_counter
from typing import Literal, Protocol

import numpy as np
from scipy import sparse

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, Route, SensorSpec
from cpp_search.core.probability import PolarProbabilityMap, TargetPrior
from cpp_search.core.profiles import TargetOperationalProfile
from cpp_search.theory.markov import SpaceTimeMarkovModel, cell_adjacency, polar_cell_index
from cpp_search.theory.path_constrained import (
    InfeasiblePathError,
    PathConstrainedProblem,
    SearcherModel,
    path_detection_probability,
    team_h2_receding_horizon,
    team_receding_horizon,
)
from cpp_search.theory import transitions as transition_ops
from cpp_search.theory.stone_path import StoneTargetModel, stone_spx_cutting_plane
from cpp_search.planning.routes import local_sweep
from cpp_search.planning.terrain_belief import (
    BELIEF_TERRAIN_FULL,
    BELIEF_TERRAIN_OFF,
    TerrainParticleBelief,
)


@dataclass(frozen=True, slots=True)
class StoneSPXTargetSpec:
    profile: TargetOperationalProfile
    hazard_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class StoneSPXSearcherClass:
    name: str
    count: int
    hazard_scale: float


@dataclass(frozen=True, slots=True)
class StoneSPXRouteConfig:
    grid_width: int = 3
    grid_height: int = 3
    time_slice_count: int = 3
    particle_count: int = 600
    visit_hazard: float = 1.0
    hazard_calibration: Literal["fixed", "effective-sweep-width"] = "fixed"
    occupancy_limit: int = 1
    forbid_opposing_edge_swaps: bool = True
    relative_tolerance: float = 1e-6
    max_iterations: int = 80
    mip_relative_gap: float = 0.0
    grid_kind: Literal["square", "operational-polar"] = "square"
    radial_step_m: float = 600.0
    angular_bin_count: int = 24
    reservation_separation_m: float = 0.0
    aggregate_identical_searchers: bool = False
    master_time_limit_s: float | None = None
    persistent_master: bool = False
    continuous_relaxation_iterations: int = 0
    exact_survival_milp: bool = False
    exact_time_limit_s: float | None = None
    exact_backend: str = "highs"
    master_backend: str = "highs"
    local_improvement_passes: int = 0
    #: 지형 가중을 계획에 넣는가. ``False`` 면 지형이 들어가는 지점을
    #: **전부** 끊는다 — belief 결합(사전·전이)과 격자 사전확률. hazard 의
    #: 관측성 배율도 함께 끊지만, 합성지형에서는 그 값이 원래 전 셀 1.0 이라
    #: (``TerrainField.observability_weight`` 참조) 실질 효과는 belief 쪽뿐이다.
    #: Chapter 2 의 지형효과 대조군이 이 플래그 하나로 정의된다.
    terrain_weighting: bool = True
    #: 정사각 격자를 AOI 원에 어떻게 맞추는가. ``circumscribed`` 가 기본이며
    #: AOI 전체를 덮는다. ``inscribed`` 는 v0.3 까지의 동작으로, 원의 63.7%
    #: 만 덮어 missing-person(child) 질량의 20% 를 ``outside`` 로 흘린다.
    square_fit: str = "circumscribed"
    #: 전이행렬을 희소로 담을지. ``"auto"`` 는 셀이 많아질 때만 켠다
    #: (조밀은 셀 수^2 × 슬라이스 수라 셀=탐지폭 격자에서 메모리가 터진다).
    #: 값 자체는 조밀과 **완전히 동일**하다 — 0 을 적지 않을 뿐이다.
    sparse_transitions: str | bool = "auto"


@dataclass(frozen=True, slots=True)
class StoneSPXRouteDiagnostics:
    planning_method: str
    common_input_fingerprint: str
    detection_probability: float
    no_detection_probability: float
    target_detection_probabilities: tuple[tuple[str, float], ...]
    relative_optimality_gap: float
    lower_bound_nondetection: float
    upper_bound_nondetection: float
    iterations: int
    converged: bool
    fallback_used: bool
    fallback_reason: str | None
    runtime_s: float
    duplicate_assignment_ratio: float
    opposing_edge_swaps: int
    communication_available_ratio: float
    outside_probability: float
    searcher_classes: tuple[str, ...]
    start_cells: tuple[int, ...]
    start_quantization_max_m: float
    hazard_calibration: str
    full_slice_visit_hazard: float
    support_total_width_m: float
    effective_sweep_width_m: float
    cell_area_m2: float
    cell_area_min_m2: float
    cell_area_max_m2: float
    grid_kind: str
    grid_shape: tuple[int, int]
    terrain_weighting: bool


@dataclass(frozen=True, slots=True)
class StoneSPXRoutePlan:
    routes: list[Route]
    markov: SpaceTimeMarkovModel
    diagnostics: StoneSPXRouteDiagnostics
    assignments: tuple[tuple[int, ...], ...]
    planner_terrain: object | None
    search_model: None = None


StoneGridMethod = Literal["stone-spx", "team-h1", "team-h2"]


@dataclass(frozen=True, slots=True)
class _SquareGrid:
    center: Point2D
    half_side_m: float
    width: int
    height: int

    @classmethod
    def inscribed(cls, mission: MissionConfig, width: int, height: int):
        """AOI 원에 **내접**하는 정사각형. 원의 63.7% 만 덮는다.

        남는 36.3% 로 나간 belief 는 ``outside`` 흡수상태로 빠져 계획모형이
        영원히 탐지하지 못한다. 480 s 시나리오에서 missing-person(child) 질량의
        20.07% 가 여기로 샜다 (missing-person(adult) 는 1.50%). minimax
        목적함수의 최악 표적이 missing-person(child) 이므로 이 손실은 곧
        목적함수를 조용히 바꾼다. 비교·재현 목적으로만 남겨두고, 기본값은
        ``circumscribed`` 다.
        """

        return cls(
            mission.center,
            mission.search_radius_m / 2.0**0.5,
            width,
            height,
        )

    @classmethod
    def circumscribed(cls, mission: MissionConfig, width: int, height: int):
        """AOI 원을 **완전히 덮는** 정사각형 (반변 = R).

        AOI 를 "순변위 99% 반경"으로 선언한 이상, 그 안의 일부만 탐색하면
        선언과 실제 탐색 범위가 어긋난다. 대가는 같은 셀 수에서 셀변이
        √2 배로 커지는 것이다 — 센서 정합이 나빠지므로 셀 수로 되사야 한다.
        """

        return cls(mission.center, float(mission.search_radius_m), width, height)

    @classmethod
    def fitted(cls, mission: MissionConfig, width: int, height: int, fit: str):
        if fit == "circumscribed":
            return cls.circumscribed(mission, width, height)
        if fit == "inscribed":
            return cls.inscribed(mission, width, height)
        raise ValueError(f"unknown square grid fit: {fit!r}")

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def cell_width_m(self) -> float:
        return 2.0 * self.half_side_m / self.width

    @property
    def cell_height_m(self) -> float:
        return 2.0 * self.half_side_m / self.height

    @property
    def cell_area_m2(self) -> float:
        return self.cell_width_m * self.cell_height_m

    @property
    def cell_areas_m2(self) -> np.ndarray:
        return np.full(self.cell_count, self.cell_area_m2, dtype=float)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def fingerprint(self) -> str:
        return f"square:{self.width}:{self.height}:{self.half_side_m:.17g}"

    def center_of(self, cell: int) -> Point2D:
        x, y = cell % self.width, cell // self.width
        return Point2D(
            self.center.x - self.half_side_m + (x + 0.5) * self.cell_width_m,
            self.center.y - self.half_side_m + (y + 0.5) * self.cell_height_m,
        )

    def cell_of(self, point: Point2D) -> int | None:
        # ``int()`` 은 0 을 향해 절단하므로 (-1, 0) 구간이 열 0 으로 들어간다.
        # 그러면 정사각형 **서쪽·남쪽** 바깥 한 셀 폭의 점이 가장자리 셀로
        # 잘못 배정되고 동쪽·북쪽은 정상이라, 표적 belief 가 비대칭으로
        # 왜곡됐다 (v0.3 부터 있던 결함). ``floor`` 는 음수에서도 아래로 내린다.
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        x = floor((dx + self.half_side_m) / self.cell_width_m)
        y = floor((dy + self.half_side_m) / self.cell_height_m)
        # 구간을 [-half, +half) 로 두면 **위·오른쪽 모서리 위의 점**이 밖으로
        # 나간다. 내접에서는 모서리가 AOI 내부라 무해했지만, 외접에서는
        # 격자 모서리 = AOI 경계라서 경계 전체가 탐지 불가로 떨어진다.
        # 바깥 모서리를 닫아 마지막 셀에 넣는다.
        if x == self.width and dx <= self.half_side_m:
            x = self.width - 1
        if y == self.height and dy <= self.half_side_m:
            y = self.height - 1
        if 0 <= x < self.width and 0 <= y < self.height:
            return y * self.width + x
        return None

    def nearest_cell(self, point: Point2D) -> int:
        cell = self.cell_of(point)
        if cell is not None:
            return cell
        return min(
            range(self.cell_count),
            key=lambda index: self.center_of(index).distance_to(point),
        )


@dataclass(frozen=True, slots=True)
class _PolarGrid:
    mission: MissionConfig
    probability_map: PolarProbabilityMap

    @property
    def cell_count(self) -> int:
        return len(self.probability_map.cells)

    @property
    def cell_areas_m2(self) -> np.ndarray:
        return np.asarray(
            [cell.area_m2 for cell in self.probability_map.cells], dtype=float
        )

    @property
    def cell_area_m2(self) -> float:
        return float(np.mean(self.cell_areas_m2))

    @property
    def grid_shape(self) -> tuple[int, int]:
        radial_count = self.cell_count // self.probability_map.angular_bin_count
        return (radial_count, self.probability_map.angular_bin_count)

    @property
    def fingerprint(self) -> str:
        return (
            f"operational-polar:{self.probability_map.radial_step_m:.17g}:"
            f"{self.probability_map.angular_bin_count}:{self.mission.search_radius_m:.17g}"
        )

    def center_of(self, cell: int) -> Point2D:
        return self.probability_map.cells[cell].center

    def cell_of(self, point: Point2D) -> int | None:
        if point.distance_to(self.mission.center) > self.mission.search_radius_m:
            return None
        return polar_cell_index(point, self.mission, self.probability_map)

    def nearest_cell(self, point: Point2D) -> int:
        cell = self.cell_of(point)
        if cell is not None:
            return cell
        return min(
            range(self.cell_count),
            key=lambda index: self.center_of(index).distance_to(point),
        )


class _OperationalGrid(Protocol):
    cell_count: int
    cell_area_m2: float
    cell_areas_m2: np.ndarray
    grid_shape: tuple[int, int]
    fingerprint: str

    def center_of(self, cell: int) -> Point2D: ...
    def cell_of(self, point: Point2D) -> int | None: ...
    def nearest_cell(self, point: Point2D) -> int: ...



#: ``"auto"`` 에서 희소로 넘어가는 셀 수. 20x20(400셀) 까지는 조밀이
#: 0.09 GB 라 문제가 없고, 24x24 부터 빠르게 커진다.
SPARSE_TRANSITION_CELL_THRESHOLD = 400


def _use_sparse_transitions(setting: str | bool, cell_count: int) -> bool:
    if isinstance(setting, bool):
        return setting
    if setting == "auto":
        return cell_count > SPARSE_TRANSITION_CELL_THRESHOLD
    raise ValueError(f"unknown sparse_transitions setting: {setting!r}")


def _target_markov(
    particle_filter,
    grid: _OperationalGrid,
    *,
    time_slice_s: float,
    time_slice_count: int,
    sparse_transitions: bool = False,
) -> tuple[np.ndarray, object]:
    outside = grid.cell_count
    state_count = outside + 1

    def states() -> list[int]:
        result = []
        for point, escaped in zip(particle_filter.points, particle_filter.escaped):
            cell = None if escaped else grid.cell_of(point)
            result.append(outside if cell is None else cell)
        return result

    initial = np.zeros(state_count, dtype=float)
    for state, weight in zip(states(), particle_filter.weights):
        initial[state] += weight
    initial /= initial.sum()

    steps = max(time_slice_count - 1, 0)
    if not sparse_transitions:
        transitions = np.zeros((steps, state_count, state_count), dtype=float)
        for time_index in range(steps):
            before = states()
            weights = tuple(particle_filter.weights)
            particle_filter.predict(time_slice_s)
            after = states()
            for source, destination, weight in zip(before, after, weights):
                transitions[time_index, source, destination] += weight
            row_sum = transitions[time_index].sum(axis=1)
            for state in range(state_count):
                if row_sum[state] > 0.0:
                    transitions[time_index, state] /= row_sum[state]
                else:
                    transitions[time_index, state, state] = 1.0
        return initial, transitions

    # 희소 경로. 입자가 실제로 지나간 (i -> k) 쌍만 담는다. 조밀 경로와
    # **같은 수를 만든다** — 0 을 적지 않을 뿐이다. 셀=탐지폭 격자에서
    # 조밀 배열은 표적 하나당 2.9 GB 라 빌드 자체가 불가능하다.
    sparse_steps: list[sparse.csr_matrix] = []
    for _ in range(steps):
        before = np.asarray(states(), dtype=np.int64)
        weights = np.asarray(particle_filter.weights, dtype=float)
        particle_filter.predict(time_slice_s)
        after = np.asarray(states(), dtype=np.int64)
        step = sparse.coo_matrix(
            (weights, (before, after)), shape=(state_count, state_count)
        ).tocsr()
        step.sum_duplicates()
        row_sum = np.asarray(step.sum(axis=1)).ravel()
        # 입자가 하나도 없던 셀은 조밀 경로와 똑같이 "제자리"로 둔다.
        empty = np.flatnonzero(row_sum <= 0.0)
        scale = np.where(row_sum > 0.0, 1.0 / np.maximum(row_sum, 1e-300), 0.0)
        step = sparse.diags(scale) @ step
        if empty.size:
            step = step + sparse.coo_matrix(
                (np.ones(empty.size), (empty, empty)),
                shape=(state_count, state_count),
            ).tocsr()
        sparse_steps.append(step.tocsr())
    return initial, tuple(sparse_steps)


def _segment_cell_lengths(
    grid: _SquareGrid, start: Point2D, end: Point2D
) -> dict[int, float]:
    """직선 구간이 각 셀 안에서 지나간 길이 (m).

    이동 중에도 gimbal 이 ±300 m 를 보고 있으므로, 이동 구간의 소인량은
    **지나간 셀들**에 떨어져야 한다. 도착셀에 전부 몰아주면 총량은 맞아도
    위치가 틀린다 (외접 5x5·96 s 대각 이동에서 1.082 km^2 가 엉뚱한 셀로
    간다).

    격자선을 만나는 파라미터 t 를 전부 모아 구간을 쪼개고, 각 구간의
    중점으로 셀을 판정한다. 부동소수 오차로 셀 경계에 정확히 앉는 경우를
    피하려고 **중점**을 쓴다.
    """

    dx = end.x - start.x
    dy = end.y - start.y
    total = (dx * dx + dy * dy) ** 0.5
    if total <= 1e-9:
        cell = grid.cell_of(start)
        return {} if cell is None else {cell: 0.0}

    cuts = {0.0, 1.0}
    left = grid.center.x - grid.half_side_m
    bottom = grid.center.y - grid.half_side_m
    for axis_start, delta, step, count in (
        (start.x - left, dx, grid.cell_width_m, grid.width),
        (start.y - bottom, dy, grid.cell_height_m, grid.height),
    ):
        if abs(delta) <= 1e-12:
            continue
        for line in range(count + 1):
            t = (line * step - axis_start) / delta
            if 0.0 < t < 1.0:
                cuts.add(t)

    ordered = sorted(cuts)
    lengths: dict[int, float] = {}
    for lower, upper in zip(ordered, ordered[1:]):
        if upper - lower <= 1e-12:
            continue
        middle = 0.5 * (lower + upper)
        cell = grid.cell_of(
            Point2D(start.x + dx * middle, start.y + dy * middle)
        )
        if cell is None:
            continue
        lengths[cell] = lengths.get(cell, 0.0) + (upper - lower) * total
    return lengths


def _adjacency(grid: _SquareGrid, *, reach_m: float) -> np.ndarray:
    """한 슬라이스에 실제로 갈 수 있는 셀만 잇는다.

    전에는 8방향 이웃을 **무조건** 이었다. 극좌표 격자는 처음부터
    ``reach_m`` 으로 끊고 있었으므로, 정사각 격자만 기구학을 안 보고 있었다.
    20 s 슬라이스에서 갈 수 있는 거리는 889 m 인데 외접 5x5 의 셀은
    1,914 m 다 — 이을 이유가 없는 이동이었다.

    이웃에 국한하지 않고 **도달 반경 안의 모든 셀**을 잇는다. 소인량을
    지나간 셀에 나눠 주므로 (``_swept_hazard``) 여러 셀을 가로지르는 이동도
    자기 값을 정직하게 받는다.
    """

    centers = [grid.center_of(cell) for cell in range(grid.cell_count)]
    result = np.zeros((grid.cell_count + 1, grid.cell_count + 1), dtype=bool)
    for source in range(grid.cell_count):
        result[source, source] = True
        for destination in range(grid.cell_count):
            if destination == source:
                continue
            if centers[source].distance_to(centers[destination]) <= reach_m:
                result[source, destination] = True
    result[grid.cell_count, grid.cell_count] = True
    if not result[: grid.cell_count, : grid.cell_count].sum() > grid.cell_count:
        raise ValueError(
            "no searcher can leave its cell within one time slice: cell size "
            f"{grid.cell_width_m:.0f} m vs reach {reach_m:.0f} m. Lower the "
            "time-slice count or the grid resolution."
        )
    return result


def _swept_hazard(
    grid: _SquareGrid,
    adjacency: np.ndarray,
    *,
    sweep_width_m: float,
    transit_speed_mps: float,
    search_speed_mps: float,
    slice_s: float,
    cell_scale: np.ndarray,
) -> dict[tuple[int, int], dict[int, float]]:
    """이동 중 소인까지 포함한 arc 별 셀 hazard.

    이동 구간에서도 search camera gimbal 은 켜져 있다. 그러니 이동시간을 통째로
    버리는 것은 틀렸고, 도착셀에 몰아주는 것도 틀렸다. 실제로 지나간
    셀들에 지나간 길이만큼 나눠 준다.

        이동 중  : hazard(j) += W * (셀 j 안에서 지나간 길이) / area(j)
        도착 후  : hazard(k) += W * 비행속도 * 남은시간 / area(k)

    **두 항 모두 비행속도를 쓴다.** 단위시간에 지면에 쌓이는 hazard 는

        ∫∫ gamma(x, z) dx dz = W * v

    로 경로 모양과 무관하다 (측방 적분 W 는 진행 '방향'에 대한 것이라
    경로가 휘어도 변하지 않는다). weave 를 해도 비행속도는 그대로이므로
    초당 소인량은 직선비행과 같다. 전에는 체류 항에 중심선속도 v_c 를
    써서 12.3 % 를 잃었다 — weave 비율만큼이다. 이동 항은 실제 거리를
    쓰고 체류 항만 중심선을 쓰니 두 항의 규약이 서로 달랐고, 그 결과
    **제자리 대기가 이동보다 11 % 불리하게** 값매겨졌다.

    ``cell_scale`` 은 셀별 지형 관측성 배율이다.
    """

    hazard: dict[tuple[int, int], dict[int, float]] = {}
    for source in range(grid.cell_count):
        start = grid.center_of(source)
        for destination in np.flatnonzero(adjacency[source, : grid.cell_count]):
            destination = int(destination)
            end = grid.center_of(destination)
            distance = start.distance_to(end)
            transit_s = distance / max(transit_speed_mps, 1e-12)
            cells: dict[int, float] = {}
            if distance > 1e-9:
                for cell, length in _segment_cell_lengths(grid, start, end).items():
                    if length <= 0.0:
                        continue
                    cells[cell] = cells.get(cell, 0.0) + (
                        sweep_width_m * length / grid.cell_area_m2
                    )
            remaining_s = max(0.0, slice_s - transit_s)
            if remaining_s > 0.0:
                cells[destination] = cells.get(destination, 0.0) + (
                    sweep_width_m * search_speed_mps * remaining_s
                    / grid.cell_area_m2
                )
            scaled = {
                cell: value * float(cell_scale[cell])
                for cell, value in cells.items()
                if value > 0.0
            }
            if scaled:
                hazard[(source, destination)] = scaled
    return hazard


def _polar_adjacency(
    grid: _PolarGrid,
    *,
    reach_m: float,
) -> np.ndarray:
    result = cell_adjacency(
        grid.probability_map,
        grid.cell_count,
        reach_m,
    )
    result[:, grid.cell_count] = False
    result[grid.cell_count, :] = False
    result[grid.cell_count, grid.cell_count] = True
    return result


def _destination_conflicts(
    grid: _OperationalGrid,
    separation_m: float,
) -> tuple[tuple[int, int], ...]:
    if separation_m <= 0.0:
        return ()
    pairs = []
    for left in range(grid.cell_count):
        for right in range(left + 1, grid.cell_count):
            if grid.center_of(left).distance_to(grid.center_of(right)) < separation_m:
                pairs.append((left, right))
    return tuple(pairs)


def _digest_transitions(digest, transitions) -> None:
    """전이열을 조밀/희소 공통으로, 그리고 **결정적으로** 해시에 넣는다.

    ``np.ascontiguousarray`` 에 희소행렬 목록을 넣으면 object 배열이 되고
    ``tobytes()`` 는 값이 아니라 **포인터**를 뱉는다. 그러면 지문이 실행마다
    달라져, 같은 인스턴스를 증명하기는커녕 같은 인스턴스를 서로 다르다고
    말하게 된다. 셀=탐지폭 격자(2,500 셀)는 희소로만 만들어지므로 그 규모에서
    조용히 깨질 자리였다.
    """

    steps = transition_ops.as_sequence(transitions)
    digest.update(f"steps:{len(steps)}".encode())
    for step in steps:
        if transition_ops.is_sparse(step):
            csr = step.tocsr()
            csr.sort_indices()
            digest.update(b"sparse")
            for part in (csr.indptr, csr.indices, csr.data):
                value = np.ascontiguousarray(part)
                digest.update(str(value.shape).encode())
                digest.update(value.dtype.str.encode())
                digest.update(value.tobytes())
        else:
            value = np.ascontiguousarray(step, dtype=float)
            digest.update(b"dense")
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())


def _common_input_fingerprint(
    problem: PathConstrainedProblem,
    targets: tuple[StoneTargetModel, ...],
    *,
    seed: int,
    grid: _OperationalGrid,
) -> str:
    """Stable proof that two planners received the same reduced problem.

    탐색자에서 **문제를 바꾸는 모든 것**이 들어가야 한다. 예전에는
    ``detection_rate`` 와 ``adjacency`` 만 담아서, 출발셀만 다른 두 인스턴스가
    같은 지문을 냈다 — 고리비 0.21 과 0.45 의 missing-person(adult) 8x8 이 실제로 그랬다.
    지문이 "같은 인스턴스"를 **증명**하는 물건인 이상 그건 결함이다.
    ``transit_fraction`` 과 ``swept_hazard`` 도 같은 이유로 들어간다.
    """

    digest = sha256()
    digest.update(f"{seed}:{grid.fingerprint}".encode())
    digest.update(
        ("starts:" + ",".join(str(s.start_state) for s in problem.searchers)).encode()
    )
    for searcher in problem.searchers:
        if searcher.swept_hazard is None:
            digest.update(b"swept:none")
            continue
        # 매핑은 순서가 보장되지 않으므로 정렬해서 넣는다.
        digest.update(b"swept:")
        for arc in sorted(searcher.swept_hazard):
            cells = searcher.swept_hazard[arc]
            digest.update(str(arc).encode())
            for cell in sorted(cells):
                digest.update(f"{cell}={float(cells[cell]):.17g};".encode())
    for array in (
        problem.initial_mass,
        *(searcher.detection_rate for searcher in problem.searchers),
        *(searcher.adjacency for searcher in problem.searchers),
        *(
            np.zeros(0) if searcher.transit_fraction is None
            else searcher.transit_fraction
            for searcher in problem.searchers
        ),
        *(target.initial_mass for target in targets),
    ):
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    _digest_transitions(digest, problem.transitions)
    for target in targets:
        _digest_transitions(digest, target.transitions)
    for target in targets:
        multiplier = np.ascontiguousarray(np.asarray(target.hazard_multiplier))
        digest.update(str(multiplier.shape).encode())
        digest.update(multiplier.dtype.str.encode())
        digest.update(multiplier.tobytes())
    return digest.hexdigest()


def _target_path_detection_probabilities(
    problem: PathConstrainedProblem,
    targets: tuple[StoneTargetModel, ...],
    paths: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    """Score any path bundle with every SPX target model on the common grid.

    표적 hazard 배율은 ``detection_rate`` 와 ``swept_hazard`` **둘 다**에
    걸어야 한다. 경로 배분 소인이 켜지면 ``swept_cells`` 가 ``swept_hazard``
    를 그대로 돌려주고 ``detection_rate`` 는 쓰이지 않으므로, 전자에만 걸면
    배율이 조용히 사라진다. 그렇게 되면 이 함수는 절단평면이 최적화한 것과
    **다른 목적함수**를 채점하게 되고, 계획법 간 비교의 근거가 무너진다.
    """

    values = []
    for target in targets:
        multiplier = np.asarray(target.hazard_multiplier, dtype=float)
        if multiplier.ndim != 0:
            raise ValueError(
                "same-grid H1 audit currently requires scalar target hazard multipliers"
            )
        scale = float(multiplier)
        searchers = tuple(
            replace(
                searcher,
                detection_rate=(
                    np.asarray(searcher.detection_rate, dtype=float) * scale
                ),
                swept_hazard=(
                    None
                    if searcher.swept_hazard is None
                    else {
                        arc: {cell: value * scale for cell, value in cells.items()}
                        for arc, cells in searcher.swept_hazard.items()
                    }
                ),
            )
            for searcher in problem.searchers
        )
        target_problem = PathConstrainedProblem(
            np.asarray(target.initial_mass, dtype=float),
            transition_ops.as_sequence(target.transitions),
            searchers,
        )
        values.append(path_detection_probability(target_problem, paths))
    return tuple(values)


@dataclass(frozen=True)
class StoneGridInstance:
    """SPX 와 MAPPO 가 **공유하는** 축소 문제 인스턴스 하나.

    v0.4 의 비교가 성립하는 지점이 여기다. 두 계획법이 같은 격자, 같은 시간
    슬라이스, 같은 지형결합 belief, 같은 탐색자 hazard, 같은 출발셀, 같은
    예약반경을 본다. ``fingerprint`` 가 그 사실의 증거이고, 두 계획법의
    결과 JSON 에 같은 값이 박히지 않으면 비교 주장은 무효다.

    경로를 고르는 방법만 다르다.

    * ``solve_stone_grid_paths``  — Stone 절단평면 (최적성 인증 있음)
    * ``learning/spx_env``        — MAPPO 정책 (인증 없음, 대신 확장 가능)
    """

    mission: MissionConfig
    sensor: SensorSpec
    config: StoneSPXRouteConfig
    seed: int
    grid: _OperationalGrid
    problem: PathConstrainedProblem
    target_models: tuple[StoneTargetModel, ...]
    target_names: tuple[str, ...]
    starts: tuple[int, ...]
    start_quantization_max_m: float
    scales: tuple[float, ...]
    class_names: tuple[str, ...]
    slice_s: float
    reservation_neighborhoods: tuple[frozenset[int], ...] | None
    full_slice_visit_hazards: np.ndarray
    initial_positions: tuple[Point2D, ...]
    terrain: object | None

    @property
    def time_slice_count(self) -> int:
        return int(self.config.time_slice_count)

    @property
    def cell_count(self) -> int:
        return int(self.grid.cell_count)

    @property
    def searcher_count(self) -> int:
        return len(self.problem.searchers)

    @property
    def fingerprint(self) -> str:
        return _common_input_fingerprint(
            self.problem, self.target_models, seed=self.seed, grid=self.grid
        )

    def score_paths(self, paths: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
        """표적별 정확한 탐지확률. **모든 계획법이 이 함수만 통과한다.**"""

        return _target_path_detection_probabilities(
            self.problem, self.target_models, paths
        )

    def worst_target_detection_probability(
        self, paths: tuple[tuple[int, ...], ...]
    ) -> float:
        """SPX 의 minimax 목적함수값. 작은 쪽이 곧 최악 표적이다."""

        return min(self.score_paths(paths))


@dataclass(frozen=True, slots=True)
class StoneGridPathSolution:
    """경로 묶음 하나와, 그 경로에 대해 말할 수 있는 것 전부."""

    paths: tuple[tuple[int, ...], ...]
    target_detection_probabilities: tuple[float, ...]
    lower_bound_nondetection: float
    upper_bound_nondetection: float
    relative_optimality_gap: float
    iterations: int
    converged: bool
    fallback_used: bool
    fallback_reason: str | None
    runtime_s: float
    discarded_warm_starts: int = 0
    #: 반복별 (iteration, 인증하한, 최선상한, 상대갭). SPX 만 채운다
    #: (휴리스틱에는 인증이 없다). Ch2 수렴성 그림의 유일한 입력.
    bound_history: tuple[tuple[int, float, float, float], ...] = ()
    #: 반복별 (iteration, 단계, 상태, 초). 시간이 병목인지 절단이 병목인지
    #: 가른다 — 예산을 어디에 넣을지는 이것으로 정한다.
    master_status_history: tuple[tuple[int, str, str, float], ...] = ()
    #: 최종 계획의 출처. ``"warm-start"`` 면 master 가 휴리스틱을 못 이겼다.
    incumbent_source: str = "unknown"
    #: 휴리스틱 기준선. master 개선폭 = 이 값 - upper_bound_nondetection.
    warm_start_nondetection: float = float("nan")


def build_stone_grid_instance(
    mission: MissionConfig,
    sensor: SensorSpec,
    *,
    prior: TargetPrior,
    terrain,
    targets: tuple[StoneSPXTargetSpec, ...],
    searcher_classes: tuple[StoneSPXSearcherClass, ...],
    initial_positions: tuple[Point2D, ...],
    planning_time_s: float,
    initial_belief_delay_s: float,
    seed: int,
    config: StoneSPXRouteConfig,
) -> StoneGridInstance:
    """지형결합 belief -> 시공간 Markov -> 축소 경로제약 문제.

    계획법을 고르기 **전에** 끝나는 모든 일을 여기서 한다. 그래서 SPX 와
    MAPPO 가 같은 인스턴스를 받는다는 것이 호출 구조로 보장된다.
    """

    if len(initial_positions) != mission.uav_count:
        raise ValueError("SPX requires one initial position per searcher")
    if sum(item.count for item in searcher_classes) != mission.uav_count:
        raise ValueError("SPX searcher class counts must equal the UAV count")
    if not targets:
        raise ValueError("SPX requires at least one target model")
    if any(not 0.0 < item.hazard_scale <= 1.0 for item in searcher_classes):
        raise ValueError("operational SPX hazard scales must be in (0, 1]")
    if planning_time_s <= 0.0 or initial_belief_delay_s < 0.0:
        raise ValueError("SPX planning time must be positive and delay nonnegative")
    if config.hazard_calibration not in {"fixed", "effective-sweep-width"}:
        raise ValueError(f"unknown SPX hazard calibration: {config.hazard_calibration}")
    if config.square_fit not in {"inscribed", "circumscribed"}:
        raise ValueError(f"unknown square grid fit: {config.square_fit!r}")

    slice_s = planning_time_s / config.time_slice_count
    if config.grid_kind == "square":
        grid: _OperationalGrid = _SquareGrid.fitted(
            mission, config.grid_width, config.grid_height, config.square_fit
        )
    elif config.grid_kind == "operational-polar":
        probability_map = PolarProbabilityMap.build(
            mission,
            prior,
            config.radial_step_m,
            config.angular_bin_count,
            terrain=terrain if config.terrain_weighting else None,
            terrain_mode_probabilities=(
                targets[0].profile.imm5_behavior.mode_probabilities
                if config.terrain_weighting
                else None
            ),
        )
        grid = _PolarGrid(mission, probability_map)
    else:
        raise ValueError(f"unknown SPX grid kind: {config.grid_kind}")

    # 한 슬라이스를 한 셀 안에서 다 쓸 때의 소인 면적. 위 ``_swept_hazard``
    # 와 **같은 규약**이어야 한다 — 소인량은 W x (비행속도) x 시간이다.
    full_slice_swept_area_m2 = (
        sensor.effective_sweep_width_m * mission.search_speed_mps * slice_s
    )
    full_slice_visit_hazards = (
        full_slice_swept_area_m2 / grid.cell_areas_m2
        if config.hazard_calibration == "effective-sweep-width"
        else np.full(grid.cell_count, config.visit_hazard, dtype=float)
    )

    target_models: list[StoneTargetModel] = []
    target_names: list[str] = []
    for index, target in enumerate(targets):
        belief = TerrainParticleBelief(
            mission,
            target.profile,
            prior=prior,
            particle_count=config.particle_count,
            seed=seed ^ (0x5350_5800 + index),
            radial_step_m=max(mission.search_radius_m, 1.0),
            angular_bin_count=8,
            terrain=terrain,
            terrain_coupling=(
                BELIEF_TERRAIN_FULL
                if config.terrain_weighting
                else BELIEF_TERRAIN_OFF
            ),
            roughening=False,
        )
        if initial_belief_delay_s > 0.0:
            belief.filter.predict(initial_belief_delay_s)
        initial, transitions = _target_markov(
            belief.filter,
            grid,
            sparse_transitions=_use_sparse_transitions(
                config.sparse_transitions, grid.cell_count
            ),
            time_slice_s=slice_s,
            time_slice_count=config.time_slice_count,
        )
        target_models.append(
            StoneTargetModel(initial, transitions, target.hazard_multiplier)
        )
        target_names.append(target.profile.name)

    adjacency = (
        _adjacency(
            grid, reach_m=mission.transit_speed_mps * slice_s
        )
        if isinstance(grid, _SquareGrid)
        else _polar_adjacency(
            grid,
            reach_m=(
                sensor.centerline_search_speed_mps(mission.search_speed_mps)
                * slice_s
            ),
        )
    )
    state_count = grid.cell_count + 1
    terrain_effectiveness = (
        np.asarray(
            [
                terrain.observability_weight(
                    grid.center_of(cell).x, grid.center_of(cell).y
                )
                for cell in range(grid.cell_count)
            ],
            dtype=float,
        )
        if config.terrain_weighting
        else np.ones(grid.cell_count, dtype=float)
    )
    class_names: list[str] = []
    scales: list[float] = []
    for item in searcher_classes:
        class_names.extend([item.name] * item.count)
        scales.extend([item.hazard_scale] * item.count)

    transit_fraction = np.zeros((state_count, state_count), dtype=float)
    for source in range(grid.cell_count):
        for destination in np.flatnonzero(adjacency[source, : grid.cell_count]):
            distance = grid.center_of(source).distance_to(grid.center_of(int(destination)))
            transit_fraction[source, destination] = max(
                0.0,
                1.0 - distance / max(mission.transit_speed_mps * slice_s, 1e-12),
            )
    transit_fraction[grid.cell_count, grid.cell_count] = 1.0

    # Multiple aircraft may share a discretised start cell. Occupancy applies
    # to the destinations selected for each search slice, not to the launch
    # state before slice 0. Forcing distinct starts can otherwise move
    # co-located aircraft several kilometres on a reduced 3x3 grid.
    starts = tuple(grid.nearest_cell(point) for point in initial_positions)
    start_quantization_max_m = max(
        (
            point.distance_to(grid.center_of(cell))
            for point, cell in zip(initial_positions, starts)
        ),
        default=0.0,
    )
    # 정사각 격자에서는 이동 중 소인을 지나간 셀에 나눠 준다. 그러면
    # ``transit_fraction`` (이동시간을 통째로 버리는 모형)은 쓰지 않는다.
    # 극좌표 격자는 아직 도착셀 모형이라 기존 경로를 유지한다.
    base_swept = (
        _swept_hazard(
            grid,
            adjacency,
            sweep_width_m=sensor.effective_sweep_width_m,
            transit_speed_mps=mission.transit_speed_mps,
            search_speed_mps=mission.search_speed_mps,
            slice_s=slice_s,
            cell_scale=terrain_effectiveness,
        )
        if isinstance(grid, _SquareGrid)
        else None
    )
    searchers = tuple(
        SearcherModel(
            start,
            np.concatenate(
                (
                    np.broadcast_to(
                        full_slice_visit_hazards * scale * terrain_effectiveness,
                        (config.time_slice_count, grid.cell_count),
                    ).copy(),
                    np.zeros((config.time_slice_count, 1)),
                ),
                axis=1,
            ),
            adjacency,
            None if base_swept is not None else transit_fraction,
            swept_hazard=(
                None
                if base_swept is None
                else {
                    arc: {cell: value * scale for cell, value in cells.items()}
                    for arc, cells in base_swept.items()
                }
            ),
        )
        for start, scale in zip(starts, scales)
    )
    frozen_targets = tuple(target_models)
    representative_initial = np.mean(
        [target.initial_mass for target in frozen_targets], axis=0
    )
    representative_transitions = transition_ops.mean_of(
        [transition_ops.as_sequence(target.transitions) for target in frozen_targets]
    )
    problem = PathConstrainedProblem(
        representative_initial,
        representative_transitions,
        searchers,
    )
    reservation_neighborhoods = (
        tuple(
            frozenset(
                {
                    other
                    for other in range(grid.cell_count)
                    if grid.center_of(cell).distance_to(grid.center_of(other))
                    < config.reservation_separation_m
                }
                | {cell}
            )
            for cell in range(grid.cell_count)
        )
        + (frozenset({grid.cell_count}),)
        if config.reservation_separation_m > 0.0
        else None
    )
    return StoneGridInstance(
        mission=mission,
        sensor=sensor,
        config=config,
        seed=seed,
        grid=grid,
        problem=problem,
        target_models=frozen_targets,
        target_names=tuple(target_names),
        starts=starts,
        start_quantization_max_m=start_quantization_max_m,
        scales=tuple(scales),
        class_names=tuple(class_names),
        slice_s=slice_s,
        reservation_neighborhoods=reservation_neighborhoods,
        full_slice_visit_hazards=full_slice_visit_hazards,
        initial_positions=tuple(initial_positions),
        terrain=terrain,
    )


def _warm_start_candidates(
    instance: StoneGridInstance,
) -> list[tuple[tuple[int, ...], ...]]:
    """SPX 정수 master 에 넣을 초기 실행가능 경로들.

    v0.4 는 H1/H2 를 **비교군으로 보고하지 않는다.** 여기서만 쓴다 — 절단평면
    master 는 초기 incumbent 가 있을 때 상한이 훨씬 빨리 내려오고, 그 상한이
    곧 최적성 인증폭의 절반이다. 대표 표적 하나와 표적별 문제 각각에 대해
    돌린 뒤 **최악 표적 PD 가 가장 큰** 후보를 고른다.
    """

    problem = instance.problem
    config = instance.config

    def attempt(solver, target_problem):
        """휴리스틱이 실행 불가를 보고하면 후보 하나를 포기한다.

        H1/H2 는 예약이 도달 가능한 첫 수를 전부 막으면
        :class:`InfeasiblePathError` 를 낸다 (6대가 16셀 격자의 중앙에 몰려
        출발할 때 실제로 난다). 그건 **warm start 가 없다**는 뜻이지
        **SPX 가 풀 수 없다**는 뜻이 아니다. 예전에는 이 예외가 그대로
        올라가 solve 전체를 죽였다 — 축소 격자 seed 하나가 통째로 날아갔다.
        """

        try:
            return solver(
                target_problem,
                sharing="shared-reserved",
                seed=instance.seed,
                reservation_neighborhoods=instance.reservation_neighborhoods,
                forbid_opposing_edge_swaps=config.forbid_opposing_edge_swaps,
            ).paths
        except InfeasiblePathError:
            return None

    candidates = [
        paths
        for paths in (
            attempt(team_receding_horizon, problem),
            attempt(team_h2_receding_horizon, problem),
        )
        if paths is not None
    ]
    for target in instance.target_models:
        multiplier = np.asarray(target.hazard_multiplier, dtype=float)
        if multiplier.ndim != 0:
            continue
        target_problem = PathConstrainedProblem(
            np.asarray(target.initial_mass, dtype=float),
            transition_ops.as_sequence(target.transitions),
            tuple(
                replace(
                    searcher,
                    detection_rate=(
                        np.asarray(searcher.detection_rate, dtype=float)
                        * float(multiplier)
                    ),
                )
                for searcher in problem.searchers
            ),
        )
        candidates.extend(
            paths
            for paths in (
                attempt(team_receding_horizon, target_problem),
                attempt(team_h2_receding_horizon, target_problem),
            )
            if paths is not None
        )
    return candidates


def solve_stone_grid_paths(
    instance: StoneGridInstance,
    *,
    method: StoneGridMethod = "stone-spx",
) -> StoneGridPathSolution:
    """Stone 절단평면(또는 그 warm start 휴리스틱)으로 셀 경로를 고른다."""

    config = instance.config
    started = perf_counter()
    if method == "stone-spx":
        warm_candidates = _warm_start_candidates(instance)
        # 후보가 하나도 없을 수 있다. 6대가 한 셀에서 출발하면 H1/H2 는
        # 예약이 모든 첫 수를 막아 실행 불가를 낸다 — 이송 후 TP 상공에서
        # 시작하는 조건이 정확히 그렇다. warm start 는 **가속 장치**이지
        # 필수가 아니므로, 없으면 master 가 스스로 찾게 둔다
        # (tests: test_solving_without_any_warm_start_still_certifies).
        initial_paths = (
            max(
                warm_candidates,
                key=lambda candidate: min(instance.score_paths(candidate)),
            )
            if warm_candidates
            else None
        )
        solution = stone_spx_cutting_plane(
            instance.problem,
            targets=instance.target_models,
            occupancy_limits=config.occupancy_limit,
            relative_tolerance=config.relative_tolerance,
            max_iterations=config.max_iterations,
            mip_relative_gap=config.mip_relative_gap,
            aggregate_identical_searchers=config.aggregate_identical_searchers,
            destination_conflicts=_destination_conflicts(
                instance.grid, config.reservation_separation_m
            ),
            forbid_opposing_edge_swaps=config.forbid_opposing_edge_swaps,
            master_time_limit_s=config.master_time_limit_s,
            initial_paths=initial_paths,
            initial_path_candidates=tuple(warm_candidates),
            persistent_master=config.persistent_master,
            continuous_relaxation_iterations=config.continuous_relaxation_iterations,
            exact_survival_milp=config.exact_survival_milp,
            exact_time_limit_s=config.exact_time_limit_s,
            exact_backend=config.exact_backend,
            master_backend=config.master_backend,
            local_improvement_passes=config.local_improvement_passes,
        )
        paths = solution.paths
        return StoneGridPathSolution(
            paths=paths,
            target_detection_probabilities=tuple(
                1.0 - value for value in solution.target_nondetection_probabilities
            ),
            lower_bound_nondetection=solution.lower_bound_nondetection,
            upper_bound_nondetection=solution.upper_bound_nondetection,
            relative_optimality_gap=solution.relative_optimality_gap,
            iterations=solution.iterations,
            converged=solution.converged,
            fallback_used=solution.fallback_used,
            fallback_reason=solution.fallback_reason,
            runtime_s=perf_counter() - started,
            discarded_warm_starts=solution.discarded_warm_starts,
            bound_history=solution.bound_history,
            master_status_history=solution.master_status_history,
            incumbent_source=solution.incumbent_source,
            warm_start_nondetection=solution.warm_start_nondetection,
        )
    if method in {"team-h1", "team-h2"}:
        solver = (
            team_receding_horizon if method == "team-h1" else team_h2_receding_horizon
        )
        result = solver(
            instance.problem,
            sharing="shared-reserved",
            seed=instance.seed,
            reservation_neighborhoods=instance.reservation_neighborhoods,
            forbid_opposing_edge_swaps=config.forbid_opposing_edge_swaps,
        )
        target_pds = instance.score_paths(result.paths)
        return StoneGridPathSolution(
            paths=result.paths,
            target_detection_probabilities=target_pds,
            lower_bound_nondetection=float("nan"),
            upper_bound_nondetection=1.0 - min(target_pds),
            relative_optimality_gap=float("nan"),
            iterations=result.evaluations,
            converged=True,
            fallback_used=False,
            fallback_reason=None,
            runtime_s=perf_counter() - started,
        )
    raise ValueError(f"unknown Stone-grid planning method: {method}")


def flown_routes_from_paths(
    instance: StoneGridInstance,
    paths: tuple[tuple[int, ...], ...],
    *,
    label: str,
) -> list[Route]:
    """셀 경로 -> 실제 비행경로. **모든 계획법이 이 변환 하나만 쓴다.**

    이동 시간을 슬라이스에서 먼저 빼고 남은 시간만 소인에 준다. 그래서 먼
    셀로 건너뛰는 계획은 자기 비용을 자기가 낸다.
    """

    mission = instance.mission
    sensor = instance.sensor
    routes: list[Route] = []
    for vehicle_id, path in enumerate(paths):
        segments: list[PathSegment] = []
        current = instance.initial_positions[vehicle_id]
        for time_index, cell in enumerate(path):
            destination = instance.grid.center_of(cell)
            distance = current.distance_to(destination)
            if distance > 1e-9:
                segments.append(PathSegment(current, destination, False))
            remaining = max(
                0.0, instance.slice_s - distance / mission.transit_speed_mps
            )
            sweep, current = local_sweep(
                destination,
                sensor.centerline_search_speed_mps(mission.search_speed_mps)
                * remaining,
                mission,
                sensor,
                phase_rad=(vehicle_id + time_index) * 2.0 * np.pi / mission.uav_count,
            )
            segments.extend(
                replace(segment, detection_scale=instance.scales[vehicle_id])
                for segment in sweep
            )
        routes.append(Route(label, vehicle_id, tuple(segments)))
    return routes


def _deconfliction_counts(
    instance: StoneGridInstance,
    paths: tuple[tuple[int, ...], ...],
) -> tuple[int, int]:
    """(같은 슬라이스 중복 점유 수, 반대방향 edge swap 수)."""

    duplicate_count = sum(
        len(column) - len(set(column)) for column in zip(*paths)
    )
    swaps = 0
    searcher_count = len(paths)
    for time_index in range(instance.time_slice_count):
        sources = [
            instance.starts[index]
            if time_index == 0
            else paths[index][time_index - 1]
            for index in range(searcher_count)
        ]
        destinations = [path[time_index] for path in paths]
        for left in range(searcher_count):
            for right in range(left + 1, searcher_count):
                swaps += int(
                    sources[left] == destinations[right]
                    and sources[right] == destinations[left]
                    and sources[left] != sources[right]
                )
    return duplicate_count, swaps


def route_plan_from_solution(
    instance: StoneGridInstance,
    solution: StoneGridPathSolution,
    *,
    method: str,
) -> StoneSPXRoutePlan:
    """경로 묶음 하나를 챕터가 그대로 평가에 넘길 수 있는 계획으로 포장한다."""

    paths = solution.paths
    target_pds = solution.target_detection_probabilities
    duplicate_count, swaps = _deconfliction_counts(instance, paths)
    grid = instance.grid
    representative_markov = SpaceTimeMarkovModel(
        initial=np.mean(
            [target.initial_mass for target in instance.target_models], axis=0
        ),
        transitions=np.mean(
            [target.transitions for target in instance.target_models], axis=0
        ),
        cell_centers=tuple(grid.center_of(cell) for cell in range(grid.cell_count)),
        outside_index=grid.cell_count,
        time_slice_s=instance.slice_s,
        time_slice_count=instance.time_slice_count,
    )
    return StoneSPXRoutePlan(
        routes=flown_routes_from_paths(instance, paths, label=method),
        markov=representative_markov,
        diagnostics=StoneSPXRouteDiagnostics(
            planning_method=method,
            common_input_fingerprint=instance.fingerprint,
            detection_probability=min(target_pds),
            no_detection_probability=1.0 - min(target_pds),
            target_detection_probabilities=tuple(
                zip(instance.target_names, target_pds)
            ),
            relative_optimality_gap=solution.relative_optimality_gap,
            lower_bound_nondetection=solution.lower_bound_nondetection,
            upper_bound_nondetection=solution.upper_bound_nondetection,
            iterations=solution.iterations,
            converged=solution.converged,
            fallback_used=solution.fallback_used,
            fallback_reason=solution.fallback_reason,
            runtime_s=solution.runtime_s,
            duplicate_assignment_ratio=duplicate_count
            / max(instance.time_slice_count * instance.searcher_count, 1),
            opposing_edge_swaps=swaps,
            communication_available_ratio=1.0,
            outside_probability=float(representative_markov.initial[-1]),
            searcher_classes=instance.class_names,
            start_cells=instance.starts,
            start_quantization_max_m=instance.start_quantization_max_m,
            hazard_calibration=instance.config.hazard_calibration,
            full_slice_visit_hazard=float(np.mean(instance.full_slice_visit_hazards)),
            support_total_width_m=2.0 * instance.sensor.coverage_half_width_m,
            effective_sweep_width_m=instance.sensor.effective_sweep_width_m,
            cell_area_m2=grid.cell_area_m2,
            cell_area_min_m2=float(np.min(grid.cell_areas_m2)),
            cell_area_max_m2=float(np.max(grid.cell_areas_m2)),
            grid_kind=instance.config.grid_kind,
            grid_shape=grid.grid_shape,
            terrain_weighting=instance.config.terrain_weighting,
        ),
        assignments=paths,
        planner_terrain=instance.terrain,
    )


def plan_stone_grid_routes(
    mission: MissionConfig,
    sensor: SensorSpec,
    *,
    prior: TargetPrior,
    terrain,
    targets: tuple[StoneSPXTargetSpec, ...],
    searcher_classes: tuple[StoneSPXSearcherClass, ...],
    initial_positions: tuple[Point2D, ...],
    planning_time_s: float,
    initial_belief_delay_s: float,
    seed: int,
    config: StoneSPXRouteConfig,
    method: StoneGridMethod = "stone-spx",
) -> StoneSPXRoutePlan:
    """인스턴스 구축 -> 경로 선택 -> 공통 경로변환, 한 번에."""

    instance = build_stone_grid_instance(
        mission,
        sensor,
        prior=prior,
        terrain=terrain,
        targets=targets,
        searcher_classes=searcher_classes,
        initial_positions=initial_positions,
        planning_time_s=planning_time_s,
        initial_belief_delay_s=initial_belief_delay_s,
        seed=seed,
        config=config,
    )
    solution = solve_stone_grid_paths(instance, method=method)
    return route_plan_from_solution(instance, solution, method=method)


def plan_stone_spx_routes(*args, **kwargs) -> StoneSPXRoutePlan:
    """SPX entry point."""

    return plan_stone_grid_routes(*args, **kwargs, method="stone-spx")
