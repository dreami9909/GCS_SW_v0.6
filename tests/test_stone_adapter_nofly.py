"""Regression tests for ``_mask_no_fly`` (SPX no-fly masking).

These isolate the masking helper with a tiny hand-built problem so they need
neither scipy nor a terrain file. The bug they guard against: masking a cell
out of ``adjacency`` without also pruning ``swept_hazard`` makes
``SearcherModel.__post_init__`` reject the now-detached arc, which crashed
certified search whenever any no-fly cell appeared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cpp_search.theory.path_constrained import (
    PathConstrainedProblem,
    SearcherModel,
)
from qt_gcs.planning.stone_adapter import _mask_no_fly


@dataclass(frozen=True)
class _FakeInstance:
    """Minimal stand-in exposing the single attribute ``_mask_no_fly`` uses."""

    problem: PathConstrainedProblem


def _build_problem(*, self_loops: bool = True) -> PathConstrainedProblem:
    states = 4
    horizon = 2
    detection_rate = np.full((horizon, states), 0.1)
    adjacency = np.ones((states, states), dtype=bool)
    if not self_loops:
        np.fill_diagonal(adjacency, False)
    # Cell 0 is the launch cell; every arc carries swept hazard, exactly as the
    # square-grid Stone-SPX instances do.
    swept_hazard = {
        (source, destination): {source: 0.2, destination: 0.3}
        for source in range(states)
        for destination in range(states)
        if adjacency[source, destination]
    }
    searcher = SearcherModel(
        start_state=0,
        detection_rate=detection_rate,
        adjacency=adjacency,
        swept_hazard=swept_hazard,
    )
    transitions = np.stack([np.full((states, states), 1.0 / states)])
    return PathConstrainedProblem(
        initial_mass=np.full(states, 1.0 / states),
        transitions=transitions,
        searchers=(searcher,),
    )


def test_mask_no_fly_prunes_swept_hazard_without_crashing() -> None:
    instance = _FakeInstance(problem=_build_problem())

    masked = _mask_no_fly(instance, np.asarray([2], dtype=int))

    searcher = masked.problem.searchers[0]
    # The no-fly cell is unreachable in both directions.
    assert not searcher.adjacency[2, :].any()
    assert not searcher.adjacency[:, 2].any()
    # No surviving swept-hazard arc touches the masked cell as an endpoint...
    assert all(
        2 not in (source, destination)
        for (source, destination) in searcher.swept_hazard
    )
    # ...nor credits coverage inside the masked cell.
    assert all(
        2 not in cells for cells in searcher.swept_hazard.values()
    )
    # Arcs between flyable cells are kept.
    assert (0, 1) in searcher.swept_hazard
    assert searcher.swept_hazard[(0, 1)] == {0: 0.2, 1: 0.3}


def test_mask_no_fly_empty_returns_original_instance() -> None:
    instance = _FakeInstance(problem=_build_problem())
    assert _mask_no_fly(instance, np.empty(0, dtype=int)) is instance


def test_mask_no_fly_isolated_start_raises() -> None:
    # Without a self-loop the launch cell only survives through its neighbours,
    # so masking them all genuinely isolates it.
    instance = _FakeInstance(problem=_build_problem(self_loops=False))
    with_all_but_start = np.asarray([1, 2, 3], dtype=int)
    try:
        _mask_no_fly(instance, with_all_but_start)
    except ValueError as error:
        assert "no-fly" in str(error)
    else:  # pragma: no cover - explicit failure is clearer than an assert
        raise AssertionError("expected isolated start cell to raise ValueError")
