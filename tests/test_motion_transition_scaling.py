"""Regression tests for ``TargetMotionSpec.transition_matrix_for_elapsed``.

The legacy ``imm``/``isotropic`` branch referenced ``ratio`` before it was
bound (it was only assigned inside the ``imm5`` branch, which returns early),
so any non-imm5 spec raised ``NameError``. These tests pin the fix and the
scaling contract: identity at t=0, full per-step matrix at t>=step_s, linear
in between, and every row a valid probability distribution.
"""

from __future__ import annotations

from cpp_search.core.motion import TargetMotionSpec


def _spec(model: str) -> TargetMotionSpec:
    return TargetMotionSpec(max_speed_mps=11.0, step_s=30.0, motion_model=model)


def _rows_sum_to_one(matrix) -> bool:
    return all(abs(sum(row) - 1.0) < 1e-9 for row in matrix)


def test_imm_elapsed_does_not_raise_and_scales() -> None:
    spec = _spec("imm")
    base = spec.effective_transition_matrix

    # t=0 -> identity.
    identity = spec.transition_matrix_for_elapsed(0.0)
    assert identity == tuple(
        tuple(float(r == c) for c in range(len(base))) for r in range(len(base))
    )

    # t >= step_s -> the full per-step matrix, clamped (does not overshoot).
    assert spec.transition_matrix_for_elapsed(30.0) == base
    assert spec.transition_matrix_for_elapsed(120.0) == base

    # Halfway a step: linear blend between identity and the base matrix.
    half = spec.transition_matrix_for_elapsed(15.0)
    for r, base_row in enumerate(base):
        for c, target in enumerate(base_row):
            expected = float(r == c) + 0.5 * (target - float(r == c))
            assert abs(half[r][c] - expected) < 1e-9
    assert _rows_sum_to_one(half)


def test_isotropic_elapsed_does_not_raise() -> None:
    spec = _spec("isotropic")
    matrix = spec.transition_matrix_for_elapsed(5.0)
    assert _rows_sum_to_one(matrix)
