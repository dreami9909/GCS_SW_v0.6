"""시공간 Markov 전이를 조밀/희소 공통으로 다루는 얇은 계층.

전이행렬 ``P_t[i, k]`` 는 격자를 촘촘히 할수록 **셀 수의 제곱 × 슬라이스 수**
로 커진다. 셀=탐지폭(400 m) 격자는 2,500 셀 × 59 슬라이스라 조밀 배열로
표적 하나당 2.9 GB, 케이스당 8.7 GB 가 되어 SPX 를 돌리기 전에 메모리에서
죽는다.

그런데 표적은 한 슬라이스에 몇백 미터밖에 못 가므로 각 행에서 0 이 아닌
항목은 인접 몇 칸뿐이다. 0 을 저장하지 않으면 그대로 들어간다.

**이것은 근사가 아니다.** 0 은 곱해도 더해도 0 이므로 결과가 비트 단위로
같다. 값이 달라지는 경우는 작은 값을 잘라낼 때뿐이고, 여기서는 자르지
않는다 (``tests/test_review_regressions.py`` 의 조밀-희소 동치 검사 참조).

전이는 **행렬-벡터 곱으로만** 쓰인다 (전방 ``v @ P``, 후방 ``P @ v``).
numpy 배열과 scipy 희소행렬 모두 ``@`` 를 지원하므로, 여기서는 "2차원
연산자들의 순서열"이라는 공통 모양만 맞춰 준다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

__all__ = [
    "as_sequence",
    "dimensions",
    "is_sparse",
    "mean_of",
    "validate",
]


def is_sparse(matrix: Any) -> bool:
    return sparse.issparse(matrix)


def as_sequence(transitions: Any) -> tuple[Any, ...]:
    """``[step, i, k]`` 조밀 배열도, 2차원 연산자 목록도 같은 모양으로."""

    if sparse.issparse(transitions):
        raise TypeError(
            "transitions must be a sequence of per-step matrices, not one matrix"
        )
    if isinstance(transitions, np.ndarray):
        if transitions.ndim != 3:
            raise ValueError("dense transitions must be [step, state, state]")
        return tuple(np.asarray(step, dtype=float) for step in transitions)
    if isinstance(transitions, Sequence):
        return tuple(
            step if sparse.issparse(step) else np.asarray(step, dtype=float)
            for step in transitions
        )
    raise TypeError(f"unsupported transition container: {type(transitions).__name__}")


def dimensions(sequence: Sequence[Any]) -> tuple[int, int | None]:
    """``(스텝 수, 상태 수)``. 스텝이 없으면 상태 수는 알 수 없다."""

    steps = len(sequence)
    if steps == 0:
        return 0, None
    rows, columns = sequence[0].shape
    if rows != columns:
        raise ValueError("each transition step must be square")
    return steps, int(rows)


def validate(
    sequence: Sequence[Any],
    *,
    states: int,
    steps: int,
    what: str = "transitions",
) -> None:
    """모양 · 음수 없음 · 행합 1 을 조밀/희소 공통으로 검사한다."""

    if len(sequence) != steps:
        raise ValueError(
            f"{what} must have {steps} steps, found {len(sequence)}"
        )
    for index, step in enumerate(sequence):
        if step.shape != (states, states):
            raise ValueError(
                f"{what}[{index}] must be [{states}, {states}], "
                f"found {tuple(step.shape)}"
            )
        if sparse.issparse(step):
            data = step.data
            row_sums = np.asarray(step.sum(axis=1)).ravel()
        else:
            data = step
            row_sums = step.sum(axis=1)
        if data.size and float(np.min(data)) < 0.0:
            raise ValueError(f"{what} cannot be negative")
        if not np.allclose(row_sums, 1.0):
            raise ValueError(f"{what} must be row-stochastic")


def mean_of(sequences: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    """여러 표적의 전이를 슬라이스별로 평균낸다 (대표 인스턴스용).

    ``np.mean`` 은 희소행렬 목록에서 동작하지 않으므로 직접 더해서 나눈다.
    """

    if not sequences:
        raise ValueError("at least one transition sequence is required")
    steps = len(sequences[0])
    if any(len(item) != steps for item in sequences):
        raise ValueError("transition sequences must share the step count")
    count = float(len(sequences))
    result = []
    for index in range(steps):
        total = sequences[0][index]
        for other in sequences[1:]:
            total = total + other[index]
        result.append(total / count)
    return tuple(result)
