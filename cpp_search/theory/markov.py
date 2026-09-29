"""입자 belief -> 이산 시공간 Markov 모델.

FAB는 상태공간과 전이행렬을 요구하는데, 우리 belief는 입자 집합이다.
이 파일이 그 사이를 잇는다.

상태공간 = ``probability.PolarProbabilityMap``의 셀 + **흡수 외부상태 1개**

핵심 수식
---------
* 초기분포 (``estimate_space_time_markov``)

      pi_0(x) = sum_{입자 i in x} w_i

* 전이행렬 — 입자를 한 슬라이스 굴려 before/after 셀 쌍을 세는 몬테카를로 추정

      N_t(x, y) = sum_{i: before=x, after=y} w_i
      P_t(x, y) = N_t(x, y) / sum_y N_t(x, y)

  방문되지 않은 상태 행은 P(x,x) = 1로 둔다(행 확률 보존). 외부상태는
  흡수 상태이므로 자기 자신으로만 간다.
  **부작용**: 이 함수는 입자필터를 time_slice_count-1 슬라이스만큼 실제로
  전진시킨다. 반환된 모델과 필터 상태가 일치하도록 의도한 것이다.

* 탐지효율 (``effectiveness``)

      w[t, x] = max(floor, terrain.observability_weight(x))
      w[t, outside] = 1e-12          (외부는 사실상 탐지 불가)

  Ch2/Ch5에서 지형 관측성이 FAB에 들어가는 통로가 바로 여기다.

* 도달 가능성 (``cell_adjacency``)

      A(x, y) = 1  iff  |c_x - c_y| <= reach

  reach = 중심선 진행속도 * 슬라이스 길이. 경로제약 FAB와 빔 탐색이 쓴다.

의존
----
* 위: ``cpp_search.core.models``, ``cpp_search.core.probability``, numpy.
* 아래: ``planning/team_planner``, ``chapters/chapter2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt, tau

import numpy as np

from cpp_search.core.models import MissionConfig, Point2D
from cpp_search.core.probability import PolarProbabilityMap


def polar_cell_index(
    point: Point2D,
    mission: MissionConfig,
    probability_map: PolarProbabilityMap,
) -> int:
    """Return the probability-map cell containing ``point``."""

    dx = point.x - mission.center.x
    dy = point.y - mission.center.y
    radius = sqrt(dx * dx + dy * dy)
    radial_count = len(probability_map.cells) // probability_map.angular_bin_count
    radial_index = min(
        int(radius / probability_map.radial_step_m),
        radial_count - 1,
    )
    angle = (tau + atan2(dy, dx)) % tau
    angular_index = min(
        int(angle / tau * probability_map.angular_bin_count),
        probability_map.angular_bin_count - 1,
    )
    return radial_index * probability_map.angular_bin_count + angular_index


@dataclass(frozen=True, slots=True)
class SpaceTimeMarkovModel:
    """Initial distribution and per-interval transitions over map cells."""

    initial: np.ndarray
    transitions: np.ndarray
    cell_centers: tuple[Point2D, ...]
    outside_index: int
    time_slice_s: float
    time_slice_count: int

    @property
    def state_count(self) -> int:
        return int(self.initial.size)

    @property
    def in_area_count(self) -> int:
        return self.state_count - 1

    def effectiveness(
        self,
        *,
        terrain=None,
        floor: float = 1e-3,
        outside_value: float = 1e-12,
    ) -> np.ndarray:
        """Per-cell detection effectiveness for the search-effort model."""

        effectiveness = np.ones(
            (self.time_slice_count, self.state_count),
            dtype=float,
        )
        if terrain is not None:
            effectiveness[:, : self.in_area_count] = np.asarray(
                [
                    max(floor, terrain.observability_weight(center.x, center.y))
                    for center in self.cell_centers
                ]
            )
        effectiveness[:, self.outside_index] = outside_value
        return effectiveness

    def feasible_mask(self) -> np.ndarray:
        """Search effort may never be assigned to the absorbing outside state."""

        mask = np.ones((self.time_slice_count, self.state_count), dtype=bool)
        mask[:, self.outside_index] = False
        return mask


def estimate_space_time_markov(
    particle_filter,
    mission: MissionConfig,
    probability_map: PolarProbabilityMap,
    *,
    time_slice_s: float,
    time_slice_count: int,
) -> SpaceTimeMarkovModel:
    """Estimate the space-time Markov model by rolling the particle belief.

    The particle filter is advanced ``time_slice_count - 1`` intervals, so the
    caller receives a belief that is consistent with the returned model.
    """

    if time_slice_s <= 0.0 or time_slice_count <= 0:
        raise ValueError("time discretization must be positive")

    state_count = len(probability_map.cells) + 1
    outside_index = state_count - 1

    initial = np.zeros(state_count, dtype=float)
    for point, weight, escaped in zip(
        particle_filter.points,
        particle_filter.weights,
        particle_filter.escaped,
    ):
        state = (
            outside_index
            if escaped
            else polar_cell_index(point, mission, probability_map)
        )
        initial[state] += weight

    transitions = np.zeros(
        (max(time_slice_count - 1, 0), state_count, state_count),
        dtype=float,
    )
    for time_index in range(time_slice_count - 1):
        before = [
            outside_index
            if escaped
            else polar_cell_index(point, mission, probability_map)
            for point, escaped in zip(
                particle_filter.points,
                particle_filter.escaped,
            )
        ]
        weights = tuple(particle_filter.weights)
        particle_filter.predict(time_slice_s)
        after = [
            outside_index
            if escaped
            else polar_cell_index(point, mission, probability_map)
            for point, escaped in zip(
                particle_filter.points,
                particle_filter.escaped,
            )
        ]
        for source, destination, weight in zip(before, after, weights):
            transitions[time_index, source, destination] += weight
        # 몬테카를로 전이 추정을 행 확률로 정규화:
        #   P_t(x, y) = N_t(x, y) / sum_y N_t(x, y)
        # 입자가 한 번도 지나가지 않은 상태는 자기 자신으로 보내 행 합 1을
        # 지킨다(그런 상태에는 어차피 확률질량이 없다).
        row_sum = transitions[time_index].sum(axis=1)
        for state in range(state_count):
            if row_sum[state] > 0.0:
                transitions[time_index, state] /= row_sum[state]
            else:
                transitions[time_index, state, state] = 1.0

    total = float(initial.sum())
    if total <= 0.0:
        raise ValueError("particle belief carries no probability mass")
    return SpaceTimeMarkovModel(
        initial=initial / total,
        transitions=transitions,
        cell_centers=tuple(cell.center for cell in probability_map.cells),
        outside_index=outside_index,
        time_slice_s=time_slice_s,
        time_slice_count=time_slice_count,
    )


def cell_adjacency(
    probability_map: PolarProbabilityMap,
    outside_index: int,
    reach_m: float,
) -> np.ndarray:
    """Boolean reachability between cells for one planning interval."""

    if reach_m <= 0.0:
        raise ValueError("reach_m must be positive")
    centers = [cell.center for cell in probability_map.cells]
    state_count = len(centers) + 1
    adjacency = np.eye(state_count, dtype=bool)
    coordinates = np.asarray([[point.x, point.y] for point in centers], dtype=float)
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt((deltas**2).sum(axis=2))
    adjacency[: len(centers), : len(centers)] = distances <= reach_m
    adjacency[outside_index, outside_index] = True
    return adjacency
