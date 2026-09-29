"""경로제약 탐색 — 이상적 노력지도를 실행 가능한 경로로.

Chapter 6의 이론 계층. 이 모듈이 답하는 질문은 하나다.

    Koopman-Stone의 최적 노력지도 e*(t, i)를, 이동제약을 만족하는
    탐색자 경로 s_j(0..T-1)로 어떻게 바꾸는가?

**일반적인 무손실 변환식은 없다.** 연속 노력배분과 경로계획은 결정변수와
제약이 서로 다른 문제다(``fab.py``는 전자, 이 모듈은 후자). 정적이고 연결된
구역이라면 노력량을 소인 길이와 간격으로 바꾸는 것으로 충분하고 그것은
``effort_route.py``가 한다. 이동표적 + 짧은 재계획 주기 + 인접성 제약에서는
노력지도를 사후 변환하는 대신 **처음부터 시간확장 네트워크 위에서 경로를
결정변수로** 최적화해야 한다.

    표적 사전분포 -> 시공간 미탐지 질량 u_t(i)
                  -> 한 번 방문의 한계 탐지이득 v_t(i)
                  -> 시간확장 네트워크의 노드 보상
                  -> 도달 가능한 경로 최적화
                  -> 첫 구간 실행 -> 미탐지 베이즈 갱신 -> 반복

수식과 구현 위치
----------------
* 미탐지 질량 재귀 (보고서 4.3절)  : ``nondetection_trace``
* 다중 탐색자 hazard 합 (4.2절)     : ``joint_survival``
* Expected Detections 완화 (6절)    : ``expected_detections``, ``search_free_marginals``
* ED 최장경로 (6절, 7.1절 부문제)   : ``expected_detection_paths``
* H1 후퇴지평 (7.2절)               : ``h1_receding_horizon``
* H2 후보 PD 비교 (7.3절)           : ``h2_receding_horizon``
* exact 오라클 (7.1절)              : ``exact_best_paths``

의존
----
위: ``src/cpp_search/chapters/chapter6``
아래: numpy만. 도메인 계층을 import 하지 않는다(``theory`` 계층 규칙).

참고문헌
--------
[1] Eagle & Yee, "An Optimal Branch-and-Bound Procedure for the Constrained
    Path, Moving Target Search Problem", Oper. Res. 38(1):110-114, 1990.
[2] Dell, Eagle, Martins & Santos, "Using Multiple Searchers in
    Constrained-Path, Moving-Target Search Problems", Naval Research
    Logistics 43(4):463-480, 1996.
[3] Stone, Royset & Washburn, "Path-Constrained Search in Discrete Time and
    Space", in *Optimal Search for Moving Targets*, Springer ISORMS 237,
    2016, pp. 81-120.
[4] Brown, "Optimal Search for a Moving Target in Discrete Time and Space",
    Oper. Res. 28(6):1275-1289, 1980.
"""

from __future__ import annotations

from cpp_search.theory import transitions as transition_ops

from collections.abc import Mapping

from dataclasses import dataclass, replace
from itertools import product
from typing import Literal

import numpy as np

__all__ = [
    "PathConstrainedProblem",
    "InfeasiblePathError",
    "PathPlanResult",
    "NonDetectionTrace",
    "SearcherModel",
    "effort_greedy_paths",
    "exact_best_paths",
    "expected_detection_paths",
    "joint_expected_detection_paths",
    "expected_detections",
    "h1_receding_horizon",
    "h2_receding_horizon",
    "team_receding_horizon",
    "team_h2_receding_horizon",
    "joint_survival",
    "hazard_field",
    "nondetection_trace",
    "path_detection_probability",
    "search_free_marginals",
]


class InfeasiblePathError(ValueError):
    """경로제약 아래 실행 가능한 경로가 없다.

    예약 규칙이 한 슬라이스의 도달 가능한 첫 수를 전부 막으면 발생한다.
    조용히 아무 셀이나 고르는 대신 명시적으로 실패한다 — 호출한 쪽이
    예약을 완화할지, 그 조건을 실패로 기록할지 정해야 한다.
    """


@dataclass(frozen=True, slots=True)
class SearcherModel:
    """한 탐색자의 출발점, 이동제약, 탐지율.

    ``detection_rate[t, i]`` 는 시각 ``t`` 에 셀 ``i`` 를 훑을 때의 **hazard**
    ``a`` 이고, 단일 방문 탐지확률은 ``q = 1 - exp(-a)`` 다. 확률이 아니라
    hazard 로 두는 이유는 같은 셀에 여러 탐색자가 들어왔을 때 **더할 수
    있기** 때문이다(보고서 4.2절). 확률을 그냥 더하면 1을 넘는다.

    ``adjacency[i, k]`` 가 참이면 한 슬라이스 안에 ``i`` 에서 ``k`` 로 갈 수
    있다. 제자리 대기를 허용하려면 대각을 참으로 둔다.
    """

    start_state: int
    detection_rate: np.ndarray
    adjacency: np.ndarray
    transit_fraction: np.ndarray | None = None
    """``transit_fraction[i, k]`` = i 에서 k 로 이동한 뒤 그 슬라이스에 **남는**
    탐색시간 비율 (0~1). ``None`` 이면 이동이 공짜라고 본다.

    hazard 는 훑은 길이에 비례하므로 유효 hazard 는

        a_eff(t, i->k) = a(t, k) * transit_fraction[i, k]

    다. 이 항이 배분 **안에** 들어가는 것이 Chapter 6의 핵심 결함을 고치는
    지점이다 — 전에는 배분이 이동을 0으로 놓고 최적화한 뒤 선택 단계에서
    사후 할인했다.

    ``swept_hazard`` 를 주면 이 항은 쓰이지 않는다. 이동을 '잃는 시간'으로
    보는 모형이기 때문이다."""

    swept_hazard: Mapping[tuple[int, int], Mapping[int, float]] | None = None
    """``swept_hazard[(i, k)][j]`` = i 에서 k 로 가는 동안 셀 ``j`` 에 쌓이는
    hazard. 도착셀뿐 아니라 **지나간 셀 전부**를 담는다.

    이동 중에도 search camera gimbal 은 꺼지지 않는다. 직선 이동은 오히려 weave
    보다 초당 소인량이 크다 (전진이 빨라서). 그런데 hazard 를 도착셀에만
    주면 그 소인량이 실제와 다른 곳에 쌓인다 — 외접 5x5·96 s 대각 이동에서
    1.082 km^2 가 지나온 셀이 아니라 도착셀로 간다.

    Stone 의 ``alpha[l,k](j', j, t)`` 를 j 로 한 겹 더 일반화한 것이다. X 에
    대해 여전히 선형이라 SPX 의 master LP 와 절단평면 인증은 그대로 성립한다.
    SP1 은 arc 독립 hazard 를 요구하므로 쓸 수 없다."""

    def __post_init__(self) -> None:
        rate = np.asarray(self.detection_rate, dtype=float)
        graph = np.asarray(self.adjacency, dtype=bool)
        if rate.ndim != 2:
            raise ValueError("detection_rate must be indexed [time, state]")
        if (rate < 0.0).any():
            raise ValueError("detection_rate is a hazard and cannot be negative")
        state_count = rate.shape[1]
        if graph.shape != (state_count, state_count):
            raise ValueError("adjacency must be [state, state]")
        if not 0 <= self.start_state < state_count:
            raise ValueError("start_state is outside the state space")
        if not graph[self.start_state].any():
            raise ValueError("start_state cannot reach any state")
        if self.transit_fraction is not None:
            fraction = np.asarray(self.transit_fraction, dtype=float)
            if fraction.shape != (state_count, state_count):
                raise ValueError("transit_fraction must be [state, state]")
            if (fraction < 0.0).any() or (fraction > 1.0).any():
                raise ValueError("transit_fraction must lie in [0, 1]")
        if self.swept_hazard is not None:
            for (source, destination), cells in self.swept_hazard.items():
                if not 0 <= source < state_count or not 0 <= destination < state_count:
                    raise ValueError("swept_hazard names a state outside the space")
                if not graph[source, destination]:
                    raise ValueError(
                        "swept_hazard names an arc that is not adjacent: "
                        f"{source} -> {destination}"
                    )
                for cell, value in cells.items():
                    if not 0 <= cell < state_count:
                        raise ValueError(
                            "swept_hazard sweeps a state outside the space"
                        )
                    if value < 0.0:
                        raise ValueError("swept hazard cannot be negative")

    @property
    def state_count(self) -> int:
        return int(self.detection_rate.shape[1])

    def successors(self, state: int) -> tuple[int, ...]:
        return tuple(int(k) for k in np.flatnonzero(self.adjacency[state]))

    def arc_hazard(self, time_index: int) -> np.ndarray:
        """``[i, k]`` 유효 hazard 표. 이동으로 잃는 탐색시간을 반영한다."""

        rate = self.detection_rate[time_index][None, :]
        if self.transit_fraction is None:
            return np.broadcast_to(rate, (self.state_count, self.state_count))
        return rate * np.asarray(self.transit_fraction, dtype=float)

    def effective_hazard(self, time_index: int, from_state: int, to_state: int) -> float:
        rate = float(self.detection_rate[time_index, to_state])
        if self.transit_fraction is None:
            return rate
        return rate * float(self.transit_fraction[from_state, to_state])

    def swept_cells(
        self, time_index: int, from_state: int, to_state: int
    ) -> Mapping[int, float]:
        """이 이동으로 hazard 가 쌓이는 셀들과 그 양.

        ``swept_hazard`` 가 없으면 기존 의미 그대로 **도착셀 하나**만 돌려준다.
        """

        if self.swept_hazard is None:
            return {to_state: self.effective_hazard(time_index, from_state, to_state)}
        return self.swept_hazard.get((from_state, to_state), {})


@dataclass(frozen=True, slots=True)
class PathConstrainedProblem:
    """이동표적 + 경로제약 탐색 문제 한 건.

    ``initial_mass`` 는 정규화된 표적 위치분포 pi_0, ``transitions[t]`` 는
    ``P_t(i, k) = Pr(X_{t+1}=k | X_t=i)`` 다.
    """

    initial_mass: np.ndarray
    transitions: np.ndarray
    searchers: tuple[SearcherModel, ...]

    def __post_init__(self) -> None:
        mass = np.asarray(self.initial_mass, dtype=float)
        moves = transition_ops.as_sequence(self.transitions)
        if mass.ndim != 1:
            raise ValueError("initial_mass must be one-dimensional")
        if (mass < 0.0).any():
            raise ValueError("initial_mass cannot be negative")
        if not self.searchers:
            raise ValueError("at least one searcher is required")
        states = mass.size
        horizon = self.searchers[0].detection_rate.shape[0]
        # 조밀 배열이든 희소행렬 목록이든 같은 계약을 건다.
        transition_ops.validate(
            moves, states=states, steps=max(horizon - 1, 0), what="transitions"
        )
        for searcher in self.searchers:
            if searcher.state_count != states:
                raise ValueError("every searcher must share the state space")
            if searcher.detection_rate.shape[0] != horizon:
                raise ValueError("every searcher must share the horizon")

    @property
    def state_count(self) -> int:
        return int(np.asarray(self.initial_mass).size)

    @property
    def time_count(self) -> int:
        return int(self.searchers[0].detection_rate.shape[0])

    @property
    def searcher_count(self) -> int:
        return len(self.searchers)

    def start_states(self) -> tuple[int, ...]:
        return tuple(searcher.start_state for searcher in self.searchers)


@dataclass(frozen=True, slots=True)
class NonDetectionTrace:
    """미탐지 질량 재귀의 전체 이력.

    ``mass[t]`` 는 시각 t **시작 시점**에 아직 탐지되지 않은 절대 질량이고,
    ``surviving_mass[t]`` 는 그 시각 탐색 **직후** 값이다. 둘의 차가 그
    슬라이스의 탐지 기여 ``detection_increment[t]`` 다.
    """

    mass: np.ndarray
    surviving_mass: np.ndarray
    detection_increment: np.ndarray
    probability_of_detection: float
    probability_of_no_detection: float

    def conditional_belief(self, time_index: int) -> np.ndarray:
        """미탐지를 조건으로 정규화한 운용 신념 b_t.

        ``mass`` 는 임무 시작부터의 **절대** 미탐지 질량이라 합이 1이 아니다.
        누계 PD 는 절대 질량으로, 재계획은 조건부 신념으로 해야 둘 다 맞는다
        (보고서 4.3절).
        """

        row = self.mass[time_index]
        total = float(row.sum())
        if total <= 0.0:
            raise ValueError("no undetected mass remains at this time index")
        return row / total


#: ED 대리목적의 두 정의. **원문과 우리 변형을 이름으로 구분한다.**
#:
#:   "dell"        Dell 1996 p.466 의 선형 ED — hazard 에 비례.
#:                     ED = sum_t sum_i m_t(i) * h_t(i)
#:                 탐색자별 보상이 그냥 더해진다(선형).
#:   "saturating"  우리 변형 — 같은 시각·같은 셀에서 포화.
#:                     ED = sum_t sum_i m_t(i) * (1 - exp(-h_t(i)))
#:                 PD 에 더 가깝지만 원문이 아니다.
#:
#: 두 식은 **다른 셀을 고를 수 있다**. 단일 탐색자·한 슬라이스 반례:
#: 셀 A(m=0.7, h=0.5) 는 dell 0.350 / sat 0.275, 셀 B(m=0.3, h=2.0) 는
#: dell 0.600 / sat 0.259 — dell 은 B, sat 는 A 를 고른다. 탐지율이 셀마다
#: 다르면 M=1 에서도 갈리므로 "다중일 때만 근사"라는 설명은 틀렸다.
#: 자세한 것은 docs/ALGORITHM_DEFINITIONS_KO.md 2절.
EDObjective = Literal["dell", "saturating"]

#: 다중 탐색자 조율 방식.
#:
#:   "joint"       Dell 1996 3절이 지시한 **곱 상태공간**. 원문 그대로지만
#:                 상태가 n^M, 분기가 d^M 이라 작은 격자에서만 실행 가능.
#:   "sequential"  탐색자를 하나씩 배정하는 우리 근사. M=1 에서는 joint 와
#:                 완전히 같다(실측 8/8 seed 일치).
Coordination = Literal["joint", "sequential"]


def hazard_field(
    problem: "PathConstrainedProblem",
    paths: tuple[tuple[int, ...], ...],
    time_index: int,
) -> np.ndarray:
    """시각 t 의 셀별 hazard 합 ``h_t(i) = sum_j a_j``.

    ``joint_survival`` 이 ``exp(-h)`` 를 내는 것과 같은 양이다. Dell 의 선형
    ED 는 지수를 취하기 전 값을 쓰므로 여기서 따로 낸다.
    """

    hazard = np.zeros(problem.state_count, dtype=float)
    for searcher, path in zip(problem.searchers, paths):
        previous = (
            searcher.start_state if time_index == 0 else path[time_index - 1]
        )
        hazard[path[time_index]] += searcher.effective_hazard(
            time_index, previous, path[time_index]
        )
    return hazard


def joint_survival(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    time_index: int,
) -> np.ndarray:
    """시각 t 의 셀별 생존확률 ``exp(-sum_j a_j)``.

    같은 셀에 여러 탐색자가 있으면 **hazard 를 더한다**(보고서 4.2절).
    단일 탐지확률 0.5 와 0.8 이면 결합 생존은 0.5*0.2 = 0.1 이지 1-(0.5+0.8)
    이 아니다. hazard 로 더하면 자동으로 맞는다.
    """

    hazard = np.zeros(problem.state_count, dtype=float)
    for searcher, path in zip(problem.searchers, paths):
        previous = (
            searcher.start_state if time_index == 0 else path[time_index - 1]
        )
        # ``swept_cells`` 는 ``swept_hazard`` 가 없으면 도착셀 하나만 준다.
        # 있으면 이동 중 지나간 셀들에도 hazard 가 쌓인다.
        for cell, value in searcher.swept_cells(
            time_index, previous, path[time_index]
        ).items():
            hazard[cell] += value
    return np.exp(-hazard)


def nondetection_trace(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
) -> NonDetectionTrace:
    """미탐지 질량 재귀식 (보고서 4.3절).

        u_0        = pi_0
        u_t^+(i)   = u_t(i) * exp(-sum_j a_{j,t,i} [s_{j,t} = i])
        Delta_t    = sum_i ( u_t(i) - u_t^+(i) )
        u_{t+1}    = u_t^+ @ P_t
        PD         = 1 - sum_i u_{T-1}^+(i)

    **이 재귀식이 이 챕터의 기준 평가함수다.** 표적 경로를 지수 개로 열거하지
    않고도 후보 탐색자 경로의 정확한 PD 를 준다. FAB, H1/H2, 탐욕 팀
    할당을 전부 같은 자로 재려면 전부 이 함수를 통과시킨다.
    """

    _validate_paths(problem, paths)
    time_count = problem.time_count
    mass = np.zeros((time_count, problem.state_count), dtype=float)
    surviving = np.zeros_like(mass)
    increment = np.zeros(time_count, dtype=float)

    mass[0] = np.asarray(problem.initial_mass, dtype=float)
    for time_index in range(time_count):
        survival = joint_survival(problem, paths, time_index)
        surviving[time_index] = mass[time_index] * survival
        increment[time_index] = float(
            mass[time_index].sum() - surviving[time_index].sum()
        )
        if time_index < time_count - 1:
            mass[time_index + 1] = (
                surviving[time_index] @ problem.transitions[time_index]
            )

    no_detection = float(surviving[-1].sum())
    return NonDetectionTrace(
        mass=mass,
        surviving_mass=surviving,
        detection_increment=increment,
        probability_of_detection=1.0 - no_detection,
        probability_of_no_detection=no_detection,
    )


def path_detection_probability(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
) -> float:
    """후보 경로 묶음의 정확한 탐지확률."""

    return nondetection_trace(problem, paths).probability_of_detection


def search_free_marginals(
    problem: PathConstrainedProblem,
    *,
    initial_mass: np.ndarray | None = None,
    start_time: int = 0,
) -> np.ndarray:
    """탐색의 영향을 받지 않은 표적 주변분포 m_t (보고서 6절).

    ED 의 노드보상을 미리 계산하는 데 쓴다. 탐색 실패가 미래 질량을 바꾸는
    효과를 **무시**하므로 ED 는 PD 의 상계가 된다.
    """

    mass = (
        np.asarray(problem.initial_mass, dtype=float)
        if initial_mass is None
        else np.asarray(initial_mass, dtype=float)
    )
    horizon = problem.time_count - start_time
    marginals = np.zeros((horizon, problem.state_count), dtype=float)
    marginals[0] = mass
    for offset in range(1, horizon):
        marginals[offset] = (
            marginals[offset - 1] @ problem.transitions[start_time + offset - 1]
        )
    return marginals


def expected_detections(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    *,
    objective: EDObjective = "saturating",
) -> float:
    """예상 탐지횟수 ED (보고서 6절).

    ``PD <= ED`` 가 항상 성립한다 — 적어도 한 번 탐지될 확률은 탐지횟수의
    기댓값을 넘을 수 없다. ED 는 보상이 가산적이라 DAG 최장경로로 빠르게
    최대화되지만, 이미 탐지됐을 상황을 이후 시각에 다시 세므로 중복을
    과대평가한다. **빠른 경로 생성기이자 상계이지 최종 판정 기준이 아니다.**
    """

    _validate_paths(problem, paths)
    marginals = search_free_marginals(problem)
    total = 0.0
    for time_index in range(problem.time_count):
        if objective == "dell":
            # Dell 1996 p.466: hazard 에 **선형**. 지수를 취하지 않는다.
            gain = hazard_field(problem, paths, time_index)
        else:
            gain = 1.0 - joint_survival(problem, paths, time_index)
        total += float((marginals[time_index] * gain).sum())
    return total


def _arc_reward(
    searcher: SearcherModel,
    marginals: np.ndarray,
    committed: np.ndarray,
    start_time: int,
    objective: EDObjective = "saturating",
) -> np.ndarray:
    """ED 의 아크 보상 ``[t, i, k]``.

    ``objective="dell"`` (원문, 선형)::

        reward_j(t, i->k) = m_t(k) * a_eff(t, i->k)

    선형이므로 탐색자별 보상이 그냥 더해진다. 앞선 탐색자가 무엇을 했는지
    (``committed``) 는 보상에 영향을 주지 않는다 — 그것이 선형의 뜻이다.

    ``objective="saturating"`` (우리 변형)::

        reward_j(t, i->k) = m_t(k) * exp(-A_{<j}(t,k)) * (1 - exp(-a_eff(t,i->k)))

    ``exp(-A_{<j})`` 항이 있어야 앞선 탐색자가 이미 건 hazard 를 이중으로
    세지 않고, 전체 합이 ``m_t(k)(1 - exp(-sum_j a_j))`` 로 정확히 접힌다.
    """

    horizon = marginals.shape[0]
    reward = np.empty(
        (horizon, searcher.state_count, searcher.state_count),
        dtype=float,
    )
    for offset in range(horizon):
        arc = searcher.arc_hazard(start_time + offset)
        if objective == "dell":
            reward[offset] = marginals[offset][None, :] * arc
        else:
            weight = marginals[offset] * np.exp(-committed[offset])
            reward[offset] = weight[None, :] * (1.0 - np.exp(-arc))
    return reward


def _longest_reward_path(
    reward: np.ndarray,
    adjacency: np.ndarray,
    start_state: int,
) -> tuple[int, ...]:
    """시간확장 DAG 의 최장 보상경로 (Viterbi).

    ``reward[t, i, k]`` 는 시각 t 에 ``i`` 에서 ``k`` 로 이동해 얻는 가산
    보상이다(``t = 0`` 이면 ``i`` 는 출발점). 보상을 **아크**에 두는 것이
    핵심이다 — 이동거리가 그 슬라이스에 남는 탐색시간을 정하므로, 노드
    보상만으로는 전개비용을 표현할 수 없다.

    보상이 가산적일 때만 옳다 — ED 가 그 경우다.
    """

    horizon, states = reward.shape[0], reward.shape[1]
    best = np.full((horizon, states), -np.inf, dtype=float)
    back = np.full((horizon, states), -1, dtype=int)
    reachable = np.flatnonzero(adjacency[start_state])
    if reachable.size == 0:
        raise ValueError("start_state cannot reach any state")
    best[0, reachable] = reward[0, start_state, reachable]

    # 슬라이스마다 [출발상태, 도착상태] 후보표를 한 번에 만든다. 파이썬
    # 이중루프는 241셀 x 2500회 평가(H2)에서 병목이 된다.
    blocked = ~adjacency
    for time_index in range(1, horizon):
        candidate = best[time_index - 1, :, None] + reward[time_index]
        candidate = np.where(blocked, -np.inf, candidate)
        back[time_index] = np.argmax(candidate, axis=0)
        best[time_index] = candidate[back[time_index], np.arange(states)]

    # 도달 가능한 첫 수가 전부 막히면(예약 등) 전 열이 -inf 가 된다. 그때
    # argmax 는 조용히 0 을 돌려주고, 셀 0 이 경로에 들어가 나중에 도달성
    # 검사에서 터진다 — 원인에서 멀리 떨어진 자리에서. 여기서 잡는다
    # (검토서 F9).
    if not np.isfinite(best[-1]).any():
        raise InfeasiblePathError(
            "no feasible path remains; every reachable first move is blocked "
            "(reservation may be over-constrained for this slice)"
        )
    last = int(np.argmax(best[-1]))
    path = [last]
    for time_index in range(horizon - 1, 0, -1):
        last = int(back[time_index, last])
        path.append(last)
    path.reverse()
    return tuple(path)


def expected_detection_paths(
    problem: PathConstrainedProblem,
    *,
    initial_mass: np.ndarray | None = None,
    start_time: int = 0,
    fixed_first: tuple[int, ...] | None = None,
    forbidden_first: frozenset[int] | None = None,
    objective: EDObjective = "saturating",
) -> tuple[tuple[int, ...], ...]:
    """ED 를 최대화하는 경로 묶음 (탐색자별 순차 배정).

    단일 탐색자면 DAG 최장경로라 **정확한** ED 최적해다. 여러 대면 결합상태
    공간이 ``d^M`` 로 커지므로(보고서 7.1절) 한 대씩 순차로 배정한다. 이때
    뒤 탐색자의 노드보상은 앞 탐색자가 이미 건 hazard 를 반영한다.

        reward_j(t,i) = m_t(i) * exp(-A_{<j}(t,i)) * (1 - exp(-a_j(t,i)))

    이 형태여야 합이 ``m_t(i)(1 - exp(-sum_j a_j))`` 로 정확히 접힌다. 단순히
    ``m_t(i)(1-exp(-a_j))`` 를 쓰면 겹치는 셀을 이중으로 센다.

    ``fixed_first`` 가 주어지면 각 탐색자의 첫 행동을 그 값으로 고정한다
    (H2 가 후보 첫 행동을 평가할 때 쓴다).
    """

    marginals = search_free_marginals(
        problem,
        initial_mass=initial_mass,
        start_time=start_time,
    )
    horizon = marginals.shape[0]
    committed = np.zeros_like(marginals)
    paths: list[tuple[int, ...]] = []

    for index, searcher in enumerate(problem.searchers):
        reward = _arc_reward(searcher, marginals, committed, start_time, objective)
        if fixed_first is not None:
            # 첫 행동을 고정한다: 그 셀 열만 남기고 나머지를 막는다.
            forced = np.full_like(reward, -np.inf)
            forced[0, :, fixed_first[index]] = reward[0, :, fixed_first[index]]
            forced[1:] = reward[1:]
            reward = forced
        elif forbidden_first:
            # 예약된 셀은 첫 수로 고를 수 없다 (shared-reserved).
            reward = reward.copy()
            for cell in forbidden_first:
                if 0 <= cell < reward.shape[2]:
                    reward[0, :, cell] = -np.inf
        path = _longest_reward_path(
            reward,
            searcher.adjacency,
            searcher.start_state,
        )
        previous = searcher.start_state
        for offset, state in enumerate(path):
            committed[offset, state] += searcher.effective_hazard(
                start_time + offset, previous, state
            )
            previous = state
        paths.append(path)

    return tuple(paths)


@dataclass(frozen=True, slots=True)
class PathPlanResult:
    """한 알고리즘이 낸 경로 묶음과 그 성능."""

    method: str
    paths: tuple[tuple[int, ...], ...]
    probability_of_detection: float
    expected_detections: float
    evaluations: int
    duplicate_assignment_ratio: float = 0.0
    communication_available_ratio: float = 1.0

    @property
    def relaxation_gap(self) -> float:
        """ED - PD. ED 완화가 얼마나 낙관적이었는지."""

        return self.expected_detections - self.probability_of_detection


def _score(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    method: str,
    evaluations: int,
    *,
    objective: EDObjective = "saturating",
) -> PathPlanResult:
    """PD 는 언제나 공통 자로, ED 는 그 방법이 최대화한 정의로 낸다.

    ``relaxation_gap = ED - PD`` 가 뜻을 가지려면 그 방법이 **실제로 쓴**
    대리목적으로 재야 한다. 선형 ED 를 쓴 방법을 포화형 ED 로 채점하면
    그 방법이 무엇을 낙관했는지 알 수 없다.
    """

    return PathPlanResult(
        method=method,
        paths=paths,
        probability_of_detection=path_detection_probability(problem, paths),
        expected_detections=expected_detections(problem, paths, objective=objective),
        evaluations=evaluations,
    )


def h1_receding_horizon(
    problem: PathConstrainedProblem,
    *,
    objective: EDObjective = "saturating",
    coordination: Coordination = "sequential",
) -> PathPlanResult:
    """H1 — ED 최장경로를 매 시각 다시 풀고 **첫 행동만** 확정 (보고서 7.2절).

    1. 남은 지평 전체에 대해 최대 ED 경로를 계산한다
    2. 그 경로의 첫 결합행동만 확정한다
    3. 미탐지를 조건으로 신념을 갱신한다
    4. 다음 시각에 다시 푼다

    매 스텝 베이즈 갱신이 ED 의 중복탐지 과대평가를 일부 교정한다. Santos
    (1993)와 Dell 등(1996)은 H1 계열이 탐색자 1-3대 시험문제에서 best known
    의 2% 이내였다고 보고한다.
    """

    committed: list[list[int]] = [[] for _ in problem.searchers]
    mass = np.asarray(problem.initial_mass, dtype=float).copy()
    evaluations = 0

    for time_index in range(problem.time_count):
        suffix = _solve_suffix(
            problem, committed, mass, time_index,
            objective=objective, coordination=coordination,
        )
        evaluations += 1
        for index, path in enumerate(suffix):
            committed[index].append(path[0])
        mass = _advance(problem, committed, mass, time_index)

    return _score(
        problem,
        tuple(tuple(path) for path in committed),
        f"h1-{objective}-{coordination}",
        evaluations,
        objective=objective,
    )


def h2_receding_horizon(
    problem: PathConstrainedProblem,
    *,
    objective: EDObjective = "saturating",
    coordination: Coordination = "sequential",
) -> PathPlanResult:
    """H2 — 후보 첫 행동을 **정확한 PD** 로 비교 (보고서 7.3절).

    suffix 생성은 ED 로 빠르게, 지금의 결정은 PD 로 정확하게 한다. 원문은
    가능한 결합 다음상태를 전부 열거하지만 ``d^M`` 이라 대수가 늘면 불가능
    하다. 여기서는 **좌표하강**으로 근사한다 — 탐색자를 하나씩 돌며 그 대의
    다음 셀만 바꿔보고, 나머지는 현재 최선에 고정한 채 정확한 PD 로 고른다.
    ``M * d`` 번 평가로 끝난다. 단일 탐색자면 원문 H2 와 동일하다.
    """

    committed: list[list[int]] = [[] for _ in problem.searchers]
    mass = np.asarray(problem.initial_mass, dtype=float).copy()
    evaluations = 0

    for time_index in range(problem.time_count):
        current = tuple(
            committed[index][-1] if committed[index] else searcher.start_state
            for index, searcher in enumerate(problem.searchers)
        )

        def score_first_move(first: tuple[int, ...]) -> float:
            suffix = _solve_suffix(
                problem, committed, mass, time_index, fixed_first=first,
                objective=objective, coordination=coordination,
            )
            full = tuple(
                tuple(committed[j]) + tuple(suffix[j])
                for j in range(problem.searcher_count)
            )
            return path_detection_probability(problem, full)

        if coordination == "joint":
            # 원문 H2: 가능한 **결합** 첫 수를 전부 열거한다. d^M 이라 대수가
            # 늘면 불가능하고, Dell 자신도 3대에서 H2 를 비교군에서 뺐다.
            candidates = [
                candidate
                for candidate in product(
                    *(
                        searcher.successors(current[index])
                        for index, searcher in enumerate(problem.searchers)
                    )
                )
            ]
            best_value, choice = -np.inf, None
            for candidate in candidates:
                value = score_first_move(candidate)
                evaluations += 1
                if value > best_value:
                    best_value, choice = value, candidate
            if choice is None:
                raise ValueError("no feasible joint first move")
            choice = list(choice)
        else:
            # 우리 근사: 좌표하강. 탐색자를 하나씩 돌며 그 대의 첫 수만 바꾼다.
            baseline = _solve_suffix(
                problem, committed, mass, time_index,
                objective=objective, coordination=coordination,
            )
            choice = [path[0] for path in baseline]
            best_value = -np.inf
            for index, searcher in enumerate(problem.searchers):
                for candidate in searcher.successors(current[index]):
                    trial = list(choice)
                    trial[index] = candidate
                    value = score_first_move(tuple(trial))
                    evaluations += 1
                    if value > best_value:
                        best_value = value
                        choice = trial

        for index, state in enumerate(choice):
            committed[index].append(state)
        mass = _advance(problem, committed, mass, time_index)

    return _score(
        problem,
        tuple(tuple(path) for path in committed),
        f"h2-{objective}-{coordination}",
        evaluations,
        objective=objective,
    )



def team_receding_horizon(
    problem: PathConstrainedProblem,
    *,
    sharing: str = "shared-reserved",
    communication_loss_probability: float = 0.0,
    seed: int = 0,
    reservation_neighborhoods: tuple[frozenset[int], ...] | None = None,
    neighbourhood_decay: np.ndarray | None = None,
    forbid_opposing_edge_swaps: bool = False,
) -> PathPlanResult:
    """공유 규칙을 지키는 팀 후퇴지평 계획 (H1의 팀 버전).

    ``h1_receding_horizon`` 은 전체 belief 를 공유한 채 k대 경로를 **중앙에서
    한 번에** 뽑는다. 그래서 정보공유가 실험축인 챕터에 그대로 넣으면 축이
    사라진다 — independent / shared / shared-reserved 가 **완전히 같은 계획**을
    낸다. 실제로 그렇게 됐고(Ch5-a 협동 이득이 구조적으로 0.0), 이 함수가 그
    결함을 고친다.

    ``planning/team_planner._assign_cells`` 의 공유 의미를 그대로 옮긴다.

        independent      기체마다 **자기 이력만** 반영한 질량. 예약 없음.
        shared-past      **과거 슬라이스까지의** 팀 질량. 같은 슬라이스에
                         동료가 무엇을 하기로 했는지는 모른다. 예약 없음.
        shared           **실시간** 팀 질량. 같은 슬라이스에서 먼저 움직인
                         동료의 갱신이 곧바로 보인다. 예약 없음.
        shared-reserved  실시간 팀 질량 + 이번 슬라이스에 이미 예약된 셀 제외.

    네 조건이 **세 요인**을 가른다 (검토서 F3).

        independent -> shared-past      과거 관측을 공유하는 이득
        shared-past -> shared           같은 슬라이스의 계획을 조율하는 이득
        shared      -> shared-reserved  하드 예약을 더하는 이득

    ``shared-past`` 를 넣기 전에는 앞의 두 요인이 한 덩어리였다. 관측이 아직
    하나도 없는 첫 슬라이스에서도 independent 와 shared 가 갈렸는데, 그것은
    과거 관측 공유가 아니라 **동료의 예정된 탐색을 고려한 순차 조율** 때문이다.
    그래서 예전의 ``belief_sharing_gain`` 은 순수한 관측정보 공유의 인과효과가
    아니라 규칙 전체의 효과였다.

    ``shared`` 가 실시간이어야 하는 이유는 저장소가 이미 기록한 사항이다 —
    슬라이스 시작 스냅샷을 쓰면 동일·결정론적 기체들이 전부 같은 선택을 해서
    ``shared`` 가 ``independent`` 와 같아진다. 그러면 두 조건의 차이에 '신념
    최신성'과 '예약 규칙'이 섞여 요인 분리가 성립하지 않는다.

    통신이 끊긴 기체는 공유 모드와 무관하게 자기 질량으로 후퇴한다.
    """

    if sharing not in {
        "independent",
        "shared-past",
        "shared",
        "shared-reserved",
    }:
        raise ValueError(f"unknown sharing mode: {sharing}")
    if not 0.0 <= communication_loss_probability <= 1.0:
        raise ValueError("communication_loss_probability must be in [0, 1]")
    if (
        reservation_neighborhoods is not None
        and len(reservation_neighborhoods) != problem.state_count
    ):
        raise ValueError("reservation_neighborhoods must match the state count")
    influence = None
    if neighbourhood_decay is not None:
        influence = np.asarray(neighbourhood_decay, dtype=float)
        if influence.shape != (problem.state_count, problem.state_count):
            raise ValueError("neighbourhood_decay must have shape (state, state)")
        if (
            not np.isfinite(influence).all()
            or np.any(influence < 0.0)
            or np.any(influence > 1.0)
        ):
            raise ValueError("neighbourhood_decay must be finite and in [0, 1]")

    rng = np.random.default_rng(seed)
    time_count = problem.time_count
    state_count = problem.state_count
    initial = np.asarray(problem.initial_mass, dtype=float)

    committed: list[list[int]] = [[] for _ in problem.searchers]
    shared_mass = initial.copy()
    own_mass = [initial.copy() for _ in problem.searchers]
    duplicate_count = 0
    link_count = 0
    evaluations = 0

    for time_index in range(time_count):
        reserved: set[int] = set()
        reserved_exact: set[int] = set()
        reserved_moves: list[tuple[int, int]] = []
        # 슬라이스가 시작될 때의 팀 질량. "과거 관측만 공유" 조건이 이것을 본다 —
        # 같은 슬라이스에 동료가 무엇을 하기로 했는지는 아직 모른다.
        shared_at_slice_start = shared_mass.copy()
        for index, searcher in enumerate(problem.searchers):
            link_up = bool(rng.random() >= communication_loss_probability)
            link_count += int(link_up)
            use_own = sharing == "independent" or not link_up
            if use_own:
                belief = own_mass[index]
            elif sharing == "shared-past":
                belief = shared_at_slice_start
            else:
                belief = shared_mass
            state = (
                committed[index][-1] if committed[index] else searcher.start_state
            )
            blocked_cells = (
                set(reserved)
                if sharing == "shared-reserved" and link_up
                else set()
            )
            if forbid_opposing_edge_swaps:
                blocked_cells.update(
                    source
                    for source, destination in reserved_moves
                    if destination == state and source != destination
                )
            blocked = frozenset(blocked_cells) if blocked_cells else None
            try:
                cell = _best_next_state(
                    problem,
                    searcher,
                    belief,
                    time_index=time_index,
                    start_state=state,
                    forbidden_first=blocked,
                )
            except InfeasiblePathError:
                if reservation_neighborhoods is None:
                    raise
                # 작은 도달가능집합에서 거리 예약이 모든 후보를 막으면 임무를
                # 중단하지 않고 동일 셀 예약까지만 유지한다. 분리는 soft
                # constraint, 같은 셀 동시배정 금지는 hard constraint다.
                cell = _best_next_state(
                    problem,
                    searcher,
                    belief,
                    time_index=time_index,
                    start_state=state,
                    forbidden_first=frozenset(
                        set(reserved_exact)
                        | {
                            source
                            for source, destination in reserved_moves
                            if forbid_opposing_edge_swaps
                            and destination == state
                            and source != destination
                        }
                    ),
                )
            evaluations += 1
            if cell in reserved_exact:
                duplicate_count += 1
            reserved_exact.add(cell)
            if reservation_neighborhoods is None:
                reserved.add(cell)
            else:
                reserved.update(reservation_neighborhoods[cell])
            committed[index].append(cell)
            reserved_moves.append((state, cell))

            # 이 기체의 행동만 반영해 질량을 갱신한다. 같은 슬라이스 안에서
            # 곧바로 반영하는 것이 '실시간 공유'의 뜻이다.
            hazard = searcher.effective_hazard(time_index, state, cell)
            if influence is None:
                factor = np.ones(state_count, dtype=float)
                factor[cell] = np.exp(-hazard)
            else:
                factor = np.exp(-hazard * influence[cell])
            own_mass[index] = own_mass[index] * factor
            if sharing != "independent":
                shared_mass = shared_mass * factor

        if time_index < time_count - 1:
            transition = problem.transitions[time_index]
            shared_mass = shared_mass @ transition
            own_mass = [mass @ transition for mass in own_mass]

    paths = tuple(tuple(path) for path in committed)
    result = _score(problem, paths, f"team-h1-{sharing}", evaluations)
    total = max(time_count * problem.searcher_count, 1)
    return replace(
        result,
        duplicate_assignment_ratio=duplicate_count / total,
        communication_available_ratio=link_count / total,
    )


def team_h2_receding_horizon(
    problem: PathConstrainedProblem,
    *,
    sharing: str = "shared-reserved",
    communication_loss_probability: float = 0.0,
    seed: int = 0,
    reservation_neighborhoods: tuple[frozenset[int], ...] | None = None,
    neighbourhood_decay: np.ndarray | None = None,
    forbid_opposing_edge_swaps: bool = False,
) -> PathPlanResult:
    """공유 규칙을 지키는 H2 (``team_receding_horizon`` 의 H2 판).

    ``team_receding_horizon`` 과 **구조는 같고 점수만 다르다**.

        team-h1   기체가 아는 belief 로 ED 최적 suffix 를 풀고 그 첫 수
        team-h2   후보 첫 수마다 suffix 를 완성해 **정확한 PD** 로 비교

    왜 필요한가
    -----------
    순수 ``h2_receding_horizon`` 은 공유 규칙을 모델링하지 않는다. 그것을
    제안 배정으로 쓰면 공유가 실험축인 챕터에서 세 조건이 같은 계획을 내고
    그 축이 0 으로 붕괴한다 — Chapter 6 의 ``ablation-independent`` 는
    +0.33 으로 이 저장소에서 **가장 큰 측정 효과**이므로, 그것을 버리고
    분리되지 않은 이득을 취할 수 없다. Chapter 5a 가 같은 사고를 이미 한 번
    겪었다.

    이 함수가 있어야 ``team-h1`` 대 ``team-h2`` 를 **같은 공유 조건에서**
    비교할 수 있다. 그것이 단일요인 비교다.

    비용
    ----
    후보 첫 수마다 suffix 를 풀고 전체 경로의 PD 를 계산하므로 기체당
    ``d`` 배 비싸다. ``team-h1`` 이 슬라이스당 ``M`` 번 푸는 데 비해
    ``M * d`` 번 푼다.
    """

    if sharing not in {
        "independent",
        "shared-past",
        "shared",
        "shared-reserved",
    }:
        raise ValueError(f"unknown sharing mode: {sharing}")
    if not 0.0 <= communication_loss_probability <= 1.0:
        raise ValueError("communication_loss_probability must be in [0, 1]")
    if (
        reservation_neighborhoods is not None
        and len(reservation_neighborhoods) != problem.state_count
    ):
        raise ValueError("reservation_neighborhoods must match the state count")
    influence = None
    if neighbourhood_decay is not None:
        influence = np.asarray(neighbourhood_decay, dtype=float)
        if influence.shape != (problem.state_count, problem.state_count):
            raise ValueError("neighbourhood_decay must have shape (state, state)")
        if (
            not np.isfinite(influence).all()
            or np.any(influence < 0.0)
            or np.any(influence > 1.0)
        ):
            raise ValueError("neighbourhood_decay must be finite and in [0, 1]")

    rng = np.random.default_rng(seed)
    time_count = problem.time_count
    state_count = problem.state_count
    initial = np.asarray(problem.initial_mass, dtype=float)

    committed: list[list[int]] = [[] for _ in problem.searchers]
    shared_mass = initial.copy()
    own_mass = [initial.copy() for _ in problem.searchers]
    duplicate_count = 0
    link_count = 0
    evaluations = 0

    for time_index in range(time_count):
        reserved: set[int] = set()
        reserved_exact: set[int] = set()
        reserved_moves: list[tuple[int, int]] = []
        shared_at_slice_start = shared_mass.copy()
        for index, searcher in enumerate(problem.searchers):
            link_up = bool(rng.random() >= communication_loss_probability)
            link_count += int(link_up)
            use_own = sharing == "independent" or not link_up
            if use_own:
                belief = own_mass[index]
            elif sharing == "shared-past":
                belief = shared_at_slice_start
            else:
                belief = shared_mass
            state = (
                committed[index][-1] if committed[index] else searcher.start_state
            )
            blocked_cells = (
                set(reserved)
                if sharing == "shared-reserved" and link_up
                else set()
            )
            if forbid_opposing_edge_swaps:
                blocked_cells.update(
                    source
                    for source, destination in reserved_moves
                    if destination == state and source != destination
                )
            blocked = frozenset(blocked_cells) if blocked_cells else None
            try:
                cell, used = _best_next_state_by_pd(
                    problem,
                    searcher,
                    belief,
                    time_index=time_index,
                    start_state=state,
                    forbidden_first=blocked,
                )
            except InfeasiblePathError:
                if reservation_neighborhoods is None:
                    raise
                cell, used = _best_next_state_by_pd(
                    problem,
                    searcher,
                    belief,
                    time_index=time_index,
                    start_state=state,
                    forbidden_first=frozenset(
                        set(reserved_exact)
                        | {
                            source
                            for source, destination in reserved_moves
                            if forbid_opposing_edge_swaps
                            and destination == state
                            and source != destination
                        }
                    ),
                )
            evaluations += used
            if cell in reserved_exact:
                duplicate_count += 1
            reserved_exact.add(cell)
            if reservation_neighborhoods is None:
                reserved.add(cell)
            else:
                reserved.update(reservation_neighborhoods[cell])
            committed[index].append(cell)
            reserved_moves.append((state, cell))

            hazard = searcher.effective_hazard(time_index, state, cell)
            if influence is None:
                factor = np.ones(state_count, dtype=float)
                factor[cell] = np.exp(-hazard)
            else:
                factor = np.exp(-hazard * influence[cell])
            own_mass[index] = own_mass[index] * factor
            if sharing != "independent":
                shared_mass = shared_mass * factor

        if time_index < time_count - 1:
            transition = problem.transitions[time_index]
            shared_mass = shared_mass @ transition
            own_mass = [mass @ transition for mass in own_mass]

    paths = tuple(tuple(path) for path in committed)
    result = _score(problem, paths, f"team-h2-{sharing}", evaluations)
    total = max(time_count * problem.searcher_count, 1)
    return replace(
        result,
        duplicate_assignment_ratio=duplicate_count / total,
        communication_available_ratio=link_count / total,
    )


def _best_next_state_by_pd(
    problem: PathConstrainedProblem,
    searcher: SearcherModel,
    belief: np.ndarray,
    *,
    time_index: int,
    start_state: int,
    forbidden_first: frozenset[int] | None,
) -> tuple[int, int]:
    """이 기체가 아는 belief 기준 **정확한 PD** 가 가장 큰 첫 수.

    H2 의 핵심이다 — suffix 는 ED 로 빠르게 완성하고 **지금의 결정만** PD 로
    고른다. 반환값은 ``(첫 수, 평가 횟수)``.

    한 대만 놓고 푼다. 팀 조율은 호출한 쪽이 질량 갱신 순서와 예약으로
    표현한다 — ``team_receding_horizon`` 과 같은 규약이다.
    """

    solo = PathConstrainedProblem(
        initial_mass=problem.initial_mass,
        transitions=problem.transitions,
        searchers=(replace(searcher, start_state=start_state),),
    )
    candidates = [
        cell
        for cell in searcher.successors(start_state)
        if forbidden_first is None or cell not in forbidden_first
    ]
    if not candidates:
        raise InfeasiblePathError(
            "no feasible first move remains for this searcher; every reachable "
            "cell is reserved"
        )

    best_cell, best_value = candidates[0], -np.inf
    for cell in candidates:
        suffix = expected_detection_paths(
            solo,
            initial_mass=belief,
            start_time=time_index,
            fixed_first=(cell,),
        )
        # 이 기체 혼자의 남은 경로를 그 belief 위에서 정확한 PD 로 잰다.
        value = _suffix_detection_probability(solo, suffix, belief, time_index)
        if value > best_value:
            best_cell, best_value = cell, value
    return best_cell, len(candidates)


def _suffix_detection_probability(
    solo: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
    belief: np.ndarray,
    start_time: int,
) -> float:
    """주어진 belief 에서 출발한 suffix 의 정확한 미탐지 재귀 PD."""

    mass = np.asarray(belief, dtype=float).copy()
    survival_total = 0.0
    for offset, _ in enumerate(paths[0]):
        hazard = np.zeros(solo.state_count, dtype=float)
        searcher = solo.searchers[0]
        previous = (
            searcher.start_state if offset == 0 else paths[0][offset - 1]
        )
        cell = paths[0][offset]
        hazard[cell] += searcher.effective_hazard(start_time + offset, previous, cell)
        mass = mass * np.exp(-hazard)
        time_index = start_time + offset
        if time_index < solo.time_count - 1:
            mass = mass @ solo.transitions[time_index]
    survival_total = float(mass.sum())
    return 1.0 - survival_total


def _best_next_state(
    problem: PathConstrainedProblem,
    searcher: SearcherModel,
    belief: np.ndarray,
    *,
    time_index: int,
    start_state: int,
    forbidden_first: frozenset[int] | None,
) -> int:
    """이 기체가 아는 belief 기준 ED 최적 suffix 의 **첫 수**.

    한 대만 놓고 푼다 — 팀 조율은 호출한 쪽이 질량 갱신 순서로 표현한다.
    """

    solo = PathConstrainedProblem(
        initial_mass=problem.initial_mass,
        transitions=problem.transitions,
        searchers=(replace(searcher, start_state=start_state),),
    )
    paths = expected_detection_paths(
        solo,
        initial_mass=belief,
        start_time=time_index,
        forbidden_first=forbidden_first,
    )
    return int(paths[0][0])


def joint_expected_detection_paths(
    problem: PathConstrainedProblem,
    *,
    initial_mass: np.ndarray | None = None,
    start_time: int = 0,
    fixed_first: tuple[int, ...] | None = None,
    objective: EDObjective = "dell",
    max_joint_states: int = 200_000,
) -> tuple[tuple[int, ...], ...]:
    """**곱 상태공간** 위의 ED 최장경로 — Dell 1996 3절이 지시한 그대로.

    원문은 다중 탐색자를 위해 별도 알고리즘을 두지 않는다. 단일 탐색자
    의사코드를 **확장된 상태공간** 위에서 돌리라고 한다.

        "The pseudocode is also appropriate for multiple searchers if the
         underlying network has an expanded state space to account for extra
         searchers."  -- Dell et al. 1996, 3절

    상태가 ``(c_1, ..., c_M)`` 순서쌍이므로 상태수 ``n^M``, 분기 ``d^M`` 이다.
    Dell 자신도 3대에서 멈췄고(9셀·3대 ≈ 10^18 feasible paths, p.465) H2 는
    3대에서 런타임 때문에 비교군에서 뺐다(4.3절). 우리 운용격자(241셀·6대)는
    상태만 1.96e14 라 실행 불가능하다 — 그래서 ``expected_detection_paths``
    의 순차근사가 따로 있다.

    이 함수는 **근사의 대가를 재기 위한 기준**이다. 실전 계획기가 아니다.
    ``max_joint_states`` 를 넘으면 조용히 근사하지 않고 거부한다.
    """

    searcher_count = len(problem.searchers)
    n = problem.state_count
    if n**searcher_count > max_joint_states:
        raise ValueError(
            f"joint state space is {n}^{searcher_count} which exceeds "
            f"{max_joint_states}; use expected_detection_paths (sequential) "
            f"or a smaller grid"
        )

    marginals = search_free_marginals(
        problem, initial_mass=initial_mass, start_time=start_time
    )
    horizon = marginals.shape[0]
    states = list(product(range(n), repeat=searcher_count))
    successors: dict[tuple[int, ...], list[tuple[int, ...]]] = {
        s: [
            t
            for t in states
            if all(problem.searchers[j].adjacency[s[j], t[j]] for j in range(searcher_count))
        ]
        for s in states
    }

    def joint_gain(offset: int, source: tuple[int, ...], target: tuple[int, ...]) -> float:
        hazard = np.zeros(n, dtype=float)
        for j, searcher in enumerate(problem.searchers):
            hazard[target[j]] += searcher.effective_hazard(
                start_time + offset, source[j], target[j]
            )
        gain = hazard if objective == "dell" else 1.0 - np.exp(-hazard)
        return float((marginals[offset] * gain).sum())

    start = tuple(searcher.start_state for searcher in problem.searchers)
    layers: list[dict[tuple[int, ...], tuple[float, tuple[int, ...] | None]]] = [
        {start: (0.0, None)}
    ]
    for offset in range(horizon):
        layer: dict[tuple[int, ...], tuple[float, tuple[int, ...] | None]] = {}
        for source, (value, _) in layers[-1].items():
            for target in successors[source]:
                if offset == 0 and fixed_first is not None and target != tuple(fixed_first):
                    continue
                total = value + joint_gain(offset, source, target)
                if target not in layer or total > layer[target][0]:
                    layer[target] = (total, source)
        if not layer:
            raise ValueError("no feasible joint move; check adjacency or fixed_first")
        layers.append(layer)

    end = max(layers[-1], key=lambda state: layers[-1][state][0])
    chain = [end]
    for index in range(len(layers) - 1, 1, -1):
        chain.append(layers[index][chain[-1]][1])
    chain.reverse()
    return tuple(tuple(step[j] for step in chain) for j in range(searcher_count))


def effort_greedy_paths(
    problem: PathConstrainedProblem,
    score_map: np.ndarray,
) -> PathPlanResult:
    """``[t, state]`` 점수 지도를 슬라이스마다 탐욕으로 따라가는 기준선.

    도달 가능한 셀 중 점수가 가장 큰 곳으로 순차 배정한다. 경로 전체를 보지
    않으므로, 경로제약을 인식하는 알고리즘이 얼마나 더 버는지를 재는 자다.

    **어떤 점수를 넘기느냐가 어떤 기준선인지를 정한다.** 절차는 같다.

    * ``planning.team_planner.detection_gain_map`` — 기체 1대를 보냈을 때의
      기대 탐지확률 ``alpha*beta*(1-exp(-w*e_1))``. Chapter 6 의
      ``ablation-effort-greedy`` 와 Chapter 3 의 flown 체제가 쓰는 점수이고,
      **챕터 간 비교의 기준선은 이쪽이다**.
    * 원 Koopman 노력지도 ``e*(t,i) = ln(p*c/mu)/c`` — 관측성 ``c`` 로
      **나누므로** "그 셀이 요구하는 양"이지 "그 셀에서 얻는 값"이 아니다.
      노력지도를 문자 그대로 따르면 어떻게 되는지를 재는 별도 조건이며,
      Chapter 3 에서 ``effort-map-literal`` 로 부른다.

    두 점수는 도달범위가 넓으면 거의 같은 답을 낸다(운용 설정에서 PD 0.5151
    대 0.5162). 좁히면 갈라진다 — 노력지도판은 관측성이 나쁜 셀로 기어들어가
    도달범위 1000 m 에서 PD 0.0033 까지 무너진다.
    """

    scores = np.asarray(score_map, dtype=float)
    if scores.shape != (problem.time_count, problem.state_count):
        raise ValueError("score_map must be [time, state]")

    committed: list[list[int]] = [[] for _ in problem.searchers]
    for time_index in range(problem.time_count):
        taken: set[int] = set()
        for index, searcher in enumerate(problem.searchers):
            current = (
                committed[index][-1] if committed[index] else searcher.start_state
            )
            options = [
                state
                for state in searcher.successors(current)
                if state not in taken
            ] or list(searcher.successors(current))
            pick = max(options, key=lambda state: scores[time_index, state])
            taken.add(pick)
            committed[index].append(pick)

    return _score(
        problem,
        tuple(tuple(path) for path in committed),
        "score-greedy",
        problem.time_count,
    )


def exact_best_paths(
    problem: PathConstrainedProblem,
    *,
    max_evaluations: int = 200_000,
) -> PathPlanResult:
    """작은 사례의 정확한 최적해 — 검증 오라클 (보고서 7.1절).

    도달 가능한 결합경로를 전부 열거해 정확한 PD 로 고른다. 경로 수가
    ``d^(M*T)`` 규모라 실시간 계획기로는 못 쓰지만, 다음 세 용도로 값이 크다.

    * 휴리스틱의 최적성 격차 측정
    * 회귀시험의 oracle
    * 완화된 노력지도가 실제 경로 상계로 얼마나 낙관적인지 측정

    ``max_evaluations`` 를 넘으면 ``ValueError`` 로 거부한다. 조용히 잘라내고
    "최적"이라고 부르면 오라클이 아니다.
    """

    per_searcher = [
        _enumerate_paths(searcher, problem.time_count)
        for searcher in problem.searchers
    ]
    total = 1
    for paths in per_searcher:
        total *= len(paths)
        if total > max_evaluations:
            raise ValueError(
                f"exact enumeration needs more than {max_evaluations} joint "
                f"paths; use a smaller grid, horizon, or searcher count"
            )

    best_paths: tuple[tuple[int, ...], ...] | None = None
    best_value = -np.inf
    evaluations = 0
    for combination in product(*per_searcher):
        value = path_detection_probability(problem, combination)
        evaluations += 1
        if value > best_value:
            best_value = value
            best_paths = combination

    assert best_paths is not None
    return _score(problem, best_paths, "exact-enumeration", evaluations)


def _enumerate_paths(searcher: SearcherModel, time_count: int) -> list[tuple[int, ...]]:
    partial: list[tuple[int, ...]] = [
        (state,) for state in searcher.successors(searcher.start_state)
    ]
    for _ in range(time_count - 1):
        grown: list[tuple[int, ...]] = []
        for path in partial:
            for nxt in searcher.successors(path[-1]):
                grown.append(path + (nxt,))
        partial = grown
    return partial


def _solve_suffix(
    problem: PathConstrainedProblem,
    committed: list[list[int]],
    mass: np.ndarray,
    time_index: int,
    *,
    fixed_first: tuple[int, ...] | None = None,
    objective: EDObjective = "saturating",
    coordination: Coordination = "sequential",
) -> tuple[tuple[int, ...], ...]:
    """남은 지평의 ED 최적 suffix. 확정된 접두부의 위치에서 출발한다.

    ``replace`` 로 출발점만 바꾼다. 예전에는 ``SearcherModel(...)`` 을 새로
    지으면서 ``transit_fraction`` 을 빠뜨렸고, 그래서 suffix 생성 단계만
    이동비용을 공짜로 보고 최종 채점에서는 물렸다. 같은 문제를 두 자로
    푸는 셈이었다 (검토서 F5).
    """

    searchers = tuple(
        replace(
            searcher,
            start_state=(
                committed[index][-1]
                if committed[index]
                else searcher.start_state
            ),
        )
        for index, searcher in enumerate(problem.searchers)
    )
    suffix_problem = PathConstrainedProblem(
        initial_mass=problem.initial_mass,
        transitions=problem.transitions,
        searchers=searchers,
    )
    solver = (
        joint_expected_detection_paths
        if coordination == "joint"
        else expected_detection_paths
    )
    return solver(
        suffix_problem,
        initial_mass=mass,
        start_time=time_index,
        fixed_first=fixed_first,
        objective=objective,
    )


def _advance(
    problem: PathConstrainedProblem,
    committed: list[list[int]],
    mass: np.ndarray,
    time_index: int,
) -> np.ndarray:
    """확정된 이번 슬라이스 행동을 반영해 미탐지 질량을 한 스텝 굴린다."""

    hazard = np.zeros(problem.state_count, dtype=float)
    for index, searcher in enumerate(problem.searchers):
        state = committed[index][time_index]
        previous = (
            searcher.start_state
            if time_index == 0
            else committed[index][time_index - 1]
        )
        hazard[state] += searcher.effective_hazard(time_index, previous, state)
    survived = mass * np.exp(-hazard)
    if time_index < problem.time_count - 1:
        return survived @ problem.transitions[time_index]
    return survived


def _validate_paths(
    problem: PathConstrainedProblem,
    paths: tuple[tuple[int, ...], ...],
) -> None:
    if len(paths) != problem.searcher_count:
        raise ValueError("one path per searcher is required")
    for searcher, path in zip(problem.searchers, paths):
        if len(path) != problem.time_count:
            raise ValueError("each path must cover the whole horizon")
        previous = searcher.start_state
        for state in path:
            if not 0 <= state < problem.state_count:
                raise ValueError("path visits a state outside the space")
            if not searcher.adjacency[previous, state]:
                raise ValueError(
                    "path violates the searcher's one-slice reachability"
                )
            previous = state
