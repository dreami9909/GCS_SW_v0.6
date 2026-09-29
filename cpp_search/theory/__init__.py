"""탐색 이론 — 경로제약 아래에서 무엇이 최적인가.

챕터에 종속되지 않는 순수 수학 계층이다.

- ``path_constrained`` : 미탐지 질량 재귀(기준 평가함수), hazard 합, ED 완화,
                         H1/H2 후퇴지평. v0.4 는 H1/H2 를 **SPX warm start
                         로만** 쓰고 비교군으로 보고하지 않는다.
- ``stone_path``       : Stone, Royset & Washburn (2016) 4장 SP1/SPX 를 절단
                         평면으로 푼다. 모든 master 는 MILP 이고, 접선 절단이
                         유효 하한을, 정수해가 상한을 준다 -> **최적성 인증**.
- ``markov``           : 입자 belief -> 시공간 Markov 모델, 셀 도달가능성.
"""

from __future__ import annotations

from cpp_search.theory.markov import SpaceTimeMarkovModel, cell_adjacency, estimate_space_time_markov
from cpp_search.theory.path_constrained import (
    InfeasiblePathError,
    NonDetectionTrace,
    PathConstrainedProblem,
    SearcherModel,
    joint_survival,
    nondetection_trace,
    path_detection_probability,
    team_h2_receding_horizon,
    team_receding_horizon,
)
from cpp_search.theory.stone_path import (
    StoneCuttingPlaneResult,
    StoneTargetModel,
    stone_sp1_cutting_plane,
    stone_spx_cutting_plane,
)

__all__ = [
    "InfeasiblePathError",
    "NonDetectionTrace",
    "PathConstrainedProblem",
    "SearcherModel",
    "SpaceTimeMarkovModel",
    "StoneCuttingPlaneResult",
    "StoneTargetModel",
    "cell_adjacency",
    "estimate_space_time_markov",
    "joint_survival",
    "nondetection_trace",
    "path_detection_probability",
    "stone_sp1_cutting_plane",
    "stone_spx_cutting_plane",
    "team_h2_receding_horizon",
    "team_receding_horizon",
]
