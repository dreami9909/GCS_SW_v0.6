"""계획 — belief 를 축소 문제로, 축소 문제의 답을 실제 비행경로로.

- ``terrain_belief`` : 지형결합 IMM5 입자 belief. SPX 와 MAPPO 의 **공통 입력**.
- ``stone_spx``      : 인스턴스 구축 / 절단평면 해법 / 진단 포장.
- ``routes``         : 셀 경로 -> 시간예산이 맞는 평행소인. 계획법과 무관한
                       공통 변환이라 KPI 차이가 경로기하로 새지 않는다.

이 패키지는 ``learning`` 을 **절대** import 하지 않는다.
"""

from __future__ import annotations

from cpp_search.planning.routes import local_sweep
from cpp_search.planning.stone_spx import (
    StoneGridInstance,
    StoneGridPathSolution,
    StoneSPXRouteConfig,
    StoneSPXRouteDiagnostics,
    StoneSPXRoutePlan,
    StoneSPXSearcherClass,
    StoneSPXTargetSpec,
    build_stone_grid_instance,
    flown_routes_from_paths,
    plan_stone_grid_routes,
    plan_stone_spx_routes,
    route_plan_from_solution,
    solve_stone_grid_paths,
)
from cpp_search.planning.terrain_belief import (
    BELIEF_TERRAIN_FULL,
    BELIEF_TERRAIN_OFF,
    ParticleGridSnapshot,
    BeliefTerrainCoupling,
    TerrainParticleBelief,
)

__all__ = [
    "ParticleGridSnapshot",
    "StoneGridInstance",
    "StoneGridPathSolution",
    "StoneSPXRouteConfig",
    "StoneSPXRouteDiagnostics",
    "StoneSPXRoutePlan",
    "StoneSPXSearcherClass",
    "StoneSPXTargetSpec",
    "BELIEF_TERRAIN_FULL",
    "BELIEF_TERRAIN_OFF",
    "BeliefTerrainCoupling",
    "TerrainParticleBelief",
    "build_stone_grid_instance",
    "flown_routes_from_paths",
    "local_sweep",
    "plan_stone_grid_routes",
    "plan_stone_spx_routes",
    "route_plan_from_solution",
    "solve_stone_grid_paths",
]
