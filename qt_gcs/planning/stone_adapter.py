"""GCS 도메인 → Stone-SPX 사전계획 어댑터 (v0.5).

이륙 전 1회, 배경 스레드에서 호출한다. ``SiteStore`` 미션(탐색중심·반경·탐색대상·
비행 MSL)을 벤더링된 ``cpp_search`` 입력으로 옮겨 **인증된** 탐색 경로를 얻고,
그 경로를 기체별 위경도 웨이포인트(고도 = 비행 MSL)로 되돌린다. Stone-SPX 는
실시간이 아니므로 50ms Fly 틱 루프에는 절대 넣지 않는다.

경계 원칙
--------
* GCS ``qt_gcs.planning.SearchCameraSpec`` 과 cpp_search ``SensorSpec`` 은 **별개 클래스**다.
  이 어댑터에서만 값을 옮기고, 두 타입을 섞지 않는다.
* 좌표는 양쪽 다 평면근사(111320 m/deg). SPX 는 탐색중심을 원점(0,0)으로 하는
  로컬 ENU 에서 풀고, 결과를 원점 위경도로 되돌린다.
* 지형은 belief(사전·전이)에만 들어간다. 고도(>600m 지형)는 no-fly 마스크로
  SPX 인접성에서 셀을 끊는다([[gcs-v05-terrain-altitude-decision]]).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from cpp_search.core.models import MissionConfig, Point2D, SensorSpec
from cpp_search.core.probability import TargetPrior
from cpp_search.core.motion import (
    CONTINUOUS_MOVE_PROFILE,
    MANEUVER_HEAVY_PROFILE,
    RELOCATION_HEAVY_PROFILE,
    STOP_HEAVY_PROFILE,
    TargetBehaviorProfile,
)
from cpp_search.core.profiles import (
    MISSING_PERSON_ADULT_OPERATIONAL_PROFILE,
    MISSING_PERSON_CHILD_OPERATIONAL_PROFILE,
)

# CPP 최악 거동: 잦은 선회·불규칙 이동(persistence 0.45, process noise 1.75)이라
# 탐색이 가장 어렵다. "탐색대상이 어떻게 기동할지 모른다 → 최악 거동 기준 계획"
# (CPP Ch2 minimax)을 이 단일 최악 거동으로 구현한다. 시나리오가 다른 거동을
# 지정하지 않으면 이 최악 거동이 기본이다.
WORST_CASE_BEHAVIOR = MANEUVER_HEAVY_PROFILE

# 미션 메타데이터 ``planning_motion_profile`` 로 계획 거동을 고를 수 있다
# (예: best/typical 시나리오는 CONTINUOUS_MOVE). 없거나 모르는 이름이면 최악 거동.
_BEHAVIOR_BY_NAME: dict[str, TargetBehaviorProfile] = {
    profile.name: profile
    for profile in (
        STOP_HEAVY_PROFILE,
        RELOCATION_HEAVY_PROFILE,
        CONTINUOUS_MOVE_PROFILE,
        MANEUVER_HEAVY_PROFILE,
    )
}


def _planning_behavior(store: SiteStore) -> TargetBehaviorProfile:
    name = str(
        (store.mission_metadata or {}).get("planning_motion_profile", "")
    ).strip().upper()
    return _BEHAVIOR_BY_NAME.get(name, WORST_CASE_BEHAVIOR)
from cpp_search.planning.stone_spx import (
    StoneSPXRouteConfig,
    StoneSPXSearcherClass,
    StoneSPXTargetSpec,
    build_stone_grid_instance,
    route_plan_from_solution,
    solve_stone_grid_paths,
)

from ..site_store import MissionPoint, SiteStore

EARTH_METERS_PER_DEGREE = 111_320.0

# Ch1 소인폭 표에서 유도된 탐색대상별 hazard 배율 (adult 0.874 / child 0.867).
_SIGNATURE_HAZARD = {
    "MISSING_PERSON_ADULT": 0.874,
    "MISSING_PERSON_CHILD": 0.867,
    "MISSING_PERSON_ELDERLY": 0.874,
}
_PROFILE_BY_TYPE = {
    "MISSING_PERSON_ADULT": MISSING_PERSON_ADULT_OPERATIONAL_PROFILE,
    "MISSING_PERSON_CHILD": MISSING_PERSON_CHILD_OPERATIONAL_PROFILE,
    "MISSING_PERSON_ELDERLY": MISSING_PERSON_ADULT_OPERATIONAL_PROFILE,
}


@dataclass(frozen=True)
class SearchPlanCertificate:
    """SPX 사전계획 결과 요약 (UI 상태줄/로그용)."""

    method: str
    fingerprint: str
    detection_probability: float
    relative_optimality_gap: float
    lower_bound_nondetection: float
    upper_bound_nondetection: float
    converged: bool
    certified: bool          # gap <= required_relative_gap 이고 converged
    required_relative_gap: float
    iterations: int
    runtime_s: float
    grid_shape: tuple[int, int]
    time_slice_count: int
    no_fly_cell_count: int
    target_detection_probabilities: tuple[tuple[str, float], ...]

    def summary(self) -> str:
        mark = "인증" if self.certified else "미인증"
        return (
            f"Stone-SPX {mark} · PD {self.detection_probability:.3f} · "
            f"gap {self.relative_optimality_gap * 100:.2f}% · "
            f"격자 {self.grid_shape[0]}×{self.grid_shape[1]} · "
            f"{self.runtime_s:.1f}s · no-fly {self.no_fly_cell_count}셀"
        )


@dataclass(frozen=True)
class CertifiedSearchPlan:
    certificate: SearchPlanCertificate
    # vehicle_id(1..6) -> [(lat, lon, alt_m), ...]
    vehicle_waypoints: dict[int, list[tuple[float, float, float]]]
    assignments: tuple[tuple[int, ...], ...] | None = None


# --------------------------------------------------------------------------- #
# 지리 <-> 로컬 ENU (탐색중심 원점)                                            #
# --------------------------------------------------------------------------- #
def _lonlat_scales(origin_lat: float) -> tuple[float, float]:
    lat_scale = EARTH_METERS_PER_DEGREE
    lon_scale = max(1e-6, EARTH_METERS_PER_DEGREE * math.cos(math.radians(origin_lat)))
    return lat_scale, lon_scale


def _local_to_lonlat(east_m, north_m, origin_lat, origin_lon):
    lat_scale, lon_scale = _lonlat_scales(origin_lat)
    return (
        origin_lat + north_m / lat_scale,
        origin_lon + east_m / lon_scale,
    )


def _lonlat_to_local(lat, lon, origin_lat, origin_lon):
    lat_scale, lon_scale = _lonlat_scales(origin_lat)
    return (
        (lon - origin_lon) * lon_scale,
        (lat - origin_lat) * lat_scale,
    )


def _point_in_polygon(px, py, polygon):
    """Ray-casting point-in-polygon test for 2D (east, north) coordinates."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# --------------------------------------------------------------------------- #
# 미션 → SPX 입력                                                              #
# --------------------------------------------------------------------------- #
def _search_center(store: SiteStore) -> tuple[float, float]:
    arc = (store.mission_metadata or {}).get("arc_search_pattern", {})
    center = arc.get("center") or (store.mission_metadata or {}).get(
        "rally_predicted_subject"
    )
    if center:
        return float(center["latitude"]), float(center["longitude"])
    if "GCS" in store.sites:
        gcs = store.sites["GCS"]
        return float(gcs.latitude), float(gcs.longitude)
    raise ValueError("미션에서 탐색 중심을 찾을 수 없습니다 (arc center / GCS).")


def _search_radius_m(store: SiteStore, default: float = 3333.3333333333) -> float:
    meta = store.mission_metadata or {}
    return float(meta.get("search_radius_m") or default)


def _subject_specs(
    store: SiteStore,
    exclude_track_ids: set[int] | frozenset[int] | tuple[int, ...] | None = None,
) -> tuple[StoneSPXTargetSpec, ...]:
    """미션 initial_subjects → StoneSPXTargetSpec (중복 타입 제거).

    ``exclude_track_ids`` 에 든 track_id 는 계획에서 뺀다 — 발견된 탐색대상을 제외하고
    남은 탐색대상으로 재계획할 때 쓴다(다인원 순차 탐색).

    각 탐색대상은 ``planning_motion_profile`` 로 지정한 거동(없으면 최악 거동
    MANEUVER_HEAVY)으로 계획한다 — 시그니처(탐지 난이도)는 대상별로 두되,
    기동은 어떻게 할지 모른다는 전제로 시나리오가 고른 단일 거동을 쓴다
    (CPP Ch2 '최악 거동 기준' minimax). 프로파일의 기본 거동
    (child=RELOCATION_HEAVY 등)을 이 선택 거동으로 교체한다.
    """
    behavior = _planning_behavior(store)
    excluded = {int(t) for t in (exclude_track_ids or ())}
    remaining = [
        subj
        for subj in store.initial_subjects
        if int(getattr(subj, "track_id", -1)) not in excluded
    ]
    seen: dict[str, StoneSPXTargetSpec] = {}
    for subj in remaining:
        key = str(getattr(subj, "subject_type", "")).upper()
        profile = _PROFILE_BY_TYPE.get(key)
        if profile is None or key in seen:
            continue
        seen[key] = StoneSPXTargetSpec(
            profile=replace(
                profile,
                name=f"{profile.name}/{behavior.name}",
                imm5_behavior=behavior,
            ),
            hazard_multiplier=_SIGNATURE_HAZARD.get(key, 1.0),
        )
    # Re-planning after every subject is found leaves nothing to search; the
    # child-signature fallback is only for missions that declare no subjects.
    if not seen and store.initial_subjects and excluded:
        raise ValueError(
            "재계획할 미탐지 대상이 없습니다 (모든 탐색대상을 발견/제외했습니다)."
        )
    if not seen:  # 탐색대상 정보가 없으면 아동 시그니처 + 선택 거동 하나로 계획
        seen["MISSING_PERSON_CHILD"] = StoneSPXTargetSpec(
            profile=replace(
                MISSING_PERSON_CHILD_OPERATIONAL_PROFILE,
                name=f"MissingPerson-child-nominal/{behavior.name}",
                imm5_behavior=behavior,
            ),
            hazard_multiplier=_SIGNATURE_HAZARD["MISSING_PERSON_CHILD"],
        )
    return tuple(seen.values())


def _initial_ring(
    radius_m: float,
    count: int,
    ring_ratio: float,
    *,
    arc_center_rad: float | None = None,
    arc_span_rad: float = 2.0 * math.pi,
) -> tuple[Point2D, ...]:
    """AOI 반경의 ``ring_ratio`` 고리에 count 대를 균등 배치 (로컬 ENU).

    ``arc_center_rad`` 가 주어지면 그 방향 ±``arc_span_rad``/2 호에만 배치한다
    (ingress 쪽 배제 시 전 기체를 FAR 쪽 반원 호에 두어 SPX가 far side 셀만
    배정하도록).
    """
    r = radius_m * ring_ratio
    positions = []
    if arc_center_rad is None:
        for i in range(count):
            theta = 2.0 * math.pi * i / count
            positions.append(Point2D(r * math.cos(theta), r * math.sin(theta)))
    else:
        for i in range(count):
            frac = (i + 0.5) / max(1, count)
            theta = arc_center_rad - arc_span_rad / 2.0 + frac * arc_span_rad
            positions.append(Point2D(r * math.cos(theta), r * math.sin(theta)))
    return tuple(positions)


def _prior_sigma_ratio(store: SiteStore, radius_m: float) -> float:
    # Mission override: a wider belief prior spreads the SPX search across the
    # circle (vehicles sweep cell-to-cell outward) instead of all clustering in
    # the tight central high-probability cells.  Also lets the belief actually
    # cover a subject that starts far from the TP centre.
    override = (store.mission_metadata or {}).get("search_prior_sigma_ratio")
    if override is not None:
        return max(0.005, min(0.9, float(override)))
    sigmas = [
        float(getattr(t, "position_uncertainty_m", 0.0) or 0.0)
        for t in store.initial_subjects
    ]
    sigma_m = min([s for s in sigmas if s > 0.0], default=500.0)
    return max(0.005, min(0.9, sigma_m / max(radius_m, 1.0)))


def _load_terrain(terrain_npz):
    if not terrain_npz:
        return None
    from cpp_search.core.terrain import RasterTerrainField

    return RasterTerrainField.load(terrain_npz)


def _no_fly_cells(instance, terrain, flight_msl_m, min_clearance_agl_m) -> np.ndarray:
    """SPX 셀 중심의 지면고도로 no-fly 셀 인덱스 배열 산출.

    H_eff = flight_msl - 지면고도;  no_fly = H_eff < min_clearance_agl.
    지형이 없거나 elevation 표집 불가면 빈 배열.
    """
    if terrain is None:
        return np.empty(0, dtype=int)
    grid = instance.grid
    no_fly = []
    for cell in range(grid.cell_count):
        center = grid.center_of(cell)
        elev = terrain.elevation_at(center.x, center.y)
        if elev is None:
            continue
        if (flight_msl_m - float(elev)) < min_clearance_agl_m:
            no_fly.append(cell)
    return np.asarray(no_fly, dtype=int)


def _mask_no_fly(instance, no_fly: np.ndarray):
    """no-fly 셀로 드나드는 arc 를 각 탐색자 인접성에서 끊은 새 인스턴스.

    인접성뿐 아니라 ``swept_hazard`` 도 함께 정리한다. no-fly 셀을 출발/도착으로
    하는 arc 는 삭제하고, 남은 arc 가 지나가며 hazard 를 쌓던 셀 목록에서도
    no-fly 셀을 뺀다. 이렇게 하지 않으면 ``SearcherModel.__post_init__`` 이
    끊긴 arc 를 "인접하지 않은 swept_hazard" 로 보고 예외를 던진다(사각 격자는
    항상 swept_hazard 를 가지므로 no-fly 셀이 하나만 생겨도 계획이 실패한다).

    빈 배열이면 원본을 그대로 돌려준다(평탄 AOI 의 정상 경로)."""
    if no_fly.size == 0:
        return instance
    problem = instance.problem
    new_searchers = []
    for searcher in problem.searchers:
        adjacency = np.array(searcher.adjacency, dtype=bool, copy=True)
        n = adjacency.shape[0]
        mask = no_fly[no_fly < n]
        masked_states = {int(cell) for cell in mask}
        adjacency[mask, :] = False
        adjacency[:, mask] = False
        if not adjacency[searcher.start_state].any():
            raise ValueError(
                f"탐색자 출발셀 {searcher.start_state} 가 no-fly 로 고립되었습니다 "
                "(이륙 반경를 넓히거나 비행 MSL 을 올리십시오)."
            )
        swept_hazard = searcher.swept_hazard
        if swept_hazard is not None:
            swept_hazard = {
                (source, destination): {
                    cell: value
                    for cell, value in cells.items()
                    if cell not in masked_states
                }
                for (source, destination), cells in swept_hazard.items()
                if source not in masked_states
                and destination not in masked_states
            }
        new_searchers.append(
            replace(searcher, adjacency=adjacency, swept_hazard=swept_hazard)
        )
    new_problem = replace(problem, searchers=tuple(new_searchers))
    return replace(instance, problem=new_problem)


# --------------------------------------------------------------------------- #
# 사전계획 진입점                                                              #
# --------------------------------------------------------------------------- #
def plan_certified_search(
    store: SiteStore,
    *,
    flight_msl_m: float = 1130.0,
    search_camera_altitude_agl_m: float = 600.0,
    terrain_npz=None,
    grid_n: int = 6,
    planning_time_s: float | None = None,
    time_slice_factor: float = 1.25,
    initial_belief_delay_s: float | None = None,
    min_clearance_agl_m: float = 120.0,
    # 분리반경 배율(×셀변). 0 = 끔. SPX 셀 배타성은 이미 occupancy=1 로 보장되며,
    # 경로 겹침의 실제 원인은 셀 분리가 아니라 경로변환 단계라 기본 0 으로 둔다
    # (1.0 이상은 6기+이륙반경에서 실행불가). 물리 충돌회피가 필요하면 소폭 올린다.
    separation_factor: float = 0.0,
    aggregate_searchers: bool = True,
    # 기체별 sweep 측방 오프셋 배율(×셀변). 같은 셀을 여러 기체가 커버해도
    # 서로 다른 방향(기체마다 고유)으로 살짝 밀어 동일 트랙 겹침을 없앤다.
    # 셀변보다 훨씬 작아 인증된 셀 커버리지는 유지된다. 0 = 끔.
    deconflict_factor: float = 0.18,
    uav_count: int | None = None,
    search_speed_kph: float = 160.0,
    transit_speed_kph: float = 160.0,
    subject_max_speed_kph: float = 40.0,
    seed: int = 20260201,
    required_relative_gap: float = 0.01,
    master_time_limit_s: float = 60.0,
    max_iterations: int = 20,
    exclude_track_ids: set[int] | frozenset[int] | tuple[int, ...] | None = None,
    ingress_corridor_vertices: list[tuple[float, float]] | None = None,
    ingress_survival_factor: float = 0.0,
) -> CertifiedSearchPlan:
    """미션 → 인증 탐색계획. 이륙 전 배경 스레드에서 1회 호출한다.

    ``flight_msl_m`` 은 실제 지형 기준 비행고도(no-fly 판정용). 웨이포인트에
    기록되는 **시뮬레이터 고도는 AGL(``search_camera_altitude_agl_m``, 기본 600)** 이다 —
    현 sim 은 평면세계(탐색대상 MSL 0)라 search camera 기하가 AGL 로 맞아야 탐지가 정상
    동작한다. ``separation_factor`` × 셀변 = SPX 분리반경(비행체 이격, 겹침 방지).
    """

    origin_lat, origin_lon = _search_center(store)
    radius_m = _search_radius_m(store)
    # 운용 대수: 인자 > 미션 메타(uav_count) > 기본 6. 1대 케이스 미션 지원.
    meta_uav = (store.mission_metadata or {}).get("uav_count")
    n_uav = int(uav_count or meta_uav or len(SiteStore.VEHICLE_IDS))
    n_uav = max(1, min(n_uav, len(SiteStore.VEHICLE_IDS)))

    # 비행체 160km/h 탐색이동, 탐색대상 최대 40km/h (사용자 확정). 탐색대상은 이륙과 함께
    # 움직이기 시작하고(이송 8분 동안 belief 확산), 그 뒤 탐색 구간에도 이동한다.
    mission = MissionConfig(
        center=Point2D(0.0, 0.0),
        search_radius_m=radius_m,
        uav_count=n_uav,
        transit_speed_mps=transit_speed_kph / 3.6,
        search_speed_mps=search_speed_kph / 3.6,
        subject_max_speed_mps=subject_max_speed_kph / 3.6,
    )
    sensor = replace(SensorSpec.sr_z50(), altitude_m=float(search_camera_altitude_agl_m))
    subjects = _subject_specs(store, exclude_track_ids=exclude_track_ids)
    searcher_classes = (
        StoneSPXSearcherClass(name="eo-ir-nominal", count=mission.uav_count, hazard_scale=1.0),
    )
    # When the ingress side is excluded, start all UAVs on the FAR-side arc
    # (opposite LC) so SPX assigns them far-side cells rather than the swept
    # ingress side they happen to sit near on a full 360-degree ring.
    _arc_center = None
    if (store.mission_metadata or {}).get("exclude_ingress_side"):
        _lc = store.sites.get("LC") or store.sites.get("GCS")
        if _lc is not None:
            _lx, _ly = _lonlat_to_local(
                float(_lc.latitude), float(_lc.longitude), origin_lat, origin_lon
            )
            if (_lx * _lx + _ly * _ly) ** 0.5 > 1.0:
                _arc_center = math.atan2(-_ly, -_lx)  # far direction (opposite LC)
    # Where the searchers start (fraction of the AOI radius). Default 0.45.
    # In rugged terrain the far-side start ring can land on no-fly peaks, which
    # isolates a searcher's start cell; a smaller ring starts them in the
    # flyable valley basin near the TP so they can then route around the peaks.
    _ring_ratio = float((store.mission_metadata or {}).get("initial_ring_ratio", 0.45))
    _ring_ratio = max(0.1, min(0.9, _ring_ratio))
    initial_positions = _initial_ring(
        radius_m, mission.uav_count, ring_ratio=_ring_ratio,
        arc_center_rad=_arc_center, arc_span_rad=math.pi * 0.9,
    )
    prior = TargetPrior(kind="tp-centered", sigma_ratio=_prior_sigma_ratio(store, radius_m))
    terrain = _load_terrain(terrain_npz)

    # 슬라이스 수는 격자를 한 번 훑을 만큼 (한 슬라이스에 ~한 셀 전진;
    # Ch2 auto 규칙과 정합, 8×8→~11). planning_time_s 는 이로부터 유도한다 —
    # 임의의 큰 값을 쓰면 슬라이스가 폭증해 MILP master 가 터진다.
    cell_m = 2.0 * radius_m / grid_n
    slice_s = cell_m / max(mission.transit_speed_mps, 1e-6)
    time_slice_count = max(3, round(time_slice_factor * grid_n))
    if planning_time_s is None:
        planning_time_s = time_slice_count * slice_s

    if initial_belief_delay_s is None:
        initial_belief_delay_s = float(
            (store.mission_metadata or {}).get("subject_lead_time_s") or 480.0
        )

    config = StoneSPXRouteConfig(
        grid_width=grid_n,
        grid_height=grid_n,
        time_slice_count=time_slice_count,
        particle_count=max(2000, 50 * grid_n * grid_n),
        hazard_calibration="effective-sweep-width",
        occupancy_limit=1,
        forbid_opposing_edge_swaps=True,
        relative_tolerance=required_relative_gap,
        max_iterations=max_iterations,
        aggregate_identical_searchers=aggregate_searchers,
        # 분리반경: 같은 슬라이스에 비행체가 서로 이 거리 안에 못 들어온다
        # (occupancy=1 위에 물리 이격 추가 → 경로 겹침 방지). 셀변 비례로 잡아
        # 4-이웃 셀 동시 점유를 막되 대각은 허용해 실행가능성을 유지한다.
        reservation_separation_m=separation_factor * cell_m,
        square_fit="circumscribed",
        terrain_weighting=terrain is not None,
        master_time_limit_s=master_time_limit_s,
        persistent_master=False,
        continuous_relaxation_iterations=3,
        local_improvement_passes=1,
        sparse_transitions="auto",
    )

    instance = build_stone_grid_instance(
        mission,
        sensor,
        prior=prior,
        terrain=terrain,
        targets=subjects,
        searcher_classes=searcher_classes,
        initial_positions=initial_positions,
        planning_time_s=planning_time_s,
        initial_belief_delay_s=initial_belief_delay_s,
        seed=seed,
        config=config,
    )
    no_fly = _no_fly_cells(instance, terrain, flight_msl_m, min_clearance_agl_m)
    instance = _mask_no_fly(instance, no_fly)

    if ingress_corridor_vertices and len(ingress_corridor_vertices) >= 3:
        corridor_local = [
            _lonlat_to_local(lat, lon, origin_lat, origin_lon)
            for lat, lon in ingress_corridor_vertices
        ]
        grid = instance.grid
        # A cell counts as swept if it OVERLAPS the corridor — check the centre
        # and the four corners, not just the centre, so edge cells whose sweep
        # legs clip the corridor are excluded too (complete exclusion).
        half_x = grid.cell_width_m / 2.0
        half_y = grid.cell_height_m / 2.0

        def _cell_overlaps_corridor(center) -> bool:
            if _point_in_polygon(center.x, center.y, corridor_local):
                return True
            for dx in (-half_x, half_x):
                for dy in (-half_y, half_y):
                    if _point_in_polygon(
                        center.x + dx, center.y + dy, corridor_local
                    ):
                        return True
            return False

        new_targets = []
        scanned_unique = 0
        for target in instance.target_models:
            mass = np.array(target.initial_mass, dtype=float)
            cell_count = 0
            for cell in range(grid.cell_count):
                if _cell_overlaps_corridor(grid.center_of(cell)):
                    mass[cell] *= ingress_survival_factor
                    cell_count += 1
            total = mass.sum()
            if total > 0:
                mass /= total
            new_targets.append(replace(target, initial_mass=mass))
            scanned_unique = max(scanned_unique, cell_count)
        instance = replace(instance, target_models=tuple(new_targets))
        logging.info(
            "Ingress belief carry-over: %d cells scanned (survival=%.2f)",
            scanned_unique, ingress_survival_factor,
        )

    # STEP1: probabilistically exclude the whole INGRESS-SIDE half-plane — every
    # cell on the LC side of the TP line (the line through the TP, perpendicular
    # to LC->TP).  The fleet already swept that side sensor-on during ingress, so
    # its belief mass goes to zero and SPX concentrates the search on the FAR
    # side beyond the TP line.  Gated by mission metadata ``exclude_ingress_side``.
    if (store.mission_metadata or {}).get("exclude_ingress_side"):
        lc_site = store.sites.get("LC") or store.sites.get("GCS")
        if lc_site is not None:
            lx, ly = _lonlat_to_local(
                float(lc_site.latitude), float(lc_site.longitude),
                origin_lat, origin_lon,
            )
            if (lx * lx + ly * ly) ** 0.5 > 1.0:
                grid = instance.grid
                new_targets = []
                excluded = 0
                for target in instance.target_models:
                    mass = np.array(target.initial_mass, dtype=float)
                    n = 0
                    for cell in range(grid.cell_count):
                        c = grid.center_of(cell)
                        # dot(cell_centre, LC) > 0  =>  cell is on the LC/ingress
                        # side of the TP line -> already swept -> exclude.
                        if (c.x * lx + c.y * ly) > 0.0:
                            mass[cell] *= ingress_survival_factor
                            n += 1
                    total = mass.sum()
                    if total > 0:
                        mass /= total
                    new_targets.append(replace(target, initial_mass=mass))
                    excluded = max(excluded, n)
                instance = replace(instance, target_models=tuple(new_targets))
                logging.info(
                    "Ingress-side half-plane excluded: %d cells (SPX -> far side)",
                    excluded,
                )

    solution = solve_stone_grid_paths(instance, method="stone-spx")
    plan = route_plan_from_solution(instance, solution, method="stone-spx")
    diag = plan.diagnostics

    certificate = SearchPlanCertificate(
        method=diag.planning_method,
        fingerprint=diag.common_input_fingerprint,
        detection_probability=diag.detection_probability,
        relative_optimality_gap=diag.relative_optimality_gap,
        lower_bound_nondetection=diag.lower_bound_nondetection,
        upper_bound_nondetection=diag.upper_bound_nondetection,
        converged=diag.converged,
        certified=bool(diag.converged and diag.relative_optimality_gap <= required_relative_gap),
        required_relative_gap=required_relative_gap,
        iterations=diag.iterations,
        runtime_s=diag.runtime_s,
        grid_shape=diag.grid_shape,
        time_slice_count=time_slice_count,
        no_fly_cell_count=int(no_fly.size),
        target_detection_probabilities=diag.target_detection_probabilities,
    )
    # 웨이포인트 고도는 시뮬레이터용 AGL (평면세계). 실제 비행 MSL 은 계획
    # 메타/no-fly 에만 쓴다 — 여기서 MSL 을 쓰면 search camera 가 그만큼 높이 있다고
    # 착각해(문제2) 가까운 탐색대상을 놓친다.
    waypoints = _routes_to_waypoints(
        plan.routes,
        origin_lat,
        origin_lon,
        search_camera_altitude_agl_m,
        deconflict_offset_m=deconflict_factor * cell_m,
    )
    return CertifiedSearchPlan(
        certificate=certificate,
        vehicle_waypoints=waypoints,
        assignments=plan.assignments,
    )


def _routes_to_waypoints(
    routes, origin_lat, origin_lon, altitude_m, *, deconflict_offset_m: float = 0.0
):
    """cpp_search Route(로컬 ENU) 목록 → 기체별 위경도 웨이포인트.

    ``deconflict_offset_m`` > 0 이면 기체마다 **고유 방향**(균등 분할)의 상수
    측방 오프셋을 경로 전체에 더한다. 두 기체가 같은 셀을 커버해도 동일 라인이
    아니라 평행하게 밀린 라인으로 날아 트랙 겹침(좌표 일치)이 사라진다. 오프셋은
    셀변보다 훨씬 작아 인증된 셀 커버리지는 보존된다.
    """
    out: dict[int, list[tuple[float, float, float]]] = {}
    # 경로는 탐색자 순서대로 온다. vehicle_id 는 0-기반일 수 있어 신뢰하지 않고
    # 순서(1..N)로 매핑한다 (0 or index 관용식은 vehicle_id==0 에서 충돌).
    route_count = len(routes)
    for index, route in enumerate(routes, start=1):
        vehicle_id = index
        if deconflict_offset_m > 0.0 and route_count > 1:
            angle = 2.0 * math.pi * (index - 1) / route_count
            offset_east = deconflict_offset_m * math.cos(angle)
            offset_north = deconflict_offset_m * math.sin(angle)
        else:
            offset_east = offset_north = 0.0
        pts: list[tuple[float, float, float]] = []
        segments = list(getattr(route, "segments", []))
        for seg_index, seg in enumerate(segments):
            start = seg.start
            lat, lon = _local_to_lonlat(
                start.x + offset_east, start.y + offset_north, origin_lat, origin_lon
            )
            pts.append((lat, lon, float(altitude_m)))
            if seg_index == len(segments) - 1:
                end = seg.end
                lat_e, lon_e = _local_to_lonlat(
                    end.x + offset_east, end.y + offset_north, origin_lat, origin_lon
                )
                pts.append((lat_e, lon_e, float(altitude_m)))
        out[vehicle_id] = pts
    return out


def replan_after_detection(
    store: SiteStore,
    found_track_ids: set[int] | frozenset[int] | tuple[int, ...],
    **kwargs,
) -> CertifiedSearchPlan:
    """발견된 탐색대상을 빼고 남은 대상으로 SPX 재계획(다인원 순차 탐색).

    한 명을 찾으면 그 대상을 belief 에서 제거하고 남은 탐색대상에 대해 인증 계획을
    다시 뽑는다. 온라인 증분 재해결(옵션 1)의 트리거가 이 함수를 호출한다.
    ``plan_certified_search`` 의 모든 인자를 그대로 넘길 수 있다.
    """
    return plan_certified_search(
        store,
        exclude_track_ids=set(int(t) for t in found_track_ids),
        **kwargs,
    )


def apply_plan_to_store(store: SiteStore, plan: CertifiedSearchPlan) -> None:
    """인증 계획의 웨이포인트를 SiteStore vehicle_waypoints 에 기록한다."""
    for vehicle_id in SiteStore.VEHICLE_IDS:
        pts = plan.vehicle_waypoints.get(vehicle_id, [])
        route: list[MissionPoint] = []
        for sequence, (lat, lon, alt) in enumerate(pts, start=1):
            route.append(
                MissionPoint(
                    latitude=lat,
                    longitude=lon,
                    altitude_m=alt,
                    code=f"WP{sequence:03d}",
                    label=f"UAV-{vehicle_id:02d} SPX {sequence}",
                    point_type="WAYPOINT",
                    sequence=sequence,
                )
            )
        store.vehicle_waypoints[vehicle_id] = route
    store.notify()
