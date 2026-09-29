"""경로 기하 KPI — 얼마나 덮었고 얼마나 겹쳤고 얼마나 날았나.

탐지 여부는 여기서 판단하지 않는다. 순수 기하만 다룬다.

핵심 수식
---------
* 임무시간 클리핑 (``_segments_within_window``)

      t_seg = L_flown / v,   v = 탐색속도 또는 이동속도
      L_flown = weave 적용 거리 (센서 on) 또는 직선거리 (센서 off)

  남은 시간이 모자라면 그 구간을 비율로 잘라낸다. 모든 조건이 같은
  시간창에서 평가되도록 만드는 장치다.

* 명목 소인부하 (``coverage_load_ratio``)

      load = (중심선 센싱거리 * W) / A_AOI

  중복을 세지 않으므로 1을 넘을 수 있다. 아래의 고유 면적률과 다르다.

* 고유 면적 탐색률과 중복률 (``_swarm_raster_coverage``)
  래스터 격자를 깔고 각 LM의 센서 on 발자국이 덮는 셀 집합을 만든 뒤

      unique = |합집합| * cell^2 / A_AOI
      redundancy = 1 - |합집합| / sum_k |SAR_k가 덮은 셀|

  한 LM이 같은 셀을 두 번 덮는 것은 1회로 센다. 중복률은 **SAR 사이**
  중복만 잡는다.

* 확률질량 탐색률 (``probability_mass_coverage``)

      mass = sum_{점 p: 소인선까지 거리 <= 지지반폭} p(x)

  면적이 아니라 표적이 있을 법한 곳을 얼마나 덮었는지를 잰다.

의존
----
* 위: ``models``.
* 아래: ``simulation``(같은 기하를 탐지 평가와 함께 씀),
  모든 챕터 실험이 경로 KPI를 여기서 얻는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from statistics import mean
from typing import Sequence

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, Route, SensorSpec


@dataclass(frozen=True, slots=True)
class GeometricMetrics:
    planner_name: str
    route_count: int
    total_distance_m: float
    mean_route_distance_m: float
    max_route_distance_m: float
    sensing_distance_m: float
    estimated_completion_time_s: float
    coverage_load_ratio: float
    coverage_redundancy_ratio: float
    planned_centerline_distance_m: float = 0.0
    centerline_sensing_distance_m: float = 0.0
    unique_area_coverage_ratio: float = 0.0
    equivalent_sweep_width_m: float = 0.0
    probability_mass_coverage: float = 0.0


def clip_routes_to_time(
    routes: list[Route],
    mission: MissionConfig,
    time_limit_s: float,
    sensor: SensorSpec | None = None,
) -> list[Route]:
    """Return route prefixes actually executable inside a common time window."""

    if time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive")
    active_sensor = sensor if sensor is not None else SensorSpec()
    return [
        Route(
            route.planner_name,
            route.vehicle_id,
            _segments_within_window(route, mission, active_sensor, time_limit_s),
        )
        for route in routes
    ]


def evaluate_routes(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    time_limit_s: float | None = None,
    target_support_points: Sequence[Point2D] | None = None,
    target_probability_mass: Sequence[float] | None = None,
) -> GeometricMetrics:
    if not routes:
        raise ValueError("at least one route is required")
    if time_limit_s is not None and time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive")

    clipped_routes = [
        _segments_within_window(route, mission, sensor, time_limit_s)
        for route in routes
    ]
    centerline_route_distances = [
        sum(segment.length_m for segment in segments)
        for segments in clipped_routes
    ]
    route_distances = [
        sum(segment.flown_distance_m(sensor) for segment in segments)
        for segments in clipped_routes
    ]
    route_times = [
        sum(segment.duration_s(mission, sensor) for segment in segments)
        for segments in clipped_routes
    ]
    centerline_sensing_distance = sum(
        segment.length_m
        for segments in clipped_routes
        for segment in segments
        if segment.sensor_on
    )
    sensing_distance = sum(
        segment.flown_distance_m(sensor)
        for segments in clipped_routes
        for segment in segments
        if segment.sensor_on
    )
    nominal_swept_area = centerline_sensing_distance * sensor.effective_sweep_width_m
    coverage_load_ratio = nominal_swept_area / mission.total_area_m2
    unique_coverage_ratio, overlap_proxy = _swarm_raster_coverage(
        clipped_routes,
        mission,
        sensor,
    )
    covered_probability_mass = (
        probability_mass_coverage(
            routes,
            mission,
            sensor,
            target_support_points,
            target_probability_mass,
            time_limit_s=time_limit_s,
        )
        if target_support_points is not None
        and target_probability_mass is not None
        else 0.0
    )

    return GeometricMetrics(
        planner_name=routes[0].planner_name,
        route_count=len(routes),
        total_distance_m=sum(route_distances),
        mean_route_distance_m=mean(route_distances),
        max_route_distance_m=max(route_distances),
        sensing_distance_m=sensing_distance,
        estimated_completion_time_s=max(route_times),
        coverage_load_ratio=coverage_load_ratio,
        coverage_redundancy_ratio=overlap_proxy,
        planned_centerline_distance_m=sum(centerline_route_distances),
        centerline_sensing_distance_m=centerline_sensing_distance,
        unique_area_coverage_ratio=unique_coverage_ratio,
        equivalent_sweep_width_m=sensor.effective_sweep_width_m,
        probability_mass_coverage=covered_probability_mass,
    )


def probability_mass_coverage(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    target_support_points: Sequence[Point2D],
    target_probability_mass: Sequence[float],
    *,
    time_limit_s: float | None = None,
) -> float:
    """Return initial target-location mass inside the unique search support."""

    points = tuple(target_support_points)
    masses = tuple(float(value) for value in target_probability_mass)
    if not points or len(points) != len(masses):
        raise ValueError("one probability mass is required per target support point")
    if any(value < 0.0 for value in masses) or sum(masses) <= 0.0:
        raise ValueError("target probability mass must be non-negative with positive sum")
    normalized = tuple(value / sum(masses) for value in masses)
    segments = [
        segment
        for route in routes
        for segment in _segments_within_window(route, mission, sensor, time_limit_s)
        if segment.sensor_on and segment.length_m > 1e-12
    ]
    half_width_squared = sensor.coverage_half_width_m**2
    covered_mass = 0.0
    for point, mass in zip(points, normalized):
        if any(
            _point_segment_distance_squared(point, segment) <= half_width_squared
            for segment in segments
        ):
            covered_mass += mass
    return min(max(covered_mass, 0.0), 1.0)


def _segments_within_window(
    route: Route,
    mission: MissionConfig,
    sensor: SensorSpec,
    time_limit_s: float | None,
) -> tuple[PathSegment, ...]:
    if time_limit_s is None:
        return route.segments

    elapsed_s = 0.0
    clipped: list[PathSegment] = []
    for segment in route.segments:
        duration_s = segment.duration_s(mission, sensor)
        remaining_s = time_limit_s - elapsed_s
        if remaining_s <= 1e-12:
            break
        if duration_s <= remaining_s + 1e-12:
            clipped.append(segment)
            elapsed_s += duration_s
            continue

        ratio = min(max(remaining_s / max(duration_s, 1e-12), 0.0), 1.0)
        clipped_end = Point2D(
            segment.start.x + (segment.end.x - segment.start.x) * ratio,
            segment.start.y + (segment.end.y - segment.start.y) * ratio,
        )
        clipped.append(
            PathSegment(
                segment.start,
                clipped_end,
                segment.sensor_on,
                segment.detection_scale,
                segment.sensor_mode,
                segment.speed_mps,
                segment.search_pattern,
            )
        )
        break
    return tuple(clipped)


def _swarm_raster_coverage(
    clipped_routes: list[tuple[PathSegment, ...]],
    mission: MissionConfig,
    sensor: SensorSpec,
) -> tuple[float, float]:
    """Approximate unique area coverage and cross-SAR footprint redundancy.

    Each SAR contributes a union of raster cells covered by its sensor-on path.
    Repeated coverage inside one SAR route is counted once; cells shared by two
    or more SAR routes are counted as swarm-level redundant coverage.
    """

    half_width = sensor.coverage_half_width_m
    grid_step = max(
        min(half_width / 4.0, sensor.instantaneous_swath_m / 2.0),
        1.0,
    )
    route_cells: list[set[tuple[int, int]]] = []
    radius_squared = mission.search_radius_m**2
    footprint_squared = half_width**2

    for segments in clipped_routes:
        covered: set[tuple[int, int]] = set()
        for segment in segments:
            if not segment.sensor_on or segment.length_m <= 1e-12:
                continue
            min_x = floor((min(segment.start.x, segment.end.x) - half_width) / grid_step)
            max_x = ceil((max(segment.start.x, segment.end.x) + half_width) / grid_step)
            min_y = floor((min(segment.start.y, segment.end.y) - half_width) / grid_step)
            max_y = ceil((max(segment.start.y, segment.end.y) + half_width) / grid_step)
            dx = segment.end.x - segment.start.x
            dy = segment.end.y - segment.start.y
            length_squared = dx * dx + dy * dy
            for grid_x in range(min_x, max_x + 1):
                x = grid_x * grid_step
                for grid_y in range(min_y, max_y + 1):
                    y = grid_y * grid_step
                    if (
                        (x - mission.center.x) ** 2
                        + (y - mission.center.y) ** 2
                        > radius_squared
                    ):
                        continue
                    projection = min(
                        max(
                            ((x - segment.start.x) * dx + (y - segment.start.y) * dy)
                            / length_squared,
                            0.0,
                        ),
                        1.0,
                    )
                    closest_x = segment.start.x + projection * dx
                    closest_y = segment.start.y + projection * dy
                    if (x - closest_x) ** 2 + (y - closest_y) ** 2 <= footprint_squared:
                        covered.add((grid_x, grid_y))
        route_cells.append(covered)

    attempted_cells = sum(len(cells) for cells in route_cells)
    if attempted_cells == 0:
        return 0.0, 0.0
    unique_cells: set[tuple[int, int]] = set()
    for cells in route_cells:
        unique_cells.update(cells)
    # 고유 면적률 = |셀 합집합| * cell^2 / A_AOI
    # 중복률     = 1 - |합집합| / sum_k |SAR_k가 덮은 셀|
    # 한 LM이 같은 셀을 두 번 덮는 것은 1회로 세므로, 중복률은 SAR 사이
    # 중복만 잡는다.
    unique_area_ratio = min(
        1.0,
        len(unique_cells) * grid_step * grid_step / mission.total_area_m2,
    )
    redundancy_ratio = max(0.0, 1.0 - len(unique_cells) / attempted_cells)
    return unique_area_ratio, redundancy_ratio


def _point_segment_distance_squared(point: Point2D, segment: PathSegment) -> float:
    dx = segment.end.x - segment.start.x
    dy = segment.end.y - segment.start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-15:
        return (
            (point.x - segment.start.x) ** 2
            + (point.y - segment.start.y) ** 2
        )
    projection = min(
        max(
            (
                (point.x - segment.start.x) * dx
                + (point.y - segment.start.y) * dy
            )
            / length_squared,
            0.0,
        ),
        1.0,
    )
    closest_x = segment.start.x + projection * dx
    closest_y = segment.start.y + projection * dy
    return (point.x - closest_x) ** 2 + (point.y - closest_y) ** 2
