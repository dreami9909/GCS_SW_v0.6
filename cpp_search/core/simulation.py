"""Monte Carlo 진리 생성과 탐지시간 평가.

계획기가 만든 경로를, 계획기가 보지 못한 진리 표적 앙상블에 대고 채점한다.

핵심 수식
---------
* 진리 궤적 (``generate_moving_targets``)
  표적마다 독립 난수열을 준다:

      seed_k = (base_seed XOR 0x5EED40) + k * 0x9E3779B9

  임무시간을 늘려도 앞부분 궤적이 바뀌지 않게 하려는 설계다.

* 시간이 붙은 소인 구간 (``_build_timed_sensing_segments``)
  각 구간에 [시작시각, 종료시각]과 속도를 붙인다. 표적도 움직이므로
  **상대운동**으로 노출구간을 구해야 한다.

* 상대운동 노출 (``_relative_motion_exposure``)
  표적과 센서가 동시에 움직일 때, 상대위치가 탐지 반경 원 안에 있는
  시간구간을 2차 방정식으로 푼다:

      |p_rel(t)|^2 = R^2
      -> a t^2 + b t + c = 0,  근 사이가 노출구간

* 팀 누적 위험률 -> 탐지시각 (``_detection_time_from_hazard_intervals``)
  여러 LM의 노출구간이 겹칠 수 있으므로 (시작, +rate) / (종료, -rate)
  이벤트를 시간순으로 스윕하면서

      Lambda(t) = ∫ lambda_active(s) ds

  를 누적하고, 목표 임계 threshold = -ln(U), U ~ Uniform(0,1)에 도달하는
  시각을 선형 보간으로 되돌린다. 즉 **역변환 표집**이다.

* KPI (``PerformanceMetrics``)

      detection_probability_within_limit = #{T_d <= T_lim} / N
      failure_rate_within_limit          = 1 - 위 값
      conditional_mean_detection_time_s  = mean{ T_d : T_d <= T_lim }
      restricted_mean_detection_time_s   = mean{ min(T_d, T_lim) }

  마지막 두 개는 서로 다른 양이다. 앞은 성공 표본만, 뒤는 실패를
  제한시간으로 절단해 전부 포함한다.

의존
----
* 위: ``models``, ``motion``, ``probability``, ``evaluation``,
  ``sensor_observation``(탐지모델 주입), ``terrain``(주입).
* 아래: 모든 챕터 실험의 최종 채점.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import atan2, cos, hypot, inf, log, sin, sqrt, tau
from random import Random
from statistics import mean

from cpp_search.core.evaluation import evaluate_routes
from cpp_search.core.motion import (
    KinematicState,
    RANDOM_MANEUVER_MODE,
    TargetMotionSpec,
    TargetTrajectory,
    apply_circular_boundary,
    propagate_kinematics,
    sample_initial_kinematics,
)
from cpp_search.core.models import MissionConfig, PathSegment, Point2D, Route, SensorSpec
from cpp_search.core.probability import TargetPrior


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    sample_count: int = 1_000
    seed: int = 20_260_808
    detection_time_limit_s: float = 5.0 * 60.0

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.detection_time_limit_s <= 0:
            raise ValueError("detection_time_limit_s must be positive")


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    planner_name: str
    conditional_mean_detection_time_s: float
    restricted_mean_detection_time_s: float
    detection_probability_within_limit: float
    failure_rate_within_limit: float
    total_distance_m: float
    coverage_redundancy_ratio: float
    detected_count: int
    sample_count: int
    mission_completion_time_s: float
    escaped_missed_count: int
    stayed_missed_count: int
    full_route_total_distance_m: float = 0.0
    full_route_coverage_redundancy_ratio: float = 0.0
    evaluation_window_s: float = 0.0
    full_route_failure_rate: float = 0.0
    unique_area_coverage_ratio: float = 0.0
    mean_route_distance_m: float = 0.0
    max_route_distance_m: float = 0.0
    centerline_total_distance_m: float = 0.0
    probability_mass_coverage: float = 0.0
    detection_times_s: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class _TimedSensingSegment:
    start: Point2D
    unit_x: float
    unit_y: float
    length_m: float
    start_time_s: float
    end_time_s: float
    speed_mps: float
    detection_scale: float
    sensor_mode: str | None = None


def generate_targets(
    mission: MissionConfig,
    config: EvaluationConfig,
    prior: TargetPrior,
    terrain=None,
    terrain_stop_ratio: float = 0.4,
    terrain_mode_probabilities: tuple[float, ...] | None = None,
) -> list[Point2D]:
    """Sample static targets from the configured spatial prior.

    When ``terrain`` is given, the radially-symmetric prior is reweighted by
    the terrain field so truth targets concentrate on trafficable corridors
    and concealment patches (and avoid barriers), matching the belief prior.
    """
    rng = Random(config.seed)
    targets: list[Point2D] = []
    for _ in range(config.sample_count):
        while True:
            radius = mission.search_radius_m * sqrt(rng.random())
            angle = tau * rng.random()
            x = mission.center.x + radius * cos(angle)
            y = mission.center.y + radius * sin(angle)
            acceptance = prior.relative_density(radius, mission.search_radius_m)
            if terrain is not None:
                acceptance *= terrain.prior_weight(
                    x,
                    y,
                    terrain_stop_ratio,
                    terrain_mode_probabilities,
                )
            if rng.random() <= acceptance:
                targets.append(Point2D(x, y))
                break
    return targets


def generate_moving_targets(
    mission: MissionConfig,
    config: EvaluationConfig,
    prior: TargetPrior,
    motion: TargetMotionSpec,
    horizon_s: float,
    terrain=None,
    terrain_stop_ratio: float = 0.4,
    terrain_bias_strength: float = 0.0,
    terrain_offroad_probability: float = 0.2,
    terrain_halt_probability_boost: float = 0.0,
) -> list[TargetTrajectory]:
    """Generate shared piecewise-linear Markov target trajectories."""
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be positive")

    initial_points = generate_targets(
        mission,
        config,
        prior,
        terrain,
        terrain_stop_ratio,
        (
            motion.effective_initial_mode_probabilities
            if motion.motion_model == "imm5"
            else None
        ),
    )
    trajectories: list[TargetTrajectory] = []
    for target_index, initial_point in enumerate(initial_points):
        # Give every target its own deterministic stream.  Extending the
        # mission horizon then extends each trajectory without changing the
        # prefixes (or shifting the random streams of later targets).
        trajectory_seed = (
            (config.seed ^ 0x5EED_40)
            + target_index * 0x9E37_79B9
        )
        trajectories.append(
            _generate_moving_trajectory(
                mission,
                initial_point,
                motion,
                horizon_s,
                trajectory_seed,
                (
                    motion.behavior_profile.name
                    if motion.behavior_profile is not None
                    else motion.motion_model.upper()
                ),
                terrain=terrain,
                terrain_bias_strength=terrain_bias_strength,
                terrain_offroad_probability=terrain_offroad_probability,
                terrain_halt_probability_boost=terrain_halt_probability_boost,
            )
        )
    return trajectories


def _generate_moving_trajectory(
    mission: MissionConfig,
    initial_point: Point2D,
    motion: TargetMotionSpec,
    horizon_s: float,
    seed: int,
    profile_name: str,
    initial_state: KinematicState | None = None,
    terrain=None,
    terrain_bias_strength: float = 0.0,
    terrain_offroad_probability: float = 0.2,
    terrain_halt_probability_boost: float = 0.0,
) -> TargetTrajectory:
    rng = Random(seed)
    times = [0.0]
    points = [initial_point]
    modes: list[int] = []
    elapsed_s = 0.0
    escaped_at_s: float | None = None
    kinematic_state = initial_state
    if kinematic_state is None and motion.motion_model in {"imm", "imm5"}:
        kinematic_state = sample_initial_kinematics(motion, rng)

    while elapsed_s < horizon_s - 1e-9:
        delta_s = min(motion.step_s, horizon_s - elapsed_s)
        previous = points[-1]
        if kinematic_state is None:
            speed_ratio = min(
                max(
                    rng.gauss(
                        motion.nominal_speed_ratio,
                        motion.speed_sigma_ratio,
                    ),
                    0.0,
                ),
                1.0,
            )
            speed_mps = speed_ratio * motion.max_speed_mps
            heading = tau * rng.random()
            dx = speed_mps * delta_s * cos(heading)
            dy = speed_mps * delta_s * sin(heading)
            step_state = KinematicState(
                speed_mps,
                heading,
                0.0,
                RANDOM_MANEUVER_MODE,
            )
        else:
            halt_boost = (
                terrain.halt_probability_boost(
                    previous.x,
                    previous.y,
                    terrain_halt_probability_boost,
                )
                if terrain is not None
                and terrain_halt_probability_boost > 0.0
                and motion.motion_model == "imm5"
                else 0.0
            )
            dx, dy, kinematic_state = propagate_kinematics(
                motion,
                kinematic_state,
                delta_s,
                rng,
                halt_probability_boost=halt_boost,
            )
            step_state = kinematic_state
        # Map-constrained transition: follow corridor tangents most of the
        # time, retain a stochastic off-road branch, and avoid hard barriers.
        if terrain is not None:
            dx, dy = terrain.transition_step(
                previous.x,
                previous.y,
                dx,
                dy,
                follow_strength=terrain_bias_strength,
                offroad_probability=terrain_offroad_probability,
                rng=rng,
                motion_mode=(
                    step_state.mode if motion.motion_model == "imm5" else None
                ),
            )
            if kinematic_state is not None and hypot(dx, dy) > 1e-9:
                kinematic_state = KinematicState(
                    kinematic_state.speed_mps,
                    atan2(dy, dx) % tau,
                    kinematic_state.turn_rate_rad_s,
                    kinematic_state.mode,
                )
                step_state = kinematic_state
        if motion.boundary_mode == "reflect":
            next_point, reflected_state, _ = apply_circular_boundary(
                previous,
                dx,
                dy,
                step_state,
                mission.center,
                mission.search_radius_m,
                motion.boundary_mode,
            )
            if kinematic_state is not None:
                kinematic_state = reflected_state
                step_state = reflected_state
        else:
            next_point = Point2D(previous.x + dx, previous.y + dy)
        elapsed_s += delta_s
        times.append(elapsed_s)
        points.append(next_point)
        modes.append(step_state.mode)

        if (
            motion.boundary_mode == "escape"
            and escaped_at_s is None
            and mission.center.distance_to(next_point)
            >= mission.search_radius_m
        ):
            escaped_at_s = elapsed_s

    return TargetTrajectory(
        tuple(times),
        tuple(points),
        profile_name=profile_name,
        motion_modes=tuple(modes),
        escaped_at_s=escaped_at_s,
    )


def evaluate_monte_carlo(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    targets: list[Point2D] | list[TargetTrajectory],
    config: EvaluationConfig,
    detection_model=None,
    sensor_detection_seed: int | None = None,
) -> PerformanceMetrics:
    if not routes:
        raise ValueError("at least one route is required")
    if not targets:
        raise ValueError("at least one target sample is required")

    timed_segments = _build_timed_sensing_segments(routes, mission, sensor)
    detection_radius = sensor.coverage_half_width_m
    detection_times = []
    detection_seed = (
        sensor_detection_seed
        if sensor_detection_seed is not None
        else config.seed ^ 0x0B5E_7A11
    )
    for target_index, target in enumerate(targets):
        detection_rng = Random(
            (detection_seed + target_index * 0x9E37_79B9) & 0xFFFF_FFFF
        )
        if isinstance(target, TargetTrajectory):
            detection_times.append(
                (
                    _first_moving_detection_time(
                        target,
                        timed_segments,
                        detection_radius,
                        mission.search_speed_mps,
                    )
                    if detection_model is None
                    else _first_moving_spatial_detection_time(
                        target,
                        timed_segments,
                        detection_radius,
                        mission.search_speed_mps,
                        sensor,
                        detection_model,
                        detection_rng,
                    )
                )
            )
        else:
            detection_times.append(
                (
                    _first_detection_time(
                        target,
                        timed_segments,
                        detection_radius,
                        mission.search_speed_mps,
                    )
                    if detection_model is None
                    else _first_static_spatial_detection_time(
                        target,
                        timed_segments,
                        detection_radius,
                        mission.search_speed_mps,
                        sensor,
                        detection_model,
                        detection_rng,
                    )
                )
            )
    successful_times = [time_s for time_s in detection_times if time_s < inf]
    statistics = detection_time_statistics(
        detection_times,
        config.detection_time_limit_s,
    )
    full_geometric = evaluate_routes(routes, mission, sensor)
    initial_target_points = [
        target.points[0] if isinstance(target, TargetTrajectory) else target
        for target in targets
    ]
    uniform_target_mass = [1.0 / len(initial_target_points)] * len(initial_target_points)
    window_geometric = evaluate_routes(
        routes,
        mission,
        sensor,
        time_limit_s=config.detection_time_limit_s,
        target_support_points=initial_target_points,
        target_probability_mass=uniform_target_mass,
    )
    escaped_missed_count = 0
    stayed_missed_count = 0
    for target, detection_time in zip(targets, detection_times):
        if detection_time < inf:
            continue
        # 이탈 판정은 오직 **평가 시점의 위치가 AOI 밖인지**로 한다.
        #
        # 예전 판정에는 "궤적이 경로 완주시각보다 먼저 끝났으면 이탈"이라는
        # 조건이 붙어 있었다. boundary_mode="open"으로 바뀐 뒤 진리 궤적은
        # 항상 임무시간까지만 생성되므로, 경로 완주시각이 임무시간보다 길기만
        # 하면 **모든 미탐지가 이탈로 찍혔다**(이탈률 = 실패율, 계획기 책임 0).
        # 그 조건을 제거한다.
        #
        # 평가 시점은 제한시간이며, 궤적이 그보다 짧으면 마지막 위치를 쓴다.
        escaped = False
        if isinstance(target, TargetTrajectory):
            evaluation_time_s = min(
                config.detection_time_limit_s, target.end_time_s
            )
            position = target.position_at_s(evaluation_time_s)
            escaped = (
                mission.center.distance_to(position) >= mission.search_radius_m
            )
        if escaped:
            escaped_missed_count += 1
        else:
            stayed_missed_count += 1

    return PerformanceMetrics(
        planner_name=routes[0].planner_name,
        conditional_mean_detection_time_s=float(
            statistics["conditional_mean_detection_time_s"]
        ),
        restricted_mean_detection_time_s=float(
            statistics["restricted_mean_detection_time_s"]
        ),
        detection_probability_within_limit=float(
            statistics["detection_probability_within_limit"]
        ),
        failure_rate_within_limit=float(statistics["failure_rate_within_limit"]),
        total_distance_m=window_geometric.total_distance_m,
        coverage_redundancy_ratio=window_geometric.coverage_redundancy_ratio,
        detected_count=int(statistics["detected_count"]),
        sample_count=len(targets),
        mission_completion_time_s=full_geometric.estimated_completion_time_s,
        escaped_missed_count=escaped_missed_count,
        stayed_missed_count=stayed_missed_count,
        full_route_total_distance_m=full_geometric.total_distance_m,
        full_route_coverage_redundancy_ratio=full_geometric.coverage_redundancy_ratio,
        evaluation_window_s=config.detection_time_limit_s,
        full_route_failure_rate=1.0 - len(successful_times) / len(targets),
        unique_area_coverage_ratio=window_geometric.unique_area_coverage_ratio,
        mean_route_distance_m=window_geometric.mean_route_distance_m,
        max_route_distance_m=window_geometric.max_route_distance_m,
        centerline_total_distance_m=window_geometric.planned_centerline_distance_m,
        probability_mass_coverage=window_geometric.probability_mass_coverage,
        detection_times_s=tuple(detection_times),
    )


def detection_time_statistics(
    detection_times_s: list[float] | tuple[float, ...],
    detection_time_limit_s: float,
) -> dict[str, float | int]:
    """Compute canonical detection KPIs from one trajectory ensemble."""

    if not detection_times_s:
        raise ValueError("at least one detection time is required")
    if detection_time_limit_s <= 0.0:
        raise ValueError("detection_time_limit_s must be positive")
    within_limit = [
        time_s for time_s in detection_times_s if time_s <= detection_time_limit_s
    ]
    detected_count = len(within_limit)
    sample_count = len(detection_times_s)
    detection_probability = detected_count / sample_count
    capped = [min(time_s, detection_time_limit_s) for time_s in detection_times_s]
    return {
        "conditional_mean_detection_time_s": (
            mean(within_limit) if within_limit else inf
        ),
        "restricted_mean_detection_time_s": mean(capped),
        "detection_probability_within_limit": detection_probability,
        "failure_rate_within_limit": 1.0 - detection_probability,
        "detected_count": detected_count,
        "sample_count": sample_count,
    }


def first_moving_target_detection_time(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    trajectory: TargetTrajectory,
    *,
    detection_model=None,
    detection_seed: int = 0,
) -> float:
    """Return one trajectory's detection time without aggregate route metrics."""

    if not routes:
        raise ValueError("at least one route is required")
    timed_segments = _build_timed_sensing_segments(routes, mission, sensor)
    radius_m = sensor.coverage_half_width_m
    if detection_model is None:
        return _first_moving_detection_time(
            trajectory,
            timed_segments,
            radius_m,
            mission.search_speed_mps,
        )
    return _first_moving_spatial_detection_time(
        trajectory,
        timed_segments,
        radius_m,
        mission.search_speed_mps,
        sensor,
        detection_model,
        Random(detection_seed),
    )


def moving_target_detection_indicators(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    targets: list[TargetTrajectory],
    detection_time_limit_s: float,
    *,
    detection_model=None,
    sensor_detection_seed: int = 0,
) -> tuple[bool, ...]:
    """Return paired within-limit outcomes without rebuilding route timing."""

    if not routes:
        raise ValueError("at least one route is required")
    if not targets:
        raise ValueError("at least one moving target is required")
    if detection_time_limit_s <= 0.0:
        raise ValueError("detection time limit must be positive")
    timed_segments = _build_timed_sensing_segments(routes, mission, sensor)
    detection_radius = sensor.coverage_half_width_m
    outcomes = []
    for target_index, target in enumerate(targets):
        detection_seed = (
            sensor_detection_seed + target_index * 0x9E37_79B9
        ) & 0xFFFF_FFFF
        detection_time = (
            _first_moving_detection_time(
                target,
                timed_segments,
                detection_radius,
                mission.search_speed_mps,
            )
            if detection_model is None
            else _first_moving_spatial_detection_time(
                target,
                timed_segments,
                detection_radius,
                mission.search_speed_mps,
                sensor,
                detection_model,
                Random(detection_seed),
            )
        )
        outcomes.append(detection_time <= detection_time_limit_s)
    return tuple(outcomes)


def first_moving_target_exposure_time(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    trajectory: TargetTrajectory,
    start_time_s: float,
    end_time_s: float,
) -> float:
    """Return the first geometric footprint encounter inside a time window."""

    if not 0.0 <= start_time_s < end_time_s:
        raise ValueError("exposure window must have increasing non-negative times")
    timed_segments = _build_timed_sensing_segments(routes, mission, sensor)
    radius_m = sensor.coverage_half_width_m
    best_time = inf
    for segment in timed_segments:
        if segment.start_time_s >= end_time_s or segment.end_time_s <= start_time_s:
            continue
        overlap_start = max(start_time_s, segment.start_time_s)
        overlap_end = min(end_time_s, segment.end_time_s, trajectory.end_time_s)
        if overlap_end <= overlap_start:
            continue
        target_index = max(
            0,
            bisect_right(trajectory.times_s, overlap_start) - 1,
        )
        while (
            overlap_start < overlap_end - 1e-12
            and target_index < len(trajectory.times_s) - 1
        ):
            target_interval_end = trajectory.times_s[target_index + 1]
            interval_end = min(overlap_end, target_interval_end)
            hit_time = _relative_motion_entry_time(
                trajectory,
                target_index,
                segment,
                overlap_start,
                interval_end,
                radius_m,
                segment.speed_mps,
            )
            if hit_time < best_time:
                best_time = hit_time
                break
            overlap_start = interval_end
            if overlap_start >= target_interval_end - 1e-12:
                target_index += 1
    return best_time


def evaluate_monte_carlo_by_profile(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec,
    targets: list[TargetTrajectory],
    config: EvaluationConfig,
    detection_model=None,
    sensor_detection_seed: int | None = None,
) -> dict[str, PerformanceMetrics]:
    """Evaluate the same route set separately for each truth profile."""

    grouped: dict[str, list[TargetTrajectory]] = {}
    for target in targets:
        grouped.setdefault(target.profile_name, []).append(target)
    return {
        profile_name: evaluate_monte_carlo(
            routes,
            mission,
            sensor,
            profile_targets,
            config,
            detection_model=detection_model,
            sensor_detection_seed=sensor_detection_seed,
        )
        for profile_name, profile_targets in grouped.items()
    }


def timed_sensing_segments(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec | None = None,
) -> list["_TimedSensingSegment"]:
    """센서 on 구간에 [시작시각, 종료시각]과 속도를 붙여 돌려준다.

    탐지시간을 계산하려면 "어느 구간을 언제 날았나"가 필요하다. 이동표적
    평가(``evaluate_monte_carlo``)와 정지표적 평가(Ch1)가 **같은 시간 축**을
    쓰도록 이 함수를 공개한다.
    """

    return _build_timed_sensing_segments(routes, mission, sensor)


def _build_timed_sensing_segments(
    routes: list[Route],
    mission: MissionConfig,
    sensor: SensorSpec | None = None,
) -> list[_TimedSensingSegment]:
    active_sensor = sensor if sensor is not None else SensorSpec()
    timed_segments: list[_TimedSensingSegment] = []
    for route in routes:
        elapsed_time_s = 0.0
        for segment in route.segments:
            length = segment.length_m
            segment_speed = segment.centerline_speed_mps(mission, active_sensor)
            if segment.effective_detection_scale > 0.0 and length > 0.0:
                timed_segments.append(
                    _TimedSensingSegment(
                        start=segment.start,
                        unit_x=(segment.end.x - segment.start.x) / length,
                        unit_y=(segment.end.y - segment.start.y) / length,
                        length_m=length,
                        start_time_s=elapsed_time_s,
                        end_time_s=(
                            elapsed_time_s
                            + length / segment_speed
                        ),
                        speed_mps=segment_speed,
                        detection_scale=segment.effective_detection_scale,
                        sensor_mode=segment.sensor_mode,
                    )
                )
            elapsed_time_s += length / segment_speed
    timed_segments.sort(key=lambda segment: segment.start_time_s)
    return timed_segments


def _first_detection_time(
    target: Point2D,
    segments: list[_TimedSensingSegment],
    detection_radius_m: float,
    speed_mps: float,
) -> float:
    best_time = inf
    radius_squared = detection_radius_m**2

    for segment in segments:
        if segment.start_time_s >= best_time:
            break

        offset_x = target.x - segment.start.x
        offset_y = target.y - segment.start.y
        projected = offset_x * segment.unit_x + offset_y * segment.unit_y
        offset_squared = offset_x**2 + offset_y**2
        perpendicular_squared = max(0.0, offset_squared - projected**2)
        if perpendicular_squared > radius_squared:
            continue

        half_chord = sqrt(max(0.0, radius_squared - perpendicular_squared))
        entry_distance = max(0.0, projected - half_chord)
        exit_distance = projected + half_chord
        if exit_distance < 0.0 or entry_distance > segment.length_m:
            continue

        detection_time = segment.start_time_s + entry_distance / segment.speed_mps
        best_time = min(best_time, detection_time)

    return best_time


def _first_static_spatial_detection_time(
    target: Point2D,
    segments: list[_TimedSensingSegment],
    detection_radius_m: float,
    speed_mps: float,
    sensor: SensorSpec,
    detection_model,
    rng: Random,
) -> float:
    threshold = -log(max(rng.random(), 1e-15))
    hazard_intervals: list[tuple[float, float, float]] = []
    for segment in segments:
        segment_radius_m = (
            detection_model.footprint_half_width_m(sensor, segment.sensor_mode)
            if hasattr(detection_model, "footprint_half_width_m")
            and (
                segment.sensor_mode is not None
                or sensor.has_extended_search_envelope
            )
            else detection_radius_m
        )
        radius_squared = segment_radius_m * segment_radius_m
        offset_x = target.x - segment.start.x
        offset_y = target.y - segment.start.y
        projected_m = offset_x * segment.unit_x + offset_y * segment.unit_y
        perpendicular_squared = max(
            0.0,
            offset_x * offset_x + offset_y * offset_y - projected_m * projected_m,
        )
        if perpendicular_squared > radius_squared:
            continue
        half_chord_m = sqrt(max(0.0, radius_squared - perpendicular_squared))
        entry_m = max(0.0, projected_m - half_chord_m)
        exit_m = min(segment.length_m, projected_m + half_chord_m)
        if exit_m <= entry_m:
            continue
        entry_time_s = segment.start_time_s + entry_m / segment.speed_mps
        exposure_s = (exit_m - entry_m) / segment.speed_mps
        midpoint_m = 0.5 * (entry_m + exit_m)
        observer = Point2D(
            segment.start.x + segment.unit_x * midpoint_m,
            segment.start.y + segment.unit_y * midpoint_m,
        )
        rate = (
            detection_model.hazard_rate(
                target,
                observer,
                sensor,
                sensor_mode=segment.sensor_mode,
            )
            if segment.sensor_mode is not None
            else detection_model.hazard_rate(target, observer, sensor)
        ) * segment.detection_scale * sensor.lateral_detection_scale(
            sqrt(perpendicular_squared)
        )
        hazard_intervals.append((entry_time_s, entry_time_s + exposure_s, rate))
    return _detection_time_from_hazard_intervals(threshold, hazard_intervals)


def _first_moving_detection_time(
    trajectory: TargetTrajectory,
    segments: list[_TimedSensingSegment],
    detection_radius_m: float,
    search_speed_mps: float,
) -> float:
    best_time = inf
    for segment in segments:
        if segment.start_time_s >= best_time:
            break
        if segment.start_time_s >= trajectory.end_time_s:
            continue

        overlap_start = segment.start_time_s
        overlap_end = min(segment.end_time_s, trajectory.end_time_s)
        if overlap_end <= overlap_start:
            continue

        target_index = max(
            0,
            bisect_right(trajectory.times_s, overlap_start) - 1,
        )
        while (
            overlap_start < overlap_end - 1e-12
            and target_index < len(trajectory.times_s) - 1
        ):
            target_interval_end = trajectory.times_s[target_index + 1]
            interval_end = min(overlap_end, target_interval_end)
            hit_time = _relative_motion_entry_time(
                trajectory,
                target_index,
                segment,
                overlap_start,
                interval_end,
                detection_radius_m,
                segment.speed_mps,
            )
            if hit_time < best_time:
                best_time = hit_time
                break

            overlap_start = interval_end
            if overlap_start >= target_interval_end - 1e-12:
                target_index += 1

    return best_time


def _first_moving_spatial_detection_time(
    trajectory: TargetTrajectory,
    segments: list[_TimedSensingSegment],
    detection_radius_m: float,
    search_speed_mps: float,
    sensor: SensorSpec,
    detection_model,
    rng: Random,
) -> float:
    threshold = -log(max(rng.random(), 1e-15))
    hazard_intervals: list[tuple[float, float, float]] = []
    for segment in segments:
        if segment.start_time_s >= trajectory.end_time_s:
            continue
        overlap_start = segment.start_time_s
        overlap_end = min(segment.end_time_s, trajectory.end_time_s)
        if overlap_end <= overlap_start:
            continue

        target_index = max(0, bisect_right(trajectory.times_s, overlap_start) - 1)
        while (
            overlap_start < overlap_end - 1e-12
            and target_index < len(trajectory.times_s) - 1
        ):
            target_interval_end = trajectory.times_s[target_index + 1]
            interval_end = min(overlap_end, target_interval_end)
            segment_radius_m = (
                detection_model.footprint_half_width_m(sensor, segment.sensor_mode)
                if hasattr(detection_model, "footprint_half_width_m")
                and (
                    segment.sensor_mode is not None
                    or sensor.has_extended_search_envelope
                )
                else detection_radius_m
            )
            exposure = _relative_motion_exposure(
                trajectory,
                target_index,
                segment,
                overlap_start,
                interval_end,
                segment_radius_m,
                segment.speed_mps,
            )
            if exposure is not None:
                entry_time_s, exit_time_s, target, observer, heading = exposure
                rate = detection_model.hazard_rate(
                    target,
                    observer,
                    sensor,
                    target_heading_rad=heading,
                    **(
                        {"sensor_mode": segment.sensor_mode}
                        if segment.sensor_mode is not None
                        else {}
                    ),
                ) * segment.detection_scale * sensor.lateral_detection_scale(
                    target.distance_to(observer)
                )
                hazard_intervals.append((entry_time_s, exit_time_s, rate))

            overlap_start = interval_end
            if overlap_start >= target_interval_end - 1e-12:
                target_index += 1
    return _detection_time_from_hazard_intervals(threshold, hazard_intervals)


def _detection_time_from_hazard_intervals(
    threshold: float,
    intervals: list[tuple[float, float, float]],
) -> float:
    """Invert cumulative team hazard after merging simultaneous exposures."""

    events: list[tuple[float, float]] = []
    earliest_certain = inf
    for start_s, end_s, rate in intervals:
        if end_s <= start_s or rate <= 0.0:
            continue
        if rate == inf:
            earliest_certain = min(earliest_certain, start_s)
            continue
        events.append((start_s, rate))
        events.append((end_s, -rate))
    if not events:
        return earliest_certain

    events.sort(key=lambda item: item[0])
    active_rate = 0.0
    cumulative_hazard = 0.0
    previous_time = events[0][0]
    index = 0
    while index < len(events):
        event_time = events[index][0]
        interval_hazard = active_rate * (event_time - previous_time)
        if (
            active_rate > 0.0
            and threshold <= cumulative_hazard + interval_hazard
        ):
            # 누적 위험률이 임계 threshold = -ln(U)에 도달하는 시각을
            # 선형 보간으로 되돌린다 (역변환 표집).
            #   Lambda(t) = threshold  ->  t = t_prev + (threshold - Lambda_prev) / lambda
            detected = previous_time + (
                threshold - cumulative_hazard
            ) / active_rate
            return min(detected, earliest_certain)
        cumulative_hazard += interval_hazard
        while index < len(events) and events[index][0] == event_time:
            active_rate += events[index][1]
            index += 1
        previous_time = event_time
        if earliest_certain <= previous_time:
            return earliest_certain
    return earliest_certain


def _relative_motion_exposure(
    trajectory: TargetTrajectory,
    target_index: int,
    sensing_segment: _TimedSensingSegment,
    start_time_s: float,
    end_time_s: float,
    detection_radius_m: float,
    search_speed_mps: float,
) -> tuple[float, float, Point2D, Point2D, float | None] | None:
    duration = end_time_s - start_time_s
    if duration <= 0.0:
        return None

    target_start_time = trajectory.times_s[target_index]
    target_end_time = trajectory.times_s[target_index + 1]
    target_duration = target_end_time - target_start_time
    target_start = trajectory.points[target_index]
    target_end = trajectory.points[target_index + 1]
    target_vx = (target_end.x - target_start.x) / target_duration
    target_vy = (target_end.y - target_start.y) / target_duration
    target_elapsed = start_time_s - target_start_time
    target_x = target_start.x + target_vx * target_elapsed
    target_y = target_start.y + target_vy * target_elapsed

    uav_elapsed = start_time_s - sensing_segment.start_time_s
    uav_x = (
        sensing_segment.start.x
        + sensing_segment.unit_x * search_speed_mps * uav_elapsed
    )
    uav_y = (
        sensing_segment.start.y
        + sensing_segment.unit_y * search_speed_mps * uav_elapsed
    )
    relative_x = target_x - uav_x
    relative_y = target_y - uav_y
    relative_vx = target_vx - sensing_segment.unit_x * search_speed_mps
    relative_vy = target_vy - sensing_segment.unit_y * search_speed_mps
    quadratic = relative_vx * relative_vx + relative_vy * relative_vy
    linear = 2.0 * (relative_x * relative_vx + relative_y * relative_vy)
    constant = (
        relative_x * relative_x
        + relative_y * relative_y
        - detection_radius_m * detection_radius_m
    )

    if quadratic <= 1e-15:
        if constant > 0.0:
            return None
        entry_delta = 0.0
        exit_delta = duration
    else:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return None
        root_offset = sqrt(max(discriminant, 0.0))
        lower = (-linear - root_offset) / (2.0 * quadratic)
        upper = (-linear + root_offset) / (2.0 * quadratic)
        entry_delta = max(0.0, lower)
        exit_delta = min(duration, upper)
        if exit_delta <= entry_delta:
            return None

    midpoint_delta = 0.5 * (entry_delta + exit_delta)
    target = Point2D(
        target_x + target_vx * midpoint_delta,
        target_y + target_vy * midpoint_delta,
    )
    observer = Point2D(
        uav_x
        + sensing_segment.unit_x * search_speed_mps * midpoint_delta,
        uav_y
        + sensing_segment.unit_y * search_speed_mps * midpoint_delta,
    )
    return (
        start_time_s + entry_delta,
        start_time_s + exit_delta,
        target,
        observer,
        (
            atan2(target_vy, target_vx)
            if hypot(target_vx, target_vy) > 1e-9
            else None
        ),
    )


def _relative_motion_entry_time(
    trajectory: TargetTrajectory,
    target_index: int,
    sensing_segment: _TimedSensingSegment,
    start_time_s: float,
    end_time_s: float,
    detection_radius_m: float,
    search_speed_mps: float,
) -> float:
    duration = end_time_s - start_time_s
    if duration <= 0.0:
        return inf

    target_start_time = trajectory.times_s[target_index]
    target_end_time = trajectory.times_s[target_index + 1]
    target_duration = target_end_time - target_start_time
    target_start = trajectory.points[target_index]
    target_end = trajectory.points[target_index + 1]
    target_vx = (target_end.x - target_start.x) / target_duration
    target_vy = (target_end.y - target_start.y) / target_duration
    target_elapsed = start_time_s - target_start_time
    target_x = target_start.x + target_vx * target_elapsed
    target_y = target_start.y + target_vy * target_elapsed

    uav_elapsed = start_time_s - sensing_segment.start_time_s
    uav_x = (
        sensing_segment.start.x
        + sensing_segment.unit_x * search_speed_mps * uav_elapsed
    )
    uav_y = (
        sensing_segment.start.y
        + sensing_segment.unit_y * search_speed_mps * uav_elapsed
    )
    relative_x = target_x - uav_x
    relative_y = target_y - uav_y
    relative_vx = target_vx - sensing_segment.unit_x * search_speed_mps
    relative_vy = target_vy - sensing_segment.unit_y * search_speed_mps

    entry_delta = _circle_entry_delta(
        relative_x,
        relative_y,
        relative_vx,
        relative_vy,
        detection_radius_m,
        duration,
    )
    return start_time_s + entry_delta if entry_delta < inf else inf


def _circle_entry_delta(
    relative_x: float,
    relative_y: float,
    relative_vx: float,
    relative_vy: float,
    radius_m: float,
    duration_s: float,
) -> float:
    constant = relative_x**2 + relative_y**2 - radius_m**2
    if constant <= 0.0:
        return 0.0

    quadratic = relative_vx**2 + relative_vy**2
    if quadratic <= 1e-15:
        return inf
    linear = 2.0 * (relative_x * relative_vx + relative_y * relative_vy)
    discriminant = linear**2 - 4.0 * quadratic * constant
    if discriminant < 0.0:
        return inf

    root = (-linear - sqrt(discriminant)) / (2.0 * quadratic)
    if 0.0 <= root <= duration_s:
        return root
    return inf
