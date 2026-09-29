"""기본 자료형 — 좌표, 센서 제원, 임무 설정, 경로.

이 파일에는 "정책"이 없다. 기하와 단위 변환만 있다.

핵심 수식
---------
* 순간 지상 폭 (``SensorSpec.instantaneous_swath_m``)

      swath = 2 * H * tan(HFOV / 2)

  H = 고도, HFOV = 수평 화각. EO/IR을 같은 18도로 맞췄으므로 두 채널의
  지상 발자국이 같다.

* 소인간격 (``SensorSpec.track_spacing_m``)

      s = W * (1 - overlap)

  W = 유효 탐색폭(아래), overlap = 인접 소인선 중첩률. Ch1의
  ``sarops_core``와 Ch4의 accordion 계획이 이 값을 쓴다.

* 유효 탐색폭 W (``SensorSpec.effective_sweep_width_m``)
  실제 계산은 ``search_envelope.CompositeSearchEnvelope.sweep_width_m``에 있다.
  여기서는 위임만 한다. **W != 2 * 지지반폭** 이라는 점이 중요하다.

* 횡방향 탐지확률 (``SensorSpec.lateral_detection_scale``)
  raised-cosine 곡선. 역시 ``search_envelope``로 위임.

* weave 적용 실제 비행거리 (``SensorSpec.actual_search_distance_m``)
  중심선 L을 날 때 실제로는 사인 곡선을 타므로 더 길다.
  ``search_envelope.WeavePattern.actual_distance_m``의 호 길이 적분.

의존
----
* 위(import): ``search_envelope`` 하나뿐. 그 외 표준 라이브러리.
* 아래(이 파일을 쓰는 곳): 사실상 전부. ``motion``, ``probability``,
  ``particle_filter``, ``sensor_observation``, ``terrain``, ``evaluation``,
  ``simulation``, 그리고 ``research`` 전 계층.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot, pi, radians, tan

from cpp_search.core.search_envelope import CompositeSearchEnvelope


@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        return hypot(other.x - self.x, other.y - self.y)


@dataclass(frozen=True, slots=True)
class SensorChannelSpec:
    name: str
    image_width_px: int
    image_height_px: int
    frame_rate_hz: float
    horizontal_fov_deg: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if self.horizontal_fov_deg is not None and not 0 < self.horizontal_fov_deg < 180:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")

    @property
    def horizontal_ifov_rad(self) -> float | None:
        if self.horizontal_fov_deg is None:
            return None
        return radians(self.horizontal_fov_deg) / self.image_width_px

    @property
    def horizontal_ifov_urad(self) -> float | None:
        if self.horizontal_ifov_rad is None:
            return None
        return self.horizontal_ifov_rad * 1_000_000.0


@dataclass(frozen=True, slots=True)
class SensorSpec:
    altitude_m: float = 600.0
    eo: SensorChannelSpec = field(
        default_factory=lambda: SensorChannelSpec(
            name="EO",
            image_width_px=1_920,
            image_height_px=1_080,
            frame_rate_hz=30.0,
            horizontal_fov_deg=18.0,
        )
    )
    ir: SensorChannelSpec = field(
        default_factory=lambda: SensorChannelSpec(
            name="IR",
            image_width_px=640,
            image_height_px=512,
            frame_rate_hz=30.0,
            horizontal_fov_deg=None,
        )
    )
    max_gimbal_angle_deg: float = 45.0
    gimbal_max_rate_dps: float = 120.0
    gimbal_scan_rate_dps: float = 60.0
    gimbal_settle_time_s: float = 0.10
    overlap_ratio: float = 0.20
    independent_look_interval_s: float | None = None
    search_envelope: CompositeSearchEnvelope | None = None
    ground_scan_radius_m: float | None = None

    def __post_init__(self) -> None:
        if self.altitude_m <= 0:
            raise ValueError("altitude_m must be positive")
        if not 0 <= self.max_gimbal_angle_deg < 90:
            raise ValueError("max_gimbal_angle_deg must be in [0, 90)")
        if self.gimbal_max_rate_dps <= 0:
            raise ValueError("gimbal_max_rate_dps must be positive")
        if not 0 < self.gimbal_scan_rate_dps <= self.gimbal_max_rate_dps:
            raise ValueError(
                "gimbal_scan_rate_dps must be positive and no greater than max rate"
            )
        if self.gimbal_settle_time_s < 0:
            raise ValueError("gimbal_settle_time_s must not be negative")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be in [0, 1)")
        if (
            self.independent_look_interval_s is not None
            and self.independent_look_interval_s <= 0.0
        ):
            raise ValueError("independent_look_interval_s must be positive")
        if self.ground_scan_radius_m is not None and self.ground_scan_radius_m <= 0.0:
            raise ValueError("ground_scan_radius_m must be positive")

    @classmethod
    def sr_z50(
        cls,
        *,
        eo_full_hd: bool = False,
        search_envelope: CompositeSearchEnvelope | None = None,
    ) -> "SensorSpec":
        """Return the catalog-constrained SR-Z50 fused-search configuration.

        The EO channel is operated at an 18 degree horizontal FOV so that its
        footprint matches the thermal channel. The catalog lists EO as a
        continuous 60-to-3 degree optical zoom range, so 18 degrees is an
        operational search-mode choice rather than a separately specified stop.
        """
        eo_width, eo_height = (1_920, 1_080) if eo_full_hd else (1_280, 720)
        return cls(
            altitude_m=600.0,
            eo=SensorChannelSpec(
                name="EO",
                image_width_px=eo_width,
                image_height_px=eo_height,
                frame_rate_hz=30.0,
                horizontal_fov_deg=18.0,
            ),
            ir=SensorChannelSpec(
                name="IR",
                image_width_px=1_280,
                image_height_px=720,
                frame_rate_hz=30.0,
                horizontal_fov_deg=18.0,
            ),
            max_gimbal_angle_deg=45.0,
            gimbal_max_rate_dps=60.0,
            gimbal_scan_rate_dps=30.0,
            gimbal_settle_time_s=0.10,
            overlap_ratio=0.20,
            independent_look_interval_s=3.2,
            search_envelope=(
                search_envelope or CompositeSearchEnvelope.operational_400m()
            ),
        )

    @property
    def fov_deg(self) -> float:
        if self.eo.horizontal_fov_deg is None:
            raise ValueError("EO horizontal FOV is required for coverage geometry")
        return self.eo.horizontal_fov_deg

    @property
    def instantaneous_swath_m(self) -> float:
        # 순간 지상 폭: swath = 2 * H * tan(HFOV / 2)
        """Nadir ground width visible in one frame."""
        return 2.0 * self.altitude_m * tan(radians(self.fov_deg) / 2.0)

    @property
    def gimbal_envelope_m(self) -> float:
        """Full ground width mechanically reachable by a +/- gimbal angle."""
        return 2.0 * self.altitude_m * tan(radians(self.max_gimbal_angle_deg))

    @property
    def track_spacing_m(self) -> float:
        """Default adjacent search-track spacing after overlap is applied."""
        # s = W * (1 - overlap).  W는 지지폭이 아니라 적분값이라는 점에 주의.
        return self.effective_sweep_width_m * (1.0 - self.overlap_ratio)

    @property
    def has_extended_search_envelope(self) -> bool:
        return self.search_envelope is not None or self.ground_scan_radius_m is not None

    @property
    def coverage_half_width_m(self) -> float:
        """Cross-track support used for unique-area coverage accounting."""

        if self.search_envelope is not None:
            return self.search_envelope.support_half_width_m
        if self.ground_scan_radius_m is not None:
            return self.ground_scan_radius_m
        return self.instantaneous_swath_m / 2.0

    @property
    def effective_sweep_width_m(self) -> float:
        """Equivalent sweep width, distinct from twice the support radius."""

        if self.search_envelope is not None:
            return self.search_envelope.sweep_width_m()
        if self.ground_scan_radius_m is not None:
            return 2.0 * self.ground_scan_radius_m
        return self.instantaneous_swath_m

    @property
    def gimbal_scan_period_s(self) -> float:
        """Time for a -limit -> +limit -> -limit scan, including two settles."""
        angular_travel_deg = 4.0 * self.max_gimbal_angle_deg
        return (
            angular_travel_deg / self.gimbal_scan_rate_dps
            + 2.0 * self.gimbal_settle_time_s
        )

    def effective_ground_scan_radius_m(self, horizontal_fov_deg: float) -> float:
        """Return system-level cross-track support for detection calculations."""
        instantaneous_radius = self.altitude_m * tan(
            radians(horizontal_fov_deg) / 2.0
        )
        if self.search_envelope is not None:
            return self.search_envelope.support_half_width_m
        if self.ground_scan_radius_m is None:
            return instantaneous_radius
        maximum_edge_angle_deg = min(
            89.0,
            self.max_gimbal_angle_deg + horizontal_fov_deg / 2.0,
        )
        mechanical_radius = self.altitude_m * tan(radians(maximum_edge_angle_deg))
        return min(self.ground_scan_radius_m, mechanical_radius)

    def seeker_ground_scan_radius_m(self, horizontal_fov_deg: float) -> float:
        """Return search-camera-only reach used to calculate mechanical revisit time."""

        if self.search_envelope is not None:
            requested_radius = self.search_envelope.seeker_half_width_m
        elif self.ground_scan_radius_m is not None:
            requested_radius = self.ground_scan_radius_m
        else:
            return self.altitude_m * tan(radians(horizontal_fov_deg) / 2.0)
        maximum_edge_angle_deg = min(
            89.0,
            self.max_gimbal_angle_deg + horizontal_fov_deg / 2.0,
        )
        mechanical_radius = self.altitude_m * tan(radians(maximum_edge_angle_deg))
        return min(requested_radius, mechanical_radius)

    def gimbal_scan_cycle_s(self, horizontal_fov_deg: float) -> float:
        """Return the full revisit cycle for one FOV over the scan radius."""
        if not self.has_extended_search_envelope:
            return self.gimbal_scan_period_s
        radius_m = self.seeker_ground_scan_radius_m(horizontal_fov_deg)
        edge_angle_deg = degrees(atan2(radius_m, self.altitude_m))
        boresight_excursion_deg = min(
            self.max_gimbal_angle_deg,
            max(0.0, edge_angle_deg - horizontal_fov_deg / 2.0),
        )
        return (
            4.0 * boresight_excursion_deg / self.gimbal_scan_rate_dps
            + 2.0 * self.gimbal_settle_time_s
        )

    def effective_observation_interval_s(self, horizontal_fov_deg: float) -> float:
        """Combine detector correlation and mechanical target revisit time."""
        if not self.has_extended_search_envelope:
            return self.reference_observation_interval_s
        return max(
            self.reference_observation_interval_s,
            self.gimbal_scan_cycle_s(horizontal_fov_deg),
        )

    @property
    def reference_observation_interval_s(self) -> float:
        """Effective interval between independent detector opportunities."""
        if self.independent_look_interval_s is not None:
            return self.independent_look_interval_s
        return self.gimbal_scan_period_s

    def along_track_distance_per_scan_m(self, speed_mps: float) -> float:
        if speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        return speed_mps * self.gimbal_scan_period_s

    def actual_search_distance_m(self, centerline_distance_m: float) -> float:
        """Return flown distance after applying the vehicle weave pattern."""

        if self.search_envelope is None:
            return centerline_distance_m
        return self.search_envelope.actual_distance_m(centerline_distance_m)

    def centerline_search_speed_mps(self, actual_speed_mps: float) -> float:
        """Return centerline progress speed at the configured actual airspeed."""

        if self.search_envelope is None:
            return actual_speed_mps
        return self.search_envelope.centerline_progress_speed_mps(actual_speed_mps)

    def lateral_detection_scale(self, cross_track_offset_m: float) -> float:
        if self.search_envelope is None:
            return 1.0 if abs(cross_track_offset_m) <= self.coverage_half_width_m else 0.0
        return self.search_envelope.lateral_detection_probability(cross_track_offset_m)


@dataclass(frozen=True, slots=True)
class MissionConfig:
    center: Point2D = Point2D(0.0, 0.0)
    search_radius_m: float = 40_000.0 / 3_600.0 * 5.0 * 60.0
    uav_count: int = 6
    max_speed_mps: float = 160_000.0 / 3_600.0
    transit_speed_mps: float = 160_000.0 / 3_600.0
    search_speed_mps: float = 100_000.0 / 3_600.0
    subject_max_speed_mps: float = 40_000.0 / 3_600.0
    lead_time_s: float = 5.0 * 60.0
    sector_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.search_radius_m <= 0:
            raise ValueError("search_radius_m must be positive")
        if self.uav_count <= 0:
            raise ValueError("uav_count must be positive")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if not 0 < self.transit_speed_mps <= self.max_speed_mps:
            raise ValueError("transit_speed_mps must be in (0, max_speed_mps]")
        if not 0 < self.search_speed_mps <= self.max_speed_mps:
            raise ValueError("search_speed_mps must be in (0, max_speed_mps]")
        if self.subject_max_speed_mps <= 0:
            raise ValueError("subject_max_speed_mps must be positive")
        if self.lead_time_s <= 0:
            raise ValueError("lead_time_s must be positive")

    @property
    def speed_mps(self) -> float:
        """Compatibility alias: coverage planners use the search speed."""
        return self.search_speed_mps

    @property
    def reachable_radius_at_search_start_m(self) -> float:
        return self.subject_max_speed_mps * self.lead_time_s

    @property
    def sector_angle_rad(self) -> float:
        return 2.0 * pi / self.uav_count

    @property
    def total_area_m2(self) -> float:
        return pi * self.search_radius_m**2

    @property
    def sector_area_m2(self) -> float:
        return self.total_area_m2 / self.uav_count


@dataclass(frozen=True, slots=True)
class PathSegment:
    start: Point2D
    end: Point2D
    sensor_on: bool
    detection_scale: float | None = None
    sensor_mode: str | None = None
    speed_mps: float | None = None
    search_pattern: bool | None = None

    def __post_init__(self) -> None:
        if self.speed_mps is not None and self.speed_mps <= 0.0:
            raise ValueError("segment speed_mps must be positive")

    @property
    def length_m(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def effective_detection_scale(self) -> float:
        if self.detection_scale is not None:
            return min(max(self.detection_scale, 0.0), 1.0)
        return 1.0 if self.sensor_on else 0.0

    @property
    def uses_search_pattern(self) -> bool:
        """Whether flown distance includes the configured search weave."""

        return self.sensor_on if self.search_pattern is None else self.search_pattern

    def actual_speed_mps(self, mission: MissionConfig) -> float:
        if self.speed_mps is not None:
            return self.speed_mps
        return (
            mission.search_speed_mps
            if self.sensor_on
            else mission.transit_speed_mps
        )

    def flown_distance_m(self, sensor: SensorSpec) -> float:
        if self.uses_search_pattern:
            return sensor.actual_search_distance_m(self.length_m)
        return self.length_m

    def duration_s(self, mission: MissionConfig, sensor: SensorSpec) -> float:
        return self.flown_distance_m(sensor) / self.actual_speed_mps(mission)

    def centerline_speed_mps(
        self,
        mission: MissionConfig,
        sensor: SensorSpec,
    ) -> float:
        if self.length_m <= 0.0:
            return self.actual_speed_mps(mission)
        return self.length_m / self.duration_s(mission, sensor)


@dataclass(frozen=True, slots=True)
class Route:
    planner_name: str
    vehicle_id: int
    segments: tuple[PathSegment, ...]

    @property
    def total_distance_m(self) -> float:
        return sum(segment.length_m for segment in self.segments)

    @property
    def sensing_distance_m(self) -> float:
        return sum(
            segment.length_m for segment in self.segments if segment.sensor_on
        )
