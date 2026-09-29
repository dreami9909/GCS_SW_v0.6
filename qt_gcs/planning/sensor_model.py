from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .geometry import LocalFrame, LocalPoint, offset_point


@dataclass(frozen=True, slots=True)
class SearchCameraSpec:
    """Single source of truth for the simulated EO search camera.

    The 1,200 m value is the full centre-line ground reach from -45 to +45
    degrees at 600 m altitude; it is not an instantaneous detection radius.
    The 850 m/43.2 s values remain named operational cell metadata until their
    detailed scan-law meaning is supplied by the equipment specification.
    """

    altitude_m: float = 600.0
    horizontal_fov_deg: float = 18.0
    max_gimbal_angle_deg: float = 45.0
    gimbal_max_rate_dps: float = 120.0
    gimbal_scan_rate_dps: float = 60.0
    gimbal_settle_time_s: float = 0.10
    overlap_ratio: float = 0.20
    detection_probability: float = 1.0
    # Demo/wide-area option: when True, a subject counts as detected as soon as a
    # sensing vehicle comes within ``gimbal_centerline_reach_m`` of it, instead of
    # having to fall inside the small instantaneous scanning-gimbal footprint.
    # This makes detection of a fast-drifting subject robust (the narrow scanning
    # footprint otherwise slips past a 19 km/h target between discrete looks).
    detect_within_reach: bool = False
    # Optional cap (metres) on the ``detect_within_reach`` centre-line reach.  The
    # raw gimbal reach (altitude*tan(gimbal), ~1,177 m at 63deg) detects a subject
    # from clear across a search cell, so a lawnmower vehicle "sees" it long before
    # a sweep lane passes over it.  Capping the reach to roughly one sweep width
    # ties detection to an actual lane pass (matches the visible track spacing).
    # ``None`` keeps the full gimbal reach (unchanged behaviour for other missions).
    detect_reach_m: float | None = None
    eo_width_px: int = 1_920
    eo_height_px: int = 1_080
    eo_frame_rate_hz: float = 30.0
    ir_width_px: int = 640
    ir_height_px: int = 512
    ir_frame_rate_hz: float = 30.0
    effective_sweep_width_m: float = 400.0
    nominal_cell_m: float = 850.0
    nominal_cell_search_time_s: float = 43.2
    nominal_search_area_km2: float = 89.4
    nominal_search_time_min: float = 44.0

    def __post_init__(self) -> None:
        if self.altitude_m <= 0.0:
            raise ValueError("altitude_m must be positive")
        if not 0.0 < self.horizontal_fov_deg < 90.0:
            raise ValueError("horizontal_fov_deg must be in (0, 90)")
        if not 0.0 <= self.max_gimbal_angle_deg < 90.0:
            raise ValueError("max_gimbal_angle_deg must be in [0, 90)")
        if not 0.0 < self.gimbal_scan_rate_dps <= self.gimbal_max_rate_dps:
            raise ValueError("invalid gimbal scan rate")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        if not 0.0 < self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in (0, 1]")
        if self.effective_sweep_width_m <= 0.0:
            raise ValueError("effective_sweep_width_m must be positive")
        if self.detect_reach_m is not None and self.detect_reach_m <= 0.0:
            raise ValueError("detect_reach_m must be positive when set")

    @property
    def instantaneous_swath_m(self) -> float:
        return 2.0 * self.altitude_m * math.tan(
            math.radians(self.horizontal_fov_deg) / 2.0
        )

    @property
    def ideal_detection_radius_m(self) -> float:
        return self.effective_sweep_width_m / 2.0

    @property
    def track_spacing_m(self) -> float:
        return self.effective_sweep_width_m * (1.0 - self.overlap_ratio)

    @property
    def gimbal_centerline_envelope_m(self) -> float:
        return 2.0 * self.altitude_m * math.tan(
            math.radians(self.max_gimbal_angle_deg)
        )

    @property
    def gimbal_centerline_reach_m(self) -> float:
        """Maximum ground reach to either side of the vehicle centreline."""
        return self.gimbal_centerline_envelope_m / 2.0

    @property
    def effective_detect_reach_m(self) -> float:
        """The distance at which a ``detect_within_reach`` contact may register.

        Defaults to the full gimbal centre-line reach, but ``detect_reach_m``
        (when set) clamps it so detection only fires once a sweep lane is within
        ~one track spacing of the subject, matching the visible lawnmower.
        """
        reach = self.gimbal_centerline_reach_m
        if self.detect_reach_m is not None:
            reach = min(reach, float(self.detect_reach_m))
        return reach

    @property
    def scan_sample_interval_s(self) -> float:
        """Maximum time step that preserves the configured FOV overlap."""
        angular_step_deg = self.horizontal_fov_deg * (1.0 - self.overlap_ratio)
        return angular_step_deg / self.gimbal_scan_rate_dps

    @property
    def gimbal_scan_period_s(self) -> float:
        angular_travel_deg = 4.0 * self.max_gimbal_angle_deg
        return (
            angular_travel_deg / self.gimbal_scan_rate_dps
            + 2.0 * self.gimbal_settle_time_s
        )

    def scan_angle_deg(self, elapsed_s: float) -> float:
        """Return a deterministic -limit/+limit triangular scan angle."""
        travel_s = 2.0 * self.max_gimbal_angle_deg / self.gimbal_scan_rate_dps
        settle_s = self.gimbal_settle_time_s
        period_s = 2.0 * travel_s + 2.0 * settle_s
        phase = max(0.0, float(elapsed_s)) % period_s
        if phase < travel_s:
            return -self.max_gimbal_angle_deg + (
                2.0 * self.max_gimbal_angle_deg * phase / travel_s
            )
        phase -= travel_s
        if phase < settle_s:
            return self.max_gimbal_angle_deg
        phase -= settle_s
        if phase < travel_s:
            return self.max_gimbal_angle_deg - (
                2.0 * self.max_gimbal_angle_deg * phase / travel_s
            )
        return -self.max_gimbal_angle_deg

    def footprint_radius_m(
        self,
        altitude_m: float,
        gimbal_angle_deg: float,
    ) -> float:
        """Approximate the cross-track half-footprint on level ground."""
        altitude = max(1.0, float(altitude_m))
        half_fov = self.horizontal_fov_deg / 2.0
        minimum_angle = max(-89.0, gimbal_angle_deg - half_fov)
        maximum_angle = min(89.0, gimbal_angle_deg + half_fov)
        near_m = altitude * math.tan(math.radians(minimum_angle))
        far_m = altitude * math.tan(math.radians(maximum_angle))
        return max(1.0, abs(far_m - near_m) / 2.0)

    def display_dict(self) -> dict[str, float | int]:
        payload = asdict(self)
        payload.update(
            {
                "instantaneous_swath_m": self.instantaneous_swath_m,
                "ideal_detection_radius_m": self.ideal_detection_radius_m,
                "track_spacing_m": self.track_spacing_m,
                "gimbal_centerline_envelope_m": (
                    self.gimbal_centerline_envelope_m
                ),
                "gimbal_centerline_reach_m": self.gimbal_centerline_reach_m,
                "effective_detect_reach_m": self.effective_detect_reach_m,
                "gimbal_scan_period_s": self.gimbal_scan_period_s,
                "scan_sample_interval_s": self.scan_sample_interval_s,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class SensorFootprint:
    vehicle_id: int
    center: LocalPoint
    radius_m: float
    gimbal_angle_deg: float
    detection_probability: float

    def contains(self, point: LocalPoint) -> bool:
        return self.detection_probability_at(point) > 0.0

    def detection_probability_at(self, point: LocalPoint) -> float:
        """Return the edge-tapered probability for one instantaneous look.

        The declared probability is the optical-axis value.  A quadratic
        taper avoids treating a subject on the FOV boundary as equivalent to
        one at the centre and gives negative observations a stable likelihood
        instead of an all-or-nothing particle deletion.
        """
        radius = max(1.0, self.radius_m)
        delta_east = point.east_m - self.center.east_m
        delta_north = point.north_m - self.center.north_m
        normalized_distance_squared = (
            delta_east * delta_east + delta_north * delta_north
        ) / (radius * radius)
        if normalized_distance_squared >= 1.0:
            return 0.0
        edge_taper = 1.0 - normalized_distance_squared
        return max(
            0.0,
            min(1.0, self.detection_probability * edge_taper),
        )

    def render_dict(self, frame: LocalFrame) -> dict[str, float | int]:
        latitude, longitude = frame.to_geographic(self.center)
        return {
            "vehicle_id": self.vehicle_id,
            "latitude": latitude,
            "longitude": longitude,
            "radius_m": self.radius_m,
            "gimbal_angle_deg": self.gimbal_angle_deg,
            "detection_probability": self.detection_probability,
        }


def build_footprint(
    spec: SearchCameraSpec,
    frame: LocalFrame,
    *,
    vehicle_id: int,
    latitude: float,
    longitude: float,
    altitude_m: float,
    heading_deg: float,
    elapsed_s: float,
) -> SensorFootprint:
    vehicle_point = frame.to_local(latitude, longitude)
    return build_local_footprint(
        spec,
        vehicle_id=vehicle_id,
        vehicle_point=vehicle_point,
        altitude_m=altitude_m,
        heading_deg=heading_deg,
        elapsed_s=elapsed_s,
    )


def build_local_footprint(
    spec: SearchCameraSpec,
    *,
    vehicle_id: int,
    vehicle_point: LocalPoint,
    altitude_m: float,
    heading_deg: float,
    elapsed_s: float,
) -> SensorFootprint:
    """Build the same footprint directly in a mission-local frame."""
    gimbal_angle = spec.scan_angle_deg(elapsed_s)
    ground_offset_m = max(1.0, float(altitude_m)) * math.tan(
        math.radians(gimbal_angle)
    )
    # Positive scan angles look to the vehicle's right side.
    footprint_center = offset_point(
        vehicle_point,
        abs(ground_offset_m),
        heading_deg + (90.0 if ground_offset_m >= 0.0 else -90.0),
    )
    return SensorFootprint(
        vehicle_id=vehicle_id,
        center=footprint_center,
        radius_m=spec.footprint_radius_m(altitude_m, gimbal_angle),
        gimbal_angle_deg=gimbal_angle,
        detection_probability=spec.detection_probability,
    )
