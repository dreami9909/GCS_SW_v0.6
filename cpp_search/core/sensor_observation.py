"""EO/IR 탐지모델 — 표적 신호, 채널 성능, 공간 탐지확률.

카탈로그가 탐지확률 곡선을 공개하지 않으므로, 아래 계수들은 **A급 모델링
가정**이지 제조사 성능 주장이 아니다.

핵심 수식
---------
* 지상 표본거리와 표적 픽셀 수 (``_channel_probability``)

      GSD = 2 * R * tan(HFOV/2) / W_px
      n_px = 표적폭 / GSD

* 픽셀 항 (Johnson 기준의 로지스틱 근사)

      P_px = sigma( (n_px - n50) / dn )

* SNR 항 (거리 제곱 감쇠 + 대비)

      SNR = SNR_ref * (R_ref / R)^2 * contrast
      P_snr = sigma( (SNR - SNR50) / dSNR )

* 거리 롤오프 항

      P_R = sigma( (R_nom - R) / (rho * R_nom) )

* 채널 확률

      P_channel = eta * P_px * P_snr * P_R          (eta = 알고리즘 효율)

* EO/IR 융합 (``clear_sensor_probability``)

      P_fused = 1 - (1 - P_eo) * (1 - P_ir)

  독립 가정. 여기에 종횡비(aspect) 항과 위장(camouflage) 계수를 곱한다.

* 최종 탐지확률 (``detection_probability``)

      PD = scale * P_sensor * P_surface * P_LOS

  P_surface = 지표 피복 가림, P_LOS = DEM 가시선.

* 위험률과 노출 (``hazard_rate`` / ``probability_for_exposure``)

      lambda = -ln(1 - PD) / t_scan
      P(노출 t) = 1 - exp(-lambda * t)

  한 번의 주사에서 PD를 얻으므로, 노출시간이 길면 지수적으로 누적된다.
  ``simulation``의 탐지시간 계산과 ``particle_filter``의 음성 갱신이
  **같은 위험률**을 쓴다. 계획과 평가가 어긋나지 않게 하기 위해서다.

의존
----
* 위: ``models``. 지형은 주입.
* 아래: ``simulation``(탐지시간), ``particle_filter``(음성 갱신),
  ``chapters/chapter0``(표적·채널별 W 테이블).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, exp, hypot, log1p, pi, radians, sin, sqrt, tan
from typing import Literal

from cpp_search.core.models import PathSegment, Point2D, SensorSpec


ObservationCondition = Literal[
    "ideal",
    "worldcover",
    "los",
    "worldcover_los",
    "sensor",
    "sensor_worldcover",
    "sensor_worldcover_los",
]


def _logistic(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class TargetSignatureSpec:
    """Nominal geometric and channel signature of one ground-target class."""

    name: str
    length_m: float
    width_m: float
    height_m: float
    eo_contrast: float
    ir_contrast: float
    head_on_multiplier: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("target signature name must not be empty")
        if min(self.length_m, self.width_m, self.height_m) <= 0.0:
            raise ValueError("target dimensions must be positive")
        if min(self.eo_contrast, self.ir_contrast) <= 0.0:
            raise ValueError("target contrasts must be positive")
        if not 0.0 < self.head_on_multiplier <= 1.0:
            raise ValueError("head_on_multiplier must be in (0, 1]")


MISSING_PERSON_ADULT_NOMINAL = TargetSignatureSpec(
    "Missing person (adult) nominal",
    length_m=0.6,
    width_m=0.5,
    height_m=1.7,
    eo_contrast=0.90,
    ir_contrast=1.00,
    head_on_multiplier=0.90,
)

MISSING_PERSON_CHILD_NOMINAL = TargetSignatureSpec(
    "Missing person (child) nominal",
    length_m=0.4,
    width_m=0.4,
    height_m=1.2,
    eo_contrast=0.85,
    ir_contrast=0.85,
    head_on_multiplier=0.88,
)


@dataclass(frozen=True, slots=True)
class CamouflageSpec:
    """Uncalibrated target-signature attenuation applied by channel."""

    name: str
    eo_contrast_multiplier: float
    ir_contrast_multiplier: float
    algorithm_probability_multiplier: float

    def __post_init__(self) -> None:
        values = (
            self.eo_contrast_multiplier,
            self.ir_contrast_multiplier,
            self.algorithm_probability_multiplier,
        )
        if any(not 0.0 < value <= 1.0 for value in values):
            raise ValueError("camouflage multipliers must be in (0, 1]")


CAMOUFLAGE_NONE = CamouflageSpec("None", 1.00, 1.00, 1.00)
CAMOUFLAGE_NOMINAL = CamouflageSpec("Nominal", 0.75, 0.85, 0.92)
CAMOUFLAGE_HEAVY = CamouflageSpec("Heavy", 0.45, 0.65, 0.75)


@dataclass(frozen=True, slots=True)
class ChannelDetectionConfig:
    """Nominal channel range, pixel, SNR, and algorithm assumptions."""

    name: str
    nominal_detection_range_m: float
    horizontal_fov_deg: float
    pixels_50: float
    pixel_transition_width: float
    snr_reference_range_m: float
    snr_at_reference: float
    snr_50: float
    snr_transition_width: float
    range_rolloff_ratio: float
    algorithm_efficiency: float

    def __post_init__(self) -> None:
        positive = (
            self.nominal_detection_range_m,
            self.horizontal_fov_deg,
            self.pixels_50,
            self.pixel_transition_width,
            self.snr_reference_range_m,
            self.snr_at_reference,
            self.snr_50,
            self.snr_transition_width,
            self.range_rolloff_ratio,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("channel detection parameters must be positive")
        if not 0.0 < self.horizontal_fov_deg < 180.0:
            raise ValueError("horizontal_fov_deg must be in (0, 180)")
        if not 0.0 < self.algorithm_efficiency <= 1.0:
            raise ValueError("algorithm_efficiency must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SensorPerformanceConfig:
    """A-grade EO/IR performance assumptions; not an equipment data sheet."""

    eo: ChannelDetectionConfig = ChannelDetectionConfig(
        name="EO",
        nominal_detection_range_m=5_000.0,
        horizontal_fov_deg=18.0,
        pixels_50=8.0,
        pixel_transition_width=2.0,
        snr_reference_range_m=1_000.0,
        snr_at_reference=10.0,
        snr_50=6.0,
        snr_transition_width=1.5,
        range_rolloff_ratio=0.12,
        algorithm_efficiency=0.90,
    )
    ir: ChannelDetectionConfig = ChannelDetectionConfig(
        name="IR",
        nominal_detection_range_m=3_500.0,
        horizontal_fov_deg=24.0,
        pixels_50=6.0,
        pixel_transition_width=1.5,
        snr_reference_range_m=1_000.0,
        snr_at_reference=9.0,
        snr_50=5.0,
        snr_transition_width=1.3,
        range_rolloff_ratio=0.12,
        algorithm_efficiency=0.86,
    )


SENSOR_PERFORMANCE_NOMINAL = SensorPerformanceConfig()


# SR-Z50 catalog geometry with the prior A-grade range/SNR/detector coefficients.
# The catalog does not publish EO/IR target-detection probability or DRI curves,
# so those coefficients remain sensitivity assumptions rather than manufacturer
# performance claims.
SENSOR_PERFORMANCE_SR_Z50 = SensorPerformanceConfig(
    eo=ChannelDetectionConfig(
        name="EO",
        nominal_detection_range_m=5_000.0,
        horizontal_fov_deg=18.0,
        pixels_50=8.0,
        pixel_transition_width=2.0,
        snr_reference_range_m=1_000.0,
        snr_at_reference=10.0,
        snr_50=6.0,
        snr_transition_width=1.5,
        range_rolloff_ratio=0.12,
        algorithm_efficiency=0.90,
    ),
    ir=ChannelDetectionConfig(
        name="IR",
        nominal_detection_range_m=3_500.0,
        horizontal_fov_deg=18.0,
        pixels_50=6.0,
        pixel_transition_width=1.5,
        snr_reference_range_m=1_000.0,
        snr_at_reference=9.0,
        snr_50=5.0,
        snr_transition_width=1.3,
        range_rolloff_ratio=0.12,
        algorithm_efficiency=0.86,
    ),
)


@dataclass(frozen=True, slots=True)
class SensorSearchMode:
    """One SR-Z50 search action with channel-specific footprint geometry."""

    name: str
    channel_mode: Literal["eo", "ir", "fused"]
    footprint_hfov_deg: float
    eo_hfov_deg: float | None
    ir_hfov_deg: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sensor search-mode name must not be empty")
        if self.channel_mode not in {"eo", "ir", "fused"}:
            raise ValueError("unsupported sensor search-mode channel")
        angles = (
            self.footprint_hfov_deg,
            *(
                value
                for value in (self.eo_hfov_deg, self.ir_hfov_deg)
                if value is not None
            ),
        )
        if any(not 0.0 < value < 180.0 for value in angles):
            raise ValueError("sensor search-mode FOVs must be in (0, 180)")
        if self.channel_mode == "eo" and self.eo_hfov_deg is None:
            raise ValueError("EO mode requires eo_hfov_deg")
        if self.channel_mode == "ir" and self.ir_hfov_deg is None:
            raise ValueError("IR mode requires ir_hfov_deg")
        if self.channel_mode == "fused" and (
            self.eo_hfov_deg is None or self.ir_hfov_deg is None
        ):
            raise ValueError("fused mode requires EO and IR FOVs")


SR_Z50_EO_WIDE = SensorSearchMode("eo_wide", "eo", 60.0, 60.0, None)
SR_Z50_FUSED_18 = SensorSearchMode("fused_18", "fused", 18.0, 18.0, 18.0)
SR_Z50_EO_TELE = SensorSearchMode("eo_tele", "eo", 3.0, 3.0, None)
SR_Z50_SEARCH_MODES = {
    mode.name: mode
    for mode in (SR_Z50_EO_WIDE, SR_Z50_FUSED_18, SR_Z50_EO_TELE)
}


@dataclass(frozen=True, slots=True)
class LandCoverOcclusionProfile:
    """WorldCover class-to-occlusion conversion for relative attenuation."""

    name: str
    class_occlusion_probabilities: tuple[tuple[int, float], ...]
    building_occlusion_probability: float
    unknown_occlusion_probability: float
    occluded_detection_ratio: float

    def __post_init__(self) -> None:
        values = [value for _, value in self.class_occlusion_probabilities]
        values.extend(
            (
                self.building_occlusion_probability,
                self.unknown_occlusion_probability,
                self.occluded_detection_ratio,
            )
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("occlusion probabilities must be in [0, 1]")

    def visibility_multiplier(
        self,
        class_code: int | None,
        is_building: bool,
    ) -> float:
        lookup = dict(self.class_occlusion_probabilities)
        occlusion = lookup.get(class_code, self.unknown_occlusion_probability)
        if is_building:
            occlusion = max(occlusion, self.building_occlusion_probability)
        return 1.0 - occlusion * (1.0 - self.occluded_detection_ratio)


WORLDCOVER_OCCLUSION_NOMINAL = LandCoverOcclusionProfile(
    "nominal",
    (
        (10, 0.45),   # tree cover
        (20, 0.25),   # shrubland
        (30, 0.05),   # grassland
        (40, 0.12),   # cropland
        (50, 0.35),   # built-up
        (60, 0.02),   # bare / sparse vegetation
        (70, 0.03),   # snow and ice
        (80, 0.00),   # permanent water
        (90, 0.18),   # herbaceous wetland
        (95, 0.50),   # mangroves
        (100, 0.20),  # moss and lichen
    ),
    building_occlusion_probability=0.50,
    unknown_occlusion_probability=0.20,
    occluded_detection_ratio=0.35,
)


@dataclass(frozen=True, slots=True)
class LandCoverPDProfile:
    """WorldCover class-to-detection conversion for one reference scan."""

    name: str
    class_probabilities: tuple[tuple[int, float], ...]
    building_probability: float
    unknown_probability: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        codes = [code for code, _ in self.class_probabilities]
        if len(codes) != len(set(codes)):
            raise ValueError("WorldCover class codes must be unique")
        values = [value for _, value in self.class_probabilities]
        values.extend((self.building_probability, self.unknown_probability))
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("detection probabilities must be in [0, 1]")

    def probability(self, class_code: int | None, is_building: bool) -> float:
        lookup = dict(self.class_probabilities)
        probability = lookup.get(class_code, self.unknown_probability)
        if is_building:
            probability = min(probability, self.building_probability)
        return probability


# These ranges deliberately bracket the uncalibrated mapping.  They are not
# claimed sensor performance specifications.
WORLDCOVER_PD_LOW = LandCoverPDProfile(
    "low",
    (
        (10, 0.15),   # tree cover
        (20, 0.38),   # shrubland
        (30, 0.78),   # grassland
        (40, 0.62),   # cropland
        (50, 0.22),   # built-up
        (60, 0.88),   # bare / sparse vegetation
        (70, 0.85),   # snow and ice
        (80, 0.95),   # permanent water
        (90, 0.45),   # herbaceous wetland
        (95, 0.18),   # mangroves
        (100, 0.40),  # moss and lichen
    ),
    building_probability=0.20,
    unknown_probability=0.50,
)

WORLDCOVER_PD_NOMINAL = LandCoverPDProfile(
    "nominal",
    (
        (10, 0.28),
        (20, 0.55),
        (30, 0.92),
        (40, 0.78),
        (50, 0.42),
        (60, 0.98),
        (70, 0.95),
        (80, 0.98),
        (90, 0.62),
        (95, 0.30),
        (100, 0.58),
    ),
    building_probability=0.35,
    unknown_probability=0.70,
)

WORLDCOVER_PD_HIGH = LandCoverPDProfile(
    "high",
    (
        (10, 0.45),
        (20, 0.70),
        (30, 0.98),
        (40, 0.90),
        (50, 0.62),
        (60, 1.00),
        (70, 0.99),
        (80, 1.00),
        (90, 0.78),
        (95, 0.48),
        (100, 0.72),
    ),
    building_probability=0.55,
    unknown_probability=0.85,
)

WORLDCOVER_PD_PROFILES = {
    profile.name: profile
    for profile in (
        WORLDCOVER_PD_LOW,
        WORLDCOVER_PD_NOMINAL,
        WORLDCOVER_PD_HIGH,
    )
}


@dataclass(frozen=True, slots=True)
class DEMLineOfSightConfig:
    """Geometry assumptions for a terrain-only line-of-sight test."""

    target_height_m: float = 3.0
    terrain_clearance_m: float = 0.0
    sample_spacing_m: float = 30.0
    unavailable_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.target_height_m < 0.0:
            raise ValueError("target_height_m must be non-negative")
        if self.terrain_clearance_m < 0.0:
            raise ValueError("terrain_clearance_m must be non-negative")
        if self.sample_spacing_m <= 0.0:
            raise ValueError("sample_spacing_m must be positive")
        if not 0.0 <= self.unavailable_probability <= 1.0:
            raise ValueError("unavailable_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SpatialDetectionModel:
    """Matched truth/planner detection model for Figures 2-6 and 2-7.

    ``sensor.altitude_m`` is interpreted as constant height above the DEM at
    the UAV ground-track position.  DEM LOS therefore uses a constant-AGL
    platform and a target located ``target_height_m`` above its local terrain.
    """

    condition: ObservationCondition = "ideal"
    terrain: object | None = None
    worldcover_profile: LandCoverPDProfile = WORLDCOVER_PD_NOMINAL
    los_config: DEMLineOfSightConfig = DEMLineOfSightConfig()
    reference_exposure_s: float | None = None
    base_detection_probability: float = 1.0
    sensor_performance: SensorPerformanceConfig = SENSOR_PERFORMANCE_NOMINAL
    target_signature: TargetSignatureSpec = MISSING_PERSON_ADULT_NOMINAL
    camouflage: CamouflageSpec = CAMOUFLAGE_NOMINAL
    worldcover_occlusion: LandCoverOcclusionProfile = (
        WORLDCOVER_OCCLUSION_NOMINAL
    )
    channel_mode: Literal["eo", "ir", "fused"] = "fused"

    def __post_init__(self) -> None:
        if self.condition not in {
            "ideal",
            "worldcover",
            "los",
            "worldcover_los",
            "sensor",
            "sensor_worldcover",
            "sensor_worldcover_los",
        }:
            raise ValueError(f"unsupported observation condition: {self.condition}")
        if (self.uses_worldcover or self.uses_los) and self.terrain is None:
            raise ValueError("spatial observation conditions require terrain")
        if self.reference_exposure_s is not None and self.reference_exposure_s <= 0.0:
            raise ValueError("reference_exposure_s must be positive")
        if not 0.0 < self.base_detection_probability <= 1.0:
            raise ValueError("base_detection_probability must be in (0, 1]")
        if self.channel_mode not in {"eo", "ir", "fused"}:
            raise ValueError("channel_mode must be eo, ir, or fused")

    @property
    def uses_worldcover(self) -> bool:
        return self.condition in {
            "worldcover",
            "worldcover_los",
            "sensor_worldcover",
            "sensor_worldcover_los",
        }

    @property
    def uses_los(self) -> bool:
        return self.condition in {
            "los",
            "worldcover_los",
            "sensor_worldcover_los",
        }

    @property
    def uses_sensor_performance(self) -> bool:
        return self.condition in {
            "sensor",
            "sensor_worldcover",
            "sensor_worldcover_los",
        }

    @property
    def label(self) -> str:
        if self.condition == "worldcover":
            return f"WorldCover ({self.worldcover_profile.name})"
        if self.condition == "worldcover_los":
            return f"WorldCover+LOS ({self.worldcover_profile.name})"
        if self.condition == "sensor":
            return f"EO/IR nominal ({self.target_signature.name})"
        if self.condition == "sensor_worldcover":
            return f"EO/IR+WorldCover ({self.target_signature.name})"
        if self.condition == "sensor_worldcover_los":
            return f"EO/IR+WorldCover+LOS ({self.target_signature.name})"
        return {"ideal": "Ideal", "los": "DEM LOS"}[self.condition]

    def reference_scan_s(
        self,
        sensor: SensorSpec,
        sensor_mode: str | None = None,
    ) -> float:
        if self.reference_exposure_s is not None:
            return self.reference_exposure_s
        mode = self.search_mode(sensor_mode)
        horizontal_fov_deg = (
            mode.footprint_hfov_deg if mode is not None else sensor.fov_deg
        )
        return sensor.effective_observation_interval_s(horizontal_fov_deg)

    @staticmethod
    def search_mode(sensor_mode: str | None) -> SensorSearchMode | None:
        if sensor_mode is None:
            return None
        try:
            return SR_Z50_SEARCH_MODES[sensor_mode]
        except KeyError as exc:
            raise ValueError(f"unsupported sensor mode: {sensor_mode}") from exc

    def footprint_half_width_m(
        self,
        sensor: SensorSpec,
        sensor_mode: str | None = None,
    ) -> float:
        mode = self.search_mode(sensor_mode)
        horizontal_fov_deg = (
            mode.footprint_hfov_deg if mode is not None else sensor.fov_deg
        )
        return sensor.effective_ground_scan_radius_m(horizontal_fov_deg)

    def instantaneous_footprint_half_width_m(
        self,
        sensor: SensorSpec,
        sensor_mode: str | None = None,
    ) -> float:
        mode = self.search_mode(sensor_mode)
        horizontal_fov_deg = (
            mode.footprint_hfov_deg if mode is not None else sensor.fov_deg
        )
        return sensor.altitude_m * tan(radians(horizontal_fov_deg) / 2.0)

    def surface_probability(self, target: Point2D) -> float:
        if not self.uses_worldcover:
            return 1.0
        terrain = self.terrain
        class_code = (
            terrain.landcover_code_at(target.x, target.y)
            if hasattr(terrain, "landcover_code_at")
            else None
        )
        is_building = (
            terrain.is_building(target.x, target.y)
            if hasattr(terrain, "is_building")
            else False
        )
        if self.uses_sensor_performance:
            return self.worldcover_occlusion.visibility_multiplier(
                class_code,
                is_building,
            )
        return self.worldcover_profile.probability(class_code, is_building)

    def _aspect_terms(
        self,
        target: Point2D,
        observer: Point2D | None,
        target_heading_rad: float | None,
    ) -> tuple[float, float]:
        if observer is None or target_heading_rad is None:
            side_fraction = 2.0 / pi
        else:
            viewing_azimuth = atan2(
                observer.y - target.y,
                observer.x - target.x,
            )
            side_fraction = abs(sin(target_heading_rad - viewing_azimuth))
        projected_width_m = (
            self.target_signature.width_m
            + (self.target_signature.length_m - self.target_signature.width_m)
            * side_fraction
        )
        aspect_probability = (
            self.target_signature.head_on_multiplier
            + (1.0 - self.target_signature.head_on_multiplier) * side_fraction
        )
        return projected_width_m, aspect_probability

    def _channel_probability(
        self,
        config: ChannelDetectionConfig,
        image_width_px: int,
        horizontal_fov_deg: float,
        contrast: float,
        target_width_m: float,
        sensor: SensorSpec,
        range_m: float,
    ) -> float:
        # 지상 표본거리 GSD = 2 R tan(HFOV/2) / W_px,
        # 표적을 가로지르는 픽셀 수 n_px = 표적폭 / GSD.
        ground_width_m = 2.0 * range_m * tan(
            radians(horizontal_fov_deg) / 2.0
        )
        ground_sample_distance_m = ground_width_m / image_width_px
        target_pixels = target_width_m / ground_sample_distance_m
        pixel_probability = _logistic(
            (target_pixels - config.pixels_50) / config.pixel_transition_width
        )
        # SNR은 거리 제곱에 반비례(수신 전력 ~ 1/R^2)하고 대비에 비례한다.
        #   SNR = SNR_ref * (R_ref / R)^2 * contrast
        snr = (
            config.snr_at_reference
            * (config.snr_reference_range_m / range_m) ** 2
            * contrast
        )
        snr_probability = _logistic(
            (snr - config.snr_50) / config.snr_transition_width
        )
        range_probability = _logistic(
            (config.nominal_detection_range_m - range_m)
            / (config.range_rolloff_ratio * config.nominal_detection_range_m)
        )
        return min(
            1.0,
            config.algorithm_efficiency
            * pixel_probability
            * snr_probability
            * range_probability,
        )

    def channel_detection_probabilities(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
        target_heading_rad: float | None = None,
        sensor_mode: str | None = None,
    ) -> tuple[float, float]:
        """Return nominal clear-view EO and IR probabilities for one scan."""
        target_width_m, _ = self._aspect_terms(
            target,
            observer,
            target_heading_rad,
        )
        mode = self.search_mode(sensor_mode)
        range_m = (
            hypot(sensor.altitude_m, target.distance_to(observer))
            if mode is not None and observer is not None
            else sensor.altitude_m
        )
        eo_enabled = mode is None or mode.channel_mode in {"eo", "fused"}
        ir_enabled = mode is None or mode.channel_mode in {"ir", "fused"}
        eo_probability = (
            self._channel_probability(
                self.sensor_performance.eo,
                sensor.eo.image_width_px,
                (
                    mode.eo_hfov_deg
                    if mode is not None and mode.eo_hfov_deg is not None
                    else self.sensor_performance.eo.horizontal_fov_deg
                ),
                self.target_signature.eo_contrast
                * self.camouflage.eo_contrast_multiplier,
                target_width_m,
                sensor,
                range_m,
            )
            if eo_enabled
            else 0.0
        )
        ir_probability = (
            self._channel_probability(
                self.sensor_performance.ir,
                sensor.ir.image_width_px,
                (
                    mode.ir_hfov_deg
                    if mode is not None and mode.ir_hfov_deg is not None
                    else self.sensor_performance.ir.horizontal_fov_deg
                ),
                self.target_signature.ir_contrast
                * self.camouflage.ir_contrast_multiplier,
                target_width_m,
                sensor,
                range_m,
            )
            if ir_enabled
            else 0.0
        )
        return eo_probability, ir_probability

    def clear_sensor_probability(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
        target_heading_rad: float | None = None,
        sensor_mode: str | None = None,
    ) -> float:
        eo_probability, ir_probability = self.channel_detection_probabilities(
            target,
            observer,
            sensor,
            target_heading_rad,
            sensor_mode,
        )
        _, aspect_probability = self._aspect_terms(
            target,
            observer,
            target_heading_rad,
        )
        mode = self.search_mode(sensor_mode)
        channel_mode = mode.channel_mode if mode is not None else self.channel_mode
        if channel_mode == "eo":
            fused_probability = eo_probability
        elif channel_mode == "ir":
            fused_probability = ir_probability
        else:
            fused_probability = 1.0 - (
                (1.0 - eo_probability) * (1.0 - ir_probability)
            )
        return min(
            1.0,
            fused_probability
            * aspect_probability
            * self.camouflage.algorithm_probability_multiplier,
        )

    def line_of_sight_probability(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
    ) -> float:
        if not self.uses_los:
            return 1.0
        if observer is None:
            raise ValueError("DEM LOS requires the UAV ground-track position")
        terrain = self.terrain
        if not hasattr(terrain, "elevation_at"):
            return self.los_config.unavailable_probability
        observer_ground = terrain.elevation_at(observer.x, observer.y)
        target_ground = terrain.elevation_at(target.x, target.y)
        if observer_ground is None or target_ground is None:
            return self.los_config.unavailable_probability

        horizontal_distance = hypot(target.x - observer.x, target.y - observer.y)
        if horizontal_distance <= self.los_config.sample_spacing_m:
            return 1.0
        observer_z = observer_ground + sensor.altitude_m
        target_height_m = (
            self.target_signature.height_m
            if self.uses_sensor_performance
            else self.los_config.target_height_m
        )
        target_z = target_ground + target_height_m
        sample_count = max(
            1,
            ceil(horizontal_distance / self.los_config.sample_spacing_m),
        )
        for index in range(1, sample_count):
            fraction = index / sample_count
            x = observer.x + fraction * (target.x - observer.x)
            y = observer.y + fraction * (target.y - observer.y)
            terrain_z = terrain.elevation_at(x, y)
            if terrain_z is None:
                return self.los_config.unavailable_probability
            ray_z = observer_z + fraction * (target_z - observer_z)
            if terrain_z + self.los_config.terrain_clearance_m >= ray_z:
                return 0.0
        return 1.0

    def detection_probability(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
        probability_scale: float = 1.0,
        target_heading_rad: float | None = None,
        sensor_mode: str | None = None,
    ) -> float:
        if not 0.0 < probability_scale <= 1.0:
            raise ValueError("probability_scale must be in (0, 1]")
        sensor_probability = (
            self.clear_sensor_probability(
                target,
                observer,
                sensor,
                target_heading_rad,
                sensor_mode,
            )
            if self.uses_sensor_performance
            else self.base_detection_probability
        )
        return min(
            1.0,
            probability_scale
            * sensor_probability
            * self.surface_probability(target)
            * self.line_of_sight_probability(target, observer, sensor),
        )

    def hazard_rate(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
        probability_scale: float = 1.0,
        target_heading_rad: float | None = None,
        sensor_mode: str | None = None,
    ) -> float:
        probability = self.detection_probability(
            target,
            observer,
            sensor,
            probability_scale,
            target_heading_rad,
            sensor_mode,
        )
        if probability >= 1.0:
            return float("inf")
        if probability <= 0.0:
            return 0.0
        # 한 번의 주사에서 PD를 얻는다면 위험률(단위시간 탐지율)은
        #   lambda = -ln(1 - PD) / t_scan
        # 이 lambda를 노출시간에 곱해 누적하면 P = 1 - exp(-lambda * t).
        return -log1p(-probability) / self.reference_scan_s(sensor, sensor_mode)

    def probability_for_exposure(
        self,
        target: Point2D,
        observer: Point2D | None,
        sensor: SensorSpec,
        exposure_s: float,
        probability_scale: float = 1.0,
        target_heading_rad: float | None = None,
        sensor_mode: str | None = None,
    ) -> float:
        if exposure_s <= 0.0:
            return 0.0
        rate = self.hazard_rate(
            target,
            observer,
            sensor,
            probability_scale,
            target_heading_rad,
            sensor_mode,
        )
        if rate == float("inf"):
            return 1.0
        return 1.0 - exp(-rate * exposure_s)


def static_path_detection_probability(
    target: Point2D,
    sensing_segments: list[PathSegment] | tuple[PathSegment, ...],
    detection_radius_m: float,
    search_speed_mps: float,
    sensor: SensorSpec,
    model: SpatialDetectionModel,
    probability_scale: float = 1.0,
    target_heading_rad: float | None = None,
) -> tuple[float, float]:
    """Return effective P_D and exposure time for one stationary hypothesis."""
    if detection_radius_m <= 0.0:
        raise ValueError("detection_radius_m must be positive")
    if search_speed_mps <= 0.0:
        raise ValueError("search_speed_mps must be positive")
    cumulative_hazard = 0.0
    total_exposure_s = 0.0
    for segment in sensing_segments:
        length_m = segment.length_m
        if length_m <= 1e-12:
            continue
        unit_x = (segment.end.x - segment.start.x) / length_m
        unit_y = (segment.end.y - segment.start.y) / length_m
        offset_x = target.x - segment.start.x
        offset_y = target.y - segment.start.y
        projected_m = offset_x * unit_x + offset_y * unit_y
        perpendicular_squared = max(
            0.0,
            offset_x * offset_x + offset_y * offset_y - projected_m * projected_m,
        )
        segment_radius_m = (
            model.footprint_half_width_m(sensor, segment.sensor_mode)
            if segment.sensor_mode is not None
            or sensor.has_extended_search_envelope
            else detection_radius_m
        )
        radius_squared = segment_radius_m * segment_radius_m
        if perpendicular_squared > radius_squared:
            continue
        half_chord_m = sqrt(max(0.0, radius_squared - perpendicular_squared))
        entry_m = max(0.0, projected_m - half_chord_m)
        exit_m = min(length_m, projected_m + half_chord_m)
        if exit_m <= entry_m:
            continue
        exposure_s = (exit_m - entry_m) / search_speed_mps
        midpoint_m = 0.5 * (entry_m + exit_m)
        observer = Point2D(
            segment.start.x + unit_x * midpoint_m,
            segment.start.y + unit_y * midpoint_m,
        )
        rate = model.hazard_rate(
            target,
            observer,
            sensor,
            probability_scale,
            target_heading_rad,
            segment.sensor_mode,
        ) * segment.effective_detection_scale * sensor.lateral_detection_scale(
            sqrt(perpendicular_squared)
        )
        total_exposure_s += exposure_s
        if rate == float("inf"):
            return 1.0, total_exposure_s
        cumulative_hazard += rate * exposure_s
    return 1.0 - exp(-cumulative_hazard), total_exposure_s
