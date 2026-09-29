"""SIR 입자필터 — 숨은 표적 위치에 대한 belief.

필터는 Monte Carlo 진리값을 **절대 보지 않는다**. 사전분포, Markov 전이
커널, 그리고 음성(미탐지) 관측만 쓴다.

핵심 수식
---------
* 예측 (``predict``)
  ``motion``의 IMM5 커널을 입자마다 독립 적용한다. 원을 벗어나면
  escaped=True로 흡수시키고 가중치는 그대로 둔다. 따라서

      P(이탈) = sum_{escaped} w_i        (``outside_probability``)

* 음성 관측 갱신 (``observe_no_detection`` / ``observe_no_detection_segments``)
  이상 센서일 때

      w_i <- w_i * (1 - PD)      (입자 i가 소인 발자국 안일 때)

  공간 탐지모델을 쓸 때는 누적 위험률로

      Lambda_i = sum_seg rate(x_i, observer, sensor) * exposure_s
      w_i <- w_i * exp(-Lambda_i)

  exposure_s는 발자국 원과 소인선의 **현(chord) 길이 / 속도**로 구한다.
  rate는 ``sensor_observation.SpatialDetectionModel.hazard_rate``.

* 유효 표본수와 재표본추출 임계 (``effective_particle_count``)

      ESS = 1 / sum_i w_i^2

  ESS < ``resample_threshold_ratio`` * N 이면 재표본추출한다.

* 재표본추출 (``_resample``)

      systematic : u_i = (u0 + i) / N,  u0 ~ U(0, 1/N)
      stratified : u_i = (i + u_i') / N, u_i' ~ U(0,1) 각각 독립

* roughening (``_roughen``) — Gordon-Salmond-Smith jitter

      sigma = K * E * N^(-1/d),   d = 2 (평면)

  E = 현재 영역 내 입자 산포(축별 최대폭), K = ``roughening_gain``.
  원 밖으로 나가는 jitter는 **기각**해서 이탈 확률질량을 보존한다.
  재표본추출 직후에만 적용하며, 기본은 꺼져 있다
  (``ParticleFilterConfig.sarops()``에서만 켠다).

* 격자 투영 (``cell_masses``)
  입자 가중치를 ``probability.PolarProbabilityMap`` 셀로 모은다.
  Ch2/Ch6의 Markov 상태 질량이 여기서 나온다.

의존
----
* 위: ``models``, ``motion``, ``probability``. 지형·탐지모델은 주입.
* 아래: ``planning/sarops_adapted``(Ch4 탐색모델), ``theory/markov``
  (belief -> Markov 추정), ``planning/estimation``(belief 품질 평가).
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import atan2, cos, exp, hypot, inf, pi, sin, sqrt, tau
from random import Random
from statistics import NormalDist

import numpy as np

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, SensorSpec
from cpp_search.core.motion import (
    MOTION_MODE_NAMES,
    KinematicState,
    RANDOM_MANEUVER_MODE,
    TargetMotionSpec,
    apply_circular_boundary,
    propagate_kinematics,
    sample_initial_kinematics,
)
from cpp_search.core.probability import PolarProbabilityMap, TargetPrior


_USE_PRIOR_TERRAIN = object()


@dataclass(frozen=True, slots=True)
class ParticleFilterConfig:
    """Numerical choices that do not change the target motion hypothesis.

    ``baseline`` preserves the historical bootstrap/SIR implementation.
    ``optimized`` uses randomized stratification for prior/transition draws,
    the actual sensing polylines for negative observations, and stratified
    resampling.  ``sarops`` adds post-resampling roughening, the SAROPS-style
    remedy for sample impoverishment after repeated negative observations
    (Gordon-Salmond-Smith jitter with a Silverman-style bandwidth).
    Roughening stays opt-in because it perturbs positions with noise the
    common truth process does not contain, so it must be a declared
    experiment condition rather than a silent default.
    """

    initialization_scheme: str = "random"
    isotropic_transition_scheme: str = "random"
    observation_geometry: str = "cell"
    resampling_scheme: str = "systematic"
    roughening_scheme: str = "none"
    roughening_gain: float = 0.2

    def __post_init__(self) -> None:
        if self.initialization_scheme not in {"random", "stratified"}:
            raise ValueError("unsupported particle initialization scheme")
        if self.isotropic_transition_scheme not in {"random", "stratified"}:
            raise ValueError("unsupported isotropic transition scheme")
        if self.observation_geometry not in {"cell", "segment"}:
            raise ValueError("unsupported observation geometry")
        if self.resampling_scheme not in {"systematic", "stratified"}:
            raise ValueError("unsupported particle resampling scheme")
        if self.roughening_scheme not in {"none", "jitter"}:
            raise ValueError("unsupported particle roughening scheme")
        if self.roughening_gain <= 0.0:
            raise ValueError("roughening_gain must be positive")

    @classmethod
    def optimized(cls) -> "ParticleFilterConfig":
        return cls(
            initialization_scheme="stratified",
            isotropic_transition_scheme="stratified",
            observation_geometry="segment",
            resampling_scheme="stratified",
        )

    @classmethod
    def sarops(cls, *, roughening_gain: float = 0.2) -> "ParticleFilterConfig":
        """Optimized SIR plus post-resampling roughening."""

        return cls(
            initialization_scheme="stratified",
            isotropic_transition_scheme="stratified",
            observation_geometry="segment",
            resampling_scheme="stratified",
            roughening_scheme="jitter",
            roughening_gain=roughening_gain,
        )

    @property
    def label(self) -> str:
        if self == ParticleFilterConfig():
            return "baseline-sir"
        if self == ParticleFilterConfig.optimized():
            return "optimized-stratified-segment"
        if self == ParticleFilterConfig.sarops(roughening_gain=self.roughening_gain):
            return "sarops-stratified-segment-roughened"
        return "custom"


@dataclass(frozen=True, slots=True)
class TargetStateObservation:
    """Noisy external report of target position, speed, and heading."""

    position: Point2D
    speed_mps: float
    heading_rad: float
    position_sigma_m: float
    speed_sigma_mps: float
    heading_sigma_rad: float
    report_time_s: float = 60.0

    def __post_init__(self) -> None:
        if self.speed_mps < 0.0:
            raise ValueError("observed target speed must not be negative")
        if self.position_sigma_m <= 0.0:
            raise ValueError("position_sigma_m must be positive")
        if self.speed_sigma_mps <= 0.0:
            raise ValueError("speed_sigma_mps must be positive")
        if self.heading_sigma_rad <= 0.0:
            raise ValueError("heading_sigma_rad must be positive")
        if self.report_time_s < 0.0:
            raise ValueError("report_time_s must not be negative")


@dataclass(frozen=True, slots=True)
class ParticleFilterDiagnostics:
    particle_count: int
    effective_particle_count: float
    outside_probability: float
    prediction_step_count: int
    observation_update_count: int
    resample_count: int
    degenerate: bool
    mean_speed_mps: float = 0.0
    mean_absolute_turn_rate_dps: float = 0.0
    motion_model_probabilities: tuple[float, ...] = (0.0, 0.0, 0.0)
    configuration_name: str = "baseline-sir"
    roughening_count: int = 0
    roughening_sigma_m: float = 0.0


class TargetParticleFilter:
    """Bootstrap/SIR particle filter for the hidden target position.

    The filter never receives the Monte Carlo ground-truth position.  It only
    uses the configured prior, the Markov transition model, and negative
    sensor observations.  Particles that leave the fixed search circle are
    retained in an absorbing outside state so escape probability is not
    accidentally normalised back into the search area.
    """

    def __init__(
        self,
        mission: MissionConfig,
        prior: TargetPrior,
        motion: TargetMotionSpec | None,
        particle_count: int = 3_000,
        seed: int = 20_260_809,
        resample_threshold_ratio: float = 0.5,
        initial_angle_range: tuple[float, float] | None = None,
        config: ParticleFilterConfig | None = None,
        terrain=None,
        terrain_stop_ratio: float = 0.4,
        terrain_bias_strength: float = 0.0,
        transition_terrain=_USE_PRIOR_TERRAIN,
        terrain_offroad_probability: float = 0.2,
        terrain_halt_probability_boost: float = 0.0,
    ) -> None:
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if not 0.0 < resample_threshold_ratio <= 1.0:
            raise ValueError("resample_threshold_ratio must be in (0, 1]")
        if not 0.0 <= terrain_bias_strength <= 1.0:
            raise ValueError("terrain_bias_strength must be in [0, 1]")
        if not 0.0 <= terrain_offroad_probability <= 1.0:
            raise ValueError("terrain_offroad_probability must be in [0, 1]")
        if not 0.0 <= terrain_halt_probability_boost <= 1.0:
            raise ValueError("terrain_halt_probability_boost must be in [0, 1]")

        self.mission = mission
        self.motion = motion
        self.particle_count = particle_count
        self.resample_threshold_ratio = resample_threshold_ratio
        self.terrain = terrain
        self.transition_terrain = (
            terrain
            if transition_terrain is _USE_PRIOR_TERRAIN
            else transition_terrain
        )
        self.terrain_stop_ratio = terrain_stop_ratio
        self.terrain_bias_strength = terrain_bias_strength
        self.terrain_offroad_probability = terrain_offroad_probability
        self.terrain_halt_probability_boost = terrain_halt_probability_boost
        if (
            initial_angle_range is not None
            and initial_angle_range[1] <= initial_angle_range[0]
        ):
            raise ValueError("initial_angle_range must have positive width")
        self.initial_angle_range = initial_angle_range
        self.config = config or ParticleFilterConfig()
        self.rng = Random(seed)
        self.points = self._sample_prior(prior)
        if motion is not None and motion.motion_model == "isotropic":
            initial_states = self._sample_isotropic_states(particle_count)
        else:
            initial_states = [
                KinematicState(0.0, 0.0, 0.0, 0)
                if motion is None
                else sample_initial_kinematics(motion, self.rng)
                for _ in range(particle_count)
            ]
        self.speeds_mps = [state.speed_mps for state in initial_states]
        self.headings_rad = [state.heading_rad for state in initial_states]
        self.turn_rates_rad_s = [
            state.turn_rate_rad_s for state in initial_states
        ]
        self.motion_modes = [state.mode for state in initial_states]
        self.escaped = [False] * particle_count
        self.weights = [1.0 / particle_count] * particle_count
        self.prediction_step_count = 0
        self.observation_update_count = 0
        self.resample_count = 0
        self.roughening_count = 0
        self.last_roughening_sigma_m = 0.0
        self.degenerate = False
        self.motion_phase_s = 0.0

    def _sample_isotropic_state(self) -> KinematicState:
        assert self.motion is not None
        speed_ratio = min(
            max(
                self.rng.gauss(
                    self.motion.nominal_speed_ratio,
                    self.motion.speed_sigma_ratio,
                ),
                0.0,
            ),
            1.0,
        )
        return KinematicState(
            speed_ratio * self.motion.max_speed_mps,
            tau * self.rng.random(),
            0.0,
            RANDOM_MANEUVER_MODE,
        )

    def _sample_isotropic_states(self, count: int) -> list[KinematicState]:
        if self.config.isotropic_transition_scheme == "random":
            return [self._sample_isotropic_state() for _ in range(count)]

        assert self.motion is not None
        normal = NormalDist()
        speed_ratios = []
        headings = []
        for index in range(count):
            speed_quantile = (index + self.rng.random()) / count
            sampled_ratio = (
                self.motion.nominal_speed_ratio
                + self.motion.speed_sigma_ratio * normal.inv_cdf(speed_quantile)
            )
            speed_ratios.append(min(max(sampled_ratio, 0.0), 1.0))
            headings.append(tau * (index + self.rng.random()) / count)
        # Break the artificial radial/heading pairing while retaining one
        # sample in every heading stratum.
        self.rng.shuffle(headings)
        return [
            KinematicState(
                speed_ratio * self.motion.max_speed_mps,
                heading,
                0.0,
                RANDOM_MANEUVER_MODE,
            )
            for speed_ratio, heading in zip(speed_ratios, headings)
        ]

    def _sample_prior(self, prior: TargetPrior) -> list[Point2D]:
        # Stratified sampling only models the radial prior; when a terrain
        # field imposes angular structure, fall back to rejection sampling so
        # the per-position terrain weight is honoured.
        if self.config.initialization_scheme == "stratified" and self.terrain is None:
            return self._sample_prior_stratified(prior)

        points: list[Point2D] = []
        while len(points) < self.particle_count:
            radius = self.mission.search_radius_m * sqrt(self.rng.random())
            angle = (
                tau * self.rng.random()
                if self.initial_angle_range is None
                else self.initial_angle_range[0]
                + (self.initial_angle_range[1] - self.initial_angle_range[0])
                * self.rng.random()
            )
            x = self.mission.center.x + radius * cos(angle)
            y = self.mission.center.y + radius * sin(angle)
            acceptance = prior.relative_density(
                radius, self.mission.search_radius_m
            )
            if self.terrain is not None:
                acceptance *= self.terrain.prior_weight(
                    x,
                    y,
                    self.terrain_stop_ratio,
                    (
                        self.motion.effective_initial_mode_probabilities
                        if self.motion is not None
                        and self.motion.motion_model == "imm5"
                        else None
                    ),
                )
            if self.rng.random() > acceptance:
                continue
            points.append(Point2D(x, y))
        return points

    def _sample_prior_stratified(self, prior: TargetPrior) -> list[Point2D]:
        """Randomized Latin-hypercube draw from the radial target prior."""

        radial_bin_count = 4_096
        radius_limit = self.mission.search_radius_m
        radial_edges = [
            radius_limit * index / radial_bin_count
            for index in range(radial_bin_count + 1)
        ]
        cumulative = []
        total = 0.0
        for inner, outer in zip(radial_edges, radial_edges[1:]):
            midpoint = (inner + outer) / 2.0
            # Annular area is proportional to outer^2-inner^2.
            total += prior.relative_density(midpoint, radius_limit) * (
                outer * outer - inner * inner
            )
            cumulative.append(total)
        if total <= 0.0:
            raise ValueError("target prior has no probability mass")
        cumulative = [value / total for value in cumulative]

        radii = []
        for index in range(self.particle_count):
            quantile = (index + self.rng.random()) / self.particle_count
            bin_index = min(bisect_left(cumulative, quantile), radial_bin_count - 1)
            lower_cdf = cumulative[bin_index - 1] if bin_index else 0.0
            upper_cdf = cumulative[bin_index]
            local_fraction = (
                (quantile - lower_cdf) / (upper_cdf - lower_cdf)
                if upper_cdf > lower_cdf
                else 0.5
            )
            inner = radial_edges[bin_index]
            outer = radial_edges[bin_index + 1]
            radii.append(
                sqrt(inner * inner + local_fraction * (outer * outer - inner * inner))
            )

        min_angle, max_angle = (
            (0.0, tau)
            if self.initial_angle_range is None
            else self.initial_angle_range
        )
        angle_width = max_angle - min_angle
        angles = [
            min_angle
            + angle_width * (index + self.rng.random()) / self.particle_count
            for index in range(self.particle_count)
        ]
        self.rng.shuffle(angles)
        return [
            Point2D(
                self.mission.center.x + radius * cos(angle),
                self.mission.center.y + radius * sin(angle),
            )
            for radius, angle in zip(radii, angles)
        ]

    def predict(self, elapsed_s: float) -> None:
        """Apply the target transition model without consulting ground truth."""
        if self.motion is None or elapsed_s <= 0.0 or self.degenerate:
            return

        if self.motion.motion_model == "isotropic":
            self._predict_isotropic(elapsed_s)
            return

        remaining_s = elapsed_s
        while remaining_s > 1e-9:
            delta_s = min(self.motion.step_s, remaining_s)
            for index, point in enumerate(self.points):
                if self.escaped[index]:
                    continue
                halt_boost = (
                    self.transition_terrain.halt_probability_boost(
                        point.x,
                        point.y,
                        self.terrain_halt_probability_boost,
                    )
                    if self.transition_terrain is not None
                    and self.terrain_halt_probability_boost > 0.0
                    and self.motion.motion_model == "imm5"
                    else 0.0
                )
                dx, dy, state = propagate_kinematics(
                    self.motion,
                    KinematicState(
                        self.speeds_mps[index],
                        self.headings_rad[index],
                        self.turn_rates_rad_s[index],
                        self.motion_modes[index],
                    ),
                    delta_s,
                    self.rng,
                    halt_probability_boost=halt_boost,
                )
                if self.transition_terrain is not None:
                    dx, dy = self.transition_terrain.transition_step(
                        point.x,
                        point.y,
                        dx,
                        dy,
                        follow_strength=self.terrain_bias_strength,
                        offroad_probability=self.terrain_offroad_probability,
                        rng=self.rng,
                        motion_mode=(
                            state.mode
                            if self.motion.motion_model == "imm5"
                            else None
                        ),
                    )
                    if hypot(dx, dy) > 1e-9:
                        state = KinematicState(
                            state.speed_mps,
                            atan2(dy, dx) % tau,
                            state.turn_rate_rad_s,
                            state.mode,
                        )
                predicted, state, escaped = apply_circular_boundary(
                    point,
                    dx,
                    dy,
                    state,
                    self.mission.center,
                    self.mission.search_radius_m,
                    self.motion.boundary_mode,
                )
                self.points[index] = predicted
                self.speeds_mps[index] = state.speed_mps
                self.headings_rad[index] = state.heading_rad
                self.turn_rates_rad_s[index] = state.turn_rate_rad_s
                self.motion_modes[index] = state.mode
                self.escaped[index] = escaped
            remaining_s -= delta_s
            self.prediction_step_count += 1

    def _predict_isotropic(self, elapsed_s: float) -> None:
        """Advance the random-heading process on its fixed Markov clock.

        A planner may call ``predict`` at different intervals, but the target
        must retain one sampled velocity for the configured ``step_s``.  The
        target process therefore remains identical for every replanning period.
        """

        assert self.motion is not None
        remaining_s = elapsed_s
        while remaining_s > 1e-9:
            until_transition_s = self.motion.step_s - self.motion_phase_s
            delta_s = min(remaining_s, until_transition_s)
            for index, point in enumerate(self.points):
                if self.escaped[index]:
                    continue
                state = KinematicState(
                    self.speeds_mps[index],
                    self.headings_rad[index],
                    0.0,
                    self.motion_modes[index],
                )
                dx = state.speed_mps * delta_s * cos(state.heading_rad)
                dy = state.speed_mps * delta_s * sin(state.heading_rad)
                predicted, state, escaped = apply_circular_boundary(
                    point,
                    dx,
                    dy,
                    state,
                    self.mission.center,
                    self.mission.search_radius_m,
                    self.motion.boundary_mode,
                )
                self.points[index] = predicted
                self.speeds_mps[index] = state.speed_mps
                self.headings_rad[index] = state.heading_rad
                self.turn_rates_rad_s[index] = 0.0
                self.motion_modes[index] = state.mode
                self.escaped[index] = escaped

            remaining_s -= delta_s
            self.motion_phase_s += delta_s
            self.prediction_step_count += 1
            if self.motion_phase_s >= self.motion.step_s - 1e-9:
                self.motion_phase_s = 0.0
                active_indices = [
                    index
                    for index in range(self.particle_count)
                    if not self.escaped[index]
                ]
                states = self._sample_isotropic_states(len(active_indices))
                for index, state in zip(active_indices, states):
                    self.speeds_mps[index] = state.speed_mps
                    self.headings_rad[index] = state.heading_rad
                    self.turn_rates_rad_s[index] = 0.0
                    self.motion_modes[index] = state.mode

    def observe_no_detection(
        self,
        observed_cell_indices: set[int],
        probability_map: PolarProbabilityMap,
        detection_probability: float,
    ) -> None:
        """Apply p(no detection | x) to every particle and resample if needed."""
        if not 0.0 < detection_probability <= 1.0:
            raise ValueError("detection_probability must be in (0, 1]")
        if not observed_cell_indices or self.degenerate:
            return

        missed_probability = 1.0 - detection_probability
        for index, (point, weight) in enumerate(zip(self.points, self.weights)):
            if self.escaped[index]:
                continue
            if self._cell_index(point, probability_map) in observed_cell_indices:
                self.weights[index] = weight * missed_probability

        self._finish_observation_update()

    def observe_no_detection_segments(
        self,
        sensing_segments: list[PathSegment] | tuple[PathSegment, ...],
        detection_radius_m: float,
        detection_probability: float,
        detection_model=None,
        sensor: SensorSpec | None = None,
        search_speed_mps: float | None = None,
    ) -> None:
        """Update weights from the actual sensing polyline footprint.

        This removes the polar-cell quantisation used by the baseline filter.
        It intentionally applies the same ideal-PD likelihood assumed by the
        Monte Carlo evaluator, so the optimization changes numerical accuracy
        rather than the sensor model.
        """

        if detection_radius_m <= 0.0:
            raise ValueError("detection_radius_m must be positive")
        if not 0.0 < detection_probability <= 1.0:
            raise ValueError("detection_probability must be in (0, 1]")
        if not sensing_segments or self.degenerate:
            return
        if detection_model is not None and (
            sensor is None
            or search_speed_mps is None
            or search_speed_mps <= 0.0
        ):
            raise ValueError(
                "spatial detection updates require sensor and search speed"
            )

        active_indices = np.asarray(
            [index for index, escaped in enumerate(self.escaped) if not escaped],
            dtype=int,
        )
        if active_indices.size == 0:
            return
        x = np.asarray([self.points[index].x for index in active_indices])
        y = np.asarray([self.points[index].y for index in active_indices])
        observed = np.zeros(active_indices.size, dtype=bool)
        cumulative_hazard = np.zeros(active_indices.size, dtype=float)
        radius_squared = detection_radius_m * detection_radius_m
        for segment in sensing_segments:
            dx = segment.end.x - segment.start.x
            dy = segment.end.y - segment.start.y
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-15:
                distance_squared = (
                    (x - segment.start.x) ** 2 + (y - segment.start.y) ** 2
                )
            else:
                raw_projection = (
                    (x - segment.start.x) * dx + (y - segment.start.y) * dy
                ) / length_squared
                projection = np.clip(raw_projection, 0.0, 1.0)
                closest_x = segment.start.x + projection * dx
                closest_y = segment.start.y + projection * dy
                distance_squared = (x - closest_x) ** 2 + (y - closest_y) ** 2
            segment_observed = distance_squared <= radius_squared
            observed |= segment_observed
            if detection_model is None or not np.any(segment_observed):
                continue
            if length_squared <= 1e-15:
                continue

            along_track_m = raw_projection * sqrt(length_squared)
            offset_squared = (x - segment.start.x) ** 2 + (y - segment.start.y) ** 2
            perpendicular_squared = np.maximum(
                0.0,
                offset_squared - along_track_m * along_track_m,
            )
            half_chord = np.sqrt(
                np.maximum(radius_squared - perpendicular_squared, 0.0)
            )
            entry_m = np.maximum(0.0, along_track_m - half_chord)
            exit_m = np.minimum(sqrt(length_squared), along_track_m + half_chord)
            exposure_s = np.maximum(exit_m - entry_m, 0.0) / search_speed_mps
            for local_index in np.flatnonzero(segment_observed & (exposure_s > 0.0)):
                particle_index = int(active_indices[local_index])
                midpoint_ratio = (
                    0.5 * (entry_m[local_index] + exit_m[local_index])
                    / sqrt(length_squared)
                )
                observer = Point2D(
                    segment.start.x + midpoint_ratio * dx,
                    segment.start.y + midpoint_ratio * dy,
                )
                rate = detection_model.hazard_rate(
                    self.points[particle_index],
                    observer,
                    sensor,
                    detection_probability,
                    self.headings_rad[particle_index],
                )
                if rate == inf:
                    cumulative_hazard[local_index] = inf
                elif cumulative_hazard[local_index] != inf:
                    cumulative_hazard[local_index] += (
                        rate * exposure_s[local_index]
                    )

        if detection_model is None:
            missed_probability = 1.0 - detection_probability
            for particle_index in active_indices[observed]:
                self.weights[int(particle_index)] *= missed_probability
        else:
            likelihoods = np.exp(-cumulative_hazard)
            for local_index, particle_index in enumerate(active_indices):
                self.weights[int(particle_index)] *= float(likelihoods[local_index])
        self._finish_observation_update()

    def observe_state(self, observation: TargetStateObservation) -> None:
        """Assimilate one independent Gaussian position/speed/heading report."""

        if self.degenerate:
            return
        log_likelihoods: list[float] = []
        maximum_log_likelihood = -inf
        for point, speed, heading, escaped in zip(
            self.points,
            self.speeds_mps,
            self.headings_rad,
            self.escaped,
        ):
            if escaped:
                log_likelihoods.append(-inf)
                continue
            position_error = point.distance_to(observation.position)
            speed_error = speed - observation.speed_mps
            heading_error = (heading - observation.heading_rad + pi) % tau - pi
            log_likelihood = -0.5 * (
                (position_error / observation.position_sigma_m) ** 2
                + (speed_error / observation.speed_sigma_mps) ** 2
                + (heading_error / observation.heading_sigma_rad) ** 2
            )
            log_likelihoods.append(log_likelihood)
            maximum_log_likelihood = max(maximum_log_likelihood, log_likelihood)

        if maximum_log_likelihood == -inf:
            self.weights = [0.0] * self.particle_count
            self.degenerate = True
            return
        self.weights = [
            weight * exp(log_likelihood - maximum_log_likelihood)
            if log_likelihood > -inf
            else 0.0
            for weight, log_likelihood in zip(self.weights, log_likelihoods)
        ]
        self._finish_observation_update()

    def _finish_observation_update(self) -> None:
        self.observation_update_count += 1
        total_weight = sum(self.weights)
        if total_weight <= 1e-15:
            self.weights = [0.0] * self.particle_count
            self.degenerate = True
            return
        self.weights = [weight / total_weight for weight in self.weights]
        if (
            self.effective_particle_count
            < self.resample_threshold_ratio * self.particle_count
        ):
            self._resample()

    def cell_masses(self, probability_map: PolarProbabilityMap) -> list[float]:
        """Return the current posterior mass in each in-area polar cell."""
        masses = [0.0] * len(probability_map.cells)
        if self.degenerate:
            return masses
        for point, weight, escaped in zip(
            self.points, self.weights, self.escaped
        ):
            if not escaped:
                masses[self._cell_index(point, probability_map)] += weight
        return masses

    @property
    def effective_particle_count(self) -> float:
        # 유효 표본수 ESS = 1 / sum_i w_i^2.
        # 가중치가 고르면 N, 하나에 몰리면 1에 가까워진다.
        # ESS < resample_threshold_ratio * N 이면 재표본추출한다.
        squared_weight_sum = sum(weight * weight for weight in self.weights)
        return 1.0 / squared_weight_sum if squared_weight_sum > 0.0 else 0.0

    @property
    def outside_probability(self) -> float:
        return sum(
            weight
            for weight, escaped in zip(self.weights, self.escaped)
            if escaped
        )

    @property
    def diagnostics(self) -> ParticleFilterDiagnostics:
        mode_names = self.motion.mode_names if self.motion is not None else MOTION_MODE_NAMES
        mode_probabilities = [0.0] * len(mode_names)
        mean_speed_mps = 0.0
        mean_absolute_turn_rate = 0.0
        for speed, turn_rate, mode, weight in zip(
            self.speeds_mps,
            self.turn_rates_rad_s,
            self.motion_modes,
            self.weights,
        ):
            mean_speed_mps += speed * weight
            mean_absolute_turn_rate += abs(turn_rate) * weight
            mode_probabilities[mode] += weight
        return ParticleFilterDiagnostics(
            particle_count=self.particle_count,
            effective_particle_count=self.effective_particle_count,
            outside_probability=self.outside_probability,
            prediction_step_count=self.prediction_step_count,
            observation_update_count=self.observation_update_count,
            resample_count=self.resample_count,
            degenerate=self.degenerate,
            mean_speed_mps=mean_speed_mps,
            mean_absolute_turn_rate_dps=mean_absolute_turn_rate * 180.0 / 3.141592653589793,
            motion_model_probabilities=tuple(mode_probabilities),
            configuration_name=self.config.label,
            roughening_count=self.roughening_count,
            roughening_sigma_m=self.last_roughening_sigma_m,
        )

    def _cell_index(
        self, point: Point2D, probability_map: PolarProbabilityMap
    ) -> int:
        dx = point.x - self.mission.center.x
        dy = point.y - self.mission.center.y
        radius = sqrt(dx * dx + dy * dy)
        radial_bin_count = (
            len(probability_map.cells) // probability_map.angular_bin_count
        )
        radial_index = min(
            int(radius / probability_map.radial_step_m),
            radial_bin_count - 1,
        )
        angle = (tau + atan2(dy, dx)) % tau
        angular_index = min(
            int(angle / tau * probability_map.angular_bin_count),
            probability_map.angular_bin_count - 1,
        )
        return radial_index * probability_map.angular_bin_count + angular_index

    def _resample(self) -> None:
        if self.config.resampling_scheme == "stratified":
            positions = [
                (index + self.rng.random()) / self.particle_count
                for index in range(self.particle_count)
            ]
        else:
            start = self.rng.random() / self.particle_count
            positions = [
                start + index / self.particle_count
                for index in range(self.particle_count)
            ]
        self._resample_at_positions(positions)

    # Kept for notebooks/tests that used the previous private helper.
    def _systematic_resample(self) -> None:
        start = self.rng.random() / self.particle_count
        self._resample_at_positions(
            [start + index / self.particle_count for index in range(self.particle_count)]
        )

    def _resample_at_positions(self, positions: list[float]) -> None:
        selected_indices: list[int] = []
        cumulative_weight = self.weights[0]
        source_index = 0
        for position in positions:
            while position > cumulative_weight and source_index < self.particle_count - 1:
                source_index += 1
                cumulative_weight += self.weights[source_index]
            selected_indices.append(source_index)

        self.points = [self.points[index] for index in selected_indices]
        self.speeds_mps = [self.speeds_mps[index] for index in selected_indices]
        self.headings_rad = [
            self.headings_rad[index] for index in selected_indices
        ]
        self.turn_rates_rad_s = [
            self.turn_rates_rad_s[index] for index in selected_indices
        ]
        self.motion_modes = [
            self.motion_modes[index] for index in selected_indices
        ]
        self.escaped = [self.escaped[index] for index in selected_indices]
        self.weights = [1.0 / self.particle_count] * self.particle_count
        self.resample_count += 1
        self._roughen()

    def _roughen(self) -> None:
        """Jitter resampled positions to counter sample impoverishment.

        Bandwidth follows Gordon-Salmond-Smith: ``sigma = K * E * N**(-1/d)``
        with ``E`` the current in-area particle spread and ``d = 2``.  A jitter
        that would move a particle outside the fixed search circle is rejected
        so the absorbing outside state keeps its exact probability mass.
        """

        if self.config.roughening_scheme != "jitter" or self.degenerate:
            return
        interior = [
            point
            for point, escaped in zip(self.points, self.escaped)
            if not escaped
        ]
        if len(interior) < 2:
            self.last_roughening_sigma_m = 0.0
            return
        spread = max(
            max(point.x for point in interior) - min(point.x for point in interior),
            max(point.y for point in interior) - min(point.y for point in interior),
        )
        if spread <= 0.0:
            self.last_roughening_sigma_m = 0.0
            return
        # Gordon-Salmond-Smith 대역폭: sigma = K * E * N^(-1/d), d = 2.
        # 입자 수가 늘수록 jitter를 줄여 원 분포를 왜곡하지 않게 한다.
        sigma = (
            self.config.roughening_gain
            * spread
            * self.particle_count ** (-0.5)
        )
        if sigma <= 0.0:
            self.last_roughening_sigma_m = 0.0
            return
        radius = self.mission.search_radius_m
        center = self.mission.center
        for index, (point, escaped) in enumerate(zip(self.points, self.escaped)):
            if escaped:
                continue
            candidate = Point2D(
                point.x + self.rng.gauss(0.0, sigma),
                point.y + self.rng.gauss(0.0, sigma),
            )
            if candidate.distance_to(center) <= radius:
                self.points[index] = candidate
        self.roughening_count += 1
        self.last_roughening_sigma_m = sigma
