from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np

from .geometry import LocalFrame, LocalPoint, polyline_length


MODEL_NAME = "RHP-FE-PF-PW-ARC"
PF_CONFIGURATION = "optimized-stratified-segment"
LOCAL_SECTOR_COUNT = 6
LOCAL_SECTOR_WIDTH_DEG = 360.0 / LOCAL_SECTOR_COUNT


@dataclass(frozen=True, slots=True)
class RHPBeliefSummary:
    mean_east_m: float
    mean_north_m: float
    mean_speed_mps: float
    mean_heading_deg: float
    uncertainty_east_95_m: float
    uncertainty_north_95_m: float
    effective_particle_count: float
    outside_probability: float
    revision: int


@dataclass(frozen=True, slots=True)
class RHPRouteAction:
    vehicle_id: int
    radius_m: float
    direction: int
    signature: str
    previous_position: LocalPoint
    scan_points: tuple[LocalPoint, ...]
    execution_points: tuple[tuple[LocalPoint, bool], ...]
    transit_time_s: float
    full_action_time_s: float
    radial_probability: float
    detection_probability: float = 0.0
    expected_detection_time_s: float = 0.0
    score: float = 0.0
    first_hits_s: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class RHPPlanningDecision:
    candidates: dict[int, dict[str, Any]]
    route_updates: dict[int, tuple[dict[str, Any], ...]]


class IsotropicTargetParticleFilter:
    """Optimized PF used by the validated RHP-FE-PF-PW-ARC baseline.

    The implementation is intentionally self-contained so the GCS repository
    remains portable.  It follows the research simulator's numerical choices:
    stratified moving-ring prior, stratified isotropic Markov transitions,
    segment-based negative observations, stratified resampling, and an
    absorbing outside state.
    """

    MOTION_STEP_S = 30.0
    NOMINAL_SPEED_RATIO = 0.65
    SPEED_SIGMA_RATIO = 0.15
    PRIOR_MEAN_RADIUS_RATIO = 0.65
    PRIOR_SIGMA_RATIO = 0.15

    def __init__(
        self,
        center: LocalPoint,
        search_radius_m: float,
        *,
        maximum_speed_mps: float,
        particle_count: int = 3_000,
        seed: int = 20_260_811,
        resample_threshold_ratio: float = 0.5,
    ) -> None:
        if search_radius_m <= 0.0:
            raise ValueError("search_radius_m must be positive")
        if maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive")
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if not 0.0 < resample_threshold_ratio <= 1.0:
            raise ValueError("resample_threshold_ratio must be in (0, 1]")
        self.center = center
        self.search_radius_m = float(search_radius_m)
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.particle_count = int(particle_count)
        self.resample_threshold_ratio = float(resample_threshold_ratio)
        self.rng = np.random.default_rng(seed)
        self.east_m, self.north_m = self._sample_prior()
        self.speed_mps, self.heading_rad = self._sample_isotropic_states(
            self.particle_count
        )
        self.escaped = np.zeros(self.particle_count, dtype=bool)
        self.weights = np.full(
            self.particle_count,
            1.0 / self.particle_count,
            dtype=float,
        )
        self.motion_phase_s = 0.0
        self.revision = 0
        self.prediction_step_count = 0
        self.observation_update_count = 0
        self.resample_count = 0
        self.degenerate = False

    def _sample_prior(self) -> tuple[np.ndarray, np.ndarray]:
        radial_bin_count = 4_096
        edges = np.linspace(0.0, self.search_radius_m, radial_bin_count + 1)
        midpoint = (edges[:-1] + edges[1:]) / 2.0
        sigma_m = self.PRIOR_SIGMA_RATIO * self.search_radius_m
        mean_m = self.PRIOR_MEAN_RADIUS_RATIO * self.search_radius_m
        density = np.exp(-0.5 * ((midpoint - mean_m) / sigma_m) ** 2)
        annular_mass = density * (edges[1:] ** 2 - edges[:-1] ** 2)
        cumulative = np.cumsum(annular_mass)
        cumulative /= cumulative[-1]
        quantiles = (
            np.arange(self.particle_count, dtype=float)
            + self.rng.random(self.particle_count)
        ) / self.particle_count
        indices = np.searchsorted(cumulative, quantiles, side="left")
        indices = np.clip(indices, 0, radial_bin_count - 1)
        lower_cdf = np.where(indices > 0, cumulative[np.maximum(indices - 1, 0)], 0.0)
        upper_cdf = cumulative[indices]
        local_fraction = np.divide(
            quantiles - lower_cdf,
            upper_cdf - lower_cdf,
            out=np.full(self.particle_count, 0.5),
            where=(upper_cdf - lower_cdf) > 1e-15,
        )
        inner = edges[indices]
        outer = edges[indices + 1]
        radii = np.sqrt(
            inner * inner + local_fraction * (outer * outer - inner * inner)
        )
        angles = math.tau * (
            np.arange(self.particle_count, dtype=float)
            + self.rng.random(self.particle_count)
        ) / self.particle_count
        self.rng.shuffle(angles)
        return (
            self.center.east_m + radii * np.cos(angles),
            self.center.north_m + radii * np.sin(angles),
        )

    def _sample_isotropic_states(
        self,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if count <= 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        normal = NormalDist()
        speed_quantiles = (
            np.arange(count, dtype=float) + self.rng.random(count)
        ) / count
        ratios = np.asarray(
            [
                self.NOMINAL_SPEED_RATIO
                + self.SPEED_SIGMA_RATIO * normal.inv_cdf(float(value))
                for value in speed_quantiles
            ],
            dtype=float,
        )
        speeds = np.clip(ratios, 0.0, 1.0) * self.maximum_speed_mps
        headings = math.tau * (
            np.arange(count, dtype=float) + self.rng.random(count)
        ) / count
        self.rng.shuffle(headings)
        return speeds, headings

    def predict(self, elapsed_s: float) -> None:
        remaining_s = max(0.0, float(elapsed_s))
        if remaining_s <= 0.0 or self.degenerate:
            return
        while remaining_s > 1e-9:
            until_transition_s = self.MOTION_STEP_S - self.motion_phase_s
            delta_s = min(remaining_s, until_transition_s)
            active = ~self.escaped
            self.east_m[active] += (
                self.speed_mps[active]
                * delta_s
                * np.cos(self.heading_rad[active])
            )
            self.north_m[active] += (
                self.speed_mps[active]
                * delta_s
                * np.sin(self.heading_rad[active])
            )
            radius_squared = (
                (self.east_m - self.center.east_m) ** 2
                + (self.north_m - self.center.north_m) ** 2
            )
            self.escaped |= radius_squared >= self.search_radius_m**2
            remaining_s -= delta_s
            self.motion_phase_s += delta_s
            self.prediction_step_count += 1
            if self.motion_phase_s >= self.MOTION_STEP_S - 1e-9:
                self.motion_phase_s = 0.0
                active_indices = np.flatnonzero(~self.escaped)
                speeds, headings = self._sample_isotropic_states(
                    int(active_indices.size)
                )
                self.speed_mps[active_indices] = speeds
                self.heading_rad[active_indices] = headings
        self.revision += 1

    def observe_position(
        self,
        measurement: LocalPoint,
        *,
        measurement_std_m: float,
    ) -> None:
        if self.degenerate:
            return
        sigma = max(1.0, float(measurement_std_m))
        distance_squared = (
            (self.east_m - measurement.east_m) ** 2
            + (self.north_m - measurement.north_m) ** 2
        )
        log_likelihood = -0.5 * distance_squared / (sigma * sigma)
        log_likelihood[self.escaped] = -np.inf
        maximum = float(np.max(log_likelihood))
        if not math.isfinite(maximum):
            return
        self.weights *= np.exp(log_likelihood - maximum)
        self._finish_observation()

    def observe_no_detection_segments(
        self,
        segments: Iterable[tuple[LocalPoint, LocalPoint]],
        *,
        detection_radius_m: float,
        detection_probability: float,
    ) -> None:
        if self.degenerate:
            return
        observed = np.zeros(self.particle_count, dtype=bool)
        radius_squared = float(detection_radius_m) ** 2
        for start, end in segments:
            dx = end.east_m - start.east_m
            dy = end.north_m - start.north_m
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-15:
                distance_squared = (
                    (self.east_m - start.east_m) ** 2
                    + (self.north_m - start.north_m) ** 2
                )
            else:
                projection = (
                    (self.east_m - start.east_m) * dx
                    + (self.north_m - start.north_m) * dy
                ) / length_squared
                projection = np.clip(projection, 0.0, 1.0)
                closest_east = start.east_m + projection * dx
                closest_north = start.north_m + projection * dy
                distance_squared = (
                    (self.east_m - closest_east) ** 2
                    + (self.north_m - closest_north) ** 2
                )
            observed |= (~self.escaped) & (distance_squared <= radius_squared)
        if not np.any(observed):
            return
        self.weights[observed] *= 1.0 - float(detection_probability)
        self._finish_observation()

    def _finish_observation(self) -> None:
        self.observation_update_count += 1
        total = float(np.sum(self.weights))
        if total <= 1e-15:
            self.degenerate = True
            return
        self.weights /= total
        if self.effective_particle_count < (
            self.resample_threshold_ratio * self.particle_count
        ):
            self._resample_stratified()
        self.revision += 1

    def _resample_stratified(self) -> None:
        positions = (
            np.arange(self.particle_count, dtype=float)
            + self.rng.random(self.particle_count)
        ) / self.particle_count
        cumulative = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative, positions, side="left")
        indices = np.clip(indices, 0, self.particle_count - 1)
        self.east_m = self.east_m[indices].copy()
        self.north_m = self.north_m[indices].copy()
        self.speed_mps = self.speed_mps[indices].copy()
        self.heading_rad = self.heading_rad[indices].copy()
        self.escaped = self.escaped[indices].copy()
        self.weights.fill(1.0 / self.particle_count)
        self.resample_count += 1

    @property
    def effective_particle_count(self) -> float:
        squared = float(np.sum(self.weights * self.weights))
        return 1.0 / squared if squared > 0.0 else 0.0

    @property
    def outside_probability(self) -> float:
        return float(np.sum(self.weights[self.escaped]))

    def summary(self) -> RHPBeliefSummary:
        active_weights = np.where(self.escaped, 0.0, self.weights)
        total = float(np.sum(active_weights))
        if total <= 1e-15:
            active_weights = self.weights
            total = max(float(np.sum(active_weights)), 1.0)
        normalized = active_weights / total
        mean_east = float(np.sum(self.east_m * normalized))
        mean_north = float(np.sum(self.north_m * normalized))
        mean_speed = float(np.sum(self.speed_mps * normalized))
        heading_x = float(np.sum(np.cos(self.heading_rad) * normalized))
        heading_y = float(np.sum(np.sin(self.heading_rad) * normalized))
        variance_east = float(
            np.sum((self.east_m - mean_east) ** 2 * normalized)
        )
        variance_north = float(
            np.sum((self.north_m - mean_north) ** 2 * normalized)
        )
        confidence_scale = math.sqrt(5.991)
        return RHPBeliefSummary(
            mean_east,
            mean_north,
            mean_speed,
            math.degrees(math.atan2(heading_y, heading_x)) % 360.0,
            confidence_scale * math.sqrt(max(0.0, variance_east)),
            confidence_scale * math.sqrt(max(0.0, variance_north)),
            self.effective_particle_count,
            self.outside_probability,
            self.revision,
        )

    def sampled_forecast(
        self,
        sample_count: int,
        horizon_s: float,
        *,
        seed: int,
        time_step_s: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return common-random-number future positions for candidate scoring."""
        sample_count = max(1, int(sample_count))
        step_count = max(1, int(math.ceil(horizon_s / time_step_s)))
        rng = np.random.default_rng(seed)
        total = float(np.sum(self.weights))
        probabilities = self.weights / total if total > 0.0 else None
        indices = rng.choice(
            self.particle_count,
            size=sample_count,
            replace=True,
            p=probabilities,
        )
        positions = np.empty((step_count + 1, sample_count, 2), dtype=float)
        positions[0, :, 0] = self.east_m[indices]
        positions[0, :, 1] = self.north_m[indices]
        escaped = np.empty((step_count + 1, sample_count), dtype=bool)
        escaped[0] = self.escaped[indices]
        speed = np.zeros(sample_count, dtype=float)
        heading = np.zeros(sample_count, dtype=float)
        phase_s = self.MOTION_STEP_S
        for step_index in range(step_count):
            if phase_s >= self.MOTION_STEP_S - 1e-9:
                ratios = np.clip(
                    rng.normal(
                        self.NOMINAL_SPEED_RATIO,
                        self.SPEED_SIGMA_RATIO,
                        sample_count,
                    ),
                    0.0,
                    1.0,
                )
                speed = ratios * self.maximum_speed_mps
                heading = rng.uniform(0.0, math.tau, sample_count)
                phase_s = 0.0
            delta_s = min(time_step_s, horizon_s - step_index * time_step_s)
            positions[step_index + 1, :, 0] = (
                positions[step_index, :, 0] + speed * np.cos(heading) * delta_s
            )
            positions[step_index + 1, :, 1] = (
                positions[step_index, :, 1] + speed * np.sin(heading) * delta_s
            )
            radius_squared = (
                (positions[step_index + 1, :, 0] - self.center.east_m) ** 2
                + (positions[step_index + 1, :, 1] - self.center.north_m) ** 2
            )
            escaped[step_index + 1] = (
                escaped[step_index] | (radius_squared >= self.search_radius_m**2)
            )
            phase_s += delta_s
        return positions, escaped, 1.0 / sample_count


class RHPFEPFPWARCPlanner:
    """Live GCS adapter for the validated RHP-FE-PF-PW-ARC planner.

    Candidates are evaluated from the current PF every GCS planning cycle.
    The best action is displayed immediately, while only a 25-second prefix is
    committed at each RHP decision epoch.  Positive detection is handled by
    the caller, which stops invoking search planning and transfers to AUTO.
    """

    def __init__(
        self,
        frame: LocalFrame,
        search_center: LocalPoint,
        *,
        search_radius_m: float,
        track_spacing_m: float,
        detection_radius_m: float,
        transit_speed_mps: float,
        search_speed_mps: float,
        decision_interval_s: float = 25.0,
        encounter_sample_count: int = 64,
        radial_shortlist_count: int = 5,
        detection_probability: float = 1.0,
        seed: int = 20_260_811,
        swarm_coordination: bool = False,
    ) -> None:
        self.frame = frame
        self.search_center = search_center
        self.search_radius_m = float(search_radius_m)
        self.track_spacing_m = float(track_spacing_m)
        self.detection_radius_m = float(detection_radius_m)
        self.transit_speed_mps = float(transit_speed_mps)
        self.search_speed_mps = float(search_speed_mps)
        self.decision_interval_s = float(decision_interval_s)
        self.encounter_sample_count = int(encounter_sample_count)
        self.radial_shortlist_count = int(radial_shortlist_count)
        self.detection_probability = float(detection_probability)
        self.seed = int(seed)
        self.swarm_coordination = bool(swarm_coordination)
        self.assigned_radii: dict[int, float] = {}
        self.last_applied_s: dict[int, float] = {}
        self.last_scores: dict[int, float] = {}
        self.cached_candidates: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self.assigned_radii.clear()
        self.last_applied_s.clear()
        self.last_scores.clear()
        self.cached_candidates.clear()

    def evaluate(
        self,
        *,
        elapsed_s: float,
        revision: int,
        track_id: int,
        belief: IsotropicTargetParticleFilter,
        vehicles: list[dict[str, Any]],
    ) -> RHPPlanningDecision:
        eligible = [
            vehicle
            for vehicle in vehicles
            if bool(vehicle.get("mission_launched"))
            and not bool(vehicle.get("emergency_mode"))
            and str(vehicle.get("flight_phase")) == "ROUTE"
            and bool(vehicle.get("search_started", False))
        ]
        if not eligible:
            return RHPPlanningDecision({}, {})
        decision_due = any(
            int(vehicle.get("runtime_route_revision", 0)) == 0
            or float(elapsed_s)
            - self.last_applied_s.get(int(vehicle["vehicle_id"]), -1e9)
            >= self.decision_interval_s
            for vehicle in eligible
        )
        if not decision_due and self.cached_candidates:
            candidates = {}
            for vehicle in eligible:
                vehicle_id = int(vehicle["vehicle_id"])
                cached = self.cached_candidates.get(vehicle_id)
                if cached is None:
                    continue
                remaining_s = max(
                    0.0,
                    self.decision_interval_s
                    - (
                        float(elapsed_s)
                        - self.last_applied_s.get(vehicle_id, float(elapsed_s))
                    ),
                )
                candidates[vehicle_id] = {
                    **cached,
                    "action_commit_due": False,
                    "receding_horizon_refresh": False,
                    "decision_epoch_s": (
                        self.last_applied_s.get(
                            vehicle_id,
                            float(elapsed_s),
                        )
                        + self.decision_interval_s
                    ),
                    "seconds_until_decision": remaining_s,
                }
            return RHPPlanningDecision(candidates, {})
        radii = self._candidate_radii()
        actions: list[RHPRouteAction] = []
        for vehicle in eligible:
            vehicle_id = int(vehicle["vehicle_id"])
            position = self.frame.to_local(
                float(vehicle["latitude"]),
                float(vehicle["longitude"]),
            )
            # RHP-FE-PF-PW-ARC owns a fixed 60-degree sector per SAR.  The
            # common PF is global, but PW shortlisting is local to that SAR's
            # sector.  Cross-sector reassignment belongs to the separate
            # Global-Dynamic/CSA comparison model and must not happen here.
            radial_values = self._radial_values(
                belief,
                radii,
                vehicle_id=vehicle_id,
            )
            shortlisted = sorted(
                radii,
                key=lambda radius: (-radial_values[radius], radius),
            )[: self.radial_shortlist_count]
            current_radius = position.distance_to(self.search_center)
            committed_radius = self.assigned_radii.get(vehicle_id)
            nearest_radius = min(
                radii,
                key=lambda radius: abs(radius - current_radius),
            )
            if self.swarm_coordination:
                # A coordinated refresh must be allowed to move an SAR away from
                # its previous ARC track.  Restricting a vehicle in transit to
                # only ``committed_radius`` made two LMs reuse the same ring
                # when the belief peak changed.  Keep that radius as a valid
                # candidate for route continuity, but expose the complete
                # shortlist so the team-level marginal selector can reserve a
                # different ring for every vehicle.
                vehicle_radii = tuple(
                    dict.fromkeys(
                        [
                            *(
                                [committed_radius]
                                if committed_radius is not None
                                else []
                            ),
                            *shortlisted,
                            nearest_radius,
                        ]
                    )
                )
            elif (
                committed_radius is not None
                and abs(current_radius - committed_radius)
                > self.track_spacing_m / 2.0
            ):
                vehicle_radii = (committed_radius,)
            else:
                vehicle_radii = tuple(
                    dict.fromkeys(
                        [
                            *shortlisted,
                            nearest_radius,
                        ]
                    )
                )
            for radius in vehicle_radii:
                for direction in (-1, 1):
                    actions.append(
                        self._build_action(
                            vehicle_id,
                            track_id,
                            position,
                            radius,
                            direction,
                            radial_values.get(radius, 0.0),
                        )
                    )
        max_horizon_s = max(action.full_action_time_s for action in actions)
        forecast, escaped, sample_weight = belief.sampled_forecast(
            self.encounter_sample_count,
            max_horizon_s,
            seed=self._deterministic_seed(self.seed, revision, track_id),
        )
        scored = [
            self._score_action(action, forecast, escaped, sample_weight)
            for action in actions
        ]
        selected_actions, marginal_probabilities = (
            self._select_swarm_actions(scored, sample_weight)
            if self.swarm_coordination
            else (
                {
                    vehicle_id: max(
                        (
                            action
                            for action in scored
                            if action.vehicle_id == vehicle_id
                        ),
                        key=lambda action: (
                            action.score,
                            action.radial_probability,
                            action.detection_probability,
                            -action.transit_time_s,
                        ),
                    )
                    for vehicle_id in sorted(
                        {action.vehicle_id for action in scored}
                    )
                },
                {},
            )
        )
        candidates: dict[int, dict[str, Any]] = {}
        updates: dict[int, tuple[dict[str, Any], ...]] = {}
        for vehicle in eligible:
            vehicle_id = int(vehicle["vehicle_id"])
            best = selected_actions[vehicle_id]
            previous_score = self.last_scores.get(vehicle_id, 0.0)
            last_applied = self.last_applied_s.get(vehicle_id, -1e9)
            commit_due = (
                int(vehicle.get("runtime_route_revision", 0)) == 0
                or float(elapsed_s) - last_applied >= self.decision_interval_s
            )
            if vehicle_id not in self.last_applied_s:
                decision_epoch_s = float(elapsed_s)
            elif commit_due:
                elapsed_intervals = max(
                    1,
                    int(
                        math.floor(
                            (
                                float(elapsed_s)
                                - self.last_applied_s[vehicle_id]
                            )
                            / self.decision_interval_s
                            + 1e-9
                        )
                    ),
                )
                decision_epoch_s = (
                    self.last_applied_s[vehicle_id]
                    + elapsed_intervals * self.decision_interval_s
                )
            else:
                decision_epoch_s = (
                    self.last_applied_s[vehicle_id]
                    + self.decision_interval_s
                )
            payload = self._route_payload(best.execution_points)
            candidates[vehicle_id] = {
                "vehicle_id": vehicle_id,
                "track_id": track_id,
                "signature": best.signature,
                "radius_m": best.radius_m,
                "direction": "CCW" if best.direction > 0 else "CW",
                "detection_probability": best.detection_probability,
                "expected_detection_time_s": best.expected_detection_time_s,
                "score": best.score,
                "current_score": previous_score,
                "improvement_ratio": (
                    (best.score - previous_score) / max(abs(previous_score), 1e-12)
                    if previous_score > 0.0
                    # The first accepted route has no finite baseline.  Use a
                    # bounded 100% gain for telemetry instead of Infinity;
                    # strict JSON consumers (the Google 3D renderer) reject
                    # non-finite numeric literals and would stop repainting.
                    else 1.0 if best.score > 0.0 else 0.0
                ),
                "probability_delta": best.detection_probability,
                "swarm_marginal_probability": marginal_probabilities.get(
                    vehicle_id,
                    best.detection_probability,
                ),
                "swarm_coordinated": self.swarm_coordination,
                "confirmation_streak": 1,
                "confirmation_required": 1,
                "receding_horizon_refresh": commit_due,
                "evaluation_revision": revision,
                "decision_interval_s": self.decision_interval_s,
                "action_commit_due": commit_due,
                "decision_epoch_s": decision_epoch_s,
                "seconds_until_decision": (
                    0.0
                    if commit_due
                    else max(0.0, decision_epoch_s - float(elapsed_s))
                ),
                "particle_count": belief.particle_count,
                "encounter_sample_count": self.encounter_sample_count,
                "belief_representation": "particle-filter",
                "pf_configuration": PF_CONFIGURATION,
                "planning_scope": "LOCAL_FIXED_SECTOR",
                "sector_index": vehicle_id,
                "sector_center_deg": math.degrees(
                    self._sector_center_angle(vehicle_id)
                ),
                "sector_width_deg": LOCAL_SECTOR_WIDTH_DEG,
                "waypoints": list(payload),
            }
            if commit_due and payload:
                updates[vehicle_id] = payload
                self.assigned_radii[vehicle_id] = best.radius_m
                # Anchor to the nominal RHP epoch instead of the GUI polling
                # instant.  A 25-second research clock therefore never drifts
                # to 27, 54, 81... merely because telemetry is sampled every
                # few simulated seconds.
                self.last_applied_s[vehicle_id] = decision_epoch_s
                self.last_scores[vehicle_id] = best.score
            self.cached_candidates[vehicle_id] = dict(candidates[vehicle_id])
        return RHPPlanningDecision(candidates, updates)

    def _select_swarm_actions(
        self,
        actions: list[RHPRouteAction],
        sample_weight: float,
    ) -> tuple[dict[int, RHPRouteAction], dict[int, float]]:
        """Assign deconflicted actions by marginal swarm encounter gain.

        This ports the latest research planner's common-hit-mask selection and
        adds a deterministic route reservation.  A second SAR cannot take an
        already reserved radius while another radius remains available, which
        prevents six independently optimal vehicles from stacking on one ARC.
        """
        selected: dict[int, RHPRouteAction] = {}
        marginals: dict[int, float] = {}
        unassigned = {action.vehicle_id for action in actions}
        swarm_hits = np.zeros(self.encounter_sample_count, dtype=bool)
        reserved_radii: set[float] = set()
        while unassigned:
            best_action: RHPRouteAction | None = None
            best_key = (
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
                -math.inf,
            )
            best_marginal = 0.0
            for action in actions:
                if action.vehicle_id not in unassigned:
                    continue
                radius_key = round(action.radius_m, 3)
                vehicle_has_free_radius = any(
                    candidate.vehicle_id == action.vehicle_id
                    and round(candidate.radius_m, 3) not in reserved_radii
                    for candidate in actions
                )
                if radius_key in reserved_radii and vehicle_has_free_radius:
                    continue
                first_hits = action.first_hits_s
                if first_hits is None:
                    continue
                hit_mask = np.isfinite(first_hits)
                new_hits = hit_mask & ~swarm_hits
                marginal_probability = min(
                    1.0,
                    float(np.count_nonzero(new_hits))
                    * sample_weight
                    * self.detection_probability,
                )
                conditional_time = (
                    float(np.mean(first_hits[new_hits]))
                    if np.any(new_hits)
                    else action.full_action_time_s
                )
                expected_time = (
                    marginal_probability * conditional_time
                    + (1.0 - marginal_probability) * action.full_action_time_s
                )
                marginal_rate = (
                    marginal_probability / expected_time
                    if marginal_probability > 0.0 and expected_time > 0.0
                    else 0.0
                )
                route_separation = self._selected_route_separation(
                    action,
                    selected.values(),
                )
                key = (
                    marginal_rate,
                    action.radial_probability,
                    route_separation,
                    action.detection_probability,
                    -action.transit_time_s,
                )
                if key > best_key:
                    best_key = key
                    best_action = action
                    best_marginal = marginal_probability
            if best_action is None:
                raise RuntimeError("Unable to assign a coordinated RHP action")
            selected[best_action.vehicle_id] = best_action
            marginals[best_action.vehicle_id] = best_marginal
            reserved_radii.add(round(best_action.radius_m, 3))
            if best_action.first_hits_s is not None:
                swarm_hits |= np.isfinite(best_action.first_hits_s)
            unassigned.remove(best_action.vehicle_id)
        return selected, marginals

    @staticmethod
    def _selected_route_separation(
        action: RHPRouteAction,
        selected: Iterable[RHPRouteAction],
    ) -> float:
        selected_points = [
            point
            for selected_action in selected
            for point, _sensor_on in selected_action.execution_points
        ]
        if not selected_points:
            return math.inf
        candidate_points = [
            point for point, _sensor_on in action.execution_points
        ]
        return min(
            candidate.distance_to(other)
            for candidate in candidate_points
            for other in selected_points
        )

    def _candidate_radii(self) -> tuple[float, ...]:
        radii = []
        radius = self.track_spacing_m / 2.0
        while radius < self.search_radius_m:
            radii.append(radius)
            radius += self.track_spacing_m
        if not radii:
            radii.append(self.search_radius_m / 2.0)
        return tuple(radii)

    def _radial_values(
        self,
        belief: IsotropicTargetParticleFilter,
        radii: tuple[float, ...],
        *,
        vehicle_id: int | None = None,
    ) -> dict[float, float]:
        distance = np.hypot(
            belief.east_m - self.search_center.east_m,
            belief.north_m - self.search_center.north_m,
        )
        active_weights = np.where(belief.escaped, 0.0, belief.weights)
        if vehicle_id is not None:
            center_angle = self._sector_center_angle(vehicle_id)
            particle_angles = np.arctan2(
                belief.north_m - self.search_center.north_m,
                belief.east_m - self.search_center.east_m,
            )
            relative_angles = (
                particle_angles - center_angle + math.pi
            ) % math.tau - math.pi
            active_weights = np.where(
                np.abs(relative_angles) <= math.pi / LOCAL_SECTOR_COUNT,
                active_weights,
                0.0,
            )
        return {
            radius: float(
                np.sum(
                    active_weights[
                        np.abs(distance - radius) <= self.detection_radius_m
                    ]
                )
            )
            for radius in radii
        }

    def _build_action(
        self,
        vehicle_id: int,
        track_id: int,
        position: LocalPoint,
        radius_m: float,
        direction: int,
        radial_probability: float,
    ) -> RHPRouteAction:
        start_angle = self._local_start_angle(position, vehicle_id)
        scan_start = self._polar_point(radius_m, start_angle)
        transit_distance = position.distance_to(scan_start)
        transit_time_s = transit_distance / self.transit_speed_mps
        planned_scan_points = self._arc_points(
            radius_m,
            start_angle,
            direction,
            self.search_speed_mps * self.decision_interval_s,
            vehicle_id,
        )
        execution: list[tuple[LocalPoint, bool]] = []
        if transit_time_s >= self.decision_interval_s - 1e-12:
            committed_fraction = min(
                self.transit_speed_mps * self.decision_interval_s
                / max(transit_distance, 1e-12),
                1.0,
            )
            # Keep the same 25-second straight action prefix, but expose three
            # time-ordered execution points.  This gives the operator an
            # observable 1/3, 2/3 and endpoint at every RHP revision without
            # changing distance, speed, probability scoring, or trajectory.
            for sample_fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0):
                fraction = committed_fraction * sample_fraction
                execution.append(
                    (
                        LocalPoint(
                            position.east_m
                            + (scan_start.east_m - position.east_m) * fraction,
                            position.north_m
                            + (scan_start.north_m - position.north_m) * fraction,
                        ),
                        False,
                    )
                )
        else:
            if transit_distance > 1e-6:
                execution.append((scan_start, False))
            scan_distance = self.search_speed_mps * (
                self.decision_interval_s - transit_time_s
            )
            executed_scan = self._arc_points(
                radius_m,
                start_angle,
                direction,
                scan_distance,
                vehicle_id,
            )
            execution.extend((point, True) for point in executed_scan[1:])
        signature = (
            f"RHP-T{track_id}-V{vehicle_id}-R{radius_m:.1f}-D{direction:+d}"
        )
        return RHPRouteAction(
            vehicle_id,
            radius_m,
            direction,
            signature,
            position,
            planned_scan_points,
            tuple(execution),
            transit_time_s,
            transit_time_s + self.decision_interval_s,
            radial_probability,
        )

    def _score_action(
        self,
        action: RHPRouteAction,
        forecast: np.ndarray,
        escaped: np.ndarray,
        sample_weight: float,
    ) -> RHPRouteAction:
        first_hits = np.full(forecast.shape[1], np.inf, dtype=float)
        scan_length = polyline_length(action.scan_points)
        full_time_s = action.transit_time_s + scan_length / self.search_speed_mps
        start_step = max(0, int(math.floor(action.transit_time_s)))
        end_step = min(int(math.ceil(full_time_s)), forecast.shape[0] - 1)
        steps = np.arange(start_step, end_step + 1, dtype=np.int64)
        times = steps.astype(float)
        valid = times >= action.transit_time_s
        steps = steps[valid]
        times = times[valid]
        if times.size:
            scan_distances = np.minimum(
                scan_length,
                (times - action.transit_time_s) * self.search_speed_mps,
            )
            scan_coordinates = np.asarray(
                [
                    (point.east_m, point.north_m)
                    for point in action.scan_points
                ],
                dtype=float,
            )
            segment_vectors = np.diff(scan_coordinates, axis=0)
            segment_lengths = np.hypot(
                segment_vectors[:, 0],
                segment_vectors[:, 1],
            )
            cumulative_lengths = np.concatenate(
                ([0.0], np.cumsum(segment_lengths))
            )
            segment_indices = np.searchsorted(
                cumulative_lengths,
                scan_distances,
                side="right",
            ) - 1
            segment_indices = np.clip(
                segment_indices,
                0,
                len(segment_lengths) - 1,
            )
            local_distances = (
                scan_distances - cumulative_lengths[segment_indices]
            )
            fractions = local_distances / np.maximum(
                segment_lengths[segment_indices],
                1e-12,
            )
            vehicle_positions = (
                scan_coordinates[segment_indices]
                + segment_vectors[segment_indices] * fractions[:, None]
            )
            delta_east = (
                forecast[steps, :, 0] - vehicle_positions[:, None, 0]
            )
            delta_north = (
                forecast[steps, :, 1] - vehicle_positions[:, None, 1]
            )
            hit_matrix = (
                ~escaped[steps]
                & (
                    delta_east * delta_east + delta_north * delta_north
                    <= self.detection_radius_m**2
                )
            )
            hit_mask = np.any(hit_matrix, axis=0)
            if np.any(hit_mask):
                first_indices = np.argmax(hit_matrix[:, hit_mask], axis=0)
                first_hits[hit_mask] = times[first_indices]
        hit_mask = np.isfinite(first_hits)
        probability = min(
            1.0,
            float(np.count_nonzero(hit_mask))
            * sample_weight
            * self.detection_probability,
        )
        conditional_time = (
            float(np.mean(first_hits[hit_mask])) if np.any(hit_mask) else full_time_s
        )
        expected_time = (
            probability * conditional_time + (1.0 - probability) * full_time_s
        )
        score = probability / expected_time if probability > 0.0 else 0.0
        return RHPRouteAction(
            action.vehicle_id,
            action.radius_m,
            action.direction,
            action.signature,
            action.previous_position,
            action.scan_points,
            action.execution_points,
            action.transit_time_s,
            full_time_s,
            action.radial_probability,
            probability,
            conditional_time,
            score,
            first_hits,
        )

    def _arc_points(
        self,
        radius_m: float,
        start_angle: float,
        direction: int,
        scan_distance_m: float,
        vehicle_id: int,
    ) -> tuple[LocalPoint, ...]:
        if scan_distance_m <= 1e-9:
            point = self._polar_point(radius_m, start_angle)
            return (point, point)
        subdivisions = max(
            1,
            int(
                math.ceil(
                    scan_distance_m / max(self.track_spacing_m / 2.0, 1.0)
                )
            ),
        )
        sector_center = self._sector_center_angle(vehicle_id)
        half_width = math.pi / LOCAL_SECTOR_COUNT
        lower_bound = sector_center - half_width
        upper_bound = sector_center + half_width
        angle = self._clamp_angle_to_sector(
            start_angle,
            sector_center,
        )
        travel_direction = 1 if direction >= 0 else -1
        angular_step = scan_distance_m / subdivisions / max(radius_m, 1e-9)
        points = [self._polar_point(radius_m, angle)]
        for _ in range(subdivisions):
            remaining = angular_step
            while remaining > 1e-12:
                boundary = upper_bound if travel_direction > 0 else lower_bound
                available = abs(boundary - angle)
                if available <= 1e-12:
                    travel_direction *= -1
                    continue
                travelled = min(available, remaining)
                angle += travel_direction * travelled
                remaining -= travelled
                if remaining > 1e-12:
                    travel_direction *= -1
            points.append(self._polar_point(radius_m, angle))
        return tuple(points)

    @staticmethod
    def _sector_center_angle(vehicle_id: int) -> float:
        return math.tau * (int(vehicle_id) - 1) / LOCAL_SECTOR_COUNT

    @staticmethod
    def _relative_angle(angle: float, center: float) -> float:
        return (angle - center + math.pi) % math.tau - math.pi

    def _clamp_angle_to_sector(self, angle: float, center: float) -> float:
        half_width = math.pi / LOCAL_SECTOR_COUNT
        relative = self._relative_angle(angle, center)
        return center + max(-half_width, min(half_width, relative))

    def _local_start_angle(
        self,
        position: LocalPoint,
        vehicle_id: int,
    ) -> float:
        sector_center = self._sector_center_angle(vehicle_id)
        delta_east = position.east_m - self.search_center.east_m
        delta_north = position.north_m - self.search_center.north_m
        if delta_east * delta_east + delta_north * delta_north <= 1e-12:
            return sector_center
        return self._clamp_angle_to_sector(
            math.atan2(delta_north, delta_east),
            sector_center,
        )

    def _polar_point(self, radius_m: float, angle_rad: float) -> LocalPoint:
        return LocalPoint(
            self.search_center.east_m + radius_m * math.cos(angle_rad),
            self.search_center.north_m + radius_m * math.sin(angle_rad),
        )

    def _route_payload(
        self,
        points: tuple[tuple[LocalPoint, bool], ...],
    ) -> tuple[dict[str, Any], ...]:
        payload = []
        for sequence, (point, sensor_on) in enumerate(points, start=1):
            latitude, longitude = self.frame.to_geographic(point)
            phase_code = "S" if sensor_on else "T"
            payload.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_m": 600.0,
                    "code": f"RHP-{phase_code}{sequence:02d}",
                    "label": (
                        f"RHP SCAN {sequence}"
                        if sensor_on
                        else f"RHP TRANSIT {sequence}"
                    ),
                    "point_type": (
                        "RHP_SCAN_WAYPOINT"
                        if sensor_on
                        else "RHP_TRANSIT_WAYPOINT"
                    ),
                    "sequence": sequence,
                }
            )
        return tuple(payload)

    @staticmethod
    def _deterministic_seed(*values: int) -> int:
        seed = 0x3456_789A
        for value in values:
            seed = (seed * 1_000_003 + int(value)) & 0xFFFF_FFFF
        return seed
