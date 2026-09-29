from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .geometry import LocalPoint
from .sensor_model import SensorFootprint


CV_MODE = 0
CTRV_MODE = 1
RANDOM_MODE = 2
MODE_NAMES = ("CV", "CTRV", "RANDOM")


@dataclass(slots=True)
class IMMParticle:
    east_m: float
    north_m: float
    speed_mps: float
    heading_rad: float
    turn_rate_rad_s: float
    mode: int
    weight: float

    def copy(self) -> "IMMParticle":
        return IMMParticle(
            self.east_m,
            self.north_m,
            self.speed_mps,
            self.heading_rad,
            self.turn_rate_rad_s,
            self.mode,
            self.weight,
        )


@dataclass(frozen=True, slots=True)
class BeliefSummary:
    mean_east_m: float
    mean_north_m: float
    mean_speed_mps: float
    mean_heading_deg: float
    uncertainty_east_95_m: float
    uncertainty_north_95_m: float
    effective_particle_count: float
    mode_probabilities: tuple[float, float, float]
    revision: int


@dataclass(frozen=True, slots=True)
class ForecastPath:
    weight: float
    times_s: tuple[float, ...]
    points: tuple[LocalPoint, ...]


class IMMParticleFilter:
    """Time-scaled Markov-jump particle filter for a moving ground subject.

    The reference transition matrix and process noise are defined at 30 s.
    Smaller runtime updates scale transition probability and standard deviation
    so a 1 s GCS loop does not accidentally make the subject thirty times more
    manoeuvrable.
    """

    BASE_STEP_S = 30.0
    BASE_TRANSITION = (
        (0.92, 0.06, 0.02),
        (0.08, 0.88, 0.04),
        (0.25, 0.15, 0.60),
    )

    def __init__(
        self,
        initial_position: LocalPoint,
        *,
        initial_speed_mps: float,
        initial_heading_deg: float,
        maximum_speed_mps: float = 90_000.0 / 3_600.0,
        particle_count: int = 360,
        initial_position_std_m: float = 450.0,
        seed: int = 20_260_809,
        resample_threshold_ratio: float = 0.5,
    ) -> None:
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if maximum_speed_mps <= 0.0:
            raise ValueError("maximum_speed_mps must be positive")
        self.maximum_speed_mps = float(maximum_speed_mps)
        self.particle_count = int(particle_count)
        self.resample_threshold_ratio = float(resample_threshold_ratio)
        self.rng = random.Random(seed)
        self.revision = 0
        heading_rad = math.radians(float(initial_heading_deg))
        mode_probabilities = (0.55, 0.35, 0.10)
        self.particles: list[IMMParticle] = []
        for _ in range(self.particle_count):
            mode = self._weighted_choice(mode_probabilities)
            speed = max(
                0.0,
                min(
                    self.maximum_speed_mps,
                    self.rng.gauss(float(initial_speed_mps), 1.5),
                ),
            )
            turn_rate = (
                math.radians(self.rng.gauss(0.0, 1.2))
                if mode == CTRV_MODE
                else 0.0
            )
            self.particles.append(
                IMMParticle(
                    self.rng.gauss(
                        initial_position.east_m,
                        initial_position_std_m,
                    ),
                    self.rng.gauss(
                        initial_position.north_m,
                        initial_position_std_m,
                    ),
                    speed,
                    (heading_rad + math.radians(self.rng.gauss(0.0, 8.0)))
                    % math.tau,
                    turn_rate,
                    mode,
                    1.0 / self.particle_count,
                )
            )

    def _weighted_choice(self, probabilities: tuple[float, ...]) -> int:
        threshold = self.rng.random()
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if threshold <= cumulative:
                return index
        return len(probabilities) - 1

    @classmethod
    def _scaled_transition(
        cls,
        mode: int,
        elapsed_s: float,
    ) -> tuple[float, float, float]:
        row = cls.BASE_TRANSITION[mode]
        scale = max(0.0, min(float(elapsed_s) / cls.BASE_STEP_S, 1.0))
        probabilities = [value * scale for value in row]
        probabilities[mode] = 1.0 - sum(
            probabilities[index]
            for index in range(len(probabilities))
            if index != mode
        )
        return tuple(probabilities)  # type: ignore[return-value]

    def _propagate_particle(
        self,
        particle: IMMParticle,
        elapsed_s: float,
        rng: random.Random,
    ) -> None:
        if elapsed_s <= 0.0:
            return
        transition = self._scaled_transition(particle.mode, elapsed_s)
        threshold = rng.random()
        cumulative = 0.0
        mode = particle.mode
        for candidate_mode, probability in enumerate(transition):
            cumulative += probability
            if threshold <= cumulative:
                mode = candidate_mode
                break

        noise_scale = math.sqrt(elapsed_s / self.BASE_STEP_S)
        particle.speed_mps = max(
            0.0,
            min(
                self.maximum_speed_mps,
                particle.speed_mps
                + rng.gauss(
                    0.0,
                    0.04 * self.maximum_speed_mps * noise_scale,
                ),
            ),
        )
        heading_noise = math.radians(rng.gauss(0.0, 2.0 * noise_scale))
        maximum_turn_rate = math.radians(12.0)
        if mode == CV_MODE:
            particle.turn_rate_rad_s *= math.pow(0.25, elapsed_s / 30.0)
            particle.heading_rad += (
                particle.turn_rate_rad_s * elapsed_s + heading_noise
            )
        elif mode == CTRV_MODE:
            particle.turn_rate_rad_s = max(
                -maximum_turn_rate,
                min(
                    maximum_turn_rate,
                    particle.turn_rate_rad_s
                    + math.radians(rng.gauss(0.0, 0.6 * noise_scale)),
                ),
            )
            particle.heading_rad += (
                particle.turn_rate_rad_s * elapsed_s + heading_noise
            )
        else:
            particle.turn_rate_rad_s = 0.0
            particle.heading_rad += math.radians(
                rng.gauss(0.0, 35.0 * noise_scale)
            )
        particle.heading_rad %= math.tau
        particle.east_m += (
            math.sin(particle.heading_rad)
            * particle.speed_mps
            * elapsed_s
        )
        particle.north_m += (
            math.cos(particle.heading_rad)
            * particle.speed_mps
            * elapsed_s
        )
        particle.mode = mode

    def predict(self, elapsed_s: float) -> None:
        remaining = max(0.0, float(elapsed_s))
        while remaining > 1e-9:
            delta_s = min(1.0, remaining)
            for particle in self.particles:
                self._propagate_particle(particle, delta_s, self.rng)
            remaining -= delta_s
        if elapsed_s > 0.0:
            self.revision += 1

    def observe_position(
        self,
        measurement: LocalPoint,
        *,
        measurement_std_m: float,
    ) -> None:
        variance = max(1.0, float(measurement_std_m)) ** 2
        likelihoods = []
        for particle in self.particles:
            dx = particle.east_m - measurement.east_m
            dy = particle.north_m - measurement.north_m
            likelihoods.append(math.exp(-0.5 * (dx * dx + dy * dy) / variance))
        maximum = max(likelihoods, default=0.0)
        if maximum <= 1e-300:
            return
        for particle, likelihood in zip(self.particles, likelihoods):
            particle.weight *= likelihood / maximum
        self._normalise_and_resample()
        self.revision += 1

    def observe_no_detection(
        self,
        footprints: list[SensorFootprint],
    ) -> None:
        if not footprints:
            return
        changed = False
        for particle in self.particles:
            position = LocalPoint(particle.east_m, particle.north_m)
            missed_probability = 1.0
            for footprint in footprints:
                detection_probability = (
                    footprint.detection_probability_at(position)
                )
                missed_probability *= 1.0 - detection_probability
            if missed_probability < 1.0:
                particle.weight *= missed_probability
                changed = True
        if changed:
            self._normalise_and_resample()
            self.revision += 1

    def _normalise_and_resample(self) -> None:
        total = sum(particle.weight for particle in self.particles)
        if total <= 1e-300:
            uniform = 1.0 / self.particle_count
            for particle in self.particles:
                particle.weight = uniform
            return
        for particle in self.particles:
            particle.weight /= total
        if self.effective_particle_count < (
            self.resample_threshold_ratio * self.particle_count
        ):
            self._systematic_resample()

    @property
    def effective_particle_count(self) -> float:
        squared = sum(
            particle.weight * particle.weight
            for particle in self.particles
        )
        return 1.0 / squared if squared > 0.0 else 0.0

    def _systematic_resample(self) -> None:
        start = self.rng.random() / self.particle_count
        positions = [
            start + index / self.particle_count
            for index in range(self.particle_count)
        ]
        cumulative = self.particles[0].weight
        source_index = 0
        selected: list[IMMParticle] = []
        for position in positions:
            while position > cumulative and source_index < self.particle_count - 1:
                source_index += 1
                cumulative += self.particles[source_index].weight
            selected.append(self.particles[source_index].copy())
        uniform = 1.0 / self.particle_count
        for particle in selected:
            particle.weight = uniform
        self.particles = selected

    def summary(self) -> BeliefSummary:
        total = sum(particle.weight for particle in self.particles)
        if total <= 0.0:
            total = 1.0
        mean_east = sum(p.east_m * p.weight for p in self.particles) / total
        mean_north = sum(p.north_m * p.weight for p in self.particles) / total
        mean_speed = sum(p.speed_mps * p.weight for p in self.particles) / total
        heading_x = sum(
            math.sin(p.heading_rad) * p.weight for p in self.particles
        ) / total
        heading_y = sum(
            math.cos(p.heading_rad) * p.weight for p in self.particles
        ) / total
        variance_east = sum(
            (p.east_m - mean_east) ** 2 * p.weight
            for p in self.particles
        ) / total
        variance_north = sum(
            (p.north_m - mean_north) ** 2 * p.weight
            for p in self.particles
        ) / total
        mode_probabilities = [0.0, 0.0, 0.0]
        for particle in self.particles:
            mode_probabilities[particle.mode] += particle.weight / total
        confidence_scale = math.sqrt(5.991)
        return BeliefSummary(
            mean_east,
            mean_north,
            mean_speed,
            (
                math.degrees(math.atan2(heading_x, heading_y)) + 360.0
            ) % 360.0,
            confidence_scale * math.sqrt(max(0.0, variance_east)),
            confidence_scale * math.sqrt(max(0.0, variance_north)),
            self.effective_particle_count,
            tuple(mode_probabilities),  # type: ignore[arg-type]
            self.revision,
        )

    def forecast(
        self,
        horizon_s: float,
        *,
        step_s: float = 5.0,
        maximum_paths: int = 180,
        seed: int | None = None,
    ) -> tuple[ForecastPath, ...]:
        if horizon_s <= 0.0 or step_s <= 0.0:
            raise ValueError("forecast horizon and step must be positive")
        path_count = max(1, min(maximum_paths, len(self.particles)))
        total_weight = sum(particle.weight for particle in self.particles)
        if path_count == len(self.particles):
            selected = list(self.particles)
            selected_weights = [
                particle.weight / total_weight
                if total_weight > 0.0
                else 1.0 / path_count
                for particle in selected
            ]
        else:
            # Deterministic systematic sampling remains representative even
            # immediately after resampling, when every particle has the same
            # weight and a simple top-N slice would be biased by list order.
            spacing = total_weight / path_count if total_weight > 0.0 else 0.0
            thresholds = [
                (index + 0.5) * spacing for index in range(path_count)
            ]
            selected = []
            source_index = 0
            cumulative = self.particles[0].weight
            for threshold in thresholds:
                while (
                    threshold > cumulative
                    and source_index < len(self.particles) - 1
                ):
                    source_index += 1
                    cumulative += self.particles[source_index].weight
                selected.append(self.particles[source_index])
            selected_weights = [1.0 / path_count] * path_count
        rng_seed = (
            seed
            if seed is not None
            else 20_260_809 + self.revision * 1_009
        )
        paths: list[ForecastPath] = []
        for index, (source, path_weight) in enumerate(
            zip(selected, selected_weights)
        ):
            particle = source.copy()
            rng = random.Random(rng_seed + index * 7_919)
            times = [0.0]
            points = [LocalPoint(particle.east_m, particle.north_m)]
            elapsed = 0.0
            while elapsed < horizon_s - 1e-9:
                delta_s = min(step_s, horizon_s - elapsed)
                self._propagate_particle(particle, delta_s, rng)
                elapsed += delta_s
                times.append(elapsed)
                points.append(LocalPoint(particle.east_m, particle.north_m))
            paths.append(
                ForecastPath(
                    path_weight,
                    tuple(times),
                    tuple(points),
                )
            )
        return tuple(paths)
