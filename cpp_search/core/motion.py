"""IMM5 표적 운동모델과 Monte Carlo 진리 궤적.

IMM5는 비행체도 무기도 계획기도 아니다. **표적이 어떻게 움직이는가**에
대한 5-모드 가정이다.

모드: 0 HALT / 1 LOW_CV / 2 HIGH_CV / 3 CTRV / 4 MANEUVER

핵심 수식
---------
* 모드 전이행렬 (``TargetBehaviorProfile.transition_matrix``)

      P = alpha * I + (1 - alpha) * 1 pi^T

  alpha = ``persistence_alpha`` (현재 모드를 유지할 성향),
  pi = ``mode_probabilities`` (정상 상태 점유 벡터).
  이 형태는 정상분포가 정확히 pi가 되도록 만든 것이다.

* CV 전파 (LOW_CV / HIGH_CV)

      x <- x + v * cos(psi) * dt
      y <- y + v * sin(psi) * dt

* CTRV 전파 (선회율 omega가 0이 아닐 때)

      psi' = psi + omega * dt
      x <- x + (v/omega) * ( sin(psi') - sin(psi) )
      y <- y + (v/omega) * ( -cos(psi') + cos(psi) )

  omega -> 0이면 CV 식으로 수렴한다.

* 경계 처리 (``apply_circular_boundary``)
  탐색원을 벗어난 입자/표적은 흡수 상태(escaped)로 표시한다. 다시
  정규화해서 원 안으로 되돌리지 않는다 — 이탈 확률을 실제로 세기 위해서다.

의존
----
* 위: ``models``.
* 아래: ``particle_filter``(예측 커널), ``simulation``(진리 궤적 생성),
  ``profiles``(missing-person 성인/아동에 거동 프로파일을 묶음).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt, tau
from random import Random

from cpp_search.core.models import MissionConfig, Point2D


@dataclass(frozen=True, slots=True)
class TargetBehaviorProfile:
    """Reproducible five-mode ground-target stress scenario.

    All numeric values are uncertainty-set design choices, not operational
    estimates. ``persistence_alpha`` controls how likely the target is to
    retain its current mode at the next nominal transition. The remaining
    probability is distributed according to the profile occupancy vector.
    Noise and turn-rate scales deliberately separate the truth scenarios from
    the planner's aggregate nominal kernel.
    """

    name: str
    target_class: str
    description: str
    mode_probabilities: tuple[float, float, float, float, float]
    persistence_alpha: float = 0.5
    persistence_reference_s: float = 30.0
    process_noise_reference_s: float = 30.0
    process_noise_scale: float = 1.0
    turn_rate_scale: float = 1.0
    #: 모드별 속도구간 [km/h]. 각 모드 안에서 균등분포로 뽑는다.
    #: 여기 값이 앙상블 평균속도를 결정한다 — 최대속도 40 km/h를 선언해도
    #: 점유율 가중 평균은 그보다 훨씬 낮다. 검증 대상 가정이므로
    #: ``config/chapter0_common_experiment.json``에서 덮어쓸 수 있다.
    mode_speed_bounds_kph: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),    # HALT
        (0.0, 20.0),   # LOW_CV
        (20.0, 40.0),  # HIGH_CV
        (2.0, 32.0),   # CTRV
        (0.0, 40.0),   # MANEUVER
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty")
        if self.target_class not in {"GROUND", "MIXED"}:
            raise ValueError("target_class must be GROUND or MIXED")
        if any(value < 0.0 for value in self.mode_probabilities):
            raise ValueError("profile mode probabilities must not be negative")
        if abs(sum(self.mode_probabilities) - 1.0) > 1e-9:
            raise ValueError("profile mode probabilities must sum to one")
        if not 0.0 <= self.persistence_alpha < 1.0:
            raise ValueError("persistence_alpha must be in [0, 1)")
        if self.persistence_reference_s <= 0.0:
            raise ValueError("persistence_reference_s must be positive")
        if self.process_noise_reference_s <= 0.0:
            raise ValueError("process_noise_reference_s must be positive")
        if self.process_noise_scale <= 0.0:
            raise ValueError("process_noise_scale must be positive")
        if self.turn_rate_scale <= 0.0:
            raise ValueError("turn_rate_scale must be positive")
        if len(self.mode_speed_bounds_kph) != len(self.mode_probabilities):
            raise ValueError("one speed band is required per motion mode")
        for low, high in self.mode_speed_bounds_kph:
            if low < 0.0 or high < low:
                raise ValueError("mode speed bands must satisfy 0 <= low <= high")

    @property
    def mean_speed_kph(self) -> float:
        """점유율로 가중한 앙상블 평균속도.

        모드마다 구간 균등분포이므로 평균은 (low+high)/2 이고, 여기에 정상상태
        점유율 pi를 가중한다. AOI를 최대속도로 잡을지 이 값으로 잡을지가
        탐색영역 설계의 갈림길이다.
        """

        return sum(
            probability * 0.5 * (low + high)
            for probability, (low, high) in zip(
                self.mode_probabilities, self.mode_speed_bounds_kph
            )
        )

    @property
    def mode_speed_ranges_kph(self) -> tuple[tuple[float, float], ...]:
        """Compatibility name for the configured per-mode speed bounds."""

        return self.mode_speed_bounds_kph

    def speed_range_kph(self, mode: int) -> tuple[float, float]:
        return self.mode_speed_bounds_kph[mode]

    @property
    def transition_matrix(self) -> tuple[tuple[float, ...], ...]:
        """모드 전이행렬 P = alpha * I + (1 - alpha) * 1 pi^T.

        alpha는 현재 모드를 유지할 성향, pi는 정상상태 점유 벡터다.
        이 형태를 쓰면 P의 정상분포가 정확히 pi가 된다 (pi P = pi).
        """

        alpha = self.persistence_alpha
        return tuple(
            tuple(
                alpha * float(row == column)
                + (1.0 - alpha) * probability
                for column, probability in enumerate(self.mode_probabilities)
            )
            for row in range(len(self.mode_probabilities))
        )


HALT_MODE = 0
LOW_CV_MODE = 1
HIGH_CV_MODE = 2
GROUND_CTRV_MODE = 3
GROUND_MANEUVER_MODE = 4
GROUND_MOTION_MODE_NAMES = ("HALT", "LOW_CV", "HIGH_CV", "CTRV", "MANEUVER")

# Compatibility aliases for callers written against the original experiment.
STOP_MODE = HALT_MODE
GROUND_RANDOM_MODE = GROUND_MANEUVER_MODE


STOP_HEAVY_PROFILE = TargetBehaviorProfile(
    "STOP_HEAVY",
    "GROUND",
    "Intermittent movement with extended halts",
    (0.60, 0.15, 0.10, 0.10, 0.05),
    persistence_alpha=0.70,
    process_noise_scale=0.75,
    turn_rate_scale=0.75,
)
RELOCATION_HEAVY_PROFILE = TargetBehaviorProfile(
    "RELOCATION_HEAVY",
    "GROUND",
    "Sustained faster relocation with occasional turns and halts",
    (0.05, 0.10, 0.65, 0.15, 0.05),
    persistence_alpha=0.65,
    process_noise_scale=0.75,
    turn_rate_scale=0.75,
)
CONTINUOUS_MOVE_PROFILE = TargetBehaviorProfile(
    "CONTINUOUS_MOVE",
    "GROUND",
    "Persistent movement across lower and higher straight-line speeds",
    (0.05, 0.35, 0.40, 0.15, 0.05),
    persistence_alpha=0.75,
)
MANEUVER_HEAVY_PROFILE = TargetBehaviorProfile(
    "MANEUVER_HEAVY",
    "GROUND",
    "Frequent turning and irregular but temporally correlated movement",
    (0.15, 0.15, 0.10, 0.35, 0.25),
    persistence_alpha=0.45,
    process_noise_scale=1.75,
    turn_rate_scale=1.50,
)
MIXED_GROUND_PROFILE = TargetBehaviorProfile(
    "MIXED_GROUND",
    "MIXED",
    "Aggregate nominal prior over four ground-motion stress scenarios",
    (0.2125, 0.1875, 0.3125, 0.1875, 0.10),
    persistence_alpha=0.6375,
    process_noise_scale=1.0625,
)
GROUND_TRUTH_SCENARIOS = (
    STOP_HEAVY_PROFILE,
    RELOCATION_HEAVY_PROFILE,
    CONTINUOUS_MOVE_PROFILE,
    MANEUVER_HEAVY_PROFILE,
)

# Backward-compatible collection name. New code should use the scenario name.
GROUND_TARGET_PROFILES = GROUND_TRUTH_SCENARIOS


@dataclass(frozen=True, slots=True)
class TargetMotionSpec:
    """Target transition model shared by truth generation and Bayesian filters.

    ``isotropic`` reproduces the original position-only random-heading model.
    ``imm`` is the legacy Markov-jump CV/CTRV/random-maneuver model.
    ``imm5`` adds HALT/low-CV/high-CV/CTRV/maneuver modes for the generic
    ground-target stress scenarios. Both retain state
    ``(x, y, speed, heading, turn_rate, mode)``.
    """

    max_speed_mps: float
    step_s: float = 30.0
    nominal_speed_ratio: float = 0.65
    speed_sigma_ratio: float = 0.15
    prediction_direction_count: int = 16
    motion_model: str = "isotropic"
    speed_process_sigma_ratio: float = 0.04
    heading_process_sigma_deg: float = 2.0
    turn_rate_process_sigma_dps: float = 0.6
    max_turn_rate_dps: float = 12.0
    initial_mode_probabilities: tuple[float, float, float] = (0.55, 0.35, 0.10)
    #: ``escape``  원 밖을 흡수 상태로 표시 (입자필터 belief 용)
    #: ``reflect`` 원 경계에서 반사
    #: ``open``    경계를 두지 않고 그대로 전파 (진리 궤적 관측용)
    #:
    #: 진리 생성에 ``escape``를 쓰면 원을 벗어나는 순간 궤적이 끊겨서,
    #: "표적이 실제로 어디까지 가는가"를 관측할 수 없다. 그 관측이 AOI 크기를
    #: 정하는 근거이므로 판정(원 밖 = 실패)과 관측을 분리한다.
    boundary_mode: str = "escape"
    behavior_profile: TargetBehaviorProfile | None = None

    def __post_init__(self) -> None:
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be positive")
        if self.step_s <= 0.0:
            raise ValueError("step_s must be positive")
        if not 0.0 <= self.nominal_speed_ratio <= 1.0:
            raise ValueError("nominal_speed_ratio must be in [0, 1]")
        if self.speed_sigma_ratio < 0.0:
            raise ValueError("speed_sigma_ratio must not be negative")
        if self.prediction_direction_count < 4:
            raise ValueError("prediction_direction_count must be at least 4")
        if self.motion_model not in {"isotropic", "imm", "imm5"}:
            raise ValueError("motion_model must be isotropic, imm, or imm5")
        if self.speed_process_sigma_ratio < 0.0:
            raise ValueError("speed_process_sigma_ratio must not be negative")
        if self.heading_process_sigma_deg < 0.0:
            raise ValueError("heading_process_sigma_deg must not be negative")
        if self.turn_rate_process_sigma_dps < 0.0:
            raise ValueError("turn_rate_process_sigma_dps must not be negative")
        if self.max_turn_rate_dps <= 0.0:
            raise ValueError("max_turn_rate_dps must be positive")
        if len(self.initial_mode_probabilities) != 3:
            raise ValueError("initial_mode_probabilities must contain CV/CTRV/random")
        if any(value < 0.0 for value in self.initial_mode_probabilities):
            raise ValueError("initial mode probabilities must not be negative")
        if abs(sum(self.initial_mode_probabilities) - 1.0) > 1e-9:
            raise ValueError("initial mode probabilities must sum to one")
        if self.boundary_mode not in {"escape", "reflect", "open"}:
            raise ValueError("boundary_mode must be 'escape', 'reflect', or 'open'")
        if self.motion_model == "imm5" and self.behavior_profile is None:
            raise ValueError("motion_model='imm5' requires a behavior_profile")
        if self.motion_model != "imm5" and self.behavior_profile is not None:
            raise ValueError("behavior_profile is only valid for motion_model='imm5'")

    @property
    def nominal_speed_mps(self) -> float:
        return self.nominal_speed_ratio * self.max_speed_mps

    @property
    def ground_process_noise_scale(self) -> float:
        if self.motion_model != "imm5":
            return 1.0
        assert self.behavior_profile is not None
        return self.behavior_profile.process_noise_scale

    @property
    def ground_turn_rate_scale(self) -> float:
        if self.motion_model != "imm5":
            return 1.0
        assert self.behavior_profile is not None
        return self.behavior_profile.turn_rate_scale

    @property
    def mode_names(self) -> tuple[str, ...]:
        return (
            GROUND_MOTION_MODE_NAMES
            if self.motion_model == "imm5"
            else MOTION_MODE_NAMES
        )

    @property
    def effective_initial_mode_probabilities(self) -> tuple[float, ...]:
        if self.motion_model == "imm5":
            assert self.behavior_profile is not None
            return self.behavior_profile.mode_probabilities
        return self.initial_mode_probabilities

    @property
    def effective_transition_matrix(self) -> tuple[tuple[float, ...], ...]:
        if self.motion_model == "imm5":
            assert self.behavior_profile is not None
            return self.behavior_profile.transition_matrix
        return (
            (0.92, 0.06, 0.02),
            (0.08, 0.88, 0.04),
            (0.25, 0.15, 0.60),
        )

    def transition_matrix_for_elapsed(
        self,
        elapsed_s: float,
    ) -> tuple[tuple[float, ...], ...]:
        """Scale a per-step mode transition to an arbitrary elapsed time.

        This keeps the target-mode clock independent of the planner update
        interval.  In particular, a 1 s planner update must not apply the
        nominal 30 s transition matrix once every second.
        """

        if elapsed_s <= 0.0:
            size = len(self.mode_names)
            return tuple(
                tuple(float(row == column) for column in range(size))
                for row in range(size)
            )
        if self.motion_model == "imm5":
            assert self.behavior_profile is not None
            ratio = elapsed_s / self.behavior_profile.persistence_reference_s
            alpha = self.behavior_profile.persistence_alpha**ratio
            probabilities = self.behavior_profile.mode_probabilities
            return tuple(
                tuple(
                    alpha * float(row == column)
                    + (1.0 - alpha) * probability
                    for column, probability in enumerate(probabilities)
                )
                for row in range(len(probabilities))
            )

        # The legacy isotropic/imm matrices are defined per nominal ``step_s``
        # (30 s by default). Scale the same way the imm5 branch does: identity
        # at ``elapsed_s -> 0``, the full per-step matrix once a step elapses.
        base = self.effective_transition_matrix
        ratio = elapsed_s / self.step_s
        interpolation = min(ratio, 1.0)
        return tuple(
            tuple(
                float(row == column)
                + interpolation * (probability - float(row == column))
                for column, probability in enumerate(base_row)
            )
            for row, base_row in enumerate(base)
        )

    def speed_quadrature(self) -> tuple[tuple[float, float], ...]:
        if self.motion_model == "imm5":
            probabilities = self.effective_initial_mode_probabilities
            # Three-point Gauss-Legendre integration for each mode's uniform
            # speed envelope. HALT contributes one exact zero-speed atom.
            uniform_nodes_and_weights = (
                (-0.7745966692414834, 5.0 / 18.0),
                (0.0, 8.0 / 18.0),
                (0.7745966692414834, 5.0 / 18.0),
            )
            combined: dict[float, float] = {}
            for mode, mode_probability in enumerate(probabilities):
                low, high = _ground_speed_bounds(self, mode)
                if high <= low:
                    combined[low] = combined.get(low, 0.0) + mode_probability
                    continue
                midpoint = 0.5 * (low + high)
                half_range = 0.5 * (high - low)
                for node, weight in uniform_nodes_and_weights:
                    speed = midpoint + node * half_range
                    combined[speed] = (
                        combined.get(speed, 0.0) + mode_probability * weight
                    )
            total = sum(combined.values())
            return tuple(
                (speed, weight / total)
                for speed, weight in sorted(combined.items())
            )

        offsets_and_weights = (
            (-2.0, 0.06136),
            (-1.0, 0.24477),
            (0.0, 0.38774),
            (1.0, 0.24477),
            (2.0, 0.06136),
        )
        combined: dict[float, float] = {}
        for offset, weight in offsets_and_weights:
            ratio = min(
                max(
                    self.nominal_speed_ratio + offset * self.speed_sigma_ratio,
                    0.0,
                ),
                1.0,
            )
            speed = ratio * self.max_speed_mps
            combined[speed] = combined.get(speed, 0.0) + weight
        total = sum(combined.values())
        return tuple((speed, weight / total) for speed, weight in combined.items())


CV_MODE = 0
CTRV_MODE = 1
RANDOM_MANEUVER_MODE = 2
MOTION_MODE_NAMES = ("CV", "CTRV", "RANDOM")


@dataclass(frozen=True, slots=True)
class KinematicState:
    speed_mps: float
    heading_rad: float
    turn_rate_rad_s: float
    mode: int


def apply_circular_boundary(
    start: Point2D,
    dx: float,
    dy: float,
    state: KinematicState,
    center: Point2D,
    radius_m: float,
    boundary_mode: str,
) -> tuple[Point2D, KinematicState, bool]:
    """Apply either absorbing escape or ideal specular circle reflection."""

    end = Point2D(start.x + dx, start.y + dy)
    offset_x = end.x - center.x
    offset_y = end.y - center.y
    if offset_x * offset_x + offset_y * offset_y < radius_m**2:
        return end, state, False
    if boundary_mode == "escape":
        return end, state, True

    start_x = start.x - center.x
    start_y = start.y - center.y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-15:
        scale = (radius_m * (1.0 - 1e-12)) / max(
            sqrt(start_x * start_x + start_y * start_y),
            1e-12,
        )
        return (
            Point2D(center.x + start_x * scale, center.y + start_y * scale),
            state,
            False,
        )

    linear = 2.0 * (start_x * dx + start_y * dy)
    constant = start_x * start_x + start_y * start_y - radius_m**2
    discriminant = max(linear * linear - 4.0 * length_squared * constant, 0.0)
    hit_fraction = (-linear + sqrt(discriminant)) / (2.0 * length_squared)
    hit_fraction = min(max(hit_fraction, 0.0), 1.0)
    hit_x = start_x + hit_fraction * dx
    hit_y = start_y + hit_fraction * dy
    hit_norm = max(sqrt(hit_x * hit_x + hit_y * hit_y), 1e-12)
    normal_x = hit_x / hit_norm
    normal_y = hit_y / hit_norm

    remaining_x = (1.0 - hit_fraction) * dx
    remaining_y = (1.0 - hit_fraction) * dy
    normal_component = remaining_x * normal_x + remaining_y * normal_y
    reflected_x = remaining_x - 2.0 * normal_component * normal_x
    reflected_y = remaining_y - 2.0 * normal_component * normal_y
    bounded_x = hit_x + reflected_x
    bounded_y = hit_y + reflected_y
    bounded_radius = sqrt(bounded_x * bounded_x + bounded_y * bounded_y)
    if bounded_radius >= radius_m:
        scale = radius_m * (1.0 - 1e-12) / max(bounded_radius, 1e-12)
        bounded_x *= scale
        bounded_y *= scale

    heading_x = cos(state.heading_rad)
    heading_y = sin(state.heading_rad)
    heading_normal = heading_x * normal_x + heading_y * normal_y
    reflected_heading_x = heading_x - 2.0 * heading_normal * normal_x
    reflected_heading_y = heading_y - 2.0 * heading_normal * normal_y
    reflected_state = KinematicState(
        state.speed_mps,
        atan2(reflected_heading_y, reflected_heading_x) % tau,
        -state.turn_rate_rad_s,
        state.mode,
    )
    return (
        Point2D(center.x + bounded_x, center.y + bounded_y),
        reflected_state,
        False,
    )


def sample_initial_kinematics(
    motion: TargetMotionSpec,
    rng: Random,
) -> KinematicState:
    mode = _weighted_choice(rng, motion.effective_initial_mode_probabilities)
    if motion.motion_model == "imm5":
        speed = _sample_ground_mode_speed(motion, mode, rng)
        if mode == GROUND_CTRV_MODE:
            turn_rate = _sample_ground_turn_rate(motion, rng)
        elif mode == GROUND_MANEUVER_MODE:
            turn_rate = _sample_ground_maneuver_turn_rate(motion, rng)
        else:
            turn_rate = 0.0
        return KinematicState(speed, tau * rng.random(), turn_rate, mode)

    speed_ratio = min(
        max(rng.gauss(motion.nominal_speed_ratio, motion.speed_sigma_ratio), 0.0),
        1.0,
    )
    turn_rate = (
        max(
            -radians(motion.max_turn_rate_dps),
            min(
                radians(motion.max_turn_rate_dps),
                rng.gauss(0.0, radians(motion.turn_rate_process_sigma_dps) * 2.0),
            ),
        )
        if mode == CTRV_MODE
        else 0.0
    )
    return KinematicState(
        speed_ratio * motion.max_speed_mps,
        tau * rng.random(),
        turn_rate,
        mode,
    )


def propagate_kinematics(
    motion: TargetMotionSpec,
    state: KinematicState,
    elapsed_s: float,
    rng: Random,
    halt_probability_boost: float = 0.0,
) -> tuple[float, float, KinematicState]:
    """Draw one motion transition and return dx, dy, and the new state."""

    if elapsed_s <= 0.0:
        return 0.0, 0.0, state
    if motion.motion_model == "isotropic":
        speed_ratio = min(
            max(rng.gauss(motion.nominal_speed_ratio, motion.speed_sigma_ratio), 0.0),
            1.0,
        )
        speed = speed_ratio * motion.max_speed_mps
        heading = tau * rng.random()
        return (
            speed * elapsed_s * cos(heading),
            speed * elapsed_s * sin(heading),
            KinematicState(speed, heading, 0.0, RANDOM_MANEUVER_MODE),
        )

    if motion.motion_model == "imm5":
        return _propagate_ground_five_mode(
            motion,
            state,
            elapsed_s,
            rng,
            halt_probability_boost,
        )

    transition_matrix = motion.transition_matrix_for_elapsed(elapsed_s)
    mode = _weighted_choice(rng, transition_matrix[state.mode])
    noise_scale = sqrt(elapsed_s / motion.step_s)
    speed = min(
        max(
            state.speed_mps
            + rng.gauss(
                0.0,
                motion.speed_process_sigma_ratio
                * motion.max_speed_mps
                * noise_scale,
            ),
            0.0,
        ),
        motion.max_speed_mps,
    )
    previous_heading = state.heading_rad
    heading_noise = rng.gauss(
        0.0,
        radians(motion.heading_process_sigma_deg) * noise_scale,
    )
    max_turn_rate = radians(motion.max_turn_rate_dps)

    if mode == CV_MODE:
        turn_rate = state.turn_rate_rad_s * 0.25
        heading = previous_heading + turn_rate * elapsed_s + heading_noise
    elif mode == CTRV_MODE:
        turn_rate = min(
            max(
                state.turn_rate_rad_s
                + rng.gauss(
                    0.0,
                    radians(motion.turn_rate_process_sigma_dps) * noise_scale,
                ),
                -max_turn_rate,
            ),
            max_turn_rate,
        )
        heading = previous_heading + turn_rate * elapsed_s + heading_noise
    else:
        turn_rate = 0.0
        heading = tau * rng.random()

    effective_turn_rate = (heading - previous_heading) / elapsed_s
    if mode == CTRV_MODE and abs(effective_turn_rate) > 1e-8:
        dx = speed / effective_turn_rate * (sin(heading) - sin(previous_heading))
        dy = speed / effective_turn_rate * (-cos(heading) + cos(previous_heading))
    else:
        dx = speed * elapsed_s * cos(heading)
        dy = speed * elapsed_s * sin(heading)
    return dx, dy, KinematicState(speed, heading % tau, turn_rate, mode)


def _ground_speed_bounds(motion: TargetMotionSpec, mode: int) -> tuple[float, float]:
    # 속도구간은 이제 거동 프로파일이 들고 있다(= config에서 선언 가능).
    # 임무 최대속도로 항상 상한을 자른다.
    profile = motion.behavior_profile
    bands = (
        profile.mode_speed_bounds_kph
        if profile is not None
        else TargetBehaviorProfile.__dataclass_fields__[
            "mode_speed_bounds_kph"
        ].default
    )
    low_kph, high_kph = bands[mode]
    return (
        min(low_kph / 3.6, motion.max_speed_mps),
        min(high_kph / 3.6, motion.max_speed_mps),
    )


def _sample_ground_mode_speed(
    motion: TargetMotionSpec,
    mode: int,
    rng: Random,
) -> float:
    low, high = _ground_speed_bounds(motion, mode)
    if high <= low:
        return low
    return rng.uniform(low, high)


def _sample_ground_turn_rate(motion: TargetMotionSpec, rng: Random) -> float:
    maximum_dps = _ground_turn_rate_limit_dps(motion)
    minimum_dps = min(0.5, maximum_dps)
    magnitude = rng.uniform(minimum_dps, maximum_dps)
    return radians(magnitude if rng.random() < 0.5 else -magnitude)


def _sample_ground_maneuver_turn_rate(
    motion: TargetMotionSpec,
    rng: Random,
) -> float:
    maximum = radians(_ground_turn_rate_limit_dps(motion))
    return rng.uniform(-maximum, maximum)


def _ground_turn_rate_limit_dps(motion: TargetMotionSpec) -> float:
    return min(
        motion.max_turn_rate_dps,
        3.0 * motion.ground_turn_rate_scale,
    )


def _propagate_ground_five_mode(
    motion: TargetMotionSpec,
    state: KinematicState,
    elapsed_s: float,
    rng: Random,
    halt_probability_boost: float = 0.0,
) -> tuple[float, float, KinematicState]:
    transition_matrix = motion.transition_matrix_for_elapsed(elapsed_s)
    mode_probabilities = _with_halt_probability_boost(
        transition_matrix[state.mode],
        halt_probability_boost,
    )
    mode = _weighted_choice(rng, mode_probabilities)
    assert motion.behavior_profile is not None
    noise_scale = sqrt(
        elapsed_s / motion.behavior_profile.process_noise_reference_s
    )
    low_speed, high_speed = _ground_speed_bounds(motion, mode)
    if mode != state.mode or not low_speed <= state.speed_mps <= high_speed:
        speed = _sample_ground_mode_speed(motion, mode, rng)
    else:
        speed_noise_scale = motion.ground_process_noise_scale
        if mode == GROUND_MANEUVER_MODE:
            speed_noise_scale *= 2.0
        speed = min(
            max(
                state.speed_mps
                + rng.gauss(
                    0.0,
                    motion.speed_process_sigma_ratio
                    * motion.max_speed_mps
                    * speed_noise_scale
                    * noise_scale,
                ),
                low_speed,
            ),
            high_speed,
        )

    previous_heading = state.heading_rad
    heading_noise = rng.gauss(
        0.0,
        radians(motion.heading_process_sigma_deg)
        * motion.ground_process_noise_scale
        * noise_scale,
    )
    if mode == HALT_MODE:
        turn_rate = 0.0
        heading = previous_heading
    elif mode in {LOW_CV_MODE, HIGH_CV_MODE}:
        turn_rate = state.turn_rate_rad_s * 0.2
        heading = previous_heading + turn_rate * elapsed_s + heading_noise
    elif mode == GROUND_CTRV_MODE:
        maximum = radians(_ground_turn_rate_limit_dps(motion))
        minimum = radians(min(0.5, _ground_turn_rate_limit_dps(motion)))
        proposed = state.turn_rate_rad_s + rng.gauss(
            0.0,
            radians(motion.turn_rate_process_sigma_dps)
            * motion.ground_process_noise_scale
            * noise_scale,
        )
        if abs(proposed) < minimum:
            proposed = minimum if (proposed >= 0.0) else -minimum
        turn_rate = min(max(proposed, -maximum), maximum)
        heading = previous_heading + turn_rate * elapsed_s + heading_noise
    else:
        maximum = radians(_ground_turn_rate_limit_dps(motion))
        proposed = 0.6 * state.turn_rate_rad_s + rng.gauss(
            0.0,
            radians(motion.turn_rate_process_sigma_dps)
            * motion.ground_process_noise_scale
            * 2.0
            * noise_scale,
        )
        turn_rate = min(max(proposed, -maximum), maximum)
        heading = previous_heading + turn_rate * elapsed_s + 2.0 * heading_noise

    effective_turn_rate = (heading - previous_heading) / elapsed_s
    if mode in {GROUND_CTRV_MODE, GROUND_MANEUVER_MODE} and abs(effective_turn_rate) > 1e-8:
        dx = speed / effective_turn_rate * (sin(heading) - sin(previous_heading))
        dy = speed / effective_turn_rate * (-cos(heading) + cos(previous_heading))
    else:
        dx = speed * elapsed_s * cos(heading)
        dy = speed * elapsed_s * sin(heading)
    return dx, dy, KinematicState(speed, heading % tau, turn_rate, mode)


def _with_halt_probability_boost(
    probabilities: tuple[float, ...],
    boost: float,
) -> tuple[float, ...]:
    """Move a bounded share of non-HALT probability mass into HALT."""
    if not 0.0 <= boost <= 1.0:
        raise ValueError("halt_probability_boost must be in [0, 1]")
    if boost <= 0.0:
        return probabilities
    adjusted = [probability * (1.0 - boost) for probability in probabilities]
    adjusted[HALT_MODE] += boost
    return tuple(adjusted)


def _weighted_choice(rng: Random, probabilities: tuple[float, ...]) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1


@dataclass(frozen=True, slots=True)
class TargetTrajectory:
    """One Monte Carlo ground-truth target path, piecewise linear in time."""

    times_s: tuple[float, ...]
    points: tuple[Point2D, ...]
    profile_name: str = "UNSPECIFIED"
    motion_modes: tuple[int, ...] = ()
    escaped_at_s: float | None = None

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.points) or len(self.times_s) < 2:
            raise ValueError("trajectory needs matching time and point sequences")
        if any(
            later <= earlier
            for earlier, later in zip(self.times_s, self.times_s[1:])
        ):
            raise ValueError("trajectory times must be strictly increasing")
        if self.motion_modes and len(self.motion_modes) != len(self.points) - 1:
            raise ValueError("motion_modes must contain one mode per time interval")
        if self.escaped_at_s is not None and not (
            self.times_s[0] <= self.escaped_at_s <= self.times_s[-1]
        ):
            raise ValueError("escaped_at_s must lie within the trajectory horizon")

    @property
    def end_time_s(self) -> float:
        return self.times_s[-1]

    def position_at_s(self, time_s: float) -> Point2D:
        """구간 선형보간으로 임의 시각의 위치를 준다.

        궤적은 구간별 등속 직선이므로 보간이 곧 정확한 위치다. 범위 밖 시각은
        양끝으로 클램프한다 (임무 종료 후 위치는 마지막 위치로 본다).

        쓰는 곳: ``simulation``의 이탈 판정, ``planning/estimation``의
        belief RMSE 계산.
        """

        times = self.times_s
        points = self.points
        clamped = min(max(time_s, times[0]), times[-1])
        index = 0
        while index + 2 < len(times) and times[index + 1] < clamped:
            index += 1
        span = times[index + 1] - times[index]
        ratio = 0.0 if span <= 0.0 else (clamped - times[index]) / span
        start = points[index]
        end = points[index + 1]
        return Point2D(
            start.x + ratio * (end.x - start.x),
            start.y + ratio * (end.y - start.y),
        )



def predict_radial_belief(
    radial_mass: list[float],
    outside_mass: float,
    elapsed_s: float,
    radial_step_m: float,
    mission: MissionConfig,
    motion: TargetMotionSpec,
) -> tuple[list[float], float]:
    """Predict a radially symmetric belief with an isotropic Markov kernel.

    Probability leaving the fixed search circle enters an absorbing outside
    state so it is not incorrectly renormalised back into the mission area.
    """
    if elapsed_s <= 0.0:
        return list(radial_mass), outside_mass

    predicted = list(radial_mass)
    escaped = outside_mass
    remaining_s = elapsed_s
    while remaining_s > 1e-9:
        delta_s = min(motion.step_s, remaining_s)
        predicted, newly_outside = _predict_one_step(
            predicted,
            delta_s,
            radial_step_m,
            mission,
            motion,
        )
        escaped += newly_outside
        remaining_s -= delta_s
    return predicted, min(1.0, escaped)


def _predict_one_step(
    radial_mass: list[float],
    elapsed_s: float,
    radial_step_m: float,
    mission: MissionConfig,
    motion: TargetMotionSpec,
) -> tuple[list[float], float]:
    destination_mass = [0.0] * len(radial_mass)
    outside_mass = 0.0
    directions = tuple(
        tau * index / motion.prediction_direction_count
        for index in range(motion.prediction_direction_count)
    )
    speed_samples = motion.speed_quadrature()

    for radial_index, source_mass in enumerate(radial_mass):
        if source_mass <= 0.0:
            continue
        radius = min(
            (radial_index + 0.5) * radial_step_m,
            mission.search_radius_m,
        )
        for speed_mps, speed_weight in speed_samples:
            displacement = speed_mps * elapsed_s
            sample_mass = (
                source_mass
                * speed_weight
                / motion.prediction_direction_count
            )
            for direction in directions:
                new_radius = sqrt(
                    max(
                        0.0,
                        radius**2
                        + displacement**2
                        + 2.0 * radius * displacement * cos(direction),
                    )
                )
                if new_radius >= mission.search_radius_m:
                    outside_mass += sample_mass
                    continue
                destination_index = min(
                    int(new_radius / radial_step_m),
                    len(destination_mass) - 1,
                )
                destination_mass[destination_index] += sample_mass

    return destination_mass, outside_mass
