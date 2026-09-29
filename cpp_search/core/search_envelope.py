"""시스템 탐색범위 — 400 m 지지반폭과 유효 탐색폭 W.

400 m는 "김벌 반경"이 아니라 **비행체 횡방향 weave + EO/IR 주사**를 합친
시스템 수준의 횡방향 지지폭이다. 이 파일이 그 분해와 W 적분을 담당한다.

    지지반폭 = 탐색기 반폭 + weave 반진폭
             = 300 m + 100 m = 400 m        (operational_400m)

핵심 수식
---------
* 횡방향 탐지확률 곡선 (``lateral_detection_probability``) — raised cosine

      u = |x| / X0                          (X0 = 지지반폭)
      PD(x) = PD0 * [ e + (1 - e) * (1 + cos(pi * u)) / 2 ],  |x| < X0
      PD(x) = 0,                                              |x| >= X0

  e = ``edge_relative_probability`` (기본 0). PD0 = 중심선 탐지확률.

* 유효 탐색폭 (``sweep_width_m``) — Koopman의 정의 그대로

      W = ∫_{-X0}^{+X0} PD(x) dx

  Simpson 적분으로 계산한다. PD0 = 1, e = 0이면 raised cosine의 평균이
  정확히 1/2이므로 **W = X0 = 400 m**가 나온다. 지지폭 800 m와 다르다는
  것이 이 프로젝트의 핵심 해석 규칙 중 하나다.

* weave 실제 비행거리 (``WeavePattern.actual_distance_m``) — 사인 곡선 호 길이

      y(s) = A * sin(k*s + phi),  k = 2*pi / lambda
      L_actual = ∫_0^L sqrt(1 + (A*k*cos(k*s + phi))^2) ds

  Simpson 적분. 계획 중심선 거리와 실제 비행거리를 KPI에서 따로 보고하는
  근거가 이 식이다.

* 중심선 진행속도 (``progress_speed_mps``)

      v_centerline = v_actual / (L_actual(lambda) / lambda)

  실제 대기속도로 날아도 중심선을 따라가는 속도는 그만큼 느리다.

의존
----
* 위: 표준 라이브러리만.
* 아래: ``models.SensorSpec``이 이 객체를 들고 W와 소인간격을 위임한다.
  Ch0(``chapters/chapter0.py``)이 표적·채널별 W 테이블을 만들 때,
  Ch1(``theory/effort_route.py``)이 달성 POS를 잴 때 직접 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sqrt


@dataclass(frozen=True, slots=True)
class WeavePattern:
    """Sinusoidal lateral vehicle motion around a planned centerline."""

    half_amplitude_m: float
    wavelength_m: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.half_amplitude_m < 0.0:
            raise ValueError("half_amplitude_m must be non-negative")
        if self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be positive")

    def actual_distance_m(self, centerline_distance_m: float) -> float:
        """Return arc length of the sinusoidal path over a centerline distance."""

        if centerline_distance_m < 0.0:
            raise ValueError("centerline_distance_m must be non-negative")
        if centerline_distance_m == 0.0 or self.half_amplitude_m == 0.0:
            return centerline_distance_m

        cycles = centerline_distance_m / self.wavelength_m
        steps = max(32, int(cycles * 128.0) + 2)
        if steps % 2:
            steps += 1
        step_m = centerline_distance_m / steps
        wave_number = 2.0 * pi / self.wavelength_m

        def integrand(centerline_m: float) -> float:
            # 호 길이 적분의 피적분 함수 sqrt(1 + (dy/ds)^2).
            # y(s) = A sin(k s + phi)  ->  dy/ds = A k cos(k s + phi)
            slope = (
                self.half_amplitude_m
                * wave_number
                * cos(wave_number * centerline_m + self.phase_rad)
            )
            return sqrt(1.0 + slope * slope)

        total = integrand(0.0) + integrand(centerline_distance_m)
        total += 4.0 * sum(integrand(index * step_m) for index in range(1, steps, 2))
        total += 2.0 * sum(integrand(index * step_m) for index in range(2, steps, 2))
        return step_m * total / 3.0

    def progress_speed_mps(self, actual_speed_mps: float) -> float:
        """Convert actual vehicle speed to average centerline progress speed."""

        if actual_speed_mps <= 0.0:
            raise ValueError("actual_speed_mps must be positive")
        ratio = self.actual_distance_m(self.wavelength_m) / self.wavelength_m
        return actual_speed_mps / ratio


@dataclass(frozen=True, slots=True)
class CompositeSearchEnvelope:
    """System-level search envelope from vehicle weave and search camera scanning.

    ``support_half_width_m`` is where the modeled lateral detection response
    reaches zero. It is not the sweep width. Sweep width is obtained by
    integrating ``lateral_detection_probability``.
    """

    name: str
    support_half_width_m: float
    seeker_half_width_m: float
    weave: WeavePattern
    edge_relative_probability: float = 0.0
    calibration_status: str = "uncalibrated operational assumption"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.support_half_width_m <= 0.0:
            raise ValueError("support_half_width_m must be positive")
        if self.seeker_half_width_m <= 0.0:
            raise ValueError("seeker_half_width_m must be positive")
        if not 0.0 <= self.edge_relative_probability < 1.0:
            raise ValueError("edge_relative_probability must be in [0, 1)")
        composed_width = self.seeker_half_width_m + self.weave.half_amplitude_m
        if abs(composed_width - self.support_half_width_m) > 1e-6:
            raise ValueError(
                "search camera reach plus vehicle weave amplitude must equal support half-width"
            )

    @classmethod
    def operational(
        cls,
        *,
        support_half_width_m: float,
        seeker_half_width_m: float,
        vehicle_weave_half_amplitude_m: float = 100.0,
        vehicle_weave_wavelength_m: float = 800.0,
        calibration_status: str = "uncalibrated operational assumption",
    ) -> "CompositeSearchEnvelope":
        """Build an explicit system envelope from search camera and vehicle motion."""

        return cls(
            name="SAR weave + SR-Z50 EO/IR operational envelope",
            support_half_width_m=support_half_width_m,
            seeker_half_width_m=seeker_half_width_m,
            weave=WeavePattern(
                half_amplitude_m=vehicle_weave_half_amplitude_m,
                wavelength_m=vehicle_weave_wavelength_m,
            ),
            calibration_status=calibration_status,
        )

    @classmethod
    def operational_400m(
        cls,
        *,
        vehicle_weave_half_amplitude_m: float = 100.0,
        vehicle_weave_wavelength_m: float = 800.0,
    ) -> "CompositeSearchEnvelope":
        """Return the declared 300 m search camera + 100 m weave decomposition."""

        return cls.operational(
            support_half_width_m=400.0,
            seeker_half_width_m=400.0 - vehicle_weave_half_amplitude_m,
            vehicle_weave_half_amplitude_m=vehicle_weave_half_amplitude_m,
            vehicle_weave_wavelength_m=vehicle_weave_wavelength_m,
        )

    def lateral_detection_probability(
        self,
        cross_track_offset_m: float,
        centerline_detection_probability: float = 1.0,
    ) -> float:
        """Raised-cosine lateral detection curve for one independent pass."""

        if not 0.0 <= centerline_detection_probability <= 1.0:
            raise ValueError("centerline_detection_probability must be in [0, 1]")
        normalized = abs(cross_track_offset_m) / self.support_half_width_m
        if normalized >= 1.0:
            return 0.0
        # raised cosine: e + (1 - e) * (1 + cos(pi * u)) / 2,  u = |x| / X0
        # u = 1(지지폭 경계)에서 e로, u = 0(중심선)에서 1로 떨어진다.
        relative = self.edge_relative_probability + (
            1.0 - self.edge_relative_probability
        ) * 0.5 * (1.0 + cos(pi * normalized))
        return centerline_detection_probability * relative

    def sweep_width_m(
        self,
        centerline_detection_probability: float = 1.0,
        integration_steps: int = 2048,
    ) -> float:
        """Numerically evaluate W = integral P_D(x) dx across the support."""

        if integration_steps < 2:
            raise ValueError("integration_steps must be at least two")
        if integration_steps % 2:
            integration_steps += 1
        lower = -self.support_half_width_m
        upper = self.support_half_width_m
        step_m = (upper - lower) / integration_steps
        values = [
            self.lateral_detection_probability(
                lower + index * step_m,
                centerline_detection_probability,
            )
            for index in range(integration_steps + 1)
        ]
        # W = ∫ PD(x) dx 를 Simpson 공식으로:
        #   (h/3) * [ f0 + 4*(홀수항 합) + 2*(짝수항 합) + fn ]
        # PD0 = 1, e = 0이면 raised cosine의 평균이 1/2이므로 W = X0.
        weighted_sum = values[0] + values[-1]
        weighted_sum += 4.0 * sum(values[1:-1:2])
        weighted_sum += 2.0 * sum(values[2:-1:2])
        return step_m * weighted_sum / 3.0

    def actual_distance_m(self, centerline_distance_m: float) -> float:
        return self.weave.actual_distance_m(centerline_distance_m)

    def centerline_progress_speed_mps(self, actual_speed_mps: float) -> float:
        return self.weave.progress_speed_mps(actual_speed_mps)
