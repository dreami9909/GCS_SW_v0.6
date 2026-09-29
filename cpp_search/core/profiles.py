"""Missing-person 운용 프로파일 — 시그니처 + IMM5 거동을 한 객체로 묶는다.

진리 생성기, 계획기, 센서 모델이 **같은 표적 정의**를 보게 만드는 것이
이 파일의 유일한 목적이다. 여기가 갈리면 계획과 평가가 서로 다른 표적을
가정하게 된다.

    TargetOperationalProfile
      ├── signature      : sensor_observation.TargetSignatureSpec (크기·EO/IR 대비)
      ├── imm5_behavior  : motion.TargetBehaviorProfile (5-모드 점유·지속성)
      └── motion_spec()  : 임무 최대속도를 받아 motion.TargetMotionSpec 생성

의존
----
* 위: ``motion``, ``sensor_observation``.
* 아래: ``planning/*``, ``teamwork/transfer``, 그리고 모든 챕터 실험.
"""

from __future__ import annotations

from dataclasses import dataclass

from cpp_search.core.motion import (
    MANEUVER_HEAVY_PROFILE,
    RELOCATION_HEAVY_PROFILE,
    TargetBehaviorProfile,
    TargetMotionSpec,
)
from cpp_search.core.sensor_observation import (
    MISSING_PERSON_ADULT_NOMINAL,
    MISSING_PERSON_CHILD_NOMINAL,
    TargetSignatureSpec,
)


@dataclass(frozen=True, slots=True)
class TargetOperationalProfile:
    """Target class declaration shared by truth, planner, and sensor models."""

    name: str
    signature: TargetSignatureSpec
    imm5_behavior: TargetBehaviorProfile
    integration_step_s: float = 30.0
    boundary_mode: str = "escape"
    calibration_status: str = "uncalibrated research prior; track calibration pending"

    def __post_init__(self) -> None:
        if self.integration_step_s <= 0.0:
            raise ValueError("integration_step_s must be positive")
        if self.boundary_mode not in {"escape", "reflect"}:
            raise ValueError("boundary_mode must be escape or reflect")

    def motion_spec(
        self,
        *,
        max_speed_mps: float,
        step_s: float | None = None,
        boundary_mode: str | None = None,
    ) -> TargetMotionSpec:
        return TargetMotionSpec(
            max_speed_mps=max_speed_mps,
            step_s=self.integration_step_s if step_s is None else step_s,
            motion_model="imm5",
            boundary_mode=self.boundary_mode if boundary_mode is None else boundary_mode,
            behavior_profile=self.imm5_behavior,
        )


MISSING_PERSON_ADULT_OPERATIONAL_PROFILE = TargetOperationalProfile(
    name="MissingPerson-adult-nominal",
    signature=MISSING_PERSON_ADULT_NOMINAL,
    imm5_behavior=MANEUVER_HEAVY_PROFILE,
)

MISSING_PERSON_CHILD_OPERATIONAL_PROFILE = TargetOperationalProfile(
    name="MissingPerson-child-nominal",
    signature=MISSING_PERSON_CHILD_NOMINAL,
    imm5_behavior=RELOCATION_HEAVY_PROFILE,
)

NOMINAL_TARGET_PROFILES = (
    MISSING_PERSON_ADULT_OPERATIONAL_PROFILE,
    MISSING_PERSON_CHILD_OPERATIONAL_PROFILE,
)
