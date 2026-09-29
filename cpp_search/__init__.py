"""협동 표적탐색 연구 시뮬레이터 — 도메인 계층.

이 패키지의 최상위 모듈은 "세계"를 기술한다. 챕터·탐색전략 개념이 전혀 없고,
표적이 어떻게 움직이고 센서가 무엇을 보며 경로가 얼마나 덮는지만 안다.

```
cpp_search/
├── models.py             Point2D, SensorSpec(SR-Z50 기하·W·weave), MissionConfig, Route
├── geometry.py           극좌표 <-> 월드좌표 변환
├── search_envelope.py    SAR weave + EO/IR 주사를 합친 400 m 지지반폭, W = ∫PD dx
├── motion.py             IMM5 표적 운동모델(5모드)과 진리 궤적
├── probability.py        표적 사전확률과 극좌표 확률지도
├── belief.py             로그오즈 증거지도, 정규화 표적위치지도, POC/POD/POS
├── particle_filter.py    SIR 입자필터(예측·음성관측·재표본추출·roughening)
├── sensor_observation.py Missing-person 시그니처, EO/IR 성능, 공간 탐지모델
├── terrain.py            합성 지형(기동성·사전·관측성 가중치)
├── profiles.py           Missing-person 운용 프로파일 = 시그니처 + IMM5 거동
├── evaluation.py         경로 기하 KPI(고유 면적·중복률·중심선 대 실제 거리)
├── simulation.py         Monte Carlo 진리 생성과 탐지시간 평가 -> PerformanceMetrics
└── research/             연구 계층(Stone SPX · MAPPO · 챕터 실험)
```

의존은 한 방향이다. ``research``는 이 계층을 자유롭게 쓰지만, 이 계층은
``research``를 절대 import 하지 않는다. ``tests/test_architecture.py``가 강제한다.
"""

from cpp_search.core.models import MissionConfig, Point2D, Route, SensorChannelSpec, SensorSpec
from cpp_search.core.profiles import (
    MISSING_PERSON_ADULT_OPERATIONAL_PROFILE,
    MISSING_PERSON_CHILD_OPERATIONAL_PROFILE,
    NOMINAL_TARGET_PROFILES,
    TargetOperationalProfile,
)
from cpp_search.core.search_envelope import CompositeSearchEnvelope
from cpp_search.core.sensor_observation import (
    MISSING_PERSON_ADULT_NOMINAL,
    MISSING_PERSON_CHILD_NOMINAL,
    SENSOR_PERFORMANCE_SR_Z50,
    SpatialDetectionModel,
)

__all__ = [
    "CompositeSearchEnvelope",
    "MISSING_PERSON_ADULT_NOMINAL",
    "MISSING_PERSON_ADULT_OPERATIONAL_PROFILE",
    "MISSING_PERSON_CHILD_NOMINAL",
    "MISSING_PERSON_CHILD_OPERATIONAL_PROFILE",
    "MissionConfig",
    "NOMINAL_TARGET_PROFILES",
    "Point2D",
    "Route",
    "SENSOR_PERFORMANCE_SR_Z50",
    "SensorChannelSpec",
    "SensorSpec",
    "SpatialDetectionModel",
    "TargetOperationalProfile",
]
