"""지형 가중 이동표적 belief — SPX 와 MAPPO 가 **공유하는** 단 하나의 입력.

Stone SPX 와 MAPPO 를 같은 척도로 비교하려면 둘이 같은 표적 belief 를 봐야
한다. 이 모듈이 그 belief 를 만든다. 여기서 나온 셀 질량 하나가
``theory/markov`` 를 거쳐 SPX 의 ``StoneTargetModel`` 과 MAPPO 환경의 상태로
**동시에** 들어간다.

지형 3성분을 서로 다른 지점에 주입한다.

    coupling.prior         -> PolarProbabilityMap 의 셀 사전질량
                              (통행성·은폐 가중 -> 표적이 어디 있을 법한가)
    coupling.transition    -> 입자필터 예측 커널
                              (이동 편향·off-road 확률·정지확률 상승)
    coupling.observability -> **음성 관측** 갱신에서 공간 탐지모델 사용 여부

세 성분을 한 플래그로 묶지 않은 이유는 "지형을 넣었다"가 세 개의 다른 주장이기
때문이다. v0.4 의 기본 조건은 ``BELIEF_TERRAIN_FULL``(전부 on) 이고, ``BELIEF_TERRAIN_OFF``
는 지형 가중의 효과를 재는 대조군으로만 쓴다.

**SPX 인스턴스에는 앞의 두 지점만 실제로 작용한다.** 인스턴스 구축은 belief 를
``predict`` 만 하고 음성 관측을 넣지 않으므로 ``observability`` 는 읽히지
않는다. 그리고 합성지형(``cpp_search.core.terrain.TerrainField``)의
``observability_weight`` 는 **선언된 모델 범위상 항상 1.0** 이다 — 지형은
탐색자의 센서에 작용하지 않고 표적 belief 에만 작용한다. 따라서 v0.4 에서
"지형 가중"이란 **사전질량 + 전이커널** 두 가지이고, 셀별 hazard 는 지형과
무관하다 (2026-09-13 검토에서 수치로 확인: ``terrain_effectiveness`` 전 셀 1.0).

주요 절차
---------
* ``predict(dt)``                 — 필터 전진 후 셀 질량으로 투영.
* ``observe_no_detection(routes)``— 센서 on 구간만 뽑아 음성 베이즈 갱신.
  observability 가 켜졌을 때만 ``SpatialDetectionModel`` 을 넘긴다.
* ``project_to_grid()``           — 현재 셀 질량 + 필터 진단값 스냅샷.

의존
----
* 위: ``cpp_search.{particle_filter, probability, profiles, models}``.
* 아래: ``planning/stone_spx``, ``learning/spx_env``, ``chapters/*``.
"""

from __future__ import annotations

from dataclasses import dataclass

from cpp_search.core.models import MissionConfig, Route, SensorSpec
from cpp_search.core.particle_filter import ParticleFilterConfig, TargetParticleFilter
from cpp_search.core.probability import PolarProbabilityMap, TargetPrior
from cpp_search.core.profiles import TargetOperationalProfile


@dataclass(frozen=True, slots=True)
class BeliefTerrainCoupling:
    """지형이 **belief 의 어느 지점에** 들어가는지 선언한다.

    ``research/truth.TerrainCoupling`` 과 혼동하면 안 된다. 그쪽은 **진리
    표적의 운동**에 지형이 개입하는 세 계수(편향·off-road·정지확률)이고,
    이쪽은 **탐색자가 가진 추정**에 지형이 들어가는 세 지점이다. 하나는
    세계가 어떻게 움직이는지, 다른 하나는 우리가 그것을 어떻게 믿는지다.
    """

    name: str
    prior: bool = False
    transition: bool = False
    observability: bool = False


#: 지형 가중 없음 — 지형 효과를 재는 대조군.
BELIEF_TERRAIN_OFF = BeliefTerrainCoupling("Terrain-Off")

#: v0.4 기본 조건 — 사전·전이·관측성 전부 지형에 결합.
BELIEF_TERRAIN_FULL = BeliefTerrainCoupling(
    "Terrain-Full",
    prior=True,
    transition=True,
    observability=True,
)

BELIEF_TERRAIN_COUPLINGS_BY_NAME: dict[str, BeliefTerrainCoupling] = {
    coupling.name: coupling for coupling in (BELIEF_TERRAIN_OFF, BELIEF_TERRAIN_FULL)
}


@dataclass(frozen=True, slots=True)
class ParticleGridSnapshot:
    cell_mass: tuple[float, ...]
    outside_probability: float
    effective_particle_count: float
    resample_count: int
    prediction_step_count: int
    observation_update_count: int

    @property
    def in_area_probability(self) -> float:
        return sum(self.cell_mass)


class TerrainParticleBelief:
    """입자 예측 · 음성 베이즈 갱신 · 재표본추출 · 격자 투영.

    공개된 탐색이론에 기반한 자체 구현이며, 비공개 USCG SAROPS 구현의 재현을
    주장하지 않는다.
    """

    def __init__(
        self,
        mission: MissionConfig,
        target_profile: TargetOperationalProfile,
        *,
        prior: TargetPrior | None = None,
        particle_count: int = 3_000,
        seed: int = 20_260_830,
        radial_step_m: float = 100.0,
        angular_bin_count: int = 180,
        terrain=None,
        terrain_coupling: BeliefTerrainCoupling = BELIEF_TERRAIN_FULL,
        terrain_bias_strength: float = 0.6,
        terrain_offroad_probability: float = 0.2,
        terrain_halt_probability_boost: float = 0.35,
        roughening: bool = True,
        roughening_gain: float = 0.2,
    ) -> None:
        if terrain_coupling != BELIEF_TERRAIN_OFF and terrain is None:
            raise ValueError("terrain coupling requires a terrain model")
        self.mission = mission
        self.target_profile = target_profile
        self.prior = prior or TargetPrior(kind="moving-ring")
        self.terrain = terrain
        self.terrain_coupling = terrain_coupling
        self.probability_map = PolarProbabilityMap.build(
            mission,
            self.prior,
            radial_step_m,
            angular_bin_count,
            terrain=terrain if terrain_coupling.prior else None,
            terrain_mode_probabilities=(
                target_profile.imm5_behavior.mode_probabilities
                if terrain_coupling.prior
                else None
            ),
        )
        self.filter = TargetParticleFilter(
            mission,
            self.prior,
            target_profile.motion_spec(max_speed_mps=mission.subject_max_speed_mps),
            particle_count=particle_count,
            seed=seed,
            config=(
                ParticleFilterConfig.sarops(roughening_gain=roughening_gain)
                if roughening
                else ParticleFilterConfig.optimized()
            ),
            terrain=terrain if terrain_coupling.prior else None,
            transition_terrain=terrain if terrain_coupling.transition else None,
            terrain_bias_strength=(
                terrain_bias_strength if terrain_coupling.transition else 0.0
            ),
            terrain_offroad_probability=terrain_offroad_probability,
            terrain_halt_probability_boost=(
                terrain_halt_probability_boost
                if terrain_coupling.transition
                else 0.0
            ),
        )

    def predict(self, elapsed_s: float) -> ParticleGridSnapshot:
        if elapsed_s < 0.0:
            raise ValueError("elapsed_s must be non-negative")
        self.filter.predict(elapsed_s)
        return self.project_to_grid()

    def observe_no_detection(
        self,
        routes: list[Route],
        sensor: SensorSpec,
        *,
        detection_probability: float,
        detection_model=None,
    ) -> ParticleGridSnapshot:
        sensing_segments = tuple(
            segment
            for route in routes
            for segment in route.segments
            if segment.sensor_on
        )
        active_detection_model = (
            detection_model if self.terrain_coupling.observability else None
        )
        self.filter.observe_no_detection_segments(
            sensing_segments,
            detection_radius_m=sensor.coverage_half_width_m,
            detection_probability=detection_probability,
            detection_model=active_detection_model,
            sensor=sensor if active_detection_model is not None else None,
            search_speed_mps=(
                sensor.centerline_search_speed_mps(self.mission.search_speed_mps)
                if active_detection_model is not None
                else None
            ),
        )
        return self.project_to_grid()

    def project_to_grid(self) -> ParticleGridSnapshot:
        diagnostics = self.filter.diagnostics
        return ParticleGridSnapshot(
            cell_mass=tuple(self.filter.cell_masses(self.probability_map)),
            outside_probability=self.filter.outside_probability,
            effective_particle_count=self.filter.effective_particle_count,
            resample_count=diagnostics.resample_count,
            prediction_step_count=diagnostics.prediction_step_count,
            observation_update_count=diagnostics.observation_update_count,
        )
