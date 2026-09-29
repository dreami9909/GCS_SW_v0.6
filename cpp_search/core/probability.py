"""표적 사전확률과 극좌표 확률지도.

핵심 수식
---------
* 반경 방향 사전밀도 (``TargetPrior.relative_density``)

      uniform      : f(r) = 1
      tp-centered  : f(r) = exp(-r^2 / (2 sigma^2))
      moving-ring  : f(r) = exp(-(r - mu)^2 / (2 sigma^2))

  mu = ``mean_radius_ratio`` * R, sigma = ``sigma_ratio`` * R.
  moving-ring은 "발사 시점 이후 표적이 반경 mu 부근 고리에 있을 것"이라는
  가정이다. 정규화는 하지 않는다 (셀 면적을 곱한 뒤 전체로 나눈다).

* 극좌표 셀 면적 (``PolarProbabilityMap.build``) — 면적 정확 이산화

      A_cell = (r_out^2 - r_in^2) * dtheta / 2

  단순히 dr * r * dtheta로 근사하지 않는다. 바깥 링이 넓다는 사실이
  노력배분에 그대로 반영되어야 하기 때문이다.

* 셀 사전질량

      w_cell = f(r) * g_terrain(x, y) * A_cell
      p_cell = w_cell / sum(w)

  g_terrain은 ``terrain.TerrainField.prior_weight``. Ch4의 Terrain-Prior
  ablation이 이 항을 켜고 끈다.

* 셀 관측성 ``observability`` = ``terrain.observability_weight``.
  Ch2/Ch6의 FAB 탐지효율 w[x,t]가 이 값에서 나온다.

의존
----
* 위: ``geometry``(좌표 변환), ``models``. 지형은 덕 타이핑으로 주입받는다.
* 아래: ``particle_filter.cell_masses``(입자 -> 셀 질량 투영),
  ``theory/markov``(셀 = Markov 상태), ``planning/sarops_adapted``,
  ``planning/team_planner``, ``planning/estimation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, pi, tau

from cpp_search.core.geometry import polar_to_world
from cpp_search.core.models import MissionConfig, Point2D


@dataclass(frozen=True, slots=True)
class TargetPrior:
    """Radially symmetric prior for the target position at search start."""

    kind: str = "moving-ring"
    mean_radius_ratio: float = 0.65
    sigma_ratio: float = 0.15

    def __post_init__(self) -> None:
        if self.kind not in {"uniform", "tp-centered", "moving-ring"}:
            raise ValueError(f"unsupported target prior: {self.kind}")
        if not 0.0 <= self.mean_radius_ratio <= 1.0:
            raise ValueError("mean_radius_ratio must be in [0, 1]")
        if self.sigma_ratio <= 0.0:
            raise ValueError("sigma_ratio must be positive")
        # 기각표본추출은 밀도 최대/평균 비로 수용률이 정해진다. sigma 가 AOI 의
        # 0.5% 아래면 수용률이 ~0 이라 표본 생성이 사실상 멈춘다 (실측: 50분 정지).
        if self.kind != "uniform" and self.sigma_ratio < 0.005:
            raise ValueError("sigma_ratio below 0.005 makes rejection sampling stall; use >= 0.005")

    def relative_density(self, radius_m: float, search_radius_m: float) -> float:
        """Return an unnormalised spatial density in the search circle."""
        if not 0.0 <= radius_m <= search_radius_m:
            return 0.0
        if self.kind == "uniform":
            return 1.0

        mean_radius_m = (
            0.0
            if self.kind == "tp-centered"
            else self.mean_radius_ratio * search_radius_m
        )
        sigma_m = self.sigma_ratio * search_radius_m
        # Ring-Gaussian: f(r) = exp( -(r - mu)^2 / (2 sigma^2) ).
        # tp-centered는 mu = 0이므로 중심 집중형 절단 Gaussian이 된다.
        # 정규화하지 않는다 — 셀 면적을 곱한 뒤 전체 합으로 나눈다.
        standardised = (radius_m - mean_radius_m) / sigma_m
        return exp(-0.5 * standardised**2)

    @property
    def label(self) -> str:
        labels = {
            "uniform": "원형 영역 내 균일분포",
            "tp-centered": "TP 중심 절단 Gaussian 분포",
            "moving-ring": "이동 반경 중심 Ring-Gaussian 분포",
        }
        return labels[self.kind]


@dataclass(frozen=True, slots=True)
class ProbabilityCell:
    center: Point2D
    radius_m: float
    angle_rad: float
    area_m2: float
    prior_mass: float
    observability: float = 1.0


@dataclass(frozen=True, slots=True)
class PolarProbabilityMap:
    """Area-correct polar discretisation of a target-position prior."""

    prior: TargetPrior
    cells: tuple[ProbabilityCell, ...]
    radial_step_m: float
    angular_bin_count: int

    @classmethod
    def build(
        cls,
        mission: MissionConfig,
        prior: TargetPrior,
        radial_step_m: float,
        angular_bin_count: int = 180,
        terrain=None,
        terrain_stop_ratio: float = 0.4,
        terrain_mode_probabilities: tuple[float, ...] | None = None,
    ) -> "PolarProbabilityMap":
        if radial_step_m <= 0.0:
            raise ValueError("radial_step_m must be positive")
        if angular_bin_count < mission.uav_count:
            raise ValueError("angular_bin_count must cover all sectors")

        radial_bin_count = ceil(mission.search_radius_m / radial_step_m)
        angle_step = tau / angular_bin_count
        raw_cells: list[tuple[Point2D, float, float, float, float, float]] = []
        total_weight = 0.0

        for radial_index in range(radial_bin_count):
            inner_radius = radial_index * radial_step_m
            outer_radius = min(
                (radial_index + 1) * radial_step_m,
                mission.search_radius_m,
            )
            radius = (inner_radius + outer_radius) / 2.0
            # 면적 정확 이산화: A = (r_out^2 - r_in^2) * dtheta / 2.
            # dr * r * dtheta 근사를 쓰지 않는 이유는 바깥 링이 실제로 훨씬
            # 넓고, 그 사실이 노력배분에 그대로 반영되어야 하기 때문이다.
            cell_area = 0.5 * (outer_radius**2 - inner_radius**2) * angle_step
            radial_density = prior.relative_density(radius, mission.search_radius_m)

            for angle_index in range(angular_bin_count):
                angle = (angle_index + 0.5) * angle_step
                point = polar_to_world(radius, angle, mission.center)
                # Terrain reweights the radially-symmetric prior per angle.
                if terrain is not None:
                    terrain_gain = terrain.prior_weight(
                        point.x,
                        point.y,
                        terrain_stop_ratio,
                        terrain_mode_probabilities,
                    )
                    observability = (
                        terrain.observability_weight(point.x, point.y)
                        if hasattr(terrain, "observability_weight")
                        else 1.0
                    )
                else:
                    terrain_gain = 1.0
                    observability = 1.0
                density = radial_density * terrain_gain
                weight = density * cell_area
                raw_cells.append(
                    (point, radius, angle, cell_area, weight, observability)
                )
                total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("target prior has no probability mass")

        cells = tuple(
            ProbabilityCell(
                point,
                radius,
                angle,
                area,
                weight / total_weight,
                observability,
            )
            for point, radius, angle, area, weight, observability in raw_cells
        )
        return cls(prior, cells, radial_step_m, angular_bin_count)

    @property
    def total_mass(self) -> float:
        return sum(cell.prior_mass for cell in self.cells)
