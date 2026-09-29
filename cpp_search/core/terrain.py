"""합성 지형 — 기동성 / 사전 / 관측성 세 가지 가중치.

Ch4의 지형 ablation이 이 세 성분을 하나씩 켜고 끈다. 세 성분이 서로 다른
곳에 들어간다는 점이 중요하다.

    prior         -> probability.PolarProbabilityMap의 셀 사전질량
    transition    -> particle_filter의 예측 커널 (이동 편향, 정지 확률)
    observability -> FAB 탐지효율 w[x,t], 그리고 센서 가림

핵심 수식
---------
* 통로 가중 (``Corridor.weight``)

      w(x,y) = exp( -d(x,y)^2 / (2 * sigma^2) )

  d = 통로 중심선까지의 수직거리. 차량은 도로를 따라 움직인다는 가정.

* 장애물 억제 (``Barrier.suppression``) — 내부는 0, 경계 부근은 완만한 감쇠.

* 기동성 (``mobility_weight``) = 통로 가중의 합 * 장애물 억제.

* 사전 가중 (``prior_weight``)
  기동성과 은폐도를 IMM5 모드 점유확률로 섞는다. 정지 성향이 큰 표적일수록
  은폐지에 있을 사전확률이 커진다.

* 이동 편향 (``bias_step``)

      v <- normalize( (1-b) * v + b * grad(mobility) )

  b = ``terrain_bias_strength``. 기동성이 커지는 방향으로 끌린다.

* 관측성 (``observability_weight``) — 은폐도가 높을수록 낮아진다.
  Ch2/Ch5에서 FAB의 탐지효율 w[x,t]가 되고, 값이 낮은 셀은 같은 노력으로도
  탐지확률이 낮으므로 배분이 자연히 다른 곳으로 간다.

의존
----
* 위: ``models``만.
* 아래: ``probability``, ``particle_filter``, ``simulation``(진리 궤적),
  ``planning/sarops_adapted``, ``planning/team_planner``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, hypot, sin, tau
from pathlib import Path
from random import Random

import numpy as np

from cpp_search.core.models import Point2D


@dataclass(frozen=True, slots=True)
class Corridor:
    """선분 [start, end] 주변으로 통행성이 높은 띠."""
    start: Point2D
    end: Point2D
    half_width_m: float          # 띠의 특성 폭(가우시안 sigma)
    strength: float = 1.0        # 통행성 가중 최대 기여

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """Return the closest centerline point and segment fraction."""
        ax, ay = self.start.x, self.start.y
        dx = self.end.x - ax
        dy = self.end.y - ay
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-9:
            return ax, ay, 0.0
        fraction = ((x - ax) * dx + (y - ay) * dy) / length_squared
        fraction = max(0.0, min(1.0, fraction))
        return ax + fraction * dx, ay + fraction * dy, fraction

    def tangent(self) -> tuple[float, float]:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = hypot(dx, dy)
        if length <= 1e-9:
            return 0.0, 0.0
        return dx / length, dy / length

    def distance_to_centerline(self, x: float, y: float) -> float:
        projected_x, projected_y, _ = self.project(x, y)
        return hypot(x - projected_x, y - projected_y)

    def weight(self, x: float, y: float) -> float:
        dist = self.distance_to_centerline(x, y)
        return self.strength * exp(-0.5 * (dist / self.half_width_m) ** 2)


@dataclass(frozen=True, slots=True)
class Barrier:
    """원형/곡선 장애물: 중심 근처에서 통행성을 강하게 억제."""
    center: Point2D
    radius_m: float
    softness_m: float = 150.0    # 가장자리 전이 폭

    def contains(self, x: float, y: float) -> bool:
        return hypot(x - self.center.x, y - self.center.y) <= self.radius_m

    def intersects_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
    ) -> bool:
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            return self.contains(x, y)
        fraction = (
            (self.center.x - x) * dx + (self.center.y - y) * dy
        ) / length_squared
        fraction = max(0.0, min(1.0, fraction))
        closest_x = x + fraction * dx
        closest_y = y + fraction * dy
        return self.contains(closest_x, closest_y)

    def suppression(self, x: float, y: float) -> float:
        """0(완전 차단) ~ 1(영향 없음)을 반환."""
        dist = hypot(x - self.center.x, y - self.center.y)
        # 내부는 0에 가깝고, radius+softness 밖은 1
        edge = (dist - self.radius_m) / self.softness_m
        # 로지스틱: edge<0 이면 0쪽, edge>0 이면 1쪽
        return 1.0 / (1.0 + exp(-edge))


@dataclass(frozen=True, slots=True)
class ConcealmentPatch:
    """표적이 정지·은신할 확률이 높은 원형 영역(산림·시가지)."""
    center: Point2D
    radius_m: float
    strength: float = 1.0

    def weight(self, x: float, y: float) -> float:
        dist = hypot(x - self.center.x, y - self.center.y)
        return self.strength * exp(-0.5 * (dist / self.radius_m) ** 2)


@dataclass(frozen=True, slots=True)
class TerrainField:
    """합성 지형: 통행성 회랑 + 장애물 + 은폐 패치의 조합.

    mobility_weight(x, y):  이동표적의 통행 가능성 (회랑↑, 장애물↓)
    concealment_weight(x, y): 정지표적의 은신 선호 (패치↑)
    prior_weight(x, y, stop_ratio): 두 가중을 stop_ratio로 섞은 존재확률 가중
    """
    corridors: tuple[Corridor, ...]
    barriers: tuple[Barrier, ...]
    concealments: tuple[ConcealmentPatch, ...]
    background: float = 0.06     # 지형 밖 최소 통행성(야지 기동 허용, 그러나 낮음)
    contrast: float = 2.0        # 가중 대비를 키우는 지수(>1 이면 회랑/은폐에 더 집중)
    search_radius_m: float = 3333.0

    # --- mobility: 이동표적이 지나갈/머무를 통행성 ---
    def mobility_weight(self, x: float, y: float) -> float:
        corridor_gain = 0.0
        for corridor in self.corridors:
            corridor_gain = max(corridor_gain, corridor.weight(x, y))
        base = self.background + (1.0 - self.background) * corridor_gain
        suppression = 1.0
        for barrier in self.barriers:
            suppression *= barrier.suppression(x, y)
        return base * suppression

    # --- concealment: 정지표적이 은신할 선호 ---
    def concealment_weight(self, x: float, y: float) -> float:
        gain = 0.0
        for patch in self.concealments:
            gain = max(gain, patch.weight(x, y))
        base = self.background + (1.0 - self.background) * gain
        # 은폐지도 장애물(하천 등) 안이면 존재 불가
        suppression = 1.0
        for barrier in self.barriers:
            suppression *= barrier.suppression(x, y)
        return base * suppression

    def prior_weight(
        self,
        x: float,
        y: float,
        stop_ratio: float = 0.4,
        mode_probabilities: tuple[float, ...] | None = None,
    ) -> float:
        """모드 혼합 존재확률 가중.

        stop_ratio: 표적이 정지(은신) 상태일 비율. 나머지는 이동(통행성).
        MIXED 프로파일의 평균 HALT 점유율을 넘기면 자연스럽게 은폐 쪽으로 기운다.
        contrast 지수로 회랑·은폐 구역과 배경의 대비를 키운다.
        """
        if mode_probabilities is not None:
            if len(mode_probabilities) != 5 or any(
                value < 0.0 for value in mode_probabilities
            ):
                raise ValueError(
                    "mode_probabilities must contain five non-negative values"
                )
            total = sum(mode_probabilities)
            if total <= 0.0:
                raise ValueError("mode_probabilities must have positive mass")
            stop_ratio = mode_probabilities[0] / total
        if any(barrier.contains(x, y) for barrier in self.barriers):
            return 0.0
        mob = self.mobility_weight(x, y)
        con = self.concealment_weight(x, y)
        mixed = stop_ratio * con + (1.0 - stop_ratio) * mob
        return mixed ** self.contrast

    def mobility_gradient(self, x: float, y: float, step_m: float = 40.0) -> tuple[float, float]:
        """통행성이 증가하는 방향의 단위벡터(대략). 없으면 (0,0)."""
        gx = self.mobility_weight(x + step_m, y) - self.mobility_weight(x - step_m, y)
        gy = self.mobility_weight(x, y + step_m) - self.mobility_weight(x, y - step_m)
        norm = hypot(gx, gy)
        if norm <= 1e-9:
            return (0.0, 0.0)
        return (gx / norm, gy / norm)

    def concealment_affinity(self, x: float, y: float) -> float:
        """Return the local concealment preference on a 0-1 scale."""
        if any(barrier.contains(x, y) for barrier in self.barriers):
            return 0.0
        return min(
            1.0,
            max(
                (patch.weight(x, y) for patch in self.concealments),
                default=0.0,
            ),
        )

    def halt_probability_boost(
        self,
        x: float,
        y: float,
        maximum_boost: float,
    ) -> float:
        """Increase HALT propensity near concealment without fixing the mode."""
        if not 0.0 <= maximum_boost <= 1.0:
            raise ValueError("maximum_boost must be in [0, 1]")
        return maximum_boost * self.concealment_affinity(x, y)

    def transition_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        follow_strength: float = 0.5,
        offroad_probability: float = 0.2,
        rng=None,
        motion_mode: int | None = None,
    ) -> tuple[float, float]:
        """Apply corridor-following and hard-barrier constraints to one step.

        Corridor following aligns motion with the nearest centerline tangent
        and adds cross-track correction. A Bernoulli off-road branch preserves
        the unconstrained IMM displacement. Hard barriers constrain both
        branches; an intersecting step is redirected around the obstacle and
        blocked only when no collision-free detour is found.
        """
        del motion_mode
        if not 0.0 <= follow_strength <= 1.0:
            raise ValueError("follow_strength must be in [0, 1]")
        if not 0.0 <= offroad_probability <= 1.0:
            raise ValueError("offroad_probability must be in [0, 1]")
        distance = hypot(dx, dy)
        if distance <= 1e-9:
            return dx, dy

        direction_x = dx / distance
        direction_y = dy / distance
        choose_offroad = (
            offroad_probability > 0.0
            and rng is not None
            and rng.random() < offroad_probability
        )
        if not choose_offroad and self.corridors and follow_strength > 0.0:
            corridor = min(
                self.corridors,
                key=lambda item: item.distance_to_centerline(x, y),
            )
            projected_x, projected_y, _ = corridor.project(x, y)
            tangent_x, tangent_y = corridor.tangent()
            if tangent_x != 0.0 or tangent_y != 0.0:
                if tangent_x * direction_x + tangent_y * direction_y < 0.0:
                    tangent_x = -tangent_x
                    tangent_y = -tangent_y
                cross_x = projected_x - x
                cross_y = projected_y - y
                cross_distance = hypot(cross_x, cross_y)
                if cross_distance > 1e-9:
                    cross_x /= cross_distance
                    cross_y /= cross_distance
                cross_gain = min(
                    cross_distance / max(corridor.half_width_m, 1e-9),
                    1.0,
                )
                route_x = tangent_x + 0.75 * cross_gain * cross_x
                route_y = tangent_y + 0.75 * cross_gain * cross_y
                route_norm = hypot(route_x, route_y)
                if route_norm > 1e-9:
                    route_x /= route_norm
                    route_y /= route_norm
                    effective_strength = follow_strength * corridor.weight(x, y)
                    direction_x = (
                        (1.0 - effective_strength) * direction_x
                        + effective_strength * route_x
                    )
                    direction_y = (
                        (1.0 - effective_strength) * direction_y
                        + effective_strength * route_y
                    )
                    direction_norm = hypot(direction_x, direction_y)
                    if direction_norm > 1e-9:
                        direction_x /= direction_norm
                        direction_y /= direction_norm

        return self._avoid_barriers(
            x,
            y,
            distance * direction_x,
            distance * direction_y,
        )

    def _avoid_barriers(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
    ) -> tuple[float, float]:
        distance = hypot(dx, dy)
        if distance <= 1e-9:
            return dx, dy
        adjusted_x, adjusted_y = dx, dy
        for barrier in self.barriers:
            if not barrier.intersects_step(x, y, adjusted_x, adjusted_y):
                continue
            radial_x = x - barrier.center.x
            radial_y = y - barrier.center.y
            radial_norm = hypot(radial_x, radial_y)
            if radial_norm <= 1e-9 or barrier.contains(x, y):
                radial_x, radial_y = 1.0, 0.0
            else:
                radial_x /= radial_norm
                radial_y /= radial_norm
            current_x = adjusted_x / distance
            current_y = adjusted_y / distance
            tangents = ((-radial_y, radial_x), (radial_y, -radial_x))
            tangent_x, tangent_y = max(
                tangents,
                key=lambda tangent: tangent[0] * current_x + tangent[1] * current_y,
            )
            detour_x = tangent_x + 0.2 * radial_x
            detour_y = tangent_y + 0.2 * radial_y
            detour_norm = hypot(detour_x, detour_y)
            adjusted_x = distance * detour_x / detour_norm
            adjusted_y = distance * detour_y / detour_norm

        if any(
            barrier.intersects_step(x, y, adjusted_x, adjusted_y)
            for barrier in self.barriers
        ):
            return 0.0, 0.0
        return adjusted_x, adjusted_y

    def bias_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        bias_strength: float = 0.5,
        motion_mode: int | None = None,
    ) -> tuple[float, float]:
        """Backward-compatible deterministic corridor transition."""
        return self.transition_step(
            x,
            y,
            dx,
            dy,
            follow_strength=bias_strength,
            offroad_probability=0.0,
            motion_mode=motion_mode,
        )

    def observability_weight(self, x: float, y: float) -> float:
        """항상 1.0 — **이 모델의 지형은 센서에 작용하지 않는다** (선언된 범위).

        2026-09-06 결정. 지형은 belief 에만 들어간다.

            prior       셀 사전질량      은폐지에 표적이 있을 확률이 높다
            transition  입자 예측 커널    회랑을 따라 가고 장애물에서 멈춘다
            negative    음성관측 갱신     못 봤을 때 belief 를 어떻게 깎을지

        **계획 탐지율에는 안 들어간다.** 그래서 Koopman 배분의 관측성
        ``c_i`` 가 전 셀 1.0 이고, ``e_i = ln(p_i c_i/mu)/c_i`` 가
        ``ln(p_i/mu)`` 로 축약된다. ``use_terrain`` 을 켜고 꺼도
        ``markov.effectiveness`` 는 바뀌지 않는다 — 실측 확인.

        **이것은 측정 결과가 아니라 모델 범위다.** "지형 관측성은 효과가
        없다"고 읽으면 안 된다. 그 축이 애초에 움직일 수 없다.

        실제 센서층(DEM 가시선 + 지표피복 차폐)을 넣으려면 재료는 이미 있다 —
        ``sensor_observation.WORLDCOVER_OCCLUSION_NOMINAL`` 과
        ``WORLDCOVER_PD_*``. 물리가 바뀌므로 전 챕터 재실행이 필요하고,
        향후 과제로 남긴다.

        주의: ``TerrainAblation.observability`` 는 **다른 것**이다. 그쪽은
        음성관측 갱신에 ``SpatialDetectionModel`` 을 쓸지 여부이고, belief
        경로라 살아 있다(Chapter 4 가 측정한다).
        """

        del x, y
        return 1.0


class RasterTerrainField:
    """실제 DEM/WorldCover 격자를 기존 지형 인터페이스로 노출한다.

    ``tools/fetch_terrain.py``가 만든 NPZ는 남쪽에서 북쪽, 서쪽에서 동쪽
    순서의 정규 로컬 격자다. 연속 가중치와 고도는 쌍선형 보간하고,
    토지피복·통행불가 마스크는 최근접 셀을 사용한다.
    """

    def __init__(
        self,
        *,
        mobility: np.ndarray,
        concealment: np.ndarray,
        observability: np.ndarray,
        impassable: np.ndarray,
        elevation_m: np.ndarray,
        landcover_class: np.ndarray,
        resolution_m: float,
        search_radius_m: float,
        contrast: float = 2.0,
        metadata: dict | None = None,
    ) -> None:
        arrays = {
            "mobility": np.asarray(mobility, dtype=float),
            "concealment": np.asarray(concealment, dtype=float),
            "observability": np.asarray(observability, dtype=float),
            "impassable": np.asarray(impassable, dtype=bool),
            "elevation_m": np.asarray(elevation_m, dtype=float),
            "landcover_class": np.asarray(landcover_class, dtype=np.uint8),
        }
        shapes = {value.shape for value in arrays.values()}
        if len(shapes) != 1:
            raise ValueError("all raster terrain layers must have the same shape")
        shape = next(iter(shapes))
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError("raster terrain layers must be two-dimensional")
        if resolution_m <= 0.0 or search_radius_m <= 0.0:
            raise ValueError("resolution_m and search_radius_m must be positive")
        for key in ("mobility", "concealment", "observability"):
            values = arrays[key]
            if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
                raise ValueError(f"{key} values must be finite and in [0, 1]")
        if not np.isfinite(arrays["elevation_m"]).all():
            raise ValueError("elevation values must be finite")

        for name, value in arrays.items():
            value = value.copy()
            value.setflags(write=False)
            setattr(self, name, value)
        self.resolution_m = float(resolution_m)
        self.search_radius_m = float(search_radius_m)
        self.contrast = float(contrast)
        self.metadata = dict(metadata or {})
        height, width = shape
        self.x_min_m = -0.5 * (width - 1) * self.resolution_m
        self.y_min_m = -0.5 * (height - 1) * self.resolution_m
        self.x_max_m = -self.x_min_m
        self.y_max_m = -self.y_min_m

    @classmethod
    def load(cls, path: str | Path, metadata: dict | None = None) -> "RasterTerrainField":
        """Load a terrain NPZ and its adjacent JSON metadata."""

        source = Path(path)
        if metadata is None:
            from json import loads

            metadata_path = source.with_suffix(".json")
            metadata = loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(source, allow_pickle=False) as data:
            required = {
                "mobility",
                "concealment",
                "observability",
                "impassable",
                "elevation_m",
                "landcover_class",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"terrain archive is missing layers: {sorted(missing)}")
            layers = {name: data[name] for name in required}
        return cls(
            **layers,
            resolution_m=float(metadata["resolution_m"]),
            search_radius_m=float(metadata["radius_m"]),
            contrast=float(metadata.get("contrast", 2.0)),
            metadata=metadata,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.mobility.shape

    def _fractional_index(self, x: float, y: float) -> tuple[float, float] | None:
        column = (x - self.x_min_m) / self.resolution_m
        row = (y - self.y_min_m) / self.resolution_m
        height, width = self.shape
        if column < 0.0 or row < 0.0 or column > width - 1 or row > height - 1:
            return None
        return row, column

    def _nearest(self, layer: np.ndarray, x: float, y: float):
        position = self._fractional_index(x, y)
        if position is None:
            return None
        row, column = position
        return layer[int(round(row)), int(round(column))]

    def _bilinear(self, layer: np.ndarray, x: float, y: float) -> float | None:
        position = self._fractional_index(x, y)
        if position is None:
            return None
        row, column = position
        row0, column0 = int(row), int(column)
        row1 = min(row0 + 1, layer.shape[0] - 1)
        column1 = min(column0 + 1, layer.shape[1] - 1)
        fy, fx = row - row0, column - column0
        return float(
            layer[row0, column0] * (1.0 - fx) * (1.0 - fy)
            + layer[row0, column1] * fx * (1.0 - fy)
            + layer[row1, column0] * (1.0 - fx) * fy
            + layer[row1, column1] * fx * fy
        )

    def mobility_weight(self, x: float, y: float) -> float:
        value = self._bilinear(self.mobility, x, y)
        return 0.0 if value is None else value

    def concealment_weight(self, x: float, y: float) -> float:
        value = self._bilinear(self.concealment, x, y)
        return 0.0 if value is None else value

    def observability_weight(self, x: float, y: float) -> float:
        value = self._bilinear(self.observability, x, y)
        return 0.0 if value is None else value

    def prior_weight(
        self,
        x: float,
        y: float,
        stop_ratio: float = 0.4,
        mode_probabilities: tuple[float, ...] | None = None,
    ) -> float:
        if mode_probabilities is not None:
            if len(mode_probabilities) != 5 or any(
                value < 0.0 for value in mode_probabilities
            ):
                raise ValueError(
                    "mode_probabilities must contain five non-negative values"
                )
            total = sum(mode_probabilities)
            if total <= 0.0:
                raise ValueError("mode_probabilities must have positive mass")
            stop_ratio = mode_probabilities[0] / total
        if not 0.0 <= stop_ratio <= 1.0:
            raise ValueError("stop_ratio must be in [0, 1]")
        if bool(self._nearest(self.impassable, x, y)):
            return 0.0
        mixed = (
            stop_ratio * self.concealment_weight(x, y)
            + (1.0 - stop_ratio) * self.mobility_weight(x, y)
        )
        return mixed**self.contrast

    def mobility_gradient(
        self, x: float, y: float, step_m: float | None = None
    ) -> tuple[float, float]:
        step = float(step_m or self.resolution_m)
        gx = self.mobility_weight(x + step, y) - self.mobility_weight(x - step, y)
        gy = self.mobility_weight(x, y + step) - self.mobility_weight(x, y - step)
        norm = hypot(gx, gy)
        return (0.0, 0.0) if norm <= 1e-12 else (gx / norm, gy / norm)

    def concealment_affinity(self, x: float, y: float) -> float:
        return self.concealment_weight(x, y)

    def halt_probability_boost(self, x: float, y: float, maximum_boost: float) -> float:
        if not 0.0 <= maximum_boost <= 1.0:
            raise ValueError("maximum_boost must be in [0, 1]")
        return maximum_boost * self.concealment_affinity(x, y)

    def _path_clear(self, x: float, y: float, dx: float, dy: float) -> bool:
        distance = hypot(dx, dy)
        steps = max(1, int(distance / max(self.resolution_m * 0.5, 1.0)))
        for index in range(1, steps + 1):
            fraction = index / steps
            blocked = self._nearest(
                self.impassable,
                x + fraction * dx,
                y + fraction * dy,
            )
            if blocked is not None and bool(blocked):
                return False
        return True

    def transition_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        follow_strength: float = 0.5,
        offroad_probability: float = 0.2,
        rng=None,
        motion_mode: int | None = None,
    ) -> tuple[float, float]:
        del motion_mode
        if not 0.0 <= follow_strength <= 1.0:
            raise ValueError("follow_strength must be in [0, 1]")
        if not 0.0 <= offroad_probability <= 1.0:
            raise ValueError("offroad_probability must be in [0, 1]")
        distance = hypot(dx, dy)
        if distance <= 1e-9:
            return dx, dy

        candidate_x, candidate_y = dx, dy
        follows_terrain = not (
            rng is not None
            and offroad_probability > 0.0
            and rng.random() < offroad_probability
        )
        if follows_terrain and follow_strength > 0.0:
            gradient_x, gradient_y = self.mobility_gradient(x, y)
            if gradient_x != 0.0 or gradient_y != 0.0:
                direction_x, direction_y = dx / distance, dy / distance
                direction_x = (1.0 - follow_strength) * direction_x + follow_strength * gradient_x
                direction_y = (1.0 - follow_strength) * direction_y + follow_strength * gradient_y
                norm = hypot(direction_x, direction_y)
                if norm > 1e-12:
                    candidate_x = distance * direction_x / norm
                    candidate_y = distance * direction_y / norm
        if self._path_clear(x, y, candidate_x, candidate_y):
            return candidate_x, candidate_y

        angle = np.arctan2(candidate_y, candidate_x)
        alternatives: list[tuple[float, float, float]] = []
        for offset in (30, -30, 60, -60, 90, -90, 180):
            trial_angle = angle + np.radians(offset)
            trial_x = distance * float(np.cos(trial_angle))
            trial_y = distance * float(np.sin(trial_angle))
            if self._path_clear(x, y, trial_x, trial_y):
                score = self.mobility_weight(x + trial_x, y + trial_y)
                alternatives.append((score, trial_x, trial_y))
        if not alternatives:
            return 0.0, 0.0
        _, best_x, best_y = max(alternatives, key=lambda item: item[0])
        return best_x, best_y

    def bias_step(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        bias_strength: float = 0.5,
        motion_mode: int | None = None,
    ) -> tuple[float, float]:
        return self.transition_step(
            x,
            y,
            dx,
            dy,
            follow_strength=bias_strength,
            offroad_probability=0.0,
            motion_mode=motion_mode,
        )

    def elevation_at(self, x: float, y: float) -> float | None:
        return self._bilinear(self.elevation_m, x, y)

    def landcover_code_at(self, x: float, y: float) -> int | None:
        value = self._nearest(self.landcover_class, x, y)
        return None if value is None else int(value)

    def is_building(self, x: float, y: float) -> bool:
        return self.landcover_code_at(x, y) == 50


def build_synthetic_terrain(
    search_radius_m: float,
    seed: int = 20260817,
    corridor_count: int = 2,
    barrier_count: int = 1,
    concealment_count: int = 3,
    background: float = 0.06,
) -> TerrainField:
    """탐색 원 안에 재현 가능한 합성 지형을 생성한다.

    - 회랑: 원을 가로지르는 선형 축(도로/기동로).
    - 장애물: 원 내부 어딘가의 원형 차단(하천/호수).
    - 은폐 패치: 통행성 축에서 약간 떨어진 은신처(산림).
    """
    rng = Random(seed)
    R = search_radius_m

    # 회랑: 원 경계의 한 점에서 반대편 근처로 이어지는 선분 몇 개
    corridors: list[Corridor] = []
    for _ in range(corridor_count):
        a0 = rng.uniform(0.0, tau)
        # 반대편에서 ±60도 흔들어 직선 축이 아니게
        a1 = a0 + tau / 2.0 + rng.uniform(-tau / 6.0, tau / 6.0)
        start = Point2D(R * cos(a0), R * sin(a0))
        end = Point2D(R * cos(a1), R * sin(a1))
        half_width = rng.uniform(0.10, 0.16) * R
        corridors.append(Corridor(start, end, half_width_m=half_width, strength=1.0))

    # 장애물: 중심에서 벗어난 원형 차단
    barriers: list[Barrier] = []
    for _ in range(barrier_count):
        ang = rng.uniform(0.0, tau)
        rad = rng.uniform(0.25, 0.55) * R
        center = Point2D(rad * cos(ang), rad * sin(ang))
        barriers.append(Barrier(center, radius_m=rng.uniform(0.12, 0.20) * R,
                                softness_m=0.05 * R))

    # 은폐 패치: 회랑 근처지만 정확히 겹치지 않는 은신처
    concealments: list[ConcealmentPatch] = []
    for _ in range(concealment_count):
        ang = rng.uniform(0.0, tau)
        rad = rng.uniform(0.35, 0.85) * R
        center = Point2D(rad * cos(ang), rad * sin(ang))
        concealments.append(ConcealmentPatch(center, radius_m=rng.uniform(0.10, 0.18) * R,
                                             strength=1.0))

    return TerrainField(
        corridors=tuple(corridors),
        barriers=tuple(barriers),
        concealments=tuple(concealments),
        background=background,
        search_radius_m=search_radius_m,
    )


def build_control_terrain(
    search_radius_m: float,
    seed: int = 20260817,
    patch_count: int = 5,
    background: float = 0.06,
) -> TerrainField:
    """대조군 지형: 방향성 회랑 없이 임의 위치의 은폐 패치만 배치.

    build_synthetic_terrain과 '집중도'는 맞추되(비슷한 수의 좁은 고밀도
    구역), 선형 회랑이 없어 이동 방향 정보를 주지 않는다. 표적은 이 패치들에
    몰려 있지만 등방성으로 퍼지므로, "위치 집중"은 같고 "방향 예측 가능성"만
    빠진 조건이 된다. 지형의 방향정보가 주는 순수 가치를 분리하기 위한 대조.
    """
    rng = Random(seed)
    R = search_radius_m

    # 회랑(2개)이 차지하던 고통행성 면적을, 방향 없는 패치 여러 개로 대체.
    # 원 지형의 concealment 3개 + 회랑 대체분 → 총 patch_count개.
    concealments: list[ConcealmentPatch] = []
    for _ in range(patch_count):
        ang = rng.uniform(0.0, tau)
        rad = rng.uniform(0.20, 0.85) * R
        center = Point2D(rad * cos(ang), rad * sin(ang))
        concealments.append(
            ConcealmentPatch(center, radius_m=rng.uniform(0.10, 0.18) * R, strength=1.0)
        )

    # 장애물은 A와 동일한 통계로 1개 배치(공정성).
    ang = rng.uniform(0.0, tau)
    rad = rng.uniform(0.25, 0.55) * R
    barriers = (
        Barrier(
            Point2D(rad * cos(ang), rad * sin(ang)),
            radius_m=rng.uniform(0.12, 0.20) * R,
            softness_m=0.05 * R,
        ),
    )

    return TerrainField(
        corridors=(),                 # 회랑 없음 = 방향정보 없음
        barriers=barriers,
        concealments=tuple(concealments),
        background=background,
        search_radius_m=search_radius_m,
    )
