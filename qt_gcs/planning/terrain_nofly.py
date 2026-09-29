"""런타임: baked DEM(NPZ) → SPX 셀별 H_eff + no-fly 마스크 (numpy 만).

``bake_terrain_from_mission.py`` 가 만든 RasterTerrainField NPZ 를 읽어, SPX 격자
셀 중심(로컬 ENU 미터)마다 지면고도를 표집하고, 고정-MSL 가정으로 센서-지면
높이 H_eff 와 no-fly 마스크를 돌려준다 ([[gcs-v05-terrain-altitude-decision]]).

SPX 연결 지점:
  1) ``no_fly`` True 셀을 ``stone_spx._adjacency`` 에서 끊는다(arc 제거) → 우회.
  2) no-fly 셀의 belief 질량은 탐색 불가 → 달성가능 탐지확률 천장(정직하게 보고).

cpp_search 가 이미 v0.5 에 벤더링돼 있으면 core.terrain.RasterTerrainField 를
그대로 써도 되지만(동일 elevation_at), 런타임 의존을 numpy 로만 두려고 여기서는
독립 NPZ 로더 + 쌍선형 표집을 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BakedTerrain:
    """RasterTerrainField 포맷 NPZ 의 최소 런타임 로더 (elevation 표집만)."""

    def __init__(self, elevation_m: np.ndarray, resolution_m: float, metadata: dict | None = None):
        self.elevation_m = np.asarray(elevation_m, dtype=float)
        if self.elevation_m.ndim != 2 or min(self.elevation_m.shape) < 2:
            raise ValueError("elevation_m must be a 2-D grid")
        self.resolution_m = float(resolution_m)
        self.metadata = dict(metadata or {})
        height, width = self.elevation_m.shape
        # RasterTerrainField 와 동일한 중심대칭 로컬 프레임.
        self.x_min_m = -0.5 * (width - 1) * self.resolution_m
        self.y_min_m = -0.5 * (height - 1) * self.resolution_m

    @classmethod
    def load(cls, npz_path: str | Path) -> "BakedTerrain":
        source = Path(npz_path)
        meta_path = source.with_suffix(".json")
        metadata = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else {}
        )
        with np.load(source, allow_pickle=False) as data:
            elevation = data["elevation_m"]
        resolution = float(metadata.get("resolution_m", 30.0))
        return cls(elevation, resolution, metadata)

    def elevation_at(self, x_m, y_m):
        """로컬 ENU (east=x, north=y) 미터에서 지면고도(MSL) 쌍선형 표집. 벡터화."""
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        h, w = self.elevation_m.shape
        col = (x - self.x_min_m) / self.resolution_m
        row = (y - self.y_min_m) / self.resolution_m
        c0 = np.clip(np.floor(col).astype(int), 0, w - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, h - 2)
        fc = np.clip(col - c0, 0.0, 1.0)
        fr = np.clip(row - r0, 0.0, 1.0)
        g = self.elevation_m
        top = g[r0, c0] * (1 - fc) + g[r0, c0 + 1] * fc
        bot = g[r0 + 1, c0] * (1 - fc) + g[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr


def cell_geometry_from_dem(
    terrain: BakedTerrain,
    cell_centers_xy_m,
    *,
    flight_msl_m: float = 600.0,
    min_clearance_agl_m: float = 120.0,
):
    """SPX 셀 중심(로컬 ENU (x,y) m) 배열 → (elevation, H_eff, no_fly).

    ``cell_centers_xy_m``: shape (N, 2) 또는 (x_array, y_array). 반환 배열은 셀 순서.

        H_eff  = flight_msl - elevation
        no_fly = H_eff < min_clearance_agl   (지형이 비행고도에 근접/초과)
    """
    centers = np.asarray(cell_centers_xy_m, dtype=float)
    if centers.ndim == 2 and centers.shape[1] == 2:
        x, y = centers[:, 0], centers[:, 1]
    else:
        x, y = centers  # (x_array, y_array)
    elevation = terrain.elevation_at(x, y)
    h_eff = float(flight_msl_m) - elevation
    no_fly = h_eff < float(min_clearance_agl_m)
    return elevation, h_eff, no_fly


def apply_no_fly_to_adjacency(adjacency: np.ndarray, no_fly: np.ndarray) -> np.ndarray:
    """no-fly 셀로 들어가고 나가는 arc 를 모두 끊는다 (stone_spx._adjacency 산출물).

    ``adjacency`` 는 (cell_count, cell_count) bool (또는 outside 열 포함). no-fly 는
    길이 cell_count bool. 반환은 새 배열(원본 불변)."""
    result = np.array(adjacency, dtype=bool, copy=True)
    n = no_fly.shape[0]
    result[:n, :][no_fly, :] = False
    result[:, :n][:, no_fly] = False
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baked DEM → SPX cell H_eff/no-fly probe.")
    parser.add_argument("npz", type=Path, help="baked terrain NPZ")
    parser.add_argument("--flight-msl", type=float, default=600.0)
    parser.add_argument("--min-clearance-agl", type=float, default=120.0)
    parser.add_argument("--grid", type=int, default=10, help="probe an NxN cell grid over the AOI")
    args = parser.parse_args()

    terrain = BakedTerrain.load(args.npz)
    radius = float(terrain.metadata.get("radius_m", -terrain.x_min_m))
    # NxN 셀 중심을 AOI 에 균등 배치 (실제로는 stone_spx grid.center_of 를 쓴다).
    edges = np.linspace(-radius, radius, args.grid + 1)
    centers_1d = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers_1d, centers_1d)
    elev, h_eff, no_fly = cell_geometry_from_dem(
        terrain, np.column_stack([xx.ravel(), yy.ravel()]),
        flight_msl_m=args.flight_msl, min_clearance_agl_m=args.min_clearance_agl,
    )
    print(f"grid {args.grid}x{args.grid} over radius {radius:.0f} m @ MSL {args.flight_msl:.0f} m")
    print(f"  elevation  {elev.min():.1f} ~ {elev.max():.1f} m")
    print(f"  H_eff      {h_eff.min():.1f} ~ {h_eff.max():.1f} m")
    print(f"  no-fly cells: {int(no_fly.sum())} / {no_fly.size} "
          f"(clearance floor {args.min_clearance_agl:.0f} m → 지형 > "
          f"{args.flight_msl - args.min_clearance_agl:.0f} m 이면 no-fly)")
