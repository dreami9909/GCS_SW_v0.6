"""미션 JSON → 실제 DEM(COP30/SRTM) → RasterTerrainField NPZ 베이커 (오프라인 준비).

역할
----
GCS 미션 파일(saudi_desert_mission.json 등)의 AOI(중심 위경도 + 탐색반경)를 읽어,
그 영역의 실제 지형고도를 로컬 ENU 격자로 굽고, cpp_search 의
``core.terrain.RasterTerrainField.load`` 가 그대로 읽는 **NPZ + metadata JSON** 으로
저장한다. 이 산출물이 v0.5 런타임에서 셀별 지면고도 → H_eff → no-fly 마스크의
입력이 된다 ([[gcs-v05-terrain-altitude-decision]] 고정-MSL 600m).

무거운 GIS 의존성(rasterio)은 **이 준비 단계에만** 필요하다. 런타임(GCS)은
numpy 로 NPZ 만 읽는다.

좌표 규약
--------
두 저장소와 동일한 평면근사(EARTH_METERS_PER_DEGREE = 111320)를 쓴다. 로컬 ENU
격자점(east,north)을 위경도로 역변환해 DEM 을 쌍선형 표집한다::

    lat = center_lat + north / 111320
    lon = center_lon + east  / (111320 * cos(center_lat))

격자는 RasterTerrainField 규약대로 남→북(row 증가=북), 서→동(col 증가=동),
중심이 (0,0), 간격 resolution_m.

DEM 소스
-------
1) 로컬 GeoTIFF (``--geotiff PATH``) — 가장 확실. 먼저 한 번 받아두면 재현 가능.
   Copernicus GLO-30 예시 (AWS Open Data, 인증 불필요, 타일 1°x1°)::

       # N23 E047 타일 (사우디 시나리오 23.74N 47.33E)
       aws s3 cp --no-sign-request \\
         s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N23_00_E047_00_DEM/Copernicus_DSM_COG_10_N23_00_E047_00_DEM.tif \\
         cop30_N23E047.tif

2) OpenTopography API (``--opentopo``, 무료 API key 필요) — bbox 클립 GeoTIFF 자동 취득::

       demtype=COP30 (또는 SRTMGL1), south/north/west/east=bbox, outputFormat=GTiff

실행::

    python3 bake_terrain_from_mission.py saudi_desert_mission.json \\
        --geotiff cop30_N23E047.tif --resolution 30 --out terrain/saudi.npz
    # 지오메트리/포맷만 검증 (rasterio·네트워크 불필요):
    python3 bake_terrain_from_mission.py saudi_desert_mission.json --selftest --out /tmp/selftest.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EARTH_METERS_PER_DEGREE = 111_320.0


# --------------------------------------------------------------------------- #
# 미션 AOI                                                                     #
# --------------------------------------------------------------------------- #
def read_mission_aoi(path: str | Path) -> tuple[float, float, float]:
    """미션 JSON → (center_lat, center_lon, search_radius_m).

    탐색 중심은 ``mission.arc_search_pattern.center`` 를 우선 쓰고, 없으면
    ``mission.rally_predicted_subject`` 를 쓴다. 반경은 ``mission.search_radius_m``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mission = data.get("mission", {})
    center = (
        mission.get("arc_search_pattern", {}).get("center")
        or mission.get("rally_predicted_subject")
    )
    if not center:
        raise ValueError("미션에 arc_search_pattern.center / rally_predicted_subject 가 없다")
    radius = float(mission.get("search_radius_m") or data.get("search_radius_m") or 0.0)
    if radius <= 0.0:
        raise ValueError("mission.search_radius_m 가 없거나 0 이다")
    return float(center["latitude"]), float(center["longitude"]), radius


# --------------------------------------------------------------------------- #
# 로컬 ENU 격자                                                                #
# --------------------------------------------------------------------------- #
def build_local_axes(radius_m: float, resolution_m: float, *, margin_ratio: float = 0.05):
    """중심(0,0) 기준 대칭 ENU 축 (east, north). RasterTerrainField 규약과 동일."""
    half = radius_m * (1.0 + margin_ratio)
    n = int(math.ceil(2.0 * half / resolution_m))
    if n % 2 == 0:
        n += 1  # 중심 셀이 정확히 (0,0) 에 오도록 홀수로
    x_min = -0.5 * (n - 1) * resolution_m
    axis = x_min + resolution_m * np.arange(n, dtype=float)
    return axis, axis  # east_axis, north_axis (정사각)


def local_grid_to_lonlat(east_axis, north_axis, center_lat, center_lon):
    """격자점(east,north) → (lon_grid, lat_grid), shape (H, W). row=north, col=east."""
    lon_scale = max(1e-6, EARTH_METERS_PER_DEGREE * math.cos(math.radians(center_lat)))
    east_grid, north_grid = np.meshgrid(east_axis, north_axis)  # (H,W)
    lat_grid = center_lat + north_grid / EARTH_METERS_PER_DEGREE
    lon_grid = center_lon + east_grid / lon_scale
    return lon_grid, lat_grid


# --------------------------------------------------------------------------- #
# DEM 표집 (쌍선형, 순수 numpy)                                                #
# --------------------------------------------------------------------------- #
def _bilinear_sample(grid: np.ndarray, cols: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """정규 픽셀 격자 ``grid[row, col]`` 을 소수 좌표(cols, rows)에서 쌍선형 표집."""
    h, w = grid.shape
    c0 = np.clip(np.floor(cols).astype(int), 0, w - 2)
    r0 = np.clip(np.floor(rows).astype(int), 0, h - 2)
    fc = np.clip(cols - c0, 0.0, 1.0)
    fr = np.clip(rows - r0, 0.0, 1.0)
    v00 = grid[r0, c0]
    v01 = grid[r0, c0 + 1]
    v10 = grid[r0 + 1, c0]
    v11 = grid[r0 + 1, c0 + 1]
    top = v00 * (1 - fc) + v01 * fc
    bot = v10 * (1 - fc) + v11 * fc
    return top * (1 - fr) + bot * fr


def sample_geotiff_elevation(geotiff_path, lon_grid, lat_grid) -> np.ndarray:
    """EPSG:4326 GeoTIFF 를 위경도 격자에서 표집해 elevation_m 배열 반환.

    rasterio 가 필요하다(지연 import). DEM 픽셀은 도(degree) 단위 affine 이라
    (lon,lat) → (col,row) 역변환 후 쌍선형 표집한다.
    """
    try:
        import rasterio  # noqa: F401
        from rasterio.transform import rowcol
    except ImportError as error:  # pragma: no cover
        raise SystemExit(
            "rasterio 가 필요하다: pip install rasterio  (베이커 전용, 런타임엔 불필요)"
        ) from error

    with rasterio.open(geotiff_path) as dataset:
        band = dataset.read(1).astype(float)
        transform = dataset.transform
        nodata = dataset.nodata
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
        if np.isnan(band).any():  # nodata 를 최근접 유효값으로 채움(사막이라 드묾)
            band = _fill_nan(band)

    # affine 역변환: (lon,lat) → (col,row), 소수 좌표
    inv = ~transform
    cols = inv.a * lon_grid + inv.b * lat_grid + inv.c
    rows = inv.d * lon_grid + inv.e * lat_grid + inv.f
    return _bilinear_sample(band, cols, rows)


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return np.zeros_like(out)
    out[mask] = np.nanmean(out)
    return out


# --------------------------------------------------------------------------- #
# 자기검증용 합성 DEM (rasterio·네트워크 없이 지오메트리 확인)                 #
# --------------------------------------------------------------------------- #
def synthetic_elevation(lon_grid, lat_grid, center_lat, center_lon, *, base_m=250.0):
    """중심에서 멀어질수록 완만히 상승 + 한쪽에 600m 넘는 능선 하나(테스트용)."""
    lon_scale = EARTH_METERS_PER_DEGREE * math.cos(math.radians(center_lat))
    east = (lon_grid - center_lon) * lon_scale
    north = (lat_grid - center_lat) * EARTH_METERS_PER_DEGREE
    r = np.hypot(east, north)
    elev = base_m + 0.01 * r  # 완만한 사막 경사
    ridge = 500.0 * np.exp(-(((east - 2500.0) ** 2 + north ** 2) / (2 * 700.0 ** 2)))
    return elev + ridge  # 능선 마루 ~ base+500 > 600m → no-fly 테스트


# --------------------------------------------------------------------------- #
# 굽기                                                                         #
# --------------------------------------------------------------------------- #
def bake(
    mission_path,
    out_path,
    *,
    resolution_m: float = 30.0,
    geotiff: str | None = None,
    selftest: bool = False,
    impassable_slope_deg: float | None = None,
) -> dict:
    center_lat, center_lon, radius_m = read_mission_aoi(mission_path)
    east_axis, north_axis = build_local_axes(radius_m, resolution_m)
    lon_grid, lat_grid = local_grid_to_lonlat(east_axis, north_axis, center_lat, center_lon)

    if selftest:
        elevation = synthetic_elevation(lon_grid, lat_grid, center_lat, center_lon)
    elif geotiff:
        elevation = sample_geotiff_elevation(geotiff, lon_grid, lat_grid)
    else:
        raise SystemExit("--geotiff PATH 또는 --selftest 중 하나가 필요하다")

    elevation = np.asarray(elevation, dtype=float)
    shape = elevation.shape

    # DEM-only: 관측성/통행성 중립(지형은 belief 에만, 센서 관측성은 1.0 규약).
    mobility = np.ones(shape, dtype=float)
    concealment = np.zeros(shape, dtype=float)
    observability = np.ones(shape, dtype=float)
    landcover_class = np.zeros(shape, dtype=np.uint8)
    impassable = np.zeros(shape, dtype=bool)
    if impassable_slope_deg is not None:
        gy, gx = np.gradient(elevation, resolution_m)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        impassable = slope_deg > float(impassable_slope_deg)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mobility=mobility,
        concealment=concealment,
        observability=observability,
        impassable=impassable,
        elevation_m=elevation,
        landcover_class=landcover_class,
    )
    metadata = {
        "resolution_m": float(resolution_m),
        "radius_m": float(radius_m),
        "contrast": 2.0,
        "center_latitude": center_lat,
        "center_longitude": center_lon,
        "grid_shape": [int(shape[0]), int(shape[1])],
        "source": "selftest-synthetic" if selftest else f"geotiff:{geotiff}",
        "elevation_min_m": float(np.min(elevation)),
        "elevation_max_m": float(np.max(elevation)),
        "note": (
            "RasterTerrainField.load 호환. elevation_m 은 로컬 ENU(남→북·서→동) "
            "지면고도(MSL). observability=1.0 (지형은 belief 에만)."
        ),
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mission AOI → real-DEM RasterTerrainField NPZ.")
    parser.add_argument("mission", help="GCS 미션 JSON 경로")
    parser.add_argument("--geotiff", help="EPSG:4326 DEM GeoTIFF (COP30/SRTM)")
    parser.add_argument("--resolution", type=float, default=30.0, help="격자 간격 m (기본 30)")
    parser.add_argument("--impassable-slope-deg", type=float, default=None,
                        help="이 경사(도) 초과 셀을 impassable 로 표시")
    parser.add_argument("--selftest", action="store_true",
                        help="합성 DEM 으로 지오메트리/포맷만 검증 (rasterio 불필요)")
    parser.add_argument("--out", type=Path, default=Path("terrain/terrain.npz"))
    args = parser.parse_args(argv)

    meta = bake(
        args.mission, args.out,
        resolution_m=args.resolution, geotiff=args.geotiff,
        selftest=args.selftest, impassable_slope_deg=args.impassable_slope_deg,
    )
    print(f"wrote {args.out}  (+ {args.out.with_suffix('.json').name})")
    print(f"  center=({meta['center_latitude']:.5f},{meta['center_longitude']:.5f}) "
          f"radius={meta['radius_m']:.0f} m  grid={meta['grid_shape']} @ {meta['resolution_m']:.0f} m")
    print(f"  elevation {meta['elevation_min_m']:.1f} ~ {meta['elevation_max_m']:.1f} m MSL")


if __name__ == "__main__":
    main()
