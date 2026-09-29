"""Parallel ingress-sweep formation geometry (Stage 2).

The six vehicles do not converge on the predicted subject point (TP). They fly a
**fixed heading** (the launcher->TP azimuth), abreast, spaced ``spacing_m`` apart
on a line perpendicular to that heading, so their sensors sweep a corridor while
transiting to the search area. The ingress-sweep phase ends when the formation
line reaches the TP-crossing line (perpendicular to the heading through TP);
after that the SPX plan takes over.

This module is pure geometry (no Qt, no scipy) so it is unit-testable headless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_METERS_PER_DEGREE = 111_320.0


def _lon_scale(latitude: float) -> float:
    return max(10_000.0, EARTH_METERS_PER_DEGREE * math.cos(math.radians(latitude)))


def _to_local(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Geographic -> local ENU metres (east, north) about an origin."""
    east = (lon - origin_lon) * _lon_scale(origin_lat)
    north = (lat - origin_lat) * EARTH_METERS_PER_DEGREE
    return east, north


def _to_geographic(east: float, north: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    lat = origin_lat + north / EARTH_METERS_PER_DEGREE
    lon = origin_lon + east / _lon_scale(origin_lat)
    return lat, lon


@dataclass(frozen=True)
class IngressFormation:
    """Result of :func:`build_parallel_ingress`."""

    #: vehicle_id -> ordered (lat, lon, alt) ingress waypoints
    routes: dict[int, list[tuple[float, float, float]]]
    #: launcher->TP azimuth in degrees (the fixed parallel heading)
    heading_deg: float
    #: total formation front width in metres (== (count-1) * spacing_m)
    front_m: float
    #: per-vehicle lateral offsets (metres) along the perpendicular, launcher-frame order
    offsets_m: tuple[float, ...]
    #: vehicle_id -> (lat, lon, alt) launch-formation point: a tight cluster at
    #: the launcher (compressed lateral offset, lane order) that fans out to the
    #: abreast line on the way in, so the fleet emerges from LC rather than
    #: starting pre-spread in parallel
    launch_positions: dict[int, tuple[float, float, float]] = None  # type: ignore[assignment]


def build_parallel_ingress(
    launcher_lat: float,
    launcher_lon: float,
    tp_lat: float,
    tp_lon: float,
    *,
    count: int = 6,
    spacing_m: float = 800.0,
    altitude_m: float = 600.0,
    formup_fraction: float = 0.3,
    launch_cluster_ratio: float = 0.15,
) -> IngressFormation:
    """Build a parallel abreast ingress sweep from the launcher to the TP line.

    Each vehicle gets two waypoints: a *form-up* point where the abreast line is
    established (``formup_fraction`` of the way in) and the *TP-line* point where
    the ingress sweep ends (on the perpendicular-through-TP line, at that
    vehicle's lateral offset). The segment between them is parallel to the
    launcher->TP heading, so the six vehicles sweep a corridor of width
    ``(count - 1) * spacing_m`` without converging on TP.

    Vehicle ids are ``1..count`` assigned left-to-right across the front.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")
    if not 0.0 <= formup_fraction < 1.0:
        raise ValueError("formup_fraction must be in [0, 1)")

    # Local frame centred on the launcher.
    tp_e, tp_n = _to_local(tp_lat, tp_lon, launcher_lat, launcher_lon)
    axis_len = math.hypot(tp_e, tp_n)
    if axis_len < 1.0:
        raise ValueError("launcher and TP are effectively coincident")
    # Unit heading (launcher->TP) and its left perpendicular.
    hx, hy = tp_e / axis_len, tp_n / axis_len
    px, py = -hy, hx
    heading_deg = math.degrees(math.atan2(tp_e, tp_n)) % 360.0  # compass bearing

    offsets = tuple(
        (index - (count - 1) / 2.0) * spacing_m for index in range(count)
    )
    formup_along = formup_fraction * axis_len

    routes: dict[int, list[tuple[float, float, float]]] = {}
    launch_positions: dict[int, tuple[float, float, float]] = {}
    for index, offset in enumerate(offsets):
        vehicle_id = index + 1
        # launch-formation point: a TIGHT cluster at the launcher (lateral offset
        # compressed by ``launch_cluster_ratio``, still in lane order).  The fleet
        # thus EMERGES from the launch site and FANS OUT to the abreast line on
        # the way to the form-up point, instead of starting pre-spread in
        # parallel.  Lane order is preserved, so the fan-out never crosses.
        le = px * offset * launch_cluster_ratio
        ln = py * offset * launch_cluster_ratio
        launch = _to_geographic(le, ln, launcher_lat, launcher_lon)
        # form-up point: partway in, already spread onto the abreast line
        fe = hx * formup_along + px * offset
        fn = hy * formup_along + py * offset
        # TP-line point: at TP along-track distance, same lateral offset
        te = hx * axis_len + px * offset
        tn = hy * axis_len + py * offset
        formup = _to_geographic(fe, fn, launcher_lat, launcher_lon)
        tp_line = _to_geographic(te, tn, launcher_lat, launcher_lon)
        launch_positions[vehicle_id] = (launch[0], launch[1], float(altitude_m))
        routes[vehicle_id] = [
            (formup[0], formup[1], float(altitude_m)),
            (tp_line[0], tp_line[1], float(altitude_m)),
        ]

    return IngressFormation(
        routes=routes,
        heading_deg=heading_deg,
        front_m=(count - 1) * spacing_m,
        offsets_m=offsets,
        launch_positions=launch_positions,
    )


def ingress_corridor_polygon(
    launcher_lat: float,
    launcher_lon: float,
    tp_lat: float,
    tp_lon: float,
    *,
    count: int = 6,
    spacing_m: float = 800.0,
    formup_fraction: float = 0.3,
    buffer_m: float = 400.0,
) -> list[tuple[float, float]]:
    """Return the 4 corner vertices (lat, lon) of the ingress sweep corridor.

    The corridor is a parallelogram from the formup line to the TP line,
    extending *buffer_m* beyond the outermost vehicles on each side for
    sensor coverage.  Vertices are in clockwise winding order.
    """
    tp_e, tp_n = _to_local(tp_lat, tp_lon, launcher_lat, launcher_lon)
    axis_len = math.hypot(tp_e, tp_n)
    if axis_len < 1.0:
        return []
    hx, hy = tp_e / axis_len, tp_n / axis_len
    px, py = -hy, hx

    half_width = (count - 1) / 2.0 * spacing_m + buffer_m
    formup_along = formup_fraction * axis_len

    corners_local = [
        (hx * formup_along - px * half_width, hy * formup_along - py * half_width),
        (hx * formup_along + px * half_width, hy * formup_along + py * half_width),
        (hx * axis_len + px * half_width, hy * axis_len + py * half_width),
        (hx * axis_len - px * half_width, hy * axis_len - py * half_width),
    ]
    return [
        _to_geographic(e, n, launcher_lat, launcher_lon)
        for e, n in corners_local
    ]
