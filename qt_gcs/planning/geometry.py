from __future__ import annotations

import math
from dataclasses import dataclass


EARTH_METERS_PER_DEGREE = 111_320.0


@dataclass(frozen=True, slots=True)
class LocalPoint:
    """East/north point in metres in one mission-local tangent frame."""

    east_m: float
    north_m: float

    def distance_to(self, other: "LocalPoint") -> float:
        return math.hypot(
            other.east_m - self.east_m,
            other.north_m - self.north_m,
        )


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """Small-area WGS84/local conversion used by the runtime planner.

    The GCS demonstration operates over tens of kilometres, so the same local
    tangent approximation already used by the map and track predictor is both
    deterministic and sufficiently accurate for the rule-based simulation.
    """

    reference_latitude: float
    reference_longitude: float

    @property
    def longitude_scale(self) -> float:
        return max(
            10_000.0,
            EARTH_METERS_PER_DEGREE
            * math.cos(math.radians(self.reference_latitude)),
        )

    def to_local(self, latitude: float, longitude: float) -> LocalPoint:
        return LocalPoint(
            (float(longitude) - self.reference_longitude)
            * self.longitude_scale,
            (float(latitude) - self.reference_latitude)
            * EARTH_METERS_PER_DEGREE,
        )

    def to_geographic(self, point: LocalPoint) -> tuple[float, float]:
        return (
            self.reference_latitude
            + point.north_m / EARTH_METERS_PER_DEGREE,
            self.reference_longitude
            + point.east_m / self.longitude_scale,
        )


def offset_point(
    point: LocalPoint,
    distance_m: float,
    heading_deg: float,
) -> LocalPoint:
    heading_rad = math.radians(float(heading_deg))
    return LocalPoint(
        point.east_m + math.sin(heading_rad) * float(distance_m),
        point.north_m + math.cos(heading_rad) * float(distance_m),
    )


def distance_to_segment(
    point: LocalPoint,
    start: LocalPoint,
    end: LocalPoint,
) -> float:
    dx = end.east_m - start.east_m
    dy = end.north_m - start.north_m
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return point.distance_to(start)
    fraction = (
        (point.east_m - start.east_m) * dx
        + (point.north_m - start.north_m) * dy
    ) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    closest = LocalPoint(
        start.east_m + fraction * dx,
        start.north_m + fraction * dy,
    )
    return point.distance_to(closest)


def point_along_polyline(
    points: list[LocalPoint] | tuple[LocalPoint, ...],
    distance_m: float,
) -> LocalPoint:
    if not points:
        raise ValueError("polyline must contain at least one point")
    if len(points) == 1 or distance_m <= 0.0:
        return points[0]
    remaining = float(distance_m)
    for start, end in zip(points, points[1:]):
        segment_length = start.distance_to(end)
        if segment_length <= 1e-9:
            continue
        if remaining <= segment_length:
            fraction = remaining / segment_length
            return LocalPoint(
                start.east_m
                + (end.east_m - start.east_m) * fraction,
                start.north_m
                + (end.north_m - start.north_m) * fraction,
            )
        remaining -= segment_length
    return points[-1]


def polyline_length(
    points: list[LocalPoint] | tuple[LocalPoint, ...],
) -> float:
    return sum(start.distance_to(end) for start, end in zip(points, points[1:]))
