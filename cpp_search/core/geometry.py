"""극좌표 <-> 월드좌표 변환.

    x = cx + r * cos(theta),   y = cy + r * sin(theta)

``probability.PolarProbabilityMap.build``가 셀 중심 좌표를 만들 때 쓴다.

의존: ``models``만 참조. ``probability``가 이 파일을 쓴다.
"""

from __future__ import annotations

from math import cos, pi, radians, sin

from cpp_search.core.models import MissionConfig, Point2D


def sector_center_angle(mission: MissionConfig, vehicle_id: int) -> float:
    if not 0 <= vehicle_id < mission.uav_count:
        raise ValueError("vehicle_id is outside the configured fleet")
    return radians(mission.sector_offset_deg) + vehicle_id * 2.0 * pi / mission.uav_count


def local_to_world(local: Point2D, origin: Point2D, angle_rad: float) -> Point2D:
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    return Point2D(
        origin.x + local.x * cos_a - local.y * sin_a,
        origin.y + local.x * sin_a + local.y * cos_a,
    )


def polar_to_world(
    radius_m: float, angle_rad: float, origin: Point2D
) -> Point2D:
    return Point2D(
        origin.x + radius_m * cos(angle_rad),
        origin.y + radius_m * sin(angle_rad),
    )

