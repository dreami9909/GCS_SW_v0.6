"""셀 배정 -> 실제 비행경로. 계획의 마지막 한 칸.

SPX 든 MAPPO 든 결정하는 것은 "어느 시간 슬라이스에 어느 셀에 있을지"까지다.
그 셀 안에서 무엇을 그리는지는 계획법과 무관한 공통 변환이고, 이 모듈이 그
변환 **하나만** 담당한다. 두 계획법이 같은 변환을 쓰기 때문에 Monte Carlo
KPI 차이가 경로기하 차이가 아니라 셀 선택 차이로 귀속된다.

    할당 셀 중심 + 중심선 예산 L  ->  평행소인 블록 (n_track x leg)

        n_track = round( sqrt(L / s) ),    leg = L / n_track

``n_track`` 을 이렇게 잡으면 블록 두 변이 모두 ``sqrt(L*s)`` 에 가까워져
정사각형이 된다 = 같은 중심선 길이로 **새로 덮는 면적이 최대**다. 짧은 선분을
왕복하는 셔틀은 같은 땅을 다시 덮으므로 쓰지 않는다. 소인 방향은 접선(반경에
수직)이라 블록이 자기 고리 안에 머문다.

시간 예산 회계
--------------
``n_track * leg = L`` 로만 잡으면 트랙 사이 연결이동과 셀 중심에서 첫 트랙까지
가는 이동이 회계에서 빠진다. 그만큼 실제 소요시간이 예산을 넘고, 초과분은
평가 단계의 시간 절단이 잘라내므로 **계획의 어디가 버려질지를 세그먼트 순서가
정하게 된다**. 그래서 만들어진 세그먼트의 실제 소요시간을 직접 계산해 맞춘다
(소인 구간은 중심선 탐색속도, 이동 구간은 이동속도).

의존
----
* 위: ``cpp_search.core.models``.
* 아래: ``planning/stone_spx``, ``learning/spx_policy``.
"""

from __future__ import annotations

from math import cos, sin, sqrt

from cpp_search.core.models import MissionConfig, PathSegment, Point2D, SensorSpec


def local_sweep(
    center: Point2D,
    centerline_budget_m: float,
    mission: MissionConfig,
    sensor: SensorSpec,
    *,
    phase_rad: float,
) -> tuple[list[PathSegment], Point2D]:
    """할당 셀 위의 평행소인. 시간 예산 안에 들어오도록 크기를 맞춘다."""

    if centerline_budget_m <= 1e-9:
        return [], center

    spacing = max(sensor.track_spacing_m, 1.0)
    track_count = max(1, int(round(sqrt(centerline_budget_m / spacing))))

    radial_x = center.x - mission.center.x
    radial_y = center.y - mission.center.y
    radial_norm = sqrt(radial_x * radial_x + radial_y * radial_y)
    if radial_norm <= 1e-9:
        along_x, along_y = cos(phase_rad), sin(phase_rad)
    else:
        # Tangential heading keeps a sweep block inside its own annulus.
        along_x, along_y = -radial_y / radial_norm, radial_x / radial_norm
    step_x, step_y = -along_y, along_x

    search_speed = sensor.centerline_search_speed_mps(mission.search_speed_mps)
    transit_speed = mission.transit_speed_mps
    time_budget_s = centerline_budget_m / search_speed

    def build(leg_length: float) -> tuple[list[PathSegment], Point2D, float]:
        segments: list[PathSegment] = []
        current = center
        elapsed_s = 0.0
        offset_start = -0.5 * (track_count - 1) * spacing
        for index in range(track_count):
            offset = offset_start + index * spacing
            anchor = Point2D(
                center.x + step_x * offset,
                center.y + step_y * offset,
            )
            direction = 1.0 if index % 2 == 0 else -1.0
            leg_start = _clamp_to_area(
                Point2D(
                    anchor.x - direction * along_x * leg_length / 2.0,
                    anchor.y - direction * along_y * leg_length / 2.0,
                ),
                mission,
            )
            leg_end = _clamp_to_area(
                Point2D(
                    anchor.x + direction * along_x * leg_length / 2.0,
                    anchor.y + direction * along_y * leg_length / 2.0,
                ),
                mission,
            )
            hop = current.distance_to(leg_start)
            if hop > 1e-9:
                segments.append(PathSegment(current, leg_start, False))
                elapsed_s += hop / transit_speed
            span = leg_start.distance_to(leg_end)
            if span > 1e-9:
                segments.append(PathSegment(leg_start, leg_end, True))
                elapsed_s += span / search_speed
            current = leg_end
        return segments, current, elapsed_s

    leg_length = centerline_budget_m / track_count
    segments, current, elapsed_s = build(leg_length)
    # 연결 이동은 leg 길이에 거의 무관한 고정비라서 비례축소 한 번으로는 안
    # 맞는다. 몇 번 줄여가며 시간 예산 안으로 들인다.
    for _ in range(6):
        if elapsed_s <= time_budget_s or leg_length <= 1.0:
            break
        leg_length *= 0.9 * (time_budget_s / elapsed_s)
        segments, current, elapsed_s = build(leg_length)
    return segments, current


def _clamp_to_area(point: Point2D, mission: MissionConfig) -> Point2D:
    dx = point.x - mission.center.x
    dy = point.y - mission.center.y
    radius = sqrt(dx * dx + dy * dy)
    limit = mission.search_radius_m
    if radius <= limit or radius <= 1e-9:
        return point
    scale = limit / radius
    return Point2D(mission.center.x + dx * scale, mission.center.y + dy * scale)
