from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import asin, cos, radians, sin, sqrt
from typing import Callable


class MissionCommand(str, Enum):
    TAKEOFF = "Takeoff"
    WAYPOINT = "Waypoint"
    LOITER = "Loiter"
    RTL = "Return"
    LAND = "Land"


@dataclass
class Waypoint:
    sequence: int
    command: MissionCommand
    latitude: float
    longitude: float
    altitude_m: float = 50.0
    hold_s: float = 0.0


@dataclass
class VehicleState:
    connected: bool = False
    ready_text: str = "Not Connected"
    flight_mode: str = "Manual"
    armed: bool = False
    gps_satellites: int = 0
    gps_fix: str = "No GPS"
    battery_percent: int = 0
    link_percent: int = 0
    latitude: float = 37.3422
    longitude: float = 127.9202
    relative_altitude_m: float = 0.0
    ground_speed_mps: float = 0.0
    vertical_speed_mps: float = 0.0
    heading_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    flight_time_s: int = 0


@dataclass
class MissionStore:
    home_latitude: float = 37.3422
    home_longitude: float = 127.9202
    waypoints: list[Waypoint] = field(default_factory=list)
    selected_sequence: int | None = None
    _listeners: list[Callable[[], None]] = field(default_factory=list, repr=False)

    def subscribe(self, callback: Callable[[], None]) -> None:
        self._listeners.append(callback)

    def notify(self) -> None:
        for callback in tuple(self._listeners):
            callback()

    def seed_demo(self) -> None:
        self.waypoints = [
            Waypoint(1, MissionCommand.TAKEOFF, 37.3422, 127.9202, 30),
            Waypoint(2, MissionCommand.WAYPOINT, 37.3460, 127.9240, 60),
            Waypoint(3, MissionCommand.WAYPOINT, 37.3440, 127.9300, 70),
            Waypoint(4, MissionCommand.LOITER, 37.3390, 127.9280, 70, 20),
            Waypoint(5, MissionCommand.RTL, 37.3422, 127.9202, 30),
        ]
        self.selected_sequence = 2
        self.notify()

    def add_waypoint(
        self,
        latitude: float,
        longitude: float,
        command: MissionCommand = MissionCommand.WAYPOINT,
    ) -> Waypoint:
        default_altitude = 30.0 if command in {MissionCommand.TAKEOFF, MissionCommand.LAND} else 60.0
        waypoint = Waypoint(
            sequence=len(self.waypoints) + 1,
            command=command,
            latitude=latitude,
            longitude=longitude,
            altitude_m=default_altitude,
        )
        self.waypoints.append(waypoint)
        self.selected_sequence = waypoint.sequence
        self.notify()
        return waypoint

    def get_selected(self) -> Waypoint | None:
        return next(
            (waypoint for waypoint in self.waypoints if waypoint.sequence == self.selected_sequence),
            None,
        )

    def select(self, sequence: int | None) -> None:
        if self.selected_sequence == sequence:
            return
        self.selected_sequence = sequence
        self.notify()

    def move(self, sequence: int, latitude: float, longitude: float) -> None:
        waypoint = next(
            (item for item in self.waypoints if item.sequence == sequence),
            None,
        )
        if waypoint is None:
            return
        waypoint.latitude = latitude
        waypoint.longitude = longitude
        self.notify()

    def update_selected(
        self,
        *,
        latitude: float,
        longitude: float,
        altitude_m: float,
        hold_s: float,
    ) -> bool:
        waypoint = self.get_selected()
        if waypoint is None:
            return False
        waypoint.latitude = latitude
        waypoint.longitude = longitude
        waypoint.altitude_m = altitude_m
        waypoint.hold_s = hold_s
        self.notify()
        return True

    def delete_selected(self) -> bool:
        selected = self.selected_sequence
        if selected is None:
            return False
        before = len(self.waypoints)
        self.waypoints = [
            waypoint for waypoint in self.waypoints if waypoint.sequence != selected
        ]
        if len(self.waypoints) == before:
            return False
        for sequence, waypoint in enumerate(self.waypoints, start=1):
            waypoint.sequence = sequence
        self.selected_sequence = min(selected, len(self.waypoints)) if self.waypoints else None
        self.notify()
        return True

    def clear(self) -> None:
        self.waypoints.clear()
        self.selected_sequence = None
        self.notify()

    def total_distance_m(self) -> float:
        points = [(self.home_latitude, self.home_longitude)]
        points.extend((waypoint.latitude, waypoint.longitude) for waypoint in self.waypoints)
        return sum(
            haversine_m(first[0], first[1], second[0], second[1])
            for first, second in zip(points, points[1:])
        )

    def estimated_time_s(self, speed_mps: float = 8.0) -> float:
        travel = self.total_distance_m() / max(speed_mps, 0.1)
        holds = sum(waypoint.hold_s for waypoint in self.waypoints)
        return travel + holds


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    value = (
        sin(delta_phi / 2) ** 2
        + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_m * asin(sqrt(value))
