from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator


EARTH_METERS_PER_DEGREE = 111_320.0


def _offset_coordinate(
    latitude: float,
    longitude: float,
    distance_m: float,
    heading_deg: float,
) -> tuple[float, float]:
    heading = math.radians(heading_deg)
    north_m = math.cos(heading) * distance_m
    east_m = math.sin(heading) * distance_m
    latitude_out = latitude + north_m / EARTH_METERS_PER_DEGREE
    longitude_scale = max(
        10_000.0,
        EARTH_METERS_PER_DEGREE * math.cos(math.radians(latitude)),
    )
    longitude_out = longitude + east_m / longitude_scale
    return latitude_out, longitude_out


def _build_arc_seed_waypoints(
    center_latitude: float,
    center_longitude: float,
    vehicle_id: int,
    arc_actions: list[dict],
    *,
    altitude_m: float = 600.0,
    sector_width_deg: float = 60.0,
    sample_step_deg: float = 10.0,
) -> list[dict]:
    """Expand a compact CPP ARC action prefix into visible map waypoints.

    Every action is one complete sector arc. Consecutive actions are joined by
    a sensor-off transit line, matching ``PathSegment`` construction in the CPP
    ARC/PW-ARC planners.
    """
    if sector_width_deg <= 0.0 or sample_step_deg <= 0.0:
        return []
    sector_center = (vehicle_id - 1) * sector_width_deg
    minimum_heading = sector_center - sector_width_deg / 2.0
    maximum_heading = sector_center + sector_width_deg / 2.0
    subdivisions = max(1, round(sector_width_deg / sample_step_deg))
    points = [
        {
            "latitude": center_latitude,
            "longitude": center_longitude,
            "altitude_m": altitude_m,
        }
    ]
    for action in arc_actions:
        radius_m = max(1.0, float(action.get("radius_m", 0.0)))
        headings = [
            minimum_heading + sector_width_deg * index / subdivisions
            for index in range(subdivisions + 1)
        ]
        if str(action.get("start_boundary", "MIN")).upper() == "MAX":
            headings.reverse()
        for heading_deg in headings:
            latitude, longitude = _offset_coordinate(
                center_latitude,
                center_longitude,
                radius_m,
                heading_deg,
            )
            points.append(
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_m": altitude_m,
                }
            )
    return points


@dataclass
class GeoPoint:
    latitude: float
    longitude: float
    altitude_m: float = 0.0


@dataclass
class SitePosition(GeoPoint):
    code: str = ""
    label: str = ""


@dataclass
class MissionPoint(GeoPoint):
    code: str = ""
    label: str = ""
    point_type: str = "WAYPOINT"
    sequence: int = 0


@dataclass
class MissionSubject:
    track_id: int
    subject_type: str
    latitude: float
    longitude: float
    altitude_m: float = 0.0
    country: str = "--"
    platform_name: str = "GROUND SUBJECT"
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    position_uncertainty_m: float = 500.0
    source: str = "SAR-C2"
    track_age_s: float = 0.0
    motion_profile: list[dict] = field(default_factory=list)


@dataclass
class MissionZone:
    code: str
    label: str
    zone_type: str
    vertices: list[GeoPoint]

    def center(self) -> GeoPoint:
        if not self.vertices:
            return GeoPoint(0.0, 0.0, 0.0)
        count = len(self.vertices)
        return GeoPoint(
            sum(point.latitude for point in self.vertices) / count,
            sum(point.longitude for point in self.vertices) / count,
            0.0,
        )


class SiteStore:
    """Editable or loaded local mission-plan data."""

    VEHICLE_IDS = tuple(range(1, 7))
    ALLOWED_CODES = {
        "GCS": "Ground Control Station",
        "RDR": "Radar",
        "LC": "Launcher",
    }
    ZONE_LABELS = {
        "SAFE": "Safe Zone",
    }

    def __init__(self) -> None:
        self.mission_metadata: dict = {}
        self.sites: dict[str, SitePosition] = {}
        self.active_vehicle_id = 1
        self.vehicle_waypoints: dict[int, list[MissionPoint]] = {
            vehicle_id: [] for vehicle_id in self.VEHICLE_IDS
        }
        self.return_point: MissionPoint | None = None
        self.zones: list[MissionZone] = []
        self.initial_subjects: list[MissionSubject] = []
        self.draft_zone_type: str | None = None
        self.draft_vertices: list[GeoPoint] = []
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    @property
    def is_mission_ready(self) -> bool:
        return (
            self.shared_configuration_ready
            and any(self.vehicle_waypoints[vehicle_id] for vehicle_id in self.VEHICLE_IDS)
        )

    @property
    def shared_configuration_ready(self) -> bool:
        return (
            {"GCS", "RDR", "LC"}.issubset(self.sites)
            and any(zone.zone_type == "SAFE" for zone in self.zones)
        )

    @property
    def active_vehicle_mission_ready(self) -> bool:
        return self.shared_configuration_ready and bool(self.waypoints)

    @property
    def waypoints(self) -> list[MissionPoint]:
        return self.vehicle_waypoints[self.active_vehicle_id]

    @waypoints.setter
    def waypoints(self, points: list[MissionPoint]) -> None:
        self.vehicle_waypoints[self.active_vehicle_id] = points

    @property
    def active_vehicle_code(self) -> str:
        return f"UAV-{self.active_vehicle_id:02d}"

    @property
    def configured_vehicle_ids(self) -> tuple[int, ...]:
        return tuple(
            vehicle_id
            for vehicle_id in self.VEHICLE_IDS
            if self.vehicle_waypoints[vehicle_id]
        )

    @property
    def total_waypoint_count(self) -> int:
        return sum(len(points) for points in self.vehicle_waypoints.values())

    def waypoints_for(self, vehicle_id: int) -> list[MissionPoint]:
        vehicle_id = int(vehicle_id)
        if vehicle_id not in self.VEHICLE_IDS:
            raise ValueError("Vehicle ID must be between 1 and 6.")
        return self.vehicle_waypoints[vehicle_id]

    def set_active_vehicle(self, vehicle_id: int) -> None:
        vehicle_id = int(vehicle_id)
        if vehicle_id not in self.VEHICLE_IDS:
            raise ValueError("Vehicle ID must be between 1 and 6.")
        if vehicle_id == self.active_vehicle_id:
            return
        self.active_vehicle_id = vehicle_id
        self.notify()

    def notify(self) -> None:
        for callback in tuple(self._listeners):
            callback()

    @staticmethod
    def _validated_point(
        latitude: float,
        longitude: float,
        altitude_m: float = 0.0,
    ) -> GeoPoint:
        latitude = float(latitude)
        longitude = float(longitude)
        altitude_m = float(altitude_m)
        if not -90 <= latitude <= 90:
            raise ValueError("Latitude must be between -90 and 90.")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180.")
        return GeoPoint(latitude, longitude, max(0.0, altitude_m))

    def set_site(
        self,
        code: str,
        latitude: float,
        longitude: float,
        altitude_m: float = 0.0,
    ) -> SitePosition:
        if code not in self.ALLOWED_CODES:
            raise ValueError(f"Unsupported site code: {code}")
        point = self._validated_point(latitude, longitude, altitude_m)
        site = SitePosition(
            latitude=point.latitude,
            longitude=point.longitude,
            altitude_m=point.altitude_m,
            code=code,
            label=self.ALLOWED_CODES[code],
        )
        self.sites[code] = site
        self.notify()
        return site

    def add_waypoint(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float = 60.0,
    ) -> MissionPoint:
        point = self._validated_point(latitude, longitude, altitude_m)
        sequence = len(self.waypoints) + 1
        waypoint = MissionPoint(
            latitude=point.latitude,
            longitude=point.longitude,
            altitude_m=point.altitude_m,
            code=f"WP{sequence:03d}",
            label=f"{self.active_vehicle_code} Waypoint {sequence}",
            point_type="WAYPOINT",
            sequence=sequence,
        )
        self.waypoints.append(waypoint)
        self.notify()
        return waypoint

    def set_return_point(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float = 30.0,
    ) -> MissionPoint:
        point = self._validated_point(latitude, longitude, altitude_m)
        self.return_point = MissionPoint(
            latitude=point.latitude,
            longitude=point.longitude,
            altitude_m=point.altitude_m,
            code="RTB",
            label="Return Point",
            point_type="RETURN",
            sequence=0,
        )
        self.notify()
        return self.return_point

    def begin_zone(self, zone_type: str) -> None:
        if zone_type not in self.ZONE_LABELS:
            raise ValueError(f"Unsupported zone type: {zone_type}")
        if self.draft_zone_type != zone_type:
            self.draft_vertices.clear()
        self.draft_zone_type = zone_type
        self.notify()

    def add_draft_vertex(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float = 0.0,
    ) -> GeoPoint:
        if self.draft_zone_type not in self.ZONE_LABELS:
            raise ValueError("Select a safe or danger zone tool first.")
        point = self._validated_point(latitude, longitude, altitude_m)
        self.draft_vertices.append(point)
        self.notify()
        return point

    def undo_draft_vertex(self) -> bool:
        if not self.draft_vertices:
            return False
        self.draft_vertices.pop()
        self.notify()
        return True

    def cancel_draft_zone(self) -> None:
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()

    def commit_zone(self) -> MissionZone:
        zone_type = self.draft_zone_type
        if zone_type not in self.ZONE_LABELS:
            raise ValueError("No zone is being drawn.")
        if len(self.draft_vertices) < 3:
            raise ValueError("A zone requires at least three vertices.")
        prefix = "SAFE"
        existing = {zone.code for zone in self.zones}
        sequence = 1
        while f"{prefix}{sequence:02d}" in existing:
            sequence += 1
        zone = MissionZone(
            code=f"{prefix}{sequence:02d}",
            label=f"{self.ZONE_LABELS[zone_type]} {sequence}",
            zone_type=zone_type,
            vertices=list(self.draft_vertices),
        )
        self.zones.append(zone)
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()
        return zone

    def iter_points(self) -> Iterator[SitePosition | MissionPoint]:
        yield from self.sites.values()
        yield from self.waypoints

    def iter_features(
        self,
    ) -> Iterator[SitePosition | MissionPoint | MissionZone]:
        yield from self.iter_points()
        yield from self.zones

    def iter_all_features(
        self,
    ) -> Iterator[SitePosition | MissionPoint | MissionZone]:
        """Yield shared features and every vehicle route.

        PLAN deliberately uses ``iter_features`` so only the route currently
        being edited is drawn. Destructive/file operations use this method so
        routes belonging to the other five vehicles are never overlooked.
        """
        yield from self.sites.values()
        for vehicle_id in self.VEHICLE_IDS:
            yield from self.vehicle_waypoints[vehicle_id]
        yield from self.zones

    @property
    def has_any_data(self) -> bool:
        return any(True for _ in self.iter_all_features())

    def get_feature(
        self,
        code: str,
    ) -> SitePosition | MissionPoint | MissionZone | None:
        if code in self.sites:
            return self.sites[code]
        if code == "RTB":
            return self.return_point
        for waypoint in self.waypoints:
            if waypoint.code == code:
                return waypoint
        return next((zone for zone in self.zones if zone.code == code), None)

    def update_point(
        self,
        code: str,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> SitePosition | MissionPoint:
        if code in self.ALLOWED_CODES:
            return self.set_site(code, latitude, longitude, altitude_m)
        point = self._validated_point(latitude, longitude, altitude_m)
        feature = self.get_feature(code)
        if not isinstance(feature, MissionPoint):
            raise ValueError("The selected feature is not a point.")
        feature.latitude = point.latitude
        feature.longitude = point.longitude
        feature.altitude_m = point.altitude_m
        self.notify()
        return feature

    def remove(self, code: str) -> bool:
        return self.remove_feature(code)

    def remove_feature(self, code: str) -> bool:
        if code in self.sites:
            del self.sites[code]
            self.notify()
            return True
        if code == "RTB" and self.return_point is not None:
            self.return_point = None
            self.notify()
            return True
        before = len(self.waypoints)
        self.waypoints = [
            waypoint for waypoint in self.waypoints if waypoint.code != code
        ]
        if len(self.waypoints) != before:
            self._renumber_waypoints()
            self.notify()
            return True
        before = len(self.zones)
        self.zones = [zone for zone in self.zones if zone.code != code]
        if len(self.zones) != before:
            self.notify()
            return True
        return False

    def _renumber_waypoints(self) -> None:
        for sequence, waypoint in enumerate(self.waypoints, start=1):
            waypoint.sequence = sequence
            waypoint.code = f"WP{sequence:03d}"
            waypoint.label = f"{self.active_vehicle_code} Waypoint {sequence}"

    def clear(self) -> None:
        self.mission_metadata.clear()
        self.sites.clear()
        for vehicle_id in self.VEHICLE_IDS:
            self.vehicle_waypoints[vehicle_id].clear()
        self.return_point = None
        self.zones.clear()
        self.initial_subjects.clear()
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()

    def seed_demo(self, latitude: float, longitude: float) -> None:
        self.mission_metadata = {
            "scenario_id": "LOCAL-DEMO",
            "name": "Local demonstration mission",
            "source": "LOCAL",
        }
        self.sites = {
            "GCS": SitePosition(latitude, longitude, 42, "GCS", self.ALLOWED_CODES["GCS"]),
            "RDR": SitePosition(
                latitude + 0.0030,
                longitude - 0.0045,
                68,
                "RDR",
                self.ALLOWED_CODES["RDR"],
            ),
            "LC": SitePosition(
                latitude - 0.0023,
                longitude - 0.0035,
                51,
                "LC",
                self.ALLOWED_CODES["LC"],
            ),
        }
        self.vehicle_waypoints = {
            vehicle_id: [] for vehicle_id in self.VEHICLE_IDS
        }
        launcher = self.sites["LC"]
        staging_latitude, staging_longitude = _offset_coordinate(
            launcher.latitude, launcher.longitude, 30_000.0, 84.0
        )
        for vehicle_id in self.VEHICLE_IDS:
            route = self.vehicle_waypoints[vehicle_id]
            route.append(
                MissionPoint(
                    staging_latitude,
                    staging_longitude,
                    600.0,
                    "WP001",
                    f"UAV-{vehicle_id:02d} Waypoint 1",
                    "WAYPOINT",
                    1,
                )
            )
            sector_heading = (vehicle_id - 1) * 60.0
            for sequence, distance_m in enumerate(
                (4_000.0, 8_000.0, 12_000.0, 15_500.0),
                start=2,
            ):
                zigzag_heading = (
                    sector_heading
                    if sequence == 5
                    else sector_heading + (
                        5.0 if sequence % 2 == 0 else -5.0
                    )
                )
                point_latitude, point_longitude = _offset_coordinate(
                    staging_latitude,
                    staging_longitude,
                    distance_m,
                    zigzag_heading,
                )
                route.append(
                    MissionPoint(
                        point_latitude,
                        point_longitude,
                        600.0,
                        f"WP{sequence:03d}",
                        f"UAV-{vehicle_id:02d} Waypoint {sequence}",
                        "WAYPOINT",
                        sequence,
                    )
                )
        self.return_point = None
        safe_center_latitude, safe_center_longitude = _offset_coordinate(
            latitude, longitude, 7_000.0, 230.0
        )
        self.zones = [
            MissionZone(
                "SAFE01",
                "Safe Zone 1",
                "SAFE",
                [
                    GeoPoint(
                        *_offset_coordinate(
                            safe_center_latitude,
                            safe_center_longitude,
                            1_600.0,
                            0.0,
                        )
                    ),
                    GeoPoint(
                        *_offset_coordinate(
                            safe_center_latitude,
                            safe_center_longitude,
                            1_600.0,
                            120.0,
                        )
                    ),
                    GeoPoint(
                        *_offset_coordinate(
                            safe_center_latitude,
                            safe_center_longitude,
                            1_600.0,
                            240.0,
                        )
                    ),
                ],
            ),
        ]
        self.initial_subjects = []
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()

    def replace_from(self, source: SiteStore) -> None:
        """Atomically replace this store with an independent source snapshot."""
        self.mission_metadata = copy.deepcopy(source.mission_metadata)
        self.sites = copy.deepcopy(source.sites)
        self.vehicle_waypoints = copy.deepcopy(source.vehicle_waypoints)
        self.active_vehicle_id = source.active_vehicle_id
        self.return_point = None
        self.zones = copy.deepcopy(source.zones)
        self.initial_subjects = copy.deepcopy(source.initial_subjects)
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()

    def render_dict(self) -> dict:
        active_waypoints = [
            {
                **asdict(point),
                "vehicle_id": self.active_vehicle_id,
                "vehicle_code": self.active_vehicle_code,
            }
            for point in self.waypoints
        ]
        return {
            "mission": copy.deepcopy(self.mission_metadata),
            "active_vehicle_id": self.active_vehicle_id,
            "active_vehicle_code": self.active_vehicle_code,
            "sites": [asdict(site) for site in self.sites.values()],
            "waypoints": active_waypoints,
            "vehicle_routes": [
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_code": f"UAV-{vehicle_id:02d}",
                    "waypoints": [
                        asdict(point)
                        for point in self.vehicle_waypoints[vehicle_id]
                    ],
                }
                for vehicle_id in self.VEHICLE_IDS
            ],
            "return_point": (
                None
            ),
            "zones": [
                {
                    "code": zone.code,
                    "label": zone.label,
                    "zone_type": zone.zone_type,
                    "vertices": [asdict(vertex) for vertex in zone.vertices],
                }
                for zone in self.zones
            ],
            "initial_subjects": [
                asdict(subj) for subj in self.initial_subjects
            ],
            "draft_zone": {
                "zone_type": self.draft_zone_type,
                "vertices": [asdict(vertex) for vertex in self.draft_vertices],
            },
        }

    def to_dict(self) -> dict:
        payload = self.render_dict()
        payload.pop("draft_zone", None)
        return {
            "format": "python-gcs-mission-plan",
            "version": 4,
            **payload,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        file_format = data.get("format")
        if file_format not in {
            "python-gcs-site-config",
            "python-gcs-mission-plan",
        }:
            raise ValueError("Unsupported mission configuration format.")

        raw_mission = data.get("mission", {})
        mission_metadata = (
            copy.deepcopy(raw_mission)
            if isinstance(raw_mission, dict)
            else {}
        )

        sites: dict[str, SitePosition] = {}
        for raw in data.get("sites", []):
            code = str(raw["code"])
            if code not in self.ALLOWED_CODES:
                continue
            point = self._validated_point(
                raw["latitude"],
                raw["longitude"],
                raw.get("altitude_m", 0.0),
            )
            sites[code] = SitePosition(
                point.latitude,
                point.longitude,
                point.altitude_m,
                code,
                self.ALLOWED_CODES[code],
            )

        def parse_waypoints(
            raw_waypoints: list[dict],
            vehicle_id: int,
        ) -> list[MissionPoint]:
            parsed: list[MissionPoint] = []
            for sequence, raw in enumerate(raw_waypoints, start=1):
                point = self._validated_point(
                    raw["latitude"],
                    raw["longitude"],
                    raw.get("altitude_m", 60.0),
                )
                parsed.append(
                    MissionPoint(
                        point.latitude,
                        point.longitude,
                        point.altitude_m,
                        f"WP{sequence:03d}",
                        f"UAV-{vehicle_id:02d} Waypoint {sequence}",
                        "WAYPOINT",
                        sequence,
                    )
                )
            return parsed

        vehicle_waypoints: dict[int, list[MissionPoint]] = {
            vehicle_id: [] for vehicle_id in self.VEHICLE_IDS
        }
        raw_routes = data.get("vehicle_routes")
        if isinstance(raw_routes, list):
            for raw_route in raw_routes:
                vehicle_id = int(raw_route.get("vehicle_id", 0))
                if vehicle_id not in self.VEHICLE_IDS:
                    continue
                vehicle_waypoints[vehicle_id] = parse_waypoints(
                    list(raw_route.get("waypoints", [])),
                    vehicle_id,
                )
        else:
            # Version 1/2 mission plans had one route. Preserve it as SAR-01.
            vehicle_waypoints[1] = parse_waypoints(
                list(data.get("waypoints", [])),
                1,
            )

        arc_pattern = mission_metadata.get("arc_search_pattern")
        if isinstance(arc_pattern, dict):
            center = arc_pattern.get("center", {})
            profiles = arc_pattern.get("vehicle_arc_sequences", {})
            if isinstance(center, dict) and isinstance(profiles, dict):
                center_point = self._validated_point(
                    center.get("latitude"),
                    center.get("longitude"),
                    arc_pattern.get("altitude_m", 600.0),
                )
                for vehicle_id in self.VEHICLE_IDS:
                    actions = profiles.get(str(vehicle_id), [])
                    if not isinstance(actions, list) or not actions:
                        continue
                    expanded = _build_arc_seed_waypoints(
                        center_point.latitude,
                        center_point.longitude,
                        vehicle_id,
                        actions,
                        altitude_m=center_point.altitude_m,
                        sector_width_deg=float(
                            arc_pattern.get("sector_width_deg", 60.0)
                        ),
                        sample_step_deg=float(
                            arc_pattern.get("sample_step_deg", 10.0)
                        ),
                    )
                    vehicle_waypoints[vehicle_id] = parse_waypoints(
                        expanded,
                        vehicle_id,
                    )

        return_point = None
        # Version 1/2 files may contain RTB; the current mission model no
        # longer uses or displays a return point.

        zones: list[MissionZone] = []
        for raw in data.get("zones", []):
            zone_type = str(raw.get("zone_type", ""))
            vertices = [
                self._validated_point(
                    vertex["latitude"],
                    vertex["longitude"],
                    vertex.get("altitude_m", 0.0),
                )
                for vertex in raw.get("vertices", [])
            ]
            if zone_type not in self.ZONE_LABELS or len(vertices) < 3:
                continue
            zones.append(
                MissionZone(
                    code=str(raw.get("code") or ""),
                    label=str(raw.get("label") or self.ZONE_LABELS[zone_type]),
                    zone_type=zone_type,
                    vertices=vertices,
                )
            )

        initial_subjects: list[MissionSubject] = []
        for raw in data.get("initial_subjects", []):
            if not isinstance(raw, dict):
                continue
            point = self._validated_point(
                raw["latitude"],
                raw["longitude"],
                raw.get("altitude_m", 0.0),
            )
            speed_mps = float(
                raw.get(
                    "speed_mps",
                    float(raw.get("speed_kph", 0.0)) / 3.6,
                )
            )
            motion_profile = []
            for segment in raw.get("motion_profile", []):
                if not isinstance(segment, dict):
                    continue
                parsed_segment = copy.deepcopy(segment)
                if "speed_kph" in parsed_segment:
                    parsed_segment["speed_kph"] = max(
                        0.0,
                        min(40.0, float(parsed_segment["speed_kph"])),
                    )
                motion_profile.append(parsed_segment)
            initial_subjects.append(
                MissionSubject(
                    track_id=int(raw["track_id"]),
                    subject_type=str(raw.get("subject_type", raw.get("target_type", "GROUND"))),
                    latitude=point.latitude,
                    longitude=point.longitude,
                    altitude_m=point.altitude_m,
                    country=str(raw.get("country", "--")),
                    platform_name=str(
                        raw.get("platform_name", "GROUND SUBJECT")
                    ),
                    speed_mps=max(0.0, min(40.0 / 3.6, speed_mps)),
                    heading_deg=float(raw.get("heading_deg", 0.0)) % 360.0,
                    position_uncertainty_m=max(
                        1.0,
                        float(raw.get("position_uncertainty_m", 500.0)),
                    ),
                    source=str(raw.get("source", "SAR-C2")),
                    track_age_s=max(0.0, float(raw.get("track_age_s", 0.0))),
                    motion_profile=motion_profile,
                )
            )

        self.mission_metadata = mission_metadata
        self.sites = sites
        self.vehicle_waypoints = vehicle_waypoints
        if self.active_vehicle_id not in self.VEHICLE_IDS:
            self.active_vehicle_id = 1
        self.return_point = return_point
        self.zones = zones
        self.initial_subjects = initial_subjects
        self.draft_zone_type = None
        self.draft_vertices.clear()
        self.notify()
