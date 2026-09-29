from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from .site_store import SiteStore
from .track_predictor import ConstantVelocityTrackPredictor


EARTH_METERS_PER_DEGREE = 111_320.0
SAR_CRUISE_SPEED_MPS = 160.0 / 3.6
SAR_INGRESS_SPEED_MPS = 100_000.0 / (8.0 * 60.0)
SAR_SEARCH_SPEED_MPS = 160.0 / 3.6
PREDICTION_HORIZON_S = 8.0 * 60.0
SIMULATION_TIME_SCALE = 3.0
# The Fly view timer fires every 50 ms.  When the WebGL map hitches (e.g. right
# after a re-solve injects a dense route) a single frame's wall-clock gap can
# balloon, and because each step is speed*dt*TIME_SCALE that turns one frame
# into a large forward jump — the vehicle appears to "suddenly speed up".  Cap
# the auto-derived dt to a small multiple of the frame time so a hitch makes the
# sim briefly fall behind wall-clock (invisible) instead of lurching (visible).
MAX_TICK_DT_S = 0.15
DEMO_FLIGHT_TIME_SCALE = SIMULATION_TIME_SCALE
DEMO_GROUND_TIME_SCALE = SIMULATION_TIME_SCALE
INITIAL_APPROACH_SPEED_MPS = 160.0 / 3.6
MIDCOURSE_APPROACH_SPEED_MPS = 160.0 / 3.6
FINAL_APPROACH_SPEED_MPS = 200.0 / 3.6
FINAL_SPEED_RAMP_S = 5.0
FINAL_CONTACT_DWELL_S = 0.75
APPROACH_VISUAL_DURATION_S = PREDICTION_HORIZON_S / DEMO_FLIGHT_TIME_SCALE
INITIAL_APPROACH_DURATION_S = 8.0
# Keep MIDCOURSE visible for three wall-clock seconds in the 3x demo.  This
# makes the yellow AUTO track state operationally distinguishable from the red
# final state even when the SAR reaches the horizontal approach point early.
MINIMUM_MIDCOURSE_APPROACH_DURATION_S = 9.0
# Keep a visible midcourse tracking interval inside the nominal 1.2 km
# search camera ground strip before final approach begins.
FINAL_ENTRY_RADIUS_M = 350.0
MIDCOURSE_APPROACH_TIME_SCALE = SIMULATION_TIME_SCALE

DEMO_SUBJECT_SPECS = (
    # track, type, country, platform, range, bearing, speed, course, age
    (101, "MISSING_PERSON_CHILD", "KR", "TARGET", 36_000.0, 58.0, 4 / 3.6, 224.0, 94),
    (204, "MISSING_PERSON_ADULT", "KR", "TARGET", 32_000.0, 112.0, 5 / 3.6, 291.0, 61),
)


def horizontal_distance_m(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    mean_latitude = math.radians((latitude_1 + latitude_2) / 2.0)
    north_m = (latitude_2 - latitude_1) * EARTH_METERS_PER_DEGREE
    east_m = (
        (longitude_2 - longitude_1)
        * EARTH_METERS_PER_DEGREE
        * math.cos(mean_latitude)
    )
    return math.hypot(east_m, north_m)


def bearing_deg(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    mean_latitude = math.radians((latitude_1 + latitude_2) / 2.0)
    north_m = (latitude_2 - latitude_1) * EARTH_METERS_PER_DEGREE
    east_m = (
        (longitude_2 - longitude_1)
        * EARTH_METERS_PER_DEGREE
        * math.cos(mean_latitude)
    )
    return (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0


def offset_position(
    latitude: float,
    longitude: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    latitude_out = latitude + north_m / EARTH_METERS_PER_DEGREE
    longitude_scale = max(
        10_000.0,
        EARTH_METERS_PER_DEGREE * math.cos(math.radians(latitude)),
    )
    longitude_out = longitude + east_m / longitude_scale
    return latitude_out, longitude_out


def destination_position(
    latitude: float,
    longitude: float,
    distance_m: float,
    heading_deg: float,
) -> tuple[float, float]:
    heading = math.radians(heading_deg)
    return offset_position(
        latitude,
        longitude,
        math.sin(heading) * distance_m,
        math.cos(heading) * distance_m,
    )


def constant_velocity_approach_time(
    relative_east_m: float,
    relative_north_m: float,
    relative_up_m: float,
    subject_velocity_east_mps: float,
    subject_velocity_north_mps: float,
    subject_velocity_up_mps: float,
    pursuer_speed_mps: float,
    max_horizon_s: float = PREDICTION_HORIZON_S,
) -> tuple[float, bool]:
    """Return reachable approach time for constant subject velocity.

    The quadratic solves ``|relative_position + subject_velocity * t| =
    pursuer_speed * t``. The boolean is false when the mathematical solution
    is unavailable or beyond the configured prediction horizon.
    """
    pursuer_speed_mps = max(1.0, float(pursuer_speed_mps))
    relative = (
        float(relative_east_m),
        float(relative_north_m),
        float(relative_up_m),
    )
    velocity = (
        float(subject_velocity_east_mps),
        float(subject_velocity_north_mps),
        float(subject_velocity_up_mps),
    )
    a = sum(component * component for component in velocity) - (
        pursuer_speed_mps * pursuer_speed_mps
    )
    b = 2.0 * sum(
        position * speed
        for position, speed in zip(relative, velocity)
    )
    c = sum(component * component for component in relative)
    positive_roots: list[float] = []
    if abs(a) < 1e-9:
        if abs(b) > 1e-9:
            root = -c / b
            if root > 0.0:
                positive_roots.append(root)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root_scale = math.sqrt(discriminant)
            for root in (
                (-b - root_scale) / (2.0 * a),
                (-b + root_scale) / (2.0 * a),
            ):
                if root > 0.0:
                    positive_roots.append(root)
    if not positive_roots:
        return float(max_horizon_s), False
    approach_time_s = min(positive_roots)
    reachable = approach_time_s <= max_horizon_s
    return min(approach_time_s, float(max_horizon_s)), reachable


@dataclass
class VehicleTrack:
    code: str = "UAV-01"
    latitude: float = 37.3422
    longitude: float = 127.9202
    altitude_m: float = 51.0
    speed_mps: float = SAR_CRUISE_SPEED_MPS
    heading_deg: float = 0.0


@dataclass
class SubjectTrack:
    track_id: int
    subject_type: str
    country: str
    platform_name: str
    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float
    heading_deg: float
    first_tracked_at: float
    found: bool = False
    position_uncertainty_m: float = 500.0
    source: str = "LOCAL"
    motion_profile: list[dict] = field(default_factory=list)

    @property
    def code(self) -> str:
        return f"PSN-{self.track_id}"

    @property
    def first_tracked_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.first_tracked_at))


def hide_preplanned_search_routes(plan: dict, mission_metadata: dict | None) -> None:
    """For an RHP-SPX mission, strip the pre-baked SPX search lawnmower (WP*)
    from a fly-map plan payload in place, keeping only the ingress lanes (IGR*).

    The online RHP planner regenerates each UAV's search route live from the
    belief and overlays it as a ``runtime_route``; the baked routes are never
    flown.  Drawing them just clutters the search disk from launch with paths
    that are immediately replaced -- and, drawn from the launcher, they visibly
    snap when the live route takes over (the SAR-03 "yellow route" jump).
    """
    if str((mission_metadata or {}).get("planning_mode", "")) != "RHP-SPX":
        return
    for route in plan.get("vehicle_routes", []):
        route["waypoints"] = [
            waypoint
            for waypoint in route.get("waypoints", [])
            if str(waypoint.get("code", "")).startswith("IGR")
        ]


@dataclass
class FlyState:
    """Local-only mission demonstration state.

    No method in this class transmits MAVLink or an external launch command.
    The accelerated motion is a visualization timeline; displayed speed and
    ETA remain based on SAR_CRUISE_SPEED_MPS.
    """

    center_latitude: float
    center_longitude: float
    vehicle: VehicleTrack
    subjects: list[SubjectTrack]
    selected_track_id: int = 101
    readiness: dict[str, bool] = field(
        default_factory=lambda: {
            "AVS": True,
            "LC": False,
            "RDR": False,
            "DL": True,
            "GCS": False,
        }
    )
    mission_status: dict[str, bool] = field(
        default_factory=lambda: {
            "초기접근": False,
            "중기접근": False,
            "최종접근": False,
            "셔터 ON": False,
            "TRACK HOLD": False,
            "TDD 탐지": False,
            "확인 완료": False,
        }
    )
    mission_loaded: bool = False
    mission_launched: bool = False
    subject_designated: bool = False
    subject_detected: bool = False
    approach_approved: bool = False
    launch_requested: bool = False
    approach_requested: bool = False
    subject_found: bool = False
    detection_source_vehicle_id: int | None = None
    emergency_mode: bool = False
    flight_phase: str = "STANDBY"
    current_waypoint_index: int = 0
    completed_route_segment_count: int = 0
    guidance_elapsed_s: float = 0.0
    simulation_elapsed_s: float = 0.0
    search_elapsed_s: float = 0.0
    last_launch_action: str = "NONE"
    shutdown_position: tuple[float, float, float] | None = None
    flight_path: list[dict[str, float]] = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic, repr=False)
    _route_points: list[tuple[float, float, float, str]] = field(
        default_factory=list,
        repr=False,
    )
    _mission_signature: tuple = field(default_factory=tuple, repr=False)
    _track_predictors: dict[int, ConstantVelocityTrackPredictor] = field(
        default_factory=dict,
        repr=False,
    )
    _last_tick_s: float = field(default=0.0, repr=False)
    _phase_before_emergency: str = field(default="ROUTE", repr=False)
    _loiter_center: tuple[float, float, float, str] | None = field(
        default=None,
        repr=False,
    )
    _loiter_angle_rad: float = field(default=0.0, repr=False)
    _detection_waypoint: tuple[float, float, float, str] | None = field(
        default=None,
        repr=False,
    )
    _return_point: tuple[float, float, float, str] | None = field(
        default=None,
        repr=False,
    )
    _mitl_return_waypoint: tuple[float, float, float, str] | None = field(
        default=None,
        repr=False,
    )
    _phase_elapsed_s: float = field(default=0.0, repr=False)
    _rendezvous_solution: dict | None = field(default=None, repr=False)
    _rendezvous_elapsed_s: float = field(default=0.0, repr=False)
    _rendezvous_vehicle_start: tuple[float, float, float] | None = field(
        default=None,
        repr=False,
    )
    _rendezvous_subject_start: tuple[float, float, float] | None = field(
        default=None,
        repr=False,
    )
    _runtime_route_revision: int = field(default=0, repr=False)
    _runtime_route_update_count: int = field(default=0, repr=False)
    _search_started: bool = field(default=False, repr=False)
    _pending_runtime_route_revision: int = field(default=0, repr=False)
    _pending_runtime_route: list[tuple[float, float, float, str]] | None = field(
        default=None,
        repr=False,
    )
    _manual_route_hold_until_s: float = field(default=0.0, repr=False)
    _manual_route_active: bool = field(default=False, repr=False)

    @classmethod
    def demo(cls, latitude: float, longitude: float) -> "FlyState":
        now = time.time()
        subjects: list[SubjectTrack] = []
        for (
            track_id,
            subject_type,
            country,
            platform_name,
            distance_m,
            radial_heading,
            speed_mps,
            movement_heading,
            age_s,
        ) in DEMO_SUBJECT_SPECS:
            track_latitude, track_longitude = destination_position(
                latitude,
                longitude,
                distance_m,
                radial_heading,
            )
            subjects.append(
                SubjectTrack(
                    track_id=track_id,
                    subject_type=subject_type,
                    country=country,
                    platform_name=platform_name,
                    latitude=track_latitude,
                    longitude=track_longitude,
                    altitude_m=0.0,
                    speed_mps=speed_mps,
                    heading_deg=movement_heading,
                    first_tracked_at=now - age_s,
                )
            )

        state = cls(
            center_latitude=latitude,
            center_longitude=longitude,
            vehicle=VehicleTrack(latitude=latitude, longitude=longitude),
            subjects=subjects,
        )
        state._reset_predictors()
        return state

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self._started_at)

    @property
    def selected_subject(self) -> SubjectTrack | None:
        return next(
            (
                track
                for track in self.subjects
                if track.track_id == self.selected_track_id
            ),
            None,
        )

    @property
    def launch_ready(self) -> bool:
        return (
            self.mission_loaded
            and all(self.readiness.values())
            and not self.emergency_mode
        )

    @property
    def automatic_mode(self) -> str:
        return (
            "ARM"
            if self.approach_approved
            and not self.emergency_mode
            and not self.subject_found
            else "SAFE"
        )

    @property
    def camera_mode(self) -> str:
        if self.subject_found:
            return "SHUT DOWN"
        if self.emergency_mode or self.flight_phase in {
            "EMERGENCY_SAFE_RETURN",
            "MANUAL_WAYPOINT_RETURN",
            "MANUAL_SAFE_RETURN",
            "RETURNED",
        }:
            return "STOW"
        if not self.mission_launched:
            return "STANDBY"
        if not self.subject_detected:
            return "SEARCH"
        return {
            "DETECTION_TRANSIT": "AUTO HANDOFF",
            "INITIAL_APPROACH": "AUTO SCAN",
            "MIDCOURSE_APPROACH": "AUTO TRACK",
            "FINAL_APPROACH": "AUTO HOLD",
            "FOUND": "SHUT DOWN",
        }.get(self.flight_phase, "AUTO TRACK")

    @property
    def rendezvous_ready(self) -> bool:
        return (
            self.mission_launched
            and self.subject_detected
            and self.approach_requested
            and not self.subject_found
        )

    @property
    def can_press_launch(self) -> bool:
        return self.launch_ready and not self.mission_launched

    @property
    def approach_success(self) -> bool:
        return self.subject_found

    @property
    def runtime_route_revision(self) -> int:
        return self._runtime_route_revision

    @property
    def runtime_route_update_count(self) -> int:
        return self._runtime_route_update_count

    @property
    def search_started(self) -> bool:
        return self._search_started

    @property
    def ingress_sweep_active(self) -> bool:
        """Stage 2: sensors sweep during the parallel ingress, before TP.

        True while the vehicle is transiting to the search area in the ROUTE
        phase and the TP-arrival search clock (``_search_started``) has not yet
        started. The planning engine treats such a vehicle as sensing so the
        ingress corridor is searched (and its belief reduced) en route.
        """
        return (
            self.mission_launched
            and not self.emergency_mode
            and self.flight_phase == "ROUTE"
            and not self._search_started
        )

    @property
    def pending_runtime_route_revision(self) -> int:
        return self._pending_runtime_route_revision

    @property
    def manual_route_active(self) -> bool:
        return self._manual_route_active

    @property
    def manual_route_hold_active(self) -> bool:
        return self._manual_route_active and self.manual_route_hold_remaining_s > 0.0

    @property
    def manual_route_hold_remaining_s(self) -> float:
        return max(
            0.0,
            self._manual_route_hold_until_s - self.simulation_elapsed_s,
        )

    def load_mission(
        self,
        store: SiteStore,
        vehicle_id: int | None = None,
    ) -> None:
        vehicle_id = (
            store.active_vehicle_id if vehicle_id is None else int(vehicle_id)
        )
        route_points = store.waypoints_for(vehicle_id)
        signature = (
            vehicle_id,
            tuple(
                (
                    code,
                    site.latitude,
                    site.longitude,
                    site.altitude_m,
                )
                for code, site in sorted(store.sites.items())
            ),
            tuple(
                (
                    point.code,
                    point.latitude,
                    point.longitude,
                    point.altitude_m,
                )
                for point in route_points
            ),
            tuple(
                (
                    subj.track_id,
                    subj.latitude,
                    subj.longitude,
                    subj.speed_mps,
                    subj.heading_deg,
                    subj.position_uncertainty_m,
                    subj.source,
                    tuple(
                        tuple(sorted(segment.items()))
                        for segment in subj.motion_profile
                    ),
                )
                for subj in store.initial_subjects
            ),
        )
        self.sync_plan_readiness(store, vehicle_id)
        if signature == self._mission_signature:
            return
        self._mission_signature = signature
        self.vehicle.code = f"UAV-{vehicle_id:02d}"
        self._route_points = [
            (
                point.latitude,
                point.longitude,
                point.altitude_m,
                point.code,
            )
            for point in route_points
        ]
        safe_zone = next(
            (zone for zone in store.zones if zone.zone_type == "SAFE"),
            None,
        )
        safe_center = safe_zone.center() if safe_zone is not None else None
        self._return_point = (
            (
                safe_center.latitude,
                safe_center.longitude,
                60.0,
                "SAFE",
            )
            if safe_center is not None
            else None
        )
        self.mission_loaded = (
            store.shared_configuration_ready and bool(route_points)
        )
        self._reset_execution()

        launcher = store.sites.get("LC") or store.sites.get("GCS")
        if launcher is not None:
            self.center_latitude = launcher.latitude
            self.center_longitude = launcher.longitude
            self.vehicle.latitude = launcher.latitude
            self.vehicle.longitude = launcher.longitude
            self.vehicle.altitude_m = launcher.altitude_m
            if store.initial_subjects:
                self._load_mission_subjects(store)
            else:
                self._rebase_demo_subjects(
                    launcher.latitude,
                    launcher.longitude,
                )

    def update_route_from_store(
        self,
        store: SiteStore,
        vehicle_id: int,
    ) -> None:
        """Replace only an in-flight route while preserving execution state."""
        vehicle_id = int(vehicle_id)
        route_points = store.waypoints_for(vehicle_id)
        previous_waypoint = (
            self._route_points[self.current_waypoint_index]
            if self._route_points
            and self.current_waypoint_index < len(self._route_points)
            else None
        )
        self._route_points = [
            (
                point.latitude,
                point.longitude,
                point.altitude_m,
                point.code,
            )
            for point in route_points
        ]
        self._mission_signature = (
            vehicle_id,
            tuple(
                (
                    code,
                    site.latitude,
                    site.longitude,
                    site.altitude_m,
                )
                for code, site in sorted(store.sites.items())
            ),
            tuple(
                (
                    point.code,
                    point.latitude,
                    point.longitude,
                    point.altitude_m,
                )
                for point in route_points
            ),
        )
        self.sync_plan_readiness(store, vehicle_id)
        if not self._route_points:
            self.current_waypoint_index = 0
            self.completed_route_segment_count = 0
            return
        if previous_waypoint is not None:
            self.current_waypoint_index = min(
                range(len(self._route_points)),
                key=lambda index: horizontal_distance_m(
                    previous_waypoint[0],
                    previous_waypoint[1],
                    self._route_points[index][0],
                    self._route_points[index][1],
                ),
            )
        else:
            self.current_waypoint_index = min(
                self.current_waypoint_index,
                len(self._route_points) - 1,
            )
        self.completed_route_segment_count = min(
            self.current_waypoint_index,
            len(self._route_points),
        )

    def set_external_approach(self, solution: dict | None) -> None:
        pass

    def approach_route_payload(self) -> list[dict[str, float | str]]:
        """Return the live AUTO approach route for map display."""
        if not self.approach_requested or self.subject_found:
            return []
        points: list[dict[str, float | str]] = []
        if (
            self.flight_phase == "DETECTION_TRANSIT"
            and self._detection_waypoint is not None
        ):
            points.append(
                {
                    "latitude": self._detection_waypoint[0],
                    "longitude": self._detection_waypoint[1],
                    "altitude_m": self._detection_waypoint[2],
                    "code": self._detection_waypoint[3],
                }
            )
        return points

    def route_points_payload(self) -> list[dict[str, float | str]]:
        if not self._route_points:
            return []
        start_index = min(
            max(0, self.current_waypoint_index),
            len(self._route_points) - 1,
        )
        return [
            {
                "latitude": point[0],
                "longitude": point[1],
                "altitude_m": point[2],
                "code": point[3],
            }
            for point in self._route_points[start_index:]
        ]

    def queue_runtime_route(
        self,
        revision: int,
        points: tuple[dict, ...] | list[dict],
    ) -> bool:
        """Accept an automatic probability-weighted route revision.

        Ingress is protected until the common rally point. During search, a
        confirmed planner revision immediately replaces the AUTO waypoint
        suffix so the operator can see the route being recomputed in real time.
        """
        if self.manual_route_hold_active:
            # An operator-applied route owns the command path for a bounded
            # hold window.  The background RHP/PF estimator keeps running, but
            # its candidate must not overwrite the manual edit immediately.
            return False
        revision = int(revision)
        if revision <= max(
            self._runtime_route_revision,
            self._pending_runtime_route_revision,
        ):
            return False
        parsed = [
            (
                float(point["latitude"]),
                float(point["longitude"]),
                max(0.0, float(point.get("altitude_m", 600.0))),
                str(point.get("code") or f"AUTO{index:02d}"),
            )
            for index, point in enumerate(points, start=1)
        ]
        if not parsed:
            return False
        self._pending_runtime_route = parsed
        self._pending_runtime_route_revision = revision
        if (
            self.flight_phase == "ROUTE"
            and (
                self._search_started
                or self.completed_route_segment_count >= 1
            )
        ):
            self._search_started = True
            self._commit_pending_runtime_route()
        return True

    def _commit_pending_runtime_route(self) -> bool:
        if not self._pending_runtime_route:
            return False
        self._route_points = self._pending_runtime_route
        self._runtime_route_revision = self._pending_runtime_route_revision
        self._runtime_route_update_count += 1
        self._pending_runtime_route = None
        self._pending_runtime_route_revision = 0
        self._manual_route_active = False
        self._manual_route_hold_until_s = 0.0
        self.current_waypoint_index = 0
        self.completed_route_segment_count = 0
        return True

    def apply_manual_runtime_route(
        self,
        points: tuple[dict, ...] | list[dict],
        *,
        hold_duration_s: float = 50.0,
    ) -> bool:
        """Apply an operator-edited route without resetting flight execution.

        Automatic RHP proposals continue to be evaluated during the hold, but
        ``queue_runtime_route`` rejects them until the bounded hold expires.
        This keeps a just-applied map edit visible and commandable while still
        returning control to the automatic planner afterwards.
        """
        parsed = [
            (
                float(point["latitude"]),
                float(point["longitude"]),
                max(0.0, float(point.get("altitude_m", 600.0))),
                str(point.get("code") or f"MWP{index:03d}"),
            )
            for index, point in enumerate(points, start=1)
        ]
        if not parsed:
            return False
        previous_waypoint = (
            self._route_points[self.current_waypoint_index]
            if self._route_points
            and self.current_waypoint_index < len(self._route_points)
            else None
        )
        self._route_points = parsed
        self._pending_runtime_route = None
        self._pending_runtime_route_revision = 0
        self._runtime_route_revision = max(1, self._runtime_route_revision)
        self._runtime_route_update_count += 1
        self._manual_route_active = True
        self._manual_route_hold_until_s = (
            self.simulation_elapsed_s + max(1.0, float(hold_duration_s))
        )
        if previous_waypoint is None:
            self.current_waypoint_index = min(
                self.current_waypoint_index,
                len(parsed) - 1,
            )
        else:
            self.current_waypoint_index = min(
                range(len(parsed)),
                key=lambda index: horizontal_distance_m(
                    previous_waypoint[0],
                    previous_waypoint[1],
                    parsed[index][0],
                    parsed[index][1],
                ),
            )
        self.completed_route_segment_count = min(
            self.current_waypoint_index,
            len(parsed),
        )
        return True

    def runtime_route_payload(self) -> list[dict[str, float | str]]:
        if self._runtime_route_revision <= 0:
            return []
        return [
            {
                "latitude": point[0],
                "longitude": point[1],
                "altitude_m": point[2],
                "code": point[3],
            }
            for point in self._route_points
        ]

    def pending_runtime_route_payload(self) -> list[dict[str, float | str]]:
        return [
            {
                "latitude": point[0],
                "longitude": point[1],
                "altitude_m": point[2],
                "code": point[3],
            }
            for point in (self._pending_runtime_route or [])
        ]

    def _rebase_demo_subjects(
        self,
        latitude: float,
        longitude: float,
    ) -> None:
        now = time.time()
        tracks_by_id = {track.track_id: track for track in self.subjects}
        for (
            track_id,
            subject_type,
            country,
            platform_name,
            distance_m,
            radial_heading,
            speed_mps,
            movement_heading,
            age_s,
        ) in DEMO_SUBJECT_SPECS:
            track_latitude, track_longitude = destination_position(
                latitude,
                longitude,
                distance_m,
                radial_heading,
            )
            track = tracks_by_id.get(track_id)
            if track is None:
                track = SubjectTrack(
                    track_id=track_id,
                    subject_type=subject_type,
                    country=country,
                    platform_name=platform_name,
                    latitude=track_latitude,
                    longitude=track_longitude,
                    altitude_m=0.0,
                    speed_mps=speed_mps,
                    heading_deg=movement_heading,
                    first_tracked_at=now - age_s,
                )
                self.subjects.append(track)
            else:
                track.subject_type = subject_type
                track.country = country
                track.platform_name = platform_name
                track.latitude = track_latitude
                track.longitude = track_longitude
                track.altitude_m = 0.0
                track.speed_mps = speed_mps
                track.heading_deg = movement_heading
                track.first_tracked_at = now - age_s
                track.found = False
                track.position_uncertainty_m = 500.0
                track.source = "LOCAL"
                track.motion_profile = []
        self._reset_predictors()

    def _load_mission_subjects(self, store: SiteStore) -> None:
        now = time.time()
        self.subjects = [
            SubjectTrack(
                track_id=subj.track_id,
                subject_type=subj.subject_type,
                country=subj.country,
                platform_name=subj.platform_name,
                latitude=subj.latitude,
                longitude=subj.longitude,
                altitude_m=subj.altitude_m,
                speed_mps=min(40.0 / 3.6, subj.speed_mps),
                heading_deg=subj.heading_deg,
                first_tracked_at=now - subj.track_age_s,
                position_uncertainty_m=subj.position_uncertainty_m,
                source=subj.source,
                motion_profile=[
                    dict(segment) for segment in subj.motion_profile
                ],
            )
            for subj in store.initial_subjects
        ]
        if self.subjects and not any(
            track.track_id == self.selected_track_id
            for track in self.subjects
        ):
            self.selected_track_id = self.subjects[0].track_id
        self._reset_predictors()

    def _reset_execution(self) -> None:
        self.mission_launched = False
        self.subject_designated = False
        self.subject_detected = False
        self.approach_approved = False
        self.launch_requested = False
        self.approach_requested = False
        self.subject_found = False
        self.detection_source_vehicle_id = None
        self.emergency_mode = False
        self.flight_phase = "STANDBY"
        self.current_waypoint_index = 0
        self.completed_route_segment_count = 0
        self.guidance_elapsed_s = 0.0
        self.simulation_elapsed_s = 0.0
        self.search_elapsed_s = 0.0
        self.last_launch_action = "NONE"
        self.shutdown_position = None
        self.flight_path = []
        self._loiter_center = None
        self._detection_waypoint = None
        self._mitl_return_waypoint = None
        self._phase_elapsed_s = 0.0
        self._rendezvous_solution = None
        self._rendezvous_elapsed_s = 0.0
        self._rendezvous_vehicle_start = None
        self._rendezvous_subject_start = None
        self._runtime_route_revision = 0
        self._runtime_route_update_count = 0
        self._search_started = False
        self._pending_runtime_route_revision = 0
        self._pending_runtime_route = None
        self._manual_route_hold_until_s = 0.0
        self._manual_route_active = False
        self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        for name in self.mission_status:
            self.mission_status[name] = False
        for subject in self.subjects:
            subject.found = False

    def resume_search(self) -> bool:
        """Transition from FOUND back to ROUTE for the next search cycle.

        Called when the current subject has been found and there are remaining
        unfound subjects.  Resets approach/detection state but preserves
        launch, elapsed time, and flight path history.
        """
        if not self.subject_found:
            return False
        self.subject_found = False
        self.subject_designated = False
        self.subject_detected = False
        self.approach_approved = False
        self.approach_requested = False
        self.detection_source_vehicle_id = None
        self.shutdown_position = None
        self.flight_phase = "ROUTE"
        self.current_waypoint_index = 0
        self.completed_route_segment_count = 0
        self.guidance_elapsed_s = 0.0
        self._phase_elapsed_s = 0.0
        self._loiter_center = None
        self._detection_waypoint = None
        self._rendezvous_solution = None
        self._rendezvous_elapsed_s = 0.0
        self._rendezvous_vehicle_start = None
        self._rendezvous_subject_start = None
        self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        for name in self.mission_status:
            self.mission_status[name] = False
        self._reset_predictors()
        return True

    def select_subject(self, track_id: int) -> bool:
        if not any(track.track_id == track_id for track in self.subjects):
            return False
        self.selected_track_id = track_id
        return True

    def designate_subject(
        self,
        track_id: int,
        *,
        cooperative_approach: tuple[float, float, float, str] | None = None,
        detector_vehicle_id: int | None = None,
    ) -> bool:
        if not self.select_subject(track_id):
            return False
        if not self.mission_launched:
            return True
        self.subject_designated = True
        self.subject_detected = True
        self.approach_approved = True
        self.guidance_elapsed_s = 0.0
        self._phase_elapsed_s = 0.0
        self.launch_requested = False
        self.approach_requested = True
        self.subject_found = False
        self.detection_source_vehicle_id = detector_vehicle_id
        self.shutdown_position = None
        self.vehicle.speed_mps = INITIAL_APPROACH_SPEED_MPS
        self._rendezvous_solution = self._calculate_predicted_rendezvous()
        self._detection_waypoint = cooperative_approach
        self._pending_runtime_route = None
        self._pending_runtime_route_revision = 0
        self._manual_route_hold_until_s = 0.0
        self._manual_route_active = False
        self._loiter_center = None
        self._rendezvous_elapsed_s = 0.0
        self._rendezvous_vehicle_start = None
        self._rendezvous_subject_start = None
        for name in self.mission_status:
            self.mission_status[name] = False
        if not self.emergency_mode:
            if cooperative_approach is not None:
                self.flight_phase = "DETECTION_TRANSIT"
            else:
                # AUTO doctrine: the first detection aborts every search arc.
                # All six SAR immediately turn toward the live ground subject.
                self.flight_phase = "INITIAL_APPROACH"
                self.mission_status["초기접근"] = True
        return True

    def sync_plan_readiness(
        self,
        store: SiteStore,
        vehicle_id: int | None = None,
    ) -> None:
        if vehicle_id is None:
            try:
                vehicle_id = int(self.vehicle.code.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                vehicle_id = store.active_vehicle_id
        ready = (
            store.shared_configuration_ready
            and bool(store.waypoints_for(vehicle_id))
        )
        self.readiness["AVS"] = True
        self.readiness["LC"] = "LC" in store.sites
        self.readiness["RDR"] = "RDR" in store.sites
        self.readiness["DL"] = True
        self.readiness["GCS"] = "GCS" in store.sites
        self.mission_loaded = ready

    def request_simulated_launch(self) -> bool:
        self.last_launch_action = "DENIED"
        if not self.launch_ready:
            return False
        if not self.mission_launched:
            self.mission_launched = True
            self.flight_phase = "ROUTE"
            self.vehicle.speed_mps = SAR_INGRESS_SPEED_MPS
            self.current_waypoint_index = 0
            self.completed_route_segment_count = 0
            self.last_launch_action = "MISSION_LAUNCH"
            return True
        return False

    def stop_approach(self) -> None:
        self.approach_approved = False
        self.approach_requested = False
        self.subject_designated = False
        self.subject_detected = False
        self.detection_source_vehicle_id = None
        if self.mission_launched and not self.emergency_mode:
            self.flight_phase = "ROUTE"
            self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        self._mitl_return_waypoint = None
        self._detection_waypoint = None
        self._rendezvous_solution = None
        for name in self.mission_status:
            self.mission_status[name] = False

    def request_safe_return_via_waypoint(self) -> bool:
        self.approach_approved = False
        self.approach_requested = False
        self.subject_designated = False
        self.subject_detected = False
        self._rendezvous_solution = None
        self._loiter_center = None
        self._detection_waypoint = None
        for name in self.mission_status:
            self.mission_status[name] = False
        if not self.mission_launched or self.emergency_mode:
            return False

        self._mitl_return_waypoint = self._nearest_route_point_to_vehicle()
        self.flight_phase = (
            "MANUAL_WAYPOINT_RETURN"
            if self._mitl_return_waypoint is not None
            else "MANUAL_SAFE_RETURN"
        )
        self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        return True

    def toggle_emergency(self) -> bool:
        self.emergency_mode = not self.emergency_mode
        if self.emergency_mode:
            self._phase_before_emergency = self.flight_phase
            self.flight_phase = "EMERGENCY_SAFE_RETURN"
            self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        else:
            self.flight_phase = (
                self._phase_before_emergency
                if self.mission_launched
                else "STANDBY"
            )
            self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
            if self.flight_phase == "FINAL_APPROACH":
                self._begin_rendezvous_animation()
        return self.emergency_mode

    def _nearest_route_point_to_subject(
        self,
        subject: SubjectTrack | None,
    ) -> tuple[float, float, float, str] | None:
        if subject is None or not self._route_points:
            return None
        return min(
            self._route_points,
            key=lambda candidate: horizontal_distance_m(
                subject.latitude,
                subject.longitude,
                candidate[0],
                candidate[1],
            ),
        )

    def _nearest_route_point_to_vehicle(
        self,
    ) -> tuple[float, float, float, str] | None:
        if not self._route_points:
            return None
        return min(
            self._route_points,
            key=lambda candidate: horizontal_distance_m(
                self.vehicle.latitude,
                self.vehicle.longitude,
                candidate[0],
                candidate[1],
            ),
        )

    def tick(self, dt_override_s: float | None = None) -> None:
        elapsed = self.elapsed_s
        dt = (
            max(0.01, min(float(dt_override_s), 1.0))
            if dt_override_s is not None
            else (
                max(0.05, min(elapsed - self._last_tick_s, MAX_TICK_DT_S))
                if self._last_tick_s
                else 0.25
            )
        )
        self._last_tick_s = elapsed
        search_was_active = self._search_started
        if self.mission_launched:
            self.simulation_elapsed_s += dt * DEMO_GROUND_TIME_SCALE
            if search_was_active:
                # The CPP/RHP experiment defines t=0 at physical TP arrival.
                # Ingress and subject-scenario time must not consume the first
                # 25-second receding-horizon interval.
                self.search_elapsed_s += dt * DEMO_GROUND_TIME_SCALE
            self._advance_subjects(dt)

        if self.emergency_mode:
            self._advance_emergency_return(dt)
            return
        if not self.mission_launched:
            return

        if self.flight_phase == "ROUTE":
            self._advance_route(dt)
        elif self.flight_phase == "DETECTION_TRANSIT":
            self._advance_detection_transit(dt)
        elif self.flight_phase == "INITIAL_APPROACH":
            self._advance_initial_guidance(dt)
        elif self.flight_phase == "MIDCOURSE_APPROACH":
            self._advance_midcourse_guidance(dt)
        elif self.flight_phase == "FINAL_APPROACH":
            self._advance_approach(dt)
        elif self.flight_phase in {
            "MANUAL_WAYPOINT_RETURN",
            "MANUAL_SAFE_RETURN",
        }:
            self._advance_mitl_safe_return(dt)
        if self.mission_launched:
            if not self.flight_path or horizontal_distance_m(
                self.flight_path[-1]["latitude"],
                self.flight_path[-1]["longitude"],
                self.vehicle.latitude,
                self.vehicle.longitude,
            ) >= 120.0:
                self.flight_path.append(
                    {
                        "latitude": self.vehicle.latitude,
                        "longitude": self.vehicle.longitude,
                        "altitude_m": self.vehicle.altitude_m,
                    }
                )
                self.flight_path = self.flight_path[-1200:]

    def _advance_route(self, dt: float) -> None:
        if not self._route_points:
            return
        target_code = str(self._route_points[self.current_waypoint_index][3])
        has_ingress = any(
            str(point[3]).startswith("IGR") for point in self._route_points
        )
        if has_ingress:
            # Formation ingress: hold ingress speed while heading to any ingress
            # (IGR) waypoint, i.e. all the way to the TP line, then slow to the
            # SPX search speed only after crossing it.
            on_initial_ingress = (
                self._runtime_route_revision == 0
                and target_code.startswith("IGR")
            )
        else:
            on_initial_ingress = (
                self._runtime_route_revision == 0
                and self.completed_route_segment_count == 0
                and self.current_waypoint_index == 0
            )
        self.vehicle.speed_mps = (
            SAR_INGRESS_SPEED_MPS
            if on_initial_ingress
            else SAR_SEARCH_SPEED_MPS
        )
        waypoint = self._route_points[self.current_waypoint_index]
        if str(waypoint[3]).startswith("RHP-T"):
            self.vehicle.speed_mps = SAR_CRUISE_SPEED_MPS
        reached = self._move_vehicle_toward(
            waypoint[0],
            waypoint[1],
            waypoint[2],
            self.vehicle.speed_mps * dt * DEMO_FLIGHT_TIME_SCALE,
        )
        if reached:
            if on_initial_ingress:
                # Search begins only once the LAST ingress leg (the TP-line
                # crossing) is complete — the next waypoint is a search leg.
                next_index = (
                    self.current_waypoint_index + 1
                ) % len(self._route_points)
                next_code = str(self._route_points[next_index][3])
                if not has_ingress or not next_code.startswith("IGR"):
                    self._search_started = True
            if self._commit_pending_runtime_route():
                return
            self.completed_route_segment_count = min(
                len(self._route_points),
                self.completed_route_segment_count + 1,
            )
            self.current_waypoint_index = (
                self.current_waypoint_index + 1
            ) % len(self._route_points)

    def _advance_detection_transit(self, dt: float) -> None:
        if self._detection_waypoint is None:
            self.flight_phase = "INITIAL_APPROACH"
            self.mission_status["초기접근"] = True
            self._phase_elapsed_s = 0.0
            return
        reached = self._move_vehicle_toward(
            self._detection_waypoint[0],
            self._detection_waypoint[1],
            max(80.0, self._detection_waypoint[2]),
            INITIAL_APPROACH_SPEED_MPS * dt * DEMO_FLIGHT_TIME_SCALE,
        )
        if reached:
            self.flight_phase = "INITIAL_APPROACH"
            self.vehicle.speed_mps = INITIAL_APPROACH_SPEED_MPS
            self._phase_elapsed_s = 0.0
            self.mission_status["초기접근"] = True

    def _advance_initial_guidance(self, dt: float) -> None:
        subject = self.selected_subject
        if subject is not None:
            self._move_vehicle_toward(
                subject.latitude,
                subject.longitude,
                600.0,
                INITIAL_APPROACH_SPEED_MPS * dt * MIDCOURSE_APPROACH_TIME_SCALE,
            )
        self._phase_elapsed_s += dt * MIDCOURSE_APPROACH_TIME_SCALE
        if self._phase_elapsed_s >= INITIAL_APPROACH_DURATION_S:
            self.flight_phase = "MIDCOURSE_APPROACH"
            self.vehicle.speed_mps = MIDCOURSE_APPROACH_SPEED_MPS
            self.mission_status["중기접근"] = True
            self._phase_elapsed_s = 0.0

    def _advance_midcourse_guidance(self, dt: float) -> None:
        self._phase_elapsed_s += dt * MIDCOURSE_APPROACH_TIME_SCALE
        solution = self._rendezvous_solution
        if solution is None or str(solution.get("model", "")).startswith(
            "CV-KF"
        ):
            solution = self._calculate_predicted_rendezvous()
            self._rendezvous_solution = solution
        if solution is None:
            return
        distance = horizontal_distance_m(
            self.vehicle.latitude,
            self.vehicle.longitude,
            float(solution["latitude"]),
            float(solution["longitude"]),
        )
        minimum_midcourse_complete = (
            self._phase_elapsed_s >= MINIMUM_MIDCOURSE_APPROACH_DURATION_S
        )
        if distance <= FINAL_ENTRY_RADIUS_M:
            if minimum_midcourse_complete:
                self._start_final_guidance()
                return
            subject = self.selected_subject
            if subject is not None:
                self._move_vehicle_toward(
                    subject.latitude,
                    subject.longitude,
                    600.0,
                    MIDCOURSE_APPROACH_SPEED_MPS
                    * dt
                    * MIDCOURSE_APPROACH_TIME_SCALE,
                )
            return
        step = min(
            MIDCOURSE_APPROACH_SPEED_MPS * dt * MIDCOURSE_APPROACH_TIME_SCALE,
            max(1.0, distance - FINAL_ENTRY_RADIUS_M),
        )
        self._move_vehicle_toward(
            float(solution["latitude"]),
            float(solution["longitude"]),
            600.0,
            step,
        )
        remaining = horizontal_distance_m(
            self.vehicle.latitude,
            self.vehicle.longitude,
            float(solution["latitude"]),
            float(solution["longitude"]),
        )
        if (
            minimum_midcourse_complete
            and remaining <= FINAL_ENTRY_RADIUS_M + 1.0
        ):
            self._start_final_guidance()

    def _start_final_guidance(self) -> None:
        self.flight_phase = "FINAL_APPROACH"
        self.vehicle.speed_mps = MIDCOURSE_APPROACH_SPEED_MPS
        self.mission_status["최종접근"] = True
        self._phase_elapsed_s = 0.0
        self._begin_rendezvous_animation()

    def _advance_approach(self, dt: float) -> None:
        subject = self.selected_subject
        if subject is None or subject.found:
            return
        if (
            self._rendezvous_solution is None
            or str(self._rendezvous_solution.get("model", "")).startswith(
                "CV-KF"
            )
        ):
            self._rendezvous_solution = self._calculate_predicted_rendezvous()
        self._phase_elapsed_s += dt * MIDCOURSE_APPROACH_TIME_SCALE
        fraction = min(1.0, self._phase_elapsed_s / 5.0)
        ramp_fraction = min(
            1.0,
            self._phase_elapsed_s / FINAL_SPEED_RAMP_S,
        )
        smooth_ramp = ramp_fraction * ramp_fraction * (
            3.0 - 2.0 * ramp_fraction
        )
        self.vehicle.speed_mps = (
            MIDCOURSE_APPROACH_SPEED_MPS
            + (FINAL_APPROACH_SPEED_MPS - MIDCOURSE_APPROACH_SPEED_MPS)
            * smooth_ramp
        )
        self.mission_status["최종접근"] = True
        self.mission_status["셔터 ON"] = fraction >= 0.20
        self.mission_status["TRACK HOLD"] = fraction >= 0.40
        self.mission_status["TDD 탐지"] = fraction >= 0.65
        self.mission_status["확인 완료"] = fraction >= 0.85
        reached = self._move_vehicle_toward_3d(
            subject.latitude,
            subject.longitude,
            subject.altitude_m,
            self.vehicle.speed_mps * dt * MIDCOURSE_APPROACH_TIME_SCALE,
        )
        horizontal_separation_m = horizontal_distance_m(
            self.vehicle.latitude,
            self.vehicle.longitude,
            subject.latitude,
            subject.longitude,
        )
        altitude_separation_m = abs(
            self.vehicle.altitude_m - subject.altitude_m
        )
        slant_separation_m = math.hypot(
            horizontal_separation_m,
            altitude_separation_m,
        )
        # Hold a full-frame red final hold at close range before changing to
        # SHUT DOWN.  This lets the search camera image reach essentially 100% width
        # instead of jumping from a small box directly to the found overlay.
        if reached or slant_separation_m <= 5.0:
            self._rendezvous_elapsed_s += dt * MIDCOURSE_APPROACH_TIME_SCALE
        else:
            self._rendezvous_elapsed_s = 0.0
        if self._rendezvous_elapsed_s >= FINAL_CONTACT_DWELL_S:
            destination = (
                subject.latitude,
                subject.longitude,
                subject.altitude_m,
            )
            self._complete_rendezvous(subject, destination)

    def _begin_rendezvous_animation(self) -> None:
        subject = self.selected_subject
        if subject is None:
            return
        if self._rendezvous_solution is None:
            self._rendezvous_solution = self._calculate_predicted_rendezvous()
        self._rendezvous_elapsed_s = 0.0
        self._rendezvous_vehicle_start = (
            self.vehicle.latitude,
            self.vehicle.longitude,
            self.vehicle.altitude_m,
        )
        self._rendezvous_subject_start = (
            subject.latitude,
            subject.longitude,
            subject.altitude_m,
        )

    @staticmethod
    def _interpolate(start: float, end: float, fraction: float) -> float:
        return start + (end - start) * max(0.0, min(1.0, fraction))

    def _complete_rendezvous(
        self,
        subject: SubjectTrack,
        rendezvous_position: tuple[float, float, float],
    ) -> None:
        subject.found = True
        self.subject_found = True
        self.approach_requested = False
        self.flight_phase = "FOUND"
        self.vehicle.speed_mps = FINAL_APPROACH_SPEED_MPS
        subject.latitude, subject.longitude, subject.altitude_m = rendezvous_position
        self.vehicle.latitude = rendezvous_position[0]
        self.vehicle.longitude = rendezvous_position[1]
        self.vehicle.altitude_m = rendezvous_position[2]
        self.shutdown_position = rendezvous_position
        for name in self.mission_status:
            self.mission_status[name] = True

    def _advance_emergency_return(self, dt: float) -> None:
        if self._return_point is None:
            self.flight_phase = "RETURNED"
            self.vehicle.speed_mps = 0.0
            return
        reached = self._move_vehicle_toward(
            self._return_point[0],
            self._return_point[1],
            max(30.0, self._return_point[2]),
            SAR_CRUISE_SPEED_MPS * dt * DEMO_FLIGHT_TIME_SCALE,
        )
        if reached:
            self.flight_phase = "RETURNED"
            self.vehicle.speed_mps = 0.0

    def _advance_mitl_safe_return(self, dt: float) -> None:
        step_m = SAR_CRUISE_SPEED_MPS * dt * DEMO_FLIGHT_TIME_SCALE
        if (
            self.flight_phase == "MANUAL_WAYPOINT_RETURN"
            and self._mitl_return_waypoint is not None
        ):
            reached_waypoint = self._move_vehicle_toward(
                self._mitl_return_waypoint[0],
                self._mitl_return_waypoint[1],
                max(80.0, self._mitl_return_waypoint[2]),
                step_m,
            )
            if not reached_waypoint:
                return
            self.flight_phase = "MANUAL_SAFE_RETURN"
            return

        if self._return_point is None:
            self.flight_phase = "RETURNED"
            self.vehicle.speed_mps = 0.0
            return
        reached_safe_zone = self._move_vehicle_toward(
            self._return_point[0],
            self._return_point[1],
            max(30.0, self._return_point[2]),
            step_m,
        )
        if reached_safe_zone:
            self.flight_phase = "RETURNED"
            self.vehicle.speed_mps = 0.0

    def _move_vehicle_toward(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        step_m: float,
    ) -> bool:
        distance = horizontal_distance_m(
            self.vehicle.latitude,
            self.vehicle.longitude,
            latitude,
            longitude,
        )
        if distance <= max(1.0, step_m):
            self.vehicle.latitude = latitude
            self.vehicle.longitude = longitude
            self.vehicle.altitude_m = altitude_m
            return True
        heading = bearing_deg(
            self.vehicle.latitude,
            self.vehicle.longitude,
            latitude,
            longitude,
        )
        self.vehicle.heading_deg = heading
        self.vehicle.latitude, self.vehicle.longitude = destination_position(
            self.vehicle.latitude,
            self.vehicle.longitude,
            step_m,
            heading,
        )
        altitude_fraction = min(1.0, step_m / max(1.0, distance))
        self.vehicle.altitude_m += (
            altitude_m - self.vehicle.altitude_m
        ) * altitude_fraction
        return False

    def _move_vehicle_toward_3d(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        step_m: float,
    ) -> bool:
        """Move along the full slant vector for final approach camera guidance."""
        horizontal_m = horizontal_distance_m(
            self.vehicle.latitude,
            self.vehicle.longitude,
            latitude,
            longitude,
        )
        vertical_m = float(altitude_m) - self.vehicle.altitude_m
        slant_m = math.hypot(horizontal_m, vertical_m)
        if slant_m <= max(1.0, step_m):
            self.vehicle.latitude = latitude
            self.vehicle.longitude = longitude
            self.vehicle.altitude_m = altitude_m
            return True
        fraction = min(1.0, max(0.0, step_m) / slant_m)
        horizontal_step_m = horizontal_m * fraction
        if horizontal_step_m > 1e-6:
            heading = bearing_deg(
                self.vehicle.latitude,
                self.vehicle.longitude,
                latitude,
                longitude,
            )
            self.vehicle.heading_deg = heading
            self.vehicle.latitude, self.vehicle.longitude = destination_position(
                self.vehicle.latitude,
                self.vehicle.longitude,
                horizontal_step_m,
                heading,
            )
        self.vehicle.altitude_m += vertical_m * fraction
        return False

    def _advance_subjects(self, dt: float) -> None:
        for track in self.subjects:
            if not track.found:
                simulated_dt = dt * DEMO_GROUND_TIME_SCALE
                orbital_motion = False
                if track.motion_profile:
                    segment = next(
                        (
                            item
                            for item in track.motion_profile
                            if float(item.get("start_s", 0.0))
                            <= self.simulation_elapsed_s
                            < float(item.get("end_s", float("inf")))
                        ),
                        track.motion_profile[-1],
                    )
                    track.speed_mps = min(
                        40.0 / 3.6,
                        max(
                            0.0,
                            float(
                                segment.get(
                                    "speed_kph",
                                    track.speed_mps * 3.6,
                                )
                            )
                            / 3.6,
                        ),
                    )
                    if str(segment.get("mode", "")).upper() == "ORBIT":
                        center_latitude = float(
                            segment.get("center_latitude", track.latitude)
                        )
                        center_longitude = float(
                            segment.get("center_longitude", track.longitude)
                        )
                        radius_m = max(
                            50.0,
                            float(segment.get("radius_m", 2_400.0)),
                        )
                        clockwise = bool(segment.get("clockwise", True))
                        radial_bearing = bearing_deg(
                            center_latitude,
                            center_longitude,
                            track.latitude,
                            track.longitude,
                        )
                        angular_step_deg = math.degrees(
                            track.speed_mps * simulated_dt / radius_m
                        )
                        if not clockwise:
                            angular_step_deg *= -1.0
                        radial_bearing = (
                            radial_bearing + angular_step_deg
                        ) % 360.0
                        track.latitude, track.longitude = destination_position(
                            center_latitude,
                            center_longitude,
                            radius_m,
                            radial_bearing,
                        )
                        track.heading_deg = (
                            radial_bearing + (90.0 if clockwise else -90.0)
                        ) % 360.0
                        orbital_motion = True
                    elif "turn_rate_dps" in segment:
                        track.heading_deg = (
                            track.heading_deg
                            + float(segment["turn_rate_dps"])
                            * simulated_dt
                        ) % 360.0
                    elif "heading_deg" in segment:
                        track.heading_deg = float(
                            segment["heading_deg"]
                        ) % 360.0
                if not orbital_motion:
                    track.latitude, track.longitude = destination_position(
                        track.latitude,
                        track.longitude,
                        track.speed_mps * simulated_dt,
                        track.heading_deg,
                    )
                if not track.motion_profile:
                    track.heading_deg = (
                        track.heading_deg
                        + math.sin(
                            self.elapsed_s / 13.0 + track.track_id
                        )
                        * 0.12
                    ) % 360.0
            predictor = self._track_predictors.get(track.track_id)
            if predictor is None:
                predictor = self._new_predictor(track)
                self._track_predictors[track.track_id] = predictor
            predictor.update(
                track.latitude,
                track.longitude,
                track.altitude_m,
                dt,
            )

    def _new_predictor(
        self,
        track: SubjectTrack,
    ) -> ConstantVelocityTrackPredictor:
        predictor = ConstantVelocityTrackPredictor(
            self.center_latitude,
            self.center_longitude,
            track.latitude,
            track.longitude,
            track.altitude_m,
        )
        heading = math.radians(track.heading_deg)
        predictor.east.velocity = math.sin(heading) * track.speed_mps
        predictor.north.velocity = math.cos(heading) * track.speed_mps
        return predictor

    def _reset_predictors(self) -> None:
        self._track_predictors = {
            track.track_id: self._new_predictor(track)
            for track in self.subjects
        }

    def _calculate_predicted_rendezvous(self) -> dict | None:
        subject = self.selected_subject
        if subject is None:
            return None
        predictor = self._track_predictors.get(subject.track_id)
        if predictor is None:
            return None
        estimate = predictor.estimate()
        mean_latitude = math.radians(
            (self.vehicle.latitude + estimate.latitude) / 2.0
        )
        relative_east_m = (
            (estimate.longitude - self.vehicle.longitude)
            * EARTH_METERS_PER_DEGREE
            * math.cos(mean_latitude)
        )
        relative_north_m = (
            estimate.latitude - self.vehicle.latitude
        ) * EARTH_METERS_PER_DEGREE
        relative_up_m = estimate.altitude_m - self.vehicle.altitude_m
        pursuer_speed_mps = max(1.0, self.vehicle.speed_mps)
        approach_time_s, reachable = constant_velocity_approach_time(
            relative_east_m,
            relative_north_m,
            relative_up_m,
            estimate.velocity_east_mps,
            estimate.velocity_north_mps,
            estimate.velocity_up_mps,
            pursuer_speed_mps,
        )
        prediction = predictor.predict(approach_time_s)
        return {
            "code": "INT",
            "model": "CV-KF RELATIVE APPROACH",
            "horizon_s": approach_time_s,
            "max_horizon_s": PREDICTION_HORIZON_S,
            "reachable": reachable,
            "vehicle_code": self.vehicle.code,
            "vehicle_speed_mps": pursuer_speed_mps,
            "vehicle_latitude": self.vehicle.latitude,
            "vehicle_longitude": self.vehicle.longitude,
            "vehicle_altitude_m": self.vehicle.altitude_m,
            "latitude": prediction.latitude,
            "longitude": prediction.longitude,
            "altitude_m": prediction.altitude_m,
            "estimated_speed_mps": prediction.estimated_speed_mps,
            "estimated_heading_deg": prediction.estimated_heading_deg,
            "uncertainty_east_95_m": prediction.uncertainty_east_95_m,
            "uncertainty_north_95_m": prediction.uncertainty_north_95_m,
            "uncertainty_altitude_95_m": prediction.uncertainty_altitude_95_m,
        }

    def predicted_approach(self) -> dict | None:
        return None

    def render_dict(
        self,
        store: SiteStore,
        *,
        include_plan: bool = True,
        include_flight_path: bool = True,
    ) -> dict:
        subject_payloads = []
        for track in self.subjects:
            distance_m = horizontal_distance_m(
                self.vehicle.latitude,
                self.vehicle.longitude,
                track.latitude,
                track.longitude,
            )
            subject_payloads.append(
                {
                    **asdict(track),
                    "code": track.code,
                    "first_tracked_text": track.first_tracked_text,
                    "selected": track.track_id == self.selected_track_id,
                    "distance_m": distance_m,
                    "eta_s": distance_m / SAR_CRUISE_SPEED_MPS,
                }
            )
        shutdown_marker = None
        if self.shutdown_position is not None:
            shutdown_marker = {
                "latitude": self.shutdown_position[0],
                "longitude": self.shutdown_position[1],
                "altitude_m": self.shutdown_position[2],
                "label": "SHUT DOWN",
            }
        guidance_orbit = None
        emergency_return = None
        if self.emergency_mode and self._return_point is not None:
            emergency_return = {
                "latitude": self._return_point[0],
                "longitude": self._return_point[1],
                "altitude_m": self._return_point[2],
                "code": self._return_point[3],
                "phase": self.flight_phase,
            }
        mitl_return = None
        if self.flight_phase in {
            "MANUAL_WAYPOINT_RETURN",
            "MANUAL_SAFE_RETURN",
        } and self._return_point is not None:
            mitl_return = {
                "phase": self.flight_phase,
                "via_waypoint": (
                    {
                        "latitude": self._mitl_return_waypoint[0],
                        "longitude": self._mitl_return_waypoint[1],
                        "altitude_m": self._mitl_return_waypoint[2],
                        "code": self._mitl_return_waypoint[3],
                    }
                    if self.flight_phase == "MANUAL_WAYPOINT_RETURN"
                    and self._mitl_return_waypoint is not None
                    else None
                ),
                "safe_zone": {
                    "latitude": self._return_point[0],
                    "longitude": self._return_point[1],
                    "altitude_m": self._return_point[2],
                    "code": self._return_point[3],
                },
            }
        payload = {
            "vehicle": asdict(self.vehicle),
            "subjects": subject_payloads,
            "selected_track_id": self.selected_track_id,
            "subject_designated": self.subject_designated,
            "subject_detected": self.subject_detected,
            "detection_source_vehicle_id": self.detection_source_vehicle_id,
            "approach_approved": self.approach_approved,
            "predicted_approach": None,
            "readiness": dict(self.readiness),
            "mission_status": dict(self.mission_status),
            "mission_loaded": self.mission_loaded,
            "mission_launched": self.mission_launched,
            "flight_phase": self.flight_phase,
            "search_elapsed_s": self.search_elapsed_s,
            "current_waypoint_index": self.current_waypoint_index,
            "completed_route_segment_count": (
                self.completed_route_segment_count
            ),
            "automatic_mode": self.automatic_mode,
            "camera_mode": self.camera_mode,
            "launch_ready": self.launch_ready,
            "can_press_launch": self.can_press_launch,
            "launch_requested": self.launch_requested,
            "rendezvous_ready": self.rendezvous_ready,
            "approach_success": self.approach_success,
            "emergency_mode": self.emergency_mode,
            "guidance_orbit": guidance_orbit,
            "emergency_return": emergency_return,
            "mitl_return": mitl_return,
            "shutdown_marker": shutdown_marker,
            "sar_cruise_speed_mps": SAR_CRUISE_SPEED_MPS,
            "prediction_horizon_s": PREDICTION_HORIZON_S,
            "demo_time_scale": DEMO_FLIGHT_TIME_SCALE,
            "display_source": "LOCAL_SIMULATION",
        }
        if include_plan:
            plan = store.render_dict()
            hide_preplanned_search_routes(plan, store.mission_metadata)
            payload["plan"] = plan
        if include_flight_path:
            payload["flight_path"] = list(self.flight_path)
        return payload
