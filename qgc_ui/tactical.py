from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class SubjectTrack:
    """Synthetic track used by the non-transmitting training UI."""

    track_id: int
    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float
    heading_deg: float
    first_tracked_at: float

    @property
    def first_tracked_text(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.first_tracked_at))


@dataclass(frozen=True)
class TacticalSite:
    code: str
    label: str
    latitude: float
    longitude: float
    affiliation: str = "team"


@dataclass
class TacticalState:
    """Inert situational-awareness state.

    This model deliberately contains no transport or command-sending code.
    Launch and approach controls only change local simulation flags.
    """

    subjects: list[SubjectTrack] = field(default_factory=list)
    sites: list[TacticalSite] = field(default_factory=list)
    selected_track_id: int | None = None
    readiness: dict[str, bool] = field(
        default_factory=lambda: {
            "AVS": True,
            "LC": True,
            "RDR": True,
            "DL": True,
            "GCS": True,
        }
    )
    mission_status: dict[str, bool] = field(
        default_factory=lambda: {
            "초기접근": True,
            "중기접근": False,
            "최종접근": False,
            "셔터 ON": False,
            "TRACK HOLD": False,
            "TDD 탐지": False,
            "확인 완료": False,
        }
    )
    launch_requested: bool = False
    emergency_mode: bool = False
    _origin: dict[int, tuple[float, float]] = field(default_factory=dict, repr=False)

    @classmethod
    def demo(cls, center_lat: float, center_lon: float) -> "TacticalState":
        now = time.time()
        subjects = [
            SubjectTrack(101, center_lat + 0.0062, center_lon + 0.0080, 3150, 238, 224, now - 94),
            SubjectTrack(204, center_lat - 0.0045, center_lon + 0.0105, 2210, 184, 291, now - 61),
            SubjectTrack(317, center_lat + 0.0090, center_lon - 0.0060, 4020, 266, 142, now - 38),
            SubjectTrack(422, center_lat - 0.0082, center_lon - 0.0075, 1680, 156, 54, now - 19),
        ]
        state = cls(
            subjects=subjects,
            sites=[
                TacticalSite("LC", "이륙지점", center_lat - 0.0023, center_lon - 0.0035),
                TacticalSite("RDR", "레이다", center_lat + 0.0030, center_lon - 0.0045),
                TacticalSite("GCS", "GCS", center_lat, center_lon),
            ],
            selected_track_id=101,
        )
        state._origin = {
            track.track_id: (track.latitude, track.longitude) for track in subjects
        }
        return state

    @property
    def selected_subject(self) -> SubjectTrack | None:
        return next(
            (track for track in self.subjects if track.track_id == self.selected_track_id),
            None,
        )

    @property
    def launch_ready(self) -> bool:
        return all(self.readiness.values()) and not self.emergency_mode

    @property
    def automatic_mode(self) -> str:
        arming_inputs = (
            self.mission_status["셔터 ON"],
            self.mission_status["TRACK HOLD"],
            self.mission_status["TDD 탐지"],
            self.mission_status["확인 완료"],
        )
        return "ARM" if all(arming_inputs) and not self.emergency_mode else "SAFE"

    @property
    def approach_success(self) -> bool:
        return self.launch_requested and all(self.mission_status.values())

    def select_subject(self, track_id: int) -> None:
        if any(track.track_id == track_id for track in self.subjects):
            self.selected_track_id = track_id

    def request_simulated_launch(self) -> bool:
        if not self.launch_ready or self.automatic_mode != "ARM":
            return False
        self.launch_requested = True
        return True

    def toggle_emergency(self) -> bool:
        self.emergency_mode = not self.emergency_mode
        if self.emergency_mode:
            self.launch_requested = False
        return self.emergency_mode

    def tick(self, elapsed_s: float, *, link_connected: bool) -> None:
        """Advance deterministic display-only telemetry."""
        self.readiness["DL"] = link_connected
        phase = elapsed_s % 36.0
        thresholds = {
            "초기접근": 0,
            "중기접근": 4,
            "최종접근": 8,
            "셔터 ON": 12,
            "TRACK HOLD": 15,
            "TDD 탐지": 19,
            "확인 완료": 23,
        }
        for name, threshold in thresholds.items():
            self.mission_status[name] = phase >= threshold
        if phase < 2:
            self.launch_requested = False

        for index, track in enumerate(self.subjects):
            origin_lat, origin_lon = self._origin[track.track_id]
            track.latitude = origin_lat + math.sin(elapsed_s / (8 + index)) * 0.0014
            track.longitude = origin_lon + math.cos(elapsed_s / (9 + index)) * 0.0014
            track.heading_deg = (track.heading_deg + 0.35 + index * 0.07) % 360

