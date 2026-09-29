from __future__ import annotations

import math
import os
import time
import json
from concurrent.futures import Future, ThreadPoolExecutor

from PySide6.QtCore import QEvent, QSignalBlocker, QRectF, QUrl, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPen
# QtWebChannel / QtWebEngine are the heavy GUI add-on modules. They are imported
# lazily inside the widget/bridge methods that actually build a web view, so the
# pure payload helpers in this module (and their tests) import without the
# add-on installed. Type annotations use ``from __future__ import annotations``
# and stay as strings, so they need no runtime import.
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .fly_bridge import FlyMapBridge
from .fly_map_html import (
    ASSET_DIR,
    load_search_camera_html,
    resolve_fly_map_html,
)
from .fly_state import (
    FlyState,
    SAR_CRUISE_SPEED_MPS,
    bearing_deg,
    destination_position,
    hide_preplanned_search_routes,
    horizontal_distance_m,
)
from .map_bridge import MapBridge
from .planning import MODEL_NAME, RuleBasedPlanningEngine, SearchCameraSpec
from .site_store import MissionPoint, SiteStore


GREEN = "#55e77a"
AMBER = "#f3b52d"
RED = "#ff554c"
SUBJECT_YELLOW = "#ffd34f"
BLUE = "#4db3ff"
MUTED = "#869187"
WAYPOINT_CONTEXT_DELETE_RADIUS_M = 2500.0
MANUAL_ROUTE_HOLD_S = 50.0


def json_safe_payload(value):
    """Return a strict-JSON-safe copy of a map telemetry payload.

    Python's JSON encoder writes NaN/Infinity by default, while the browser's
    ``JSON.parse`` rejects them.  A single non-finite planning metric would
    therefore suspend every vehicle repaint until a later valid frame, which
    appears to the operator as a freeze followed by a position jump.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            key: json_safe_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe_payload(item) for item in value]
    return value


def simplify_flight_path(
    points: list[dict],
    *,
    tolerance_m: float = 24.0,
    maximum_points: int = 96,
) -> list[dict]:
    """Keep the drawn track accurate without feeding hundreds of collinear
    vertices into Google Maps 3D on every telemetry frame."""
    if len(points) <= 2:
        return list(points)

    reference_latitude = sum(
        float(point["latitude"]) for point in points
    ) / len(points)
    longitude_scale = 111_320.0 * math.cos(math.radians(reference_latitude))
    projected = [
        (
            float(point["longitude"]) * longitude_scale,
            float(point["latitude"]) * 111_320.0,
        )
        for point in points
    ]
    retained = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start_index, end_index = stack.pop()
        start_x, start_y = projected[start_index]
        end_x, end_y = projected[end_index]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        length_squared = delta_x * delta_x + delta_y * delta_y
        best_distance = -1.0
        best_index = -1
        for index in range(start_index + 1, end_index):
            point_x, point_y = projected[index]
            if length_squared <= 1e-9:
                distance = math.hypot(point_x - start_x, point_y - start_y)
            else:
                fraction = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (point_x - start_x) * delta_x
                            + (point_y - start_y) * delta_y
                        )
                        / length_squared,
                    ),
                )
                distance = math.hypot(
                    point_x - (start_x + delta_x * fraction),
                    point_y - (start_y + delta_y * fraction),
                )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance > tolerance_m:
            retained.add(best_index)
            stack.append((start_index, best_index))
            stack.append((best_index, end_index))

    ordered = sorted(retained)
    if len(ordered) > maximum_points:
        ordered = sorted(
            {
                ordered[
                    round(index * (len(ordered) - 1) / (maximum_points - 1))
                ]
                for index in range(maximum_points)
            }
        )
    return [points[index] for index in ordered]


def build_mission_map_plan_payload(
    live_store: SiteStore,
    pending_store: SiteStore,
    pending_dirty: bool,
) -> dict:
    """Build a visual-only plan with newly added waypoints marked as pending."""
    display_store = pending_store if pending_dirty else live_store
    plan = display_store.render_dict()
    # RHP-SPX: hide the pre-baked SPX search routes on the fly map (the online
    # planner draws each route live).  This is the SHARED-map plan path
    # (setFlyPlan); FlyState.render_dict handles the non-shared path.
    hide_preplanned_search_routes(plan, display_store.mission_metadata)
    plan["pending_edit"] = pending_dirty
    if not pending_dirty:
        return plan

    original_coordinates = {
        vehicle_id: {
            (
                round(point.latitude, 7),
                round(point.longitude, 7),
                round(point.altitude_m, 1),
            )
            for point in live_store.waypoints_for(vehicle_id)
        }
        for vehicle_id in SiteStore.VEHICLE_IDS
    }

    def mark_waypoints(waypoints: list[dict], vehicle_id: int) -> None:
        known_coordinates = original_coordinates.get(vehicle_id, set())
        for waypoint in waypoints:
            coordinate = (
                round(float(waypoint["latitude"]), 7),
                round(float(waypoint["longitude"]), 7),
                round(float(waypoint.get("altitude_m", 0.0)), 1),
            )
            waypoint["pending_preview"] = coordinate not in known_coordinates

    for route in plan.get("vehicle_routes", []):
        mark_waypoints(
            route.get("waypoints", []),
            int(route.get("vehicle_id", 0)),
        )
    mark_waypoints(
        plan.get("waypoints", []),
        int(plan.get("active_vehicle_id", 1)),
    )
    return plan


def waypoint_for_context_key(
    route,
    vehicle_id: int,
    context_key: str | None,
):
    """Resolve the exact waypoint named by a map marker context key."""
    if not context_key or ":" not in context_key:
        return None
    owner, code = context_key.split(":", 1)
    if owner != "FLEET":
        try:
            if int(owner) != int(vehicle_id):
                return None
        except ValueError:
            return None
    if not code.startswith("WP"):
        return None
    return next((point for point in route if point.code == code), None)


def nearest_waypoint_within(
    route,
    latitude: float,
    longitude: float,
    radius_m: float = WAYPOINT_CONTEXT_DELETE_RADIUS_M,
):
    """Return a nearby waypoint without ever selecting a distant route point."""
    if not route:
        return None
    waypoint = min(
        route,
        key=lambda point: horizontal_distance_m(
            latitude,
            longitude,
            point.latitude,
            point.longitude,
        ),
    )
    distance_m = horizontal_distance_m(
        latitude,
        longitude,
        waypoint.latitude,
        waypoint.longitude,
    )
    return waypoint if distance_m <= radius_m else None


def runtime_route_index_for_context_key(
    route: list[dict],
    vehicle_id: int,
    context_key: str | None,
) -> int | None:
    """Resolve a PLAN or live AUTO marker to one exact editable route index."""
    if not context_key:
        return None
    parts = context_key.split(":")
    if len(parts) == 3 and parts[0] == "AUTO":
        try:
            owner = int(parts[1])
            index = int(parts[2])
        except ValueError:
            return None
        return index if owner == int(vehicle_id) and 0 <= index < len(route) else None
    if len(parts) != 2:
        return None
    owner, code = parts
    if owner != "FLEET":
        try:
            if int(owner) != int(vehicle_id):
                return None
        except ValueError:
            return None
    return next(
        (
            index
            for index, point in enumerate(route)
            if str(point.get("code", "")) == code
        ),
        None,
    )


def nearest_runtime_route_index_within(
    route: list[dict],
    latitude: float,
    longitude: float,
    radius_m: float = WAYPOINT_CONTEXT_DELETE_RADIUS_M,
) -> int | None:
    if not route:
        return None
    index = min(
        range(len(route)),
        key=lambda candidate: horizontal_distance_m(
            latitude,
            longitude,
            float(route[candidate]["latitude"]),
            float(route[candidate]["longitude"]),
        ),
    )
    distance_m = horizontal_distance_m(
        latitude,
        longitude,
        float(route[index]["latitude"]),
        float(route[index]["longitude"]),
    )
    return index if distance_m <= radius_m else None


class SearchCameraWidget(QWidget):
    activated = Signal()

    def __init__(self, state: FlyState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PySide6.QtWebEngineCore import QWebEngineSettings
        from PySide6.QtWebEngineWidgets import QWebEngineView

        self.state = state
        self.search_camera_spec = SearchCameraSpec()
        self._map_loaded = False
        self._last_payload = ""
        self.setMinimumSize(390, 220)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.web_view = QWebEngineView(self)
        self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        settings = self.web_view.settings()
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls,
            True,
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled,
            True,
        )
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.WebGLEnabled,
            True,
        )
        self.web_view.installEventFilter(self)
        layout.addWidget(self.web_view)
        self.web_view.loadFinished.connect(self._on_map_loaded)
        self.web_view.setHtml(
            load_search_camera_html(
                os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
            ),
            QUrl.fromLocalFile(str(ASSET_DIR.resolve()) + os.sep),
        )

    def _on_map_loaded(self, ok: bool) -> None:
        self._map_loaded = bool(ok)
        self._last_payload = ""
        self.refresh()

    def _state_payload(self) -> dict:
        vehicle = self.state.vehicle
        subject = self.state.selected_subject
        subject_latitude = subject.latitude if subject is not None else vehicle.latitude
        subject_longitude = (
            subject.longitude if subject is not None else vehicle.longitude
        )
        subject_distance_m = (
            horizontal_distance_m(
                vehicle.latitude,
                vehicle.longitude,
                subject_latitude,
                subject_longitude,
            )
            if subject is not None
            else float("inf")
        )
        altitude_separation_m = abs(
            float(vehicle.altitude_m)
            - float(subject.altitude_m if subject is not None else 0.0)
        )
        slant_distance_m = math.hypot(
            subject_distance_m,
            altitude_separation_m,
        )
        off_nadir_deg = math.degrees(
            math.atan2(
                subject_distance_m,
                max(1.0, altitude_separation_m),
            )
        )
        coverage_m = self.search_camera_spec.gimbal_centerline_envelope_m
        subject_visible = bool(
            subject is not None
            and subject_distance_m <= coverage_m / 2.0
            and off_nadir_deg <= self.search_camera_spec.max_gimbal_angle_deg
        )
        return {
            "vehicle": {
                "latitude": vehicle.latitude,
                "longitude": vehicle.longitude,
                "heading_deg": vehicle.heading_deg,
                "altitude_m": vehicle.altitude_m,
            },
            "subject": {
                "latitude": subject_latitude,
                "longitude": subject_longitude,
                "code": subject.code if subject is not None else "NO SUBJECT",
                "altitude_m": (
                    subject.altitude_m if subject is not None else 0.0
                ),
            },
            "subject_visible": subject_visible,
            "subject_distance_m": subject_distance_m,
            "horizontal_distance_m": subject_distance_m,
            "altitude_separation_m": altitude_separation_m,
            "slant_distance_m": slant_distance_m,
            "off_nadir_deg": off_nadir_deg,
            "coverage_m": coverage_m,
            "phase": self.state.flight_phase,
            "camera_mode": self.state.camera_mode,
            "held": bool(self.state.mission_status["TRACK HOLD"]),
            "found": bool(self.state.subject_found),
        }

    def refresh(self) -> None:
        if not self._map_loaded:
            return
        payload = json.dumps(
            self._state_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if payload == self._last_payload:
            return
        self._last_payload = payload
        self.web_view.page().runJavaScript(
            f"window.updateSearchCamera && window.updateSearchCamera({payload});"
        )

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if (
            watched is self.web_view
            and event.type() == QEvent.Type.MouseButtonDblClick
        ):
            self.activated.emit()
            return True
        return super().eventFilter(watched, event)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        self.refresh()
        return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#050b07"))
        width = self.width()
        height = self.height()
        phase = time.monotonic()

        for index in range(34):
            y = int((index * 31 + phase * 19) % max(1, height))
            tone = 20 + (index * 5) % 28
            painter.setPen(QPen(QColor(tone, tone + 12, tone + 2, 175), 1))
            painter.drawLine(0, y, width, y)

        painter.setPen(QPen(QColor("#31513b"), 1, Qt.PenStyle.DashLine))
        center_x = width / 2
        center_y = height / 2
        for radius in (38, 72, 106):
            painter.drawEllipse(QRectF(
                center_x - radius,
                center_y - radius,
                radius * 2,
                radius * 2,
            ))

        painter.setPen(QPen(QColor("#77b889"), 1))
        painter.drawLine(int(center_x), 14, int(center_x), height - 14)
        painter.drawLine(14, int(center_y), width - 14, int(center_y))
        painter.drawLine(int(center_x - 14), int(center_y), int(center_x + 14), int(center_y))
        painter.drawLine(int(center_x), int(center_y - 14), int(center_x), int(center_y + 14))

        subject = self.state.selected_subject
        camera_mode = self.state.camera_mode
        held = self.state.mission_status["TRACK HOLD"] and subject is not None
        if camera_mode == "SEARCH":
            sweep_angle = (phase * 0.9) % (math.pi * 2.0)
            sweep_radius = min(width, height) * 0.38
            painter.setPen(QPen(QColor("#55e77a"), 2))
            painter.drawLine(
                int(center_x),
                int(center_y),
                int(center_x + math.sin(sweep_angle) * sweep_radius),
                int(center_y - math.cos(sweep_angle) * sweep_radius),
            )
        offset_x = 0.0 if held else math.sin(phase * 0.65) * 56
        offset_y = 0.0 if held else math.cos(phase * 0.52) * 34
        if (
            subject is not None
            and self.state.subject_detected
            and self.state.flight_phase
            in {
                "DETECTION_TRANSIT",
                "INITIAL_APPROACH",
                "MIDCOURSE_APPROACH",
                "FINAL_APPROACH",
                "FOUND",
            }
        ):
            final_phase = self.state.flight_phase in {
                "FINAL_APPROACH",
                "FOUND",
            }
            acquiring = self.state.flight_phase in {
                "DETECTION_TRANSIT",
                "INITIAL_APPROACH",
            }
            box_color = QColor(
                RED if final_phase else (AMBER if acquiring else SUBJECT_YELLOW)
            )
            subject_box = QRectF(
                center_x + offset_x - 34,
                center_y + offset_y - 25,
                68,
                50,
            )
            painter.setPen(QPen(box_color, 2))
            painter.drawRect(subject_box)
            painter.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.Bold))
            painter.drawText(
                QRectF(
                    subject_box.left(),
                    subject_box.top() - 18,
                    150,
                    16,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                (
                    "SUBJECT / FINAL"
                    if final_phase
                    else (
                        "SUBJECT / SCANNING"
                        if acquiring
                        else "SUBJECT / CLOSING"
                    )
                ),
            )

        painter.setFont(QFont("IBM Plex Mono", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor(GREEN if held else RED))
        painter.drawText(10, 18, "HOLD" if held else "NO HOLD")
        mode_color = (
            RED
            if camera_mode in {"AUTO HOLD", "SHUT DOWN"}
            else (SUBJECT_YELLOW if camera_mode.startswith("AUTO") else GREEN)
        )
        painter.setPen(QColor(mode_color))
        painter.drawText(
            QRectF(0, 7, width - 10, 18),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            camera_mode,
        )
        painter.setPen(QColor("#9cc5a6"))
        subject_text = (
            f"{subject.code}  AZ {subject.heading_deg:03.0f}°  "
            f"ALT {subject.altitude_m:.0f}M"
            if subject is not None and self.state.subject_detected
            else "AUTO SEARCH / NO DETECTION"
        )
        painter.drawText(10, height - 10, subject_text)
        if self.state.subject_found:
            painter.setPen(QPen(QColor(RED), 5))
            cross_size = min(width, height) * 0.18
            painter.drawLine(
                int(center_x - cross_size),
                int(center_y - cross_size),
                int(center_x + cross_size),
                int(center_y + cross_size),
            )
            painter.drawLine(
                int(center_x - cross_size),
                int(center_y + cross_size),
                int(center_x + cross_size),
                int(center_y - cross_size),
            )
            painter.setFont(
                QFont("IBM Plex Mono", 15, QFont.Weight.Bold)
            )
            painter.drawText(
                QRectF(0, center_y + cross_size + 8, width, 30),
                Qt.AlignmentFlag.AlignCenter,
                "SHUT DOWN",
            )
        painter.end()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.activated.emit()
        event.accept()


class FlyMapStage(QWidget):
    def __init__(
        self,
        web_view: QWebEngineView | None,
        state: FlyState,
        search_camera_spec: SearchCameraSpec | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.web_view = web_view
        self.map_layout = QVBoxLayout(self)
        self.map_layout.setContentsMargins(0, 0, 0, 0)
        if web_view is not None:
            self.map_layout.addWidget(web_view)

        self.search_camera_panel = QFrame(self)
        self.search_camera_panel.setObjectName("searchCameraPanel")
        self.search_camera_panel.setStyleSheet(
            """
            QFrame#searchCameraPanel {
                background: #050906;
                border-top: 1px solid #69766a;
                border-left: 1px solid #59665a;
                border-right: 3px solid #020403;
                border-bottom: 3px solid #020403;
            }
            """
        )
        self.search_camera_panel.setFixedSize(440, 280)
        camera_layout = QVBoxLayout(self.search_camera_panel)
        camera_layout.setContentsMargins(6, 5, 6, 6)
        camera_layout.setSpacing(4)
        header = QHBoxLayout()
        title = QLabel("SEARCH CAMERA // EO")
        title.setObjectName("fieldCaption")
        self.hold_label = QLabel("● NO HOLD")
        self.hold_label.setObjectName("dataValue")
        header.addWidget(title)
        header.addStretch(1)
        self.split_button = QPushButton("탐지 카메라 (6분할)")
        self.split_button.setMaximumHeight(26)
        self.split_button.setStyleSheet("padding:2px 6px; font-size:8pt;")
        header.addWidget(self.split_button)
        header.addWidget(self.hold_label)
        camera_layout.addLayout(header)
        self.search_camera_video = SearchCameraWidget(state)
        camera_layout.addWidget(self.search_camera_video, 1)
        search_camera_spec = search_camera_spec or SearchCameraSpec()
        specification = QLabel(
            "AUTO // "
            f"ALT {search_camera_spec.altitude_m:.0f}m · "
            f"Af {search_camera_spec.horizontal_fov_deg:.0f}° · "
            f"Ag {search_camera_spec.max_gimbal_angle_deg:.0f}° · "
            f"COVER {search_camera_spec.gimbal_centerline_envelope_m:.0f}m · "
            f"CELL {search_camera_spec.nominal_cell_m:.0f}m/"
            f"{search_camera_spec.nominal_cell_search_time_s:.1f}s · "
            f"AREA {search_camera_spec.nominal_search_area_km2:.1f}km²/"
            f"{search_camera_spec.nominal_search_time_min:.0f}min"
        )
        specification.setObjectName("mutedText")
        specification.setStyleSheet("font-size:7pt;")
        camera_layout.addWidget(specification)
        self.search_camera_panel.raise_()
        self.mission_modify_button = QPushButton("임무지도 수정", self)
        self.mission_modify_button.setObjectName("primaryButton")
        self.mission_modify_button.setFixedSize(110, 28)
        self.mission_modify_button.raise_()

    def attach_map_view(self, web_view: QWebEngineView) -> None:
        self.web_view = web_view
        self.map_layout.addWidget(web_view)
        web_view.show()
        self.search_camera_panel.raise_()

    def refresh_camera(self) -> None:
        held = self.search_camera_video.state.mission_status["TRACK HOLD"]
        found = self.search_camera_video.state.subject_found
        camera_mode = self.search_camera_video.state.camera_mode
        self.hold_label.setText(
            "SHUT DOWN"
            if found
            else ("● HOLD" if held else f"● {camera_mode}")
        )
        self.hold_label.setStyleSheet(
            f"color: {RED if found else (GREEN if held else RED)};"
        )
        self.search_camera_video.refresh()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        margin = 16
        self.search_camera_panel.move(
            max(margin, self.width() - self.search_camera_panel.width() - margin),
            max(margin, self.height() - self.search_camera_panel.height() - margin),
        )
        self.mission_modify_button.move(
            max(margin, self.width() - self.mission_modify_button.width() - margin),
            64,
        )
        self.search_camera_panel.raise_()
        self.mission_modify_button.raise_()


class Fly3DView(QWidget):
    statusMessage = Signal(str)
    vehicleSelectionRequested = Signal(int)

    def __init__(
        self,
        store: SiteStore,
        parent: QWidget | None = None,
        *,
        shared_web_view: QWebEngineView | None = None,
        shared_bridge: MapBridge | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self._shared_web_view = shared_web_view
        self._shared_bridge = shared_bridge
        self._uses_shared_map = (
            shared_web_view is not None and shared_bridge is not None
        )
        center = store.sites.get("GCS")
        center_lat = center.latitude if center else 37.3422
        center_lon = center.longitude if center else 127.9202
        self.states = {
            vehicle_id: FlyState.demo(center_lat, center_lon)
            for vehicle_id in SiteStore.VEHICLE_IDS
        }
        self.selected_vehicle_id = 0
        self.state = self.states[1]
        self.search_camera_spec = self._build_search_camera_spec()
        self.planning_engine = self._new_planning_engine(center_lat, center_lon)
        # v0.5: 탐색 경로는 이륙 전 Stone-SPX 사전계획(PLAN 화면 "인증 탐색계획
        # 생성")이 만든 웨이포인트가 권위다. 비행 중 RHP 라이브 재계획이 그 경로를
        # 덮어쓰지 않도록 route_updates 적용을 끈다. 탐지·접근(approach)·협동 구조 접근은
        # 그대로 유지된다. RHP 를 다시 쓰려면 이 플래그를 False 로.
        self._spx_preplanned = True
        self._ingress_enabled = bool(
            self.store.mission_metadata.get("ingress_sweep", False)
        )
        self._ingress_prepended = False
        self._ingress_tp_transition_done = False
        self._planning_mode = str(
            (self.store.mission_metadata or {}).get("planning_mode", "")
        )
        self._arc_patrol_active = False
        self._arc_patrol_done = False
        self._arc_patrol_start_s = 0.0
        self._arc_patrol_duration_s = float(
            (self.store.mission_metadata or {}).get(
                "arc_patrol_duration_s", 75.0
            )
        )
        self._rhp_spx_search_active = False
        self._continuous_search_active = False
        self._found_track_ids: set[int] = set()
        self._phase2_research_ids: set[int] = set()
        self._resume_search_triggered = False
        self._spx_replan_pending = False
        self._spx_replan_reason = ""
        self._spx_replan_exclude_track_ids: set[int] = set()
        self._spx_replan_future: Future | None = None
        self._spx_replan_generation = 0
        baked_spx = (self.store.mission_metadata or {}).get("spx_certified", {})
        self._spx_detection_probability: float | None = (
            baked_spx.get("detection_probability") if baked_spx else None
        )
        self._planning_payload: dict = {
            "model": MODEL_NAME,
            "status": "INITIALIZING",
            "sensor": self.search_camera_spec.display_dict(),
        }
        self._planning_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gcs-rhp-planner",
        )
        self._spx_replan_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="gcs-spx-replan",
        )
        self._planning_future: Future | None = None
        self._planning_future_generation = 0
        self._planning_future_elapsed_s = 0.0
        self._planning_future_initial_precompute = False
        self._planning_generation = 0
        self._initial_rhp_precompute_requested = False
        self._last_planning_cycle_s = -1e9
        self._last_planning_search_elapsed_s = -1e9
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        self.map_html, self.provider_name = resolve_fly_map_html(
            self.api_key,
            uses_shared_map=self._uses_shared_map,
        )
        self._map_loaded = False
        self._has_activated = False
        self._active = False
        self._refreshing_table = False
        self._last_map_track_click: tuple[int, float] | None = None
        self._reported_shutdown = False
        self._detection_dialog: QMessageBox | None = None
        self._pending_mission = SiteStore()
        self._pending_dirty = False
        self._pending_runtime_routes: dict[int, list[dict]] = {}
        self._pending_runtime_originals: dict[int, list[dict]] = {}
        self._manual_route_applied_vehicle_ids: set[int] = set()
        self._plan_render_revision = 0
        self._cached_plan_key: tuple[int, bool] | None = None
        self._cached_plan_payload: dict | None = None
        self._last_sent_plan_key: tuple[int, bool] | None = None
        self._flight_path_payload_cache: dict[
            int,
            tuple[tuple, list[dict]],
        ] = {}
        self._preserve_execution_on_next_plan_change = False
        self._pending_context_position: tuple[float, float, float, float] | None = None
        self._pending_context_feature: tuple[str, float] | None = None
        self._context_menu_block_until = 0.0
        self._split_active_vehicle_id = 1
        self._last_console_refresh_at = 0.0
        self._last_camera_refresh_at = 0.0
        self._last_map_emit_at = 0.0

        self._build_ui()
        self._connect_map()
        self.store.subscribe(self._on_plan_changed)
        for vehicle_id, state in self.states.items():
            state.load_mission(self.store, vehicle_id)
        self._refresh_display()

        self.update_timer = QTimer(self)
        self.update_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.update_timer.timeout.connect(self._tick)
        self.update_timer.setInterval(50)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._build_console())

        if self._shared_web_view is None:
            from PySide6.QtWebEngineCore import QWebEngineSettings
            from PySide6.QtWebEngineWidgets import QWebEngineView

            self.web_view = QWebEngineView()
            self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            settings = self.web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled,
                True,
            )
        else:
            self.web_view = self._shared_web_view
        self.map_stage = FlyMapStage(
            None if self._uses_shared_map else self.web_view,
            self.state,
            self.search_camera_spec,
        )
        self.map_stage.split_button.clicked.connect(self._show_split_seekers)
        self.map_stage.mission_modify_button.clicked.connect(
            self._apply_pending_mission
        )
        self.map_stage.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        layout.addWidget(self.map_stage, 1)

    def _build_console(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("flyConsole")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(410)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content.setObjectName("toolPanel")
        panel = QVBoxLayout(content)
        panel.setContentsMargins(9, 9, 9, 12)
        panel.setSpacing(6)

        banner = QFrame()
        banner.setStyleSheet(
            "background:#17231a; border:1px solid #435346;"
        )
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(9, 6, 9, 6)
        title = QLabel("임무 지도 // MISSION MAP")
        title.setObjectName("panelTitle")
        title.setStyleSheet("border:0; padding:0;")
        self.source_label = QLabel("LOCAL DATA")
        self.source_label.setObjectName("dataValue")
        banner_layout.addWidget(title)
        banner_layout.addStretch(1)
        banner_layout.addWidget(self.source_label)
        panel.addWidget(banner)

        panel.addWidget(self._build_subject_group())
        panel.addWidget(self._build_information_group())
        panel.addWidget(self._build_readiness_group())
        panel.addWidget(self._build_mission_group())
        panel.addWidget(self._build_control_group())
        panel.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_subject_group(self) -> QWidget:
        group = QGroupBox("탐지 대상 / TARGETS")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(7, 12, 7, 7)
        layout.setSpacing(4)
        self.subject_table = QTableWidget(0, 4)
        self.subject_table.setHorizontalHeaderLabels(
            ("NO/TYPE", "속도", "방향", "최초 추적")
        )
        self.subject_table.setAlternatingRowColors(True)
        self.subject_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.subject_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.subject_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.subject_table.verticalHeader().setVisible(False)
        header = self.subject_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.subject_table.setFixedHeight(120)
        self.subject_table.cellDoubleClicked.connect(self._select_subject_from_table)
        layout.addWidget(self.subject_table)
        return group

    def _build_information_group(self) -> QWidget:
        group = QGroupBox("UAV / 선택 대상 정보")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(7, 12, 7, 7)
        layout.setSpacing(6)
        (
            vehicle_frame,
            self.vehicle_title_label,
            self.vehicle_values,
        ) = self._information_column(
            "UAV-01",
            BLUE,
        )
        subject_frame, _subject_title, self.subject_values = self._information_column(
            "SUBJECT",
            RED,
        )
        layout.addWidget(vehicle_frame, 1)
        layout.addWidget(subject_frame, 1)
        return group

    @staticmethod
    def _information_column(
        title: str,
        color: str,
    ) -> tuple[QWidget, QLabel, dict[str, QLabel]]:
        frame = QFrame()
        frame.setStyleSheet("background:#09100c; border:1px solid #2e3930;")
        layout = QGridLayout(frame)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)
        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color:{color}; font-family:'IBM Plex Mono'; font-weight:700; border:0;"
        )
        layout.addWidget(title_label, 0, 0, 1, 2)
        values: dict[str, QLabel] = {}
        for row, (caption, key) in enumerate(
            (
                ("고도", "altitude"),
                ("위도", "latitude"),
                ("경도", "longitude"),
                ("속도", "speed"),
                ("방위각", "heading"),
                ("거리", "distance"),
                ("도달시간", "eta"),
            ),
            start=1,
        ):
            caption_label = QLabel(caption)
            caption_label.setObjectName("mutedText")
            caption_label.setStyleSheet("border:0;")
            value_label = QLabel("--")
            value_label.setObjectName("dataValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setStyleSheet("border:0;")
            layout.addWidget(caption_label, row, 0)
            layout.addWidget(value_label, row, 1)
            values[key] = value_label
        return frame, title_label, values

    def _build_readiness_group(self) -> QWidget:
        group = QGroupBox("이륙 준비 상태 / READINESS")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(7, 12, 7, 7)
        lamp_row = QHBoxLayout()
        lamp_row.setSpacing(3)
        self.readiness_lamps: dict[str, QLabel] = {}
        for name in ("AVS", "LC", "RDR", "DL", "GCS"):
            lamp, label = self._lamp_cell(name)
            self.readiness_lamps[name] = label
            lamp_row.addWidget(lamp, 1)
        layout.addLayout(lamp_row)
        self.launch_ready_label = QLabel("● 이륙 불가")
        self.launch_ready_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.launch_ready_label.setStyleSheet(
            f"color:{RED}; background:#120a08; border:1px solid #5f2824; padding:5px;"
        )
        layout.addWidget(self.launch_ready_label)
        return group

    def _build_mission_group(self) -> QWidget:
        group = QGroupBox("임무 진행 상태 / MISSION")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(7, 12, 7, 7)
        layout.setSpacing(3)
        self.mission_lamps: dict[str, QLabel] = {}

        status_names = list(self.state.mission_status)
        guidance_row = QHBoxLayout()
        guidance_row.setSpacing(3)
        for name in status_names[:3]:
            lamp, label = self._lamp_cell(name)
            self.mission_lamps[name] = label
            guidance_row.addWidget(lamp, 1)
        layout.addLayout(guidance_row)

        terminal_row = QHBoxLayout()
        terminal_row.setSpacing(3)
        for name in status_names[3:]:
            display_name = "근접센서 탐지" if name == "TDD 탐지" else name
            lamp, label = self._lamp_cell(display_name)
            self.mission_lamps[name] = label
            terminal_row.addWidget(lamp, 1)
        layout.addLayout(terminal_row)

        return group

    @staticmethod
    def _lamp_cell(name: str) -> tuple[QWidget, QLabel]:
        cell = QFrame()
        cell.setStyleSheet("background:#090f0b; border:1px solid #2d382f;")
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(3, 2, 3, 3)
        layout.setSpacing(0)
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet(
            "color:#8b968c; font-family:'IBM Plex Mono'; font-size:8pt; border:0;"
        )
        lamp = QLabel("●")
        lamp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lamp.setStyleSheet(f"color:{RED}; font-size:14pt; border:0;")
        layout.addWidget(name_label)
        layout.addWidget(lamp)
        return cell, lamp

    def _build_control_group(self) -> QWidget:
        group = QGroupBox("운용 모드 / CONTROLS")
        layout = QGridLayout(group)
        layout.setContentsMargins(7, 12, 7, 7)
        layout.setSpacing(5)
        caption = QLabel("◈ AUTO MODE")
        caption.setObjectName("fieldCaption")
        self.mode_label = QLabel("SAFE")
        self.mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_label.setStyleSheet(
            f"color:{AMBER}; background:#251c0d; border:1px solid #715822; padding:7px;"
        )
        self.launch_button = QPushButton("이륙")
        self.launch_button.setStyleSheet(
            f"background:#12351c; color:{GREEN}; border:1px solid #2e7640;"
            "font-weight:700; padding:7px;"
        )
        self.launch_button.clicked.connect(self._request_launch)
        self.emergency_button = QPushButton("비상모드")
        self.emergency_button.clicked.connect(self._toggle_emergency)
        layout.addWidget(caption, 0, 0)
        layout.addWidget(self.mode_label, 0, 1)
        layout.addWidget(self.launch_button, 1, 0)
        layout.addWidget(self.emergency_button, 1, 1)
        return group

    def _connect_map(self) -> None:
        if self._uses_shared_map:
            assert self._shared_bridge is not None
            self.map_bridge = self._shared_bridge
            self.map_bridge.featureSelected.connect(
                self._on_shared_feature_selected
            )
            self.map_bridge.mapRightClicked.connect(
                self._on_map_right_clicked
            )
            self.map_bridge.featureRightClicked.connect(
                self._on_waypoint_right_clicked
            )
            self._map_loaded = True
            return
        self.map_bridge = FlyMapBridge(self.store, self.state)
        self.map_bridge.spx_pd = self._spx_detection_probability
        self.map_bridge.subjectSelected.connect(self._select_subject)
        self.map_bridge.mapRightClicked.connect(self._on_map_right_clicked)
        self.map_bridge.featureRightClicked.connect(
            self._on_waypoint_right_clicked
        )
        self.map_bridge.mapStatusChanged.connect(self.statusMessage)
        from PySide6.QtWebChannel import QWebChannel

        self.web_channel = QWebChannel(self.web_view.page())
        self.web_channel.registerObject("bridge", self.map_bridge)
        self.web_view.page().setWebChannel(self.web_channel)
        self.web_view.loadFinished.connect(self._on_map_loaded)

    def _build_search_camera_spec(self) -> SearchCameraSpec:
        """Detection sensor spec, with optional wide-FOV overrides from mission
        metadata.  The default narrow scanning gimbal FOV (~195 m instantaneous
        footprint, ±45deg) misses a fast-drifting subject (19 km/h child) that
        has moved off the swept track by the time a vehicle passes.  A mission
        may widen the DETECTION footprint via ``search_camera_fov_deg`` /
        ``search_camera_gimbal_deg`` / ``search_camera_sweep_width_m`` so a pass
        within the wider corridor detects the moving subject.  This changes only
        the detection/belief footprint — the baked SPX routes come from the
        separate ``SensorSpec.sr_z50`` and are unaffected (no re-bake needed).
        """
        meta = self.store.mission_metadata or {}
        kwargs: dict = {}
        fov = meta.get("search_camera_fov_deg")
        gimbal = meta.get("search_camera_gimbal_deg")
        sweep = meta.get("search_camera_sweep_width_m")
        if fov is not None:
            kwargs["horizontal_fov_deg"] = float(fov)
        if gimbal is not None:
            kwargs["max_gimbal_angle_deg"] = float(gimbal)
        if sweep is not None:
            kwargs["effective_sweep_width_m"] = float(sweep)
        if meta.get("search_camera_detect_within_reach") is not None:
            kwargs["detect_within_reach"] = bool(
                meta.get("search_camera_detect_within_reach")
            )
        if meta.get("search_camera_detect_reach_m") is not None:
            kwargs["detect_reach_m"] = float(
                meta.get("search_camera_detect_reach_m")
            )
        try:
            return SearchCameraSpec(**kwargs)
        except (ValueError, TypeError):
            return SearchCameraSpec()

    def _new_planning_engine(
        self,
        center_latitude: float,
        center_longitude: float,
    ) -> RuleBasedPlanningEngine:
        metadata = self.store.mission_metadata
        raw_arc = metadata.get("arc_search_pattern", {})
        arc = raw_arc if isinstance(raw_arc, dict) else {}
        route = self.store.waypoints_for(1)
        # Anchor the search circle on the predicted subject point (TP), not on
        # the first route waypoint. With arc patterns route[0] is the TP rally,
        # but with an SPX-baked plan route[0] is an off-centre search leg, which
        # would drag the boundary ring off the search area.
        tp_center = metadata.get("rally_predicted_subject")
        if not isinstance(tp_center, dict):
            tp_center = arc.get("center") if isinstance(arc.get("center"), dict) else {}
        search_center_latitude = float(
            tp_center.get(
                "latitude",
                route[0].latitude if route else center_latitude,
            )
        )
        search_center_longitude = float(
            tp_center.get(
                "longitude",
                route[0].longitude if route else center_longitude,
            )
        )
        return RuleBasedPlanningEngine(
            center_latitude,
            center_longitude,
            search_camera=self.search_camera_spec,
            particle_count=max(120, int(arc.get("particle_count", 3_000))),
            rhp_decision_interval_s=max(
                1.0,
                float(arc.get("decision_interval_s", 25.0)),
            ),
            rhp_encounter_sample_count=max(
                16,
                int(arc.get("encounter_sample_count", 64)),
            ),
            rhp_radial_shortlist_count=max(
                1,
                int(arc.get("radial_shortlist_count", 5)),
            ),
            search_radius_m=max(
                self.search_camera_spec.track_spacing_m,
                float(metadata.get("search_radius_m", 3_333.3333333333)),
            ),
            minimum_route_updates_before_detection=max(
                0,
                int(
                    metadata.get(
                        "visualization_min_rhp_updates_before_atr",
                        0,
                    )
                ),
            ),
            search_center_latitude=search_center_latitude,
            search_center_longitude=search_center_longitude,
        )

    def activate(self) -> None:
        if not self._has_activated:
            for vehicle_id, state in self.states.items():
                state.load_mission(self.store, vehicle_id)
            self._has_activated = True
        # ``_refresh_display`` can emit a state while the shared WebView is
        # still displaying PLAN.  That payload is intentionally ignored by
        # the JavaScript FLY renderer, so it must not count as the one plan
        # snapshot sent for this revision.  Force a full Mission Map plan on
        # every activation; otherwise TP and the per-SAR waypoint layer can be
        # absent until the operator edits the mission again.
        self._last_sent_plan_key = None
        if self._uses_shared_map:
            # The FLY view now owns the shared map: hide the pre-baked SPX
            # routes so a store-change plan notification cannot repaint them.
            if self.map_bridge is not None:
                self.map_bridge.hide_preplanned_routes = True
            self.map_stage.attach_map_view(self.web_view)
            self.web_view.page().runJavaScript(
                "window.setMapMode && window.setMapMode('FLY');"
            )
        elif not self._map_loaded:
            base_url = QUrl.fromLocalFile(str(ASSET_DIR.resolve()) + os.sep)
            self.web_view.setHtml(self.map_html, base_url)
            self._map_loaded = True
        self._active = True
        if not self.update_timer.isActive():
            self.update_timer.start()
        self._tick()
        self.statusMessage.emit(
            f"MISSION MAP READY // {self.provider_name} // UI 내부 모의 데이터"
        )

    def deactivate(self) -> None:
        self._active = False
        self.update_timer.stop()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.update_timer.stop()
        self._planning_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        self._spx_replan_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        super().closeEvent(event)

    def _on_map_loaded(self, ok: bool) -> None:
        if ok:
            self._last_sent_plan_key = None
            self.map_bridge.emit_state()
        else:
            self.statusMessage.emit("MISSION MAP 지도를 불러오지 못했습니다.")

    def _on_plan_changed(self) -> None:
        self._plan_render_revision += 1
        self._cached_plan_key = None
        self._cached_plan_payload = None
        self._last_sent_plan_key = None
        preserve_execution = self._preserve_execution_on_next_plan_change
        self._preserve_execution_on_next_plan_change = False
        manually_applied = set(self._manual_route_applied_vehicle_ids)
        self._manual_route_applied_vehicle_ids.clear()
        for vehicle_id, state in self.states.items():
            if preserve_execution and manually_applied:
                # Edited routes have already been applied through
                # ``apply_manual_runtime_route`` and every other SAR must keep
                # its own current RHP path.  Refresh readiness only; replacing
                # any state from the store here would collapse live routes
                # back onto their original PLAN definitions.
                state.sync_plan_readiness(self.store, vehicle_id)
            elif preserve_execution and state.mission_launched:
                state.update_route_from_store(self.store, vehicle_id)
            else:
                state.load_mission(self.store, vehicle_id)
        if not preserve_execution:
            self._planning_generation += 1
            self._planning_future = None
            center = self.store.sites.get("GCS")
            center_latitude = (
                center.latitude if center is not None else self.state.center_latitude
            )
            center_longitude = (
                center.longitude if center is not None else self.state.center_longitude
            )
            # Refresh the detection sensor spec from the newly loaded mission so
            # wide-FOV / detect_within_reach overrides take effect (the __init__
            # spec was built before any mission metadata existed).
            self.search_camera_spec = self._build_search_camera_spec()
            self.planning_engine = self._new_planning_engine(
                center_latitude,
                center_longitude,
            )
            self._initial_rhp_precompute_requested = False
            self._planning_future_initial_precompute = False
            self._last_planning_cycle_s = -1e9
            self._last_planning_search_elapsed_s = -1e9
            self._planning_payload = {
                "model": MODEL_NAME,
                "status": "MISSION RESET",
                "sensor": self.search_camera_spec.display_dict(),
            }
            self._ingress_enabled = bool(
                self.store.mission_metadata.get("ingress_sweep", False)
            )
            self._ingress_prepended = False
            self._ingress_tp_transition_done = False
            self._planning_mode = str(
                (self.store.mission_metadata or {}).get("planning_mode", "")
            )
            self._arc_patrol_active = False
            self._arc_patrol_done = False
            self._rhp_spx_search_active = False
            self._continuous_search_active = False
            self._found_track_ids = set()
            self._phase2_research_ids = set()
            self._resume_search_triggered = False
            self._spx_replan_pending = False
            self._spx_replan_reason = ""
            self._spx_replan_exclude_track_ids = set()
            if self._spx_replan_future is not None:
                self._spx_replan_future.cancel()
                self._spx_replan_future = None
            self._spx_replan_generation += 1
        self._refresh_display()
        if self._map_loaded:
            self._emit_map_state()

    def _run_planning_cycle(self, *, force: bool = False) -> None:
        if force:
            self._collect_planning_result(wait=True)
        elif self._planning_future is not None:
            return

        wall_elapsed_s = self.states[1].elapsed_s
        canonical_state = self.states[1]
        approach_track_id = next(
            (
                state.selected_track_id
                for state in self.states.values()
                if state.approach_requested and not state.subject_found
            ),
            None,
        )
        # "Fleet launched" = at least one vehicle is airborne.  all() wrongly
        # counted the five never-launched slots of a single-UAV mission as
        # blocking, so its t=0 planner preview -- and thus the two reachability
        # rings -- never ran during ingress (they only appeared after TP).
        # Launches are simultaneous, so any()==all() for a full fleet.
        fleet_launched = any(
            state.mission_launched for state in self.states.values()
        )
        search_active = any(
            state.search_started for state in self.states.values()
        )
        initial_prefix_precompute = bool(
            approach_track_id is None
            and fleet_launched
            and not search_active
            and not self._initial_rhp_precompute_requested
        )
        if (
            approach_track_id is None
            and not search_active
            and not initial_prefix_precompute
        ):
            # Before TP, the sole permitted planner operation is one t=0
            # initial-prefix cache.  Ingress must not advance the PF, create
            # camera observations, or consume the 25-second RHP clock.
            return

        if not force:
            if search_active:
                if (
                    canonical_state.search_elapsed_s
                    - self._last_planning_search_elapsed_s
                    < 1.0
                ):
                    return
            elif wall_elapsed_s - self._last_planning_cycle_s < 1.0:
                return

        self._last_planning_cycle_s = wall_elapsed_s
        planning_elapsed_s = (
            0.0
            if initial_prefix_precompute
            else canonical_state.search_elapsed_s
        )
        if initial_prefix_precompute:
            self._initial_rhp_precompute_requested = True
        elif search_active:
            self._last_planning_search_elapsed_s = planning_elapsed_s
        vehicles = []
        for vehicle_id, state in self.states.items():
            route = self.store.waypoints_for(vehicle_id)
            rally = route[0] if route else None
            if initial_prefix_precompute and rally is None:
                # A single-/few-UAV mission leaves the other vehicle slots without
                # a route; skip those inactive slots rather than aborting the
                # whole t=0 preface (which suppressed the reachability rings
                # during a single-UAV ingress).
                if not state.mission_launched:
                    continue
                self._initial_rhp_precompute_requested = False
                return
            vehicles.append(
                {
                "vehicle_id": vehicle_id,
                # CPP defines search t=0 with all six SAR physically at TP.
                # This synthetic position is used only for a side-effect-free
                # first-prefix cache; the live vehicle remains on ingress.
                "latitude": (
                    rally.latitude
                    if initial_prefix_precompute and rally is not None
                    else state.vehicle.latitude
                ),
                "longitude": (
                    rally.longitude
                    if initial_prefix_precompute and rally is not None
                    else state.vehicle.longitude
                ),
                "altitude_m": (
                    rally.altitude_m
                    if initial_prefix_precompute and rally is not None
                    else state.vehicle.altitude_m
                ),
                "speed_mps": state.vehicle.speed_mps,
                "heading_deg": state.vehicle.heading_deg,
                "mission_launched": state.mission_launched,
                "emergency_mode": state.emergency_mode,
                "flight_phase": state.flight_phase,
                "completed_route_segment_count": (
                    state.completed_route_segment_count
                ),
                "search_started": (
                    state.search_started or initial_prefix_precompute
                ),
                "rhp_preview_only": initial_prefix_precompute,
                "runtime_route_revision": state.runtime_route_revision,
                "runtime_route_update_count": (
                    state.runtime_route_update_count
                ),
                "ingress_sweep": state.ingress_sweep_active and self._ingress_enabled,
                }
            )
        c4i_center = self.store.mission_metadata.get(
            "rally_predicted_subject",
            {},
        )
        subjects = [
            {
                "track_id": track.track_id,
                "latitude": track.latitude,
                "longitude": track.longitude,
                "altitude_m": track.altitude_m,
                "speed_mps": track.speed_mps,
                "heading_deg": track.heading_deg,
                "position_uncertainty_m": track.position_uncertainty_m,
                "found": track.found,
                "measurement_latitude": (
                    track.latitude
                    if approach_track_id == track.track_id
                    else float(c4i_center.get("latitude", track.latitude))
                ),
                "measurement_longitude": (
                    track.longitude
                    if approach_track_id == track.track_id
                    else float(c4i_center.get("longitude", track.longitude))
                ),
                "estimator_speed_mps": (
                    track.speed_mps
                    if approach_track_id == track.track_id
                    else 0.0
                ),
                "static_observation_measurement": (
                    approach_track_id != track.track_id
                    and bool(c4i_center)
                ),
            }
            for track in canonical_state.subjects
        ]
        routes = {
            vehicle_id: state.route_points_payload()
            for vehicle_id, state in self.states.items()
        }
        arguments = {
            "elapsed_s": planning_elapsed_s,
            "vehicles": vehicles,
            "subjects": subjects,
            "routes": routes,
            "selected_track_id": canonical_state.selected_track_id,
            "approach_track_id": approach_track_id,
        }
        if not force:
            self._planning_future_generation = self._planning_generation
            self._planning_future_elapsed_s = planning_elapsed_s
            self._planning_future_initial_precompute = (
                initial_prefix_precompute
            )
            self._planning_future = self._planning_executor.submit(
                self.planning_engine.update,
                **arguments,
            )
            return
        try:
            result = self.planning_engine.update(**arguments)
        except Exception as error:  # Keep the last valid route on planner failure.
            if initial_prefix_precompute:
                self._initial_rhp_precompute_requested = False
            self._planning_payload = {
                **self._planning_payload,
                "status": "PLANNER ERROR",
                "error": str(error),
            }
            return

        self._apply_planning_result(
            result,
            planning_elapsed_s,
            initial_prefix_precompute=initial_prefix_precompute,
        )

    def _collect_planning_result(self, *, wait: bool = False) -> None:
        future = self._planning_future
        if future is None or (not wait and not future.done()):
            return
        generation = self._planning_future_generation
        simulation_elapsed_s = self._planning_future_elapsed_s
        initial_prefix_precompute = (
            self._planning_future_initial_precompute
        )
        self._planning_future = None
        self._planning_future_initial_precompute = False
        try:
            result = future.result()
        except Exception as error:  # Keep the last valid route on planner failure.
            if initial_prefix_precompute:
                self._initial_rhp_precompute_requested = False
            if generation == self._planning_generation:
                self._planning_payload = {
                    **self._planning_payload,
                    "status": "PLANNER ERROR",
                    "error": str(error),
                }
            return
        if generation != self._planning_generation:
            return
        self._apply_planning_result(
            result,
            simulation_elapsed_s,
            initial_prefix_precompute=initial_prefix_precompute,
        )

    def _apply_planning_result(
        self,
        result,
        simulation_elapsed_s: float,
        *,
        initial_prefix_precompute: bool = False,
    ) -> None:
        canonical_state = self.states[1]

        for vehicle_id, solution in result.approaches.items():
            state = self.states.get(vehicle_id)
            if state is not None:
                state.set_external_approach(solution)
        queued_vehicle_ids = []
        precomputed_vehicle_ids = []
        manual_hold_vehicle_ids = []
        # v0.5: SPX 사전계획이 권위이면 RHP 라이브 탐색 재계획(route_updates)을
        # 적용하지 않는다. 위 approach 적용과 아래 탐지→구조 접근은 그대로 수행된다.
        if not self._spx_preplanned:
            for vehicle_id, points in result.route_updates.items():
                state = self.states.get(vehicle_id)
                if state is None:
                    continue
                if state.manual_route_hold_active:
                    manual_hold_vehicle_ids.append(vehicle_id)
                    continue
                if state.queue_runtime_route(result.revision, points):
                    queued_vehicle_ids.append(vehicle_id)
                    if (
                        state.pending_runtime_route_revision == result.revision
                        and state.runtime_route_revision != result.revision
                    ):
                        precomputed_vehicle_ids.append(vehicle_id)
        rendered = result.render_dict()
        search_active = any(
            state.search_started for state in self.states.values()
        )
        if not search_active:
            rendered["beliefs"] = []
        self._planning_payload = {
            **rendered,
            "status": "ACTIVE",
            "update_interval_s": 1.0,
            "simulation_elapsed_s": simulation_elapsed_s,
            "queued_vehicle_ids": queued_vehicle_ids,
            "precomputed_vehicle_ids": precomputed_vehicle_ids,
            "initial_prefix_precomputed": initial_prefix_precompute,
            "manual_hold_vehicle_ids": manual_hold_vehicle_ids,
        }
        # Two reachability rings: the OUTER ring is the planning-max radius (the
        # subject could be anywhere up to subject_max_speed); the INNER ring is
        # how far the subject reaches at ITS OWN average speed — the most likely
        # extent.  The map draws the outer ring faint and the inner ring bold.
        planner_payload = self._planning_payload.get("planner")
        if isinstance(planner_payload, dict):
            max_kph = float(
                (self.store.mission_metadata or {}).get(
                    "subject_max_speed_kph", 40.0
                )
            ) or 40.0
            subj = canonical_state.selected_subject
            subj_mps = float(subj.speed_mps) if subj is not None else 0.0
            frac = max(0.0, min(1.0, subj_mps / max(1e-6, max_kph / 3.6)))
            radius = float(planner_payload.get("search_radius_m", 0.0) or 0.0)
            planner_payload["search_radius_expected_m"] = radius * frac
        if queued_vehicle_ids:
            vehicles_text = ", ".join(
                f"UAV-{vehicle_id:02d}" for vehicle_id in queued_vehicle_ids
            )
            decision_interval_s = result.planner["decision_interval_s"]
            if set(precomputed_vehicle_ids) == set(queued_vehicle_ids):
                self.statusMessage.emit(
                    "CPP TP t=0 RHP 초기 PREFIX 캐시 완료 // "
                    f"{vehicles_text} // PF·센서·25초 시계 미개시"
                )
            else:
                self.statusMessage.emit(
                    "RHP 확률경로 적용 // "
                    f"{vehicles_text} // {decision_interval_s:g}초 ACTION PREFIX"
                )
        elif manual_hold_vehicle_ids:
            vehicles_text = ", ".join(
                f"UAV-{vehicle_id:02d}" for vehicle_id in manual_hold_vehicle_ids
            )
            self.statusMessage.emit(
                f"수동 경로 보호 중 // {vehicles_text} // RHP 후보는 평가만 수행"
            )
        if result.detections:
            # A subject is "busy" once it is found or already being approached by
            # a designated vehicle.  STEP3 sequential: the FIRST subject stays
            # busy (rescue group approaching) while the research group is still
            # free to detect and approach the SECOND subject.  (The old guard
            # `not any(subject_detected)` blocked every detection after the first
            # subject, so the second was never engaged.)
            busy_track_ids = set(self._found_track_ids)
            for state in self.states.values():
                if state.subject_detected and state.selected_track_id is not None:
                    busy_track_ids.add(int(state.selected_track_id))
            fresh_detections = [
                detection
                for detection in result.detections
                if int(detection["track_id"]) not in busy_track_ids
            ]
            if fresh_detections:
                preferred = next(
                    (
                        detection
                        for detection in fresh_detections
                        if int(detection["track_id"])
                        == canonical_state.selected_track_id
                    ),
                    fresh_detections[0],
                )
                self._approach_detected_subject(
                    int(preferred["track_id"]),
                    int(preferred["vehicle_id"]),
                    automatic=True,
                    run_planning_cycle=False,
                )

    def _check_ingress_tp_transition(self) -> None:
        """Check if all vehicles crossed the TP-line.

        Called every tick.  Once all launched SAR have
        ``search_started == True`` (each reached its first waypoint),
        the ingress flag is set.  Arc patrol start is handled by
        ``_check_tp_arrival`` which runs for all multi-vehicle missions.
        """
        if not self._ingress_enabled:
            return
        if self._ingress_tp_transition_done:
            return
        if not all(
            state.search_started
            for state in self.states.values()
            if state.mission_launched
        ):
            return
        self._ingress_tp_transition_done = True

    def _check_tp_arrival(self) -> None:
        """Start the search phase once every launched vehicle crosses the TP line.

        Works for a single UAV as well as the full fleet (a lone searcher still
        needs the belief filter, the search-boundary ring and detection to come
        online after it reaches the TP).
        """
        if self._continuous_search_active or self._rhp_spx_search_active:
            # Already started (either flow) — do NOT re-enter, or the RHP-SPX
            # planner would be re-created every tick (resetting its cell queues
            # and freezing every vehicle on its first cell).
            return
        launched = [s for s in self.states.values() if s.mission_launched]
        if not launched:
            return
        if not all(s.search_started for s in launched):
            return
        if self._planning_mode == "RHP-SPX":
            # Real-time adaptive cell re-selection: the planner re-scores cells by
            # PF belief every epoch and hops each vehicle to its next cell (no
            # fixed baked lawnmower).
            self._start_rhp_spx_search()
        else:
            self._start_continuous_search()

    def _start_continuous_search(self) -> None:
        """Begin searching after the fleet crosses the TP line.

        Per the operator's continuous-lawnmower requirement, the vehicles keep
        flying their baked, continuous SPX lawnmower — no adaptive per-cell
        re-planning or arc-sector fragmentation.  ``_spx_preplanned`` stays True
        so live RHP route updates are not applied; the planning engine still
        runs each cycle for belief tracking, the search-boundary ring and
        automatic detection.
        """
        self._continuous_search_active = True
        self._spx_preplanned = True
        # The RHP "wait for N route updates before detecting" gate assumes live
        # re-planning refines the belief first.  A fixed continuous lawnmower
        # never emits route updates, so that gate would block detection forever
        # — open it now that the fleet is sweeping.
        self.planning_engine.minimum_route_updates_before_detection = 0
        # If the fleet swept an ingress corridor on the way in, re-solve the SPX
        # lawnmower to EXCLUDE that already-searched corridor — otherwise the
        # baked routes send vehicles back over ground they just swept.
        corridor = (
            self._ingress_corridor_vertices()
            if self._ingress_tp_transition_done
            else None
        )
        if corridor:
            self._trigger_spx_replan(
                reason="ingress_complete",
                ingress_corridor_vertices=corridor,
            )
        self._run_planning_cycle(force=True)
        n_launched = sum(
            1 for s in self.states.values() if s.mission_launched
        )
        self.statusMessage.emit(
            f"탐색 개시 // {n_launched}기 연속 lawnmower"
            + (" // ingress 소인 구역 제외 재계획" if corridor else "")
            + " // 탐색 경계 표시"
        )

    def _start_arc_patrol(self) -> None:
        """Activate RHP arc sector patrol (Phase 2) before SPX lawnmower."""
        self._arc_patrol_active = True
        self._arc_patrol_start_s = self.states[1].simulation_elapsed_s
        self._spx_preplanned = False
        self._run_planning_cycle(force=True)
        n_launched = sum(
            1 for s in self.states.values() if s.mission_launched
        )
        self.statusMessage.emit(
            f"아크 순찰 개시 // {n_launched}기×60° 섹터 // "
            f"{self._arc_patrol_duration_s:g}초 후 SPX lawnmower 전환"
        )

    def _start_rhp_spx_search(self) -> None:
        """Activate RHP-SPX adaptive cell-order search (Phase 2)."""
        self._rhp_spx_search_active = True
        self._spx_preplanned = False
        # Open the detection gate now (as the continuous flow does).  Otherwise a
        # vehicle passing the subject in the first ~3 route updates is NOT
        # detected — the "flew right past but didn't find" bug.
        self.planning_engine.minimum_route_updates_before_detection = 0
        self._setup_rhp_spx_planner()
        self._run_planning_cycle(force=True)
        n_launched = sum(
            1 for s in self.states.values() if s.mission_launched
        )
        self.statusMessage.emit(
            f"RHP-SPX 적응 탐색 개시 // {n_launched}기 "
            "셀순서 PF 재배열 lawnmower"
        )

    def _spatial_cell_partition(
        self, spx_assignments, grid_w: int, grid_h: int, radius_m: float,
    ) -> "tuple[tuple[int, ...], ...]":
        """Re-partition the searchable SPX cells spatially: each cell goes to the
        UAV whose current (ingress-exit) position is nearest, then each UAV's
        cells are ordered nearest-neighbour from that position.  Vehicles cover
        contiguous nearby regions, so they don't cross each other going from the
        TP line to their search area and transit distance is minimised."""
        frame = self.planning_engine.frame
        center = self.planning_engine.search_center
        cw = 2.0 * radius_m / max(1, grid_w)
        ch = 2.0 * radius_m / max(1, grid_h)

        def cell_xy(cell: int) -> tuple[float, float]:
            x = cell % grid_w
            y = cell // grid_w
            return (
                center.east_m - radius_m + (x + 0.5) * cw,
                center.north_m - radius_m + (y + 0.5) * ch,
            )

        searchable = sorted({int(c) for path in spx_assignments for c in path})
        veh_xy: dict[int, tuple[float, float]] = {}
        for vid, state in self.states.items():
            local = frame.to_local(state.vehicle.latitude, state.vehicle.longitude)
            veh_xy[vid] = (local.east_m, local.north_m)
        if not veh_xy or not searchable:
            return tuple(tuple(int(c) for c in path) for path in spx_assignments)
        vids = sorted(veh_xy)
        # Lateral axis = perpendicular to the ingress heading (exit-centroid -> TP).
        # The exits are spread along it, so sorting BOTH vehicles and cells by
        # lateral position and splitting the cells into equal chunks gives each
        # UAV a balanced left-to-right slice (no crossing, even workload).
        ex = sum(veh_xy[v][0] for v in vids) / len(vids)
        ey = sum(veh_xy[v][1] for v in vids) / len(vids)
        hx, hy = center.east_m - ex, center.north_m - ey
        hn = math.hypot(hx, hy) or 1.0
        px, py = -hy / hn, hx / hn  # left perpendicular (lateral) unit

        def lateral(pt: tuple[float, float]) -> float:
            return pt[0] * px + pt[1] * py

        vids_sorted = sorted(vids, key=lambda v: lateral(veh_xy[v]))
        cells_sorted = sorted(searchable, key=lambda c: lateral(cell_xy(c)))
        n, m = len(vids_sorted), len(cells_sorted)
        result_by_vid: dict[int, tuple[int, ...]] = {}
        for i, vid in enumerate(vids_sorted):
            chunk = cells_sorted[i * m // n:(i + 1) * m // n]
            cur = veh_xy[vid]
            remaining = list(chunk)
            ordered: list[int] = []
            while remaining:  # nearest-neighbour order from the exit
                nxt = min(
                    remaining,
                    key=lambda c: (cell_xy(c)[0] - cur[0]) ** 2
                    + (cell_xy(c)[1] - cur[1]) ** 2,
                )
                ordered.append(nxt)
                cur = cell_xy(nxt)
                remaining.remove(nxt)
            result_by_vid[vid] = tuple(ordered)
        return tuple(result_by_vid[v] for v in vids)

    def _setup_rhp_spx_planner(self) -> None:
        """Create and attach an RHPSPXPlanner from baked mission metadata."""
        from cpp_search.core.models import MissionConfig, Point2D, SensorSpec
        from .planning.rhp_spx import RHPSPXPlanner

        meta = self.store.mission_metadata or {}
        spx_assignments = meta.get("spx_assignments")
        if not spx_assignments:
            self.statusMessage.emit(
                "RHP-SPX 전환 실패: SPX 셀 배정 없음 // 3단계 흐름 fallback"
            )
            self._planning_mode = ""
            self._rhp_spx_search_active = False
            self._spx_preplanned = True
            self._start_arc_patrol()
            return

        assignments = tuple(tuple(path) for path in spx_assignments)
        search_radius_m = float(meta.get("search_radius_m", 8900.0))
        grid_shape = (meta.get("spx_certified") or {}).get(
            "grid_shape", [6, 6]
        )
        grid_height, grid_width = int(grid_shape[0]), int(grid_shape[1])

        # Spatially re-partition the searchable cells so each UAV owns the cells
        # nearest its ingress-exit position (no crossing at the ingress->search
        # handoff, minimal transit).  On by default; opt out with
        # ``spatial_cell_assignment: false``.
        if meta.get("spatial_cell_assignment", True):
            spatial = self._spatial_cell_partition(
                spx_assignments, grid_width, grid_height, search_radius_m
            )
            if spatial and all(len(p) > 0 for p in spatial):
                assignments = spatial

        search_speed_kph = 160.0
        transit_speed_kph = 160.0
        n_uav = len(assignments)
        mission_config = MissionConfig(
            center=Point2D(
                self.planning_engine.search_center.east_m,
                self.planning_engine.search_center.north_m,
            ),
            search_radius_m=search_radius_m,
            uav_count=n_uav,
            transit_speed_mps=transit_speed_kph / 3.6,
            search_speed_mps=search_speed_kph / 3.6,
        )
        # Lane spacing = track_spacing = effective_sweep_width * (1 - overlap).
        # A bare SensorSpec() derives the sweep width from the instantaneous FOV
        # swath (~190 m), giving ~152 m lanes — far denser than the search
        # camera's effective sweep width and inconsistent with detection.  Drive
        # the lawnmower off the SAME effective sweep width the detection model
        # uses (default 400 m, mission-overridable) so lanes are ~320 m (20 %
        # overlap): full coverage, no gaps, and ~half the redundant lanes.
        sweep_width_m = float(self.search_camera_spec.effective_sweep_width_m)
        sensor_spec = SensorSpec(ground_scan_radius_m=sweep_width_m / 2.0)

        # Optional per-cell sweep size: default (unset) = full-cell coverage; a
        # mission wanting visible cell-to-cell hopping sets a smaller lane count
        # so each cell's box fits inside it and the vehicle completes + hops.
        cell_sweep_lanes = meta.get("rhp_cell_sweep_lanes")
        complete_cell_sweep = bool(meta.get("rhp_complete_cell_sweep", False))
        # Terrain no-fly cells (in the planner's own grid), so live routes detour
        # transits around them. Recomputed from the baked terrain + flight geometry.
        no_fly_cells = self._rhp_no_fly_cells(
            meta, self.planning_engine.search_center,
            search_radius_m, grid_width, grid_height,
        )
        planner = RHPSPXPlanner(
            frame=self.planning_engine.frame,
            search_center=self.planning_engine.search_center,
            search_radius_m=search_radius_m,
            grid_width=grid_width,
            grid_height=grid_height,
            initial_assignments=assignments,
            mission_config=mission_config,
            sensor_spec=sensor_spec,
            decision_interval_s=25.0,
            search_altitude_m=self.search_camera_spec.altitude_m,
            cell_sweep_lanes=(
                int(cell_sweep_lanes) if cell_sweep_lanes else None
            ),
            complete_cell_sweep=complete_cell_sweep,
            no_fly_cells=no_fly_cells,
        )
        self.planning_engine.set_rhp_spx_planner(planner)

    def _rhp_no_fly_cells(
        self, meta, search_center, search_radius_m, grid_width, grid_height,
    ) -> tuple[int, ...]:
        """No-fly cell indices in the RHP grid, from the baked terrain + flight
        altitude, so the live planner detours transits around no-fly terrain."""
        terr_npz = meta.get("spx_terrain_npz")
        if not terr_npz:
            return ()
        try:
            from pathlib import Path as _Path
            from cpp_search.core.terrain import RasterTerrainField
            from cpp_search.core.models import Point2D as _P2
            from .planning.rhp_spx import _CellGrid
            root = _Path(__file__).resolve().parent.parent
            path = root / terr_npz
            if not path.exists():
                return ()
            terrain = RasterTerrainField.load(str(path))
            flight = float(meta.get("spx_flight_msl_m", 1130.0))
            clear = float(meta.get("spx_clearance_agl_m", 120.0))
            grid = _CellGrid(
                center_x=search_center.east_m, center_y=search_center.north_m,
                half_side_m=search_radius_m, width=grid_width, height=grid_height,
            )
            cells = []
            for c in range(grid.cell_count):
                cc = grid.center_of(c)
                elev = terrain.elevation_at(
                    cc.x - search_center.east_m, cc.y - search_center.north_m
                )
                if elev is not None and (flight - float(elev)) < clear:
                    cells.append(c)
            return tuple(cells)
        except Exception:
            return ()

    def _check_arc_patrol_completion(self) -> None:
        """End arc patrol and transition to SPX lawnmower routes (Phase 3)."""
        if not self._arc_patrol_active:
            return
        elapsed = (
            self.states[1].simulation_elapsed_s - self._arc_patrol_start_s
        )
        if elapsed < self._arc_patrol_duration_s:
            return
        self._arc_patrol_active = False
        self._arc_patrol_done = True
        self._spx_preplanned = True
        for vehicle_id, state in self.states.items():
            if state.mission_launched:
                state.update_route_from_store(self.store, vehicle_id)
        corridor = (
            self._ingress_corridor_vertices()
            if self._ingress_tp_transition_done
            else None
        )
        self._trigger_spx_replan(
            reason="arc_patrol_complete",
            ingress_corridor_vertices=corridor,
        )
        self.statusMessage.emit(
            "아크 순찰 완료 // SPX lawnmower 소인 전환 // "
            "SPX 재계획 MILP 시작"
        )

    def _recenter_search_on_subject(self, track_id: int) -> bool:
        """Re-center the search circle on a subject that lies outside it.

        Sequential multi-zone SAR: once the fleet has rescued the subject inside
        the current circle, the next subject can be beyond it.  Move the
        predicted-subject center in mission metadata to that subject's zone and
        rebuild the planning engine so its belief filter and search geometry
        follow the fleet to the new area.  The subsequent SPX re-solve (which
        reads the same metadata center) then lays routes over the new zone.

        Returns True only when the center actually moved (subject was outside).
        """
        subject = next(
            (s for s in self.store.initial_subjects if s.track_id == track_id),
            None,
        )
        if subject is None:
            return False
        return self._recenter_on_point(subject.latitude, subject.longitude)

    def _recenter_on_point(self, latitude: float, longitude: float) -> bool:
        """Move the predicted-subject search center to an explicit point.

        Shared core of :meth:`_recenter_search_on_subject`.  Sequential STEP3
        re-centres on the *second predicted point* (TP2) rather than the moving
        subject's live position, so the phase-2 SPX sweep is deterministic and
        the subject can be placed on it.  Returns True only when the center
        actually moved (point was outside the current circle).
        """
        meta = self.store.mission_metadata or {}
        current = meta.get("rally_predicted_subject") or {}
        cur_lat = current.get("latitude")
        cur_lon = current.get("longitude")
        radius_m = float(meta.get("search_radius_m", 8900.0))
        if cur_lat is not None and cur_lon is not None:
            if horizontal_distance_m(
                cur_lat, cur_lon, latitude, longitude
            ) <= radius_m:
                return False
        new_center = {"latitude": latitude, "longitude": longitude}
        meta["rally_predicted_subject"] = new_center
        # The SPX re-solve reads arc_search_pattern.center first (see
        # stone_adapter._search_center), so move it too or the MILP would keep
        # re-solving the original circle while the belief engine re-centers.
        arc = meta.get("arc_search_pattern")
        if isinstance(arc, dict):
            arc["center"] = dict(new_center)
        self.store.mission_metadata = meta
        center = self.store.sites.get("GCS") or self.store.sites.get("LC")
        center_lat = center.latitude if center else self.state.center_latitude
        center_lon = center.longitude if center else self.state.center_longitude
        self.planning_engine = self._new_planning_engine(center_lat, center_lon)
        # A fresh engine restores the detection gate; keep it open while the
        # search is running so the next subject can still be found.  RHP-SPX
        # (STEP2/STEP3) uses its own active flag, so check both.
        if self._continuous_search_active or self._rhp_spx_search_active:
            self.planning_engine.minimum_route_updates_before_detection = 0
        return True

    def _check_found_resume_search(self) -> None:
        """When all vehicles are FOUND and unfound subjects remain, resume."""
        if self._resume_search_triggered:
            return
        if self._spx_replan_pending:
            return
        launched = [s for s in self.states.values() if s.mission_launched]
        if not launched:
            return
        if not all(s.subject_found for s in launched):
            return
        for s in launched:
            if s.selected_track_id is not None:
                self._found_track_ids.add(s.selected_track_id)
        all_subject_ids = {
            subj.track_id for subj in self.store.initial_subjects
        }
        unfound_ids = all_subject_ids - self._found_track_ids
        if not unfound_ids:
            return
        self._resume_search_triggered = True
        for state in launched:
            state.resume_search()
        next_track_id = min(unfound_ids)
        for state in launched:
            state.selected_track_id = next_track_id
        recentered = self._recenter_search_on_subject(next_track_id)
        self.statusMessage.emit(
            f"구조 완료 ({len(self._found_track_ids)}/"
            f"{len(all_subject_ids)}) // "
            f"잔여 {len(unfound_ids)}명 탐색 재개"
            + (" // 원 밖 구역 재중심화" if recentered else "")
            + " // SPX 재계획 요청"
        )
        corridor = (
            self._ingress_corridor_vertices()
            if self._ingress_tp_transition_done
            else None
        )
        self._trigger_spx_replan(
            reason="detection",
            exclude_track_ids=self._found_track_ids.copy(),
            ingress_corridor_vertices=corridor,
        )
        self._resume_search_triggered = False

    def _trigger_spx_replan(
        self,
        *,
        reason: str = "incremental",
        exclude_track_ids: "set[int] | None" = None,
        ingress_corridor_vertices: "list[tuple[float, float]] | None" = None,
    ) -> None:
        """Submit an incremental SPX re-solve on a background thread.

        The MILP re-solve (``stone_adapter.plan_certified_search``) is heavy
        (~60 s).  It runs on ``_spx_replan_executor`` (separate from the RHP
        planner) so the 1 Hz tick loop is not blocked.  The result is collected
        by ``_collect_spx_replan_result`` and applied via
        ``apply_plan_to_store``.

        Parameters
        ----------
        reason : str
            Why the replan was triggered: "ingress_complete" (TP transition),
            "detection" (subject found, re-solve for remaining subjects).
        exclude_track_ids : set[int] | None
            Track IDs of found subjects to exclude from the re-plan.
        """
        if self._spx_replan_future is not None and not self._spx_replan_future.done():
            # A previous SPX re-plan is still running; supersede it.
            self._spx_replan_future.cancel()
        self._spx_replan_pending = True
        self._spx_replan_reason = reason
        self._spx_replan_exclude_track_ids = exclude_track_ids or set()
        generation = self._spx_replan_generation

        # Build keyword arguments from mission metadata so the re-solve uses
        # the same parameters the original bake used.
        meta = self.store.mission_metadata or {}
        spx_kwargs: dict = {}
        if meta.get("flight_altitude_msl_m") is not None:
            spx_kwargs["flight_msl_m"] = float(meta["flight_altitude_msl_m"])
        if meta.get("search_speed_kph") is not None:
            spx_kwargs["search_speed_kph"] = float(meta["search_speed_kph"])
        if meta.get("subject_max_speed_kph") is not None:
            spx_kwargs["subject_max_speed_kph"] = float(meta["subject_max_speed_kph"])
        # Re-solve on the SAME grid the mission was baked with — otherwise the
        # corridor re-solve reverts to the 6x6 default, packing a whole budget
        # into one coarse cell (the "spinning in one cell" bug).  The bake writes
        # spx_certified.grid_shape; a finer grid keeps cell-to-cell progression.
        grid_shape = (meta.get("spx_certified") or {}).get("grid_shape")
        if grid_shape:
            spx_kwargs["grid_n"] = int(grid_shape[0])

        from .planning.stone_adapter import plan_certified_search

        # Capture a frozen copy of the store data the solver needs.  The solver
        # reads store.sites / initial_subjects / mission_metadata — all of
        # which could theoretically be mutated on the main thread while the
        # background solve runs.  SiteStore is lightweight; a full snapshot via
        # replace_from is the simplest thread-safety measure.
        snapshot = SiteStore()
        snapshot.replace_from(self.store)

        excluded = set(int(t) for t in self._spx_replan_exclude_track_ids)
        corridor = ingress_corridor_vertices

        def _solve():
            return plan_certified_search(
                snapshot,
                exclude_track_ids=excluded or None,
                ingress_corridor_vertices=corridor,
                **spx_kwargs,
            )

        self._spx_replan_future = self._spx_replan_executor.submit(_solve)
        self._spx_replan_future._spx_generation = generation  # type: ignore[attr-defined]
        self.statusMessage.emit(
            f"SPX 재계획 MILP 시작 // 사유: {reason} // "
            f"제외 대상: {excluded or '없음'} // 배경 스레드"
        )

    def _collect_spx_replan_result(self) -> None:
        """Collect a completed SPX re-plan future and apply it to the store."""
        future = self._spx_replan_future
        if future is None or not future.done():
            return
        self._spx_replan_future = None
        generation = getattr(future, '_spx_generation', -1)
        if generation != self._spx_replan_generation:
            # A mission reset happened while the solve was running.
            return

        from .planning.stone_adapter import apply_plan_to_store

        try:
            plan = future.result()
        except Exception as error:
            self._spx_replan_pending = False
            self.statusMessage.emit(
                f"SPX 재계획 실패 // {error}"
            )
            return

        # Preserve the live search/approach state across this mid-flight
        # re-solve.  apply_plan_to_store notifies subscribers, which fires
        # _on_plan_changed synchronously; WITHOUT this flag set first, that
        # handler takes the "new mission" branch and resets
        # _continuous_search_active / _found_track_ids and rebuilds the planning
        # engine — which restarts the search and re-triggers the re-solve in a
        # loop.  Set it before apply so the notify takes the preserve branch.
        self._preserve_execution_on_next_plan_change = True
        # Apply the new certified routes to the live mission store.
        apply_plan_to_store(self.store, plan)
        # Reload every FlyState so vehicles pick up the new routes.
        for vehicle_id, state in self.states.items():
            state.update_route_from_store(self.store, vehicle_id)
        # STEP3: prepend the parallel-formation transit to TP2 ahead of the
        # freshly-applied phase-2 SPX routes for the research group (must run
        # AFTER update_route_from_store, which would otherwise overwrite it).
        research_ids = getattr(self, "_phase2_research_ids", None)
        if research_ids:
            self._prepend_phase2_parallel(research_ids)
            self._phase2_research_ids = set()
        self._spx_replan_pending = False
        cert = plan.certificate
        self._spx_detection_probability = cert.detection_probability
        if isinstance(self.map_bridge, FlyMapBridge):
            self.map_bridge.spx_pd = cert.detection_probability
        self.statusMessage.emit(
            f"SPX 재계획 완료 // {cert.summary()} // "
            f"사유: {self._spx_replan_reason}"
        )
        self._preserve_execution_on_next_plan_change = True
        self._on_plan_changed()

    _MIN_FLEET_SEPARATION_M = 800.0

    def _enforce_fleet_separation(self) -> None:
        """Keep searching UAVs >= ~800 m apart (sensor sweep width W=400 m, so
        2W=800 m between adjacent tracks avoids overlap and near-collisions).
        Any ROUTE-phase pair that gets closer is gently pushed apart along the
        line between them so they slide past at the minimum spacing.  Approach/
        rescue and emergency phases are exempt — vehicles converge on a subject
        on purpose there."""
        sep = self._MIN_FLEET_SEPARATION_M
        # Only in the SEARCH phase (post-TP): during ingress every UAV is still
        # stacked at the single LC point, so a separation push there would shove
        # them apart in a degenerate direction and scramble the lane order.  The
        # ingress lanes are already 800 m apart by construction.
        movers = [
            s for s in self.states.values()
            if s.mission_launched and not s.emergency_mode
            and s.flight_phase == "ROUTE" and s.search_started
        ]
        for i in range(len(movers)):
            a = movers[i]
            for j in range(i + 1, len(movers)):
                b = movers[j]
                d = horizontal_distance_m(
                    a.vehicle.latitude, a.vehicle.longitude,
                    b.vehicle.latitude, b.vehicle.longitude,
                )
                if 1.0 < d < sep:
                    push = (sep - d) / 2.0
                    brg = bearing_deg(
                        a.vehicle.latitude, a.vehicle.longitude,
                        b.vehicle.latitude, b.vehicle.longitude,
                    )
                    a.vehicle.latitude, a.vehicle.longitude = destination_position(
                        a.vehicle.latitude, a.vehicle.longitude, push, brg + 180.0,
                    )
                    b.vehicle.latitude, b.vehicle.longitude = destination_position(
                        b.vehicle.latitude, b.vehicle.longitude, push, brg,
                    )

    def _tick(self) -> None:
        if not self._active:
            return
        self._collect_planning_result()
        self._collect_spx_replan_result()
        for state in self.states.values():
            state.tick()
        self._enforce_fleet_separation()
        self._check_ingress_tp_transition()
        self._check_tp_arrival()
        if self._rhp_spx_search_active:
            # Keep the detection gate open once adaptive search is running — an
            # engine rebuild (e.g. re-centring on the next subject) resets it to
            # the mission default, which would re-block near passes.
            self.planning_engine.minimum_route_updates_before_detection = 0
        if self._planning_mode != "RHP-SPX":
            self._check_arc_patrol_completion()
        self._check_found_resume_search()
        self._run_planning_cycle()
        now = time.monotonic()
        if now - self._last_console_refresh_at >= 0.20:
            for vehicle_id, state in self.states.items():
                state.sync_plan_readiness(self.store, vehicle_id)
            self._refresh_display(refresh_camera=False)
            self._last_console_refresh_at = now
        if now - self._last_camera_refresh_at >= 0.10:
            self.map_stage.refresh_camera()
            self._last_camera_refresh_at = now
        if now - self._last_map_emit_at >= 0.10:
            self._emit_map_state()
            self._last_map_emit_at = now
        all_found = all(
            s.subject_found for s in self.states.values() if s.mission_launched
        )
        all_subject_ids = {
            subj.track_id for subj in self.store.initial_subjects
        }
        no_unfound = not (all_subject_ids - self._found_track_ids)
        if all_found and no_unfound and not self._reported_shutdown:
            self._reported_shutdown = True
            self.statusMessage.emit(
                f"전체 구조 완료 // {len(all_subject_ids)}명 "
                "전원 확인 // SHUT DOWN"
            )
        elif not all_found:
            self._reported_shutdown = False

    def _emit_map_state(self) -> None:
        if self._uses_shared_map:
            has_runtime_preview = bool(self._pending_runtime_routes)
            display_store = (
                self._pending_mission
                if self._pending_dirty and not has_runtime_preview
                else self.store
            )
            render_state = self.state.render_dict(
                display_store,
                include_plan=False,
                include_flight_path=False,
            )
            plan_key = (self._plan_render_revision, self._pending_dirty)
            if self._cached_plan_key != plan_key:
                self._cached_plan_payload = build_mission_map_plan_payload(
                    self.store,
                    display_store,
                    self._pending_dirty and not has_runtime_preview,
                )
                if has_runtime_preview:
                    self._cached_plan_payload["pending_edit"] = True
                if self._ingress_enabled:
                    corridor = self._ingress_corridor_vertices()
                    if corridor:
                        self._cached_plan_payload["ingress_corridor"] = [
                            {"latitude": lat, "longitude": lon}
                            for lat, lon in corridor
                        ]
                self._cached_plan_key = plan_key
            if self._last_sent_plan_key != plan_key:
                # Send the low-rate mission geometry through its own durable
                # browser channel.  Embedding PLAN in one telemetry frame was
                # unsafe: requestAnimationFrame coalescing could replace that
                # frame with the next position-only update before rendering,
                # making TP and every planned RHP route disappear.
                plan_payload = json.dumps(
                    json_safe_payload(self._cached_plan_payload or {}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                plan_argument = json.dumps(
                    plan_payload,
                    ensure_ascii=False,
                )
                self.web_view.page().runJavaScript(
                    "window.setFlyPlan && "
                    f"window.setFlyPlan({plan_argument});"
                )
                self._last_sent_plan_key = plan_key
            render_state["plan_revision"] = self._plan_render_revision
            vehicle_payloads = []
            for vehicle_id, state in self.states.items():
                raw_path = state.flight_path[-1200:]
                path_key = (
                    len(raw_path),
                    (
                        round(float(raw_path[-1]["latitude"]), 7),
                        round(float(raw_path[-1]["longitude"]), 7),
                    )
                    if raw_path
                    else None,
                )
                cached_path = self._flight_path_payload_cache.get(vehicle_id)
                if cached_path is None or cached_path[0] != path_key:
                    cached_path = (
                        path_key,
                        simplify_flight_path(raw_path),
                    )
                    self._flight_path_payload_cache[vehicle_id] = cached_path
                vehicle_payloads.append(
                    {
                        **vars(state.vehicle),
                        "flight_path": cached_path[1],
                        "flight_phase": state.flight_phase,
                        "mission_launched": state.mission_launched,
                        "current_waypoint_index": state.current_waypoint_index,
                        "completed_route_segment_count": (
                            state.completed_route_segment_count
                        ),
                        "runtime_route": state.runtime_route_payload(),
                        "runtime_route_revision": state.runtime_route_revision,
                        "runtime_route_update_count": (
                            state.runtime_route_update_count
                        ),
                        "manual_route_preview": self._pending_runtime_routes.get(
                            vehicle_id,
                            [],
                        ),
                        "manual_route_preview_pending": (
                            vehicle_id in self._pending_runtime_routes
                        ),
                        "manual_route_active": state.manual_route_active,
                        "manual_route_hold_remaining_s": (
                            state.manual_route_hold_remaining_s
                        ),
                        "pending_runtime_route": (
                            state.pending_runtime_route_payload()
                        ),
                        "pending_runtime_route_revision": (
                            state.pending_runtime_route_revision
                        ),
                        "approach_route": state.approach_route_payload(),
                        "camera_mode": state.camera_mode,
                        "detection_source_vehicle_id": (
                            state.detection_source_vehicle_id
                        ),
                        "selected": (
                            self.selected_vehicle_id == vehicle_id
                        ),
                    }
                )
            render_state["vehicles"] = vehicle_payloads
            render_state["selected_vehicle_id"] = self.selected_vehicle_id
            render_state["planning"] = dict(self._planning_payload)
            if self._spx_detection_probability is not None:
                render_state["spx_pd"] = round(self._spx_detection_probability, 4)
            payload = json.dumps(
                json_safe_payload(render_state),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            argument = json.dumps(payload, ensure_ascii=False)
            self.web_view.page().runJavaScript(
                f"window.setFlyState && window.setFlyState({argument});"
            )
        elif isinstance(self.map_bridge, FlyMapBridge):
            self.map_bridge.emit_state()

    def _on_shared_feature_selected(self, code: str) -> None:
        if not self._active or not code.startswith("PSN-"):
            return
        try:
            track_id = int(code.removeprefix("PSN-"))
        except ValueError:
            return
        now = time.monotonic()
        if (
            self._last_map_track_click is not None
            and self._last_map_track_click[0] == track_id
            and now - self._last_map_track_click[1] <= 0.8
        ):
            self._last_map_track_click = None
            self._select_subject(track_id)
            return
        self._last_map_track_click = (track_id, now)
        self.statusMessage.emit(
            f"PSN-{track_id} 지정: 같은 대상을 한 번 더 클릭하십시오."
        )

    def _select_subject_from_table(self, row: int, _column: int) -> None:
        item = self.subject_table.item(row, 0)
        if item is None:
            return
        track_id = int(item.data(Qt.ItemDataRole.UserRole))
        self._select_subject(track_id)

    def _select_subject(self, track_id: int) -> None:
        self._approach_detected_subject(
            track_id,
            self._nearest_detector_id(track_id),
            automatic=False,
            run_planning_cycle=True,
        )

    def _next_unfound_subject_id(self, *, exclude: int) -> int | None:
        """Lowest track_id of a subject not yet found (excluding ``exclude``)."""
        found = set(self._found_track_ids)
        found.add(int(exclude))
        remaining = sorted(
            s.track_id for s in self.store.initial_subjects
            if s.track_id not in found
        )
        return remaining[0] if remaining else None

    def _rank_vehicles_by_distance(self, subject) -> list[int]:
        """Launched vehicle ids ordered nearest-first to ``subject``."""
        ranked: list[tuple[float, int]] = []
        for vehicle_id, state in self.states.items():
            if not state.mission_launched or state.emergency_mode:
                continue
            ranked.append((
                horizontal_distance_m(
                    state.vehicle.latitude,
                    state.vehicle.longitude,
                    subject.latitude,
                    subject.longitude,
                ),
                vehicle_id,
            ))
        ranked.sort()
        return [vehicle_id for _distance, vehicle_id in ranked]

    def _dispatch_research_group(
        self, research_ids: list[int], next_track_id: int
    ) -> None:
        """Send the non-rescue vehicles to re-search the next subject.

        Re-centre the belief/search on that subject's zone (which also rebuilds
        the engine) and re-solve the SPX lawnmower excluding the found subject;
        the research vehicles pick up routes over the new zone and keep flying
        in ROUTE mode while the rescue group approaches the first subject.
        """
        if not research_ids:
            return
        # STEP3: re-centre on the SECOND predicted point (TP2) when the mission
        # defines one, so the phase-2 SPX sweep is deterministic (the subject is
        # placed on it).  Fall back to the moving subject's live position.
        tp2 = (self.store.mission_metadata or {}).get("second_predicted_subject")
        if isinstance(tp2, dict) and tp2.get("latitude") is not None:
            # A focused second zone: shrink the search radius around TP2 so the
            # 3-vehicle phase-2 SPX sweep is dense and deterministic (the huge
            # 8.9 km phase-1 circle would scatter them and leave gaps the subject
            # can hide in).  Set it BEFORE re-centring so the rebuilt belief
            # filter and the SPX re-solve both use the tighter circle.
            r2 = (self.store.mission_metadata or {}).get("second_search_radius_m")
            if r2:
                meta = self.store.mission_metadata or {}
                meta["search_radius_m"] = float(r2)
                self.store.mission_metadata = meta
            self._recenter_on_point(
                float(tp2["latitude"]), float(tp2["longitude"])
            )
        else:
            self._recenter_search_on_subject(next_track_id)
        for vehicle_id in research_ids:
            state = self.states.get(vehicle_id)
            if state is not None:
                state.selected_track_id = next_track_id
        # STEP3: remember the research group so that when the phase-2 SPX solve
        # lands (_collect_spx_replan_result), a parallel-formation corridor to
        # TP2 is prepended ahead of the sweep (mirrors the LC->TP1 ingress) so
        # the group transits together without crossing paths.
        self._phase2_research_ids = {int(v) for v in research_ids}
        # Start the parallel transit to TP2 immediately (over the current routes)
        # so the group does not idle in the phase-1 zone during the ~60 s MILP
        # re-solve; _collect_spx_replan_result re-prepends a fresh corridor onto
        # the phase-2 SPX routes when they land.
        if isinstance(tp2, dict) and tp2.get("latitude") is not None:
            self._prepend_phase2_parallel(research_ids)
        corridor = (
            self._ingress_corridor_vertices()
            if self._ingress_tp_transition_done
            else None
        )
        self._trigger_spx_replan(
            reason="detection",
            exclude_track_ids=self._found_track_ids.copy(),
            ingress_corridor_vertices=corridor,
        )

    def _prepend_phase2_parallel(self, research_ids: "set[int] | list[int]") -> bool:
        """Prepend a parallel-formation corridor (current centroid -> TP2) ahead
        of each research vehicle's phase-2 SPX route (STEP3).

        Mirrors the LC->TP1 ingress: the group forms up abreast (lanes assigned
        by current lateral order to avoid crossing) and flies parallel to the
        second predicted point before the SPX sweep begins.  Called from
        ``_collect_spx_replan_result`` once the phase-2 routes have landed, so
        the prepend survives the route swap.  Returns True if it prepended.
        """
        tp2 = (self.store.mission_metadata or {}).get("second_predicted_subject")
        if not (isinstance(tp2, dict) and tp2.get("latitude") is not None):
            return False
        ids = sorted(
            int(v) for v in research_ids
            if v in self.states and self.states[int(v)].mission_launched
        )
        if not ids:
            return False
        t2lat = float(tp2["latitude"])
        t2lon = float(tp2["longitude"])
        # Launcher = centroid of the research vehicles' current positions.
        lat0 = sum(self.states[v].vehicle.latitude for v in ids) / len(ids)
        lon0 = sum(self.states[v].vehicle.longitude for v in ids) / len(ids)

        def _enu(lat: float, lon: float) -> tuple[float, float]:
            east = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
            north = (lat - lat0) * 111_320.0
            return east, north

        tx, ty = _enu(t2lat, t2lon)
        axis_len = math.hypot(tx, ty)
        if axis_len < 1.0:
            return False
        hx, hy = tx / axis_len, ty / axis_len
        px, py = -hy, hx  # left perpendicular = lateral axis
        # Order vehicles left-to-right along the lateral axis so each maps to the
        # nearest formation lane (build_parallel_ingress numbers lanes 1..n with
        # increasing lateral offset), minimising crossings.
        ordered = sorted(
            ids,
            key=lambda v: (
                _enu(self.states[v].vehicle.latitude,
                     self.states[v].vehicle.longitude)[0] * px
                + _enu(self.states[v].vehicle.latitude,
                       self.states[v].vehicle.longitude)[1] * py
            ),
        )
        from qt_gcs.planning.ingress import build_parallel_ingress

        alt = float(
            (self.store.mission_metadata or {}).get("flight_altitude_msl_m", 600.0)
        )
        formation = build_parallel_ingress(
            lat0, lon0, t2lat, t2lon, count=len(ordered), altitude_m=alt
        )
        prepended = False
        for lane_index, vehicle_id in enumerate(ordered, start=1):
            legs = formation.routes.get(lane_index, [])
            if not legs:
                continue
            existing = self.store.vehicle_waypoints[vehicle_id]
            transit_wps = [
                MissionPoint(
                    latitude=lat,
                    longitude=lon,
                    altitude_m=leg_alt,
                    code=f"P2F{seq:03d}",
                    label=f"UAV-{vehicle_id:02d} Phase2 Transit {seq}",
                    point_type="INGRESS_WAYPOINT",
                    sequence=seq,
                )
                for seq, (lat, lon, leg_alt) in enumerate(legs, start=1)
            ]
            combined = transit_wps + list(existing)
            for i, wp in enumerate(combined, start=1):
                wp.sequence = i
            self.store.vehicle_waypoints[vehicle_id] = combined
            state = self.states[vehicle_id]
            state.update_route_from_store(self.store, vehicle_id)
            # Force the vehicle onto the first transit waypoint; the nearest-match
            # in update_route_from_store would otherwise skip the parallel leg.
            state.current_waypoint_index = 0
            state.completed_route_segment_count = 0
            prepended = True
        return prepended

    def _approach_detected_subject(
        self,
        track_id: int,
        detector_id: int | None,
        *,
        automatic: bool,
        run_planning_cycle: bool,
    ) -> None:
        canonical_subject = next(
            (
                track
                for track in self.states[1].subjects
                if track.track_id == track_id and not track.found
            ),
            None,
        )
        if canonical_subject is None:
            self.statusMessage.emit(f"PSN-{track_id}는 접근 가능한 대상이 아닙니다.")
            return
        detected_track = (
            canonical_subject.latitude,
            canonical_subject.longitude,
            canonical_subject.altitude_m,
            canonical_subject.speed_mps,
            canonical_subject.heading_deg,
        )
        # One SAR detection becomes the shared AUTO track immediately. This
        # avoids six independent stale copies steering to different positions.
        for state in self.states.values():
            local_subject = next(
                (
                    track
                    for track in state.subjects
                    if track.track_id == track_id and not track.found
                ),
                None,
            )
            if local_subject is not None:
                (
                    local_subject.latitude,
                    local_subject.longitude,
                    local_subject.altitude_m,
                    local_subject.speed_mps,
                    local_subject.heading_deg,
                ) = detected_track
        # Fleet split (sequential multi-subject): the ``rescue_group_size``
        # nearest vehicles rescue THIS subject while the rest peel off to
        # re-search the NEXT subject.  Without the metadata flag the whole fleet
        # converges (the original cooperative-approach behaviour).
        rescue_size = (self.store.mission_metadata or {}).get("rescue_group_size")
        next_track_id = self._next_unfound_subject_id(exclude=track_id)
        # Vehicles currently assigned to THIS subject.  At the first detection the
        # whole launched fleet is assigned to subject #1; after the split only the
        # research group is assigned to subject #2, so the phase-2 approach uses
        # exactly those vehicles (all of them — no further split).
        assigned_ids = [
            vid
            for vid, state in self.states.items()
            if state.mission_launched
            and not state.emergency_mode
            and state.selected_track_id == track_id
        ]
        if not assigned_ids:
            assigned_ids = [
                vid
                for vid, state in self.states.items()
                if state.mission_launched and not state.emergency_mode
            ]
        # Split only when more vehicles are assigned than the rescue group needs
        # AND another subject still awaits — otherwise every assigned vehicle
        # converges (the phase-2 "all 3 approach the 2nd subject" case).
        split = (
            bool(rescue_size)
            and next_track_id is not None
            and len(assigned_ids) > int(rescue_size)
        )
        if split:
            ranked = [
                vid
                for vid in self._rank_vehicles_by_distance(canonical_subject)
                if vid in assigned_ids
            ]
            rescue_ids = ranked[: int(rescue_size)]
            research_ids = ranked[int(rescue_size):]
            results = [
                self.states[vid].designate_subject(
                    track_id, detector_vehicle_id=detector_id
                )
                for vid in rescue_ids
            ]
            if any(results):
                self._found_track_ids.add(track_id)
                self._dispatch_research_group(research_ids, next_track_id)
                self.statusMessage.emit(
                    f"PSN-{track_id} 발견 // 접근 {len(rescue_ids)}기 "
                    f"(UAV-{', UAV-'.join(f'{v:02d}' for v in rescue_ids)}) "
                    f"// 재탐색 {len(research_ids)}기 → PSN-{next_track_id} 구역 재중심화"
                )
        else:
            results = [
                self.states[vid].designate_subject(
                    track_id,
                    detector_vehicle_id=detector_id,
                )
                for vid in assigned_ids
            ]
        if not any(results):
            return
        if split:
            self._refresh_display()
            self._emit_map_state()
            return
        if detector_id is not None:
            # Display before the planning cycle so a heavy IMM/PF evaluation
            # cannot delay or hide the operator warning.
            self._show_detection_notice(detector_id, automatic=automatic)
        if run_planning_cycle:
            self._run_planning_cycle(force=True)
        self._refresh_display()
        self._emit_map_state()
        if detector_id is not None:
            self.statusMessage.emit(
                f"PSN-{track_id} {'자동' if automatic else '수동'} 발견 // "
                f"UAV-{detector_id:02d} // AUTO 협동 접근"
            )
        else:
            self.statusMessage.emit(
                f"PSN-{track_id} 선택 // 이륙 후 탐지 가능"
            )

    def _nearest_detector_id(self, track_id: int) -> int | None:
        candidates: list[tuple[float, int]] = []
        for vehicle_id, state in self.states.items():
            if not state.mission_launched or state.emergency_mode:
                continue
            subject = next(
                (
                    track
                    for track in state.subjects
                    if track.track_id == track_id and not track.found
                ),
                None,
            )
            if subject is None:
                continue
            candidates.append(
                (
                    horizontal_distance_m(
                        state.vehicle.latitude,
                        state.vehicle.longitude,
                        subject.latitude,
                        subject.longitude,
                    ),
                    vehicle_id,
                )
            )
        return min(candidates)[1] if candidates else None

    def _show_detection_notice(
        self,
        vehicle_id: int,
        *,
        automatic: bool = True,
    ) -> None:
        if self._detection_dialog is not None:
            self._detection_dialog.close()
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("발견 확인")
        dialog.setText(
            "⚠ 탐지 대상을 발견했습니다.\n"
            f"최초 발견: UAV-{vehicle_id:02d}\n"
            f"발견 방식: {'자동 인식' if automatic else 'MANUAL 수동 지정'}\n"
            "6대 AUTO 협동 구조 접근을 시작합니다."
        )
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.finished.connect(
            lambda _result, current=dialog: (
                setattr(self, "_detection_dialog", None)
                if self._detection_dialog is current
                else None
            )
        )
        self._detection_dialog = dialog
        dialog.show()

        def bring_to_front() -> None:
            if self._detection_dialog is dialog:
                dialog.raise_()
                dialog.activateWindow()

        QTimer.singleShot(0, bring_to_front)

    def _ingress_corridor_vertices(self) -> list[tuple[float, float]]:
        """Compute the 4 corners of the ingress sweep corridor for map display."""
        tp = self.store.mission_metadata.get("rally_predicted_subject")
        if not isinstance(tp, dict):
            return []
        launcher = self.store.sites.get("LC")
        if launcher is None:
            return []
        from qt_gcs.planning.ingress import ingress_corridor_polygon

        return ingress_corridor_polygon(
            launcher.latitude,
            launcher.longitude,
            float(tp["latitude"]),
            float(tp["longitude"]),
            count=len(SiteStore.VEHICLE_IDS),
        )

    def _prepend_ingress_routes(self) -> bool:
        """Insert parallel ingress waypoints ahead of each vehicle's SPX route.

        Returns True if ingress waypoints were actually prepended.
        """
        if not self._ingress_enabled:
            return False
        tp = self.store.mission_metadata.get("rally_predicted_subject")
        if not isinstance(tp, dict):
            return False
        launcher = self.store.sites.get("LC")
        if launcher is None:
            return False
        from qt_gcs.planning.ingress import build_parallel_ingress

        # Ingress lane count = active UAVs (uav_count).  A 1-UAV mission then gets
        # a single lane at offset 0 -> forced straight LC->TP transit, not the
        # leftmost lane of a 6-wide formation.
        ingress_count = int(
            (self.store.mission_metadata or {}).get("uav_count")
            or len(SiteStore.VEHICLE_IDS)
        )
        ingress_count = max(1, min(ingress_count, len(SiteStore.VEHICLE_IDS)))
        formation = build_parallel_ingress(
            launcher.latitude,
            launcher.longitude,
            float(tp["latitude"]),
            float(tp["longitude"]),
            count=ingress_count,
            altitude_m=float(
                self.store.mission_metadata.get(
                    "flight_altitude_msl_m",
                    600.0,
                )
            ),
        )
        # Launch-formation: the vehicles start already spread abreast in lane
        # order (applied after load_mission in _request_launch, which otherwise
        # resets them to the single LC point).
        self._ingress_launch_positions = formation.launch_positions or {}
        prepended = False
        for vehicle_id in SiteStore.VEHICLE_IDS:
            ingress_points = formation.routes.get(vehicle_id, [])
            if not ingress_points:
                continue
            # Start the route AT the launch cluster point so the drawn route line
            # includes the fan-out leg (launch -> form-up).  Without this the
            # route began at the form-up point while the vehicle launched at the
            # compressed cluster, so its movement did not follow its drawn plan.
            launch_pos = self._ingress_launch_positions.get(vehicle_id)
            if launch_pos is not None:
                ingress_points = [
                    (launch_pos[0], launch_pos[1], launch_pos[2])
                ] + list(ingress_points)
            existing = self.store.vehicle_waypoints[vehicle_id]
            ingress_wps = []
            for seq, (lat, lon, alt) in enumerate(ingress_points, start=1):
                ingress_wps.append(
                    MissionPoint(
                        latitude=lat,
                        longitude=lon,
                        altitude_m=alt,
                        code=f"IGR{seq:03d}",
                        label=f"UAV-{vehicle_id:02d} Ingress {seq}",
                        point_type="WAYPOINT",
                        sequence=seq,
                    )
                )
            # Renumber, but keep the ingress waypoints tagged (IGR prefix +
            # point_type) so the vehicle holds ingress speed until it crosses
            # the TP line (the last ingress waypoint), not the form-up point.
            combined = ingress_wps + existing
            n_ingress = len(ingress_wps)
            for i, wp in enumerate(combined, start=1):
                wp.sequence = i
                is_ingress = i <= n_ingress
                wp.code = f"IGR{i:03d}" if is_ingress else f"WP{i:03d}"
                wp.label = (
                    f"UAV-{vehicle_id:02d} "
                    f"{'Ingress' if is_ingress else 'Waypoint'} {i}"
                )
                if is_ingress:
                    wp.point_type = "INGRESS_WAYPOINT"
            self.store.vehicle_waypoints[vehicle_id] = combined
            prepended = True
        return prepended

    def _request_launch(self) -> None:
        if self._ingress_enabled and not getattr(self, "_ingress_prepended", False):
            if self._prepend_ingress_routes():
                self._ingress_prepended = True
                # Reload routes into FlyState after prepending
                for vehicle_id, state in self.states.items():
                    state.load_mission(self.store, vehicle_id)
                # Spread the fleet into the launch formation (lane order) so it
                # flies parallel from launch instead of fanning out from LC.
                for vehicle_id, state in self.states.items():
                    pos = getattr(self, "_ingress_launch_positions", {}).get(vehicle_id)
                    if pos is not None:
                        state.vehicle.latitude = pos[0]
                        state.vehicle.longitude = pos[1]
                        state.vehicle.altitude_m = pos[2]
        launched = [
            state.request_simulated_launch()
            for state in self.states.values()
        ]
        if any(launched):
            self._initial_rhp_precompute_requested = False
            self._last_planning_cycle_s = -1e9
            self._last_planning_search_elapsed_s = -1e9
            self.statusMessage.emit(
                "UAV-01~06 일괄 이륙 // CPP TP t=0 초기 PREFIX 백그라운드 계산"
            )
        else:
            self.statusMessage.emit(
                "이륙할 수 없습니다. 임무 장입 상태 또는 기존 이륙 여부를 확인하십시오."
            )
        self._refresh_display()
        self._emit_map_state()
        if any(launched):
            self._run_planning_cycle()

    def _toggle_emergency(self) -> None:
        target_enabled = not self.state.emergency_mode
        for state in self.states.values():
            if state.emergency_mode != target_enabled:
                state.toggle_emergency()
        enabled = target_enabled
        self.statusMessage.emit(
            "비상모드 // 6기 모두 안전지대 중심으로 직선 복귀합니다."
            if enabled
            else "비상모드 해제 // 중단했던 임무 단계를 재개합니다."
        )
        self._refresh_display()
        self._emit_map_state()

    def _stop_approach_all(self) -> None:
        for state in self.states.values():
            state.stop_approach()
        self.statusMessage.emit("MANUAL 중지 // 접근 중단 후 탐색 경로 복귀")
        self._refresh_display()
        self._emit_map_state()

    def _show_split_seekers(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("탐지 카메라 화면 // UAV-01~06")
        dialog.resize(1180, 720)
        root = QVBoxLayout(dialog)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        command_row = QHBoxLayout()
        active_label = QLabel(
            f"활성 구조기 // UAV-{self._split_active_vehicle_id:02d}"
        )
        active_label.setObjectName("dataValue")
        engage_button = QPushButton("구조 접근")
        stop_button = QPushButton("중지")
        engage_button.setStyleSheet(
            """
            QPushButton {
                background:qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #713a34, stop:0.52 #4b2622, stop:1 #281512
                );
                color:#ffd2cb;
                border-top:1px solid #a65b52;
                border-left:1px solid #86453f;
                border-right:2px solid #1b0d0c;
                border-bottom:2px solid #1b0d0c;
                font-weight:700;
                padding:5px 12px;
            }
            QPushButton:hover {
                background:#7b4039;
                color:#fff0ec;
            }
            QPushButton:pressed {
                background:#321714;
            }
            """
        )
        stop_button.setStyleSheet(
            """
            QPushButton {
                background:qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #454334, stop:0.52 #2d2d24, stop:1 #181a16
                );
                color:#ddd4b2;
                border-top:1px solid #77735b;
                border-left:1px solid #625f4c;
                border-right:2px solid #10120f;
                border-bottom:2px solid #10120f;
                font-weight:700;
                padding:5px 12px;
            }
            QPushButton:hover {
                background:#55513d;
                color:#f1e7bf;
            }
            QPushButton:pressed {
                background:#202018;
            }
            """
        )
        engage_button.setMaximumWidth(90)
        stop_button.setMaximumWidth(90)
        engage_button.clicked.connect(self._split_approach_active)
        stop_button.clicked.connect(self._split_stop_active)
        command_row.addWidget(active_label)
        command_row.addStretch(1)
        command_row.addWidget(engage_button)
        command_row.addWidget(stop_button)
        root.addLayout(command_row)

        grid = QGridLayout()
        grid.setSpacing(6)
        root.addLayout(grid, 1)
        widgets: list[SearchCameraWidget] = []
        panels: dict[int, QGroupBox] = {}
        for index, vehicle_id in enumerate(SiteStore.VEHICLE_IDS):
            panel = QGroupBox(f"UAV-{vehicle_id:02d} // AUTO CAMERA")
            panel_layout = QVBoxLayout(panel)
            camera = SearchCameraWidget(self.states[vehicle_id])
            camera.setMinimumSize(300, 190)
            camera.activated.connect(
                lambda selected_id=vehicle_id: self._activate_split_vehicle(
                    selected_id
                )
            )
            panel_layout.addWidget(camera)
            grid.addWidget(panel, index // 3, index % 3)
            widgets.append(camera)
            panels[vehicle_id] = panel
        dialog._camera_widgets = widgets
        dialog._camera_panels = panels
        dialog._active_label = active_label
        refresh_timer = QTimer(dialog)
        refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        refresh_timer.setInterval(125)
        refresh_timer.timeout.connect(
            lambda: [widget.refresh() for widget in widgets]
        )
        refresh_timer.start()
        dialog._refresh_timer = refresh_timer
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.show()
        self._split_camera_dialog = dialog
        self._refresh_split_selection()

    def _activate_split_vehicle(self, vehicle_id: int) -> None:
        self._split_active_vehicle_id = int(vehicle_id)
        self.vehicleSelectionRequested.emit(self._split_active_vehicle_id)
        self._refresh_split_selection()
        self.statusMessage.emit(
            f"UAV-{vehicle_id:02d} 탐지 카메라 활성화 // MANUAL 명령 대상"
        )

    def _refresh_split_selection(self) -> None:
        dialog = getattr(self, "_split_camera_dialog", None)
        if dialog is None:
            return
        dialog._active_label.setText(
            f"활성 구조기 // UAV-{self._split_active_vehicle_id:02d}"
        )
        for vehicle_id, panel in dialog._camera_panels.items():
            panel.setStyleSheet(
                "QGroupBox { border:2px solid #9cff00; color:#9cff00; }"
                if vehicle_id == self._split_active_vehicle_id
                else "QGroupBox { border:1px solid #3d4a40; color:#b7c1b8; }"
            )

    def _split_approach_active(self) -> None:
        vehicle_id = self._split_active_vehicle_id
        state = self.states[vehicle_id]
        if not state.mission_launched:
            self.statusMessage.emit(
                f"UAV-{vehicle_id:02d} 미이륙 // 접근 명령 거부"
            )
            return
        subject_id = state.selected_track_id
        state.designate_subject(subject_id)
        self.statusMessage.emit(
            f"MANUAL 접근 명령 // UAV-{vehicle_id:02d} → PSN-{subject_id}"
        )

    def _split_stop_active(self) -> None:
        vehicle_id = self._split_active_vehicle_id
        started = self.states[vehicle_id].request_safe_return_via_waypoint()
        self.statusMessage.emit(
            (
                f"MANUAL 중지 명령 // UAV-{vehicle_id:02d} 최근접 전용 "
                "웨이포인트 경유 후 안전구역 복귀"
            )
            if started
            else f"UAV-{vehicle_id:02d} 복귀 명령 거부 // 미이륙 또는 비상모드"
        )
        self._refresh_display()
        self._emit_map_state()

    @staticmethod
    def _copy_route(route: list[dict]) -> list[dict]:
        return [
            {
                "latitude": float(point["latitude"]),
                "longitude": float(point["longitude"]),
                "altitude_m": max(0.0, float(point.get("altitude_m", 600.0))),
                "code": str(point.get("code") or f"MWP{index:03d}"),
            }
            for index, point in enumerate(route, start=1)
        ]

    def _effective_route_for_editing(self, vehicle_id: int) -> list[dict]:
        pending = self._pending_runtime_routes.get(vehicle_id)
        if pending is not None:
            return self._copy_route(pending)
        state = self.states[vehicle_id]
        runtime = state.runtime_route_payload()
        if runtime:
            return self._copy_route(runtime)
        return [
            {
                "latitude": point.latitude,
                "longitude": point.longitude,
                "altitude_m": point.altitude_m,
                "code": point.code,
            }
            for point in self.store.waypoints_for(vehicle_id)
        ]

    @staticmethod
    def _renumber_manual_route(route: list[dict]) -> None:
        for sequence, point in enumerate(route, start=1):
            point["code"] = f"MWP{sequence:03d}"

    def _show_waypoint_context_menu(self, position=None) -> None:
        if not self._active:
            return
        if time.monotonic() < self._context_menu_block_until:
            return

        remembered = self._pending_context_position
        if remembered is not None and time.monotonic() - remembered[3] <= 2.5:
            latitude, longitude, altitude = remembered[:3]
        else:
            display_vehicle_id = (
                1 if self.selected_vehicle_id == 0 else self.selected_vehicle_id
            )
            vehicle = self.states[display_vehicle_id].vehicle
            latitude, longitude = destination_position(
                vehicle.latitude,
                vehicle.longitude,
                850.0,
                vehicle.heading_deg,
            )
            altitude = max(600.0, vehicle.altitude_m)

        menu = QMenu(self)
        action_commands: dict[object, tuple[str, int]] = {}
        nearest_by_vehicle = {}
        remembered_feature = self._pending_context_feature
        context_key = (
            remembered_feature[0]
            if remembered_feature is not None
            and time.monotonic() - remembered_feature[1] <= 2.5
            else None
        )
        vehicle_ids = (
            SiteStore.VEHICLE_IDS
            if self.selected_vehicle_id == 0
            else (self.selected_vehicle_id,)
        )
        if any(
            self.states[vehicle_id].approach_requested
            for vehicle_id in vehicle_ids
        ):
            self.statusMessage.emit(
                "AUTO 접근 중에는 탐색 웨이포인트를 수정할 수 없습니다."
            )
            return
        editable_routes = {
            vehicle_id: self._effective_route_for_editing(vehicle_id)
            for vehicle_id in vehicle_ids
        }
        for vehicle_id in vehicle_ids:
            route = editable_routes[vehicle_id]
            exact_index = runtime_route_index_for_context_key(
                route,
                vehicle_id,
                context_key,
            )
            nearest_index = (
                nearest_runtime_route_index_within(
                    route,
                    latitude,
                    longitude,
                )
                if route and context_key is None
                else None
            )
            nearest_by_vehicle[vehicle_id] = (
                exact_index if exact_index is not None else nearest_index
            )

        if self.selected_vehicle_id == 0:
            add_menu = menu.addMenu("웨이포인트 추가")
            delete_menu = menu.addMenu("웨이포인트 삭제")
            for vehicle_id in vehicle_ids:
                add_action = add_menu.addAction(f"UAV-{vehicle_id:02d}")
                delete_action = delete_menu.addAction(f"UAV-{vehicle_id:02d}")
                delete_action.setEnabled(
                    nearest_by_vehicle[vehicle_id] is not None
                    and len(editable_routes[vehicle_id]) > 1
                )
                action_commands[add_action] = ("ADD", vehicle_id)
                action_commands[delete_action] = ("DELETE", vehicle_id)
        else:
            vehicle_id = self.selected_vehicle_id
            menu.setTitle(f"UAV-{vehicle_id:02d} 경로 편집")
            add_action = menu.addAction("웨이포인트 추가")
            delete_action = menu.addAction("웨이포인트 삭제")
            delete_action.setEnabled(
                nearest_by_vehicle[vehicle_id] is not None
                and len(editable_routes[vehicle_id]) > 1
            )
            action_commands[add_action] = ("ADD", vehicle_id)
            action_commands[delete_action] = ("DELETE", vehicle_id)
        source = self.sender()
        global_position = QCursor.pos()
        if position is not None and isinstance(source, QWidget):
            global_position = source.mapToGlobal(position)
        selected_action = menu.exec(global_position)
        self._context_menu_block_until = time.monotonic() + 0.45
        if selected_action is None or selected_action not in action_commands:
            return
        command, vehicle_id = action_commands[selected_action]
        route = editable_routes[vehicle_id]
        if vehicle_id not in self._pending_runtime_routes:
            self._pending_runtime_originals[vehicle_id] = self._copy_route(route)
        delete_index = nearest_by_vehicle[vehicle_id]
        if command == "ADD":
            route.append(
                {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "altitude_m": max(600.0, float(altitude)),
                    "code": "",
                }
            )
            self._renumber_manual_route(route)
            point = route[-1]
            self._pending_runtime_routes[vehicle_id] = route
            self._pending_dirty = True
            self._plan_render_revision += 1
            self._cached_plan_key = None
            self.statusMessage.emit(
                f"UAV-{vehicle_id:02d} {point['code']} 추가 대기 // "
                "'임무지도 수정'을 누르면 반영"
            )
        elif command == "DELETE" and delete_index is not None and len(route) > 1:
            deleted = str(route[delete_index].get("code", delete_index + 1))
            route.pop(delete_index)
            self._renumber_manual_route(route)
            self._pending_runtime_routes[vehicle_id] = route
            self._pending_dirty = True
            self._plan_render_revision += 1
            self._cached_plan_key = None
            self.statusMessage.emit(
                f"UAV-{vehicle_id:02d} {deleted} 삭제 대기 // "
                "'임무지도 수정'을 누르면 반영"
            )
        self._pending_context_feature = None
        self._emit_map_state()

    def _remember_map_context_position(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        self._pending_context_position = (
            latitude,
            longitude,
            altitude,
            time.monotonic(),
        )

    def _on_map_right_clicked(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        """Open the Qt waypoint menu when the web map reports a right-click."""
        self._pending_context_feature = None
        self._remember_map_context_position(latitude, longitude, altitude)
        QTimer.singleShot(0, self._show_waypoint_context_menu)

    def _on_waypoint_right_clicked(
        self,
        feature_key: str,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        """Open a menu tied to one exact vehicle waypoint marker."""
        self._remember_map_context_position(latitude, longitude, altitude)
        self._pending_context_feature = (
            str(feature_key),
            time.monotonic(),
        )
        QTimer.singleShot(0, self._show_waypoint_context_menu)

    def _apply_pending_mission(self) -> None:
        if not self._pending_dirty:
            self.statusMessage.emit("대기 중인 웨이포인트 수정이 없습니다.")
            return
        if self._pending_runtime_routes:
            hold_duration_s = max(
                1.0,
                float(
                    self.store.mission_metadata.get(
                        "manual_route_hold_s",
                        MANUAL_ROUTE_HOLD_S,
                    )
                ),
            )
            applied_vehicle_ids: list[int] = []
            for vehicle_id, route in sorted(self._pending_runtime_routes.items()):
                state = self.states[vehicle_id]
                if not state.apply_manual_runtime_route(
                    route,
                    hold_duration_s=hold_duration_s,
                ):
                    continue
                applied_vehicle_ids.append(vehicle_id)
                self.store.vehicle_waypoints[vehicle_id] = [
                    MissionPoint(
                        latitude=float(point["latitude"]),
                        longitude=float(point["longitude"]),
                        altitude_m=max(
                            0.0,
                            float(point.get("altitude_m", 600.0)),
                        ),
                        code=f"WP{sequence:03d}",
                        label=f"UAV-{vehicle_id:02d} Waypoint {sequence}",
                        point_type="WAYPOINT",
                        sequence=sequence,
                    )
                    for sequence, point in enumerate(route, start=1)
                ]
            if not applied_vehicle_ids:
                self.statusMessage.emit("적용 가능한 웨이포인트 수정이 없습니다.")
                return
            self._pending_dirty = False
            self._pending_runtime_routes.clear()
            self._pending_runtime_originals.clear()
            self._manual_route_applied_vehicle_ids = set(applied_vehicle_ids)
            self._preserve_execution_on_next_plan_change = True
            self.store.notify()
            vehicles_text = ", ".join(
                f"UAV-{vehicle_id:02d}" for vehicle_id in applied_vehicle_ids
            )
            self.statusMessage.emit(
                f"Mission Map 수동 경로 반영 // {vehicles_text} // "
                f"자동 RHP 덮어쓰기 {hold_duration_s:g}초 보호"
            )
            return
        self._pending_dirty = False
        self._preserve_execution_on_next_plan_change = True
        self.store.replace_from(self._pending_mission)
        self.statusMessage.emit(
            "Mission Map 웨이포인트 수정 반영 완료 // 6기 임무 재동기화"
        )

    @staticmethod
    def _set_lamp(label: QLabel, enabled: bool) -> None:
        label.setStyleSheet(
            f"color:{GREEN if enabled else RED}; font-size:14pt; border:0;"
        )

    def _refresh_display(self, *, refresh_camera: bool = True) -> None:
        self._refresh_subject_table()
        self._refresh_information()

        for name, ready in self.state.readiness.items():
            self._set_lamp(self.readiness_lamps[name], ready)
        ready = self.state.launch_ready and not self.state.mission_launched
        if self.state.mission_launched:
            readiness_text = "● 이륙 완료"
            readiness_green = True
        else:
            readiness_text = "● 이륙 가능" if ready else "● 이륙 불가"
            readiness_green = ready
        self.launch_ready_label.setText(readiness_text)
        self.launch_ready_label.setStyleSheet(
            f"color:{GREEN if readiness_green else RED}; background:"
            f"{'#0b1c10' if readiness_green else '#120a08'}; border:1px solid "
            f"{'#2e7640' if readiness_green else '#5f2824'}; padding:5px;"
        )

        for name, complete in self.state.mission_status.items():
            self._set_lamp(self.mission_lamps[name], complete)

        mode = self.state.automatic_mode
        self.mode_label.setText(mode)
        self.mode_label.setStyleSheet(
            f"color:{GREEN if mode == 'ARM' else AMBER}; background:"
            f"{'#0d2c15' if mode == 'ARM' else '#251c0d'}; border:1px solid "
            f"{'#2e7640' if mode == 'ARM' else '#715822'}; padding:7px;"
        )
        self.launch_button.setEnabled(self.state.can_press_launch)
        self.launch_button.setText(
            "이륙 완료" if self.state.mission_launched else "이륙"
        )
        self.launch_button.setStyleSheet(
            (
                f"background:#12351c; color:{GREEN}; border:1px solid #2e7640;"
                if self.state.can_press_launch
                else "background:#111713; color:#586259; border:1px solid #29322b;"
            )
            + "font-weight:700; padding:7px;"
        )
        self.emergency_button.setText(
            "비상 해제" if self.state.emergency_mode else "비상모드"
        )
        if refresh_camera:
            self.map_stage.refresh_camera()

    def select_vehicle(self, vehicle_id: int) -> None:
        self.selected_vehicle_id = int(vehicle_id)
        display_id = 1 if vehicle_id == 0 else int(vehicle_id)
        self.state = self.states[display_id]
        self.map_stage.search_camera_video.state = self.state
        if isinstance(self.map_bridge, FlyMapBridge):
            self.map_bridge.state = self.state
        self._refresh_display()
        self._emit_map_state()

    def _refresh_subject_table(self) -> None:
        self._refreshing_table = True
        blocker = QSignalBlocker(self.subject_table)
        self.subject_table.setUpdatesEnabled(False)
        try:
            self.subject_table.setRowCount(len(self.state.subjects))
            selected_row = -1
            for row, track in enumerate(self.state.subjects):
                values = (
                    (
                        f"{track.track_id}/{track.subject_type}-{track.country}"
                        + (" X" if track.found else "")
                    ),
                    f"{track.speed_mps:.0f} m/s",
                    f"{track.heading_deg:03.0f}°",
                    track.first_tracked_text,
                )
                for column, value in enumerate(values):
                    item = self.subject_table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.subject_table.setItem(row, column, item)
                    item.setText(value)
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, track.track_id)
                if track.track_id == self.state.selected_track_id:
                    selected_row = row
            if selected_row >= 0:
                self.subject_table.selectRow(selected_row)
        finally:
            self.subject_table.setUpdatesEnabled(True)
            del blocker
            self._refreshing_table = False
    def _refresh_information(self) -> None:
        vehicle = self.state.vehicle
        self.vehicle_title_label.setText(vehicle.code)
        vehicle_values = {
            "altitude": f"{vehicle.altitude_m:,.1f} m",
            "latitude": f"{vehicle.latitude:.5f}",
            "longitude": f"{vehicle.longitude:.5f}",
            "speed": f"{vehicle.speed_mps:.1f} m/s",
            "heading": f"{vehicle.heading_deg:03.0f}°",
        }
        for name, value in vehicle_values.items():
            self.vehicle_values[name].setText(value)

        subject = self.state.selected_subject
        distance_m = (
            horizontal_distance_m(
                vehicle.latitude,
                vehicle.longitude,
                subject.latitude,
                subject.longitude,
            )
            if subject
            else 0.0
        )
        eta_s = distance_m / SAR_CRUISE_SPEED_MPS if subject else 0.0
        eta_text = (
            f"{int(eta_s // 60):02d}:{int(eta_s % 60):02d}"
            if subject
            else "--"
        )
        subject_values = {
            "altitude": f"{subject.altitude_m:,.0f} m" if subject else "--",
            "latitude": f"{subject.latitude:.5f}" if subject else "--",
            "longitude": f"{subject.longitude:.5f}" if subject else "--",
            "speed": f"{subject.speed_mps:.1f} m/s" if subject else "--",
            "heading": f"{subject.heading_deg:03.0f}°" if subject else "--",
            "distance": f"{distance_m / 1000:.1f} km" if subject else "--",
            "eta": eta_text,
        }
        vehicle_values["distance"] = (
            f"{distance_m / 1000:.1f} km" if subject else "--"
        )
        vehicle_values["eta"] = eta_text
        for name, value in vehicle_values.items():
            self.vehicle_values[name].setText(value)
        for name, value in subject_values.items():
            self.subject_values[name].setText(value)
