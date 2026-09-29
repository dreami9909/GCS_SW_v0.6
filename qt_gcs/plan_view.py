from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, QUrl, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .map_bridge import MapBridge
from .map_html import load_map_html
from .site_store import MissionPoint, MissionZone, SitePosition, SiteStore


class Plan3DView(QWidget):
    statusMessage = Signal(str)

    POINT_TOOLS = {"GCS", "RDR", "LC", "WP"}
    ZONE_TOOLS = {"SAFE"}

    def __init__(
        self,
        store: SiteStore,
        loaded_store: SiteStore | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.loaded_store = loaded_store if loaded_store is not None else store
        self.placement_code: str | None = None
        self.selected_code: str | None = None
        self.vehicle_editing_enabled = False
        self.tool_buttons: dict[str, QPushButton] = {}
        self._refreshing_table = False
        self.map_ready = False
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
        self.map_html, self.provider_name = load_map_html(self.api_key)

        self._build_ui()
        self.set_vehicle_editing_enabled(False)
        self._connect_map()
        self.store.subscribe(self.refresh_table)
        self.refresh_table()

    def _build_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(1)

        content = QWidget()
        layout = QHBoxLayout(content)
        self.page_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self._build_tool_panel())

        self.web_view = QWebEngineView()
        self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(
            QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled,
            True,
        )
        layout.addWidget(self.web_view, 1)
        layout.addWidget(self._build_editor_panel())
        outer_layout.addWidget(content, 1)
        outer_layout.addWidget(self._build_map_navigation_bar())

    def _build_map_navigation_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("mapNavigationBar")
        bar.setStyleSheet(
            """
            QFrame#mapNavigationBar {
                background:#0b110d;
                border-top:1px solid #4b574b;
                border-bottom:1px solid #030504;
            }
            """
        )
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(7)

        title = QLabel("지도 위치 이동")
        title.setObjectName("panelTitle")
        title.setStyleSheet(
            "font-size:10pt; padding:2px 8px 2px 2px; border:0;"
        )
        layout.addWidget(title)

        self.map_latitude_input = self._coordinate_spin(-90.0, 90.0, 7)
        self.map_longitude_input = self._coordinate_spin(-180.0, 180.0, 7)
        self.map_altitude_input = self._coordinate_spin(-500.0, 20_000.0, 1)
        self.map_altitude_input.setSuffix(" m")
        self.map_latitude_input.setValue(37.3422)
        self.map_longitude_input.setValue(127.9202)
        self.map_altitude_input.setValue(250.0)

        for label_text, control in (
            ("위도", self.map_latitude_input),
            ("경도", self.map_longitude_input),
            ("고도", self.map_altitude_input),
        ):
            label = QLabel(label_text)
            label.setObjectName("fieldCaption")
            layout.addWidget(label)
            control.setMinimumWidth(145 if label_text != "고도" else 110)
            layout.addWidget(control)
            control.lineEdit().returnPressed.connect(self._move_map_to_inputs)

        move_button = QPushButton("지도 이동")
        move_button.setObjectName("primaryButton")
        move_button.clicked.connect(self._move_map_to_inputs)
        layout.addWidget(move_button)
        layout.addStretch(1)
        return bar

    def attach_map_view(self) -> None:
        """Move the single shared map surface back into the PLAN page."""
        self.page_layout.insertWidget(1, self.web_view, 1)
        self.web_view.show()
        # The PLAN view owns the map again: show the certified SPX plan (undo
        # the FLY view's pre-baked-route hiding).
        self.bridge.hide_preplanned_routes = False
        self.web_view.page().runJavaScript(
            "window.setMapMode && window.setMapMode('PLAN');"
        )
        self._set_plan_vehicle_selection(
            self.store.active_vehicle_id if self.vehicle_editing_enabled else 0
        )

    def _set_plan_vehicle_selection(self, vehicle_id: int) -> None:
        if not self.map_ready:
            return
        self.web_view.page().runJavaScript(
            "window.setPlanVehicleSelection && "
            f"window.setPlanVehicleSelection({int(vehicle_id)});"
        )

    def _build_tool_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(245)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 12, 10, 10)
        panel_layout.setSpacing(6)

        title = QLabel("PLAN")
        title.setObjectName("panelTitle")
        panel_layout.addWidget(title)

        note = QLabel("임무장입 설정")
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        panel_layout.addWidget(note)

        self.vehicle_route_label = QLabel("")

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self._add_tool_section(
            panel_layout,
            "시설",
            (
                ("GCS", "▣  GCS 배치"),
                ("RDR", "◇  레이다 배치"),
                ("LC", "△  이륙지점 배치"),
            ),
        )
        self._add_tool_section(
            panel_layout,
            "비행계획",
            (
                ("WP", "●  웨이포인트 추가"),
            ),
        )
        self.waypoint_delete_button = QPushButton("웨이포인트 삭제")
        self.waypoint_delete_button.setToolTip(
            "지도 또는 목록에서 활성화한 웨이포인트를 삭제합니다."
        )
        self.waypoint_delete_button.clicked.connect(self._delete_point)
        self.waypoint_delete_button.setEnabled(False)
        panel_layout.addWidget(self.waypoint_delete_button)
        # Compatibility alias used by the smoke test and older integrations.
        self.undo_vertex_button = self.waypoint_delete_button
        self._add_tool_section(
            panel_layout,
            "작전구역",
            (
                ("SAFE", "▱  안전지대 설정"),
            ),
        )

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        panel_layout.addWidget(divider)

        load_button = QPushButton("임무 장입")
        load_button.setObjectName("primaryButton")
        load_button.clicked.connect(self._load_mission)
        panel_layout.addWidget(load_button)
        clear_button = QPushButton("임무 삭제")
        clear_button.clicked.connect(self._delete_mission)
        panel_layout.addWidget(clear_button)

        panel_layout.addStretch(1)
        provider_title = QLabel("MAP PROVIDER")
        provider_title.setObjectName("fieldCaption")
        panel_layout.addWidget(provider_title)
        self.provider_label = QLabel(self.provider_name)
        self.provider_label.setObjectName(
            "providerOnline" if self.api_key else "providerOffline"
        )
        self.provider_label.setWordWrap(True)
        panel_layout.addWidget(self.provider_label)

        return panel

    def _add_tool_section(
        self,
        layout: QVBoxLayout,
        caption: str,
        tools: tuple[tuple[str, str], ...],
    ) -> None:
        label = QLabel(caption.upper())
        label.setObjectName("fieldCaption")
        layout.addWidget(label)
        for code, text in tools:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setProperty("siteTool", True)
            button.clicked.connect(
                lambda checked, value=code: self._select_placement_tool(
                    value if checked else None
                )
            )
            self.tool_group.addButton(button)
            self.tool_buttons[code] = button
            layout.addWidget(button)

    def _build_editor_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("editorPanel")
        panel.setFixedWidth(395)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 12, 10, 10)
        panel_layout.setSpacing(8)

        title = QLabel("MISSION CONFIGURATION")
        title.setObjectName("panelTitle")
        panel_layout.addWidget(title)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(("CODE", "TYPE", "LAT", "LON", "ALT/VTX"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        panel_layout.addWidget(self.table, 1)

        editor = QGroupBox("요소 상세")
        form = QFormLayout(editor)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.code_value = QLabel("--")
        self.code_value.setObjectName("dataValue")
        form.addRow("요소", self.code_value)
        self.latitude_input = self._coordinate_spin(-90.0, 90.0, 7)
        self.longitude_input = self._coordinate_spin(-180.0, 180.0, 7)
        self.altitude_input = self._coordinate_spin(0.0, 20_000.0, 1)
        self.altitude_input.setSuffix(" m")
        form.addRow("위도", self.latitude_input)
        form.addRow("경도", self.longitude_input)
        form.addRow("고도", self.altitude_input)
        panel_layout.addWidget(editor)

        edit_row = QHBoxLayout()
        self.apply_button = QPushButton("좌표 적용")
        self.apply_button.setObjectName("primaryButton")
        self.apply_button.clicked.connect(self._apply_fields)
        self.delete_selected_button = QPushButton("선택 삭제")
        self.delete_selected_button.clicked.connect(self._delete_selected)
        edit_row.addWidget(self.apply_button)
        edit_row.addWidget(self.delete_selected_button)
        panel_layout.addLayout(edit_row)

        file_row = QHBoxLayout()
        new_button = QPushButton("새 임무")
        new_button.clicked.connect(self._new_plan)
        open_button = QPushButton("임무 열기")
        open_button.clicked.connect(self._open_config)
        save_button = QPushButton("임무 저장")
        save_button.clicked.connect(self._save_config)
        file_row.addWidget(new_button)
        file_row.addWidget(open_button)
        file_row.addWidget(save_button)
        panel_layout.addLayout(file_row)

        self.editor_status = QLabel("왼쪽에서 장입할 임무 요소를 선택하십시오.")
        self.editor_status.setObjectName("statusLine")
        self.editor_status.setWordWrap(True)
        panel_layout.addWidget(self.editor_status)
        return panel

    @staticmethod
    def _coordinate_spin(
        minimum: float,
        maximum: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        asset_dir = Path(__file__).resolve().parent / "assets"
        spin_style = (
            """
            QDoubleSpinBox {
                padding-right:32px;
            }
            QDoubleSpinBox::up-button {
                subcontrol-origin:border;
                subcontrol-position:top right;
                width:28px;
                background:#1b281e;
                border-left:1px solid #526054;
                border-bottom:1px solid #303a31;
            }
            QDoubleSpinBox::down-button {
                subcontrol-origin:border;
                subcontrol-position:bottom right;
                width:28px;
                background:#121b14;
                border-left:1px solid #526054;
                border-top:1px solid #303a31;
            }
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background:#344538;
            }
            QDoubleSpinBox::up-button:pressed,
            QDoubleSpinBox::down-button:pressed {
                background:#72551b;
            }
            QDoubleSpinBox::up-arrow {
                image:url(__UP_ARROW__);
                width:12px;
                height:8px;
            }
            QDoubleSpinBox::down-arrow {
                image:url(__DOWN_ARROW__);
                width:12px;
                height:8px;
            }
            """
        )
        spin.setStyleSheet(
            spin_style
            .replace(
                "__UP_ARROW__",
                (asset_dir / "spin_up.svg").as_posix(),
            )
            .replace(
                "__DOWN_ARROW__",
                (asset_dir / "spin_down.svg").as_posix(),
            )
        )
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.0001 if decimals > 2 else 1.0)
        spin.setKeyboardTracking(False)
        return spin

    def _connect_map(self) -> None:
        self.bridge = MapBridge(self.store)
        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)
        self.bridge.mapClicked.connect(self._on_map_clicked)
        self.bridge.featureSelected.connect(self._select_feature)
        self.bridge.mapStatusChanged.connect(self._map_status)
        self.web_view.loadFinished.connect(self._on_map_load_finished)
        self.web_view.setHtml(self.map_html, QUrl("https://localhost/"))

    def _on_map_load_finished(self, success: bool) -> None:
        self.map_ready = success
        if success:
            self.bridge.emit_plan()
            self._set_plan_vehicle_selection(
                self.store.active_vehicle_id
                if self.vehicle_editing_enabled
                else 0
            )
            self._set_status(f"{self.provider_name} map surface ready.")
        else:
            self._set_status("3D map surface failed to load.", error=True)

    def _select_placement_tool(self, code: str | None) -> None:
        if code == "WP" and not self.vehicle_editing_enabled:
            button = self.tool_buttons.get("WP")
            if button is not None:
                button.setChecked(False)
            self.placement_code = None
            self._set_status(
                "공통 설정 완료 후 상단의 기체 번호 1~6 중 하나를 "
                "활성화하십시오.",
                error=True,
            )
            return
        self.placement_code = code
        if code in self.ZONE_TOOLS:
            self.store.begin_zone(code)
            self._set_status(
                "안전지대: 지도에서 경계점 3개를 클릭하면 자동으로 확정됩니다."
            )
        elif code:
            self.store.cancel_draft_zone()
            self._set_status(f"{code} 도구 선택됨: 3D 지도에서 위치를 클릭하십시오.")

    def _on_map_clicked(
        self,
        latitude: float,
        longitude: float,
        map_altitude: float,
    ) -> None:
        code = self.placement_code
        if code is None:
            self._set_status("먼저 왼쪽에서 장입 도구를 선택하십시오.")
            return

        if code == "WP" and not self.vehicle_editing_enabled:
            self._set_status(
                "상단 기체 번호를 활성화한 뒤 웨이포인트를 입력하십시오.",
                error=True,
            )
            return

        if code in {"GCS", "RDR", "LC"}:
            previous = self.store.sites.get(code)
            altitude = previous.altitude_m if previous else max(0.0, map_altitude)
            feature = self.store.set_site(code, latitude, longitude, altitude)
            self._select_feature(feature.code)
        elif code == "WP":
            feature = self.store.add_waypoint(latitude, longitude, 60.0)
            self._select_feature(feature.code)
        elif code in self.ZONE_TOOLS:
            if self.store.draft_zone_type != code:
                self.store.begin_zone(code)
            self.store.add_draft_vertex(latitude, longitude)
            count = len(self.store.draft_vertices)
            if count >= 3:
                self._finish_zone()
                return
            self._set_status(
                f"{code} 경계점 {count}개 입력됨 // {3 - count}개 더 필요"
            )
            return
        else:
            return

        self._set_status(
            f"{feature.code} 위치 장입: {feature.latitude:.6f}, "
            f"{feature.longitude:.6f}"
        )

    def _delete_point(self) -> None:
        selected = (
            self.store.get_feature(self.selected_code)
            if self.selected_code
            else None
        )
        waypoint = selected if isinstance(selected, MissionPoint) else None
        if not self.vehicle_editing_enabled or waypoint is None:
            self._set_status(
                "삭제할 웨이포인트를 지도 또는 목록에서 먼저 선택하십시오.",
                error=True,
            )
            return

        deleted_code = waypoint.code
        self.store.remove_feature(deleted_code)
        self.selected_code = None
        self.table.clearSelection()
        self.code_value.setText("--")
        self._refresh_point_controls()
        self._set_status(
            f"{deleted_code} 웨이포인트 삭제 완료 // "
            f"{len(self.store.waypoints)}개 남음"
        )

    def _finish_zone(self) -> None:
        try:
            zone = self.store.commit_zone()
        except ValueError as error:
            count = len(self.store.draft_vertices)
            if self.store.draft_zone_type is None:
                message = "먼저 '안전지대 설정'을 선택하십시오."
            else:
                message = (
                    f"경계점이 {count}개입니다. 설정 완료에는 최소 3개가 필요합니다."
                )
            self._set_status(message, error=True)
            return
        self.placement_code = None
        self.tool_group.setExclusive(False)
        for button in self.tool_group.buttons():
            button.setChecked(False)
        self.tool_group.setExclusive(True)
        self._select_feature(zone.code)
        self._set_status(f"{zone.label} 설정 완료 // 경계점 {len(zone.vertices)}개")

    def _map_status(self, message: str) -> None:
        self._set_status(message)

    def _move_map_to_inputs(self) -> None:
        if not self.map_ready:
            self._set_status(
                "지도가 아직 준비되지 않았습니다. 잠시 후 다시 시도하십시오.",
                error=True,
            )
            return
        latitude = self.map_latitude_input.value()
        longitude = self.map_longitude_input.value()
        altitude = self.map_altitude_input.value()
        camera_payload = json.dumps(
            {
                "latitude": latitude,
                "longitude": longitude,
                "altitude_m": altitude,
            },
            separators=(",", ":"),
        )
        self._set_status(
            f"지도 이동 요청: {latitude:.7f}, {longitude:.7f}, {altitude:.1f} m"
        )
        self.web_view.page().runJavaScript(
            f"window.moveMapCamera && window.moveMapCamera({camera_payload});",
            lambda moved: self._map_move_finished(
                bool(moved),
                latitude,
                longitude,
                altitude,
            ),
        )

    def _map_move_finished(
        self,
        moved: bool,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> None:
        if moved:
            self._set_status(
                f"지도 중심 이동 완료: {latitude:.7f}, "
                f"{longitude:.7f}, {altitude:.1f} m"
            )
        else:
            self._set_status(
                "지도가 아직 준비되지 않았습니다. 잠시 후 다시 시도하십시오.",
                error=True,
            )

    def refresh_table(self) -> None:
        selected = self.selected_code
        if self.vehicle_editing_enabled:
            features = [
                (feature.code, feature)
                for feature in self.store.iter_features()
            ]
        else:
            features = [
                (site.code, site)
                for site in self.store.sites.values()
            ]
            for vehicle_id in SiteStore.VEHICLE_IDS:
                features.extend(
                    (
                        f"UAV-{vehicle_id:02d}/{waypoint.code}",
                        waypoint,
                    )
                    for waypoint in self.store.waypoints_for(vehicle_id)
                )
            features.extend((zone.code, zone) for zone in self.store.zones)
        self._refreshing_table = True
        self.table.setUpdatesEnabled(False)
        blocker = QSignalBlocker(self.table)
        try:
            self.table.setRowCount(len(features))
            selected_row = -1
            for row, (display_code, feature) in enumerate(features):
                if isinstance(feature, MissionZone):
                    center = feature.center()
                    values = (
                        display_code,
                        feature.label,
                        f"{center.latitude:.5f}",
                        f"{center.longitude:.5f}",
                        f"{len(feature.vertices)} VTX",
                    )
                else:
                    values = (
                        display_code,
                        feature.label,
                        f"{feature.latitude:.5f}",
                        f"{feature.longitude:.5f}",
                        f"{feature.altitude_m:.0f}",
                    )
                for column, value in enumerate(values):
                    item = self.table.item(row, column)
                    if item is None:
                        item = QTableWidgetItem()
                        item.setForeground(QColor("#c5d7c7"))
                        self.table.setItem(row, column, item)
                    item.setText(value)
                if selected == display_code:
                    selected_row = row
            if selected_row >= 0:
                self.table.selectRow(selected_row)
        finally:
            del blocker
            self.table.setUpdatesEnabled(True)
            self._refreshing_table = False

        if self.vehicle_editing_enabled:
            self.vehicle_route_label.setText(
                f"{self.store.active_vehicle_code} ROUTE EDIT // "
                f"WP {len(self.store.waypoints)}"
            )
        self._refresh_point_controls()

    def set_vehicle_editing_enabled(self, enabled: bool) -> None:
        self.vehicle_editing_enabled = bool(enabled)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
            if self.vehicle_editing_enabled
            else QAbstractItemView.SelectionMode.NoSelection
        )
        waypoint_button = self.tool_buttons.get("WP")
        if waypoint_button is not None:
            waypoint_button.setEnabled(self.vehicle_editing_enabled)
            if not self.vehicle_editing_enabled and waypoint_button.isChecked():
                waypoint_button.setChecked(False)
                self.placement_code = None
        if self.vehicle_editing_enabled:
            self.vehicle_route_label.setText(
                f"{self.store.active_vehicle_code} ROUTE EDIT // "
                f"WP {len(self.store.waypoints)}"
            )
        else:
            self.vehicle_route_label.clear()
            self.table.clearSelection()
        self._refresh_point_controls()

    def on_active_vehicle_changed(self, vehicle_id: int) -> None:
        if self.store.active_vehicle_id != vehicle_id:
            self.store.set_active_vehicle(vehicle_id)
        self.selected_code = None
        self.table.clearSelection()
        self.code_value.setText("--")
        self.set_vehicle_editing_enabled(True)
        self._set_plan_vehicle_selection(vehicle_id)
        self.refresh_table()
        self._set_status(
            f"{self.store.active_vehicle_code} 경로 편집 활성화 // "
            f"웨이포인트 {len(self.store.waypoints)}개"
        )

    def show_fleet_overview(self) -> None:
        self.selected_code = None
        self.table.clearSelection()
        self.code_value.setText("--")
        self.set_vehicle_editing_enabled(False)
        self.vehicle_route_label.clear()
        self._set_plan_vehicle_selection(0)
        self.refresh_table()

    def _refresh_point_controls(self) -> None:
        selected = (
            self.store.get_feature(self.selected_code)
            if self.selected_code
            else None
        )
        selected_waypoint = (
            selected
            if isinstance(selected, MissionPoint)
            and selected.point_type == "WAYPOINT"
            else None
        )
        self.waypoint_delete_button.setText(
            (
                f"웨이포인트 삭제 ({selected_waypoint.code})"
                if selected_waypoint is not None
                else "웨이포인트 삭제"
            )
        )
        self.waypoint_delete_button.setToolTip(
            (
                f"선택한 {selected_waypoint.code} 웨이포인트를 삭제합니다."
                if selected_waypoint is not None
                else "지도 또는 목록에서 삭제할 웨이포인트를 선택하십시오."
            )
        )
        self.waypoint_delete_button.setEnabled(
            self.vehicle_editing_enabled and selected_waypoint is not None
        )

    def _on_table_selection(self) -> None:
        if self._refreshing_table or not self.vehicle_editing_enabled:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is not None:
            self._select_feature(item.text())

    def _select_feature(self, code: str) -> None:
        feature = self.store.get_feature(code)
        if feature is None:
            return
        if isinstance(feature, MissionPoint) and not self.vehicle_editing_enabled:
            self._set_status(
                "전체 경로는 읽기 전용입니다. 상단에서 SAR 번호를 선택해 편집하십시오."
            )
            return
        self.selected_code = code
        self._load_fields(feature)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == code:
                self._refreshing_table = True
                self.table.selectRow(row)
                self._refreshing_table = False
                break
        self._refresh_point_controls()

    def _load_fields(
        self,
        feature: SitePosition | MissionPoint | MissionZone,
    ) -> None:
        self.code_value.setText(f"{feature.code} // {feature.label}")
        is_point = not isinstance(feature, MissionZone)
        self.latitude_input.setEnabled(is_point)
        self.longitude_input.setEnabled(is_point)
        self.altitude_input.setEnabled(is_point)
        self.apply_button.setEnabled(is_point)
        if is_point:
            self.latitude_input.setValue(feature.latitude)
            self.longitude_input.setValue(feature.longitude)
            self.altitude_input.setValue(feature.altitude_m)
        else:
            center = feature.center()
            self.latitude_input.setValue(center.latitude)
            self.longitude_input.setValue(center.longitude)
            self.altitude_input.setValue(0.0)

    def _apply_fields(self) -> None:
        if self.selected_code is None:
            self._set_status("수정할 위치를 먼저 선택하십시오.", error=True)
            return
        try:
            feature = self.store.update_point(
                self.selected_code,
                self.latitude_input.value(),
                self.longitude_input.value(),
                self.altitude_input.value(),
            )
        except ValueError as error:
            self._set_status(str(error), error=True)
            return
        self._select_feature(feature.code)
        self._set_status(f"{feature.code} 좌표와 고도를 수정했습니다.")

    def _delete_selected(self) -> None:
        if self.selected_code is None:
            return
        code = self.selected_code
        if self.store.remove_feature(code):
            self.selected_code = None
            self.code_value.setText("--")
            self._set_status(f"{code} 요소를 삭제했습니다.")

    def _load_mission(self) -> None:
        if self.store.draft_vertices:
            self._set_status(
                "안전지대 설정을 완료한 뒤 임무를 장입하십시오.",
                error=True,
            )
            return
        if not self.store.is_mission_ready:
            self._set_status(
                "임무 장입에는 GCS·레이다·이륙지점·안전지대와 "
                "최소 1기의 웨이포인트가 필요합니다.",
                error=True,
            )
            return
        self.loaded_store.replace_from(self.store)
        self._set_status(
            f"임무 장입 완료 // 시설 {len(self.store.sites)}개, "
            f"기체 경로 {len(self.store.configured_vehicle_ids)}/6, "
            f"웨이포인트 총 {self.store.total_waypoint_count}개, "
            f"작전구역 {len(self.store.zones)}개"
        )

    def _new_plan(self) -> None:
        if self.store.has_any_data:
            answer = QMessageBox.question(
                self,
                "새 임무",
                "현재 PLAN 편집 내용을 지우고 새 임무를 작성합니까?\n"
                "이미 장입된 Mission Map 임무는 유지됩니다.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.store.clear()
        self.selected_code = None
        self.code_value.setText("--")
        self.set_vehicle_editing_enabled(False)
        self._set_status("새 PLAN 편집본을 시작했습니다.")

    def _delete_mission(self) -> None:
        has_plan = self.store.has_any_data
        has_loaded = self.loaded_store.has_any_data
        if has_plan or has_loaded:
            answer = QMessageBox.question(
                self,
                "임무 삭제",
                "PLAN 편집본과 Mission Map에 장입된 임무를 모두 삭제합니까?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.store.clear()
        if self.loaded_store is not self.store:
            self.loaded_store.clear()
        self.selected_code = None
        self.code_value.setText("--")
        self.set_vehicle_editing_enabled(False)
        self._set_status("PLAN 편집본과 장입 임무를 삭제했습니다.")

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "임무계획 저장",
            str(Path.cwd() / "mission_plan.json"),
            "GCS Mission Plan (*.json)",
        )
        if not path:
            return
        try:
            self.store.save(path)
            self._set_status(f"저장 완료: {Path(path).name}")
        except OSError as error:
            QMessageBox.critical(self, "저장 실패", str(error))

    def _open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "임무계획 열기",
            str(Path.cwd()),
            "GCS Mission Plan (*.json)",
        )
        if not path:
            return
        try:
            self.store.load(path)
            first = next(self.store.iter_features(), None)
            self.selected_code = first.code if first else None
            if self.selected_code:
                self._select_feature(self.selected_code)
            self._set_status(
                f"불러오기 완료: {Path(path).name} // "
                "검토 후 '임무 장입'을 누르면 Mission Map에 반영됩니다."
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "불러오기 실패", str(error))

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self.editor_status.setText(message)
        self.editor_status.setProperty("error", error)
        self.editor_status.style().unpolish(self.editor_status)
        self.editor_status.style().polish(self.editor_status)
        self.statusMessage.emit(message)
