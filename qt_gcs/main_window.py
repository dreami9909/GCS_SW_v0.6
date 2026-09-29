from __future__ import annotations

from PySide6.QtCore import Qt, QTime, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .details_view import DetailsView
from .fly_view import Fly3DView
from .plan_view import Plan3DView
from .site_store import SiteStore


APP_STYLESHEET = """
QWidget {
    background: #070b09;
    color: #c7d0c5;
    font-family: "IBM Plex Sans KR", "Malgun Gothic";
    font-size: 10pt;
}
QFrame#topBar {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #293128, stop:0.45 #151b16, stop:1 #080c09
    );
    border-top: 1px solid #4d594d;
    border-bottom: 2px solid #374238;
}
QPushButton {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #2b342c, stop:0.48 #1c241d, stop:0.52 #121914, stop:1 #0b100d
    );
    color: #cbd3c9;
    border-top: 1px solid #667064;
    border-left: 1px solid #556054;
    border-right: 2px solid #171d18;
    border-bottom: 2px solid #111612;
    padding: 7px 11px;
    min-height: 24px;
    font-weight: 600;
}
QPushButton:hover {
    background: #303d31;
    color: #f2c04d;
    border-top-color: #808c7c;
    border-left-color: #707c6d;
}
QPushButton:pressed {
    background: #0c120e;
    border-top: 2px solid #111612;
    border-left: 2px solid #111612;
    border-right: 1px solid #596359;
    border-bottom: 1px solid #596359;
}
QPushButton:checked {
    background: #313a30;
    color: #f2bd3f;
    border: 1px solid #8c7130;
    border-bottom: 3px solid #d69b29;
}
QPushButton[siteTool="true"]:checked {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #5b451b, stop:0.5 #34270f, stop:1 #1a1409
    );
    color: #ffd66a;
    border-top: 1px solid #d3a642;
    border-left: 1px solid #a77d28;
    border-right: 2px solid #201707;
    border-bottom: 2px solid #201707;
}
QPushButton#dataButton {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #f3c74f, stop:0.5 #c58a1e, stop:1 #765013
    );
    color: #10140f;
    font-family: "IBM Plex Mono";
    font-weight: 700;
}
QPushButton#primaryButton {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #8d681e, stop:0.5 #5d4214, stop:1 #33240b
    );
    color: #ffe5a0;
    border-top: 1px solid #d2a23d;
    border-left: 1px solid #a77b28;
    border-right: 2px solid #211707;
    border-bottom: 2px solid #211707;
    font-weight: 700;
}
QPushButton#fleetVehicleButton {
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    color: #8f9b90;
    font-family: "IBM Plex Mono";
    font-size: 10pt;
    font-weight: 700;
}
QPushButton#fleetVehicleButton[routeConfigured="true"] {
    color: #73df89;
    border-color: #456f4e;
}
QPushButton#fleetVehicleButton:checked {
    color: #071009;
    background: #71dd88;
    border-top: 1px solid #b3f3bf;
    border-left: 1px solid #91e9a2;
    border-right: 2px solid #204329;
    border-bottom: 2px solid #18361f;
}
QPushButton#fleetVehicleButton:disabled {
    color: #485149;
    background: #0b100c;
    border-color: #232a24;
}
QLabel#toolbarTitle {
    color: #929c90;
    font-family: "IBM Plex Sans KR";
    font-size: 8pt;
    font-weight: 600;
}
QLabel#toolbarValue {
    color: #71dd88;
    font-family: "IBM Plex Mono", "IBM Plex Sans KR";
    font-size: 10pt;
    font-weight: 700;
}
QLabel#panelTitle {
    color: #f1b937;
    font-family: "IBM Plex Sans KR";
    font-size: 13pt;
    font-weight: 700;
    padding: 7px 3px;
    border-bottom: 1px solid #5f512b;
}
QLabel#mutedText, QLabel#fieldCaption {
    color: #89948a;
    font-size: 9pt;
    font-weight: 600;
}
QLabel#dataValue {
    color: #75e28b;
    font-family: "IBM Plex Mono", "IBM Plex Sans KR";
    font-weight: 700;
}
QLabel#providerOnline {
    color: #74df89;
    background: #101912;
    border-top: 1px solid #536254;
    border-left: 1px solid #465347;
    border-right: 2px solid #080c09;
    border-bottom: 2px solid #080c09;
    padding: 7px;
    font-family: "IBM Plex Mono";
    font-weight: 700;
}
QLabel#providerOffline {
    color: #f1b937;
    background: #251d0e;
    border: 1px solid #755b20;
    padding: 7px;
    font-family: "IBM Plex Mono";
    font-weight: 700;
}
QLabel#statusLine {
    color: #9fbea5;
    background: #0b110d;
    border-top: 1px solid #3f4b40;
    border-left: 1px solid #354036;
    border-right: 2px solid #050806;
    border-bottom: 2px solid #050806;
    padding: 7px;
}
QLabel#statusLine[error="true"] { color: #ff736c; border-color: #7b3935; }
QFrame#toolPanel, QFrame#editorPanel {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #111712, stop:0.5 #0c110d, stop:1 #080c09
    );
    border-right: 1px solid #354036;
}
QFrame#editorPanel {
    border-left: 1px solid #354036;
    border-right: 0;
}
QFrame#divider { color: #4b574b; background: #4b574b; }
QGroupBox {
    color: #f1b937;
    border-top: 1px solid #59645a;
    border-left: 1px solid #4a554b;
    border-right: 2px solid #060a07;
    border-bottom: 2px solid #060a07;
    margin-top: 9px;
    padding-top: 9px;
    font-weight: 700;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QTableWidget {
    background: #080d0a;
    alternate-background-color: #101711;
    gridline-color: #2f3930;
    border-top: 1px solid #586359;
    border-left: 1px solid #4a554b;
    border-right: 2px solid #050806;
    border-bottom: 2px solid #050806;
    selection-background-color: #5b4620;
}
QHeaderView::section {
    background: #202920;
    color: #e8b43c;
    border: 0;
    border-right: 1px solid #49534a;
    border-bottom: 2px solid #0b0f0c;
    padding: 5px;
    font-family: "IBM Plex Mono";
    font-size: 8pt;
    font-weight: 600;
}
QDoubleSpinBox {
    background: #060a07;
    color: #73df89;
    border-top: 1px solid #4d584e;
    border-left: 1px solid #414b42;
    border-right: 2px solid #030504;
    border-bottom: 2px solid #030504;
    padding: 5px;
    font-family: "IBM Plex Mono";
}
QLabel#footer {
    color: #8c978d;
    background: #050806;
    border-top: 2px solid #303a31;
    padding: 5px 9px;
    font-family: "IBM Plex Mono", "IBM Plex Sans KR";
    font-size: 8pt;
}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tactical Ground Control")
        self.setMinimumSize(1280, 760)
        self.resize(1600, 960)
        self.setStyleSheet(APP_STYLESHEET)
        # PLAN edits stay in ``store`` until the operator explicitly loads
        # an independent snapshot into ``mission_store``.
        self.store = SiteStore()
        self.mission_store = SiteStore()
        self._pages: dict[str, QWidget] = {}
        self._nav_buttons: dict[str, QPushButton] = {}
        self._vehicle_buttons: dict[int, QPushButton] = {}
        self._vehicle_route_selected = False
        self.selected_vehicle_id = 0

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_top_bar())

        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, 1)
        self.footer = QLabel("PLAN READY // GOOGLE 3D MAP")
        self.footer.setObjectName("footer")
        root_layout.addWidget(self.footer)
        self.setCentralWidget(root)

        self._build_pages()
        self.store.subscribe(self._refresh_vehicle_buttons)
        self._refresh_vehicle_buttons()
        self.show_page("Plan")

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(58)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 5, 9, 5)
        layout.setSpacing(5)

        group = QButtonGroup(self)
        group.setExclusive(True)
        for name, label in (
            ("Data", "▤  DATA"),
            ("Plan", "PLAN"),
            ("Fly", "MISSION MAP"),
            ("Analyze", "DETAILS"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            if name == "Data":
                button.setObjectName("dataButton")
            button.clicked.connect(lambda _checked, page=name: self.show_page(page))
            group.addButton(button)
            layout.addWidget(button)
            self._nav_buttons[name] = button

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedWidth(12)
        layout.addWidget(separator)

        for title, value in (
            ("시스템", "정상"),
            ("지도", "GOOGLE 3D"),
            ("데이터 링크", "미연결"),
        ):
            layout.addWidget(self._status_chip(title, value))

        fleet_label = QLabel("기체")
        fleet_label.setObjectName("toolbarTitle")
        fleet_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fleet_label.setFixedWidth(28)
        layout.addWidget(fleet_label)

        self._vehicle_group = QButtonGroup(self)
        self._vehicle_group.setExclusive(True)
        for vehicle_id in (0, *SiteStore.VEHICLE_IDS):
            button = QPushButton("전체" if vehicle_id == 0 else str(vehicle_id))
            button.setObjectName("fleetVehicleButton")
            if vehicle_id == 0:
                button.setFixedWidth(48)
            button.setCheckable(True)
            button.setChecked(vehicle_id == 0)
            button.setToolTip(
                "6기 전체 운용 화면을 선택합니다."
                if vehicle_id == 0
                else f"UAV-{vehicle_id:02d} 웨이포인트 경로를 선택합니다."
            )
            button.clicked.connect(
                lambda checked, selected_id=vehicle_id: (
                    self._select_vehicle(selected_id) if checked else None
                )
            )
            self._vehicle_group.addButton(button, vehicle_id)
            self._vehicle_buttons[vehicle_id] = button
            layout.addWidget(button)
        layout.addStretch(1)
        self.time_value = QLabel("--:--:--")
        self.time_value.setObjectName("toolbarValue")
        layout.addWidget(self.time_value)
        return bar

    def _refresh_vehicle_buttons(self) -> None:
        shared_ready = self.store.shared_configuration_ready
        if not shared_ready and self._vehicle_route_selected:
            self._vehicle_route_selected = False
            self._vehicle_group.setExclusive(False)
            for vehicle_id, button in self._vehicle_buttons.items():
                button.setChecked(vehicle_id == 0)
            self._vehicle_group.setExclusive(True)
            self.selected_vehicle_id = 0
            if hasattr(self, "plan_view"):
                self.plan_view.set_vehicle_editing_enabled(False)

        configured = set(self.store.configured_vehicle_ids)
        for vehicle_id, button in self._vehicle_buttons.items():
            button.setEnabled(vehicle_id == 0 or shared_ready)
            button.setProperty(
                "routeConfigured",
                "true" if vehicle_id == 0 or vehicle_id in configured else "false",
            )
            button.style().unpolish(button)
            button.style().polish(button)

    def _select_vehicle(self, vehicle_id: int) -> None:
        if vehicle_id == 0:
            self.selected_vehicle_id = 0
            self._vehicle_route_selected = False
            self.plan_view.show_fleet_overview()
            if hasattr(self, "fly_view"):
                self.fly_view.select_vehicle(0)
            if hasattr(self, "details_view"):
                self.details_view._select_contact(1)
            self.set_status("6기 전체 운용 화면 // 상세 정보는 UAV-01 기준")
            return
        if not self.store.shared_configuration_ready:
            self.set_status(
                "GCS·레이다·이륙지점·안전지대 공통 설정을 먼저 완료하십시오."
            )
            return
        self._vehicle_route_selected = True
        self.selected_vehicle_id = vehicle_id
        self.store.set_active_vehicle(vehicle_id)
        # Display selection must not reload/reset an in-progress fleet mission.
        self.mission_store.active_vehicle_id = vehicle_id
        self.plan_view.on_active_vehicle_changed(vehicle_id)
        if hasattr(self, "details_view"):
            self.details_view._select_contact(vehicle_id)
        if hasattr(self, "fly_view"):
            self.fly_view.select_vehicle(vehicle_id)
        self.set_status(f"UAV-{vehicle_id:02d} 웨이포인트 경로 편집 활성화")

    @staticmethod
    def _status_chip(title: str, value: str) -> QWidget:
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(8, 0, 8, 0)
        box.setSpacing(0)
        title_label = QLabel(title)
        title_label.setObjectName("toolbarTitle")
        value_label = QLabel(value)
        value_label.setObjectName("toolbarValue")
        box.addWidget(title_label)
        box.addWidget(value_label)
        return widget

    def _build_pages(self) -> None:
        self.plan_view = Plan3DView(self.store, self.mission_store)
        self.plan_view.statusMessage.connect(self.set_status)
        self._add_page("Plan", self.plan_view)
        self.fly_view = Fly3DView(
            self.mission_store,
            shared_web_view=self.plan_view.web_view,
            shared_bridge=self.plan_view.bridge,
        )
        self.fly_view.statusMessage.connect(self.set_status)
        self.fly_view.vehicleSelectionRequested.connect(
            self._select_vehicle_from_seeker
        )
        self._add_page("Fly", self.fly_view)
        self.details_view = DetailsView(self.mission_store, self.fly_view)
        self.details_view.statusMessage.connect(self.set_status)
        self._add_page("Analyze", self.details_view)
        self._add_page(
            "Data",
            self._placeholder(
                "SENSING DATA",
                "데이터 반영 전",
            ),
        )

    def _add_page(self, name: str, page: QWidget) -> None:
        self._pages[name] = page
        self.stack.addWidget(page)

    def _select_vehicle_from_seeker(self, vehicle_id: int) -> None:
        button = self._vehicle_buttons.get(vehicle_id)
        if button is None or not button.isEnabled():
            return
        button.setChecked(True)
        self._select_vehicle(vehicle_id)

    @staticmethod
    def _placeholder(title: str, description: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label = QLabel(description)
        description_label.setObjectName("mutedText")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return page

    def show_page(self, name: str) -> None:
        page = self._pages.get(name)
        if page is None:
            return
        if hasattr(self, "fly_view"):
            if name == "Fly":
                self.fly_view.activate()
            else:
                self.fly_view.deactivate()
                if name == "Plan":
                    self.plan_view.attach_map_view()
        if hasattr(self, "details_view"):
            if name == "Analyze":
                self.details_view.activate()
            else:
                self.details_view.deactivate()
        self.stack.setCurrentWidget(page)
        for page_name, button in self._nav_buttons.items():
            button.setChecked(page_name == name)
        display_name = {
            "Fly": "Mission map",
            "Analyze": "Details",
        }.get(name, name)
        self.set_status(f"{display_name} view")

    def set_status(self, message: str) -> None:
        self.footer.setText(message)

    def _update_clock(self) -> None:
        self.time_value.setText(QTime.currentTime().toString("HH:mm:ss"))
