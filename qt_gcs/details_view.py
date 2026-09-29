from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import (
    QDateTime,
    QPointF,
    QRectF,
    QStandardPaths,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtMultimedia import (
    QMediaCaptureSession,
    QMediaFormat,
    QMediaRecorder,
    QVideoFrame,
    QVideoFrameInput,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .fly_view import AMBER, BLUE, GREEN, MUTED, RED, Fly3DView, SearchCameraWidget
from .site_store import SiteStore


def horizontal_distance_m(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    mean_latitude = math.radians((latitude_1 + latitude_2) / 2.0)
    north_m = (latitude_2 - latitude_1) * 111_320.0
    east_m = (
        (longitude_2 - longitude_1)
        * 111_320.0
        * math.cos(mean_latitude)
    )
    return math.hypot(east_m, north_m)


class SideProfileWidget(QWidget):
    def __init__(
        self,
        state_provider: Callable,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state_provider = state_provider
        self.setMinimumSize(460, 260)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#050b08"))

        left = 58.0
        right = max(left + 100.0, self.width() - 24.0)
        top = 28.0
        bottom = max(top + 100.0, self.height() - 42.0)
        plot = QRectF(left, top, right - left, bottom - top)

        painter.setFont(QFont("IBM Plex Mono", 8))
        painter.setPen(QPen(QColor("#24362a"), 1))
        for index in range(6):
            y = plot.top() + plot.height() * index / 5
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
        for index in range(7):
            x = plot.left() + plot.width() * index / 6
            painter.drawLine(int(x), int(plot.top()), int(x), int(plot.bottom()))

        state = self.state_provider()
        vehicle = state.vehicle
        subject = state.selected_subject
        if subject is None:
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "NO DESIGNATED SUBJECT")
            painter.end()
            return

        distance_m = horizontal_distance_m(
            vehicle.latitude,
            vehicle.longitude,
            subject.latitude,
            subject.longitude,
        )
        maximum_altitude = max(800.0, vehicle.altitude_m * 1.25)

        def altitude_y(altitude_m: float) -> float:
            ratio = max(0.0, min(1.0, altitude_m / maximum_altitude))
            return plot.bottom() - ratio * plot.height()

        vehicle_point = (plot.left(), altitude_y(vehicle.altitude_m))
        subject_point = (plot.right(), altitude_y(subject.altitude_m))
        control_x = plot.left() + plot.width() * 0.58
        control_y = altitude_y(max(vehicle.altitude_m * 0.55, 90.0))

        path = QPainterPath()
        path.moveTo(*vehicle_point)
        path.quadTo(control_x, control_y, *subject_point)
        painter.setPen(QPen(QColor(AMBER), 3, Qt.PenStyle.DashLine))
        painter.drawPath(path)

        painter.setPen(QPen(QColor(BLUE), 2))
        painter.setBrush(QColor("#153a54"))
        painter.drawPolygon(
            [
                self._point(vehicle_point[0], vehicle_point[1] - 10),
                self._point(vehicle_point[0] + 12, vehicle_point[1] + 8),
                self._point(vehicle_point[0] - 12, vehicle_point[1] + 8),
            ]
        )
        painter.setPen(QPen(QColor(RED), 2))
        painter.setBrush(QColor("#431512"))
        painter.drawRect(QRectF(
            subject_point[0] - 12,
            subject_point[1] - 9,
            24,
            18,
        ))

        painter.setPen(QColor("#b9c6bb"))
        painter.drawText(
            int(plot.left() + 8),
            int(vehicle_point[1] - 13),
            f"{vehicle.code}  {vehicle.altitude_m:.0f} m",
        )
        painter.drawText(
            int(plot.right() - 145),
            int(subject_point[1] - 13),
            f"{subject.subject_type} {subject.code}  GROUND",
        )
        painter.setPen(QColor(AMBER))
        painter.drawText(
            int(plot.center().x() - 95),
            int(plot.top() + 18),
            f"비행 경로 // RANGE {distance_m / 1000:.2f} km",
        )

        painter.setPen(QColor(MUTED))
        painter.drawText(
            5,
            int(plot.top() + 8),
            f"{maximum_altitude:.0f}m",
        )
        painter.drawText(18, int(plot.bottom()), "0m")
        painter.drawText(
            int(plot.center().x() - 65),
            self.height() - 12,
            "X: 드론-대상 수평거리",
        )
        painter.drawText(8, 18, "Y: 고도")
        painter.end()

    @staticmethod
    def _point(x: float, y: float):
        return QPointF(x, y)


class ContactHudWidget(QWidget):
    """Display-only artificial-horizon HUD for the selected local contact."""

    def __init__(
        self,
        state_provider: Callable,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state_provider = state_provider
        self.contact_number = 1
        self.setMinimumSize(500, 310)

    def select_contact(self, contact_number: int) -> None:
        self.contact_number = max(1, min(6, int(contact_number)))
        self.update()

    def _contact_values(self) -> dict[str, float | bool | str]:
        state = self.state_provider()
        vehicle = state.vehicle
        index = self.contact_number - 1
        elapsed = state.elapsed_s
        return {
            "heading": vehicle.heading_deg % 360,
            "roll": math.sin(elapsed * 0.72 + index * 0.8) * (8 + index),
            "pitch": math.cos(elapsed * 0.48 + index * 0.55) * 6.5,
            "altitude": max(0.0, vehicle.altitude_m),
            "speed": max(0.0, vehicle.speed_mps),
            "battery": max(31, 96 - index * 6 - int(elapsed / 90)),
            "voltage": 12.55 - index * 0.11,
            "satellites": max(7, 14 - index),
            "armed": state.automatic_mode == "ARM" and not state.emergency_mode,
            "connected": True,
            "time": time.strftime("%H:%M:%S"),
        }

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width = float(self.width())
        height = float(self.height())
        values = self._contact_values()
        roll = float(values["roll"])
        pitch = float(values["pitch"])
        center_x = width / 2
        center_y = height * 0.51
        top_tape_height = max(22.0, height * 0.075)

        painter.fillRect(self.rect(), QColor("#526b82"))
        painter.save()
        painter.setClipRect(QRectF(0, top_tape_height, width, height - top_tape_height))
        painter.translate(center_x, center_y)
        painter.rotate(-roll)
        pitch_offset = pitch * height / 45.0
        horizon_y = pitch_offset
        painter.fillRect(
            QRectF(-width * 1.2, -height * 1.4, width * 2.4, height * 1.4 + horizon_y),
            QColor("#58738c"),
        )
        painter.fillRect(
            QRectF(-width * 1.2, horizon_y, width * 2.4, height * 1.6),
            QColor("#596a32"),
        )
        painter.setPen(QPen(QColor("#d0d2bb"), 2))
        painter.drawLine(
            int(-width),
            int(horizon_y),
            int(width),
            int(horizon_y),
        )
        painter.setFont(QFont("IBM Plex Mono", max(7, int(height / 42))))
        for pitch_mark in range(-30, 35, 5):
            if pitch_mark == 0:
                continue
            y = horizon_y - pitch_mark * height / 45.0
            line_half = width * (0.10 if pitch_mark % 10 else 0.16)
            painter.setPen(QPen(QColor("#d5d7c1"), 2))
            painter.drawLine(
                int(-line_half),
                int(y),
                int(line_half),
                int(y),
            )
            if pitch_mark % 10 == 0:
                painter.drawText(
                    int(-line_half - 32),
                    int(y + 4),
                    str(pitch_mark),
                )
                painter.drawText(
                    int(line_half + 8),
                    int(y + 4),
                    str(pitch_mark),
                )
        painter.restore()

        self._draw_heading_tape(painter, values, width, top_tape_height)
        self._draw_roll_scale(painter, roll, center_x, top_tape_height, width, height)
        self._draw_side_tapes(painter, values, width, height, top_tape_height)

        status_color = QColor("#ad504b" if not values["armed"] else "#5da873")
        painter.setFont(QFont("IBM Plex Sans KR", max(12, int(height / 20)), QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#4d1011"), 3))
        status = "ARMED" if values["armed"] else "DISARMED"
        status_rect = QRectF(0, height * 0.31, width, height * 0.12)
        painter.drawText(
            status_rect.translated(1, 1),
            Qt.AlignmentFlag.AlignCenter,
            status,
        )
        painter.setPen(status_color)
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignCenter, status)

        vehicle_y = center_y + 8
        painter.setPen(QPen(QColor("#af5149"), 3))
        painter.drawLine(
            int(center_x - width * 0.08),
            int(vehicle_y),
            int(center_x),
            int(vehicle_y - 12),
        )
        painter.drawLine(
            int(center_x),
            int(vehicle_y - 12),
            int(center_x + width * 0.08),
            int(vehicle_y),
        )
        painter.drawLine(
            int(center_x - width * 0.11),
            int(vehicle_y - 2),
            int(center_x - width * 0.05),
            int(vehicle_y - 2),
        )
        painter.drawLine(
            int(center_x + width * 0.05),
            int(vehicle_y - 2),
            int(center_x + width * 0.11),
            int(vehicle_y - 2),
        )

        painter.setFont(QFont("IBM Plex Mono", max(7, int(height / 38))))
        painter.setPen(QColor("#d4ddd4"))
        bottom_y = height - 8
        left_status = (
            f"배터리 BAT {float(values['voltage']):.2f}V / "
            f"{int(values['battery'])}%"
        )
        center_status = "EKF 정상 / 진동 VIBE 정상"
        right_status = f"GPS 3D FIX / 위성 {int(values['satellites'])}개"
        painter.drawText(8, int(bottom_y), left_status)
        painter.drawText(
            QRectF(0, height - 28, width, 22),
            Qt.AlignmentFlag.AlignCenter,
            center_status,
        )
        painter.drawText(
            QRectF(width * 0.58, height - 28, width * 0.4 - 8, 22),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            right_status,
        )
        painter.setPen(QColor("#c4cec5"))
        painter.drawText(
            8,
            int(height - 30),
            f"속도 SPD {float(values['speed']):.1f} m/s",
        )
        painter.drawText(
            QRectF(width * 0.68, height - 55, width * 0.3, 22),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"SIGHTING-{self.contact_number:02d}",
        )
        painter.drawText(
            QRectF(8, top_tape_height + 4, width * 0.5, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"방위각 HDG {float(values['heading']):03.0f}°",
        )
        painter.drawText(
            QRectF(width * 0.46, top_tape_height + 4, width * 0.52 - 8, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"ROLL {roll:+.1f}° / PITCH {pitch:+.1f}°",
        )
        painter.end()

    @staticmethod
    def _draw_heading_tape(
        painter: QPainter,
        values: dict,
        width: float,
        tape_height: float,
    ) -> None:
        heading = float(values["heading"])
        painter.fillRect(
            QRectF(0, 0, width, tape_height),
            QColor("#465d73"),
        )
        painter.setPen(QPen(QColor("#d4dcde"), 1))
        painter.drawLine(0, int(tape_height - 2), int(width), int(tape_height - 2))
        pixels_per_degree = width / 90.0
        center_x = width / 2
        painter.setFont(QFont("IBM Plex Mono", max(7, int(tape_height * 0.38))))
        for offset in range(-50, 55, 5):
            shown_heading = (heading + offset) % 360
            x = center_x + offset * pixels_per_degree
            major = int(round(shown_heading)) % 15 == 0
            length = tape_height * (0.48 if major else 0.28)
            painter.drawLine(
                int(x),
                int(tape_height - 2),
                int(x),
                int(tape_height - length),
            )
            if major:
                cardinal = {
                    0: "N",
                    90: "E",
                    180: "S",
                    270: "W",
                }.get(int(round(shown_heading)) % 360)
                label = cardinal or f"{int(round(shown_heading)) % 360:03d}"
                painter.drawText(
                    QRectF(x - 22, 0, 44, tape_height * 0.55),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
        pointer = QPolygonF(
            [
                QPointF(center_x, tape_height - 1),
                QPointF(center_x - 6, tape_height - 10),
                QPointF(center_x + 6, tape_height - 10),
            ]
        )
        painter.setBrush(QColor("#101512"))
        painter.drawPolygon(pointer)

    @staticmethod
    def _draw_roll_scale(
        painter: QPainter,
        roll: float,
        center_x: float,
        tape_height: float,
        width: float,
        height: float,
    ) -> None:
        radius = min(width * 0.27, height * 0.29)
        center_y = tape_height + radius * 0.82
        painter.setPen(QPen(QColor("#eff3e9"), 2))
        for angle in range(-60, 61, 10):
            radians = math.radians(angle - 90)
            outer_x = center_x + math.cos(radians) * radius
            outer_y = center_y + math.sin(radians) * radius
            inner_radius = radius - (12 if angle % 30 == 0 else 7)
            inner_x = center_x + math.cos(radians) * inner_radius
            inner_y = center_y + math.sin(radians) * inner_radius
            painter.drawLine(
                int(inner_x),
                int(inner_y),
                int(outer_x),
                int(outer_y),
            )
        roll_angle = math.radians(roll - 90)
        pointer_x = center_x + math.cos(roll_angle) * (radius - 18)
        pointer_y = center_y + math.sin(roll_angle) * (radius - 18)
        painter.setBrush(QColor("#aa504b"))
        painter.setPen(QPen(QColor("#aa504b"), 1))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(pointer_x, pointer_y),
                    QPointF(pointer_x - 6, pointer_y + 9),
                    QPointF(pointer_x + 6, pointer_y + 9),
                ]
            )
        )

    @staticmethod
    def _draw_side_tapes(
        painter: QPainter,
        values: dict,
        width: float,
        height: float,
        tape_height: float,
    ) -> None:
        tape_top = tape_height + 45
        tape_bottom = height - 68
        tape_width = max(35.0, width * 0.075)
        painter.setFont(QFont("IBM Plex Mono", max(7, int(height / 40))))
        painter.setPen(QPen(QColor("#d3dcd4"), 1))
        painter.drawText(
            QRectF(5, tape_top - 25, 90, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "SPD (m/s)",
        )
        painter.drawText(
            QRectF(width - 95, tape_top - 25, 90, 20),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "ALT (m)",
        )
        painter.drawLine(
            int(tape_width),
            int(tape_top),
            int(tape_width),
            int(tape_bottom),
        )
        painter.drawLine(
            int(width - tape_width),
            int(tape_top),
            int(width - tape_width),
            int(tape_bottom),
        )
        for index in range(5):
            y = tape_top + (tape_bottom - tape_top) * index / 4
            painter.drawLine(0, int(y), int(tape_width), int(y))
            painter.drawLine(
                int(width - tape_width),
                int(y),
                int(width),
                int(y),
            )
        center_y = (tape_top + tape_bottom) / 2
        painter.setBrush(QColor("#07100d"))
        painter.setPen(QPen(QColor("#dce8dc"), 1))
        painter.drawRect(QRectF(0, center_y - 15, tape_width, 30))
        painter.drawRect(QRectF(width - tape_width, center_y - 15, tape_width, 30))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            QRectF(0, center_y - 15, tape_width, 30),
            Qt.AlignmentFlag.AlignCenter,
            f"{float(values['speed']):.0f}",
        )
        painter.drawText(
            QRectF(width - tape_width, center_y - 15, tape_width, 30),
            Qt.AlignmentFlag.AlignCenter,
            f"{float(values['altitude']):.0f}",
        )


class DetailsView(QWidget):
    statusMessage = Signal(str)

    def __init__(
        self,
        store: SiteStore,
        mission_map: Fly3DView,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.mission_map = mission_map
        self._last_log_second = -1
        self._active = False
        self._recording = False
        self._recording_frame_index = 0
        self._recording_path: Path | None = None
        self._last_recording_path: Path | None = None
        self._launcher_log_entries: list[str] = []
        self._radar_log_entries: list[str] = []
        self._equipment_log_dialog: QDialog | None = None
        self._launcher_log_view: QPlainTextEdit | None = None
        self._radar_log_view: QPlainTextEdit | None = None
        self._capture_session = QMediaCaptureSession(self)
        self._media_recorder = QMediaRecorder(self)
        self._video_frame_input = QVideoFrameInput(self)
        self._capture_session.setRecorder(self._media_recorder)
        self._capture_session.setVideoFrameInput(self._video_frame_input)
        self._media_recorder.errorOccurred.connect(self._on_recording_error)
        self._media_recorder.recorderStateChanged.connect(
            self._on_recorder_state_changed
        )
        self._recording_timer = QTimer(self)
        self._recording_timer.setInterval(100)
        self._recording_timer.timeout.connect(self._capture_camera_frame)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self._refresh()

    @property
    def state(self):
        return self.mission_map.state

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        layout.addWidget(self._build_camera_panel(), 0, 0)
        layout.addWidget(self._build_vehicle_panel(), 0, 1)
        layout.addWidget(self._build_profile_panel(), 1, 0)
        layout.addWidget(self._build_control_panel(), 1, 1)

    @staticmethod
    def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("detailsPanel")
        frame.setStyleSheet(
            """
            QFrame#detailsPanel {
                background:#070c09;
                border-top:1px solid #58645a;
                border-left:1px solid #485449;
                border-right:2px solid #030504;
                border-bottom:2px solid #030504;
            }
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("panelTitle")
        layout.addWidget(label)
        return frame, layout

    def _build_camera_panel(self) -> QWidget:
        frame, layout = self._panel("탐지 카메라 화면 // DETECTION CAMERA")
        header = QHBoxLayout()
        source = QLabel("EO/IR SIMULATION")
        source.setObjectName("fieldCaption")
        self.record_button = QPushButton("● 녹화 시작")
        self.record_button.clicked.connect(self._toggle_camera_recording)
        self.save_recording_button = QPushButton("저장")
        self.save_recording_button.setEnabled(False)
        self.save_recording_button.clicked.connect(self._save_camera_recording)
        self.camera_hold = QLabel("● NO HOLD")
        header.addWidget(source)
        header.addStretch(1)
        header.addWidget(self.record_button)
        header.addWidget(self.save_recording_button)
        header.addWidget(self.camera_hold)
        layout.addLayout(header)
        self.search_camera = SearchCameraWidget(self.state)
        layout.addWidget(self.search_camera, 1)
        return frame

    def _toggle_camera_recording(self) -> None:
        if self._recording:
            self._stop_camera_recording()
        else:
            self._start_camera_recording()

    def _start_camera_recording(self) -> None:
        first_frame = self.search_camera.grab().toImage()
        if first_frame.isNull():
            self.statusMessage.emit("탐색기 녹화를 시작할 수 없습니다.")
            return

        recording_directory = (
            Path(QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.TempLocation
            ))
            / "TacticalGCS"
            / "recordings"
        )
        recording_directory.mkdir(parents=True, exist_ok=True)
        timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        self._recording_path = recording_directory / f"search_camera_{timestamp}.mp4"
        self._last_recording_path = None

        media_format = QMediaFormat()
        media_format.setFileFormat(QMediaFormat.FileFormat.MPEG4)
        supported_codecs = media_format.supportedVideoCodecs(
            QMediaFormat.ConversionMode.Encode
        )
        if QMediaFormat.VideoCodec.H264 in supported_codecs:
            media_format.setVideoCodec(QMediaFormat.VideoCodec.H264)
        elif QMediaFormat.VideoCodec.MPEG4 in supported_codecs:
            media_format.setVideoCodec(QMediaFormat.VideoCodec.MPEG4)
        self._media_recorder.setMediaFormat(media_format)
        self._media_recorder.setQuality(QMediaRecorder.Quality.HighQuality)
        self._media_recorder.setVideoFrameRate(10.0)
        even_width = max(2, first_frame.width() // 2 * 2)
        even_height = max(2, first_frame.height() // 2 * 2)
        self._media_recorder.setVideoResolution(even_width, even_height)
        self._media_recorder.setOutputLocation(
            QUrl.fromLocalFile(str(self._recording_path))
        )

        self._recording = True
        self._recording_frame_index = 0
        self.save_recording_button.setEnabled(False)
        self.record_button.setText("■ 녹화 중지")
        self.record_button.setStyleSheet(
            "background:#3c1715; color:#d9756e; border:1px solid #8e4943;"
            "font-weight:700;"
        )
        self._media_recorder.record()
        self._recording_timer.start()
        self._capture_camera_frame()
        self.statusMessage.emit("탐색기 녹화 시작 // 10 FPS / MP4")

    def _capture_camera_frame(self) -> None:
        if not self._recording:
            return
        image = self.search_camera.grab().toImage().convertToFormat(
            QImage.Format.Format_ARGB32
        )
        if image.isNull():
            return
        even_width = max(2, image.width() // 2 * 2)
        even_height = max(2, image.height() // 2 * 2)
        if image.width() != even_width or image.height() != even_height:
            image = image.scaled(
                even_width,
                even_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        frame = QVideoFrame(image)
        frame_duration_us = 100_000
        frame.setStartTime(self._recording_frame_index * frame_duration_us)
        frame.setEndTime((self._recording_frame_index + 1) * frame_duration_us)
        if self._video_frame_input.sendVideoFrame(frame):
            self._recording_frame_index += 1

    def _stop_camera_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._recording_timer.stop()
        self._video_frame_input.sendVideoFrame(QVideoFrame())
        self._media_recorder.stop()
        self.record_button.setText("● 녹화 시작")
        self.record_button.setStyleSheet("")
        self.statusMessage.emit("탐색기 녹화 종료 // 저장 버튼을 누르십시오.")

    def _on_recorder_state_changed(
        self,
        recorder_state: QMediaRecorder.RecorderState,
    ) -> None:
        if (
            recorder_state == QMediaRecorder.RecorderState.StoppedState
            and self._recording_path is not None
            and self._recording_frame_index > 0
        ):
            self._last_recording_path = self._recording_path
            self.save_recording_button.setEnabled(True)

    def _on_recording_error(
        self,
        _error: QMediaRecorder.Error,
        error_message: str,
    ) -> None:
        self._recording = False
        self._recording_timer.stop()
        self.record_button.setText("● 녹화 시작")
        self.record_button.setStyleSheet("")
        self.statusMessage.emit(f"탐색기 녹화 오류 // {error_message}")

    def _save_camera_recording(self) -> None:
        source_path = self._last_recording_path
        if source_path is None or not source_path.exists():
            self.statusMessage.emit("저장할 탐색기 녹화 영상이 없습니다.")
            return
        movies_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.MoviesLocation
        )
        suggested_path = Path(movies_directory or str(Path.home())) / source_path.name
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "탐색기 녹화 영상 저장",
            str(suggested_path),
            "MP4 영상 (*.mp4)",
        )
        if not selected_path:
            return
        destination = Path(selected_path)
        if destination.suffix.lower() != ".mp4":
            destination = destination.with_suffix(".mp4")
        try:
            shutil.copy2(source_path, destination)
        except OSError as error:
            self.statusMessage.emit(f"탐색기 영상 저장 실패 // {error}")
            return
        self.statusMessage.emit(f"탐색기 영상 저장 완료 // {destination}")

    def _build_vehicle_panel(self) -> QWidget:
        frame, layout = self._panel("선택 UAV 상태")
        selector = QHBoxLayout()
        selector.setSpacing(4)
        selector_label = QLabel("TOP BAR VEHICLE")
        selector_label.setObjectName("fieldCaption")
        selector.addWidget(selector_label)
        self.contact_buttons: list[QPushButton] = []
        selector.addStretch(1)
        self.contact_status = QLabel("● SIGHTING-01")
        self.contact_status.setObjectName("dataValue")
        selector.addWidget(self.contact_status)
        layout.addLayout(selector)

        legend = QLabel(
            "계기 표기: HDG 방위각(°) · SPD 속도(m/s) · ALT 고도(m) · "
            "ROLL 좌우 기울기(°) · PITCH 기수 상하각(°) · "
            "BAT 배터리 · GPS 위성 상태"
        )
        legend.setObjectName("mutedText")
        legend.setWordWrap(True)
        legend.setStyleSheet(
            "background:#101713; color:#9eaaa0; border:1px solid #303b32;"
            "padding:3px 5px; font-size:8pt;"
        )
        layout.addWidget(legend)

        self.contact_hud = ContactHudWidget(lambda: self.state)
        layout.addWidget(self.contact_hud, 1)
        return frame

    def _build_profile_panel(self) -> QWidget:
        frame, layout = self._panel("측면 접근 프로파일 // DISTANCE × ALTITUDE")
        self.profile = SideProfileWidget(lambda: self.state)
        layout.addWidget(self.profile, 1)
        return frame

    def _build_control_panel(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("detailsPanel")
        frame.setStyleSheet(
            """
            QFrame#detailsPanel {
                background:#070c09;
                border-top:1px solid #58645a;
                border-left:1px solid #485449;
                border-right:2px solid #030504;
                border-bottom:2px solid #030504;
            }
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel("통신 제어상태")
        title.setObjectName("panelTitle")
        self.equipment_log_button = QPushButton("LOG DATA")
        self.equipment_log_button.clicked.connect(self._show_equipment_logs)
        self.equipment_log_button.setMaximumWidth(110)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.equipment_log_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._launcher_scroll())
        splitter.addWidget(self._radar_scroll())
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([180, 180])
        layout.addWidget(splitter, 1)
        return frame

    @staticmethod
    def _number_input(
        minimum: float,
        maximum: float,
        decimals: int,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        control.setMinimumWidth(0)
        control.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        asset_directory = Path(__file__).resolve().parent / "assets"
        control.setStyleSheet(
            """
            QDoubleSpinBox {
                min-width:0px;
                min-height:28px;
                max-height:32px;
                padding:1px 28px 1px 3px;
            }
            QDoubleSpinBox::up-button {
                subcontrol-origin:border;
                subcontrol-position:top right;
                width:24px;
                background:#1b281e;
                border-left:1px solid #526054;
                border-bottom:1px solid #303a31;
            }
            QDoubleSpinBox::down-button {
                subcontrol-origin:border;
                subcontrol-position:bottom right;
                width:24px;
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
                width:11px;
                height:7px;
            }
            QDoubleSpinBox::down-arrow {
                image:url(__DOWN_ARROW__);
                width:11px;
                height:7px;
            }
            """
            .replace(
                "__UP_ARROW__",
                (asset_directory / "spin_up.svg").as_posix(),
            )
            .replace(
                "__DOWN_ARROW__",
                (asset_directory / "spin_down.svg").as_posix(),
            )
        )
        return control

    def _launcher_scroll(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("데이터링크 제어/상태")
        title.setObjectName("panelTitle")
        title.setStyleSheet("font-size:10pt; padding:2px 3px;")
        title.setMaximumHeight(20)
        self.launcher_location_label = QLabel("이륙지점 위치 // 임무 미장입")
        self.launcher_location_label.setObjectName("mutedText")
        self.launcher_location_label.setStyleSheet(
            "background:transparent; color:#aebbb0; border:0;"
            "padding:0px 3px; font-size:8pt;"
        )
        self.launcher_location_label.setMinimumWidth(0)
        self.launcher_location_label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self.launcher_location_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.launcher_location_label.setMaximumHeight(20)
        header.addWidget(title)
        header.addWidget(self.launcher_location_label, 1)
        layout.addLayout(header)
        content = QGridLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setHorizontalSpacing(5)
        content.setVerticalSpacing(3)
        content.setColumnStretch(0, 32)
        content.setColumnStretch(1, 28)
        content.setColumnStretch(2, 40)

        self.datalink_status_group = QGroupBox("전송률/주파수")
        datalink_layout = QGridLayout(self.datalink_status_group)
        datalink_layout.setContentsMargins(4, 8, 4, 3)
        datalink_layout.setHorizontalSpacing(3)
        datalink_layout.setVerticalSpacing(4)
        datalink_layout.setColumnStretch(0, 3)
        datalink_layout.setColumnStretch(1, 4)
        self.datalink_status_group.setMinimumHeight(82)

        value_style = (
            "background:#0b130d; color:#c7d8ca; border:1px solid #334638;"
            "padding:1px 3px; font-size:10pt;"
        )
        caption_style = (
            "background:transparent; color:#8f9c91; border:0;"
            "padding:0px; font-size:10pt;"
        )

        self.bitrate_value_label = QLabel("12.0 Mbps")
        self.frequency_bandwidth_value_label = QLabel("5.8 GHz / 20 MHz")

        for value_label in (
            self.bitrate_value_label,
            self.frequency_bandwidth_value_label,
        ):
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet(value_style)
            value_label.setMinimumHeight(26)
            value_label.setMaximumHeight(26)
            value_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        traffic_widget = QWidget()
        traffic_layout = QHBoxLayout(traffic_widget)
        traffic_layout.setContentsMargins(0, 0, 0, 0)
        traffic_layout.setSpacing(2)
        self.rx_status_label = QLabel("RX ●")
        self.tx_status_label = QLabel("TX ●")
        for traffic_label in (self.rx_status_label, self.tx_status_label):
            traffic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            traffic_label.setStyleSheet(
                f"background:#102518; color:{GREEN}; border:1px solid #367447;"
                "padding:1px 2px; font-size:8pt;"
            )
            traffic_label.setMinimumHeight(20)
            traffic_label.setMaximumHeight(21)
            traffic_label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            traffic_layout.addWidget(traffic_label, 1)

        self.datalink_caption_labels: list[QLabel] = []
        for row, (caption, value_widget) in enumerate(
            (
                ("전송률", self.bitrate_value_label),
                ("주파수/대역폭", self.frequency_bandwidth_value_label),
                ("송수신상태", traffic_widget),
            )
        ):
            caption_label = QLabel(caption)
            # caption_label.setAlignment(
            #     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            # )
            caption_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            caption_label.setStyleSheet(caption_style)
            self.datalink_caption_labels.append(caption_label)
            datalink_layout.addWidget(caption_label, row, 0)
            datalink_layout.addWidget(value_widget, row, 1)

        umbilical = QGroupBox("FANET 통신상태")
        umbilical_layout = QGridLayout(umbilical)
        umbilical_layout.setContentsMargins(4, 8, 4, 3)
        umbilical_layout.setSpacing(2)
        umbilical_layout.setRowStretch(0, 1)
        umbilical_layout.setRowStretch(1, 2)
        umbilical_layout.setRowStretch(2, 2)
        umbilical_layout.setRowStretch(3, 1)
        umbilical.setMinimumHeight(70)
        self.umbilical_buttons: list[QLabel] = []
        for index in range(1, 7):
            indicator = QLabel(f"{index} ●")
            indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            indicator.setStyleSheet(
                f"background:#102518; color:{GREEN}; border:1px solid #367447;"
                "min-width:0px; min-height:18px; padding:1px 2px;"
            )
            indicator.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Expanding,
            )
            self.umbilical_buttons.append(indicator)
            umbilical_layout.addWidget(
                indicator,
                1 + (index - 1) // 3,
                (index - 1) % 3,
            )

        connections = QGroupBox("ADT/GDT 연결상태")
        connections_layout = QVBoxLayout(connections)
        connections_layout.setContentsMargins(4, 8, 4, 3)
        connections_layout.setSpacing(0)
        connections.setMinimumHeight(70)
        self.canister_table = QTableWidget(2, 6)
        self.canister_table.setHorizontalHeaderLabels(
            tuple(str(number) for number in range(1, 7))
        )
        self.canister_table.setVerticalHeaderLabels(
            ("ADT", "GDT")
        )
        self.canister_table.verticalHeader().setVisible(True)
        self.canister_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.canister_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.canister_table.setStyleSheet(
            "font-size:8pt; padding:0px;"
            "QHeaderView::section { padding:2px; font-size:8pt; }"
        )
        self.canister_table.verticalHeader().setMinimumSectionSize(12)
        self.canister_table.verticalHeader().setDefaultSectionSize(14)
        self.canister_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.canister_table.verticalHeader().setFixedWidth(34)
        self.canister_table.horizontalHeader().setFixedHeight(24)
        self.canister_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.canister_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.canister_table.setMinimumWidth(0)
        self.canister_table.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.canister_table.setMinimumHeight(50)
        for column in range(6):
            for row, value in enumerate(
                ("OK", "OK")
            ):
                status_item = QTableWidgetItem(value)
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                status_item.setForeground(QColor(GREEN))
                self.canister_table.setItem(row, column, status_item)
        connections_layout.addWidget(self.canister_table)

        content.addWidget(self.datalink_status_group, 0, 0)
        content.addWidget(umbilical, 0, 1)
        content.addWidget(connections, 0, 2)
        layout.addLayout(content)
        scroll.setWidget(body)
        scroll.setMinimumHeight(118)
        return scroll

    def _radar_scroll(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(3)

        title = QLabel("레이다 제어 / 상태")
        title.setObjectName("panelTitle")
        title.setStyleSheet("font-size:10pt; padding:2px 3px;")
        title.setMaximumHeight(20)
        layout.addWidget(title)

        self.detection_table = QTableWidget(4, 4)
        self.detection_table.setHorizontalHeaderLabels(
            ("탐지", "위도", "경도", "고도")
        )
        self.detection_table.verticalHeader().setVisible(False)
        self.detection_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.detection_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.detection_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.detection_table.setStyleSheet(
            "font-size:8pt; padding:0px;"
            "QHeaderView::section { padding:2px; font-size:8pt; }"
        )
        self.detection_table.verticalHeader().setMinimumSectionSize(16)
        self.detection_table.verticalHeader().setDefaultSectionSize(18)
        self.detection_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.detection_table.horizontalHeader().setFixedHeight(24)
        self.detection_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.detection_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.detection_table.setMinimumWidth(0)
        self.detection_table.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.detection_table.setMinimumHeight(94)

        self.restricted_zone_inputs = (
            self._number_input(-90, 90, 6),
            self._number_input(-180, 180, 6),
            self._number_input(0, 20_000, 1, " m"),
        )
        self.jamming_inputs = (
            self._number_input(-90, 90, 6),
            self._number_input(-180, 180, 6),
            self._number_input(0, 20_000, 1, " m"),
        )
        self.radar_inputs = (
            self._number_input(-90, 90, 6),
            self._number_input(-180, 180, 6),
            self._number_input(0, 20_000, 1, " m"),
        )
        for control in self.radar_inputs:
            control.editingFinished.connect(self._apply_radar_location)

        self.radar_coordinate_table = QTableWidget(3, 4)
        self.radar_coordinate_table.setHorizontalHeaderLabels(
            ("구분", "위도", "경도", "고도")
        )
        self.radar_coordinate_table.verticalHeader().setVisible(False)
        self.radar_coordinate_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.radar_coordinate_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        coordinate_header = self.radar_coordinate_table.horizontalHeader()
        coordinate_header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        for column in range(1, 4):
            coordinate_header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Stretch,
            )
        self.radar_coordinate_table.setStyleSheet(
            "font-size:8pt; padding:0px;"
            "QHeaderView::section { padding:2px; font-size:8pt; }"
        )
        self.radar_coordinate_table.verticalHeader().setMinimumSectionSize(18)
        self.radar_coordinate_table.verticalHeader().setDefaultSectionSize(20)
        self.radar_coordinate_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.radar_coordinate_table.horizontalHeader().setFixedHeight(24)
        self.radar_coordinate_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.radar_coordinate_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.radar_coordinate_table.setMinimumWidth(0)
        self.radar_coordinate_table.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.radar_coordinate_table.setMinimumHeight(82)
        for row, (label, controls) in enumerate(
            (
                ("비행 제한 구역", self.restricted_zone_inputs),
                ("재밍 영역", self.jamming_inputs),
                ("레이다 위치", self.radar_inputs),
            )
        ):
            label_item = QTableWidgetItem(label)
            label_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.radar_coordinate_table.setItem(row, 0, label_item)
            for column, control in enumerate(controls, start=1):
                control.setStyleSheet(
                    "min-width:0px; min-height:17px; max-height:19px;"
                    "padding:0px 2px; font-size:7pt;"
                )
                self.radar_coordinate_table.setCellWidget(row, column, control)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(6)
        content.addWidget(self.detection_table, 4)
        content.addWidget(self.radar_coordinate_table, 5)
        layout.addLayout(content)
        scroll.setWidget(body)
        scroll.setMinimumHeight(124)
        return scroll

    def _coordinate_group(
        self,
        parent_layout: QVBoxLayout,
        title: str,
    ) -> tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        latitude = self._number_input(-90, 90, 6)
        longitude = self._number_input(-180, 180, 6)
        altitude = self._number_input(0, 20_000, 1, " m")
        form.addRow("위도", latitude)
        form.addRow("경도", longitude)
        form.addRow("고도", altitude)
        parent_layout.addWidget(group)
        return latitude, longitude, altitude

    @staticmethod
    def _log_widget() -> QPlainTextEdit:
        log = QPlainTextEdit()
        log.setReadOnly(True)
        log.setMaximumBlockCount(180)
        log.setStyleSheet(
            "background:#030604; color:#79d98c; font-family:'IBM Plex Mono';"
        )
        return log

    def _show_equipment_logs(self) -> None:
        if self._equipment_log_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("통신 제어상태 LOG DATA")
            dialog.setModal(False)
            dialog.resize(760, 520)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)

            heading = QLabel("LOG DATA // LOCAL STATUS HISTORY")
            heading.setObjectName("panelTitle")
            layout.addWidget(heading)

            splitter = QSplitter(Qt.Orientation.Vertical)
            launcher_group = QGroupBox("데이터링크 제어/상태 로그")
            launcher_layout = QVBoxLayout(launcher_group)
            self._launcher_log_view = self._log_widget()
            launcher_layout.addWidget(self._launcher_log_view)
            splitter.addWidget(launcher_group)

            radar_group = QGroupBox("레이다 제어 / 상태 로그")
            radar_layout = QVBoxLayout(radar_group)
            self._radar_log_view = self._log_widget()
            radar_layout.addWidget(self._radar_log_view)
            splitter.addWidget(radar_group)
            splitter.setChildrenCollapsible(False)
            splitter.setSizes([230, 230])
            layout.addWidget(splitter, 1)

            close_button = QPushButton("닫기")
            close_button.clicked.connect(dialog.close)
            layout.addWidget(close_button)
            self._equipment_log_dialog = dialog

        assert self._launcher_log_view is not None
        assert self._radar_log_view is not None
        self._launcher_log_view.setPlainText(
            "\n".join(self._launcher_log_entries)
        )
        self._radar_log_view.setPlainText(
            "\n".join(self._radar_log_entries)
        )
        self._equipment_log_dialog.show()
        self._equipment_log_dialog.raise_()
        self._equipment_log_dialog.activateWindow()
        self.statusMessage.emit("통신 제어상태 LOG DATA 창 열림")

    def activate(self) -> None:
        self._active = True
        if not self.timer.isActive():
            self.timer.start()
        self._load_site_inputs()
        self._tick()
        self.statusMessage.emit("DETAILS READY // LOCAL DATA // COMMAND TX OFF")

    def deactivate(self) -> None:
        if self._recording:
            self._stop_camera_recording()
        self._active = False
        self.timer.stop()

    def _tick(self) -> None:
        if not self._active:
            return
        self.state.sync_plan_readiness(self.store)
        self._refresh()

    def _refresh(self) -> None:
        state = self.state
        elapsed = state.elapsed_s
        self.contact_hud.update()

        held = state.mission_status["TRACK HOLD"]
        self.search_camera.state = state
        self.camera_hold.setText(
            "SHUT DOWN"
            if state.subject_found
            else ("● HOLD" if held else "● NO HOLD")
        )
        self.camera_hold.setStyleSheet(
            f"color:{RED if state.subject_found else (GREEN if held else RED)};"
        )
        self.search_camera.refresh()
        self.profile.update()
        self._refresh_detections()

        second = int(elapsed)
        if second != self._last_log_second:
            self._last_log_second = second
            subject = state.selected_subject
            self._launcher_log(
                "STATUS BITRATE=12.0Mbps FREQ=5.8GHz "
                f"BW=20MHz RX=ON TX=ON MODE={state.automatic_mode}"
            )
            self._radar_log(
                f"TRACKS={len(state.subjects)} "
                f"SELECTED={subject.code if subject else 'NONE'}"
            )

    def _select_contact(self, contact_number: int) -> None:
        self.contact_hud.select_contact(contact_number)
        self.contact_status.setText(f"● SIGHTING-{contact_number:02d}")
        self.statusMessage.emit(
            f"DETAILS SIGHTING-{contact_number:02d} 선택 // LOCAL DATA"
        )

    def _refresh_detections(self) -> None:
        for row, track in enumerate(self.state.subjects[:4]):
            values = (
                f"{row + 1}/{track.subject_type}",
                f"{track.latitude:.5f}",
                f"{track.longitude:.5f}",
                f"{track.altitude_m:.1f}",
            )
            for column, value in enumerate(values):
                item = self.detection_table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.detection_table.setItem(row, column, item)
                item.setText(value)

    def _load_site_inputs(self) -> None:
        launcher = self.store.sites.get("LC")
        radar = self.store.sites.get("RDR")
        if launcher:
            self.launcher_location_label.setText(
                "이륙지점 위치 // "
                f"위도 {launcher.latitude:.6f}  |  "
                f"경도 {launcher.longitude:.6f}  |  "
                f"고도 {launcher.altitude_m:.1f} m"
            )
        else:
            self.launcher_location_label.setText("이륙지점 위치 // 임무 미장입")
        if radar:
            self.radar_inputs[0].setValue(radar.latitude)
            self.radar_inputs[1].setValue(radar.longitude)
            self.radar_inputs[2].setValue(radar.altitude_m)

    def _apply_radar_location(self) -> None:
        self.store.set_site(
            "RDR",
            self.radar_inputs[0].value(),
            self.radar_inputs[1].value(),
            self.radar_inputs[2].value(),
        )
        self._radar_log("레이다 위치 로컬 적용")

    def _toggle_umbilical(self, unit: int, connected: bool) -> None:
        indicator = self.umbilical_buttons[unit - 1]
        indicator.setText(f"{unit} ●")
        indicator.setStyleSheet(
            (
                f"background:#102518; color:{GREEN}; border:1px solid #367447;"
                if connected
                else f"background:#2a0d0b; color:{RED}; border:1px solid #7a2d28;"
            )
            + "min-width:0px; min-height:18px; padding:1px 2px;"
        )
        self._launcher_log(
            f"FANET-{unit} {'연결' if connected else '분리'}"
        )

    def _launcher_log(self, message: str) -> None:
        entry = f"{time.strftime('%H:%M:%S')}  {message}"
        self._launcher_log_entries.append(entry)
        self._launcher_log_entries = self._launcher_log_entries[-180:]
        if self._launcher_log_view is not None:
            self._launcher_log_view.appendPlainText(entry)

    def _radar_log(self, message: str) -> None:
        entry = f"{time.strftime('%H:%M:%S')}  {message}"
        self._radar_log_entries.append(entry)
        self._radar_log_entries = self._radar_log_entries[-180:]
        if self._radar_log_view is not None:
            self._radar_log_view.appendPlainText(entry)
