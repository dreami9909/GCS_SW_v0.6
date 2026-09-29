from __future__ import annotations

import argparse
import os
import sys


def hydrate_google_maps_key() -> None:
    """Read a newly saved Windows user key even from an older IDE process."""
    if os.getenv("GOOGLE_MAPS_API_KEY", "").strip() or sys.platform != "win32":
        return
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "GOOGLE_MAPS_API_KEY")
    except (FileNotFoundError, OSError):
        return
    value = str(value).strip()
    if value:
        os.environ["GOOGLE_MAPS_API_KEY"] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qt 3D PLAN + MISSION MAP milestone"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Construct the Qt window, process events, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")
        os.environ["GOOGLE_MAPS_API_KEY"] = ""
        os.environ.setdefault(
            "QTWEBENGINE_CHROMIUM_FLAGS",
            "--disable-gpu --disable-software-rasterizer",
        )
    else:
        hydrate_google_maps_key()

    from PySide6.QtWidgets import QApplication
    if args.smoke_test:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import (
            QDoubleSpinBox,
            QScrollArea,
            QStyle,
            QStyleOptionSpinBox,
        )
        from qt_gcs.planning import build_footprint

    from qt_gcs.fonts import load_interface_fonts
    from qt_gcs.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Tactical Ground Control")
    load_interface_fonts(app)
    window = MainWindow()
    if args.smoke_test:
        window.show()
        for _ in range(4):
            app.processEvents()
        assert "Plan" in window._pages
        assert window.plan_view.provider_name
        window.plan_view._select_placement_tool("GCS")
        window.plan_view._on_map_clicked(37.3422, 127.9202, 42.0)
        assert window.store.sites["GCS"].altitude_m == 42.0
        assert window.plan_view.table.rowCount() == 1
        assert not window._vehicle_buttons[1].isEnabled()
        window.plan_view._select_placement_tool("RDR")
        window.plan_view._on_map_clicked(37.3440, 127.9180, 55.0)
        window.plan_view._select_placement_tool("LC")
        window.plan_view._on_map_clicked(37.3410, 127.9190, 51.0)
        window.plan_view._select_placement_tool("SAFE")
        window.plan_view._on_map_clicked(37.3400, 127.9200, 0.0)
        window.plan_view._on_map_clicked(37.3410, 127.9240, 0.0)
        window.plan_view._on_map_clicked(37.3380, 127.9230, 0.0)
        assert len(window.store.zones) == 1
        assert not window.store.draft_vertices
        assert window.store.draft_zone_type is None
        assert window.plan_view.placement_code is None
        assert window.plan_view.table.rowCount() == 4
        assert all(
            button.isEnabled()
            for button in window._vehicle_buttons.values()
        )
        QTest.mouseClick(
            window._vehicle_buttons[1],
            Qt.MouseButton.LeftButton,
        )
        assert window.plan_view.vehicle_editing_enabled
        assert window.store.active_vehicle_id == 1
        window.plan_view._select_placement_tool("WP")
        window.plan_view._on_map_clicked(37.3450, 127.9250, 0.0)
        assert len(window.store.waypoints_for(1)) == 1
        assert window.plan_view.table.rowCount() == 5
        QTest.mouseClick(
            window._vehicle_buttons[2],
            Qt.MouseButton.LeftButton,
        )
        assert window.store.active_vehicle_id == 2
        assert not window.store.waypoints
        window.plan_view._select_placement_tool("WP")
        window.plan_view._on_map_clicked(37.3460, 127.9260, 0.0)
        assert len(window.store.waypoints_for(2)) == 1
        QTest.mouseClick(
            window._vehicle_buttons[1],
            Qt.MouseButton.LeftButton,
        )
        window.plan_view._select_feature("WP001")
        assert (
            window.plan_view.undo_vertex_button.text()
            == "웨이포인트 삭제 (WP001)"
        )
        QTest.mouseClick(
            window.plan_view.undo_vertex_button,
            Qt.MouseButton.LeftButton,
        )
        assert not window.store.waypoints_for(1)
        assert len(window.store.waypoints_for(2)) == 1
        assert "웨이포인트 삭제 완료" in window.plan_view.editor_status.text()
        spin = window.plan_view.map_altitude_input
        spin.setValue(250.0)
        spin_option = QStyleOptionSpinBox()
        spin.initStyleOption(spin_option)
        up_button = spin.style().subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            spin_option,
            QStyle.SubControl.SC_SpinBoxUp,
            spin,
        )
        assert up_button.width() >= 28
        QTest.mouseClick(
            spin,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            up_button.center(),
        )
        assert spin.value() == 251.0
        for _ in range(20):
            if window.plan_view.map_ready:
                break
            QTest.qWait(100)
        assert window.plan_view.map_ready
        window.plan_view.map_latitude_input.setValue(37.4012345)
        window.plan_view.map_longitude_input.setValue(127.1012345)
        window.plan_view.map_altitude_input.setValue(320.0)
        window.plan_view._move_map_to_inputs()
        for _ in range(20):
            if "지도 중심 이동 완료" in window.plan_view.editor_status.text():
                break
            QTest.qWait(50)
        assert "지도 중심 이동 완료" in window.plan_view.editor_status.text()
        assert not window.mission_store.sites
        window.show_page("Fly")
        for _ in range(4):
            app.processEvents()
        assert window.fly_view.update_timer.isActive()
        assert window.fly_view.subject_table.rowCount() == 2
        assert window.fly_view.subject_table.height() == 120
        assert not window.fly_view.state.readiness["GCS"]
        assert not window.fly_view.state.readiness["RDR"]
        assert not window.fly_view.state.readiness["LC"]
        window.fly_view._select_subject(204)
        assert window.fly_view.state.selected_track_id == 204
        window.show_page("Analyze")
        for _ in range(4):
            app.processEvents()
        assert window.details_view.timer.isActive()
        assert window.details_view.detection_table.rowCount() == 4
        assert window.details_view.state.selected_subject.altitude_m == 0
        assert window.details_view.record_button.text() == "● 녹화 시작"
        assert window.details_view.save_recording_button.text() == "저장"
        assert not window.details_view.save_recording_button.isEnabled()
        assert window.details_view.bitrate_value_label.text() == "12.0 Mbps"
        assert (
            window.details_view.frequency_bandwidth_value_label.text()
            == "5.8 GHz / 20 MHz"
        )
        assert window.details_view.rx_status_label.text() == "RX ●"
        assert window.details_view.tx_status_label.text() == "TX ●"
        assert not window.details_view.datalink_status_group.findChildren(
            QDoubleSpinBox
        )
        window.details_view._select_contact(6)
        assert window.details_view.contact_hud.contact_number == 6
        assert not window.details_view.contact_buttons
        window.show_page("Plan")
        assert not window.fly_view.update_timer.isActive()
        assert not window.details_view.timer.isActive()
        window.store.seed_demo(37.3422, 127.9202)
        QTest.mouseClick(
            window._vehicle_buttons[0],
            Qt.MouseButton.LeftButton,
        )
        assert not window.plan_view.vehicle_editing_enabled
        assert window.plan_view.table.rowCount() == 34
        window.plan_view._load_mission()
        assert window.mission_store.is_mission_ready
        assert window.mission_store.waypoints is not window.store.waypoints
        assert window.mission_store.waypoints[0] is not window.store.waypoints[0]
        window.show_page("Analyze")
        assert "이륙지점 위치 // 위도" in (
            window.details_view.launcher_location_label.text()
        )
        assert window.details_view.canister_table.rowCount() == 2
        QTest.mouseClick(
            window.details_view.equipment_log_button,
            Qt.MouseButton.LeftButton,
        )
        app.processEvents()
        assert window.details_view._equipment_log_dialog is not None
        assert window.details_view._equipment_log_dialog.isVisible()
        window.details_view._equipment_log_dialog.close()
        window.resize(1280, 760)
        for _ in range(4):
            app.processEvents()
        for scroll_area in window.details_view.findChildren(QScrollArea):
            assert scroll_area.verticalScrollBar().maximum() == 0
            assert scroll_area.horizontalScrollBar().maximum() == 0
        datalink_labels = [
            *window.details_view.datalink_caption_labels,
            window.details_view.bitrate_value_label,
            window.details_view.frequency_bandwidth_value_label,
            window.details_view.rx_status_label,
            window.details_view.tx_status_label,
        ]
        for label in datalink_labels:
            text_width = label.fontMetrics().horizontalAdvance(label.text())
            assert label.contentsRect().width() >= text_width, (
                label.text(),
                label.contentsRect().width(),
                text_width,
            )
        window.resize(1600, 960)
        window.show_page("Fly")
        assert window.fly_view.state.launch_ready
        window.fly_view._request_launch()
        window.fly_view.update_timer.stop()
        launched_positions = {
            vehicle_id: (
                state.vehicle.latitude,
                state.vehicle.longitude,
                state.vehicle.altitude_m,
            )
            for vehicle_id, state in window.fly_view.states.items()
        }
        window.fly_view._pending_mission.replace_from(window.mission_store)
        window.fly_view._pending_mission.set_active_vehicle(2)
        pending_waypoint = window.fly_view._pending_mission.add_waypoint(
            37.455,
            127.988,
            650.0,
        )
        window.fly_view._pending_dirty = True
        window.fly_view._emit_map_state()
        QTest.qWait(300)
        rendered_map_state = {}
        window.fly_view.web_view.page().runJavaScript(
            """
            (() => {
              const markers = Array.from(document.querySelectorAll('.marker'));
              return markers.filter(marker =>
                marker.textContent.includes('수정 대기')
              ).length;
            })()
            """,
            lambda result: rendered_map_state.__setitem__(
                "pendingCount",
                result,
            ),
        )
        window.fly_view.web_view.page().runJavaScript(
            """
            (() => {
              const pending = Array.from(
                document.querySelectorAll('.marker')
              ).find(marker => marker.textContent.includes('수정 대기'));
              return pending ? Number(getComputedStyle(pending).opacity) : null;
            })()
            """,
            lambda result: rendered_map_state.__setitem__(
                "pendingOpacity",
                result,
            ),
        )
        window.fly_view.web_view.page().runJavaScript(
            """
            (() => {
              const colors = Array.from(document.querySelectorAll('.marker'))
                .filter(marker => marker.textContent.includes('▲ SAR-'))
                .map(marker => getComputedStyle(marker).color);
              return new Set(colors).size;
            })()
            """,
            lambda result: rendered_map_state.__setitem__(
                "vehicleColorCount",
                result,
            ),
        )
        window.fly_view.web_view.page().runJavaScript(
            """
            (() => {
              const colors = Array.from(
                document.querySelectorAll('#shapes polyline')
              ).map(line => line.getAttribute('stroke')).filter(Boolean);
              return new Set(colors).size;
            })()
            """,
            lambda result: rendered_map_state.__setitem__(
                "routeColorCount",
                result,
            ),
        )
        for _ in range(20):
            if len(rendered_map_state) == 4:
                break
            QTest.qWait(50)
        assert rendered_map_state["pendingCount"] == 1
        assert float(rendered_map_state["pendingOpacity"]) <= 0.31
        assert rendered_map_state["vehicleColorCount"] == 6
        assert rendered_map_state["routeColorCount"] >= 6
        window.fly_view._apply_pending_mission()
        assert not window.fly_view._pending_dirty
        assert len(window.mission_store.waypoints_for(2)) == 6
        assert pending_waypoint.code == "WP006"
        assert all(
            state.mission_launched
            for state in window.fly_view.states.values()
        )
        assert launched_positions == {
            vehicle_id: (
                state.vehicle.latitude,
                state.vehicle.longitude,
                state.vehicle.altitude_m,
            )
            for vehicle_id, state in window.fly_view.states.items()
        }
        # Detection is expected near the subject-side search waypoint, not at
        # the launcher tens of kilometres away.  Advance the smoke scenario
        # directly to that operational checkpoint so the test remains fast.
        for state in window.fly_view.states.values():
            detection_waypoint = state._nearest_route_point_to_subject(
                state.selected_subject
            )
            assert detection_waypoint is not None
            state.vehicle.latitude = detection_waypoint[0]
            state.vehicle.longitude = detection_waypoint[1]
            state.vehicle.altitude_m = detection_waypoint[2]
        detector_state = window.fly_view.states[1]
        detector_state._search_started = True
        footprint = build_footprint(
            window.fly_view.search_camera_spec,
            window.fly_view.planning_engine.frame,
            vehicle_id=1,
            latitude=detector_state.vehicle.latitude,
            longitude=detector_state.vehicle.longitude,
            altitude_m=detector_state.vehicle.altitude_m,
            heading_deg=detector_state.vehicle.heading_deg,
            elapsed_s=detector_state.simulation_elapsed_s,
        )
        detected_latitude, detected_longitude = (
            window.fly_view.planning_engine.frame.to_geographic(
                footprint.center
            )
        )
        for state in window.fly_view.states.values():
            state._search_started = True
            subject = next(track for track in state.subjects if track.track_id == 101)
            subject.latitude = detected_latitude
            subject.longitude = detected_longitude
        window.fly_view._run_planning_cycle(force=True)
        detection_dialog = window.fly_view._detection_dialog
        assert detection_dialog is not None
        assert detection_dialog.windowTitle() == "발견 확인"
        assert "수색 대상을 발견했습니다" in detection_dialog.text()
        automatic_detector_id = (
            window.fly_view.state.detection_source_vehicle_id
        )
        automatic_track_id = window.fly_view.state.selected_track_id
        assert automatic_detector_id is not None
        assert f"SAR-{automatic_detector_id:02d}" in detection_dialog.text()
        assert "자동 인식" in detection_dialog.text()
        assert "6대 AUTO 협동 구조 접근" in detection_dialog.text()
        assert (
            detection_dialog.windowModality()
            == Qt.WindowModality.ApplicationModal
        )
        assert detection_dialog.windowFlags() & (
            Qt.WindowType.WindowStaysOnTopHint
        )
        cooperative_states = [
            (
                vehicle_id,
                state.subject_detected,
                state.selected_track_id,
                state.flight_phase,
            )
            for vehicle_id, state in window.fly_view.states.items()
        ]
        assert all(
            state.subject_detected
            and state.selected_track_id == automatic_track_id
            and state.flight_phase == "INITIAL_APPROACH"
            for state in window.fly_view.states.values()
        ), cooperative_states
        shared_subject_coordinates = {
            (
                state.selected_subject.latitude,
                state.selected_subject.longitude,
            )
            for state in window.fly_view.states.values()
        }
        assert len(shared_subject_coordinates) == 1
        detection_dialog.accept()
        app.processEvents()
        window.fly_view._run_planning_cycle(force=True)
        assert window.fly_view.state.automatic_mode == "ARM"
        assert not window.fly_view.state.can_press_launch
        # The predicted-approach ("예상 접근지점") concept was removed, so
        # predicted_approach() is always None now.
        assert window.fly_view.state.predicted_approach() is None
        assert window.fly_view._planning_payload["revision"] >= 1
        for _ in range(350):
            for state in window.fly_view.states.values():
                state.tick(1.0)
            window.fly_view._run_planning_cycle(force=True)
            if window.fly_view.state.subject_found:
                break
        window.fly_view._refresh_display()
        assert window.fly_view.launch_button.text() == "이륙 완료"
        assert window.fly_view.state.subject_found, (
            window.fly_view.state.flight_phase,
            window.fly_view.state.vehicle.latitude,
            window.fly_view.state.vehicle.longitude,
            window.fly_view.state.selected_subject.latitude,
            window.fly_view.state.selected_subject.longitude,
        )
        window.close()
        print("QT_3D_UI_SMOKE_TEST_OK")
        return 0

    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
