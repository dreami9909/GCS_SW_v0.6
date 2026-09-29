from __future__ import annotations

import unittest
from pathlib import Path

from qt_gcs.fly_state import (
    FlyState,
    INITIAL_APPROACH_SPEED_MPS,
    SAR_CRUISE_SPEED_MPS,
    SAR_INGRESS_SPEED_MPS,
    SAR_SEARCH_SPEED_MPS,
    MIDCOURSE_APPROACH_SPEED_MPS,
    SIMULATION_TIME_SCALE,
    FINAL_APPROACH_SPEED_MPS,
    bearing_deg,
    constant_velocity_approach_time,
    destination_position,
    horizontal_distance_m,
)
from qt_gcs.site_store import SiteStore


class FlyStateTests(unittest.TestCase):
    def test_visual_simulation_uses_three_timescale(self) -> None:
        self.assertEqual(3.0, SIMULATION_TIME_SCALE)

    def test_relative_motion_rendezvous_time_uses_both_tracks(self) -> None:
        stationary_time, stationary_reachable = (
            constant_velocity_approach_time(
                1_000.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                100.0,
            )
        )
        receding_time, receding_reachable = (
            constant_velocity_approach_time(
                1_000.0, 0.0, 0.0,
                20.0, 0.0, 0.0,
                100.0,
            )
        )
        self.assertTrue(stationary_reachable)
        self.assertTrue(receding_reachable)
        self.assertAlmostEqual(10.0, stationary_time, places=6)
        self.assertAlmostEqual(12.5, receding_time, places=6)

    def test_plan_sites_drive_readiness(self) -> None:
        store = SiteStore()
        state = FlyState.demo(37.3422, 127.9202)
        state.sync_plan_readiness(store)
        self.assertFalse(state.readiness["GCS"])
        self.assertFalse(state.readiness["RDR"])
        self.assertFalse(state.readiness["LC"])
        self.assertEqual(
            {"MISSING_PERSON_CHILD", "MISSING_PERSON_ADULT"},
            {track.subject_type for track in state.subjects},
        )
        self.assertTrue(all(track.altitude_m == 0 for track in state.subjects))

        store.seed_demo(37.3422, 127.9202)
        state.sync_plan_readiness(store)
        self.assertTrue(state.launch_ready)

    def test_completed_route_segments_advance_and_reset(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        self.assertTrue(state.request_simulated_launch())
        first_waypoint = store.waypoints_for(1)[0]
        state.vehicle.latitude = first_waypoint.latitude
        state.vehicle.longitude = first_waypoint.longitude
        state.vehicle.altitude_m = first_waypoint.altitude_m

        state._advance_route(0.1)

        self.assertEqual(1, state.completed_route_segment_count)
        self.assertEqual(1, state.current_waypoint_index)
        self.assertEqual(
            1,
            state.render_dict(store)["completed_route_segment_count"],
        )
        state._reset_execution()
        self.assertEqual(0, state.completed_route_segment_count)

    def test_in_flight_route_update_preserves_execution_and_position(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        self.assertTrue(state.request_simulated_launch())
        state.vehicle.latitude = 37.401234
        state.vehicle.longitude = 127.951234
        state.vehicle.altitude_m = 612.0
        original_phase = state.flight_phase
        original_route_length = len(state._route_points)

        store.set_active_vehicle(1)
        store.add_waypoint(37.455, 127.988, 650.0)
        state.update_route_from_store(store, 1)

        self.assertTrue(state.mission_launched)
        self.assertEqual(original_phase, state.flight_phase)
        self.assertEqual(37.401234, state.vehicle.latitude)
        self.assertEqual(127.951234, state.vehicle.longitude)
        self.assertEqual(612.0, state.vehicle.altitude_m)
        self.assertEqual(original_route_length + 1, len(state._route_points))

    def test_runtime_route_is_queued_then_committed_at_segment_boundary(
        self,
    ) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        self.assertEqual([], state.runtime_route_payload())
        self.assertTrue(state.request_simulated_launch())
        first_waypoint = store.waypoints_for(1)[0]
        replacement = [
            {
                "latitude": first_waypoint.latitude + 0.001,
                "longitude": first_waypoint.longitude + 0.001,
                "altitude_m": 600.0,
                "code": "AUTO01",
            }
        ]

        self.assertTrue(state.queue_runtime_route(7, replacement))
        self.assertEqual(0, state.runtime_route_revision)
        self.assertEqual(7, state.pending_runtime_route_revision)
        state.vehicle.latitude = first_waypoint.latitude
        state.vehicle.longitude = first_waypoint.longitude
        state.vehicle.altitude_m = first_waypoint.altitude_m
        state._advance_route(0.1)

        self.assertEqual(7, state.runtime_route_revision)
        self.assertEqual(1, state.runtime_route_update_count)
        self.assertEqual(0, state.pending_runtime_route_revision)
        self.assertEqual("AUTO01", state.runtime_route_payload()[0]["code"])

    def test_precomputed_route_commits_at_tp_without_position_jump(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        self.assertTrue(state.request_simulated_launch())
        rally = store.waypoints_for(1)[0]
        replacement = [
            {
                "latitude": rally.latitude + 0.003,
                "longitude": rally.longitude + 0.003,
                "altitude_m": 600.0,
                "code": "RHP-01",
            },
            {
                "latitude": rally.latitude + 0.006,
                "longitude": rally.longitude,
                "altitude_m": 600.0,
                "code": "RHP-02",
            },
        ]

        # A route calculated before TP must remain pending during ingress.
        self.assertTrue(state.queue_runtime_route(7, replacement))
        self.assertEqual(7, state.pending_runtime_route_revision)
        self.assertEqual(0, state.runtime_route_revision)
        self.assertEqual(0.0, state.search_elapsed_s)

        # Place the vehicle 30 m before TP.  The ingress tick may cover only
        # that remaining distance; committing RHP must not move it again.
        approach_heading = (
            bearing_deg(
                state.vehicle.latitude,
                state.vehicle.longitude,
                rally.latitude,
                rally.longitude,
            )
            + 180.0
        ) % 360.0
        state.vehicle.latitude, state.vehicle.longitude = destination_position(
            rally.latitude,
            rally.longitude,
            30.0,
            approach_heading,
        )
        state.vehicle.altitude_m = rally.altitude_m
        state._advance_route(0.05)

        self.assertAlmostEqual(rally.latitude, state.vehicle.latitude, places=9)
        self.assertAlmostEqual(rally.longitude, state.vehicle.longitude, places=9)
        self.assertEqual(7, state.runtime_route_revision)
        self.assertEqual(0, state.pending_runtime_route_revision)
        self.assertEqual(0.0, state.search_elapsed_s)
        committed_position = (
            state.vehicle.latitude,
            state.vehicle.longitude,
        )

        # The following frame advances by the normal simulated search step,
        # proving there is no route-application teleport.
        state._advance_route(0.05)
        frame_step_m = horizontal_distance_m(
            committed_position[0],
            committed_position[1],
            state.vehicle.latitude,
            state.vehicle.longitude,
        )
        self.assertGreater(frame_step_m, 0.0)
        self.assertLessEqual(
            frame_step_m,
            SAR_SEARCH_SPEED_MPS * 0.05 * SIMULATION_TIME_SCALE + 0.1,
        )
        state.tick(1.0)
        self.assertEqual(SIMULATION_TIME_SCALE, state.search_elapsed_s)

    def test_first_automatic_route_applies_immediately_after_rally(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        state.completed_route_segment_count = 1
        replacement = [
            {
                "latitude": 37.39,
                "longitude": 127.97,
                "altitude_m": 600.0,
                "code": "AUTO01",
            }
        ]

        self.assertTrue(state.queue_runtime_route(8, replacement))
        self.assertEqual(8, state.runtime_route_revision)
        self.assertEqual(1, state.runtime_route_update_count)
        self.assertEqual(0, state.pending_runtime_route_revision)
        self.assertEqual("AUTO01", state.runtime_route_payload()[0]["code"])

        second_replacement = [
            {
                "latitude": 37.395,
                "longitude": 127.975,
                "altitude_m": 600.0,
                "code": "AUTO02",
            }
        ]
        self.assertTrue(state.queue_runtime_route(9, second_replacement))
        self.assertEqual(9, state.runtime_route_revision)
        self.assertEqual(2, state.runtime_route_update_count)
        self.assertEqual(0, state.pending_runtime_route_revision)
        self.assertEqual("AUTO02", state.runtime_route_payload()[0]["code"])

    def test_detection_abandons_search_route_for_cooperative_atr_approach(
        self,
    ) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        original_route = tuple(state._route_points)
        approach = (
            state.vehicle.latitude,
            state.vehicle.longitude,
            600.0,
            "AUTO-01",
        )

        self.assertTrue(
            state.designate_subject(
                101,
                cooperative_approach=approach,
                detector_vehicle_id=3,
            )
        )
        self.assertEqual(3, state.detection_source_vehicle_id)
        self.assertEqual("AUTO HANDOFF", state.camera_mode)
        self.assertEqual(approach, state._detection_waypoint)
        self.assertNotIn(approach, original_route)
        self.assertEqual("AUTO-01", state.approach_route_payload()[0]["code"])

        state._advance_detection_transit(0.1)
        self.assertEqual("INITIAL_APPROACH", state.flight_phase)
        self.assertEqual("AUTO SCAN", state.camera_mode)
        state._advance_initial_guidance(8.0 / 3.0)
        self.assertEqual("MIDCOURSE_APPROACH", state.flight_phase)
        self.assertEqual("AUTO TRACK", state.camera_mode)
        state._start_final_guidance()
        self.assertEqual("AUTO HOLD", state.camera_mode)

    def test_closing_track_is_visible_before_final_hold(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        state.designate_subject(101)
        subject = state.selected_subject
        self.assertIsNotNone(subject)
        state.vehicle.latitude = subject.latitude
        state.vehicle.longitude = subject.longitude
        state.vehicle.altitude_m = 600.0
        state.flight_phase = "MIDCOURSE_APPROACH"
        state._phase_elapsed_s = 0.0
        state.set_external_approach(
            {
                "latitude": subject.latitude,
                "longitude": subject.longitude,
                "altitude_m": 0.0,
                "model": "IMM-PF TEST",
            }
        )

        for _ in range(59):
            state.tick(0.05)
        self.assertEqual("MIDCOURSE_APPROACH", state.flight_phase)

        for _ in range(2):
            state.tick(0.05)
        self.assertEqual("FINAL_APPROACH", state.flight_phase)
        self.assertEqual("AUTO HOLD", state.camera_mode)

    def test_single_launch_detection_auto_guidance_and_rendezvous(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        self.assertTrue(state.request_simulated_launch())
        self.assertTrue(state.mission_launched)
        self.assertEqual("ROUTE", state.flight_phase)
        self.assertFalse(state.request_simulated_launch())

        self.assertTrue(state.designate_subject(101))
        self.assertIsNone(state.predicted_approach())
        self.assertEqual("ARM", state.automatic_mode)
        self.assertEqual("INITIAL_APPROACH", state.flight_phase)
        self.assertEqual(INITIAL_APPROACH_SPEED_MPS, state.vehicle.speed_mps)
        self.assertFalse(state.request_simulated_launch())
        phases = {state.flight_phase}
        phase_speeds = {}
        for _ in range(1_000):
            state.tick(1.0)
            phases.add(state.flight_phase)
            phase_speeds.setdefault(state.flight_phase, state.vehicle.speed_mps)
            if state.approach_success:
                break
        self.assertIsNone(state.predicted_approach())
        self.assertIn("INITIAL_APPROACH", phases)
        self.assertIn("MIDCOURSE_APPROACH", phases)
        self.assertIn("FINAL_APPROACH", phases)
        self.assertEqual(
            MIDCOURSE_APPROACH_SPEED_MPS,
            phase_speeds["MIDCOURSE_APPROACH"],
        )
        self.assertTrue(
            MIDCOURSE_APPROACH_SPEED_MPS
            <= phase_speeds["FINAL_APPROACH"]
            < FINAL_APPROACH_SPEED_MPS,
        )
        self.assertAlmostEqual(
            FINAL_APPROACH_SPEED_MPS,
            state.vehicle.speed_mps,
            places=6,
        )
        self.assertTrue(state.approach_success)
        self.assertEqual("FOUND", state.flight_phase)
        self.assertTrue(all(state.mission_status.values()))
        self.assertIsNotNone(state.shutdown_position)
        self.assertAlmostEqual(
            state.vehicle.latitude,
            state.selected_subject.latitude,
            places=7,
        )
        self.assertAlmostEqual(
            state.vehicle.longitude,
            state.selected_subject.longitude,
            places=7,
        )
        self.assertEqual(0.0, state.vehicle.altitude_m)

    def test_terminal_speed_ramps_from_160_to_200_kph(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        state.request_simulated_launch()
        state.designate_subject(101)
        state._start_final_guidance()

        speeds = [state.vehicle.speed_mps]
        for _ in range(10):
            state._advance_approach(0.5)
            speeds.append(state.vehicle.speed_mps)

        self.assertAlmostEqual(160.0, speeds[0] * 3.6, places=5)
        self.assertTrue(
            all(later >= earlier for earlier, later in zip(speeds, speeds[1:]))
        )
        self.assertGreater(speeds[1], speeds[0])
        self.assertLess(speeds[1], speeds[-1])
        self.assertAlmostEqual(200.0, speeds[-1] * 3.6, places=5)

    def test_terminal_hit_requires_slant_range_to_close_with_altitude(
        self,
    ) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        state.request_simulated_launch()
        state.designate_subject(101)
        subject = state.selected_subject
        self.assertIsNotNone(subject)
        state.vehicle.latitude = subject.latitude
        state.vehicle.longitude = subject.longitude
        state.vehicle.altitude_m = 600.0
        subject.altitude_m = 0.0
        state._start_final_guidance()

        state._advance_approach(0.01)

        self.assertFalse(state.approach_success)
        self.assertLess(state.vehicle.altitude_m, 600.0)
        for _ in range(200):
            state._advance_approach(0.1)
            if state.approach_success:
                break
        self.assertTrue(state.approach_success)
        self.assertEqual("FOUND", state.flight_phase)

    def test_emergency_returns_straight_to_safe_zone_center(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        state.request_simulated_launch()
        state.tick(1.0)
        frozen = dict(state.mission_status)
        safe_center = store.zones[0].center()
        distance_before_emergency = horizontal_distance_m(
            state.vehicle.latitude,
            state.vehicle.longitude,
            safe_center.latitude,
            safe_center.longitude,
        )

        self.assertTrue(state.toggle_emergency())
        self.assertEqual("EMERGENCY_SAFE_RETURN", state.flight_phase)
        state.tick(0.25)
        distance_during_return = horizontal_distance_m(
            state.vehicle.latitude,
            state.vehicle.longitude,
            safe_center.latitude,
            safe_center.longitude,
        )
        self.assertGreater(distance_during_return, 0.0)
        self.assertLess(distance_during_return, distance_before_emergency)
        for _ in range(100):
            state.tick(1.0)
            if state.flight_phase == "RETURNED":
                break
        self.assertEqual(frozen, state.mission_status)
        self.assertEqual("RETURNED", state.flight_phase)
        return_distance = horizontal_distance_m(
            state.vehicle.latitude,
            state.vehicle.longitude,
            safe_center.latitude,
            safe_center.longitude,
        )
        self.assertLess(return_distance, 1.0)
        self.assertEqual(0.0, state.vehicle.speed_mps)
        self.assertFalse(state.toggle_emergency())
        self.assertEqual("ROUTE", state.flight_phase)
        self.assertEqual(SAR_CRUISE_SPEED_MPS, state.vehicle.speed_mps)

    def test_mitl_stop_returns_via_own_nearest_waypoint_then_safe_zone(
        self,
    ) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 3)
        state.request_simulated_launch()

        own_waypoint = store.waypoints_for(3)[2]
        state.vehicle.latitude = own_waypoint.latitude + 0.001
        state.vehicle.longitude = own_waypoint.longitude
        state.designate_subject(101)

        self.assertTrue(state.request_safe_return_via_waypoint())
        self.assertEqual("MANUAL_WAYPOINT_RETURN", state.flight_phase)
        payload = state.render_dict(store)
        via_waypoint = payload["mitl_return"]["via_waypoint"]
        self.assertEqual("WP003", via_waypoint["code"])
        self.assertAlmostEqual(
            own_waypoint.latitude,
            via_waypoint["latitude"],
            places=7,
        )
        self.assertIsNone(payload["predicted_approach"])

        for _ in range(10):
            state.tick(0.25)
            if state.flight_phase == "MANUAL_SAFE_RETURN":
                break
        self.assertEqual("MANUAL_SAFE_RETURN", state.flight_phase)
        self.assertAlmostEqual(
            own_waypoint.latitude,
            state.vehicle.latitude,
            places=7,
        )
        self.assertAlmostEqual(
            own_waypoint.longitude,
            state.vehicle.longitude,
            places=7,
        )

        for _ in range(500):
            state.tick(1.0)
            if state.flight_phase == "RETURNED":
                break
        safe_center = store.zones[0].center()
        self.assertEqual("RETURNED", state.flight_phase)
        self.assertLess(
            horizontal_distance_m(
                state.vehicle.latitude,
                state.vehicle.longitude,
                safe_center.latitude,
                safe_center.longitude,
            ),
            1.0,
        )
        self.assertEqual(0.0, state.vehicle.speed_mps)

    def test_payload_contains_plan_and_rendezvous(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)
        payload = state.render_dict(store)
        self.assertEqual(2, len(payload["subjects"]))
        self.assertEqual("UAV-01", payload["vehicle"]["code"])
        self.assertAlmostEqual(
            160.0 / 3.6,
            payload["vehicle"]["speed_mps"],
        )
        self.assertIsNone(payload["predicted_approach"])
        state.request_simulated_launch()
        state.designate_subject(101)
        payload = state.render_dict(store)
        self.assertIsNone(payload["predicted_approach"])
        self.assertEqual(3, len(payload["plan"]["sites"]))
        distances_km = [
            track["distance_m"] / 1000
            for track in payload["subjects"]
        ]
        self.assertGreaterEqual(min(distances_km), 31.9)
        self.assertLessEqual(max(distances_km), 40.1)
        nearest_route_distances = [
            min(
                horizontal_distance_m(
                    waypoint.latitude,
                    waypoint.longitude,
                    track.latitude,
                    track.longitude,
                )
                for route in store.vehicle_waypoints.values()
                for waypoint in route
            )
            for track in state.subjects
        ]
        self.assertTrue(
            all(distance <= 1_200.0 for distance in nearest_route_distances)
        )
        self.assertAlmostEqual(44.444444, SAR_CRUISE_SPEED_MPS, places=5)

    def test_demo_mission_rebases_subjects_and_completes_rendezvous(self) -> None:
        mission_path = (
            Path(__file__).parents[1] / "demo_rendezvous_mission.json"
        )
        store = SiteStore()
        store.load(mission_path)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store)

        launcher = store.sites["LC"]
        self.assertAlmostEqual(launcher.latitude, state.vehicle.latitude, places=7)
        self.assertAlmostEqual(launcher.longitude, state.vehicle.longitude, places=7)
        ranges_km = [
            horizontal_distance_m(
                launcher.latitude,
                launcher.longitude,
                track.latitude,
                track.longitude,
            )
            / 1000.0
            for track in state.subjects
        ]
        self.assertGreaterEqual(min(ranges_km), 31.9)
        self.assertLessEqual(max(ranges_km), 40.1)

        self.assertTrue(state.request_simulated_launch())
        self.assertTrue(state.designate_subject(101))
        phases = {state.flight_phase}
        for _ in range(1_000):
            state.tick(1.0)
            phases.add(state.flight_phase)
            if state.approach_success:
                break

        self.assertIn("INITIAL_APPROACH", phases)
        self.assertIn("MIDCOURSE_APPROACH", phases)
        self.assertIn("FINAL_APPROACH", phases)
        self.assertTrue(state.approach_success)
        self.assertEqual("FOUND", state.flight_phase)

    def test_saudi_scenario_starts_at_launch_and_follows_subject_profile(
        self,
    ) -> None:
        mission_path = (
            Path(__file__).parents[1] / "saudi_desert_mission.json"
        )
        store = SiteStore()
        store.load(mission_path)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        rally = store.waypoints_for(1)[0]
        arc_pattern = store.mission_metadata["arc_search_pattern"]
        for vehicle_id in SiteStore.VEHICLE_IDS:
            route = store.waypoints_for(vehicle_id)
            vehicle_rally = route[0]
            self.assertAlmostEqual(rally.latitude, vehicle_rally.latitude, places=7)
            self.assertAlmostEqual(rally.longitude, vehicle_rally.longitude, places=7)
            self.assertEqual(36, len(route))
            sector_center = (vehicle_id - 1) * 60.0
            actions = arc_pattern["vehicle_arc_sequences"][str(vehicle_id)]
            for action_index, action in enumerate(actions):
                arc = route[1 + action_index * 7 : 1 + (action_index + 1) * 7]
                self.assertEqual(7, len(arc))
                for waypoint in arc:
                    radius_m = horizontal_distance_m(
                        rally.latitude,
                        rally.longitude,
                        waypoint.latitude,
                        waypoint.longitude,
                    )
                    self.assertAlmostEqual(
                        float(action["radius_m"]),
                        radius_m,
                        delta=8.0,
                    )
                    waypoint_bearing = bearing_deg(
                        rally.latitude,
                        rally.longitude,
                        waypoint.latitude,
                        waypoint.longitude,
                    )
                    difference = (
                        waypoint_bearing - sector_center + 180.0
                    ) % 360.0 - 180.0
                    self.assertLessEqual(abs(difference), 30.1)
        predicted_subject = store.mission_metadata["rally_predicted_subject"]
        rally_standoff_m = horizontal_distance_m(
            rally.latitude,
            rally.longitude,
            float(predicted_subject["latitude"]),
            float(predicted_subject["longitude"]),
        )
        self.assertLess(rally_standoff_m, 1.0)
        self.assertAlmostEqual(
            100_000.0 / (8.0 * 60.0),
            SAR_INGRESS_SPEED_MPS,
        )
        arrival_state = FlyState.demo(23.65, 47.05)
        arrival_state.load_mission(store, 1)
        arrival_state.request_simulated_launch()
        for _ in range(49):
            arrival_state.tick(1.0)
        subject_offset_m = horizontal_distance_m(
            rally.latitude,
            rally.longitude,
            arrival_state.selected_subject.latitude,
            arrival_state.selected_subject.longitude,
        )
        self.assertTrue(arrival_state.search_started)
        self.assertAlmostEqual(
            3_333.3333333333,
            float(store.mission_metadata["search_radius_m"]),
        )
        self.assertEqual(300.0, store.mission_metadata["subject_lead_time_s"])
        self.assertGreater(subject_offset_m, 2_990.0)
        self.assertLess(subject_offset_m, 3_010.0)
        for _ in range(120):
            arrival_state.tick(1.0)
            orbit_offset_m = horizontal_distance_m(
                rally.latitude,
                rally.longitude,
                arrival_state.selected_subject.latitude,
                arrival_state.selected_subject.longitude,
            )
            self.assertAlmostEqual(3_000.0, orbit_offset_m, delta=3.0)
        initial_position = (
            state.selected_subject.latitude,
            state.selected_subject.longitude,
        )

        state.tick(1.0)
        self.assertEqual(0.0, state.simulation_elapsed_s)
        self.assertEqual(
            initial_position,
            (
                state.selected_subject.latitude,
                state.selected_subject.longitude,
            ),
        )
        self.assertTrue(state.request_simulated_launch())
        self.assertEqual(SAR_INGRESS_SPEED_MPS, state.vehicle.speed_mps)
        state.vehicle.latitude = rally.latitude
        state.vehicle.longitude = rally.longitude
        state.vehicle.altitude_m = rally.altitude_m
        state._advance_route(0.1)
        state._advance_route(0.1)
        self.assertEqual(SAR_SEARCH_SPEED_MPS, state.vehicle.speed_mps)
        for _ in range(67):
            state.tick(1.0)
        self.assertEqual(201.0, state.simulation_elapsed_s)
        self.assertAlmostEqual(12.0, state.selected_subject.speed_mps * 3.6)
        radial_bearing = bearing_deg(
            rally.latitude,
            rally.longitude,
            state.selected_subject.latitude,
            state.selected_subject.longitude,
        )
        self.assertAlmostEqual(
            (radial_bearing + 90.0) % 360.0,
            state.selected_subject.heading_deg,
            delta=0.1,
        )
        for _ in range(34):
            state.tick(1.0)
        self.assertEqual(303.0, state.simulation_elapsed_s)
        self.assertAlmostEqual(12.0, state.selected_subject.speed_mps * 3.6)
        five_minute_offset_m = horizontal_distance_m(
            rally.latitude,
            rally.longitude,
            state.selected_subject.latitude,
            state.selected_subject.longitude,
        )
        self.assertGreater(five_minute_offset_m, 2_990.0)
        self.assertLess(five_minute_offset_m, 3_010.0)


if __name__ == "__main__":
    unittest.main()
