"""End-to-end headless scenario integration tests.

These load each mission JSON, construct FlyState instances for 6 vehicles,
simulate launch → waypoint navigation → detection → 6-vehicle convergence →
FOUND, without any Qt UI widgets.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from qt_gcs.fly_state import (
    FlyState,
    horizontal_distance_m,
    SAR_SEARCH_SPEED_MPS,
)
from qt_gcs.planning.runtime import RuleBasedPlanningEngine
from qt_gcs.planning.sensor_model import SearchCameraSpec, build_footprint
from qt_gcs.site_store import SiteStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_store(mission_file: str) -> SiteStore:
    store = SiteStore()
    store.load(PROJECT_ROOT / mission_file)
    return store


def _build_states(store: SiteStore) -> dict[int, FlyState]:
    center = store.sites.get("GCS") or store.sites.get("LC")
    lat = center.latitude if center else 37.3422
    lon = center.longitude if center else 127.9202
    states = {}
    for vid in SiteStore.VEHICLE_IDS:
        s = FlyState.demo(lat, lon)
        s.load_mission(store, vid)
        states[vid] = s
    return states


def _launch_all(states: dict[int, FlyState]) -> None:
    for s in states.values():
        if s.mission_loaded:
            s.request_simulated_launch()


def _tick_all(states: dict[int, FlyState], dt: float = 1.0, n: int = 1) -> None:
    for _ in range(n):
        for s in states.values():
            s.tick(dt)


def _advance_to_first_waypoint(
    states: dict[int, FlyState],
    max_ticks: int = 3000,
) -> bool:
    """Tick until at least one vehicle reaches its first waypoint."""
    for _ in range(max_ticks):
        _tick_all(states, dt=1.0)
        if any(s._search_started for s in states.values() if s.mission_launched):
            return True
    return False


def _move_vehicle_near_subject(state: FlyState, offset_m: float = 50.0) -> None:
    """Teleport vehicle near its selected subject for detection testing."""
    subj = state.selected_subject
    if subj is None:
        return
    lat_offset = offset_m / 111_320.0
    state.vehicle.latitude = subj.latitude + lat_offset
    state.vehicle.longitude = subj.longitude
    state.vehicle.altitude_m = 1130.0


class MissionLoadTests(unittest.TestCase):
    """Verify that each mission JSON file loads without error."""

    def test_load_continuous_move_6uav(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        self.assertTrue(store.is_mission_ready)
        self.assertEqual(len(store.configured_vehicle_ids), 6)
        self.assertIn("GCS", store.sites)
        self.assertIn("RDR", store.sites)
        self.assertIn("LC", store.sites)

    def test_load_single_uav(self) -> None:
        store = _load_store("single_uav_mission.json")
        self.assertTrue(store.shared_configuration_ready)
        self.assertIn("GCS", store.sites)
        # SAR-01 has waypoints, SAR-02~06 have empty arrays
        self.assertTrue(len(store.waypoints_for(1)) > 0)

    def test_load_ingress_sweep(self) -> None:
        store = _load_store("ingress_sweep_6uav_mission.json")
        self.assertTrue(store.is_mission_ready)
        self.assertEqual(len(store.configured_vehicle_ids), 6)
        metadata = store.mission_metadata
        self.assertTrue(metadata.get("ingress_sweep", False))

    def test_load_multi_subject(self) -> None:
        store = _load_store("multi_subject_3_mission.json")
        self.assertTrue(store.is_mission_ready)
        self.assertEqual(len(store.initial_subjects), 3)
        types = {s.subject_type for s in store.initial_subjects}
        self.assertEqual(
            types,
            {"MISSING_PERSON_CHILD", "MISSING_PERSON_ADULT", "MISSING_PERSON_ELDERLY"},
        )


class SubjectGroundAltitudeTests(unittest.TestCase):
    """Verify subjects are at ground level (altitude_m == 0)."""

    def _check_subjects_at_ground(self, mission_file: str) -> None:
        store = _load_store(mission_file)
        states = _build_states(store)
        for vid, state in states.items():
            for subj in state.subjects:
                self.assertEqual(
                    subj.altitude_m,
                    0,
                    f"Subject {subj.track_id} in {mission_file} "
                    f"vehicle {vid} has altitude_m={subj.altitude_m}, "
                    f"expected 0 (ground level)",
                )

    def test_continuous_move_subjects_at_ground(self) -> None:
        self._check_subjects_at_ground("continuous_move_mission_6uav.json")

    def test_single_uav_subjects_at_ground(self) -> None:
        self._check_subjects_at_ground("single_uav_mission.json")

    def test_ingress_subjects_at_ground(self) -> None:
        self._check_subjects_at_ground("ingress_sweep_6uav_mission.json")

    def test_multi_subject_subjects_at_ground(self) -> None:
        self._check_subjects_at_ground("multi_subject_3_mission.json")


class FlightAltitudeTests(unittest.TestCase):
    """Verify flight altitude is terrain + ~600m AGL."""

    def test_continuous_move_altitude(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        alt = store.mission_metadata.get("flight_altitude_msl_m", 0)
        terrain = store.mission_metadata.get("terrain_reference", {})
        ground_m = terrain.get("aoi_ground_center_m_msl", 0)
        agl = alt - ground_m
        self.assertAlmostEqual(agl, 601.0, delta=10.0)

    def test_waypoint_altitudes_are_consistent(self) -> None:
        """Waypoints within a mission should have a consistent altitude."""
        store = _load_store("continuous_move_mission_6uav.json")
        altitudes = set()
        for vid in SiteStore.VEHICLE_IDS:
            for wp in store.waypoints_for(vid):
                altitudes.add(round(wp.altitude_m, 1))
        # All waypoints should share the same altitude
        self.assertEqual(
            len(altitudes),
            1,
            f"Expected uniform waypoint altitude, got {altitudes}",
        )
        # The waypoint altitude should be positive (above ground)
        wp_alt = altitudes.pop()
        self.assertGreater(wp_alt, 0.0)


class TPDisplayTests(unittest.TestCase):
    """Verify rally_predicted_subject (TP) exists and is valid."""

    def _check_tp(self, mission_file: str) -> None:
        store = _load_store(mission_file)
        metadata = store.mission_metadata
        rp = metadata.get("rally_predicted_subject")
        self.assertIsNotNone(
            rp,
            f"No rally_predicted_subject in {mission_file}",
        )
        self.assertIn("latitude", rp)
        self.assertIn("longitude", rp)
        self.assertIsInstance(rp["latitude"], (int, float))
        self.assertIsInstance(rp["longitude"], (int, float))

    def test_continuous_move_has_tp(self) -> None:
        self._check_tp("continuous_move_mission_6uav.json")

    def test_single_uav_has_tp(self) -> None:
        self._check_tp("single_uav_mission.json")

    def test_ingress_sweep_has_tp(self) -> None:
        self._check_tp("ingress_sweep_6uav_mission.json")

    def test_multi_subject_has_tp(self) -> None:
        self._check_tp("multi_subject_3_mission.json")


class LaunchAndRouteTests(unittest.TestCase):
    """Verify vehicles can launch and navigate waypoints."""

    def test_6uav_all_launch(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        for vid, state in states.items():
            self.assertTrue(
                state.mission_launched,
                f"SAR-{vid:02d} did not launch",
            )
            self.assertEqual(state.flight_phase, "ROUTE")

    def test_single_uav_only_sar01_launches(self) -> None:
        store = _load_store("single_uav_mission.json")
        states = _build_states(store)
        _launch_all(states)
        self.assertTrue(states[1].mission_launched)
        self.assertEqual(states[1].flight_phase, "ROUTE")
        # Other vehicles can't launch (no waypoints)
        for vid in range(2, 7):
            self.assertFalse(states[vid].mission_launched)

    def test_search_starts_at_first_waypoint(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        self.assertFalse(states[1]._search_started)
        reached = _advance_to_first_waypoint(states)
        self.assertTrue(reached, "No vehicle reached first waypoint")
        any_searching = any(
            s._search_started for s in states.values()
        )
        self.assertTrue(any_searching)


class DesignateSubjectConvergenceTests(unittest.TestCase):
    """Verify 6-vehicle convergence on designate_subject."""

    def test_designate_triggers_all_six_approach(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        # Tick a few to get flying
        _tick_all(states, dt=1.0, n=5)
        track_id = states[1].selected_track_id
        # Simulate detection: designate on all 6 states (as _approach_detected_subject does)
        canonical = next(
            t for t in states[1].subjects if t.track_id == track_id
        )
        for state in states.values():
            local = next(
                (t for t in state.subjects if t.track_id == track_id),
                None,
            )
            if local is not None:
                local.latitude = canonical.latitude
                local.longitude = canonical.longitude
                local.altitude_m = canonical.altitude_m
        for state in states.values():
            result = state.designate_subject(
                track_id, detector_vehicle_id=1
            )
            self.assertTrue(result, "designate_subject failed")
        # All 6 vehicles must now be in INITIAL_APPROACH
        for vid, state in states.items():
            self.assertTrue(
                state.subject_detected,
                f"SAR-{vid:02d} not detected",
            )
            self.assertEqual(
                state.flight_phase,
                "INITIAL_APPROACH",
                f"SAR-{vid:02d} phase={state.flight_phase}",
            )
            self.assertEqual(
                state.selected_track_id,
                track_id,
                f"SAR-{vid:02d} tracking wrong subject",
            )

    def test_approach_to_found(self) -> None:
        """Tick from INITIAL_APPROACH through FINAL_APPROACH to FOUND."""
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        _tick_all(states, dt=1.0, n=5)
        track_id = states[1].selected_track_id
        # Move all vehicles very close to subject for fast convergence
        canonical = next(
            t for t in states[1].subjects if t.track_id == track_id
        )
        for state in states.values():
            state.vehicle.latitude = canonical.latitude + 0.001
            state.vehicle.longitude = canonical.longitude
            state.vehicle.altitude_m = 1130.0
            local = next(
                (t for t in state.subjects if t.track_id == track_id),
                None,
            )
            if local:
                local.latitude = canonical.latitude
                local.longitude = canonical.longitude
                local.altitude_m = canonical.altitude_m
            state.designate_subject(track_id, detector_vehicle_id=1)
        # Tick until FOUND
        found_count = 0
        for _ in range(2000):
            _tick_all(states, dt=1.0)
            found_count = sum(
                1 for s in states.values()
                if s.subject_found and s.mission_launched
            )
            if found_count == 6:
                break
        self.assertEqual(
            found_count,
            6,
            f"Only {found_count}/6 vehicles reached FOUND. "
            f"Phases: {[(v, s.flight_phase) for v, s in states.items()]}",
        )
        for vid, state in states.items():
            self.assertEqual(state.flight_phase, "FOUND")
            subj = state.selected_subject
            self.assertIsNotNone(subj)
            self.assertTrue(subj.found)
            # Vehicle should be at subject position (ground level)
            self.assertAlmostEqual(
                state.vehicle.altitude_m,
                subj.altitude_m,
                delta=1.0,
                msg=f"SAR-{vid:02d} not at subject altitude",
            )


class DetectionMechanismTests(unittest.TestCase):
    """Verify the sensor detection logic (600m reach + 95m footprint)."""

    def test_detection_within_sensor_reach(self) -> None:
        spec = SearchCameraSpec()
        self.assertAlmostEqual(spec.gimbal_centerline_reach_m, 600.0, delta=5.0)
        self.assertAlmostEqual(spec.effective_sweep_width_m, 400.0, delta=1.0)
        self.assertAlmostEqual(spec.ideal_detection_radius_m, 200.0, delta=1.0)
        self.assertEqual(spec.detection_probability, 1.0)

    def test_planning_engine_detects_subject_in_range(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        spec = SearchCameraSpec()
        center = store.sites["GCS"]
        # Use minimum_route_updates_before_detection=0 to bypass the gate
        engine = RuleBasedPlanningEngine(
            center.latitude,
            center.longitude,
            search_camera=spec,
            search_radius_m=float(
                store.mission_metadata.get("search_radius_m", 8900.0)
            ),
            minimum_route_updates_before_detection=0,
        )
        # Place SAR-01 right next to subject
        subj = states[1].subjects[0]
        states[1].vehicle.latitude = subj.latitude + 0.0003
        states[1].vehicle.longitude = subj.longitude
        states[1].vehicle.altitude_m = 1130.0
        states[1]._search_started = True
        states[1].flight_phase = "ROUTE"

        def _vehicle_dict(vid, s):
            return {
                "vehicle_id": vid,
                "latitude": s.vehicle.latitude,
                "longitude": s.vehicle.longitude,
                "altitude_m": s.vehicle.altitude_m,
                "speed_mps": s.vehicle.speed_mps,
                "heading_deg": s.vehicle.heading_deg,
                "mission_launched": s.mission_launched,
                "emergency_mode": s.emergency_mode,
                "flight_phase": s.flight_phase,
                "completed_route_segment_count": s.completed_route_segment_count,
                "search_started": s._search_started,
                "runtime_route_revision": s.runtime_route_revision,
                "runtime_route_update_count": s.runtime_route_update_count,
                "ingress_sweep": False,
            }

        vehicles = [_vehicle_dict(vid, s) for vid, s in states.items()]
        subjects_data = [
            {
                "track_id": t.track_id,
                "latitude": t.latitude,
                "longitude": t.longitude,
                "altitude_m": t.altitude_m,
                "speed_mps": t.speed_mps,
                "heading_deg": t.heading_deg,
                "position_uncertainty_m": t.position_uncertainty_m,
                "found": t.found,
                "measurement_latitude": t.latitude,
                "measurement_longitude": t.longitude,
                "estimator_speed_mps": t.speed_mps,
                "static_observation_measurement": False,
            }
            for t in states[1].subjects
        ]
        routes = {
            vid: s.route_points_payload()
            for vid, s in states.items()
        }
        # First update initializes the engine (previous_elapsed_s=None)
        engine.update(
            elapsed_s=0.0,
            vehicles=vehicles,
            subjects=subjects_data,
            routes=routes,
            selected_track_id=states[1].selected_track_id,
            approach_track_id=None,
        )
        # Second update with dt>0 triggers detection
        result = engine.update(
            elapsed_s=1.0,
            vehicles=vehicles,
            subjects=subjects_data,
            routes=routes,
            selected_track_id=states[1].selected_track_id,
            approach_track_id=None,
        )
        self.assertTrue(
            len(result.detections) > 0,
            "No detection when vehicle is within sensor reach",
        )

    def test_no_detection_outside_sensor_reach(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        states = _build_states(store)
        _launch_all(states)
        spec = SearchCameraSpec()
        center = store.sites["GCS"]
        engine = RuleBasedPlanningEngine(
            center.latitude,
            center.longitude,
            search_camera=spec,
            search_radius_m=8900.0,
            minimum_route_updates_before_detection=0,
        )
        # Place SAR-01 far from subject (2km away)
        subj = states[1].subjects[0]
        states[1].vehicle.latitude = subj.latitude + 0.02  # ~2.2km
        states[1].vehicle.longitude = subj.longitude
        states[1].vehicle.altitude_m = 1130.0
        states[1]._search_started = True
        states[1].flight_phase = "ROUTE"

        def _vehicle_dict(vid, s):
            return {
                "vehicle_id": vid,
                "latitude": s.vehicle.latitude,
                "longitude": s.vehicle.longitude,
                "altitude_m": s.vehicle.altitude_m,
                "speed_mps": s.vehicle.speed_mps,
                "heading_deg": s.vehicle.heading_deg,
                "mission_launched": s.mission_launched,
                "emergency_mode": s.emergency_mode,
                "flight_phase": s.flight_phase,
                "completed_route_segment_count": s.completed_route_segment_count,
                "search_started": s._search_started,
                "runtime_route_revision": s.runtime_route_revision,
                "runtime_route_update_count": s.runtime_route_update_count,
                "ingress_sweep": False,
            }

        vehicles = [_vehicle_dict(vid, s) for vid, s in states.items()]
        subjects_data = [
            {
                "track_id": t.track_id,
                "latitude": t.latitude,
                "longitude": t.longitude,
                "altitude_m": t.altitude_m,
                "speed_mps": t.speed_mps,
                "heading_deg": t.heading_deg,
                "position_uncertainty_m": t.position_uncertainty_m,
                "found": t.found,
                "measurement_latitude": t.latitude,
                "measurement_longitude": t.longitude,
                "estimator_speed_mps": t.speed_mps,
                "static_observation_measurement": False,
            }
            for t in states[1].subjects
        ]
        routes = {
            vid: s.route_points_payload()
            for vid, s in states.items()
        }
        engine.update(
            elapsed_s=0.0,
            vehicles=vehicles,
            subjects=subjects_data,
            routes=routes,
            selected_track_id=states[1].selected_track_id,
            approach_track_id=None,
        )
        result = engine.update(
            elapsed_s=1.0,
            vehicles=vehicles,
            subjects=subjects_data,
            routes=routes,
            selected_track_id=states[1].selected_track_id,
            approach_track_id=None,
        )
        self.assertEqual(
            len(result.detections),
            0,
            "Detection should not occur when vehicle is 2km from subject",
        )


class MultiSubjectResumeSearchTests(unittest.TestCase):
    """Verify multi-subject scenario resume-search flow."""

    def test_multi_subject_types_loaded(self) -> None:
        store = _load_store("multi_subject_3_mission.json")
        states = _build_states(store)
        for state in states.values():
            types = {s.subject_type for s in state.subjects}
            self.assertIn("MISSING_PERSON_CHILD", types)
            self.assertIn("MISSING_PERSON_ADULT", types)
            self.assertIn("MISSING_PERSON_ELDERLY", types)

    def test_stationary_subject_speed_zero(self) -> None:
        store = _load_store("multi_subject_3_mission.json")
        for subj in store.initial_subjects:
            if subj.track_id == 103:  # ELDERLY, STATIONARY
                self.assertAlmostEqual(
                    subj.speed_mps,
                    0.0,
                    delta=0.01,
                    msg="Stationary subject 103 should have speed 0",
                )

    def test_resume_search_transitions_found_to_route(self) -> None:
        store = _load_store("multi_subject_3_mission.json")
        states = _build_states(store)
        _launch_all(states)
        _tick_all(states, dt=1.0, n=5)
        # Manually mark first subject as found on all states
        first_track_id = states[1].selected_track_id
        for state in states.values():
            if not state.mission_launched:
                continue
            subj = next(
                (t for t in state.subjects if t.track_id == first_track_id),
                None,
            )
            if subj:
                subj.found = True
            state.subject_found = True
            state.flight_phase = "FOUND"
        # resume_search should transition back to ROUTE
        for state in states.values():
            if state.mission_launched:
                result = state.resume_search()
                self.assertTrue(result, "resume_search should succeed")
                self.assertEqual(state.flight_phase, "ROUTE")
                self.assertFalse(state.subject_found)


class IngressSweepTests(unittest.TestCase):
    """Verify ingress sweep scenario behavior."""

    def test_ingress_sweep_flag_in_metadata(self) -> None:
        store = _load_store("ingress_sweep_6uav_mission.json")
        self.assertTrue(store.mission_metadata.get("ingress_sweep", False))

    def test_ingress_sweep_active_before_search_started(self) -> None:
        store = _load_store("ingress_sweep_6uav_mission.json")
        states = _build_states(store)
        _launch_all(states)
        # Before reaching first waypoint, ingress_sweep_active should be True
        for vid, state in states.items():
            if state.mission_launched:
                self.assertTrue(
                    state.ingress_sweep_active,
                    f"SAR-{vid:02d} ingress_sweep should be active before first WP",
                )

    def test_ingress_sweep_off_after_search_started(self) -> None:
        store = _load_store("ingress_sweep_6uav_mission.json")
        states = _build_states(store)
        _launch_all(states)
        # Manually set search_started
        for state in states.values():
            if state.mission_launched:
                state._search_started = True
        for vid, state in states.items():
            if state.mission_launched:
                self.assertFalse(
                    state.ingress_sweep_active,
                    f"SAR-{vid:02d} ingress_sweep should be OFF after search_started",
                )


class SPXProbabilityTests(unittest.TestCase):
    """Verify SPX detection probability is baked in mission metadata."""

    def test_baked_spx_pd_exists(self) -> None:
        store = _load_store("continuous_move_mission_6uav.json")
        spx = store.mission_metadata.get("spx_certified", {})
        # If SPX baked data exists, it should have detection_probability
        if spx:
            self.assertIn("detection_probability", spx)
            pd = spx["detection_probability"]
            self.assertGreater(pd, 0.0)
            self.assertLessEqual(pd, 1.0)


class SingleUAVScenarioTests(unittest.TestCase):
    """Verify single-UAV scenario end-to-end."""

    def test_single_uav_route_navigation(self) -> None:
        store = _load_store("single_uav_mission.json")
        states = _build_states(store)
        _launch_all(states)
        self.assertTrue(states[1].mission_launched)
        self.assertEqual(states[1].flight_phase, "ROUTE")
        # Tick a few times to verify route navigation works
        _tick_all(states, dt=1.0, n=10)
        # SAR-01 should still be in ROUTE (navigating waypoints)
        self.assertEqual(states[1].flight_phase, "ROUTE")

    def test_single_uav_designate_and_approach(self) -> None:
        store = _load_store("single_uav_mission.json")
        states = _build_states(store)
        _launch_all(states)
        _tick_all(states, dt=1.0, n=5)
        track_id = states[1].selected_track_id
        # Designate subject on SAR-01
        subj = next(t for t in states[1].subjects if t.track_id == track_id)
        states[1].vehicle.latitude = subj.latitude + 0.001
        states[1].vehicle.longitude = subj.longitude
        states[1].vehicle.altitude_m = 1130.0
        result = states[1].designate_subject(
            track_id, detector_vehicle_id=1
        )
        self.assertTrue(result)
        self.assertEqual(states[1].flight_phase, "INITIAL_APPROACH")
        # Tick to FOUND
        for _ in range(2000):
            states[1].tick(1.0)
            if states[1].subject_found:
                break
        self.assertTrue(
            states[1].subject_found,
            f"SAR-01 did not reach FOUND. Phase={states[1].flight_phase}",
        )


class SequentialOutsideCircleTests(unittest.TestCase):
    """Scenario (2): find subject #1 inside the circle, then re-center the
    search onto subject #2 which lies outside it.

    Mirrors fly_view._recenter_search_on_subject: the resume step moves BOTH
    predicted-subject center fields (rally_predicted_subject and
    arc_search_pattern.center) onto the next subject's zone so the belief engine
    and the SPX re-solve both follow the fleet out of the original circle.
    """

    def _load_seq(self) -> SiteStore:
        return _load_store("multi_subject_sequential_mission.json")

    def _recenter(self, store: SiteStore, subject) -> None:
        nc = {"latitude": subject.latitude, "longitude": subject.longitude}
        meta = store.mission_metadata
        meta["rally_predicted_subject"] = nc
        arc = meta.get("arc_search_pattern")
        if isinstance(arc, dict):
            arc["center"] = dict(nc)

    def test_subject_one_inside_two_outside(self) -> None:
        store = self._load_seq()
        meta = store.mission_metadata
        radius = float(meta["search_radius_m"])
        center = meta["rally_predicted_subject"]
        subs = {s.track_id: s for s in store.initial_subjects}
        self.assertEqual({101, 102}, set(subs))
        d1 = horizontal_distance_m(
            center["latitude"], center["longitude"],
            subs[101].latitude, subs[101].longitude,
        )
        d2 = horizontal_distance_m(
            center["latitude"], center["longitude"],
            subs[102].latitude, subs[102].longitude,
        )
        self.assertLessEqual(d1, radius)
        self.assertGreater(d2, radius)

    def test_recenter_moves_search_center_onto_outside_subject(self) -> None:
        from qt_gcs.planning.stone_adapter import _search_center

        store = self._load_seq()
        subs = {s.track_id: s for s in store.initial_subjects}
        before = _search_center(store)
        self._recenter(store, subs[102])
        after = _search_center(store)
        # Center must jump to subject #2 (both metadata fields honoured).
        self.assertGreater(
            horizontal_distance_m(before[0], before[1], after[0], after[1]),
            float(store.mission_metadata["search_radius_m"]),
        )
        self.assertLess(
            horizontal_distance_m(
                after[0], after[1], subs[102].latitude, subs[102].longitude
            ),
            1.0,
        )

    def test_recentered_engine_belief_follows_to_outside_zone(self) -> None:
        store = self._load_seq()
        subs = {s.track_id: s for s in store.initial_subjects}
        gcs = store.sites["GCS"]
        self._recenter(store, subs[102])
        meta = store.mission_metadata
        tp = meta["rally_predicted_subject"]
        engine = RuleBasedPlanningEngine(
            gcs.latitude, gcs.longitude,
            search_center_latitude=tp["latitude"],
            search_center_longitude=tp["longitude"],
            search_radius_m=float(meta["search_radius_m"]),
        )
        c_lat, c_lon = engine.frame.to_geographic(engine.search_center)
        self.assertLess(
            horizontal_distance_m(
                c_lat, c_lon, subs[102].latitude, subs[102].longitude
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
