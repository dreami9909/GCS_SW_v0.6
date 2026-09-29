"""Stage 2: the parallel ingress sweep detects subjects before TP arrival.

A vehicle marked ``ingress_sweep`` must sense (footprints + detection) even
though ``search_started`` is still False (the RHP/PF clock starts at TP).
"""

from __future__ import annotations

import unittest

from qt_gcs.planning import RuleBasedPlanningEngine, build_footprint


def _vehicle(lat, lon, *, search_started, ingress_sweep):
    return {
        "vehicle_id": 1, "latitude": lat, "longitude": lon, "altitude_m": 600.0,
        "speed_mps": 160.0 / 3.6, "heading_deg": 0.0, "mission_launched": True,
        "emergency_mode": False, "flight_phase": "ROUTE",
        "search_started": search_started, "ingress_sweep": ingress_sweep,
        "runtime_route_revision": 0,
    }


def _subject(lat, lon):
    return {"track_id": 101, "subject_type": "MISSING_PERSON_CHILD",
            "latitude": lat, "longitude": lon, "speed_mps": 0.0, "heading_deg": 0.0}


class IngressDetectionTests(unittest.TestCase):
    LAT, LON = 37.4, 127.9

    def _subject_in_footprint(self, engine):
        fp = build_footprint(engine.search_camera, engine.frame, vehicle_id=1,
                             latitude=self.LAT, longitude=self.LON, altitude_m=600.0,
                             heading_deg=0.0, elapsed_s=0.0)
        return engine.frame.to_geographic(fp.center)

    def test_ingress_sweep_detects_before_tp(self) -> None:
        engine = RuleBasedPlanningEngine(self.LAT, self.LON, particle_count=120)
        tlat, tlon = self._subject_in_footprint(engine)
        result = engine.update(
            elapsed_s=0.0,
            vehicles=[_vehicle(self.LAT, self.LON, search_started=False, ingress_sweep=True)],
            subjects=[_subject(tlat, tlon)],
            routes={1: [{"latitude": self.LAT, "longitude": self.LON}]},
            selected_track_id=101, approach_track_id=None,
        )
        self.assertEqual(1, len(result.detections))
        self.assertEqual(101, result.detections[0]["track_id"])

    def test_no_sweep_no_search_means_no_detection(self) -> None:
        engine = RuleBasedPlanningEngine(self.LAT, self.LON, particle_count=120)
        tlat, tlon = self._subject_in_footprint(engine)
        result = engine.update(
            elapsed_s=0.0,
            vehicles=[_vehicle(self.LAT, self.LON, search_started=False, ingress_sweep=False)],
            subjects=[_subject(tlat, tlon)],
            routes={1: [{"latitude": self.LAT, "longitude": self.LON}]},
            selected_track_id=101, approach_track_id=None,
        )
        self.assertFalse(result.detections)


if __name__ == "__main__":
    unittest.main()


class IngressSweepStateTests(unittest.TestCase):
    def _launched_state(self):
        from qt_gcs.fly_state import FlyState
        from qt_gcs.site_store import SiteStore
        store = SiteStore(); store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        return state

    def test_sweep_active_during_ingress_route_before_tp(self) -> None:
        state = self._launched_state()
        self.assertEqual("ROUTE", state.flight_phase)
        self.assertFalse(state.search_started)
        self.assertTrue(state.ingress_sweep_active)

    def test_sweep_ends_once_search_starts(self) -> None:
        state = self._launched_state()
        state._search_started = True  # TP reached -> search clock running
        self.assertFalse(state.ingress_sweep_active)

    def test_sweep_inactive_before_launch(self) -> None:
        from qt_gcs.fly_state import FlyState
        state = FlyState.demo(37.3422, 127.9202)
        self.assertFalse(state.ingress_sweep_active)


class IngressDetectionApproachTests(unittest.TestCase):
    """End-to-end: ingress detection triggers cooperative approach for all 6."""

    LAT, LON = 37.4, 127.9

    def test_ingress_detection_produces_detection_event(self) -> None:
        """A subject in the footprint during ingress sweep produces a detection."""
        engine = RuleBasedPlanningEngine(self.LAT, self.LON, particle_count=120)
        fp = build_footprint(
            engine.search_camera, engine.frame, vehicle_id=1,
            latitude=self.LAT, longitude=self.LON, altitude_m=600.0,
            heading_deg=0.0, elapsed_s=0.0,
        )
        tlat, tlon = engine.frame.to_geographic(fp.center)
        result = engine.update(
            elapsed_s=0.0,
            vehicles=[{
                "vehicle_id": 1, "latitude": self.LAT, "longitude": self.LON,
                "altitude_m": 600.0, "speed_mps": 44.4, "heading_deg": 0.0,
                "mission_launched": True, "emergency_mode": False,
                "flight_phase": "ROUTE", "search_started": False,
                "ingress_sweep": True, "runtime_route_revision": 0,
            }],
            subjects=[{
                "track_id": 101, "latitude": tlat, "longitude": tlon,
                "altitude_m": 0.0, "speed_mps": 0.0, "heading_deg": 0.0,
                "position_uncertainty_m": 500.0, "found": False,
                "measurement_latitude": tlat, "measurement_longitude": tlon,
                "estimator_speed_mps": 0.0, "static_observation_measurement": False,
            }],
            routes={1: [{"latitude": self.LAT, "longitude": self.LON}]},
            selected_track_id=101,
            approach_track_id=None,
        )
        self.assertTrue(len(result.detections) >= 1)
        self.assertEqual(101, result.detections[0]["track_id"])

    def test_designate_subject_switches_to_approach(self) -> None:
        """designate_subject during ingress -> INITIAL_APPROACH for the vehicle."""
        from qt_gcs.fly_state import FlyState
        from qt_gcs.site_store import SiteStore
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        # Vehicle is in ROUTE / ingress_sweep_active
        self.assertEqual("ROUTE", state.flight_phase)
        self.assertTrue(state.ingress_sweep_active)

        # Detection event -> designate for approach
        state.designate_subject(101, detector_vehicle_id=1)
        self.assertEqual("INITIAL_APPROACH", state.flight_phase)
        self.assertTrue(state.subject_detected)
        self.assertTrue(state.approach_requested)
        # Ingress sweep ends (not in ROUTE anymore)
        self.assertFalse(state.ingress_sweep_active)

    def test_cooperative_approach_all_six_from_ingress(self) -> None:
        """All 6 vehicles switch to INITIAL_APPROACH when one detects during ingress."""
        from qt_gcs.fly_state import FlyState
        from qt_gcs.site_store import SiteStore
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        states = {}
        for vid in SiteStore.VEHICLE_IDS:
            s = FlyState.demo(37.3422, 127.9202)
            s.load_mission(store, vid)
            s.request_simulated_launch()
            states[vid] = s

        # All in ingress
        for s in states.values():
            self.assertTrue(s.ingress_sweep_active)

        # Vehicle 3 detects -> all 6 designate
        for s in states.values():
            s.designate_subject(101, detector_vehicle_id=3)

        for vid, s in states.items():
            self.assertEqual(
                "INITIAL_APPROACH", s.flight_phase,
                f"SAR-{vid:02d} should be in INITIAL_APPROACH",
            )
            self.assertTrue(s.subject_detected)
            self.assertFalse(s.ingress_sweep_active)

    def test_no_detection_without_ingress_sweep_flag(self) -> None:
        """Without ingress_sweep=True, no detection before search_started."""
        engine = RuleBasedPlanningEngine(self.LAT, self.LON, particle_count=120)
        fp = build_footprint(
            engine.search_camera, engine.frame, vehicle_id=1,
            latitude=self.LAT, longitude=self.LON, altitude_m=600.0,
            heading_deg=0.0, elapsed_s=0.0,
        )
        tlat, tlon = engine.frame.to_geographic(fp.center)
        result = engine.update(
            elapsed_s=0.0,
            vehicles=[{
                "vehicle_id": 1, "latitude": self.LAT, "longitude": self.LON,
                "altitude_m": 600.0, "speed_mps": 44.4, "heading_deg": 0.0,
                "mission_launched": True, "emergency_mode": False,
                "flight_phase": "ROUTE", "search_started": False,
                "ingress_sweep": False, "runtime_route_revision": 0,
            }],
            subjects=[{
                "track_id": 101, "latitude": tlat, "longitude": tlon,
                "altitude_m": 0.0, "speed_mps": 0.0, "heading_deg": 0.0,
                "position_uncertainty_m": 500.0, "found": False,
                "measurement_latitude": tlat, "measurement_longitude": tlon,
                "estimator_speed_mps": 0.0, "static_observation_measurement": False,
            }],
            routes={1: [{"latitude": self.LAT, "longitude": self.LON}]},
            selected_track_id=101,
            approach_track_id=None,
        )
        self.assertFalse(result.detections)
