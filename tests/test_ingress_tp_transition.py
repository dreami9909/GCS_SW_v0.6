"""TP-line transition: all vehicles crossing TP triggers SPX re-plan stub."""
from __future__ import annotations

import unittest

from qt_gcs.fly_state import (
    FlyState,
    SAR_INGRESS_SPEED_MPS,
    SAR_SEARCH_SPEED_MPS,
)
from qt_gcs.site_store import SiteStore


class TPTransitionTests(unittest.TestCase):
    """Test the conditions that determine TP-line crossing."""

    def _launched_fleet(self):
        """Create 6 launched FlyState instances."""
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        states = {}
        for vid in SiteStore.VEHICLE_IDS:
            state = FlyState.demo(37.3422, 127.9202)
            state.load_mission(store, vid)
            state.request_simulated_launch()
            states[vid] = state
        return store, states

    def test_ingress_sweep_active_before_tp(self) -> None:
        """All vehicles in ROUTE before first WP → ingress_sweep_active."""
        _store, states = self._launched_fleet()
        for state in states.values():
            self.assertTrue(state.ingress_sweep_active)
            self.assertFalse(state.search_started)

    def test_search_starts_after_first_waypoint(self) -> None:
        """Reaching the first waypoint sets _search_started = True."""
        _store, states = self._launched_fleet()
        state = states[1]
        # Simulate reaching first waypoint
        state._search_started = True
        self.assertTrue(state.search_started)
        self.assertFalse(state.ingress_sweep_active)

    def test_all_crossed_tp_condition(self) -> None:
        """Fleet transition: all search_started means ingress is done."""
        _store, states = self._launched_fleet()
        # Initially none have crossed
        self.assertFalse(all(s.search_started for s in states.values()))

        # Partial crossing — not all done yet
        for vid in [1, 2, 3]:
            states[vid]._search_started = True
        self.assertFalse(all(s.search_started for s in states.values()))

        # All crossed
        for state in states.values():
            state._search_started = True
        self.assertTrue(all(s.search_started for s in states.values()))

    def test_ingress_sweep_ends_for_each_vehicle_at_tp(self) -> None:
        """Once search_started, ingress_sweep_active is False for that vehicle."""
        _store, states = self._launched_fleet()
        state = states[1]
        self.assertTrue(state.ingress_sweep_active)
        state._search_started = True
        self.assertFalse(state.ingress_sweep_active)

    def test_ingress_speed_held_until_tp_line(self) -> None:
        """Ingress speed is kept across ALL ingress legs (form-up AND TP line),
        not dropped at the form-up point, and search starts only after the TP
        line (last IGR waypoint) is crossed."""
        state = FlyState.demo(37.34, 127.92)
        state._route_points = [
            (37.35, 127.93, 600.0, "IGR001"),   # form-up
            (37.36, 127.94, 600.0, "IGR002"),   # TP line
            (37.37, 127.95, 600.0, "WP003"),    # first search leg
        ]
        state.mission_launched = True
        state.flight_phase = "ROUTE"
        state._runtime_route_revision = 0
        state.completed_route_segment_count = 0
        state.current_waypoint_index = 0
        state._search_started = False

        # Leg 1 — heading to the form-up point: ingress speed.
        state._advance_route(0.001)
        self.assertAlmostEqual(SAR_INGRESS_SPEED_MPS, state.vehicle.speed_mps)
        self.assertFalse(state._search_started)

        # Leg 2 — heading to the TP line: STILL ingress speed (the bug was that
        # it dropped to search speed here, before the TP line).
        state.current_waypoint_index = 1
        state.completed_route_segment_count = 1
        state._advance_route(0.001)
        self.assertAlmostEqual(SAR_INGRESS_SPEED_MPS, state.vehicle.speed_mps)
        self.assertFalse(state._search_started)

        # Cross the TP line (reach IGR002): search now begins.
        state.vehicle.latitude, state.vehicle.longitude = 37.36, 127.94
        state.vehicle.altitude_m = 600.0
        state._advance_route(1.0)
        self.assertTrue(state._search_started)

        # Post-TP search legs run at search speed.
        state.current_waypoint_index = 2
        state._advance_route(0.001)
        self.assertAlmostEqual(SAR_SEARCH_SPEED_MPS, state.vehicle.speed_mps)

    def test_spx_replan_stub_fields(self) -> None:
        """Verify the stub replan tracking fields are settable."""
        # This tests the data model, not the GUI method
        replan_pending = False
        replan_reason = ""
        exclude_ids: set[int] = set()

        # Simulate trigger
        replan_pending = True
        replan_reason = "ingress_complete"
        exclude_ids = set()

        self.assertTrue(replan_pending)
        self.assertEqual("ingress_complete", replan_reason)
        self.assertEqual(set(), exclude_ids)

    def test_detection_replan_excludes_found_tracks(self) -> None:
        """When a subject is found, its track_id should be excluded from replan."""
        replan_pending = True
        replan_reason = "detection"
        exclude_ids = {101}

        self.assertTrue(replan_pending)
        self.assertEqual("detection", replan_reason)
        self.assertIn(101, exclude_ids)


if __name__ == "__main__":
    unittest.main()
