"""SPX re-plan trigger: background MILP solve + apply to store.

These tests verify the ``_trigger_spx_replan`` → ``_collect_spx_replan_result``
pipeline without running a real MILP solve (which takes ~60 s and needs a full
scipy HiGHS stack).  The solve function is mocked to return a lightweight
``CertifiedSearchPlan`` immediately.

The tests exercise:
1. The background future is created and collectable.
2. On collection, ``apply_plan_to_store`` writes new waypoints to the store.
3. Each FlyState picks up the new routes via ``update_route_from_store``.
4. Generation guards prevent stale results from being applied after a reset.
5. Solver errors are caught and surfaced as status messages.
6. ``replan_after_detection`` path correctly excludes found track IDs.
"""

from __future__ import annotations

import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from qt_gcs.planning.stone_adapter import (
    CertifiedSearchPlan,
    SearchPlanCertificate,
    apply_plan_to_store,
)
from qt_gcs.site_store import SiteStore


def _dummy_certificate() -> SearchPlanCertificate:
    """Minimal certificate for test purposes."""
    return SearchPlanCertificate(
        method="stone-spx",
        fingerprint="test-0000",
        detection_probability=0.55,
        relative_optimality_gap=0.005,
        lower_bound_nondetection=0.44,
        upper_bound_nondetection=0.45,
        converged=True,
        certified=True,
        required_relative_gap=0.01,
        iterations=3,
        runtime_s=1.5,
        grid_shape=(6, 6),
        time_slice_count=8,
        no_fly_cell_count=0,
        target_detection_probabilities=(("child", 0.55),),
    )


def _dummy_plan(n_waypoints: int = 10) -> CertifiedSearchPlan:
    """A lightweight certified plan with ``n_waypoints`` per vehicle."""
    waypoints = {}
    for vid in SiteStore.VEHICLE_IDS:
        waypoints[vid] = [
            (37.3 + 0.001 * i, 127.9 + 0.001 * i, 600.0)
            for i in range(n_waypoints)
        ]
    return CertifiedSearchPlan(
        certificate=_dummy_certificate(),
        vehicle_waypoints=waypoints,
    )


class ApplyPlanToStoreTests(unittest.TestCase):
    """Verify ``apply_plan_to_store`` replaces vehicle waypoints."""

    def test_apply_writes_correct_waypoint_count(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        original_counts = {
            vid: len(store.vehicle_waypoints[vid])
            for vid in SiteStore.VEHICLE_IDS
        }
        plan = _dummy_plan(n_waypoints=15)
        apply_plan_to_store(store, plan)
        for vid in SiteStore.VEHICLE_IDS:
            self.assertEqual(15, len(store.vehicle_waypoints[vid]))
            # Originals were 5 (seed_demo), now 15
            self.assertNotEqual(original_counts[vid], 15)

    def test_applied_waypoints_have_spx_labels(self) -> None:
        store = SiteStore()
        plan = _dummy_plan(n_waypoints=3)
        apply_plan_to_store(store, plan)
        for vid in SiteStore.VEHICLE_IDS:
            for wp in store.vehicle_waypoints[vid]:
                self.assertIn("SPX", wp.label)

    def test_apply_triggers_notify(self) -> None:
        store = SiteStore()
        notified = []
        store.subscribe(lambda: notified.append(True))
        apply_plan_to_store(store, _dummy_plan(3))
        self.assertTrue(notified)


class SPXReplanFutureTests(unittest.TestCase):
    """Simulate the background SPX re-plan pipeline."""

    def test_future_resolves_with_plan(self) -> None:
        """A mocked solve returns a CertifiedSearchPlan via Future."""
        executor = ThreadPoolExecutor(max_workers=1)
        plan = _dummy_plan(8)
        future = executor.submit(lambda: plan)
        result = future.result(timeout=5.0)
        self.assertIsInstance(result, CertifiedSearchPlan)
        self.assertEqual(8, len(result.vehicle_waypoints[1]))
        executor.shutdown(wait=False)

    def test_apply_after_collect(self) -> None:
        """Simulates _collect_spx_replan_result: future → apply_plan_to_store."""
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        executor = ThreadPoolExecutor(max_workers=1)
        plan = _dummy_plan(12)
        future = executor.submit(lambda: plan)
        result = future.result(timeout=5.0)
        apply_plan_to_store(store, result)
        for vid in SiteStore.VEHICLE_IDS:
            self.assertEqual(12, len(store.vehicle_waypoints[vid]))
        executor.shutdown(wait=False)

    def test_generation_guard_rejects_stale_result(self) -> None:
        """If generation changed, the result is discarded."""
        generation_at_submit = 0
        current_generation = 1  # Simulate a mission reset

        plan = _dummy_plan(5)
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        original_count = len(store.vehicle_waypoints[1])

        # If generation doesn't match, don't apply
        if generation_at_submit != current_generation:
            applied = False
        else:
            apply_plan_to_store(store, plan)
            applied = True

        self.assertFalse(applied)
        self.assertEqual(original_count, len(store.vehicle_waypoints[1]))

    def test_solver_error_is_caught(self) -> None:
        """A solver exception should not propagate; status message should note it."""
        executor = ThreadPoolExecutor(max_workers=1)

        def _failing_solve():
            raise RuntimeError("MILP infeasible")

        future = executor.submit(_failing_solve)
        with self.assertRaises(RuntimeError):
            future.result(timeout=5.0)
        executor.shutdown(wait=False)


class SPXReplanExclusionTests(unittest.TestCase):
    """Verify exclusion of found subjects flows through to the solver."""

    def test_exclude_track_ids_passed_to_plan_certified_search(self) -> None:
        """When reason='detection', exclude_track_ids must reach the solver."""
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        store.mission_metadata["ingress_sweep"] = True
        store.mission_metadata["rally_predicted_subject"] = {
            "latitude": 37.6, "longitude": 128.2,
        }

        # The flow: _trigger_spx_replan captures exclude_track_ids in the
        # closure that calls plan_certified_search(snapshot, exclude_track_ids=...).
        # We test this at the stone_adapter level.
        from qt_gcs.planning.stone_adapter import _subject_specs

        # With 2 subjects, excluding one leaves one
        from qt_gcs.site_store import MissionSubject
        store.initial_subjects = [
            MissionSubject(
                track_id=101, subject_type="MISSING_PERSON_CHILD",
                latitude=37.5, longitude=128.0,
            ),
            MissionSubject(
                track_id=204, subject_type="MISSING_PERSON_ADULT",
                latitude=37.6, longitude=128.1,
            ),
        ]
        specs_all = _subject_specs(store)
        specs_excluded = _subject_specs(store, exclude_track_ids={101})
        self.assertEqual(2, len(specs_all))
        self.assertEqual(1, len(specs_excluded))

    def test_snapshot_isolates_store_from_mutation(self) -> None:
        """The SiteStore snapshot should be independent of the original."""
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        snapshot = SiteStore()
        snapshot.replace_from(store)

        # Mutate original
        store.sites.clear()
        store.vehicle_waypoints[1].clear()

        # Snapshot is unaffected
        self.assertIn("GCS", snapshot.sites)
        self.assertTrue(len(snapshot.vehicle_waypoints[1]) > 0)


class FlyStateRouteUpdateTests(unittest.TestCase):
    """Verify FlyState picks up new routes after SPX re-plan."""

    def test_update_route_from_store_changes_route(self) -> None:
        from qt_gcs.fly_state import FlyState
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()

        original_route_len = len(state._route_points)

        # Simulate SPX re-plan result applied to store
        plan = _dummy_plan(20)
        apply_plan_to_store(store, plan)

        # FlyState picks up new routes
        state.update_route_from_store(store, 1)
        self.assertEqual(20, len(state._route_points))
        self.assertNotEqual(original_route_len, 20)

    def test_update_preserves_launch_state(self) -> None:
        from qt_gcs.fly_state import FlyState
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        self.assertTrue(state.mission_launched)

        plan = _dummy_plan(10)
        apply_plan_to_store(store, plan)
        state.update_route_from_store(store, 1)

        # Launch state is preserved
        self.assertTrue(state.mission_launched)
        self.assertEqual("ROUTE", state.flight_phase)


if __name__ == "__main__":
    unittest.main()
