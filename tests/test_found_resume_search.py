"""Found → resume-search automation tests.

Verifies:
1. FlyState.resume_search() transitions from FOUND back to ROUTE.
2. resume_search() preserves launch state and search_started.
3. resume_search() returns False when not in FOUND state.
4. FlyView._check_found_resume_search() triggers when all vehicles FOUND
   and unfound subjects remain.
5. Resume triggers SPX replan with correct exclude_track_ids.
6. No resume when all subjects have been found (mission complete).
7. _found_track_ids accumulates across multiple resume cycles.
"""

from __future__ import annotations

import unittest
from dataclasses import field
from unittest.mock import MagicMock, patch, PropertyMock

from qt_gcs.fly_state import FlyState, SubjectTrack, VehicleTrack
from qt_gcs.site_store import SiteStore, MissionSubject


def _make_subject(track_id: int, found: bool = False) -> SubjectTrack:
    return SubjectTrack(
        track_id=track_id,
        subject_type="MISSING_PERSON_CHILD",
        country="SIM",
        platform_name=f"SIM-SUBJECT-{track_id}",
        latitude=23.7 + 0.01 * track_id,
        longitude=47.3 + 0.01 * track_id,
        altitude_m=0.0,
        speed_mps=5.0,
        heading_deg=90.0,
        first_tracked_at=0.0,
        found=found,
    )


def _make_vehicle(code: str = "SAR-01") -> VehicleTrack:
    return VehicleTrack(
        code=code,
        latitude=23.65,
        longitude=47.05,
        altitude_m=600.0,
        heading_deg=0.0,
        speed_mps=208.0,
    )


def _make_state(
    track_id: int = 101,
    subjects: list[SubjectTrack] | None = None,
) -> FlyState:
    if subjects is None:
        subjects = [_make_subject(101), _make_subject(102), _make_subject(103)]
    return FlyState(
        center_latitude=23.65,
        center_longitude=47.05,
        vehicle=_make_vehicle(),
        subjects=subjects,
        selected_track_id=track_id,
    )


class ResumeSearchStateTests(unittest.TestCase):
    """FlyState.resume_search() unit tests."""

    def test_resume_transitions_found_to_route(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.subject_found = True
        state.flight_phase = "FOUND"
        state._search_started = True
        result = state.resume_search()
        self.assertTrue(result)
        self.assertEqual("ROUTE", state.flight_phase)
        self.assertFalse(state.subject_found)

    def test_resume_preserves_launch_and_search(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.subject_found = True
        state.flight_phase = "FOUND"
        state._search_started = True
        state.simulation_elapsed_s = 120.0
        state.resume_search()
        self.assertTrue(state.mission_launched)
        self.assertTrue(state._search_started)
        self.assertEqual(120.0, state.simulation_elapsed_s)

    def test_resume_clears_approach_state(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.subject_found = True
        state.subject_designated = True
        state.subject_detected = True
        state.approach_approved = True
        state.approach_requested = True
        state.flight_phase = "FOUND"
        state.shutdown_position = (23.7, 47.3, 600.0)
        state.resume_search()
        self.assertFalse(state.subject_designated)
        self.assertFalse(state.subject_detected)
        self.assertFalse(state.approach_approved)
        self.assertFalse(state.approach_requested)
        self.assertIsNone(state.shutdown_position)

    def test_resume_returns_false_when_not_found(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.flight_phase = "ROUTE"
        result = state.resume_search()
        self.assertFalse(result)
        self.assertEqual("ROUTE", state.flight_phase)

    def test_resume_resets_waypoint_index(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.subject_found = True
        state.flight_phase = "FOUND"
        state.current_waypoint_index = 5
        state.completed_route_segment_count = 3
        state.resume_search()
        self.assertEqual(0, state.current_waypoint_index)
        self.assertEqual(0, state.completed_route_segment_count)

    def test_resume_resets_mission_status_flags(self) -> None:
        state = _make_state()
        state.mission_launched = True
        state.subject_found = True
        state.flight_phase = "FOUND"
        for name in state.mission_status:
            state.mission_status[name] = True
        state.resume_search()
        self.assertTrue(all(v is False for v in state.mission_status.values()))


def _check_found_resume_search(self) -> None:
    """Extracted logic matching Fly3DView._check_found_resume_search.

    This standalone function duplicates the method body so it can be tested
    without importing PySide6.  The tests below verify the logic against
    the same contract; the real method in fly_view.py is the source of truth.
    """
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
    self._trigger_spx_replan(
        reason="detection",
        exclude_track_ids=self._found_track_ids.copy(),
    )
    self._resume_search_triggered = False


class CheckFoundResumeSearchTests(unittest.TestCase):
    """Fly3DView._check_found_resume_search() logic tests.

    The method is duplicated above as a standalone function to avoid
    importing PySide6.  The test verifies the same logic contract.
    """

    def _make_fly_view_mock(
        self, n_subjects: int = 3, n_found: int = 1
    ) -> MagicMock:
        fly_view = MagicMock()
        fly_view._resume_search_triggered = False
        fly_view._spx_replan_pending = False
        fly_view._found_track_ids = set()
        fly_view._reported_shutdown = False

        subjects = [
            MissionSubject(
                track_id=101 + i,
                subject_type="MISSING_PERSON_CHILD",
                latitude=23.7 + 0.01 * i,
                longitude=47.3 + 0.01 * i,
                altitude_m=0.0,
                country="SIM",
                platform_name=f"SIM-{101 + i}",
                speed_mps=5.0,
                heading_deg=90.0,
                position_uncertainty_m=500.0,
                source="SAR-C2-SIM",
                motion_profile=[],
            )
            for i in range(n_subjects)
        ]

        store = SiteStore()
        store.initial_subjects = subjects
        fly_view.store = store

        states = {}
        for vid in SiteStore.VEHICLE_IDS:
            state = _make_state(
                track_id=101,
                subjects=[_make_subject(101 + j) for j in range(n_subjects)],
            )
            state.mission_launched = True
            if n_found >= 1:
                state.subject_found = True
                state.flight_phase = "FOUND"
                state.selected_track_id = 101
            states[vid] = state
        fly_view.states = states

        fly_view.statusMessage = MagicMock()
        fly_view.statusMessage.emit = MagicMock()

        return fly_view

    def test_resume_triggers_when_all_found_with_remaining(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        for state in fly_view.states.values():
            self.assertEqual("ROUTE", state.flight_phase)
            self.assertFalse(state.subject_found)
        fly_view._trigger_spx_replan.assert_called_once()
        call_kwargs = fly_view._trigger_spx_replan.call_args[1]
        self.assertEqual("detection", call_kwargs["reason"])
        self.assertIn(101, call_kwargs["exclude_track_ids"])

    def test_no_resume_when_all_subjects_found(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=1, n_found=1)
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        fly_view._trigger_spx_replan.assert_not_called()
        for state in fly_view.states.values():
            self.assertTrue(state.subject_found)

    def test_no_resume_when_replan_pending(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        fly_view._spx_replan_pending = True
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        fly_view._trigger_spx_replan.assert_not_called()
        for state in fly_view.states.values():
            self.assertTrue(state.subject_found)

    def test_no_resume_when_not_all_vehicles_found(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        first_vid = next(iter(fly_view.states))
        fly_view.states[first_vid].subject_found = False
        fly_view.states[first_vid].flight_phase = "ROUTE"
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        fly_view._trigger_spx_replan.assert_not_called()

    def test_found_track_ids_accumulate(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        fly_view._found_track_ids = {101}
        for state in fly_view.states.values():
            state.selected_track_id = 102
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        self.assertEqual({101, 102}, fly_view._found_track_ids)
        call_kwargs = fly_view._trigger_spx_replan.call_args[1]
        self.assertEqual({101, 102}, call_kwargs["exclude_track_ids"])

    def test_resume_selects_next_unfound_subject(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        for state in fly_view.states.values():
            self.assertEqual(102, state.selected_track_id)

    def test_resume_skips_already_found_selects_next(self) -> None:
        fly_view = self._make_fly_view_mock(n_subjects=3, n_found=1)
        fly_view._found_track_ids = {101}
        for state in fly_view.states.values():
            state.selected_track_id = 102
        fly_view._trigger_spx_replan = MagicMock()

        _check_found_resume_search(fly_view)

        for state in fly_view.states.values():
            self.assertEqual(103, state.selected_track_id)


if __name__ == "__main__":
    unittest.main()
