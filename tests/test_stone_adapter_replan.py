"""Multi-subject re-planning: excluding a found subject from the SPX subject set.

These exercise the pure subject-spec selection (no scipy MILP solve needed).
"""

from __future__ import annotations

import unittest

from qt_gcs.planning import stone_adapter as sa
from qt_gcs.site_store import MissionSubject, SiteStore


def _store_with_two_subjects() -> SiteStore:
    store = SiteStore()
    store.initial_subjects = [
        MissionSubject(
            track_id=101, subject_type="MISSING_PERSON_CHILD",
            latitude=23.75, longitude=47.33, speed_mps=3.0,
        ),
        MissionSubject(
            track_id=204, subject_type="MISSING_PERSON_ADULT",
            latitude=23.80, longitude=47.36, speed_mps=5.0,
        ),
    ]
    return store


class ReplanExclusionTests(unittest.TestCase):
    def test_all_subjects_when_none_excluded(self) -> None:
        specs = sa._subject_specs(_store_with_two_subjects())
        self.assertEqual(2, len(specs))

    def test_found_subject_is_dropped(self) -> None:
        specs = sa._subject_specs(_store_with_two_subjects(), exclude_track_ids={101})
        names = [spec.profile.name for spec in specs]
        self.assertEqual(1, len(specs))
        self.assertTrue(any("adult" in n for n in names), names)
        self.assertFalse(any("child" in n for n in names), names)

    def test_excluding_every_subject_refuses(self) -> None:
        with self.assertRaises(ValueError):
            sa._subject_specs(_store_with_two_subjects(), exclude_track_ids={101, 204})

    def test_missionless_store_still_falls_back_to_child(self) -> None:
        # No initial_subjects at all -> the child-signature fallback still applies.
        specs = sa._subject_specs(SiteStore())
        self.assertEqual(1, len(specs))
        self.assertIn("child", specs[0].profile.name)

    def test_elderly_type_maps_to_adult_profile(self) -> None:
        store = SiteStore()
        store.initial_subjects = [
            MissionSubject(
                track_id=103, subject_type="MISSING_PERSON_ELDERLY",
                latitude=23.77, longitude=47.30, speed_mps=1.5,
            ),
        ]
        specs = sa._subject_specs(store)
        self.assertEqual(1, len(specs))
        self.assertAlmostEqual(0.874, specs[0].hazard_multiplier, places=3)

    def test_three_subject_types_all_recognized(self) -> None:
        store = SiteStore()
        store.initial_subjects = [
            MissionSubject(
                track_id=101, subject_type="MISSING_PERSON_CHILD",
                latitude=23.75, longitude=47.33, speed_mps=3.0,
            ),
            MissionSubject(
                track_id=102, subject_type="MISSING_PERSON_ADULT",
                latitude=23.80, longitude=47.36, speed_mps=5.0,
            ),
            MissionSubject(
                track_id=103, subject_type="MISSING_PERSON_ELDERLY",
                latitude=23.77, longitude=47.30, speed_mps=1.5,
            ),
        ]
        specs = sa._subject_specs(store)
        self.assertEqual(3, len(specs))

    def test_elderly_excluded_leaves_others(self) -> None:
        store = SiteStore()
        store.initial_subjects = [
            MissionSubject(
                track_id=101, subject_type="MISSING_PERSON_CHILD",
                latitude=23.75, longitude=47.33, speed_mps=3.0,
            ),
            MissionSubject(
                track_id=103, subject_type="MISSING_PERSON_ELDERLY",
                latitude=23.77, longitude=47.30, speed_mps=1.5,
            ),
        ]
        specs = sa._subject_specs(store, exclude_track_ids={101})
        self.assertEqual(1, len(specs))

    def test_only_elderly_remaining_does_not_crash(self) -> None:
        store = SiteStore()
        store.initial_subjects = [
            MissionSubject(
                track_id=101, subject_type="MISSING_PERSON_CHILD",
                latitude=23.75, longitude=47.33, speed_mps=3.0,
            ),
            MissionSubject(
                track_id=103, subject_type="MISSING_PERSON_ELDERLY",
                latitude=23.77, longitude=47.30, speed_mps=1.5,
            ),
        ]
        specs = sa._subject_specs(store, exclude_track_ids={101})
        self.assertEqual(1, len(specs))


if __name__ == "__main__":
    unittest.main()
