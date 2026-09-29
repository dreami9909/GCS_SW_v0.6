from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qt_gcs.site_store import SiteStore


def point_inside_zone(latitude: float, longitude: float, vertices) -> bool:
    inside = False
    previous = vertices[-1]
    for current in vertices:
        crosses_latitude = (
            current.latitude > latitude
        ) != (
            previous.latitude > latitude
        )
        if crosses_latitude:
            boundary_longitude = (
                (previous.longitude - current.longitude)
                * (latitude - current.latitude)
                / (previous.latitude - current.latitude)
                + current.longitude
            )
            if longitude < boundary_longitude:
                inside = not inside
        previous = current
    return inside


class SiteStoreTests(unittest.TestCase):
    def test_set_site_validates_code_and_coordinates(self) -> None:
        store = SiteStore()
        site = store.set_site("GCS", 37.3422, 127.9202, 42)
        self.assertEqual("GCS", site.code)
        self.assertEqual(42, site.altitude_m)

        with self.assertRaises(ValueError):
            store.set_site("UNKNOWN", 37.0, 127.0)
        with self.assertRaises(ValueError):
            store.set_site("RDR", 100.0, 127.0)

    def test_site_configuration_round_trip(self) -> None:
        source = SiteStore()
        source.seed_demo(37.3422, 127.9202)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "site_config.json"
            source.save(path)

            loaded = SiteStore()
            loaded.load(path)

        self.assertEqual({"GCS", "RDR", "LC"}, set(loaded.sites))
        self.assertAlmostEqual(
            source.sites["RDR"].latitude,
            loaded.sites["RDR"].latitude,
        )
        self.assertEqual(51, loaded.sites["LC"].altitude_m)
        self.assertEqual(5, len(loaded.waypoints))
        self.assertIsNone(loaded.return_point)
        self.assertEqual({"SAFE"}, {zone.zone_type for zone in loaded.zones})

    def test_waypoints_renumber_after_delete(self) -> None:
        store = SiteStore()
        store.add_waypoint(37.34, 127.92)
        store.add_waypoint(37.35, 127.93)
        store.add_waypoint(37.36, 127.94)

        self.assertTrue(store.remove_feature("WP002"))
        self.assertEqual(["WP001", "WP002"], [item.code for item in store.waypoints])
        self.assertEqual([1, 2], [item.sequence for item in store.waypoints])

    def test_shared_features_and_vehicle_routes_are_isolated(self) -> None:
        store = SiteStore()
        store.set_site("GCS", 37.3422, 127.9202, 42)
        store.set_site("RDR", 37.3432, 127.9192, 55)
        store.set_site("LC", 37.3412, 127.9182, 51)
        store.begin_zone("SAFE")
        store.add_draft_vertex(37.340, 127.920)
        store.add_draft_vertex(37.341, 127.924)
        store.add_draft_vertex(37.338, 127.923)
        store.commit_zone()

        self.assertTrue(store.shared_configuration_ready)
        store.add_waypoint(37.345, 127.925, 600)
        store.set_active_vehicle(2)
        self.assertFalse(store.waypoints)
        store.add_waypoint(37.346, 127.926, 700)
        store.add_waypoint(37.347, 127.927, 800)

        self.assertEqual({"GCS", "RDR", "LC"}, set(store.sites))
        self.assertEqual(1, len(store.waypoints_for(1)))
        self.assertEqual(2, len(store.waypoints_for(2)))
        self.assertEqual("UAV-02 Waypoint 1", store.waypoints_for(2)[0].label)
        self.assertEqual((1, 2), store.configured_vehicle_ids)
        self.assertEqual(3, store.total_waypoint_count)
        self.assertTrue(store.is_mission_ready)

    def test_all_six_routes_round_trip(self) -> None:
        source = SiteStore()
        source.seed_demo(37.3422, 127.9202)
        source.set_active_vehicle(2)
        source.add_waypoint(37.500, 128.000, 900)
        source.set_active_vehicle(6)
        source.add_waypoint(37.600, 128.100, 1000)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fleet_plan.json"
            source.save(path)
            loaded = SiteStore()
            loaded.load(path)

        self.assertEqual(5, len(loaded.waypoints_for(1)))
        self.assertEqual(6, len(loaded.waypoints_for(2)))
        self.assertEqual(6, len(loaded.waypoints_for(6)))
        self.assertEqual(tuple(range(1, 7)), loaded.configured_vehicle_ids)
        self.assertEqual("UAV-06 Waypoint 1", loaded.waypoints_for(6)[0].label)

    def test_saudi_v4_plan_loads_c4i_subjects_and_round_trips(self) -> None:
        mission_path = (
            Path(__file__).parents[1] / "saudi_desert_mission.json"
        )
        source = SiteStore()
        source.load(mission_path)

        self.assertTrue(source.is_mission_ready)
        self.assertEqual("SD-RHP-01", source.mission_metadata["scenario_id"])
        self.assertEqual(
            "RHP-FE-PF-PW-ARC",
            source.mission_metadata["arc_search_pattern"]["type"],
        )
        self.assertAlmostEqual(
            3_333.3333333333,
            source.mission_metadata["search_radius_m"],
        )
        self.assertEqual(
            3,
            source.mission_metadata[
                "visualization_min_rhp_updates_before_atr"
            ],
        )
        self.assertEqual(216, source.total_waypoint_count)
        self.assertEqual([101, 204], [t.track_id for t in source.initial_subjects])
        self.assertLessEqual(
            max(subj.speed_mps for subj in source.initial_subjects),
            40.0 / 3.6,
        )

        with tempfile.TemporaryDirectory() as directory:
            saved_path = Path(directory) / "saudi_saved.json"
            source.save(saved_path)
            loaded = SiteStore()
            loaded.load(saved_path)

        self.assertEqual(source.mission_metadata, loaded.mission_metadata)
        self.assertEqual(
            source.initial_subjects[0].motion_profile,
            loaded.initial_subjects[0].motion_profile,
        )

    def test_seeded_safe_zone_excludes_shared_facilities(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)

        safe_zone = store.zones[0]
        for site in store.sites.values():
            self.assertFalse(
                point_inside_zone(
                    site.latitude,
                    site.longitude,
                    safe_zone.vertices,
                ),
                f"{site.code} must be outside the safe zone",
            )

    def test_loaded_snapshot_is_independent_from_plan_edits(self) -> None:
        plan = SiteStore()
        plan.seed_demo(37.3422, 127.9202)
        loaded = SiteStore()
        loaded.replace_from(plan)

        loaded_latitude = loaded.waypoints[0].latitude
        plan.update_point("WP001", 37.9, 128.1, 777.0)

        self.assertEqual(loaded_latitude, loaded.waypoints[0].latitude)
        self.assertNotEqual(plan.waypoints[0].latitude, loaded.waypoints[0].latitude)

    def test_safe_zone_workflow(self) -> None:
        store = SiteStore()
        store.begin_zone("SAFE")
        store.add_draft_vertex(37.340, 127.920)
        store.add_draft_vertex(37.341, 127.924)
        with self.assertRaises(ValueError):
            store.commit_zone()

        store.add_draft_vertex(37.338, 127.923)
        safe_zone = store.commit_zone()
        self.assertEqual("SAFE01", safe_zone.code)
        self.assertEqual(3, len(safe_zone.vertices))
        self.assertIsNone(store.draft_zone_type)
        self.assertFalse(store.draft_vertices)

        with self.assertRaises(ValueError):
            store.begin_zone("DANGER")


if __name__ == "__main__":
    unittest.main()
