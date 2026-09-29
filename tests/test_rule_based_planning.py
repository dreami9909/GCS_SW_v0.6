import math
import unittest

from qt_gcs.planning import (
    IMMParticleFilter,
    LocalFrame,
    LocalPoint,
    MODEL_NAME,
    RuleBasedPlanningEngine,
    SearchCameraSpec,
    build_footprint,
)


class SearchCameraModelTests(unittest.TestCase):
    def test_default_geometry_matches_declared_search_camera_profile(self) -> None:
        camera = SearchCameraSpec()

        self.assertAlmostEqual(190.06, camera.instantaneous_swath_m, places=1)
        self.assertAlmostEqual(200.0, camera.ideal_detection_radius_m, places=1)
        self.assertAlmostEqual(320.0, camera.track_spacing_m, places=1)
        self.assertAlmostEqual(1_200.0, camera.gimbal_centerline_envelope_m)
        self.assertAlmostEqual(600.0, camera.gimbal_centerline_reach_m)
        self.assertAlmostEqual(0.24, camera.scan_sample_interval_s)
        self.assertAlmostEqual(3.2, camera.gimbal_scan_period_s)

    def test_scan_footprint_moves_across_vehicle_track(self) -> None:
        camera = SearchCameraSpec()
        frame = LocalFrame(37.0, 127.0)
        left = build_footprint(
            camera,
            frame,
            vehicle_id=1,
            latitude=37.0,
            longitude=127.0,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.0,
        )
        right = build_footprint(
            camera,
            frame,
            vehicle_id=1,
            latitude=37.0,
            longitude=127.0,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=1.5,
        )

        self.assertLess(left.center.east_m, 0.0)
        self.assertGreater(right.center.east_m, 0.0)
        self.assertAlmostEqual(1_200.0, right.center.east_m - left.center.east_m)

    def test_detection_probability_tapers_at_footprint_edge(self) -> None:
        camera = SearchCameraSpec()
        frame = LocalFrame(37.0, 127.0)
        footprint = build_footprint(
            camera,
            frame,
            vehicle_id=1,
            latitude=37.0,
            longitude=127.0,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.8,
        )
        center_probability = footprint.detection_probability_at(
            footprint.center
        )
        edge_probability = footprint.detection_probability_at(
            LocalPoint(
                footprint.center.east_m + footprint.radius_m * 0.8,
                footprint.center.north_m,
            )
        )
        outside_probability = footprint.detection_probability_at(
            LocalPoint(
                footprint.center.east_m + footprint.radius_m * 1.1,
                footprint.center.north_m,
            )
        )

        self.assertGreater(center_probability, edge_probability)
        self.assertGreater(edge_probability, outside_probability)
        self.assertEqual(0.0, outside_probability)


class IMMParticleFilterTests(unittest.TestCase):
    def test_one_second_transition_preserves_thirty_second_semantics(self) -> None:
        one_second = IMMParticleFilter._scaled_transition(0, 1.0)
        self.assertAlmostEqual(1.0, sum(one_second))
        self.assertGreater(one_second[0], 0.99)
        self.assertAlmostEqual(0.06 / 30.0, one_second[1])

    def test_measurement_moves_belief_toward_observation(self) -> None:
        particle_filter = IMMParticleFilter(
            LocalPoint(0.0, 0.0),
            initial_speed_mps=0.0,
            initial_heading_deg=0.0,
            particle_count=240,
            initial_position_std_m=500.0,
            seed=12,
        )
        before = particle_filter.summary()
        particle_filter.observe_position(
            LocalPoint(400.0, 0.0),
            measurement_std_m=80.0,
        )
        after = particle_filter.summary()

        self.assertGreater(after.mean_east_m, before.mean_east_m)
        self.assertLess(after.uncertainty_east_95_m, before.uncertainty_east_95_m)

    def test_ground_subject_speed_is_not_clipped_below_demo_profile(self) -> None:
        particle_filter = IMMParticleFilter(
            LocalPoint(0.0, 0.0),
            initial_speed_mps=70.0 / 3.6,
            initial_heading_deg=0.0,
            particle_count=240,
            seed=8,
        )

        self.assertGreater(particle_filter.summary().mean_speed_mps, 17.0)


class RuntimePlanningEngineTests(unittest.TestCase):
    @staticmethod
    def _vehicle(latitude: float, longitude: float) -> dict:
        return {
            "vehicle_id": 1,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": 600.0,
            "speed_mps": 160.0 / 3.6,
            "heading_deg": 0.0,
            "mission_launched": True,
            "emergency_mode": False,
            "flight_phase": "ROUTE",
            "search_started": True,
            "runtime_route_revision": 1,
        }

    @staticmethod
    def _subject(latitude: float, longitude: float) -> dict:
        return {
            "track_id": 101,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": 0.0,
            "speed_mps": 8.0,
            "heading_deg": 90.0,
            "found": False,
        }

    def test_cycle_returns_belief_rendezvous_and_rule_route_candidate(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(
            latitude,
            longitude,
            particle_count=120,
            route_improvement_ratio=-1.0,
            probability_gain=0.0,
            confirmation_cycles=1,
            minimum_route_hold_s=0.0,
        )
        vehicle = self._vehicle(latitude, longitude)
        subject = self._subject(latitude + 0.01, longitude)
        routes = {
            1: [
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_m": 600.0,
                }
            ]
        }

        search_result = engine.update(
            elapsed_s=1.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        approach_result = engine.update(
            elapsed_s=2.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=101,
        )

        self.assertEqual(1, len(search_result.beliefs))
        self.assertIn(1, search_result.route_candidates)
        self.assertIn(1, search_result.route_updates)
        self.assertEqual(MODEL_NAME, search_result.render_dict()["model"])
        self.assertEqual(
            25.0,
            search_result.route_candidates[1]["decision_interval_s"],
        )
        self.assertEqual(
            "optimized-stratified-segment",
            search_result.route_candidates[1]["pf_configuration"],
        )
        self.assertTrue(
            math.isfinite(
                float(
                    search_result.route_candidates[1][
                        "improvement_ratio"
                    ]
                )
            )
        )
        self.assertAlmostEqual(
            3_333.3333333333,
            float(search_result.planner["search_radius_m"]),
            places=6,
        )
        self.assertAlmostEqual(
            300.0,
            float(search_result.planner["subject_lead_time_s"]),
            places=6,
        )
        self.assertAlmostEqual(
            latitude,
            float(search_result.planner["search_center"]["latitude"]),
            places=6,
        )
        self.assertAlmostEqual(
            longitude,
            float(search_result.planner["search_center"]["longitude"]),
            places=6,
        )
        self.assertTrue(
            all(
                point["code"].startswith("RHP-")
                for point in search_result.route_updates[1]
            )
        )
        self.assertEqual({}, approach_result.approaches)

    def test_six_lm_actions_remain_inside_fixed_local_sectors(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(
            latitude,
            longitude,
            particle_count=240,
        )
        vehicles = []
        for vehicle_id in range(1, 7):
            vehicle = self._vehicle(latitude, longitude)
            vehicle["vehicle_id"] = vehicle_id
            vehicle["runtime_route_revision"] = 0
            vehicles.append(vehicle)

        routes = {vehicle_id: [] for vehicle_id in range(1, 7)}
        subject = self._subject(latitude + 0.01, longitude)
        # Re-evaluate repeatedly while the LMs execute local RHP prefixes.
        # Every generated point must remain in that SAR's fixed 60-degree sector.
        for revision, elapsed_s in enumerate((1.0, 26.0, 51.0, 76.0), 1):
            result = engine.update(
                elapsed_s=elapsed_s,
                vehicles=vehicles,
                subjects=[subject],
                routes=routes,
                selected_track_id=101,
                approach_track_id=None,
            )

            self.assertFalse(result.planner["swarm_coordination"])
            self.assertEqual(
                "GLOBAL_TP_THEN_LOCAL_RHP",
                result.planner["planning_scope"],
            )
            self.assertEqual(
                "FIXED_60_DEG_PER_LM",
                result.planner["sector_ownership"],
            )
            self.assertEqual(set(range(1, 7)), set(result.route_updates))
            endpoints = {
                (
                    round(float(points[-1]["latitude"]), 7),
                    round(float(points[-1]["longitude"]), 7),
                )
                for points in result.route_updates.values()
            }
            self.assertEqual(6, len(endpoints))
            for vehicle in vehicles:
                vehicle_id = int(vehicle["vehicle_id"])
                points = result.route_updates[vehicle_id]
                sector_center = math.tau * (vehicle_id - 1) / 6.0
                for point in points:
                    local = engine.frame.to_local(
                        float(point["latitude"]),
                        float(point["longitude"]),
                    )
                    if local.distance_to(engine.search_center) <= 1e-6:
                        continue
                    angle = math.atan2(
                        local.north_m - engine.search_center.north_m,
                        local.east_m - engine.search_center.east_m,
                    )
                    relative = (
                        angle - sector_center + math.pi
                    ) % math.tau - math.pi
                    self.assertLessEqual(abs(relative), math.pi / 6.0 + 1e-6)
                transit_point = points[min(1, len(points) - 1)]
                vehicle["latitude"] = float(transit_point["latitude"])
                vehicle["longitude"] = float(transit_point["longitude"])
                vehicle["runtime_route_revision"] = revision
                routes[vehicle_id] = list(points)

    def test_search_camera_footprint_automatically_reports_detection(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(latitude, longitude, particle_count=120)
        vehicle = self._vehicle(latitude, longitude)
        footprint = build_footprint(
            engine.search_camera,
            engine.frame,
            vehicle_id=1,
            latitude=latitude,
            longitude=longitude,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.0,
        )
        subject_latitude, subject_longitude = engine.frame.to_geographic(
            footprint.center
        )

        result = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[self._subject(subject_latitude, subject_longitude)],
            routes={1: [{"latitude": latitude, "longitude": longitude}]},
            selected_track_id=101,
            approach_track_id=None,
        )

        self.assertEqual(1, len(result.detections))
        self.assertEqual(101, result.detections[0]["track_id"])
        self.assertEqual(1, result.detections[0]["vehicle_id"])
        self.assertEqual(1.0, result.detections[0]["probability"])
        self.assertEqual("SEARCH_CAM_SWEPT_FOV", result.detections[0]["source"])

    def test_automatic_detection_rejects_old_1200_m_radius_shortcut(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(latitude, longitude, particle_count=120)
        vehicle = self._vehicle(latitude, longitude)
        # At the -45 degree stop the circular footprint approximation extends
        # slightly beyond the declared 600 m centre-line reach.  The explicit
        # 1,200 m full-width coverage limit must still win.
        outside_declared_coverage = LocalPoint(-780.0, 0.0)
        current_look = build_footprint(
            engine.search_camera,
            engine.frame,
            vehicle_id=1,
            latitude=latitude,
            longitude=longitude,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.0,
        )
        self.assertGreater(
            current_look.detection_probability_at(outside_declared_coverage),
            0.0,
        )
        subject_latitude, subject_longitude = engine.frame.to_geographic(
            outside_declared_coverage
        )

        result = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[self._subject(subject_latitude, subject_longitude)],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )

        self.assertFalse(result.detections)

    def test_automatic_detection_integrates_fov_between_one_hz_updates(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(latitude, longitude, particle_count=120)
        vehicle = self._vehicle(latitude, longitude)
        intervening_look = build_footprint(
            engine.search_camera,
            engine.frame,
            vehicle_id=1,
            latitude=latitude,
            longitude=longitude,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.5,
        )
        subject_latitude, subject_longitude = engine.frame.to_geographic(
            intervening_look.center
        )
        subject = self._subject(subject_latitude, subject_longitude)

        before_scan = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )
        after_scan = engine.update(
            elapsed_s=1.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )

        self.assertFalse(before_scan.detections)
        self.assertEqual(1, len(after_scan.detections))
        self.assertEqual(1.0, after_scan.detections[0]["probability"])

    def test_demo_detection_gate_waits_for_three_visible_rhp_updates(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(
            latitude,
            longitude,
            particle_count=120,
            minimum_route_updates_before_detection=3,
        )
        vehicle = self._vehicle(latitude, longitude)
        footprint = build_footprint(
            engine.search_camera,
            engine.frame,
            vehicle_id=1,
            latitude=latitude,
            longitude=longitude,
            altitude_m=600.0,
            heading_deg=0.0,
            elapsed_s=0.0,
        )
        subject_latitude, subject_longitude = engine.frame.to_geographic(
            footprint.center
        )
        subject = self._subject(subject_latitude, subject_longitude)
        vehicle["runtime_route_update_count"] = 2
        blocked = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )
        vehicle["runtime_route_update_count"] = 3
        allowed = engine.update(
            elapsed_s=1.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )

        self.assertFalse(blocked.detections)
        self.assertEqual(1, len(allowed.detections))

    def test_rhp_commits_only_at_twenty_five_second_epochs(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(
            latitude,
            longitude,
            particle_count=180,
        )
        vehicle = self._vehicle(latitude, longitude)
        vehicle["heading_deg"] = 180.0
        subject = self._subject(latitude + 0.01, longitude)
        subject["speed_mps"] = 10.0
        routes = {
            1: [
                {
                    "latitude": latitude - 0.02,
                    "longitude": longitude,
                    "altitude_m": 600.0,
                },
                {
                    "latitude": latitude - 0.03,
                    "longitude": longitude,
                    "altitude_m": 600.0,
                },
            ]
        }

        updates = []
        for elapsed_s in range(1, 55):
            result = engine.update(
                elapsed_s=float(elapsed_s),
                vehicles=[vehicle],
                subjects=[subject],
                routes=routes,
                selected_track_id=101,
                approach_track_id=None,
            )
            if result.route_updates:
                updates.append((elapsed_s, result))

        self.assertGreaterEqual(len(updates), 2)
        applied_at, applied_result = updates[0]
        self.assertEqual(1, applied_at)
        self.assertTrue(
            all(
                later[0] - earlier[0] >= 25
                for earlier, later in zip(updates, updates[1:])
            )
        )
        self.assertIn(1, applied_result.route_updates)
        self.assertTrue(
            applied_result.route_candidates[1]["action_commit_due"]
        )

    def test_tp_initial_prefix_cache_does_not_start_pf_or_rhp_clock(self) -> None:
        latitude = 37.4
        longitude = 127.9
        engine = RuleBasedPlanningEngine(
            latitude,
            longitude,
            particle_count=180,
        )
        vehicle = self._vehicle(latitude, longitude)
        vehicle["runtime_route_revision"] = 0
        vehicle["runtime_route_update_count"] = 0
        vehicle["rhp_preview_only"] = True
        subject = self._subject(latitude + 0.025, longitude)
        routes = {1: []}

        cached = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )

        search_filter = engine.search_filters[101]
        self.assertIn(1, cached.route_updates)
        self.assertFalse(cached.footprints)
        self.assertEqual(0, search_filter.prediction_step_count)
        self.assertEqual(0, search_filter.observation_update_count)
        self.assertEqual(0, search_filter.revision)
        self.assertEqual(0.0, engine.rhp_planner.last_applied_s[1])

        # Physical TP arrival starts search t=0.  The cached prefix remains
        # committed, but neither a duplicate route update nor a synthetic
        # t=0 negative observation is allowed.
        vehicle["runtime_route_revision"] = cached.revision
        vehicle["runtime_route_update_count"] = 1
        vehicle["rhp_preview_only"] = False
        at_tp = engine.update(
            elapsed_s=0.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        self.assertEqual(1, len(at_tp.footprints))
        self.assertFalse(at_tp.route_updates)
        self.assertEqual(0, search_filter.prediction_step_count)
        self.assertEqual(0, search_filter.observation_update_count)

        before_epoch = engine.update(
            elapsed_s=24.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        at_epoch = engine.update(
            elapsed_s=25.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        self.assertFalse(before_epoch.route_updates)
        self.assertIn(1, at_epoch.route_updates)
        delayed_poll = engine.update(
            elapsed_s=52.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes=routes,
            selected_track_id=101,
            approach_track_id=None,
        )
        self.assertIn(1, delayed_poll.route_updates)
        self.assertEqual(50.0, engine.rhp_planner.last_applied_s[1])

    def test_rhp_candidate_is_recomputed_between_route_commits(self) -> None:
        engine = RuleBasedPlanningEngine(
            37.4,
            127.9,
            particle_count=180,
        )
        vehicle = self._vehicle(37.4, 127.9)
        subject = self._subject(37.41, 127.9)
        first = engine.update(
            elapsed_s=1.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )
        second = engine.update(
            elapsed_s=2.0,
            vehicles=[vehicle],
            subjects=[subject],
            routes={1: []},
            selected_track_id=101,
            approach_track_id=None,
        )

        self.assertIn(1, first.route_updates)
        self.assertIn(1, second.route_candidates)
        self.assertNotIn(1, second.route_updates)
        self.assertFalse(
            second.route_candidates[1]["action_commit_due"]
        )
        self.assertGreater(
            second.route_candidates[1]["particle_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
