from __future__ import annotations

import json
import math
import unittest

from qt_gcs.fly_view import (
    build_mission_map_plan_payload,
    json_safe_payload,
    nearest_waypoint_within,
    nearest_runtime_route_index_within,
    runtime_route_index_for_context_key,
    simplify_flight_path,
    waypoint_for_context_key,
)
from qt_gcs.fly_bridge import FlyMapBridge
from qt_gcs.fly_state import FlyState
from qt_gcs.map_bridge import MapBridge
from qt_gcs.site_store import SiteStore


class MissionMapPlanPayloadTests(unittest.TestCase):
    def test_map_payload_replaces_non_finite_values_for_browser_json(self) -> None:
        payload = json_safe_payload(
            {
                "finite": 1.25,
                "metrics": [math.inf, -math.inf, math.nan],
            }
        )

        encoded = json.dumps(payload, allow_nan=False)

        self.assertEqual(
            {"finite": 1.25, "metrics": [None, None, None]},
            json.loads(encoded),
        )

    def test_straight_ingress_track_is_reduced_before_web_render(self) -> None:
        points = [
            {
                "latitude": 23.60 + index * 0.0001,
                "longitude": 47.00 + index * 0.0001,
                "altitude_m": 600.0,
            }
            for index in range(400)
        ]

        simplified = simplify_flight_path(points)

        self.assertEqual(2, len(simplified))
        self.assertEqual(points[0], simplified[0])
        self.assertEqual(points[-1], simplified[-1])

    def test_map_bridge_preserves_exact_waypoint_context(self) -> None:
        bridge = MapBridge(SiteStore())
        received = []
        bridge.featureRightClicked.connect(
            lambda key, latitude, longitude, altitude: received.append(
                (key, latitude, longitude, altitude)
            )
        )

        bridge.reportFeatureRightClick(
            "4:WP005",
            37.401,
            127.951,
            620.0,
        )

        self.assertEqual(
            [("4:WP005", 37.401, 127.951, 620.0)],
            received,
        )

        fly_bridge = FlyMapBridge(
            SiteStore(),
            FlyState.demo(37.3422, 127.9202),
        )
        fly_received = []
        fly_bridge.featureRightClicked.connect(
            lambda key, latitude, longitude, altitude: fly_received.append(
                (key, latitude, longitude, altitude)
            )
        )
        fly_bridge.reportFeatureRightClick(
            "6:WP002",
            37.402,
            127.952,
            630.0,
        )
        self.assertEqual(
            [("6:WP002", 37.402, 127.952, 630.0)],
            fly_received,
        )

    def test_nearest_waypoint_requires_a_safe_click_radius(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        route = store.waypoints_for(1)
        clicked = route[2]

        self.assertEqual(
            clicked.code,
            nearest_waypoint_within(
                route,
                clicked.latitude,
                clicked.longitude,
            ).code,
        )
        self.assertIsNone(
            nearest_waypoint_within(route, 36.0, 126.0)
        )

    def test_waypoint_context_key_selects_exact_vehicle_and_code(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        route_2 = store.waypoints_for(2)

        selected = waypoint_for_context_key(route_2, 2, "2:WP003")

        self.assertIsNotNone(selected)
        self.assertEqual("WP003", selected.code)
        self.assertIsNone(
            waypoint_for_context_key(route_2, 2, "3:WP003")
        )
        self.assertEqual(
            "WP003",
            waypoint_for_context_key(route_2, 2, "FLEET:WP003").code,
        )

    def test_live_rhp_marker_context_selects_exact_runtime_index(self) -> None:
        route = [
            {
                "latitude": 37.40 + index * 0.001,
                "longitude": 127.90 + index * 0.001,
                "altitude_m": 600.0,
                "code": f"RHP-{index + 1}",
            }
            for index in range(3)
        ]

        self.assertEqual(
            1,
            runtime_route_index_for_context_key(route, 4, "AUTO:4:1"),
        )
        self.assertIsNone(
            runtime_route_index_for_context_key(route, 4, "AUTO:3:1")
        )
        self.assertEqual(
            2,
            nearest_runtime_route_index_within(
                route,
                route[2]["latitude"],
                route[2]["longitude"],
            ),
        )

    def test_manual_runtime_route_blocks_then_releases_automatic_rhp(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        state._search_started = True
        state.flight_phase = "ROUTE"
        automatic = [
            {
                "latitude": 37.40,
                "longitude": 127.90,
                "altitude_m": 600.0,
                "code": "RHP-01",
            }
        ]
        manual = [
            {
                "latitude": 37.41,
                "longitude": 127.91,
                "altitude_m": 620.0,
                "code": "MWP001",
            }
        ]

        self.assertTrue(state.queue_runtime_route(10, automatic))
        self.assertTrue(
            state.apply_manual_runtime_route(manual, hold_duration_s=50.0)
        )
        self.assertTrue(state.manual_route_active)
        self.assertEqual("MWP001", state.runtime_route_payload()[0]["code"])
        self.assertFalse(state.queue_runtime_route(11, automatic))

        state.simulation_elapsed_s += 51.0
        self.assertTrue(state.manual_route_active)
        self.assertFalse(state.manual_route_hold_active)
        self.assertTrue(state.queue_runtime_route(12, automatic))
        self.assertFalse(state.manual_route_active)
        self.assertEqual("RHP-01", state.runtime_route_payload()[0]["code"])

    def test_only_new_waypoint_is_marked_as_pending_preview(self) -> None:
        live_store = SiteStore()
        live_store.seed_demo(37.3422, 127.9202)
        pending_store = SiteStore()
        pending_store.replace_from(live_store)
        pending_store.set_active_vehicle(2)
        added = pending_store.add_waypoint(37.455, 127.988, 650.0)

        plan = build_mission_map_plan_payload(
            live_store,
            pending_store,
            True,
        )
        route = next(
            route
            for route in plan["vehicle_routes"]
            if route["vehicle_id"] == 2
        )
        preview_flags = {
            point["code"]: point["pending_preview"]
            for point in route["waypoints"]
        }

        self.assertTrue(plan["pending_edit"])
        self.assertTrue(preview_flags[added.code])
        self.assertTrue(
            all(
                not pending
                for code, pending in preview_flags.items()
                if code != added.code
            )
        )

    def test_loaded_plan_has_no_pending_preview_state(self) -> None:
        live_store = SiteStore()
        live_store.seed_demo(37.3422, 127.9202)

        plan = build_mission_map_plan_payload(
            live_store,
            SiteStore(),
            False,
        )

        self.assertFalse(plan["pending_edit"])
        self.assertTrue(
            all(
                "pending_preview" not in point
                for route in plan["vehicle_routes"]
                for point in route["waypoints"]
            )
        )


if __name__ == "__main__":
    unittest.main()
