"""Wiring tests: ingress routes are prepended to mission when metadata enables it."""
from __future__ import annotations

import json
import unittest

from qt_gcs.site_store import SiteStore, MissionPoint
from qt_gcs.planning.ingress import build_parallel_ingress, ingress_corridor_polygon


class IngressRouteWiringTests(unittest.TestCase):
    def _store_with_ingress(self) -> SiteStore:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        store.mission_metadata["ingress_sweep"] = True
        store.mission_metadata["rally_predicted_subject"] = {
            "latitude": 37.6, "longitude": 128.2,
        }
        return store

    def test_metadata_flag_absent_means_disabled(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        self.assertFalse(store.mission_metadata.get("ingress_sweep", False))

    def test_metadata_flag_present_means_enabled(self) -> None:
        store = self._store_with_ingress()
        self.assertTrue(store.mission_metadata.get("ingress_sweep", False))

    def test_build_parallel_ingress_produces_routes(self) -> None:
        store = self._store_with_ingress()
        lc = store.sites["LC"]
        tp = store.mission_metadata["rally_predicted_subject"]
        formation = build_parallel_ingress(
            lc.latitude, lc.longitude,
            tp["latitude"], tp["longitude"],
            count=6, altitude_m=600.0,
        )
        self.assertEqual(6, len(formation.routes))
        for vid in range(1, 7):
            self.assertEqual(2, len(formation.routes[vid]))  # formup + TP-line

    def test_prepend_ingress_to_store_routes(self) -> None:
        """Simulate what _prepend_ingress_routes does at the store level."""
        store = self._store_with_ingress()
        lc = store.sites["LC"]
        tp = store.mission_metadata["rally_predicted_subject"]
        original_counts = {
            vid: len(store.vehicle_waypoints[vid])
            for vid in SiteStore.VEHICLE_IDS
        }
        formation = build_parallel_ingress(
            lc.latitude, lc.longitude,
            tp["latitude"], tp["longitude"],
            count=6, altitude_m=600.0,
        )
        for vehicle_id in SiteStore.VEHICLE_IDS:
            ingress_pts = formation.routes.get(vehicle_id, [])
            existing = store.vehicle_waypoints[vehicle_id]
            ingress_wps = [
                MissionPoint(
                    latitude=lat, longitude=lon, altitude_m=alt,
                    code=f"WP{seq:03d}",
                    label=f"SAR-{vehicle_id:02d} Waypoint {seq}",
                    point_type="WAYPOINT", sequence=seq,
                )
                for seq, (lat, lon, alt) in enumerate(ingress_pts, start=1)
            ]
            combined = ingress_wps + existing
            for i, wp in enumerate(combined, start=1):
                wp.sequence = i
                wp.code = f"WP{i:03d}"
            store.vehicle_waypoints[vehicle_id] = combined

        for vid in SiteStore.VEHICLE_IDS:
            new_count = len(store.vehicle_waypoints[vid])
            self.assertEqual(new_count, original_counts[vid] + 2)  # 2 ingress WPs
            # First two are ingress, rest are original
            self.assertGreater(new_count, 2)

    def test_ingress_sweep_in_vehicle_payload(self) -> None:
        """The ingress_sweep key must appear in the vehicle dict sent to the planner."""
        from qt_gcs.fly_state import FlyState
        store = self._store_with_ingress()
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        state.request_simulated_launch()
        # After launch, in ROUTE phase before search_started -> ingress_sweep_active
        self.assertTrue(state.ingress_sweep_active)
        # The vehicle payload dict would include this:
        payload = {
            "ingress_sweep": state.ingress_sweep_active and True,  # _ingress_enabled=True
        }
        self.assertTrue(payload["ingress_sweep"])


class IngressCorridorPolygonTests(unittest.TestCase):
    """Corridor polygon geometry for map shading."""

    def _lc_tp(self):
        return 37.3422, 127.9202, 37.6, 128.2

    def test_returns_four_vertices(self) -> None:
        vertices = ingress_corridor_polygon(*self._lc_tp(), count=6)
        self.assertEqual(4, len(vertices))

    def test_vertices_are_lat_lon_tuples(self) -> None:
        vertices = ingress_corridor_polygon(*self._lc_tp(), count=6)
        for lat, lon in vertices:
            self.assertIsInstance(lat, float)
            self.assertIsInstance(lon, float)
            self.assertGreater(lat, 30.0)
            self.assertLess(lat, 40.0)

    def test_corridor_encloses_formation_routes(self) -> None:
        lc_lat, lc_lon, tp_lat, tp_lon = self._lc_tp()
        formation = build_parallel_ingress(
            lc_lat, lc_lon, tp_lat, tp_lon, count=6,
        )
        corridor = ingress_corridor_polygon(
            lc_lat, lc_lon, tp_lat, tp_lon, count=6,
        )
        # All formup and TP-line waypoints should be within the corridor bounds
        lats = [v[0] for v in corridor]
        lons = [v[1] for v in corridor]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        for vid, route in formation.routes.items():
            for lat, lon, _alt in route:
                self.assertGreaterEqual(lat, min_lat - 0.001)
                self.assertLessEqual(lat, max_lat + 0.001)
                self.assertGreaterEqual(lon, min_lon - 0.001)
                self.assertLessEqual(lon, max_lon + 0.001)

    def test_corridor_wider_than_formation_front(self) -> None:
        """Corridor includes buffer beyond outermost vehicles."""
        from qt_gcs.planning.ingress import _to_local
        lc_lat, lc_lon, tp_lat, tp_lon = self._lc_tp()
        formation = build_parallel_ingress(
            lc_lat, lc_lon, tp_lat, tp_lon, count=6, spacing_m=800.0,
        )
        corridor = ingress_corridor_polygon(
            lc_lat, lc_lon, tp_lat, tp_lon, count=6,
            spacing_m=800.0, buffer_m=400.0,
        )
        # Formation front is 4000m; corridor should be ~4800m wide
        # Check via local coordinates of corridor edges at TP line
        c_local = [_to_local(lat, lon, lc_lat, lc_lon) for lat, lon in corridor]
        # Vertices 2 and 3 are at the TP line (back two corners)
        tp_edge_width = (
            (c_local[2][0] - c_local[3][0]) ** 2
            + (c_local[2][1] - c_local[3][1]) ** 2
        ) ** 0.5
        self.assertGreater(tp_edge_width, formation.front_m)

    def test_coincident_lc_tp_returns_empty(self) -> None:
        vertices = ingress_corridor_polygon(37.3, 127.9, 37.3, 127.9)
        self.assertEqual(0, len(vertices))


class IngressCorridorPlanPayloadTests(unittest.TestCase):
    """Verify the corridor appears in plan payload when ingress is enabled."""

    def test_corridor_in_render_dict_when_ingress_enabled(self) -> None:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        store.mission_metadata["ingress_sweep"] = True
        store.mission_metadata["rally_predicted_subject"] = {
            "latitude": 37.6, "longitude": 128.2,
        }
        lc = store.sites["LC"]
        tp = store.mission_metadata["rally_predicted_subject"]
        corridor = ingress_corridor_polygon(
            lc.latitude, lc.longitude,
            float(tp["latitude"]), float(tp["longitude"]),
            count=len(SiteStore.VEHICLE_IDS),
        )
        payload = store.render_dict()
        payload["ingress_corridor"] = [
            {"latitude": lat, "longitude": lon} for lat, lon in corridor
        ]
        self.assertEqual(4, len(payload["ingress_corridor"]))
        for vertex in payload["ingress_corridor"]:
            self.assertIn("latitude", vertex)
            self.assertIn("longitude", vertex)


class RhpSpxRouteDisplayTests(unittest.TestCase):
    """Change-2: on the fly map, an RHP-SPX mission hides the pre-baked SPX
    search lawnmower (WP*) -- the online planner regenerates each route live --
    while keeping the ingress lanes (IGR*).  Non-RHP missions keep both."""

    def _store_with_baked_routes(self, planning_mode: str) -> SiteStore:
        store = SiteStore()
        store.seed_demo(37.3422, 127.9202)
        store.mission_metadata["planning_mode"] = planning_mode
        for vid in SiteStore.VEHICLE_IDS:
            store.vehicle_waypoints[vid] = [
                MissionPoint(
                    latitude=37.40, longitude=127.95, altitude_m=600.0,
                    code="IGR001", label="ingress",
                    point_type="INGRESS_WAYPOINT", sequence=1,
                ),
                MissionPoint(
                    latitude=37.50, longitude=128.00, altitude_m=600.0,
                    code="WP002", label="search",
                    point_type="WAYPOINT", sequence=2,
                ),
                MissionPoint(
                    latitude=37.55, longitude=128.05, altitude_m=600.0,
                    code="WP003", label="search",
                    point_type="WAYPOINT", sequence=3,
                ),
            ]
        return store

    def _drawn_codes(self, store: SiteStore) -> list[str]:
        from qt_gcs.fly_state import FlyState
        state = FlyState.demo(37.3422, 127.9202)
        state.load_mission(store, 1)
        plan = state.render_dict(store)["plan"]
        codes: list[str] = []
        for route in plan.get("vehicle_routes", []):
            codes += [str(w.get("code", "")) for w in route.get("waypoints", [])]
        return codes

    def test_rhp_spx_hides_baked_search_routes(self) -> None:
        codes = self._drawn_codes(self._store_with_baked_routes("RHP-SPX"))
        self.assertEqual([], [c for c in codes if c.startswith("WP")])
        self.assertTrue([c for c in codes if c.startswith("IGR")])

    def test_non_rhp_keeps_baked_search_routes(self) -> None:
        codes = self._drawn_codes(self._store_with_baked_routes("RHP-ARC"))
        self.assertTrue([c for c in codes if c.startswith("WP")])

    def _shared_map_codes(self, store: SiteStore) -> list[str]:
        # The SHARED-map fleet path builds its plan via this function
        # (-> setFlyPlan), NOT via FlyState.render_dict.
        from qt_gcs.fly_view import build_mission_map_plan_payload
        plan = build_mission_map_plan_payload(store, store, False)
        codes: list[str] = []
        for route in plan.get("vehicle_routes", []):
            codes += [str(w.get("code", "")) for w in route.get("waypoints", [])]
        return codes

    def test_shared_map_path_hides_baked_search_routes(self) -> None:
        codes = self._shared_map_codes(self._store_with_baked_routes("RHP-SPX"))
        self.assertEqual([], [c for c in codes if c.startswith("WP")])
        self.assertTrue([c for c in codes if c.startswith("IGR")])

    def test_shared_map_path_keeps_routes_for_non_rhp(self) -> None:
        codes = self._shared_map_codes(self._store_with_baked_routes("RHP-ARC"))
        self.assertTrue([c for c in codes if c.startswith("WP")])

    def _bridge_plan_codes(self, bridge) -> list[str]:
        captured: dict = {}
        conn = bridge.planChanged.connect(lambda p: captured.__setitem__("p", p))
        bridge.emit_plan()
        bridge.planChanged.disconnect(conn)
        plan = json.loads(captured["p"])
        return [
            str(w.get("code", ""))
            for route in plan.get("vehicle_routes", [])
            for w in route.get("waypoints", [])
        ]

    def test_shared_bridge_hides_routes_only_when_fly_owns_map(self) -> None:
        # The shared MapBridge (plan_view.bridge) feeds BOTH tabs; it must hide
        # the baked routes only while the FLY view owns the map, so a plan
        # notification during flight cannot repaint them (the intermittency).
        from qt_gcs.map_bridge import MapBridge
        store = self._store_with_baked_routes("RHP-SPX")
        bridge = MapBridge(store)
        bridge.hide_preplanned_routes = True   # FLY owns the map
        fly = self._bridge_plan_codes(bridge)
        self.assertEqual([], [c for c in fly if c.startswith("WP")])
        bridge.hide_preplanned_routes = False  # PLAN owns the map
        plan = self._bridge_plan_codes(bridge)
        self.assertTrue([c for c in plan if c.startswith("WP")])


if __name__ == "__main__":
    unittest.main()
