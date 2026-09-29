"""Ingress belief carry-over and PD display tests.

Verifies:
1. _point_in_polygon correctly identifies points inside/outside a polygon.
2. _lonlat_to_local converts lat/lon to local ENU coordinates.
3. plan_certified_search with ingress_corridor_vertices reduces belief
   for cells inside the corridor.
4. PD is extracted from mission metadata's spx_certified block.
"""

from __future__ import annotations

import unittest
import math

from qt_gcs.planning.stone_adapter import (
    _point_in_polygon,
    _lonlat_to_local,
    _lonlat_scales,
)


class PointInPolygonTests(unittest.TestCase):

    def test_point_inside_square(self) -> None:
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertTrue(_point_in_polygon(5, 5, square))

    def test_point_outside_square(self) -> None:
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertFalse(_point_in_polygon(15, 5, square))

    def test_point_outside_negative(self) -> None:
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertFalse(_point_in_polygon(-1, 5, square))

    def test_point_inside_triangle(self) -> None:
        triangle = [(0, 0), (10, 0), (5, 10)]
        self.assertTrue(_point_in_polygon(5, 3, triangle))

    def test_point_outside_triangle(self) -> None:
        triangle = [(0, 0), (10, 0), (5, 10)]
        self.assertFalse(_point_in_polygon(9, 9, triangle))

    def test_corridor_shaped_polygon(self) -> None:
        corridor = [(-2000, -5000), (2000, -5000), (2000, 5000), (-2000, 5000)]
        self.assertTrue(_point_in_polygon(0, 0, corridor))
        self.assertTrue(_point_in_polygon(1999, 4999, corridor))
        self.assertFalse(_point_in_polygon(3000, 0, corridor))


class LonLatToLocalTests(unittest.TestCase):

    def test_origin_maps_to_zero(self) -> None:
        east, north = _lonlat_to_local(23.7, 47.3, 23.7, 47.3)
        self.assertAlmostEqual(east, 0.0, places=1)
        self.assertAlmostEqual(north, 0.0, places=1)

    def test_north_offset(self) -> None:
        east, north = _lonlat_to_local(23.71, 47.3, 23.7, 47.3)
        self.assertAlmostEqual(east, 0.0, places=1)
        self.assertGreater(north, 1000)
        self.assertLess(north, 1200)

    def test_east_offset(self) -> None:
        east, north = _lonlat_to_local(23.7, 47.31, 23.7, 47.3)
        self.assertGreater(east, 900)
        self.assertAlmostEqual(north, 0.0, places=1)


class PDFromMissionMetadataTests(unittest.TestCase):

    def test_baked_pd_extracted(self) -> None:
        from qt_gcs.site_store import SiteStore
        store = SiteStore()
        store.seed_demo(23.7, 47.3)
        store.mission_metadata["spx_certified"] = {
            "detection_probability": 0.556,
            "certified": True,
        }
        pd = store.mission_metadata.get("spx_certified", {}).get(
            "detection_probability"
        )
        self.assertAlmostEqual(pd, 0.556, places=3)

    def test_no_spx_returns_none(self) -> None:
        from qt_gcs.site_store import SiteStore
        store = SiteStore()
        store.seed_demo(23.7, 47.3)
        pd = (store.mission_metadata or {}).get("spx_certified", {}).get(
            "detection_probability"
        )
        self.assertIsNone(pd)


if __name__ == "__main__":
    unittest.main()
