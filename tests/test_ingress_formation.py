"""Parallel ingress-sweep formation geometry (Stage 2)."""

from __future__ import annotations

import math
import unittest

from qt_gcs.planning.ingress import build_parallel_ingress

EARTH = 111_320.0


def _local(lat, lon, olat, olon):
    e = (lon - olon) * max(10_000.0, EARTH * math.cos(math.radians(olat)))
    n = (lat - olat) * EARTH
    return e, n


class IngressFormationTests(unittest.TestCase):
    LAUNCH = (23.65, 47.05)
    TP = (23.743784095611126, 47.328064915102644)

    def _build(self, **kw):
        return build_parallel_ingress(*self.LAUNCH, *self.TP, **kw)

    def test_six_vehicles_spaced_800m_on_a_line(self) -> None:
        f = self._build(count=6, spacing_m=800.0)
        self.assertEqual(6, len(f.routes))
        # TP-line points (second waypoint) are the abreast line; adjacent gaps == spacing
        tp_pts = [f.routes[v][1] for v in range(1, 7)]
        locs = [_local(la, lo, *self.TP) for la, lo, _ in tp_pts]
        gaps = [math.dist(locs[i], locs[i + 1]) for i in range(5)]
        for g in gaps:
            self.assertAlmostEqual(800.0, g, delta=1.0)
        self.assertAlmostEqual(4000.0, f.front_m, places=6)

    def test_formation_line_passes_through_tp(self) -> None:
        # The mid of the abreast TP-line equals TP (offsets are symmetric).
        f = self._build(count=6, spacing_m=800.0)
        tp_pts = [f.routes[v][1] for v in range(1, 7)]
        locs = [_local(la, lo, *self.TP) for la, lo, _ in tp_pts]
        mid_e = sum(e for e, _ in locs) / 6
        mid_n = sum(n for _, n in locs) / 6
        self.assertAlmostEqual(0.0, math.hypot(mid_e, mid_n), delta=1.0)

    def test_vehicles_fly_parallel_fixed_heading(self) -> None:
        # Each vehicle's form-up -> TP-line leg has the same heading (parallel).
        f = self._build(count=6, spacing_m=800.0)
        headings = []
        for v in range(1, 7):
            (la0, lo0, _), (la1, lo1, _) = f.routes[v]
            e0, n0 = _local(la0, lo0, *self.LAUNCH)
            e1, n1 = _local(la1, lo1, *self.LAUNCH)
            headings.append(math.degrees(math.atan2(e1 - e0, n1 - n0)) % 360.0)
        for h in headings[1:]:
            self.assertAlmostEqual(headings[0], h, delta=0.5)
        # and equal to the launcher->TP azimuth
        self.assertAlmostEqual(f.heading_deg, headings[0], delta=0.5)

    def test_odd_count_centered(self) -> None:
        f = self._build(count=5, spacing_m=800.0)
        self.assertAlmostEqual(0.0, f.offsets_m[2], places=6)  # centre vehicle on axis
        self.assertEqual(5, len(f.routes))


if __name__ == "__main__":
    unittest.main()
