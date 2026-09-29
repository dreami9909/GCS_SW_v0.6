"""Tests for SearchCameraSpec detection-reach behaviour."""

from __future__ import annotations

import math

import pytest

from qt_gcs.planning.sensor_model import SearchCameraSpec


def _raw_reach(gimbal_deg: float, altitude_m: float = 600.0) -> float:
    return altitude_m * math.tan(math.radians(gimbal_deg))


class TestEffectiveDetectReach:
    def test_defaults_to_full_gimbal_reach(self):
        spec = SearchCameraSpec(max_gimbal_angle_deg=63.0)
        assert spec.detect_reach_m is None
        assert spec.effective_detect_reach_m == pytest.approx(
            spec.gimbal_centerline_reach_m
        )
        # ~1,177 m at 63 deg / 600 m altitude.
        assert spec.effective_detect_reach_m == pytest.approx(
            _raw_reach(63.0), rel=1e-3
        )

    def test_cap_below_raw_reach_clamps(self):
        spec = SearchCameraSpec(
            max_gimbal_angle_deg=63.0, detect_within_reach=True,
            detect_reach_m=240.0,
        )
        assert spec.effective_detect_reach_m == pytest.approx(240.0)
        assert spec.effective_detect_reach_m < spec.gimbal_centerline_reach_m

    def test_cap_above_raw_reach_keeps_raw(self):
        # A cap wider than the physical reach never widens detection.
        spec = SearchCameraSpec(max_gimbal_angle_deg=45.0, detect_reach_m=5_000.0)
        assert spec.effective_detect_reach_m == pytest.approx(
            spec.gimbal_centerline_reach_m
        )

    def test_non_positive_cap_rejected(self):
        with pytest.raises(ValueError):
            SearchCameraSpec(detect_reach_m=0.0)
        with pytest.raises(ValueError):
            SearchCameraSpec(detect_reach_m=-10.0)

    def test_display_dict_exposes_effective_reach(self):
        spec = SearchCameraSpec(
            max_gimbal_angle_deg=63.0, detect_reach_m=240.0,
        )
        payload = spec.display_dict()
        assert payload["effective_detect_reach_m"] == pytest.approx(240.0)
