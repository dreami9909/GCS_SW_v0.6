from __future__ import annotations

import math
import unittest

from qt_gcs.track_predictor import (
    EARTH_METERS_PER_DEGREE,
    ConstantVelocityTrackPredictor,
)


class ConstantVelocityTrackPredictorTests(unittest.TestCase):
    def test_learns_constant_east_velocity_and_forecasts(self) -> None:
        reference_latitude = 37.3422
        reference_longitude = 127.9202
        longitude_scale = (
            EARTH_METERS_PER_DEGREE
            * math.cos(math.radians(reference_latitude))
        )
        predictor = ConstantVelocityTrackPredictor(
            reference_latitude,
            reference_longitude,
            reference_latitude,
            reference_longitude,
            1_000.0,
        )

        speed_mps = 25.0
        dt = 0.2
        for sample in range(1, 61):
            east_m = speed_mps * sample * dt
            predictor.update(
                reference_latitude,
                reference_longitude + east_m / longitude_scale,
                1_000.0,
                dt,
            )

        estimate = predictor.estimate()
        prediction = predictor.predict(6.0)
        predicted_east_m = (
            prediction.longitude - reference_longitude
        ) * longitude_scale

        self.assertAlmostEqual(speed_mps, estimate.velocity_east_mps, delta=1.5)
        self.assertAlmostEqual(
            speed_mps * (60 * dt + 6.0),
            predicted_east_m,
            delta=18.0,
        )
        self.assertGreater(prediction.uncertainty_east_95_m, 0.0)
        self.assertGreater(prediction.uncertainty_north_95_m, 0.0)

    def test_stationary_measurements_keep_prediction_near_origin(self) -> None:
        predictor = ConstantVelocityTrackPredictor(
            37.3422,
            127.9202,
            37.3422,
            127.9202,
            500.0,
        )
        for _ in range(30):
            predictor.update(37.3422, 127.9202, 500.0, 0.25)

        prediction = predictor.predict(6.0)
        self.assertAlmostEqual(37.3422, prediction.latitude, places=5)
        self.assertAlmostEqual(127.9202, prediction.longitude, places=5)
        self.assertAlmostEqual(0.0, prediction.estimated_speed_mps, delta=0.2)


if __name__ == "__main__":
    unittest.main()
