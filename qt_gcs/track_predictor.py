from __future__ import annotations

import math
from dataclasses import dataclass


EARTH_METERS_PER_DEGREE = 111_320.0
CHI_SQUARE_95_2D = 5.991


@dataclass(frozen=True)
class AxisForecast:
    position: float
    velocity: float
    position_variance: float


class ConstantVelocityAxisFilter:
    """Two-state [position, velocity] discrete Kalman filter for one axis."""

    def __init__(
        self,
        position: float,
        *,
        velocity: float = 0.0,
        measurement_std: float = 35.0,
        acceleration_std: float = 8.0,
    ) -> None:
        self.position = float(position)
        self.velocity = float(velocity)
        self.measurement_variance = float(measurement_std) ** 2
        self.acceleration_variance = float(acceleration_std) ** 2
        self.p00 = self.measurement_variance
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 2_500.0

    def _predicted_covariance(
        self,
        dt: float,
    ) -> tuple[float, float, float, float]:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.acceleration_variance
        return (
            self.p00 + dt * (self.p01 + self.p10) + dt2 * self.p11 + q * dt4 / 4,
            self.p01 + dt * self.p11 + q * dt3 / 2,
            self.p10 + dt * self.p11 + q * dt3 / 2,
            self.p11 + q * dt2,
        )

    def update(self, measurement: float, dt: float) -> None:
        dt = max(0.01, min(float(dt), 2.0))
        predicted_position = self.position + self.velocity * dt
        predicted_velocity = self.velocity
        p00, p01, p10, p11 = self._predicted_covariance(dt)

        innovation = float(measurement) - predicted_position
        innovation_variance = max(1e-9, p00 + self.measurement_variance)
        gain_position = p00 / innovation_variance
        gain_velocity = p10 / innovation_variance

        self.position = predicted_position + gain_position * innovation
        self.velocity = predicted_velocity + gain_velocity * innovation
        updated_p00 = (1.0 - gain_position) * p00
        updated_p01 = (1.0 - gain_position) * p01
        updated_p10 = p10 - gain_velocity * p00
        updated_p11 = p11 - gain_velocity * p01
        cross = (updated_p01 + updated_p10) / 2.0
        self.p00 = max(1e-9, updated_p00)
        self.p01 = cross
        self.p10 = cross
        self.p11 = max(1e-9, updated_p11)

    def forecast(self, horizon_s: float) -> AxisForecast:
        horizon_s = max(0.0, float(horizon_s))
        p00, _p01, _p10, _p11 = self._predicted_covariance(horizon_s)
        return AxisForecast(
            position=self.position + self.velocity * horizon_s,
            velocity=self.velocity,
            position_variance=max(1e-9, p00),
        )


@dataclass(frozen=True)
class TrackPrediction:
    latitude: float
    longitude: float
    altitude_m: float
    velocity_east_mps: float
    velocity_north_mps: float
    velocity_up_mps: float
    uncertainty_east_95_m: float
    uncertainty_north_95_m: float
    uncertainty_altitude_95_m: float
    horizon_s: float

    @property
    def estimated_speed_mps(self) -> float:
        return math.hypot(self.velocity_east_mps, self.velocity_north_mps)

    @property
    def estimated_heading_deg(self) -> float:
        return (
            math.degrees(
                math.atan2(self.velocity_east_mps, self.velocity_north_mps)
            )
            + 360.0
        ) % 360.0


class ConstantVelocityTrackPredictor:
    """Independent ENU-axis CV Kalman filters with covariance prediction."""

    def __init__(
        self,
        reference_latitude: float,
        reference_longitude: float,
        latitude: float,
        longitude: float,
        altitude_m: float,
    ) -> None:
        self.reference_latitude = float(reference_latitude)
        self.reference_longitude = float(reference_longitude)
        east, north = self._to_local(latitude, longitude)
        self.east = ConstantVelocityAxisFilter(east, measurement_std=35.0)
        self.north = ConstantVelocityAxisFilter(north, measurement_std=35.0)
        self.up = ConstantVelocityAxisFilter(
            altitude_m,
            measurement_std=60.0,
            acceleration_std=5.0,
        )

    def _longitude_scale(self) -> float:
        return max(
            10_000.0,
            EARTH_METERS_PER_DEGREE
            * math.cos(math.radians(self.reference_latitude)),
        )

    def _to_local(
        self,
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:
        east = (
            float(longitude) - self.reference_longitude
        ) * self._longitude_scale()
        north = (
            float(latitude) - self.reference_latitude
        ) * EARTH_METERS_PER_DEGREE
        return east, north

    def _to_geographic(
        self,
        east: float,
        north: float,
    ) -> tuple[float, float]:
        latitude = self.reference_latitude + north / EARTH_METERS_PER_DEGREE
        longitude = self.reference_longitude + east / self._longitude_scale()
        return latitude, longitude

    def update(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        dt: float,
    ) -> None:
        east, north = self._to_local(latitude, longitude)
        self.east.update(east, dt)
        self.north.update(north, dt)
        self.up.update(float(altitude_m), dt)

    def estimate(self) -> TrackPrediction:
        return self.predict(0.0)

    def predict(self, horizon_s: float) -> TrackPrediction:
        east = self.east.forecast(horizon_s)
        north = self.north.forecast(horizon_s)
        up = self.up.forecast(horizon_s)
        latitude, longitude = self._to_geographic(
            east.position,
            north.position,
        )
        confidence_scale = math.sqrt(CHI_SQUARE_95_2D)
        return TrackPrediction(
            latitude=latitude,
            longitude=longitude,
            altitude_m=max(0.0, up.position),
            velocity_east_mps=east.velocity,
            velocity_north_mps=north.velocity,
            velocity_up_mps=up.velocity,
            uncertainty_east_95_m=confidence_scale
            * math.sqrt(east.position_variance),
            uncertainty_north_95_m=confidence_scale
            * math.sqrt(north.position_variance),
            uncertainty_altitude_95_m=1.96
            * math.sqrt(up.position_variance),
            horizon_s=float(horizon_s),
        )
