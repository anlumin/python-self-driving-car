"""Arcade car: kinematic bicycle model + crash detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CarConfig, SensorConfig
from .geometry import box_hits_segments
from .sensors import RaySensors
from .track import Track


@dataclass(frozen=True, slots=True)
class CarPose:
    """Everything a renderer (2D or 3D) needs to draw the car."""

    x: float
    y: float
    heading: float  # radians, 0 = +x, counterclockwise positive
    speed: float  # m/s, >= 0
    steer_angle: float  # front wheel angle, radians, positive = right


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


class Car:
    """State = (x, y, heading, speed, steer_angle). (x, y) is the car's center."""

    def __init__(
        self,
        track: Track,
        config: CarConfig | None = None,
        sensor_config: SensorConfig | None = None,
    ) -> None:
        self.config = config or CarConfig()
        self.track = track
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.steer_angle = 0.0
        self.sensors = RaySensors(self, track, sensor_config)
        c = self.config
        self._half_len = c.length / 2.0
        self._half_wid = c.width / 2.0
        self._crash_radius = math.hypot(self._half_len, self._half_wid)  # half diagonal
        self._max_steer = math.radians(c.max_steer_deg)
        self._steer_rate = math.radians(c.steer_rate_deg_s)

    # ---------------------------------------------------------------- state

    def reset(self, x: float, y: float, heading: float, speed: float = 0.0) -> None:
        self.x, self.y, self.heading = float(x), float(y), float(heading)
        self.speed = float(speed)
        self.steer_angle = 0.0
        self.sensors.mark_stale()

    @property
    def pose(self) -> CarPose:
        return CarPose(self.x, self.y, self.heading, self.speed, self.steer_angle)

    def max_steer_at_speed(self, speed: float) -> float:
        """Largest allowed wheel angle at this speed.

        In the bicycle model the turn radius is R = wheelbase / tan(steer), so
        the lateral acceleration is v^2 / R = v^2 * tan(steer) / wheelbase.
        Capping it at max_lateral_accel gives
            steer_max = atan(max_lateral_accel * wheelbase / v^2).
        """
        if speed < 1e-3:
            return self._max_steer
        limit = math.atan(self.config.max_lateral_accel * self.config.wheelbase / (speed * speed))
        return min(self._max_steer, limit)

    # -------------------------------------------------------------- physics

    def step(self, steer: float, throttle: float, dt: float) -> None:
        """Advance the car by dt seconds.

        steer:    [-1, 1], negative = left, positive = right
        throttle: [-1, 1], positive accelerates, negative brakes (no reverse)
        """
        c = self.config
        steer = _clamp(float(steer), -1.0, 1.0)
        throttle = _clamp(float(throttle), -1.0, 1.0)

        # Steering: move the wheels toward the target angle at a limited rate.
        max_steer = self.max_steer_at_speed(self.speed)
        target = steer * max_steer
        max_change = self._steer_rate * dt
        self.steer_angle += _clamp(target - self.steer_angle, -max_change, max_change)
        self.steer_angle = _clamp(self.steer_angle, -max_steer, max_steer)

        # Longitudinal: throttle / brake, minus drag and rolling resistance.
        accel = throttle * c.acceleration if throttle >= 0.0 else throttle * c.braking
        if self.speed > 0.0:
            accel -= c.drag * self.speed + c.rolling_resistance
        self.speed = _clamp(self.speed + accel * dt, 0.0, c.max_speed)

        # Kinematic bicycle model about the car's center (rear axle half a
        # wheelbase behind it). beta is the angle between the heading and the
        # direction the center actually moves:
        #     beta = atan(lr / wheelbase * tan(steer)),   lr = wheelbase / 2
        # and the heading turns at  v / lr * sin(beta).
        # Our steer angle is positive to the RIGHT (clockwise) while heading is
        # counterclockwise-positive, hence the minus signs.
        lr = c.wheelbase / 2.0
        beta = math.atan(0.5 * math.tan(self.steer_angle))
        move_dir = self.heading - beta
        self.x += self.speed * math.cos(move_dir) * dt
        self.y += self.speed * math.sin(move_dir) * dt
        self.heading -= self.speed / lr * math.sin(beta) * dt
        # keep heading in (-pi, pi] so it never grows without bound
        if self.heading > math.pi:
            self.heading -= 2.0 * math.pi
        elif self.heading <= -math.pi:
            self.heading += 2.0 * math.pi

        self.sensors.mark_stale()

    # ------------------------------------------------------------ geometry

    def front_center(self) -> tuple[float, float]:
        return (
            self.x + math.cos(self.heading) * self._half_len,
            self.y + math.sin(self.heading) * self._half_len,
        )

    def corners(self) -> list[tuple[float, float]]:
        """Footprint rectangle corners: front-left, front-right, rear-right, rear-left."""
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        out = []
        for lx, ly in (
            (self._half_len, self._half_wid),
            (self._half_len, -self._half_wid),
            (-self._half_len, -self._half_wid),
            (-self._half_len, self._half_wid),
        ):
            out.append((self.x + lx * ch - ly * sh, self.y + lx * sh + ly * ch))
        return out

    def is_crashed(self) -> bool:
        """True if the footprint rectangle touches any wall segment."""
        ax, ay, bx, by, _ex, _ey = self.track.walls_near(self.x, self.y, self._crash_radius)
        return box_hits_segments(
            self.x,
            self.y,
            math.cos(self.heading),
            math.sin(self.heading),
            self._half_len,
            self._half_wid,
            ax,
            ay,
            bx,
            by,
        )
