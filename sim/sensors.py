"""Ray sensors: 5 (configurable) distance sensors at the front of the car."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import SensorConfig
from .geometry import cast_rays

if TYPE_CHECKING:  # only for type hints, avoids a circular import at runtime
    from .car import Car
    from .track import Track


@dataclass(frozen=True, slots=True)
class SensorReading:
    """Result of one ray.

    hit:        True if a wall was found within max_range
    distance:   meters to the hit (== max_range when nothing was hit)
    normalized: distance / max_range  (1.0 = nothing in range, 0.0 = touching)
    point:      hit position (x, y), or the ray's end point if nothing was hit
    origin:     ray start (x, y), the front-center of the car
    angle_deg:  ray angle relative to the heading (negative = left)
    """

    hit: bool
    distance: float
    normalized: float
    point: tuple[float, float]
    origin: tuple[float, float]
    angle_deg: float


class RaySensors:
    """Casts rays from the car's front-center against the track walls.

    Results are computed lazily: the car marks them stale after every physics
    step, and they are recomputed (once) the next time someone asks.
    """

    def __init__(self, car: Car, track: Track, config: SensorConfig | None = None) -> None:
        self.car = car
        self.track = track
        self.config = config or SensorConfig()
        self.max_range = float(self.config.max_range)
        # Sensor angles use the driver convention (positive = right). Heading is
        # counterclockwise-positive, so a ray's world angle is heading - angle.
        self.angles_deg = tuple(float(a) for a in self.config.angles_deg)
        self._angles_rad = -np.radians(np.array(self.angles_deg))
        self.count = len(self.angles_deg)

        self.distances = np.full(self.count, self.max_range)
        self.hits = np.zeros(self.count, dtype=bool)
        self.origin = (0.0, 0.0)
        self._dir_x = np.zeros(self.count)
        self._dir_y = np.zeros(self.count)
        self._stale = True

    def mark_stale(self) -> None:
        self._stale = True

    def update(self) -> None:
        """Recompute all rays for the car's current pose."""
        car = self.car
        ch, sh = math.cos(car.heading), math.sin(car.heading)
        half_len = car.config.length / 2.0
        ox, oy = car.x + ch * half_len, car.y + sh * half_len  # front-center
        world = car.heading + self._angles_rad
        np.cos(world, out=self._dir_x)
        np.sin(world, out=self._dir_y)
        ax, ay, _bx, _by, ex, ey = self.track.walls_near(ox, oy, self.max_range)
        self.distances, self.hits = cast_rays(
            ox, oy, self._dir_x, self._dir_y, ax, ay, ex, ey, self.max_range
        )
        self.origin = (ox, oy)
        self._stale = False

    def _ensure_fresh(self) -> None:
        if self._stale:
            self.update()

    def normalized(self) -> np.ndarray:
        """Distances divided by max range, in [0, 1], ordered left to right."""
        self._ensure_fresh()
        return self.distances / self.max_range

    def read(self) -> list[SensorReading]:
        """One SensorReading per ray, ordered left to right."""
        self._ensure_fresh()
        ox, oy = self.origin
        readings = []
        for i in range(self.count):
            d = float(self.distances[i])
            readings.append(
                SensorReading(
                    hit=bool(self.hits[i]),
                    distance=d,
                    normalized=d / self.max_range,
                    point=(ox + float(self._dir_x[i]) * d, oy + float(self._dir_y[i]) * d),
                    origin=(ox, oy),
                    angle_deg=self.angles_deg[i],
                )
            )
        return readings
