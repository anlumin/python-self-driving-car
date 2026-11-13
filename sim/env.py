"""Gym-like, headless racing environment.

    env = RaceEnv(config=None, seed=0)
    obs = env.reset()
    obs, done, info = env.step(steer, throttle)

No reward or fitness function on purpose: build yours from `info`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .car import Car, CarPose
from .config import SimConfig
from .sensors import SensorReading
from .track import Checkpoint, Track, load_track


@dataclass(frozen=True)
class EnvSnapshot:
    """Read-only view of one environment, for renderers (2D now, 3D later)."""

    pose: CarPose
    length: float
    width: float
    sensors: list[SensorReading]
    corners: list[tuple[float, float]]
    crashed: bool
    done: bool
    done_reason: str
    laps: int
    checkpoints_passed: int
    next_checkpoint: int  # index into track.checkpoints
    progress_m: float
    time: float


class RaceEnv:
    """One car on the track, stepped at a fixed dt.

    Episode ends (`done`) when the car crashes, finishes `laps_to_finish`
    laps, exceeds `max_episode_time`, or passes no new checkpoint for
    `stall_timeout` seconds. The reason is in info["done_reason"].
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        seed: int | None = 0,
        track: Track | None = None,
    ) -> None:
        self.config = config or SimConfig()
        self.track = track or load_track(self.config.track)
        self.car = Car(self.track, self.config.car, self.config.sensors)
        self.dt = float(self.config.env.dt)
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ API

    @property
    def observation_size(self) -> int:
        return self.car.sensors.count + (1 if self.config.env.include_speed_in_obs else 0)

    def reset(
        self,
        seed: int | None = None,
        start_s: float | None = None,
        reverse: bool = False,
    ) -> np.ndarray:
        """Put the car back at the start line and clear all counters.

        seed:    re-seed the env's random generator (only used for start jitter)
        start_s: start at this arc length instead of config.env.start_s
                 (handy for tests and curriculum training)
        reverse: face against the driving direction (used by the tests)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ec = self.config.env
        s0 = ec.start_s if start_s is None else start_s
        (x, y), (tx, ty) = self.track.point_at(s0)
        heading = math.atan2(ty, tx)
        if ec.start_lateral_jitter > 0:
            off = self.rng.uniform(-ec.start_lateral_jitter, ec.start_lateral_jitter)
            x, y = x - ty * off, y + tx * off  # along the left normal
        if ec.start_heading_jitter_deg > 0:
            j = math.radians(ec.start_heading_jitter_deg)
            heading += self.rng.uniform(-j, j)
        if reverse:
            heading += math.pi
        self.car.reset(x, y, heading)

        self.time = 0.0
        self.steps = 0
        self.crashed = False
        self.done = False
        self.done_reason = ""
        self.laps = 0
        self.checkpoints_passed = 0
        self._time_of_last_checkpoint = 0.0

        # Progress bookkeeping: projection onto the centerline, unwrapped so it
        # keeps growing past the start line (and goes negative when driving backward).
        s_now, self._track_hint = self.track.project(x, y, self.track.nearest_index(x, y))
        self._last_s = s_now
        self.progress_m = 0.0
        # Next gate = first gate in front of the start position.
        self.next_checkpoint = self._first_gate_after(s_now)

        self.crashed = self.car.is_crashed()
        if self.crashed:
            self.done, self.done_reason = True, "crashed"
        return self.observation()

    def step(self, steer: float, throttle: float) -> tuple[np.ndarray, bool, dict[str, Any]]:
        """Advance one fixed timestep. Returns (observation, done, info).

        Calling step() after the episode is done does nothing and returns the
        final state again.
        """
        if self.done:
            return self.observation(), True, self.info()

        car = self.car
        x0, y0 = car.x, car.y
        car.step(steer, throttle, self.dt)
        self.time += self.dt
        self.steps += 1

        self._update_checkpoints(x0, y0, car.x, car.y)
        self._update_progress(car.x, car.y)

        ec = self.config.env
        if car.is_crashed():
            self.crashed = True
            self.done, self.done_reason = True, "crashed"
        elif self.laps >= ec.laps_to_finish:
            self.done, self.done_reason = True, "finished"
        elif self.time >= ec.max_episode_time:
            self.done, self.done_reason = True, "timeout"
        elif self.time - self._time_of_last_checkpoint >= ec.stall_timeout:
            self.done, self.done_reason = True, "stalled"

        return self.observation(), self.done, self.info()

    def observation(self) -> np.ndarray:
        """5 normalized sensor distances (left to right) [+ speed / max_speed]."""
        obs = self.car.sensors.normalized()
        if self.config.env.include_speed_in_obs:
            obs = np.append(obs, self.car.speed / self.config.car.max_speed)
        return obs

    def info(self) -> dict[str, Any]:
        return {
            "crashed": self.crashed,
            "checkpoints_passed": self.checkpoints_passed,
            "laps": self.laps,
            "progress_m": self.progress_m,
            "time": self.time,
            "speed": self.car.speed,
            "sensors": self.car.sensors.read(),
            "next_checkpoint": self.next_checkpoint,
            "done_reason": self.done_reason,
            "steps": self.steps,
        }

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            pose=self.car.pose,
            length=self.config.car.length,
            width=self.config.car.width,
            sensors=self.car.sensors.read(),
            corners=self.car.corners(),
            crashed=self.crashed,
            done=self.done,
            done_reason=self.done_reason,
            laps=self.laps,
            checkpoints_passed=self.checkpoints_passed,
            next_checkpoint=self.next_checkpoint,
            progress_m=self.progress_m,
            time=self.time,
        )

    # ------------------------------------------------------------ internals

    def _first_gate_after(self, s: float) -> int:
        for gate in self.track.checkpoints:
            if gate.s > s + 1e-9:
                return gate.index
        return 0

    @staticmethod
    def _crossed_forward(gate: Checkpoint, half_width: float,
                         x0: float, y0: float, x1: float, y1: float) -> bool:
        """Did the car's center move from behind the gate to in front of it?

        'Behind/in front' = signed distance along the gate's driving direction.
        Moving the other way (driving backward) never counts. The lateral check
        makes sure the crossing happened on the gate itself, not on the gate's
        infinite extension through some other part of the track.
        """
        cx, cy = gate.center
        tx, ty = gate.tangent
        before = (x0 - cx) * tx + (y0 - cy) * ty
        after = (x1 - cx) * tx + (y1 - cy) * ty
        if not (before < 0.0 <= after):
            return False
        lateral = -(x1 - cx) * ty + (y1 - cy) * tx  # distance along the left normal
        return abs(lateral) <= half_width + 1.0

    def _update_checkpoints(self, x0: float, y0: float, x1: float, y1: float) -> None:
        gates = self.track.checkpoints
        half_w = self.track.width / 2.0
        # A loop (not an if) in case a very fast car crosses two gates in one step.
        while self._crossed_forward(gates[self.next_checkpoint], half_w, x0, y0, x1, y1):
            gate = gates[self.next_checkpoint]
            self.checkpoints_passed += 1
            self._time_of_last_checkpoint = self.time
            if gate.is_finish:
                self.laps += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(gates)

    def _update_progress(self, x: float, y: float) -> None:
        s, self._track_hint = self.track.project(x, y, self._track_hint)
        delta = s - self._last_s
        # unwrap across the start line: a jump of ~ +/- length is really a small step
        half = self.track.length / 2.0
        if delta > half:
            delta -= self.track.length
        elif delta < -half:
            delta += self.track.length
        self.progress_m += delta
        self._last_s = s
