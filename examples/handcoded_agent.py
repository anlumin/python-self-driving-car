"""A hand-written driver that only looks at the sensor observation.

Steering: turn toward the side whose rays see more free space.
Throttle: pick a target speed from the free distance straight ahead (enough
room to brake), then accelerate / brake toward it.

    python examples/handcoded_agent.py           # headless, prints the result
    python examples/handcoded_agent.py --watch   # watch it in the 2D viewer

This is the baseline that proves the sensor data is useful for driving. It
needs the observation to include speed (config.env.include_speed_in_obs).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import RaceEnv, SimConfig  # noqa: E402


@dataclass
class HandcodedAgent:
    """Callable policy: agent(obs) -> (steer, throttle).

    obs = [n(-60), n(-30), n(0), n(+30), n(+60), speed / max_speed], where n is
    a normalized ray distance (1.0 = free up to max range).
    """

    max_range: float = 60.0
    max_speed: float = 40.0
    steer_gain: float = 1.25
    safe_decel: float = 12.0  # m/s^2 we plan to brake with (car can do 25)
    margin: float = 6.0  # m of free space to keep in front
    min_speed: float = 7.0  # m/s, never plan slower than this
    speed_gain: float = 0.5  # throttle per m/s of speed error

    def __call__(self, obs: np.ndarray) -> tuple[float, float]:
        far_left, left, front, right, far_right = (float(v) for v in obs[:5])
        speed = float(obs[5]) * self.max_speed

        # More free space on the right -> positive steer (= turn right).
        steer = self.steer_gain * ((right + 0.5 * far_right) - (left + 0.5 * far_left))
        steer = max(-1.0, min(1.0, steer))

        # Fastest speed that can still stop before `margin`: v^2 = 2 * a * d.
        free = front * self.max_range - self.margin
        target = math.sqrt(2.0 * self.safe_decel * max(free, 0.0))
        target = max(self.min_speed, min(self.max_speed, target))
        throttle = max(-1.0, min(1.0, self.speed_gain * (target - speed)))
        return steer, throttle


def make_config() -> SimConfig:
    config = SimConfig()
    config.env.include_speed_in_obs = True
    return config


def run_episode(agent: HandcodedAgent | None = None, config: SimConfig | None = None,
                seed: int = 0) -> dict:
    config = config or make_config()
    agent = agent or HandcodedAgent(max_range=config.sensors.max_range,
                                    max_speed=config.car.max_speed)
    env = RaceEnv(config, seed=seed)
    obs = env.reset()
    done = False
    info: dict = {}
    while not done:
        obs, done, info = env.step(*agent(obs))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", action="store_true", help="open the 2D viewer")
    parser.add_argument("--speed", type=int, default=1, help="sim steps per rendered frame")
    args = parser.parse_args()

    config = make_config()
    agent = HandcodedAgent(max_range=config.sensors.max_range, max_speed=config.car.max_speed)
    if args.watch:
        from render2d import Viewer

        Viewer(RaceEnv(config), policy=agent, steps_per_frame=args.speed).run()
        return

    info = run_episode(agent, config)
    print(
        f"done: {info['done_reason']}  time {info['time']:.1f}s  "
        f"checkpoints {info['checkpoints_passed']}  laps {info['laps']}  "
        f"progress {info['progress_m']:.1f} m"
    )


if __name__ == "__main__":
    main()
