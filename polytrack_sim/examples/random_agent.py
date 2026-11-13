"""Minimal example of using RaceEnv headless with a random policy.

    python examples/random_agent.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import RaceEnv, SimConfig  # noqa: E402


def main() -> None:
    config = SimConfig()
    config.env.include_speed_in_obs = True  # obs = 5 sensors + speed

    env = RaceEnv(config, seed=0)
    rng = np.random.default_rng(0)

    for episode in range(3):
        obs = env.reset()
        done = False
        while not done:
            steer = rng.uniform(-1.0, 1.0)
            throttle = rng.uniform(0.0, 1.0)
            obs, done, info = env.step(steer, throttle)

        # A fitness function would usually combine these:
        print(
            f"episode {episode}: {info['done_reason']:<8} "
            f"time={info['time']:5.1f}s  checkpoints={info['checkpoints_passed']:2d}  "
            f"progress={info['progress_m']:6.1f} m  last obs={np.round(obs, 2)}"
        )
        for r in info["sensors"]:
            print(f"    sensor {r.angle_deg:+5.0f} deg: hit={r.hit!s:<5} "
                  f"distance={r.distance:5.1f} m  normalized={r.normalized:.2f}")


if __name__ == "__main__":
    main()
