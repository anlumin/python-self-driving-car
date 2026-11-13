"""Command line entry point.

    python main.py --manual                 drive yourself (WASD / arrow keys)
    python main.py --demo                   watch the hand-coded agent
    python main.py --demo --cars 12         watch a "generation" of 12 agents at once
    python main.py --benchmark              headless speed test (does not import pygame)
    python main.py --benchmark --steps 200000

Options: --speed N (sim steps per rendered frame), --seed N.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import RaceEnv, SimConfig  # noqa: E402


def benchmark(steps: int, seed: int) -> float:
    """Run a random policy headless for `steps` steps; print and return steps/s.

    Resets after every episode end (the reset time is included). Uses all 5
    sensors because every step builds the observation and the info dict.
    """
    env = RaceEnv(SimConfig(), seed=seed)
    rng = np.random.default_rng(seed)
    # Random but "sticky" actions so the car actually drives around a bit
    # instead of jittering in place.
    actions = rng.uniform(-1.0, 1.0, size=(steps // 30 + 1, 2))
    actions[:, 1] = np.abs(actions[:, 1])  # mostly forward
    env.reset()
    episodes = 0
    t0 = time.perf_counter()
    for i in range(steps):
        steer, throttle = actions[i // 30]
        _obs, done, _info = env.step(steer, throttle)
        if done:
            episodes += 1
            env.reset()
    elapsed = time.perf_counter() - t0
    rate = steps / elapsed
    print(f"benchmark: {steps} steps in {elapsed:.2f} s -> {rate:,.0f} steps/s "
          f"({rate * env.dt:,.0f}x real time, {episodes} episodes, "
          f"{env.car.sensors.count} sensors)")
    print(f"pygame imported: {'pygame' in sys.modules}")
    return rate


def main() -> None:
    parser = argparse.ArgumentParser(description="2D top-down racing simulator")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manual", action="store_true", help="drive with the keyboard")
    mode.add_argument("--demo", action="store_true", help="watch the hand-coded agent")
    mode.add_argument("--benchmark", action="store_true", help="headless speed test")
    parser.add_argument("--steps", type=int, default=100_000, help="benchmark steps")
    parser.add_argument("--cars", type=int, default=1, help="number of cars in --demo")
    parser.add_argument("--speed", type=int, default=1, help="sim steps per rendered frame")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.steps, args.seed)
        return

    # Only the visual modes import pygame.
    from render2d import Viewer

    if args.manual:
        Viewer(RaceEnv(SimConfig(), seed=args.seed), steps_per_frame=args.speed).run()
        return

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
    from handcoded_agent import HandcodedAgent, make_config

    config = make_config()
    if args.cars <= 1:
        env = RaceEnv(config, seed=args.seed)
        Viewer(env, policy=HandcodedAgent(), steps_per_frame=args.speed).run()
        return

    # Several cars: same driver with randomly perturbed parameters, like a GA
    # generation would have. All envs share one Track object.
    rng = np.random.default_rng(args.seed)
    config.env.start_lateral_jitter = 1.5
    envs, policies = [], []
    for i in range(args.cars):
        envs.append(RaceEnv(config, seed=args.seed + i))
        policies.append(HandcodedAgent(
            steer_gain=float(rng.uniform(0.6, 3.5)),
            safe_decel=float(rng.uniform(6.0, 20.0)),
            margin=float(rng.uniform(2.0, 10.0)),
        ))
    Viewer(envs, policy=policies, steps_per_frame=args.speed).run()


if __name__ == "__main__":
    main()
