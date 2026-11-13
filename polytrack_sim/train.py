"""Your training script. Write your genetic algorithm here.

    python train.py              # run training
    python train.py --watch      # after training: watch a policy in the viewer

Everything below the "YOUR CODE" markers is yours to replace. The parts that
talk to the simulator (evaluate, watch) already work.
"""

from __future__ import annotations

import argparse

import numpy as np

from sim import RaceEnv, SimConfig, load_track

# ---------------------------------------------------------------------------
# Simulator setup (built once, shared by every car)
# ---------------------------------------------------------------------------

config = SimConfig()
config.env.include_speed_in_obs = True  # obs = 5 sensor distances + speed (6 values)
track = load_track(config.track)


def evaluate(policy) -> dict:
    """Drive one episode with policy(obs) -> (steer, throttle). Returns the final info dict.

    Useful info fields for a fitness function:
        info["progress_m"]          meters driven along the track (negative = backward)
        info["checkpoints_passed"]  gates passed in order
        info["time"]                seconds the episode lasted
        info["crashed"]             True if it hit a wall
        info["done_reason"]         "crashed" | "finished" | "timeout" | "stalled"
    """
    env = RaceEnv(config, track=track)
    obs = env.reset()
    done = False
    info: dict = {}
    while not done:
        steer, throttle = policy(obs)
        obs, done, info = env.step(steer, throttle)
    return info


def watch(policy, steps_per_frame: int = 1) -> None:
    """Open the 2D viewer and watch a policy drive (imports pygame only here)."""
    from render2d import Viewer

    Viewer(RaceEnv(config, track=track), policy=policy, steps_per_frame=steps_per_frame).run()


# ---------------------------------------------------------------------------
# YOUR CODE: neural network, fitness function, genetic algorithm
# ---------------------------------------------------------------------------


def placeholder_policy(obs: np.ndarray) -> tuple[float, float]:
    """Stand-in so this file runs. Replace with your network's output."""
    return 0.0, 0.5  # straight ahead, half throttle


def train() -> None:
    # TODO: your GA loop goes here. For now this just tests the placeholder.
    info = evaluate(placeholder_policy)
    print(
        f"placeholder policy: {info['done_reason']}, "
        f"{info['checkpoints_passed']} checkpoints, {info['progress_m']:.1f} m"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="watch the policy in the viewer")
    parser.add_argument("--speed", type=int, default=1, help="sim steps per rendered frame")
    args = parser.parse_args()

    if args.watch:
        watch(placeholder_policy, args.speed)  # TODO: watch your best policy instead
    else:
        train()
