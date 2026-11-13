import numpy as np

from sim import RaceEnv, SimConfig


def test_reset_gives_valid_observation():
    env = RaceEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (5,)
    assert np.all(np.isfinite(obs))
    assert np.all((obs >= 0.0) & (obs <= 1.0))

    cfg = SimConfig()
    cfg.env.include_speed_in_obs = True
    env = RaceEnv(cfg, seed=0)
    obs = env.reset()
    assert obs.shape == (6,)
    assert obs[5] == 0.0
    assert env.observation_size == 6


def test_info_contains_required_keys():
    env = RaceEnv(seed=0)
    env.reset()
    _obs, done, info = env.step(0.0, 1.0)
    assert not done
    for key in ("crashed", "checkpoints_passed", "laps", "progress_m", "time", "speed", "sensors"):
        assert key in info
    assert len(info["sensors"]) == 5
    assert info["speed"] > 0


def test_driving_into_wall_crashes():
    env = RaceEnv(seed=0)
    env.reset()
    done, info = False, {}
    for _ in range(600):
        _obs, done, info = env.step(-1.0, 1.0)  # hard left, full throttle
        if done:
            break
    assert done
    assert info["crashed"]
    assert info["done_reason"] == "crashed"
    assert info["time"] < 5.0


def test_crossing_gate_forward_counts():
    env = RaceEnv(seed=0)
    gate0 = env.track.checkpoints[0]
    env.reset(start_s=gate0.s - 5.0)
    for _ in range(120):
        _obs, done, info = env.step(0.0, 0.3)
    assert not done
    assert info["checkpoints_passed"] == 1
    assert info["next_checkpoint"] == 1


def test_crossing_gate_backward_does_not_count():
    env = RaceEnv(seed=0)
    gate0 = env.track.checkpoints[0]
    env.reset(start_s=gate0.s + 5.0, reverse=True)  # just past gate 0, facing backward
    for _ in range(120):  # drives back through gate 0 (and on toward the start line)
        _obs, done, info = env.step(0.0, 0.3)
    assert not info["crashed"]
    assert info["checkpoints_passed"] == 0
    assert info["progress_m"] < -5.0  # moving backward along the track


def test_stall_ends_episode():
    env = RaceEnv(seed=0)
    env.reset()
    done, info = False, {}
    while not done:
        _obs, done, info = env.step(0.0, 0.0)  # never moves
    assert info["done_reason"] == "stalled"
    assert abs(info["time"] - 10.0) < 0.05


def test_timeout_ends_episode():
    cfg = SimConfig()
    cfg.env.max_episode_time = 1.0
    env = RaceEnv(cfg, seed=0)
    done, info = False, {}
    while not done:
        _obs, done, info = env.step(0.0, 0.2)
    assert info["done_reason"] == "timeout"


def test_step_after_done_is_a_no_op():
    env = RaceEnv(seed=0)
    env.reset()
    done = False
    while not done:
        _obs, done, info = env.step(-1.0, 1.0)
    t = info["time"]
    _obs, done, info = env.step(0.0, 1.0)
    assert done and info["time"] == t


def test_deterministic_given_seed():
    def run(seed):
        cfg = SimConfig()
        cfg.env.start_lateral_jitter = 1.0
        env = RaceEnv(cfg, seed=seed)
        rng = np.random.default_rng(123)
        trace = []
        for _ in range(300):
            obs, done, info = env.step(rng.uniform(-1, 1), rng.uniform(-1, 1))
            trace.append((*obs, info["progress_m"], env.car.x, env.car.y))
            if done:
                env.reset()
        return np.array(trace)

    assert np.array_equal(run(5), run(5))
    assert not np.array_equal(run(5), run(6))


def test_handcoded_agent_passes_20_checkpoints():
    from handcoded_agent import HandcodedAgent, run_episode

    info = run_episode(HandcodedAgent())
    assert info["checkpoints_passed"] >= 20


def test_handcoded_agent_completes_a_lap():
    from handcoded_agent import HandcodedAgent, make_config, run_episode

    cfg = make_config()
    info = run_episode(HandcodedAgent(), cfg)
    assert info["done_reason"] == "finished"
    assert info["laps"] == 1
    assert info["checkpoints_passed"] == len(RaceEnv(cfg).track.checkpoints)
    assert abs(info["progress_m"] - RaceEnv(cfg).track.length) < 5.0
