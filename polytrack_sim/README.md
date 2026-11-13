# polytrack_sim — 2D top-down racing simulator

A small, fast, deterministic racing environment for training a driving AI
(e.g. a genetic algorithm + small neural network). The track is traced from
`assets/track_reference.png`. There is **no** GA, network, reward or fitness
code here on purpose — the environment just exposes everything you need.

```
sim/        pure Python + numpy simulation (no graphics imports)
render2d/   pygame viewer that only *reads* sim state
examples/   random_agent.py, handcoded_agent.py
tests/      pytest suite (sim + headless viewer checks)
tools/      check_trace.py: overlays the generated track on the reference image
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+. `pygame-ce` is only needed for the viewer; `sim/` runs with numpy alone.

## Run

| Command | What it does |
|---|---|
| `python main.py --manual` | Drive yourself: **W/↑** gas, **S/↓** brake, **A/←** **D/→** steer |
| `python main.py --demo` | Watch the hand-coded sensor-based driver |
| `python main.py --demo --cars 12` | Watch 12 cars at once (a "generation") — best car is followed and highlighted |
| `python main.py --benchmark [--steps N]` | Headless random policy, prints steps/s (does not import pygame) |
| `python examples/random_agent.py` | Minimal headless API example |
| `python examples/handcoded_agent.py [--watch]` | Hand-coded driver, headless or in the viewer |
| `python -m pytest` | Run the tests |

Add `--speed N` to the visual modes to run N sim steps per rendered frame (fast-forward).

**Viewer keys:** `R` sensor rays · `M` minimap · `C` follow / whole-track camera ·
`+`/`-` sim speed (1×…128×) · `P` pause · `Enter` restart · `Esc` quit.
After an episode ends the viewer restarts automatically after 2 s.

## Using the environment

```python
from sim import RaceEnv, SimConfig

config = SimConfig()
config.env.include_speed_in_obs = True   # obs gets a 6th value: speed / max_speed
config.car.max_speed = 35.0              # every constant lives in sim/config.py

env = RaceEnv(config, seed=0)
obs = env.reset()                        # np.ndarray, 5 (or 6) values in [0, 1]
done = False
while not done:
    steer, throttle = my_policy(obs)     # both in [-1, 1]
    obs, done, info = env.step(steer, throttle)

info["crashed"], info["checkpoints_passed"], info["laps"], info["progress_m"],
info["time"], info["speed"], info["sensors"]          # required fields
info["done_reason"]   # "crashed" | "finished" | "timeout" | "stalled" | ""
info["next_checkpoint"], info["steps"]                # extras
```

**Watching an evolved policy:**

```python
from render2d import Viewer
Viewer(RaceEnv(config), policy=best_policy, steps_per_frame=1).run()         # real time
Viewer([RaceEnv(config) for _ in pop], policy=[make_policy(g) for g in pop]).run()  # whole generation
```

For many environments, build the track once and share it:
`track = load_track(config.track); envs = [RaceEnv(config, track=track) ...]`
(`RaceEnv` already does this through a cache).

### Observation
`[n(-60°), n(-30°), n(0°), n(+30°), n(+60°)]` (+ `speed / max_speed` if
`include_speed_in_obs`). `n` = distance / max_range: **1.0 = nothing within
range, 0.0 = touching a wall.**

### Sensors
`env.car.sensors.read()` → `list[SensorReading]`, ordered left to right:
`hit`, `distance` (m, = max_range when nothing hit), `normalized`, `point`
(hit position or ray end), `origin` (front-center of the car), `angle_deg`.
Angles and range are configurable in `SensorConfig`. Intersections are exact
(analytic ray-vs-segment, see `sim/geometry.py::cast_rays`), nearest hit wins.

### Episode end (`done`)
Crash (footprint touches a wall), `laps_to_finish` laps done (default 1),
`max_episode_time` (default 120 s), or no new checkpoint for `stall_timeout`
(default 10 s). Calling `step()` after `done` is a no-op that returns the
final state.

### Checkpoints, laps and progress
- 33 gates about 20.3 m apart (the spacing is adjusted so they divide the lap evenly).
  Gate 0 is the first gate after the start; the **last gate sits on the start/finish line**,
  so crossing it completes a lap.
- A gate counts only when it is the *next* gate and the car's center crosses it
  from behind to in front (driving direction). Crossing backward never counts,
  and turning around to re-cross a gate earns nothing because it is no longer "next".
- `progress_m` is the car's position projected onto the centerline, unwrapped:
  it keeps growing across the start line (≈ 670 m per lap) and goes **negative**
  if you drive backward. Smooth and continuous, useful for fitness.

## Conventions (important for a future 3D viewer)

- Units: meters, seconds, radians. Ground plane `(x, y)`, **y points up/north**.
  A 3D viewer maps sim `y` → world `Z` (and sim `x` → world `X`, up = world `Y`).
- `heading`: 0 = facing +x, **counterclockwise positive**.
- `steer` input, `steer_angle` and sensor angles use the **driver convention:
  negative = left, positive = right** (matching "−60° … +60°, left to right").
- The 2D renderer flips y for the screen; nothing in `sim/` knows about pixels.

Everything a renderer needs is public and read-only:
`env.car.pose` (`x, y, heading, speed, steer_angle`), `env.snapshot()`
(pose + sensor readings + footprint corners + counters), and on `env.track`:
`centerline`, `left_edge`, `right_edge`, `walls`, `checkpoints`
(`left`/`right` endpoints, `center`, `tangent`), `normals`, `width`, `bounds`.
To add 3D, write `render3d/` that reads exactly these; no sim changes needed.

## Performance

Measured on this machine (Windows 11, Python 3.14, numpy 2.5), all 5 sensors on:

- `python main.py --benchmark` (random policy, resets included): **~9,300–9,700 steps/s**
  (about 155× real time at dt = 1/60 s).
- Hand-coded agent driving full laps (covers the whole track, policy included): **~9,300 steps/s**.

How: walls are 1,340 segments; a lazily built 10 m spatial grid returns only the
segments that can be within sensor range (or crash range) of the car, and the
5 rays are tested against them in one vectorized numpy expression. Car physics
uses plain `math` on scalars. To go faster for a GA: run environments in
parallel processes (`multiprocessing.Pool`), one population slice per core.

## The hand-coded agent

`examples/handcoded_agent.py`: steering = gain × (free space on the right − free
space on the left) from the rays; throttle tracks a target speed
`sqrt(2 · a · free_distance_ahead)`. It uses only the observation (5 rays +
speed). It finishes a full lap (33/33 checkpoints) in ~34.5 s, and 2-lap runs
from randomized starts all finish. The tests require ≥ 20 checkpoints.

## Assumptions and choices

- **Tracing:** 51 centerline control points were picked from the image, snapped to
  the middle of the road automatically (scan across the road to both green
  borders), then hard-coded in `sim/track.py`. Run `python tools/check_trace.py`
  to see the result drawn over the image (`tools/trace_overlay.png`).
- **Scale:** the road is 35 px wide in the image → 12 m, i.e. 0.343 m/px. The track
  is ~198 × 88 m (edges) and the lap is ~670 m.
- **Smoothing:** centripetal Catmull-Rom spline, then a light Gaussian smoothing
  (σ = 3 m) to remove hand-tracing kinks. Tightest centerline radius ≈ 13 m
  (inner edge ≈ 7 m), so the road edges never fold over.
- **Driving direction:** right along the bottom straight, up the diagonal, around
  the top-right loop, left along the lower-middle straight, through the S-section,
  left along the top straight, and down the angled left end. The start/finish
  line is on the bottom straight at image x ≈ 200 px.
- **Walls = road edges.** The white lines and curbs are drawn on/outside the edges;
  the car crashes when its 4 × 2 m rectangle touches an edge.
- **Physics:** kinematic bicycle model about the car's center. Steering is capped so
  lateral acceleration ≤ 25 m/s² (≈ 2.3° wheel angle at 40 m/s), and the wheels turn at
  ≤ 300°/s. Throttle < 0 brakes; no reverse. Linear drag 0.1/s + rolling resistance
  0.5 m/s². No tire slip or drifting.
- **Determinism:** the sim has no randomness unless you enable start jitter
  (`EnvConfig.start_lateral_jitter`, `start_heading_jitter_deg`), which uses the
  env's seeded generator. Same seed + same actions → identical trajectory.
- **Sensors:** rays start at the car's front-center. A segment lying exactly on a ray's
  line (collinear) counts as a hit at its nearest point. A tiny tolerance on the
  segment parameter guarantees rays cannot slip through the joint between two segments.
- **Viewer camera:** default view shows 60 m vertically (~107 m horizontally) on a
  1280×720 window (`Viewer(view_height_m=...)` to change). The "roughly 80 m" in the
  brief sits between those two, and a 60 m height keeps the car readable.
- **Multi-car:** cars don't collide with each other (they're independent envs).
  "Best" = largest `progress_m`. The camera, HUD, rays and minimap arrow follow it.
- The track images are pre-rendered once at 2–3× resolution and downscaled (cheap
  antialiasing). The minimap is pre-rendered once too. Only the cars, rays, HUD
  and minimap markers are drawn each frame.

## Known limitations

- No car-to-car collisions, no tire grip or drift model, no reversing (by design).
- Default pygame font (no external assets), so the HUD typography is simple.
- Fixed window size (1280×720 by default, configurable via `Viewer(window_size=...)`).
