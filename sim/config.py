"""All tunable constants for the simulation, grouped in dataclasses.

Usage:
    from sim.config import SimConfig, CarConfig
    cfg = SimConfig(car=CarConfig(max_speed=30.0))
    cfg.env.include_speed_in_obs = True

Conventions used everywhere in the sim:
    * Units are meters, seconds, radians (unless a name ends in _deg).
    * Ground plane coordinates (x, y), y points "up" (north). A 3D viewer maps sim y -> world Z.
    * Heading: 0 = facing +x, counterclockwise positive (standard math convention).
    * Steering input, steer_angle and sensor angles use the DRIVER convention:
      negative = left, positive = right (this matches the spec's "-60 ... +60, left to right").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CarConfig:
    """Arcade car physics (kinematic bicycle model)."""

    length: float = 4.0  # m, footprint used for crash detection
    width: float = 2.0  # m
    wheelbase: float = 2.6  # m, distance between front and rear axle
    max_speed: float = 40.0  # m/s (144 km/h)
    acceleration: float = 15.0  # m/s^2 at full throttle
    braking: float = 25.0  # m/s^2 at full brake (throttle = -1)
    drag: float = 0.10  # 1/s, linear air drag: decel = drag * speed
    rolling_resistance: float = 0.5  # m/s^2, constant decel while moving
    max_steer_deg: float = 35.0  # front wheel angle at low speed
    # Steering is limited at high speed so that the lateral acceleration
    # v^2 * tan(steer) / wheelbase never exceeds this value. This is what keeps
    # the car controllable: at 40 m/s the wheels can only turn ~2.3 degrees.
    max_lateral_accel: float = 25.0  # m/s^2
    steer_rate_deg_s: float = 300.0  # how fast the front wheels can turn


@dataclass
class SensorConfig:
    """The ray sensors mounted at the front-center of the car."""

    # Angles relative to the car's heading, ordered LEFT to RIGHT
    # (negative = left, positive = right).
    angles_deg: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
    max_range: float = 60.0  # m


@dataclass(frozen=True)
class TrackConfig:
    """How the track is generated from the traced reference image.

    Frozen (immutable) so it can be used as a cache key: tracks with the same
    config are built once and shared between environments.
    """

    road_width: float = 12.0  # m, constant
    centerline_spacing: float = 1.0  # m between centerline samples
    smoothing_sigma_m: float = 3.0  # Gaussian smoothing of the traced centerline
    checkpoint_spacing: float = 20.0  # m (adjusted slightly so gates are evenly spaced)
    grid_cell_size: float = 10.0  # m, spatial grid used to cull far-away walls


@dataclass
class EnvConfig:
    """Episode rules for RaceEnv."""

    dt: float = 1.0 / 60.0  # fixed timestep, s
    laps_to_finish: int = 1
    max_episode_time: float = 120.0  # s
    stall_timeout: float = 10.0  # s without passing a new checkpoint -> done
    include_speed_in_obs: bool = False  # append speed / max_speed as a 6th obs value
    start_s: float = 0.0  # where the car starts, in meters along the centerline
    # Optional random start perturbation (0 = fully deterministic start).
    start_lateral_jitter: float = 0.0  # m, uniform in [-j, j]
    start_heading_jitter_deg: float = 0.0  # deg, uniform in [-j, j]


@dataclass
class SimConfig:
    """Top-level config: one object holds everything."""

    car: CarConfig = field(default_factory=CarConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
