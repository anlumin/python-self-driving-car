import math

import numpy as np
import pytest

from sim import RaceEnv
from sim.geometry import cast_rays

MAX_RANGE = 60.0


def cast(origin, angle_deg, segments, max_range=MAX_RANGE):
    """Cast one ray (world angle in degrees, CCW) against a list of ((x0,y0),(x1,y1))."""
    segs = np.array(segments, dtype=float).reshape(-1, 2, 2)
    a, b = segs[:, 0], segs[:, 1]
    ang = math.radians(angle_deg)
    dist, hit = cast_rays(
        origin[0], origin[1],
        np.array([math.cos(ang)]), np.array([math.sin(ang)]),
        a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1],
        max_range,
    )
    return float(dist[0]), bool(hit[0])


def test_ray_straight_at_wall_returns_exact_distance():
    dist, hit = cast((0.0, 0.0), 0.0, [((7.5, -1.0), (7.5, 1.0))])
    assert hit
    assert dist == pytest.approx(7.5, abs=1e-12)


def test_ray_at_angle_returns_exact_distance():
    # Wall y = 3, ray at 30 degrees: distance = 3 / sin(30deg) = 6
    dist, hit = cast((0.0, 0.0), 30.0, [((-50.0, 3.0), (50.0, 3.0))])
    assert hit
    assert dist == pytest.approx(6.0, abs=1e-12)


def test_parallel_ray_returns_no_hit():
    dist, hit = cast((0.0, 0.0), 0.0, [((0.0, 1.0), (10.0, 1.0))])
    assert not hit
    assert dist == MAX_RANGE


def test_collinear_wall_ahead_is_hit_at_its_near_end():
    dist, hit = cast((0.0, 0.0), 0.0, [((5.0, 0.0), (10.0, 0.0))])
    assert hit
    assert dist == pytest.approx(5.0)


def test_two_walls_returns_nearer_one():
    walls = [((10.0, -1.0), (10.0, 1.0)), ((5.0, -1.0), (5.0, 1.0))]
    dist, hit = cast((0.0, 0.0), 0.0, walls)
    assert hit
    assert dist == pytest.approx(5.0)


def test_ray_pointing_away_returns_no_hit():
    dist, hit = cast((0.0, 0.0), 180.0, [((5.0, -1.0), (5.0, 1.0))])
    assert not hit
    assert dist == MAX_RANGE


def test_wall_out_of_range_is_no_hit():
    dist, hit = cast((0.0, 0.0), 0.0, [((70.0, -1.0), (70.0, 1.0))])
    assert not hit
    assert dist == MAX_RANGE


def test_wall_touching_origin_is_distance_zero():
    dist, hit = cast((0.0, 0.0), 0.0, [((0.0, -1.0), (0.0, 1.0))])
    assert hit
    assert dist == pytest.approx(0.0)


def test_ray_through_joint_of_collinear_segments_hits():
    # Two segments meet exactly at (5, 0), which is where the ray hits.
    walls = [((5.0, -2.0), (5.0, 0.0)), ((5.0, 0.0), (5.0, 2.0))]
    dist, hit = cast((0.0, 0.0), 0.0, walls)
    assert hit
    assert dist == pytest.approx(5.0)


def test_ray_through_joint_of_v_shape_hits():
    walls = [((5.0, -1.0), (6.0, 0.0)), ((6.0, 0.0), (5.0, 1.0))]
    dist, hit = cast((0.0, 0.0), 0.0, walls)
    assert hit
    assert dist == pytest.approx(6.0)


def test_no_leaks_through_joints_of_closed_polygon():
    """Shoot rays exactly at every vertex of a closed, irregular polygon around
    the origin (worst case for floating point gaps), plus thousands of rays in
    all directions. Every single ray must hit."""
    rng = np.random.default_rng(1)
    n = 97
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(5.0, 20.0, n)
    verts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1) + 0.1234567
    a = verts
    b = np.roll(verts, -1, axis=0)
    origin = (0.1234567, 0.1234567)

    to_vertex = verts - np.array(origin)
    vertex_dist = np.linalg.norm(to_vertex, axis=1)
    dirs = to_vertex / vertex_dist[:, None]
    many = np.linspace(0, 2 * np.pi, 5000, endpoint=False)
    dir_x = np.concatenate([dirs[:, 0], np.cos(many)])
    dir_y = np.concatenate([dirs[:, 1], np.sin(many)])

    dist, hit = cast_rays(origin[0], origin[1], dir_x, dir_y,
                          a[:, 0], a[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1], MAX_RANGE)
    assert hit.all()
    # Rays aimed at a vertex hit it (or something nearer, never something farther).
    assert np.all(dist[:n] <= vertex_dist + 1e-9)


def test_sensors_on_real_track_match_road_geometry():
    """At the start the car is centered on a straight 12 m road: a ray at angle
    a hits the side wall at 6 / sin(a)."""
    env = RaceEnv(seed=0)
    env.reset()
    readings = env.car.sensors.read()
    assert [r.angle_deg for r in readings] == [-60.0, -30.0, 0.0, 30.0, 60.0]
    for r in readings:
        if r.angle_deg == 0.0:
            continue
        expected = 6.0 / math.sin(math.radians(abs(r.angle_deg)))
        assert r.hit
        assert r.distance == pytest.approx(expected, abs=0.05)
        assert r.normalized == pytest.approx(r.distance / MAX_RANGE)
    # Hit points lie on the correct side: -60 is to the LEFT (north on the bottom straight).
    left, right = readings[0], readings[4]
    assert left.point[1] > left.origin[1]
    assert right.point[1] < right.origin[1]


def test_no_hit_reading_point_is_ray_end():
    env = RaceEnv(seed=0)
    env.reset()
    front = env.car.sensors.read()[2]
    if not front.hit:
        ox, oy = front.origin
        assert math.hypot(front.point[0] - ox, front.point[1] - oy) == pytest.approx(MAX_RANGE)
        assert front.normalized == 1.0
