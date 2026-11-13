import math

import numpy as np
import pytest

from sim import load_track
from sim.geometry import point_segment_distance, segments_intersect, turning_radius_closed


@pytest.fixture(scope="module")
def track():
    return load_track()


def test_track_size_is_plausible(track):
    # ~1966 px of centerline at 12/35 m per px
    assert 600 < track.length < 750
    xmin, ymin, xmax, ymax = track.bounds
    assert 180 < xmax - xmin < 230
    assert 75 < ymax - ymin < 110


def test_edges_are_closed_loops(track):
    n = track.n_points
    left = track.walls[:n]
    right = track.walls[n:]
    for loop in (left, right):
        # each segment ends where the next begins, and the last returns to the first
        assert np.allclose(loop[:-1, 1], loop[1:, 0])
        assert np.allclose(loop[-1, 1], loop[0, 0])


def test_road_width_is_constant(track):
    widths = np.linalg.norm(track.left_edge - track.right_edge, axis=1)
    assert np.allclose(widths, 12.0)
    # Nothing pinches the road: every centerline point is ~6 m from the nearest wall.
    a, b = track.walls[:, 0], track.walls[:, 1]
    nearest = np.array([point_segment_distance(x, y, a, b).min() for x, y in track.centerline])
    assert nearest.min() > 5.9
    assert nearest.max() <= 6.0 + 1e-9


def test_curves_are_wider_than_half_the_road(track):
    # Otherwise the inner edge would fold over itself.
    assert turning_radius_closed(track.centerline).min() > track.width / 2 + 2.0


def test_walls_never_cross(track):
    walls = track.walls
    m = len(walls)
    i, j = np.triu_indices(m, k=2)  # skip each segment and its direct neighbour
    n = track.n_points
    # also skip the wrap-around neighbours (last <-> first of each loop)
    wrap = ((i == 0) & (j == n - 1)) | ((i == n) & (j == m - 1))
    i, j = i[~wrap], j[~wrap]
    hits = segments_intersect(walls[i, 0], walls[i, 1], walls[j, 0], walls[j, 1])
    assert not hits.any()


def test_checkpoints_are_in_order(track):
    gates = track.checkpoints
    assert [g.index for g in gates] == list(range(len(gates)))
    s = np.array([g.s for g in gates])
    assert np.all(np.diff(s) > 0)
    gaps = np.diff(np.concatenate([[0.0], s]))
    assert np.allclose(gaps, 20.0, rtol=0.05)
    assert gates[-1].is_finish and not any(g.is_finish for g in gates[:-1])
    # the finish gate sits on the start line
    assert math.dist(gates[-1].center, track.start_position) < 1e-6
    # gates span the road
    for g in gates:
        assert math.dist(g.left, g.right) == pytest.approx(track.width)


def test_start_is_on_bottom_straight_facing_right(track):
    assert abs(track.start_heading) < math.radians(2)
    ymin = track.bounds[1]
    assert track.start_position[1] - ymin == pytest.approx(6.0, abs=0.5)


def test_grid_culling_never_misses_a_nearby_wall(track):
    rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = track.bounds
    a, b = track.walls[:, 0], track.walls[:, 1]
    for _ in range(200):
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        radius = 60.0
        ax, ay, bx, by, _ex, _ey = track.walls_near(x, y, radius)
        candidates = set(zip(np.round(ax, 9), np.round(ay, 9), np.round(bx, 9), np.round(by, 9)))
        near = point_segment_distance(x, y, a, b) <= radius
        for (p, q) in zip(a[near], b[near]):
            assert (round(p[0], 9), round(p[1], 9), round(q[0], 9), round(q[1], 9)) in candidates
