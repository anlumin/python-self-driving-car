"""Pure-numpy 2D geometry helpers used by the simulation.

Everything works in the ground plane (x, y), in meters. No graphics imports.

The 2D cross product is used a lot below:
    cross(a, b) = a.x * b.y - a.y * b.x
It is positive when b is counterclockwise from a, negative when clockwise and
zero when a and b are parallel.
"""

from __future__ import annotations

import numpy as np

# A ray and a segment whose direction cross product is smaller than this are
# treated as parallel. Ray directions are unit vectors and wall segments are
# ~1 m long, so this is a near-zero angle.
PARALLEL_EPS = 1e-12
# Tolerance on the segment parameter u (0..1). Accepting u slightly outside
# [0, 1] makes sure a ray passing exactly through the shared joint of two
# consecutive segments is caught by at least one of them (no leaks through
# floating point "gaps").
SEGMENT_U_EPS = 1e-9
# A point closer than this to the ray's line counts as lying on it (collinear).
COLLINEAR_EPS = 1e-9


# ---------------------------------------------------------------------------
# Polylines and splines
# ---------------------------------------------------------------------------


def catmull_rom_closed(
    points: np.ndarray, samples_per_segment: int = 20, alpha: float = 0.5
) -> np.ndarray:
    """Evaluate a closed centripetal Catmull-Rom spline through `points`.

    The curve passes through every control point. alpha=0.5 ("centripetal")
    spaces the spline's internal knots by sqrt(distance), which avoids the
    loops and cusps the uniform variant can make on unevenly spaced points.
    Implemented with the Barry-Goldman pyramid of linear interpolations.
    Returns an (n * samples_per_segment, 2) array (closed, last != first).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        t = t1 + (t2 - t1) * u  # parameter values between p1 and p2
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def closed_polyline_length(points: np.ndarray) -> float:
    closed = np.vstack([points, points[:1]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed polyline to (almost exactly) uniform arc-length spacing.

    The number of samples is rounded so that the loop closes evenly; the real
    spacing is length / n, which is within a fraction of a percent of `spacing`.
    """
    closed = np.vstack([points, points[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # arc length at each vertex
    length = cum[-1]
    n = max(3, int(round(length / spacing)))
    s = np.arange(n) * (length / n)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def smooth_closed(points: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Gaussian-smooth a closed polyline (wraps around the ends).

    Straight parts stay straight; sharp kinks get rounded. sigma is measured in
    samples (with 1 m spacing, samples == meters).
    """
    if sigma_samples <= 0:
        return points.copy()
    half = int(np.ceil(3 * sigma_samples))
    k = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (k / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.vstack([points[-half:], points, points[:half]])  # wrap-around padding
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([x, y], axis=1)


def tangents_and_normals_closed(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit tangents (central differences) and LEFT normals of a closed polyline.

    The left normal is the tangent rotated 90 degrees counterclockwise:
    (tx, ty) -> (-ty, tx).
    """
    t = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return t, n


def turning_radius_closed(points: np.ndarray) -> np.ndarray:
    """Approximate radius of curvature at each vertex of a closed polyline.

    radius = (segment length) / (change of direction angle in radians).
    Straight parts give a huge radius.
    """
    seg = np.roll(points, -1, axis=0) - points
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.angle(np.exp(1j * (ang - np.roll(ang, 1))))  # wrapped to (-pi, pi]
    length = 0.5 * (np.linalg.norm(seg, axis=1) + np.linalg.norm(np.roll(seg, 1, axis=0), axis=1))
    return length / np.maximum(np.abs(dang), 1e-12)


def point_segment_distance(px: float, py: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from point p to each segment a[i] -> b[i]. a, b: (S, 2)."""
    e = b - a
    w = np.array([px, py]) - a
    # Parameter of the closest point on the infinite line, clamped onto the segment.
    len2 = np.maximum((e * e).sum(axis=1), 1e-24)
    u = np.clip((w * e).sum(axis=1) / len2, 0.0, 1.0)
    closest = a + e * u[:, None]
    return np.hypot(px - closest[:, 0], py - closest[:, 1])


# ---------------------------------------------------------------------------
# Ray casting (the sensors)
# ---------------------------------------------------------------------------


def cast_rays(
    ox: float,
    oy: float,
    dir_x: np.ndarray,
    dir_y: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest intersection of several rays with several line segments.

    Rays:      R(t) = O + t * d,    t >= 0      (d is a UNIT vector, so t is meters)
    Segments:  S(u) = A + u * e,    0 <= u <= 1  (e = B - A)

    Inputs: origin (ox, oy); ray directions dir_x, dir_y with shape (R,);
    segment starts ax, ay and segment vectors ex, ey with shape (S,).

    Returns (distance, hit), both shape (R,). distance == max_range when the
    ray hits nothing within range.

    The math: setting R(t) = S(u) gives  t*d - u*e = A - O =: w.
    Take the 2D cross product of both sides with e. Because e x e = 0 the u term
    vanishes:     t * (d x e) = w x e    ->   t = (w x e) / (d x e)
    Do the same with d (d x d = 0):
                 -u * (e x d) = w x d    ->   u = (w x d) / (d x e)
    (using e x d = -(d x e)). If d x e == 0 the ray and segment are parallel and
    there is no single crossing point.

    A hit is valid when t >= 0 (in front of the origin, not behind it) and
    0 <= u <= 1 (inside the segment, not on its infinite extension).
    """
    if len(ax) == 0:
        return np.full(len(dir_x), float(max_range)), np.zeros(len(dir_x), dtype=bool)

    wx = ax - ox  # w = A - O, shape (S,)
    wy = ay - oy
    dx = dir_x[:, None]  # shape (R, 1) so it broadcasts against (S,) -> (R, S)
    dy = dir_y[:, None]

    denom = dx * ey - dy * ex  # d x e, shape (R, S)
    num_t = wx * ey - wy * ex  # w x e, shape (S,)  (does not depend on the ray)
    num_u = wx * dy - wy * dx  # w x d, shape (R, S)

    # Instead of dividing right away (which warns when denom == 0 for parallel
    # rays), make the denominator positive and compare scaled numerators:
    #   t >= 0      <=>  num_t * sign(denom) >= 0
    #   0 <= u <= 1 <=>  0 <= num_u * sign(denom) <= |denom|
    sign = np.sign(denom)
    abs_den = np.abs(denom)
    nt = num_t * sign
    nu = num_u * sign
    valid = (
        (abs_den > PARALLEL_EPS)
        & (nt >= 0.0)
        & (nu >= -SEGMENT_U_EPS * abs_den)
        & (nu <= (1.0 + SEGMENT_U_EPS) * abs_den)
    )
    t = np.where(valid, nt, np.inf) / np.where(valid, abs_den, 1.0)
    nearest = t.min(axis=1)

    # Rare edge case: a segment lying exactly ON the ray's line (collinear).
    # It is "parallel" above, but the ray still runs into it: the hit is the
    # closest segment point that is in front of the origin.
    collinear = (abs_den <= PARALLEL_EPS) & (np.abs(num_u) <= COLLINEAR_EPS)
    if collinear.any():
        ta = wx * dx + wy * dy  # projection of A onto the ray, (R, S)
        tb = (wx + ex) * dx + (wy + ey) * dy  # projection of B
        t_near = np.maximum(np.minimum(ta, tb), 0.0)
        ahead = np.maximum(ta, tb) >= 0.0
        t_col = np.where(collinear & ahead, t_near, np.inf).min(axis=1)
        nearest = np.minimum(nearest, t_col)

    hit = nearest <= max_range
    distance = np.where(hit, nearest, float(max_range))
    return distance, hit


# ---------------------------------------------------------------------------
# Collision (car footprint vs walls)
# ---------------------------------------------------------------------------


def box_hits_segments(
    cx: float,
    cy: float,
    cos_h: float,
    sin_h: float,
    half_length: float,
    half_width: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> bool:
    """True if a rotated rectangle intersects (or touches) any segment A->B.

    Uses the Separating Axis Theorem: two convex shapes do NOT overlap exactly
    when some axis exists on which their projections don't overlap. For a box
    and a segment the only axes to check are the box's two axes and the
    segment's normal.
    """
    if len(ax) == 0:
        return False
    # Move the segment endpoints into the box's local frame
    # (box center at the origin, local x = heading direction, local y = left).
    px, py = ax - cx, ay - cy
    qx, qy = bx - cx, by - cy
    p_lx = px * cos_h + py * sin_h
    p_ly = -px * sin_h + py * cos_h
    q_lx = qx * cos_h + qy * sin_h
    q_ly = -qx * sin_h + qy * cos_h

    # Axes 1 and 2: the box spans [-half_length, half_length] x [-half_width, half_width].
    overlap_x = (np.minimum(p_lx, q_lx) <= half_length) & (np.maximum(p_lx, q_lx) >= -half_length)
    overlap_y = (np.minimum(p_ly, q_ly) <= half_width) & (np.maximum(p_ly, q_ly) >= -half_width)

    # Axis 3: the segment normal n. Every point of the segment projects onto n at
    # the same value (n . p); the box projects onto [-r, r] with
    # r = half_length*|n.x| + half_width*|n.y|.
    nx = -(q_ly - p_ly)
    ny = q_lx - p_lx
    overlap_n = np.abs(nx * p_lx + ny * p_ly) <= half_length * np.abs(nx) + half_width * np.abs(ny)

    return bool(np.any(overlap_x & overlap_y & overlap_n))


def segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Element-wise proper intersection test of segments a->b and c->d (all (K, 2)).

    Used by the tests to verify the track edges never cross themselves.
    """

    def cross(o: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])

    d1 = cross(c, d, a)
    d2 = cross(c, d, b)
    d3 = cross(a, b, c)
    d4 = cross(a, b, d)
    return (d1 * d2 < 0) & (d3 * d4 < 0)
