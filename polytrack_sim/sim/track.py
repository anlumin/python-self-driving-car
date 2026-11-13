"""The race track: centerline, road edges (walls), checkpoint gates.

The centerline was traced by hand from assets/track_reference.png (600 x 331
px) and snapped to the middle of the road in the image. The pixel points are
converted to meters (the road in the image is ~35 px wide, which we scale to
12 m) and flipped vertically (image y points down, sim y points up). A
centripetal Catmull-Rom spline through the points gives the smooth loop, which
is resampled at uniform 1 m spacing.

Everything here is pure numpy and read-only after construction, so one Track
can be shared by many environments (e.g. a whole GA population).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from .config import TrackConfig
from .geometry import (
    catmull_rom_closed,
    point_segment_distance,
    resample_closed,
    smooth_closed,
    tangents_and_normals_closed,
)

# --- Reference image tracing ------------------------------------------------

TRACE_IMAGE_SIZE_PX = (600, 331)  # width, height of assets/track_reference.png
TRACE_ROAD_WIDTH_PX = 35.0  # measured road width (green-to-green) in the image

# Centerline control points in IMAGE pixels (x right, y DOWN), in driving order.
CONTROL_POINTS_PX: list[tuple[float, float]] = [
    # long bottom straight, driving to the right
    (160.0, 270.0), (300.0, 270.0), (440.0, 270.0),
    # bottom-right corner into the diagonal
    (469.6, 266.9), (492.9, 252.2),
    # diagonal up to the top-right loop
    (523.7, 200.5), (547.5, 159.7), (567.2, 126.0),
    # top-right loop (right side, over the top, down the left side)
    (573.3, 96.6), (566.6, 75.2), (550.2, 58.8), (522.0, 51.1), (496.6, 58.9),
    (480.4, 74.4), (474.0, 95.0), (476.5, 118.7), (485.0, 138.2), (485.1, 159.6),
    # curve down into the lower-middle straight
    (475.9, 181.5), (459.1, 196.0), (435.0, 202.1),
    # lower-middle straight, driving to the left
    (400.0, 202.5), (300.0, 202.0), (240.0, 202.1),
    # S-section: left arc up ...
    (196.5, 187.6), (181.5, 152.0), (195.6, 115.6), (231.9, 101.0),
    # ... the S crossover down to the right ...
    (257.7, 108.5), (275.3, 126.9), (294.5, 144.3), (318.0, 151.4),
    # ... and the right arc back up to the top straight
    (352.4, 138.5), (369.0, 104.0), (354.4, 65.5), (320.0, 51.0),
    # long top straight, driving to the left
    (280.0, 51.0), (200.0, 51.0), (125.0, 51.0),
    # angled "hexagon" turn at the left end
    (98.4, 53.1), (78.9, 65.7), (61.6, 91.7), (45.4, 120.2), (32.4, 144.8),
    (30.0, 160.0), (33.5, 178.2), (45.5, 199.7), (63.1, 230.7), (81.3, 256.9),
    (99.9, 266.9), (125.0, 270.0),
]

# Start/finish line location in image pixels (on the bottom straight).
START_POINT_PX = (200.0, 270.0)


def image_px_to_world(px: float, py: float, meters_per_px: float) -> tuple[float, float]:
    """Image pixel -> sim meters. Flips y so that 'up' in the image is +y."""
    return px * meters_per_px, (TRACE_IMAGE_SIZE_PX[1] - py) * meters_per_px


@dataclass(frozen=True)
class Checkpoint:
    """A gate across the road. Crossing it in the driving direction counts."""

    index: int
    s: float  # arc length along the centerline, m
    center: tuple[float, float]
    left: tuple[float, float]  # gate end on the left road edge
    right: tuple[float, float]  # gate end on the right road edge
    tangent: tuple[float, float]  # unit driving direction at the gate
    is_finish: bool  # the last gate sits on the start/finish line


class Track:
    """Closed-loop track built from a TrackConfig.

    Public, read-only attributes (all numpy arrays use (x, y) in meters):
        centerline   (N, 2)  uniform spacing `spacing`, index 0 = start line
        s            (N,)    arc length of each centerline point
        tangents     (N, 2)  unit driving direction
        normals      (N, 2)  unit LEFT normal
        left_edge    (N, 2)  centerline + normal * width/2
        right_edge   (N, 2)  centerline - normal * width/2
        walls        (M, 2, 2) every wall segment [start, end]; left edge loop
                     first, then the right edge loop. Sensors hit these and
                     touching one is a crash.
        checkpoints  list[Checkpoint], ordered in driving direction
        length, width, spacing, bounds (xmin, ymin, xmax, ymax)
        start_position (x, y), start_heading (radians)
    """

    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig()
        self.width = float(self.config.road_width)
        self.meters_per_px = self.width / TRACE_ROAD_WIDTH_PX

        # 1) control points: pixels -> meters
        ctrl = np.array(
            [image_px_to_world(px, py, self.meters_per_px) for px, py in CONTROL_POINTS_PX]
        )
        # 2) smooth spline through them, then uniform resampling
        dense = catmull_rom_closed(ctrl, samples_per_segment=40)
        line = resample_closed(dense, 0.25)
        # 3) light Gaussian smoothing removes small kinks from hand tracing
        line = smooth_closed(line, self.config.smoothing_sigma_m / 0.25)
        line = resample_closed(line, self.config.centerline_spacing)

        # 4) rotate the arrays so index 0 is the start/finish line
        start = np.array(image_px_to_world(*START_POINT_PX, self.meters_per_px))
        i0 = int(np.argmin(np.linalg.norm(line - start, axis=1)))
        line = np.roll(line, -i0, axis=0)

        self.centerline: np.ndarray = line
        self.n_points = len(line)
        seg = np.roll(line, -1, axis=0) - line
        seg_len = np.linalg.norm(seg, axis=1)
        self.length = float(seg_len.sum())
        self.spacing = self.length / self.n_points
        self.s: np.ndarray = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        self.tangents, self.normals = tangents_and_normals_closed(line)

        # 5) road edges = centerline offset by half the width along the normal
        half = self.width / 2.0
        self.left_edge: np.ndarray = line + self.normals * half
        self.right_edge: np.ndarray = line - self.normals * half

        # 6) wall segments from both closed edge loops
        walls = []
        for edge in (self.left_edge, self.right_edge):
            walls.append(np.stack([edge, np.roll(edge, -1, axis=0)], axis=1))
        self.walls: np.ndarray = np.concatenate(walls, axis=0)  # (M, 2, 2)
        self._wall_a = np.ascontiguousarray(self.walls[:, 0, :])
        self._wall_b = np.ascontiguousarray(self.walls[:, 1, :])

        # 7) checkpoint gates, evenly spaced (~checkpoint_spacing apart)
        self.checkpoints: list[Checkpoint] = self._make_checkpoints()

        self.start_position = (float(line[0, 0]), float(line[0, 1]))
        self.start_heading = math.atan2(self.tangents[0, 1], self.tangents[0, 0])

        both = np.vstack([self.left_edge, self.right_edge])
        self.bounds = (
            float(both[:, 0].min()),
            float(both[:, 1].min()),
            float(both[:, 0].max()),
            float(both[:, 1].max()),
        )

        # Spatial grid cache: (radius, cell_ix, cell_iy) -> wall arrays near that cell.
        self._cell_size = float(self.config.grid_cell_size)
        self._grid_cache: dict[tuple[float, int, int], tuple[np.ndarray, ...]] = {}

    # ------------------------------------------------------------------ gates

    def _make_checkpoints(self) -> list[Checkpoint]:
        n_gates = max(2, int(round(self.length / self.config.checkpoint_spacing)))
        gap = self.length / n_gates
        half = self.width / 2.0
        gates = []
        for k in range(n_gates):
            # Gate k sits at s = (k + 1) * gap, so the LAST gate is at s = length,
            # i.e. exactly on the start/finish line (s = 0).
            s = (k + 1) * gap
            (cx, cy), (tx, ty) = self.point_at(s)
            nx, ny = -ty, tx  # left normal
            gates.append(
                Checkpoint(
                    index=k,
                    s=s,
                    center=(cx, cy),
                    left=(cx + nx * half, cy + ny * half),
                    right=(cx - nx * half, cy - ny * half),
                    tangent=(tx, ty),
                    is_finish=(k == n_gates - 1),
                )
            )
        return gates

    # ------------------------------------------------------------- queries

    def point_at(self, s: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Position and unit tangent on the centerline at arc length s (wraps)."""
        s = s % self.length
        f = s / self.spacing
        i = int(f) % self.n_points
        j = (i + 1) % self.n_points
        frac = f - int(f)
        p = self.centerline[i] * (1.0 - frac) + self.centerline[j] * frac
        t = self.centerline[j] - self.centerline[i]
        t = t / np.linalg.norm(t)
        return (float(p[0]), float(p[1])), (float(t[0]), float(t[1]))

    def nearest_index(self, x: float, y: float) -> int:
        """Index of the closest centerline point (global search, O(N))."""
        d2 = (self.centerline[:, 0] - x) ** 2 + (self.centerline[:, 1] - y) ** 2
        return int(np.argmin(d2))

    def project(self, x: float, y: float, hint: int) -> tuple[float, int]:
        """Arc length s of point (x, y) projected onto the centerline.

        Local search starting at centerline index `hint` (the previous answer),
        so the result can never jump to a different, nearby part of the track.
        Returns (s, nearest_index).
        """
        c = self.centerline
        n = self.n_points
        i = hint % n

        def d2(k: int) -> float:
            px, py = c[k % n]
            return (px - x) ** 2 + (py - y) ** 2

        # Hill-climb to the locally nearest centerline vertex.
        best = d2(i)
        while True:
            fwd, back = d2(i + 1), d2(i - 1)
            if fwd < best and fwd <= back:
                i, best = (i + 1) % n, fwd
            elif back < best:
                i, best = (i - 1) % n, back
            else:
                break

        # Refine: project onto the segment after (and before) vertex i.
        s_best, dist_best = self.s[i], best
        for a in (i, (i - 1) % n):
            b = (a + 1) % n
            ax, ay = c[a]
            ex, ey = c[b, 0] - ax, c[b, 1] - ay
            u = ((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey)
            u = min(max(u, 0.0), 1.0)
            qx, qy = ax + u * ex, ay + u * ey
            dd = (qx - x) ** 2 + (qy - y) ** 2
            if dd < dist_best:
                dist_best = dd
                s_best = self.s[a] + u * self.spacing
        return float(s_best % self.length), i

    def walls_near(self, x: float, y: float, radius: float) -> tuple[np.ndarray, ...]:
        """Wall segments that may lie within `radius` of any point in (x, y)'s grid cell.

        Returns (ax, ay, bx, by, ex, ey): segment starts, ends and vectors
        (e = b - a), as contiguous 1D arrays. Results are cached per grid cell, so
        after warm-up this costs only a dictionary lookup.

        Why this is safe: for any point P in the cell, a segment within `radius`
        of P is within radius + (half cell diagonal) of the cell center.
        """
        ix = int(math.floor(x / self._cell_size))
        iy = int(math.floor(y / self._cell_size))
        key = (radius, ix, iy)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        cx = (ix + 0.5) * self._cell_size
        cy = (iy + 0.5) * self._cell_size
        reach = radius + self._cell_size * math.sqrt(2.0) / 2.0
        d = point_segment_distance(cx, cy, self._wall_a, self._wall_b)
        idx = np.nonzero(d <= reach)[0]
        a = self._wall_a[idx]
        b = self._wall_b[idx]
        arrays = tuple(
            np.ascontiguousarray(v)
            for v in (a[:, 0], a[:, 1], b[:, 0], b[:, 1], b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
        )
        self._grid_cache[key] = arrays
        return arrays


@functools.lru_cache(maxsize=8)
def load_track(config: TrackConfig | None = None) -> Track:
    """Build (once) and return the shared Track for this config."""
    return Track(config or TrackConfig())
