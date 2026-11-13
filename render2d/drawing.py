"""Low-level drawing helpers for the 2D viewer.

Only READS sim state. The renderer owns the screen convention: sim y points up,
screen y points down, so every world -> screen conversion flips y.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pygame
import pygame.gfxdraw

from sim.car import CarPose
from sim.sensors import SensorReading
from sim.track import Track

Color = tuple[int, int, int]

# --- palette ---------------------------------------------------------------
GRASS = (78, 138, 58)
ASPHALT = (62, 63, 67)
EDGE_LINE = (242, 242, 242)
CURB_RED = (204, 36, 40)
CURB_WHITE = (240, 240, 240)
GATE_LINE = (205, 205, 208)
CAR_BLUE = (30, 136, 229)
GLASS = (28, 36, 48)
TIRE = (22, 22, 24)
HUD_BG = (14, 17, 23)
HUD_TEXT = (236, 239, 244)
HUD_DIM = (150, 158, 170)
ACCENT = (255, 176, 46)

CURB_WIDTH_M = 1.0
EDGE_LINE_WIDTH_M = 0.25
GATE_LINE_WIDTH_M = 0.15


def darker(c: Color, f: float = 0.6) -> Color:
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f))


def sensor_color(normalized: float, hit: bool) -> Color:
    """Green when nothing is hit, shifting through yellow to red as the hit gets closer."""
    if not hit:
        return (60, 220, 90)
    n = max(0.0, min(1.0, normalized))
    if n < 0.5:  # red -> yellow
        return (235, int(60 + 380 * n), 50)
    return (int(235 - 350 * (n - 0.5)), 230, int(50 + 40 * (n - 0.5)))


def aaline(surface: pygame.Surface, color, a, b, width: int = 1) -> None:
    """Antialiased line; falls back to a plain line on pygame builds without `width`."""
    try:
        pygame.draw.aaline(surface, color, a, b, width)
    except TypeError:
        pygame.draw.line(surface, color, a, b, width)


def aapolygon(surface: pygame.Surface, color, points: Sequence[tuple[float, float]]) -> None:
    """Filled polygon with an antialiased outline (color may include alpha)."""
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    pygame.gfxdraw.filled_polygon(surface, pts, color)
    pygame.gfxdraw.aapolygon(surface, pts, color)


# --- camera ----------------------------------------------------------------


@dataclass
class Camera:
    """North-up camera: world point (cx, cy) is drawn at the screen center."""

    cx: float
    cy: float
    ppm: float  # pixels per meter
    screen_w: int
    screen_h: int

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.cx) * self.ppm + self.screen_w / 2,
            self.screen_h / 2 - (y - self.cy) * self.ppm,  # flip: sim y up, screen y down
        )

    def to_screen_many(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=float)
        out[:, 0] = (pts[:, 0] - self.cx) * self.ppm + self.screen_w / 2
        out[:, 1] = self.screen_h / 2 - (pts[:, 1] - self.cy) * self.ppm
        return out


# --- the static track, rendered once ---------------------------------------


@dataclass
class TrackLayer:
    """Pre-rendered picture of the whole track at a fixed zoom.

    The surface's top-left pixel corresponds to world point (x0, y0)
    (x0 = smallest x, y0 = LARGEST y, because screen y is flipped).
    """

    surface: pygame.Surface
    x0: float
    y0: float
    ppm: float

    def blit(self, screen: pygame.Surface, camera: Camera) -> None:
        sx, sy = camera.to_screen(self.x0, self.y0)
        screen.blit(self.surface, (round(sx), round(sy)))


def build_track_layer(track: Track, ppm: float, margin_m: float = 6.0,
                      supersample: int = 2) -> TrackLayer:
    """Draw grass, curbs, asphalt, edge lines, gates and the start line once.

    Everything is drawn at `supersample` x resolution and then smooth-scaled
    down, which antialiases all the polygons in one go.
    """
    xmin, ymin, xmax, ymax = track.bounds
    pad = margin_m + CURB_WIDTH_M
    x0, y0 = xmin - pad, ymax + pad
    w = int(math.ceil((xmax - xmin + 2 * pad) * ppm))
    h = int(math.ceil((ymax - ymin + 2 * pad) * ppm))
    k = ppm * supersample
    big = pygame.Surface((w * supersample, h * supersample))
    big.fill(GRASS)

    def px(pts: np.ndarray) -> list[tuple[float, float]]:
        return list(zip((pts[:, 0] - x0) * k, (y0 - pts[:, 1]) * k))

    n = track.n_points
    nrm = track.normals
    left, right = track.left_edge, track.right_edge

    def strip(inner: np.ndarray, outer: np.ndarray, colors: Sequence[Color], period: int) -> None:
        """Quads between two parallel polylines, colors alternating every `period` samples."""
        a, b = px(inner), px(outer)
        for i in range(n):
            j = (i + 1) % n
            color = colors[(i // period) % len(colors)]
            pygame.draw.polygon(big, color, [a[i], a[j], b[j], b[i]])

    # 1) red/white curbs just outside both edges (2 m stripes)
    strip(left, left + nrm * CURB_WIDTH_M, (CURB_RED, CURB_WHITE), 2)
    strip(right, right - nrm * CURB_WIDTH_M, (CURB_RED, CURB_WHITE), 2)
    # 2) asphalt
    strip(left, right, (ASPHALT,), 1)
    # 3) solid white line along each edge (inside the road)
    strip(left, left - nrm * EDGE_LINE_WIDTH_M, (EDGE_LINE,), 1)
    strip(right, right + nrm * EDGE_LINE_WIDTH_M, (EDGE_LINE,), 1)

    # 4) thin lines across the road at every checkpoint
    for gate in track.checkpoints:
        if gate.is_finish:
            continue
        line = np.array([gate.left, gate.right])
        a, b = px(line)
        pygame.draw.line(big, GATE_LINE, a, b, max(1, round(GATE_LINE_WIDTH_M * k)))

    # 5) checkered start/finish line: 2 rows x 12 squares across the road
    cells_across, rows, cell_len = 12, 2, 1.0
    half_w = track.width / 2.0
    for r in range(rows):
        for c in range(cells_across):
            s_a, s_b = (r - rows / 2) * cell_len, (r - rows / 2 + 1) * cell_len
            l_a = -half_w + c * track.width / cells_across
            l_b = l_a + track.width / cells_across
            corners = []
            for s, lat in ((s_a, l_a), (s_b, l_a), (s_b, l_b), (s_a, l_b)):
                (cx, cy), (tx, ty) = track.point_at(s)
                corners.append((cx - ty * lat, cy + tx * lat))  # along the left normal
            color = (245, 245, 245) if (r + c) % 2 == 0 else (25, 25, 25)
            pygame.draw.polygon(big, color, px(np.array(corners)))

    surface = pygame.transform.smoothscale(big, (w, h))
    return TrackLayer(surface=surface, x0=x0, y0=y0, ppm=ppm)


# --- the car ---------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def _car_shapes(length: float, width: float) -> dict[str, list[tuple[float, float]]]:
    """Car outline pieces in LOCAL meters (x forward, y left), for a car of this size."""
    hl, hw = length / 2.0, width / 2.0 * 0.9  # body slightly narrower so wheels peek out
    r = min(hl, hw) * 0.5
    body = []
    for cx, cy, start in ((hl - r, hw - r, 0), (-(hl - r), hw - r, 90),
                          (-(hl - r), -(hw - r), 180), (hl - r, -(hw - r), 270)):
        for i in range(7):
            a = math.radians(start + 90 * i / 6)
            body.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    # taper the nose a little so the car reads as "front" vs "back"
    body = [(x, y * (1.0 - 0.14 * (x / hl) ** 2 if x > 0 else 1.0)) for x, y in body]
    sx, sy = length / 4.0, width / 2.0

    def scaled(pts):
        return [(x * sx, y * sy) for x, y in pts]

    windshield = scaled([(0.95, 0.60), (0.95, -0.60), (0.22, -0.76), (0.22, 0.76)])
    rear_window = scaled([(-1.05, 0.74), (-1.05, -0.74), (-1.48, -0.62), (-1.48, 0.62)])
    return {"body": body, "windshield": windshield, "rear_window": rear_window}


def _transform(pts, x: float, y: float, c: float, s: float, camera: Camera):
    """Local car coordinates -> world (rotate by heading, then translate) -> screen."""
    return [camera.to_screen(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in pts]


def draw_car(screen: pygame.Surface, camera: Camera, pose: CarPose, length: float,
             width: float, color: Color = CAR_BLUE, alpha: int = 255,
             wheelbase: float = 2.6, highlight: bool = False) -> None:
    c, s = math.cos(pose.heading), math.sin(pose.heading)
    shapes = _car_shapes(length, width)

    if highlight:  # soft halo under the best car
        sx, sy = camera.to_screen(pose.x, pose.y)
        rad = int(length * 0.75 * camera.ppm)
        pygame.gfxdraw.filled_circle(screen, int(sx), int(sy), rad, (255, 230, 120, 70))
        pygame.gfxdraw.aacircle(screen, int(sx), int(sy), rad, (255, 230, 120, 160))

    body = _transform(shapes["body"], pose.x, pose.y, c, s, camera)
    aapolygon(screen, (*color, alpha), body)
    pygame.gfxdraw.aapolygon(screen, [(int(round(px)), int(round(py))) for px, py in body],
                             (*darker(color, 0.55), alpha))

    # wheels at the four corners, on top of the body edge so their angle is visible
    wl, ww = 0.30 * length / 4.0, 0.14 * width / 2.0  # half length / half width of a tire
    wy = width / 2.0 - ww
    for wx, side in ((wheelbase / 2, 1), (wheelbase / 2, -1), (-wheelbase / 2, 1), (-wheelbase / 2, -1)):
        # front wheels turn with the steering (positive steer = right = clockwise)
        a = -pose.steer_angle if wx > 0 else 0.0
        ca, sa = math.cos(a), math.sin(a)
        tire = [(wx + lx * ca - ly * sa, side * wy + lx * sa + ly * ca)
                for lx, ly in ((wl, ww), (wl, -ww), (-wl, -ww), (-wl, ww))]
        aapolygon(screen, (*TIRE, alpha), _transform(tire, pose.x, pose.y, c, s, camera))

    aapolygon(screen, (*GLASS, alpha), _transform(shapes["windshield"], pose.x, pose.y, c, s, camera))
    aapolygon(screen, (*GLASS, alpha), _transform(shapes["rear_window"], pose.x, pose.y, c, s, camera))


# --- sensors ---------------------------------------------------------------


def draw_rays(screen: pygame.Surface, camera: Camera, readings: Sequence[SensorReading]) -> None:
    for r in readings:
        col = sensor_color(r.normalized, r.hit)
        a = camera.to_screen(*r.origin)
        b = camera.to_screen(*r.point)
        aaline(screen, col, a, b, 2)
        if r.hit:
            bx, by = int(b[0]), int(b[1])
            pygame.gfxdraw.filled_circle(screen, bx, by, 4, col)
            pygame.gfxdraw.aacircle(screen, bx, by, 4, (20, 20, 20))


def draw_gate(screen: pygame.Surface, camera: Camera, a, b, color=(255, 214, 64, 120)) -> None:
    pa, pb = camera.to_screen(*a), camera.to_screen(*b)
    aaline(screen, color[:3], pa, pb, 3)


# --- HUD widgets -----------------------------------------------------------


def draw_panel(screen: pygame.Surface, rect: pygame.Rect, alpha: int = 170,
               radius: int = 12, border: bool = True) -> None:
    """Semi-transparent dark rounded rectangle."""
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(panel, (*HUD_BG, alpha), panel.get_rect(), border_radius=radius)
    if border:
        pygame.draw.rect(panel, (255, 255, 255, 50), panel.get_rect(), 1, border_radius=radius)
    screen.blit(panel, rect.topleft)


class Fonts:
    def __init__(self) -> None:
        self.big = pygame.font.Font(None, 64)
        self.medium = pygame.font.Font(None, 30)
        self.small = pygame.font.Font(None, 22)


def draw_text(screen: pygame.Surface, font: pygame.font.Font, text: str, pos,
              color: Color = HUD_TEXT, anchor: str = "topleft") -> pygame.Rect:
    surf = font.render(text, True, color)
    rect = surf.get_rect(**{anchor: pos})
    screen.blit(surf, rect)
    return rect


def draw_sensor_bars(screen: pygame.Surface, fonts: Fonts, topleft: tuple[int, int],
                     readings: Sequence[SensorReading]) -> None:
    """Five vertical bars (left to right), height = free distance."""
    x, y = topleft
    bar_w, bar_h, gap = 30, 70, 14
    width = len(readings) * bar_w + (len(readings) - 1) * gap + 28
    draw_panel(screen, pygame.Rect(x, y, width, bar_h + 66))
    draw_text(screen, fonts.small, "SENSORS (m)", (x + 14, y + 8), HUD_DIM)
    for i, r in enumerate(readings):
        bx = x + 14 + i * (bar_w + gap)
        by = y + 28
        pygame.draw.rect(screen, (40, 44, 52), (bx, by, bar_w, bar_h), border_radius=4)
        fill = int(bar_h * r.normalized)
        if fill > 0:
            pygame.draw.rect(screen, sensor_color(r.normalized, r.hit),
                             (bx, by + bar_h - fill, bar_w, fill), border_radius=4)
        label = f"{r.distance:.0f}" if r.hit else "--"
        draw_text(screen, fonts.small, label, (bx + bar_w // 2, by + bar_h + 4), HUD_TEXT, "midtop")
        draw_text(screen, fonts.small, f"{r.angle_deg:+.0f}", (bx + bar_w // 2, by + bar_h + 20),
                  HUD_DIM, "midtop")
