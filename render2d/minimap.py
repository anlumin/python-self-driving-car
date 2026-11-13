"""Corner minimap: whole track pre-rendered once, cars drawn on top each frame."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pygame
import pygame.gfxdraw

from sim.car import CarPose
from sim.track import Track

from .drawing import ACCENT, HUD_BG

ROAD_COLOR = (196, 199, 206)
ARROW_COLOR = (40, 230, 255)
NEXT_GATE_COLOR = (255, 226, 70)


class Minimap:
    """North-up minimap that keeps the track's true aspect ratio."""

    def __init__(self, track: Track, size: tuple[int, int] = (220, 160), padding: int = 14,
                 supersample: int = 3) -> None:
        self.track = track
        self.size = size
        w, h = size
        xmin, ymin, xmax, ymax = track.bounds
        # One uniform scale for x and y (no stretching), centered in the panel.
        self.scale = min((w - 2 * padding) / (xmax - xmin), (h - 2 * padding) / (ymax - ymin))
        self.offset_x = (w - (xmax - xmin) * self.scale) / 2 - xmin * self.scale
        self.offset_y = (h - (ymax - ymin) * self.scale) / 2 + ymax * self.scale
        self.road_px = max(3.0, track.width * self.scale)  # road drawn to scale

        self.centerline_px = self.to_map_many(track.centerline)
        self.base = self._render_base(supersample)
        # Road-only alpha mask (no decorations), used by is_on_road() and the tests.
        road = pygame.Surface((size[0] * supersample, size[1] * supersample), pygame.SRCALPHA)
        self._road_quads(road, supersample, (255, 255, 255, 255))
        self.road_mask = pygame.mask.from_surface(
            pygame.transform.smoothscale(road, size), threshold=127)
        self._font: pygame.font.Font | None = None  # created on first draw

    # ------------------------------------------------------------ mapping

    def to_map(self, x: float, y: float) -> tuple[float, float]:
        """World meters -> minimap pixel (relative to the minimap's top-left)."""
        return self.offset_x + x * self.scale, self.offset_y - y * self.scale

    def to_map_many(self, pts: np.ndarray) -> np.ndarray:
        return np.stack([self.offset_x + pts[:, 0] * self.scale,
                         self.offset_y - pts[:, 1] * self.scale], axis=1)

    def is_on_road(self, mx: float, my: float) -> bool:
        """Is minimap pixel (mx, my) on the drawn road?"""
        ix, iy = int(mx), int(my)
        if not (0 <= ix < self.size[0] and 0 <= iy < self.size[1]):
            return False
        return bool(self.road_mask.get_at((ix, iy)))

    # ------------------------------------------------------------ drawing

    def _road_quads(self, surf: pygame.Surface, k: float, color) -> None:
        """Road band, drawn to scale, as quads between the two edges (k = supersample factor)."""
        left = self.to_map_many(self.track.left_edge) * k
        right = self.to_map_many(self.track.right_edge) * k
        n = self.track.n_points
        for i in range(n):
            j = (i + 1) % n
            pygame.draw.polygon(surf, color, [left[i], left[j], right[j], right[i]])

    def _render_base(self, ss: int) -> pygame.Surface:
        w, h = self.size
        big = pygame.Surface((w * ss, h * ss), pygame.SRCALPHA)
        pygame.draw.rect(big, (*HUD_BG, 185), big.get_rect(), border_radius=14 * ss)
        pygame.draw.rect(big, (255, 255, 255, 70), big.get_rect(), ss, border_radius=14 * ss)
        self._road_quads(big, ss, (*ROAD_COLOR, 255))
        # start/finish marker: a short checkered-looking bar across the road
        gate = self.track.checkpoints[-1]
        a = np.array(self.to_map(*gate.left)) * ss
        b = np.array(self.to_map(*gate.right)) * ss
        ext = (b - a) * 0.35
        pygame.draw.line(big, (20, 20, 20), a - ext, b + ext, 3 * ss)
        pygame.draw.line(big, (255, 255, 255), a - ext, (a + b) / 2, 3 * ss)
        return pygame.transform.smoothscale(big, (w, h))

    def draw(
        self,
        screen: pygame.Surface,
        topleft: tuple[int, int],
        cars: Sequence[CarPose],
        best_index: int,
        next_checkpoint: int,
        laps: int,
        dot_colors: Sequence[tuple[int, int, int]] | None = None,
    ) -> None:
        """Draw the cached base, the completed part of the lap, the next gate and the cars.

        cars: poses of all cars; the one at best_index gets the heading arrow
        (with a single car that is simply the car).
        """
        ox, oy = topleft
        screen.blit(self.base, topleft)

        # Completed part of the current lap = everything up to the last gate passed.
        gates = self.track.checkpoints
        if next_checkpoint > 0:
            done_s = gates[next_checkpoint - 1].s
            upto = min(self.track.n_points, int(done_s / self.track.spacing))
            pts = (self.centerline_px[: upto + 1] + (ox, oy)).tolist()
            if len(pts) >= 2:
                pygame.draw.lines(screen, ACCENT, False, pts, max(2, int(self.road_px * 0.55)))

        # Next checkpoint: a small marker across the road.
        gate = gates[next_checkpoint]
        a = np.array(self.to_map(*gate.left)) + (ox, oy)
        b = np.array(self.to_map(*gate.right)) + (ox, oy)
        pygame.draw.line(screen, NEXT_GATE_COLOR, a, b, 2)

        # Other cars: small dots. Best / own car: a bright arrow.
        for i, pose in enumerate(cars):
            if i == best_index:
                continue
            mx, my = self.to_map(pose.x, pose.y)
            color = dot_colors[i % len(dot_colors)] if dot_colors else (60, 60, 70)
            pygame.gfxdraw.filled_circle(screen, int(ox + mx), int(oy + my), 3, (15, 15, 20))
            pygame.gfxdraw.filled_circle(screen, int(ox + mx), int(oy + my), 2, color)
        if cars:
            self._draw_arrow(screen, ox, oy, cars[best_index])

        if laps > 0:
            if self._font is None:
                self._font = pygame.font.Font(None, 18)
            screen.blit(self._font.render(f"lap {laps + 1}", True, (200, 200, 200)), (ox + 10, oy + 8))

    def _draw_arrow(self, screen: pygame.Surface, ox: float, oy: float, pose: CarPose) -> None:
        mx, my = self.to_map(pose.x, pose.y)
        cx, cy = ox + mx, oy + my
        # Heading on the map: sim y is up, screen y is down -> flip the y component.
        dx, dy = math.cos(pose.heading), -math.sin(pose.heading)
        px, py = -dy, dx  # perpendicular
        size = 7.0
        pts = [
            (cx + dx * size, cy + dy * size),  # tip
            (cx - dx * size * 0.6 + px * size * 0.6, cy - dy * size * 0.6 + py * size * 0.6),
            (cx - dx * size * 0.25, cy - dy * size * 0.25),  # notch
            (cx - dx * size * 0.6 - px * size * 0.6, cy - dy * size * 0.6 - py * size * 0.6),
        ]
        ipts = [(int(round(x)), int(round(y))) for x, y in pts]
        pygame.gfxdraw.filled_polygon(screen, ipts, ARROW_COLOR)
        pygame.gfxdraw.aapolygon(screen, ipts, (10, 30, 40))
