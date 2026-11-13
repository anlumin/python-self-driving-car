"""Draw the generated track on top of the reference image to check the tracing.

    python tools/check_trace.py            -> writes tools/trace_overlay.png

Yellow = centerline, red = left edge, cyan = right edge, white ticks = checkpoints,
magenta dot = start line. Uses pygame only to load/save the image.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pygame  # noqa: E402

from sim.track import TRACE_IMAGE_SIZE_PX, load_track  # noqa: E402


def main(scale: int = 3) -> str:
    track = load_track()
    img = pygame.image.load(os.path.join(ROOT, "assets", "track_reference.png"))
    w, h = TRACE_IMAGE_SIZE_PX
    img = pygame.transform.scale(img, (w * scale, h * scale))

    def to_px(p) -> tuple[float, float]:
        # inverse of track.image_px_to_world, then scaled up
        x, y = p
        return x / track.meters_per_px * scale, (h - y / track.meters_per_px) * scale

    for arr, color in (
        (track.centerline, (255, 255, 0)),
        (track.left_edge, (255, 60, 60)),
        (track.right_edge, (0, 200, 255)),
    ):
        pygame.draw.lines(img, color, True, [to_px(p) for p in arr], 2)
    for gate in track.checkpoints:
        pygame.draw.line(img, (255, 255, 255), to_px(gate.left), to_px(gate.right), 1)
    pygame.draw.circle(img, (255, 0, 255), to_px(track.start_position), 6)
    out = os.path.join(ROOT, "tools", "trace_overlay.png")
    pygame.image.save(img, out)
    return out


if __name__ == "__main__":
    print(main())
