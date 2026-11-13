"""Viewer smoke tests, run headless with SDL's dummy video driver."""

import os

import pytest

pygame = pytest.importorskip("pygame")
os.environ["SDL_VIDEODRIVER"] = "dummy"

from handcoded_agent import HandcodedAgent, make_config  # noqa: E402

from render2d.viewer import Viewer  # noqa: E402
from sim import RaceEnv  # noqa: E402


def test_minimap_arrow_stays_on_minimap_road_for_a_full_lap():
    env = RaceEnv(make_config())
    viewer = Viewer(env, policy=HandcodedAgent(), steps_per_frame=4, fps=100000)
    try:
        frames = 0
        while not env.done and frames < 2000:
            viewer.step_frame()
            frames += 1
            mx, my = viewer.minimap.to_map(env.car.x, env.car.y)
            assert viewer.minimap.is_on_road(mx, my), (frames, mx, my)
            # and the drawn pixel there is road-colored (or a marker drawn on the road),
            # never the dark panel background
            assert viewer.minimap.base.get_at((int(mx), int(my))).a > 200
        assert env.done_reason == "finished"
    finally:
        pygame.quit()


def test_minimap_keeps_aspect_ratio():
    env = RaceEnv()
    viewer = Viewer(env, policy=lambda obs: (0.0, 0.0), fps=100000)
    try:
        mm = viewer.minimap
        xmin, ymin, xmax, ymax = env.track.bounds
        a = mm.to_map(xmin, ymin)
        b = mm.to_map(xmax, ymax)
        w, h = b[0] - a[0], a[1] - b[1]
        assert w / h == pytest.approx((xmax - xmin) / (ymax - ymin))
        assert 0 <= a[0] and b[0] <= mm.size[0] and 0 <= b[1] and a[1] <= mm.size[1]
    finally:
        pygame.quit()


def test_viewer_toggles_and_multi_car_render():
    cfg = make_config()
    envs = [RaceEnv(cfg, seed=i) for i in range(3)]
    viewer = Viewer(envs, policy=[HandcodedAgent()] * 3, fps=100000)
    try:
        for key in (pygame.K_r, pygame.K_m, pygame.K_c, pygame.K_PLUS):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            viewer.step_frame()
        assert not viewer.show_rays and not viewer.show_minimap and viewer.overview
        assert viewer.steps_per_frame == 2
        for _ in range(30):
            viewer.step_frame()
        assert all(e.time > 0 for e in envs)
    finally:
        pygame.quit()
