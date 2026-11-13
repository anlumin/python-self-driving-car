"""Interactive 2D viewer: manual driving, policy replay, and multi-car viewing.

    from render2d import Viewer
    Viewer(env).run()                                   # drive with the keyboard
    Viewer(env, policy=my_policy).run()                 # watch policy(obs) -> (steer, throttle)
    Viewer(env, policy=my_policy, steps_per_frame=8).run()   # 8x fast-forward
    Viewer([env1, env2, ...], policy=[p1, p2, ...]).run()    # a whole generation

Keys: WASD / arrows drive (manual) | R rays | M minimap | C camera | +/- sim speed
      P pause | Enter restart | Esc quit

The viewer only reads simulation state (env.snapshot(), env.car.pose); all
game logic stays in sim/.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pygame

from sim.env import EnvSnapshot, RaceEnv

from .drawing import (
    CAR_BLUE,
    GRASS,
    HUD_DIM,
    HUD_TEXT,
    Camera,
    Fonts,
    build_track_layer,
    draw_car,
    draw_gate,
    draw_panel,
    draw_rays,
    draw_sensor_bars,
    draw_text,
)
from .minimap import Minimap

Policy = Callable[[np.ndarray], tuple[float, float]]

SPEED_LEVELS = (1, 2, 4, 8, 16, 32, 64, 128)
MULTI_CAR_COLORS = [
    (30, 136, 229), (229, 57, 53), (67, 160, 71), (251, 192, 45), (142, 36, 170),
    (0, 172, 193), (244, 81, 30), (124, 179, 66), (216, 27, 96), (94, 53, 177),
    (255, 143, 0), (0, 137, 123),
]


class Viewer:
    def __init__(
        self,
        env: RaceEnv | Sequence[RaceEnv],
        policy: Policy | Sequence[Policy] | None = None,
        *,
        steps_per_frame: int = 1,
        window_size: tuple[int, int] = (1280, 720),
        fps: int = 60,
        view_height_m: float = 60.0,
        auto_reset_delay: float = 2.0,
        title: str = "PolyTrack Sim",
    ) -> None:
        self.envs: list[RaceEnv] = list(env) if isinstance(env, (list, tuple)) else [env]
        if policy is None:
            if len(self.envs) != 1:
                raise ValueError("manual driving needs exactly one env")
            self.policies: list[Policy] | None = None
        elif callable(policy):
            self.policies = [policy] * len(self.envs)
        else:
            self.policies = list(policy)
            if len(self.policies) != len(self.envs):
                raise ValueError("need one policy per env")
        self.manual = self.policies is None
        self.track = self.envs[0].track

        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.width, self.height = window_size
        self.fps = fps
        self.fonts = Fonts()

        # Follow camera: view_height_m meters visible vertically (60 m -> ~107 x 60 m on 1280x720).
        self.follow_ppm = self.height / view_height_m
        self.follow_layer = build_track_layer(self.track, self.follow_ppm)
        # Overview camera: whole track fits the window.
        xmin, ymin, xmax, ymax = self.track.bounds
        self.overview_ppm = min((self.width - 80) / (xmax - xmin), (self.height - 80) / (ymax - ymin))
        self.overview_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        self.overview_layer = build_track_layer(self.track, self.overview_ppm, supersample=3)
        self.minimap = Minimap(self.track)

        self.steps_per_frame = max(1, int(steps_per_frame))
        self.auto_reset_delay = auto_reset_delay
        self.show_rays = True
        self.show_minimap = True
        self.overview = False
        self.paused = False
        self.running = True
        self.frame = 0
        self.best = 0
        self._all_done_at: float | None = None
        self._sim_clock = 0.0  # seconds of wall-clock-like time, advanced per frame

        self._obs = [e.reset() for e in self.envs]
        car = self.envs[0].car
        self.cam_x, self.cam_y = car.x, car.y

    # ------------------------------------------------------------ main loop

    def run(self, max_frames: int | None = None) -> None:
        try:
            while self.running and (max_frames is None or self.frame < max_frames):
                self.step_frame()
        finally:
            pygame.quit()

    def step_frame(self) -> None:
        """Handle input, advance the sim, draw one frame."""
        self._handle_events()
        if not self.paused:
            self._advance()
        self._update_camera()
        self._draw()
        pygame.display.flip()
        self.clock.tick(self.fps)
        self._sim_clock += 1.0 / self.fps
        self.frame += 1

    def reset(self) -> None:
        self._obs = [e.reset() for e in self.envs]
        self._all_done_at = None

    def save_screenshot(self, path: str) -> None:
        pygame.image.save(self.screen, path)

    # ------------------------------------------------------------- input

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif key == pygame.K_r:
                    self.show_rays = not self.show_rays
                elif key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                elif key == pygame.K_c:
                    self.overview = not self.overview
                elif key == pygame.K_p:
                    self.paused = not self.paused
                elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_BACKSPACE):
                    self.reset()
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.steps_per_frame = next(
                        (s for s in SPEED_LEVELS if s > self.steps_per_frame), self.steps_per_frame)
                elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.steps_per_frame = next(
                        (s for s in reversed(SPEED_LEVELS) if s < self.steps_per_frame), 1)

    def _keyboard_action(self) -> tuple[float, float]:
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        gas = keys[pygame.K_w] or keys[pygame.K_UP]
        brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
        steer = float(right) - float(left)  # positive = right
        throttle = float(gas) - float(brake)
        return steer, throttle

    # -------------------------------------------------------------- sim

    def _advance(self) -> None:
        if self._all_done_at is not None:
            if self._sim_clock - self._all_done_at >= self.auto_reset_delay:
                self.reset()
            return
        manual_action = self._keyboard_action() if self.manual else None
        for _ in range(self.steps_per_frame):
            any_running = False
            for i, env in enumerate(self.envs):
                if env.done:
                    continue
                if manual_action is not None:
                    steer, throttle = manual_action
                else:
                    steer, throttle = self.policies[i](self._obs[i])
                self._obs[i], _done, _info = env.step(steer, throttle)
                any_running = True
            if not any_running:
                break
        if all(e.done for e in self.envs):
            self._all_done_at = self._sim_clock
        # Best car = furthest along the track (camera, HUD and minimap arrow follow it).
        self.best = int(np.argmax([e.progress_m for e in self.envs]))

    def _update_camera(self) -> None:
        car = self.envs[self.best].car
        look = min(car.speed * 0.3, 10.0)  # look a bit ahead of the car
        tx = car.x + math.cos(car.heading) * look
        ty = car.y + math.sin(car.heading) * look
        self.cam_x += (tx - self.cam_x) * 0.2
        self.cam_y += (ty - self.cam_y) * 0.2

    # ------------------------------------------------------------- draw

    def _camera(self) -> Camera:
        if self.overview:
            cx, cy = self.overview_center
            return Camera(cx, cy, self.overview_ppm, self.width, self.height)
        return Camera(self.cam_x, self.cam_y, self.follow_ppm, self.width, self.height)

    def _draw(self) -> None:
        screen = self.screen
        screen.fill(GRASS)
        cam = self._camera()
        (self.overview_layer if self.overview else self.follow_layer).blit(screen, cam)

        focus = self.envs[self.best].snapshot()
        gate = self.track.checkpoints[focus.next_checkpoint]
        draw_gate(screen, cam, gate.left, gate.right)

        multi = len(self.envs) > 1
        for i, env in enumerate(self.envs):
            if i == self.best:
                continue
            color = MULTI_CAR_COLORS[i % len(MULTI_CAR_COLORS)]
            draw_car(screen, cam, env.car.pose, focus.length, focus.width, color,
                     alpha=90 if env.done else 170, wheelbase=env.config.car.wheelbase)
        best_color = MULTI_CAR_COLORS[self.best % len(MULTI_CAR_COLORS)] if multi else CAR_BLUE
        draw_car(screen, cam, focus.pose, focus.length, focus.width, best_color,
                 wheelbase=self.envs[self.best].config.car.wheelbase, highlight=multi)
        if self.show_rays:
            draw_rays(screen, cam, focus.sensors)

        self._draw_hud(focus)
        if self.show_minimap and not self.overview:  # the overview already shows everything
            self.minimap.draw(
                screen,
                (self.width - self.minimap.size[0] - 16, 16),
                [e.car.pose for e in self.envs],
                self.best,
                focus.next_checkpoint,
                focus.laps,
                dot_colors=MULTI_CAR_COLORS,
            )
        if focus.done and (not multi or self._all_done_at is not None):
            self._draw_end_banner(focus)

    def _draw_hud(self, s: EnvSnapshot) -> None:
        screen, f = self.screen, self.fonts
        env = self.envs[self.best]
        multi = len(self.envs) > 1
        rect = pygame.Rect(16, 16, 300, 196 if multi else 172)
        draw_panel(screen, rect)
        x, y = rect.x + 16, rect.y + 12
        draw_text(screen, f.small, "SPEED", (x, y), HUD_DIM)
        speed_rect = draw_text(screen, f.big, f"{s.pose.speed * 3.6:5.0f}".strip(), (x, y + 16))
        draw_text(screen, f.medium, "km/h", (speed_rect.right + 8, speed_rect.bottom - 6),
                  HUD_DIM, "bottomleft")
        rows = [
            ("LAP", f"{min(s.laps + 1, env.config.env.laps_to_finish)} / {env.config.env.laps_to_finish}"),
            ("CHECKPOINTS", f"{s.checkpoints_passed}  (next {s.next_checkpoint + 1}/{len(self.track.checkpoints)})"),
            ("TIME", f"{int(s.time // 60)}:{s.time % 60:05.2f}"),
        ]
        if multi:
            alive = sum(not e.done for e in self.envs)
            rows.append(("CAR", f"#{self.best + 1} (best)   alive {alive}/{len(self.envs)}"))
        ry = y + 70
        for label, value in rows:
            draw_text(screen, f.small, label, (x, ry), HUD_DIM)
            draw_text(screen, f.small, value, (x + 124, ry), HUD_TEXT)
            ry += 24

        draw_sensor_bars(screen, f, (16, self.height - 152), s.sensors)

        mode = "MANUAL" if self.manual else ("GENERATION" if multi else "POLICY")
        status = f"{mode}   sim x{self.steps_per_frame}   {self.clock.get_fps():.0f} fps"
        if self.paused:
            status += "   PAUSED"
        draw_text(screen, f.small, status, (self.width - 16, self.height - 34), HUD_TEXT, "bottomright")
        help_text = ("WASD/arrows drive   " if self.manual else "") + \
            "R rays   M minimap   C camera   +/- speed   P pause   Enter restart   Esc quit"
        draw_text(screen, f.small, help_text, (self.width - 16, self.height - 12), HUD_DIM, "bottomright")

    def _draw_end_banner(self, s: EnvSnapshot) -> None:
        titles = {"crashed": "CRASHED", "finished": "FINISHED!", "timeout": "TIME UP", "stalled": "STALLED"}
        rect = pygame.Rect(0, 0, 380, 110)
        rect.center = (self.width // 2, self.height // 2 - 120)
        draw_panel(self.screen, rect, alpha=200)
        draw_text(self.screen, self.fonts.big, titles.get(s.done_reason, s.done_reason.upper()),
                  (rect.centerx, rect.y + 14), HUD_TEXT, "midtop")
        draw_text(self.screen, self.fonts.small,
                  f"{s.checkpoints_passed} checkpoints  |  {s.progress_m:.0f} m  |  restarting (Enter)",
                  (rect.centerx, rect.bottom - 16), HUD_DIM, "midbottom")


def watch(policy: Policy, config=None, seed: int = 0, steps_per_frame: int = 1) -> None:
    """Convenience: replay `policy` on a fresh env in real time (or faster)."""
    Viewer(RaceEnv(config, seed=seed), policy=policy, steps_per_frame=steps_per_frame).run()
