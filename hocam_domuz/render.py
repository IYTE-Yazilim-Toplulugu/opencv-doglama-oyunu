"""Tum katmanlari tek kare uzerine cizen ana fonksiyon."""

import cv2
import math
import numpy as np
import random

from .config import (
    GOLD,
    KIND_RUZGAR,
    MAROON,
    STATE_COUNTDOWN,
    STATE_INTRO,
    STATE_NAME,
    STATE_OVER,
    STATE_PLAY,
    STATE_WAIT,
    YELLOW,
)
from .drawing import blend_rect, draw_pill, txt
from .game import Game
from .hud import draw_banner, draw_boss_bar, draw_hud
from .screens import draw_countdown, draw_intro_screen, draw_name_screen, draw_over_screen, draw_wait_screen
from .world import (
    draw_boss,
    draw_floats,
    draw_helper,
    draw_obstacle,
    draw_particles,
    draw_player,
    draw_waves,
    draw_wind_streaks,
)


def render(frame: np.ndarray, game: Game) -> np.ndarray:
    """Tum oyun katmanlarini kare uzerine cizer ve ayni kareyi dondurur."""
    H, W = frame.shape[:2]
    u, cx, cy, t = game.u, W // 2, H // 2, game.anim_t

    draw_waves(frame, t, u)
    if any(ob.kind == KIND_RUZGAR for ob in game.obstacles):
        draw_wind_streaks(frame, t, u)
    draw_boss(frame, game, u)
    for ob in game.obstacles:
        draw_obstacle(frame, ob, t, u)
    draw_player(frame, game, u)
    draw_particles(frame, game, u)
    draw_helper(frame, game, u)
    draw_floats(frame, game, u)
    if game.flash > 0:                                    # carpisma aninda kirmizi flas
        blend_rect(frame, 0, 0, W, H, (30, 30, 230), min(0.55, game.flash * 0.55))
    if game.state not in (STATE_INTRO, STATE_NAME, STATE_OVER):   # bu ekranlarda HUD gizlenir
        draw_hud(frame, game, u)
        draw_boss_bar(frame, game, u)
        if game.boss is not None and game.boss.state == "enter":      # boss girerken kirmizi uyari
            blend_rect(frame, 0, 0, W, H, (30, 30, 230), 0.14 * abs(math.sin(t * 6)))
        draw_banner(frame, game, u)

    if game.state == STATE_INTRO:
        draw_intro_screen(frame, game, u)
    elif game.state == STATE_NAME:
        draw_name_screen(frame, game, u)
    elif game.state == STATE_WAIT:
        draw_wait_screen(frame, game, u)
    elif game.state == STATE_COUNTDOWN:
        draw_countdown(frame, game, u)
    elif game.state == STATE_PLAY:
        if not game.face_visible:
            draw_pill(frame, "YÜZ KAYBOLDU - oyun durdu", cx, cy, 0.9, u, MAROON, YELLOW, 0.92, YELLOW, 2)
        elif game.survival < 0.9:                         # "BASLA!" yukari suzulerek kaybolur
            txt(frame, "BAŞLA!", cx, cy - 20 * u - game.survival * 50 * u, 2.2 - game.survival, GOLD, 5, u)
    elif game.state == STATE_OVER:
        draw_over_screen(frame, game, u)

    if game.shake > 0.05:                                 # ekran sarsintisi
        mag = game.shake * 14 * u
        M = np.float32([[1, 0, random.uniform(-mag, mag)], [0, 1, random.uniform(-mag, mag)]])
        frame[:] = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return frame
