"""Oyun dunyasi cizimi: engeller, boss, oyuncu (kep, domuz burnu), dalga, parcaciklar."""

import cv2
import math
import numpy as np

from .config import (
    CREAM,
    DARK,
    GOLD,
    HELPER_HIT_AT,
    KIND_DEVRE,
    KIND_DOMUZ,
    KIND_KALEM,
    KIND_KIMYA,
    KIND_PROJ,
    KIND_VIZE,
    MAROON,
    MAROON_LIGHT,
    STATE_OVER,
    WHITE,
    YELLOW,
)
from .drawing import blend_ellipse, blend_rect, draw_pill, draw_star, ease_out, txt
from .entities import Obstacle
from .game import Game
from .sprites import LABELS, SPRITES, boy_sprite, compose_sprite, sprite_boss


def draw_obstacle(img, ob: Obstacle, t: float, u: float):
    """Engeli animasyonlu sprite olarak cizer (domuz sallanir, vize savrulur...) + etiket."""
    w, h = int(ob.w), int(ob.h)
    sprite = SPRITES[ob.kind](w, h, t, ob.seed)
    angle, scale = 0.0, 1.0
    k = ob.kind
    if k == KIND_DOMUZ:
        angle = math.sin(t * 13 + ob.seed) * 7
        scale = 1.0 + 0.04 * math.sin(t * 26 + ob.seed)
    elif k == KIND_VIZE:
        angle = -math.cos(ob.phase) * 14             # zigzag yonune dogru egilir
    elif k == KIND_KIMYA:
        angle = math.sin(t * 4 + ob.seed) * 11       # erlen salinir
    elif k == KIND_DEVRE:
        angle = -math.cos(ob.phase) * 10
    elif k == KIND_KALEM:
        angle = -25 + math.sin(t * 5) * 12
        scale = 1.0 + 0.08 * math.sin(t * 8)
    compose_sprite(img, sprite, ob.x + ob.w / 2, ob.y + ob.h / 2, angle, scale)
    if k != KIND_PROJ:
        draw_pill(img, LABELS[k], int(ob.x + ob.w / 2), int(ob.y + ob.h + 16 * u), 0.5, u,
                  MAROON if k == KIND_KALEM else DARK, GOLD if k == KIND_KALEM else CREAM)


def draw_boss(img, game: Game, u: float):
    """MATH 255 boss'u: gezinir, nefes alir, vurulunca sarsilir, olunce solar."""
    b = game.boss
    if b is None:
        return
    look = 0.0 if game.px is None else max(-1.0, min(1.0, (game.px - b.x) / (game.W * 0.4)))
    sprite = sprite_boss(int(b.w), int(b.h), b.t, look, b.hit_flash, max(0.0, b.charge))
    jx = math.sin(b.t * 60) * b.hit_flash * 8 * u + (math.sin(b.t * 80) * 10 * u if b.state == "dying" else 0.0)
    alpha = 1.0 if b.state != "dying" else max(0.0, 1.0 - b.dying_t / 2.3)
    compose_sprite(img, sprite, b.x + jx, b.y, math.sin(b.t * 1.5) * 3, 1.0 + 0.03 * math.sin(b.t * 3), alpha)


def draw_helper(img, game: Game, u: float):
    """Kalem cocuk: oyuncudan bossa ucar, dev kalemiyle keser ve kaybolur."""
    st = game.helper_state()
    if st is None:
        return
    x, y, alpha, slash = st
    sprite = boy_sprite(max(8, int(game.H * 0.2)))
    k = ease_out(game.helper.t / HELPER_HIT_AT)
    compose_sprite(img, sprite, x, y, -25 + 25 * k + math.sin(game.helper.t * 20) * 4, 1.0, alpha)
    b = game.boss
    if slash is not None and b is not None:
        p = max(0.0, min(1.0, slash))
        x0, y0 = b.x - b.w * 0.42, b.y - b.h * 0.50
        x1, y1 = x0 + p * b.w * 0.84, y0 + p * b.h * 1.0
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), WHITE, max(2, int(10 * u)), cv2.LINE_AA)
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), GOLD, max(1, int(4 * u)), cv2.LINE_AA)


def draw_cap(img, cx, top_y, width, t):
    """Mezuniyet kepi: koyu tabla, altin puskul (t ile sallanir) ve bordo dugme."""
    hw, bh = width / 2, width * 0.15
    cap = np.array([[cx - hw * 0.55, top_y + bh * 0.6], [cx + hw * 0.55, top_y + bh * 0.6],
                    [cx + hw * 0.45, top_y + bh * 2.0], [cx - hw * 0.45, top_y + bh * 2.0]], np.int32)
    cv2.fillPoly(img, [cap], (38, 32, 30), cv2.LINE_AA)
    board = np.array([[cx - hw * 1.05, top_y], [cx, top_y - bh], [cx + hw * 1.05, top_y],
                      [cx, top_y + bh]], np.int32)
    cv2.fillPoly(img, [board], (52, 44, 42), cv2.LINE_AA)
    cv2.polylines(img, [board], True, MAROON, max(1, int(width / 40)), cv2.LINE_AA)
    sway = math.sin(t * 3.0) * hw * 0.07
    p0 = (int(cx), int(top_y))
    p1 = (int(cx + hw * 0.92), int(top_y + bh * 0.28))
    p2 = (int(cx + hw * 0.92 + sway), int(top_y + bh * 0.28 + hw * 0.55))
    cv2.line(img, p0, p1, GOLD, max(1, int(width / 45)), cv2.LINE_AA)
    cv2.line(img, p1, p2, GOLD, max(2, int(width / 35)), cv2.LINE_AA)
    cv2.ellipse(img, p2, (max(2, int(width / 28)), max(3, int(width / 16))), 0, 0, 360, GOLD, -1, cv2.LINE_AA)
    cv2.circle(img, p0, max(2, int(width / 28)), MAROON_LIGHT, -1, cv2.LINE_AA)


def draw_face_extras(img, game: Game, u: float):
    """Yuze tatli detaylar: allik, domuz burnu + kulaklar (hareketle ezilir) ve pirilti.
    Oyun bitince basin ustunde yildizlar doner."""
    t, bw, bh = game.anim_t, game.pbox_w, game.pbox_h
    nx = int(game.px)
    ny = int(game.ny if game.ny is not None else game.py)
    for sx in (-1, 1):                                        # allik yanaklar
        blend_ellipse(img, (int(nx + sx * bw * 0.30), int(ny + bh * 0.03)),
                      (int(bw * 0.10), int(bw * 0.06)), (170, 130, 255), 0.45)
    r = max(6, int(bw * 0.095))
    squash = min(0.35, abs(game.nvx) * 0.025)                 # hizli giderken yana ezilir
    snort = 0.5 + 0.5 * math.sin(t * 5)                       # nefes: burun delikleri acilip kapanir
    PINK, PINK_DARK = (185, 170, 255), (105, 90, 205)
    for sx in (-1, 1):                                        # domuz kulaklari (kasin iki ucunda)
        ex, ey = game.px + sx * bw * 0.44, game.py - bh * 0.42
        tip = (ex + sx * bw * 0.10, ey - bw * 0.15 + math.sin(t * 7 + sx) * bw * 0.012)
        ear = np.array([tip, (ex - sx * bw * 0.06, ey + bw * 0.07), (ex + sx * bw * 0.10, ey + bw * 0.11)], np.int32)
        cv2.fillPoly(img, [ear], PINK, cv2.LINE_AA)
        cv2.polylines(img, [ear], True, PINK_DARK, max(1, int(2 * u)), cv2.LINE_AA)
        mid = ear.mean(axis=0)
        cv2.fillPoly(img, [(mid + (ear - mid) * 0.5).astype(np.int32)], (150, 130, 240), cv2.LINE_AA)
    rx, ry = int(r * 1.35 * (1 + squash)), int(r * (1 - squash * 0.7))
    cv2.ellipse(img, (nx, ny), (rx + 2, ry + 2), 0, 0, 360, PINK_DARK, -1, cv2.LINE_AA)       # kenar
    cv2.ellipse(img, (nx, ny), (rx, ry), 0, 0, 360, PINK, -1, cv2.LINE_AA)                    # domuz burnu
    cv2.ellipse(img, (nx - rx // 3, ny - ry // 3), (max(2, rx // 4), max(1, ry // 6)), -20, 0, 360,
                (235, 225, 255), -1, cv2.LINE_AA)                                              # parlama
    for sx in (-1, 1):                                        # burun delikleri
        cv2.ellipse(img, (nx + sx * int(rx * 0.38), ny + int(ry * 0.05)),
                    (max(2, int(rx * 0.16 * (1 + 0.25 * snort))), max(3, int(ry * 0.42 * (1 + 0.2 * snort)))),
                    0, 0, 360, (70, 50, 140), -1, cv2.LINE_AA)
    for k in range(2):                                        # burnun etrafinda parlayan yildizlar
        a = t * 2.2 + k * math.pi
        tw = 0.5 + 0.5 * math.sin(t * 6 + k * 2)
        draw_star(img, nx + math.cos(a) * r * 2.2, ny + math.sin(a) * r * 1.8, r * (0.35 + 0.35 * tw),
                  GOLD, max(1, int(2 * u)))
    if game.state == STATE_OVER:                              # carpinca basin ustunde donen yildizlar
        top = ny - bh * 0.62
        for k in range(3):
            a = t * 5 + k * 2.1
            draw_star(img, nx + math.cos(a) * bw * 0.42, top + math.sin(a) * bh * 0.07, 7 * u,
                      YELLOW, max(2, int(3 * u)))


def draw_player(img, game: Game, u: float):
    """Oyuncu: mezuniyet kepi, kose isaretleri, nabiz atan burun halkasi, isim etiketi."""
    box = game.player_box()
    if box is None:
        return
    t = game.anim_t
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = x2 - x1, y2 - y1
    lost = not game.face_visible
    color = WHITE if not lost else (YELLOW if int(t * 4) % 2 == 0 else WHITE)
    # Kose isaretleri
    ln = int(min(bw, bh) * 0.22)
    th = max(2, int(3 * u))
    for (cx_, cy_, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx_, cy_), (cx_ + dx * ln, cy_), color, th, cv2.LINE_AA)
        cv2.line(img, (cx_, cy_), (cx_, cy_ + dy * ln), color, th, cv2.LINE_AA)
    draw_cap(img, (x1 + x2) / 2, y1 - bw * 0.16, bw * 0.95, t)
    draw_face_extras(img, game, u)
    if game.name:
        draw_pill(img, game.name, (x1 + x2) // 2, int(y2 + 18 * u), 0.5, u, MAROON, WHITE, 0.85)


def draw_waves(img, t: float, u: float):
    """Alt kenarda Ege'yi anan, yavasca dalgalanan yari saydam iki dalga."""
    H, W = img.shape[:2]
    for k, (col, alpha, amp, spd, base) in enumerate(((( 170, 110, 30), 0.38, 5, 1.6, 20),
                                                      ((210, 150, 60), 0.30, 4, -2.3, 12))):
        xs = np.arange(0, W + 8, 8)
        ys = H - base * u + np.sin(xs * 0.022 + t * spd + k) * amp * u
        poly = np.concatenate([np.stack([xs, ys], 1), [[W, H], [0, H]]]).astype(np.int32)
        y0 = max(0, int(H - (base + amp + 2) * u))
        roi = img[y0:H]
        overlay = roi.copy()
        cv2.fillPoly(overlay, [poly - np.array([0, y0], np.int32)], col, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_wind_streaks(img, t: float, u: float):
    """Ruzgar engeli varken ekrani yatay kesen seffaf rüzgar cizgileri."""
    H, W = img.shape[:2]
    for i in range(9):
        spd = 700 + 130 * (i % 4)
        x = (t * spd + i * 211) % (W + 200) - 100
        y = (i * 0.113 + 0.07) % 1.0 * H * 0.85 + 20 * u
        blend_rect(img, x, y, x + (70 + 12 * (i % 3)) * u, y + max(1, 2 * u), WHITE, 0.30)


def draw_floats(img, game: Game, u: float):
    """Engel kacinca cikan harf notlari (AA, BB...)."""
    for f in game.floats:
        k = f.life / f.max_life
        pop = 1.0 + 0.6 * max(0.0, (k - 0.85) / 0.15)         # dogarken kisa sureli buyume
        txt(img, f.text, f.x, f.y, 1.0 * pop, f.color, 3, u)


def draw_particles(img, game: Game, u: float):
    """Parcacik efektleri: omurlari boyunca kuculerek kaybolan daire/yildiz."""
    for p in game.particles:
        frac = max(0.0, p.life / p.max_life)
        r = max(1, int(p.size * u * (0.3 + 0.7 * frac)))
        pos = (int(p.x), int(p.y))
        if p.spark:
            cv2.line(img, (pos[0] - r, pos[1]), (pos[0] + r, pos[1]), p.color, max(1, r // 3 + 1), cv2.LINE_AA)
            cv2.line(img, (pos[0], pos[1] - r), (pos[0], pos[1] + r), p.color, max(1, r // 3 + 1), cv2.LINE_AA)
        else:
            cv2.circle(img, pos, r, p.color, -1, cv2.LINE_AA)
