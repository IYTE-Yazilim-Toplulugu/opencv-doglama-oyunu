"""Ekran ustu arayuz: GANO paneli, boss can cubugu, duyuru."""

import cv2
import numpy as np

from .config import CREAM, DARK, GOLD, MAROON, MAROON_LIGHT, MAX_GANO, WHITE
from .drawing import draw_pill, ease_out, rrect, txt
from .game import Game


def draw_boss_bar(img, game: Game, u: float):
    """Ekranin ustunde boss adi ve can cubugu."""
    b = game.boss
    if b is None:
        return
    W = img.shape[1]
    cx = W // 2 + int(40 * u)
    x1, x2, y1, y2 = cx - 190 * u, cx + 190 * u, 26 * u, 40 * u
    txt(img, "MATH 255  -  DİFERANSİYEL DENKLEMLER", cx, 20 * u, 0.5, (255, 200, 255), 1, u)
    rrect(img, x1, y1, x2, y2, 7 * u, (40, 20, 50), 0.9, (220, 110, 235), max(1, 2 * u))
    shown = max(0.0, (b.hp - (b.dying_t / 0.3 if b.state == "dying" else 0.0)) / b.max_hp)
    fw = int((x2 - x1 - 4) * shown)
    if fw > 2:
        grad = np.linspace((40, 40, 190), (120, 90, 255), fw).astype(np.uint8)[None, :, :]
        img[int(y1) + 2:int(y2) - 1, int(x1) + 2:int(x1) + 2 + fw] = grad
    for q in range(1, b.max_hp):                                  # can bolumleri
        xx = int(x1 + (x2 - x1) * q / b.max_hp)
        cv2.line(img, (xx, int(y1)), (xx, int(y2)), (230, 200, 240), 1, cv2.LINE_AA)


def draw_banner(img, game: Game, u: float):
    """Orta ekranda kisa sureli buyuk duyuru (boss uyarisi, 'GEÇİLDİ!')."""
    if game.banner_t <= 0:
        return
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    el = game.banner_dur - game.banner_t
    pop = 1.0 + 0.6 * (1 - ease_out(el / 0.3))
    txt(img, game.banner, cx, cy - 10 * u, 1.5 * pop, game.banner_color, 4, u)
    txt(img, game.banner_sub, cx, cy + 34 * u, 0.75, WHITE, 2, u)


def grade_title(gano: float) -> str:
    """GANO'ya gore esprili unvan."""
    if gano >= 3.5:
        return "YÜKSEK ONUR LİSTESİ"
    if gano >= 3.0:
        return "ONUR ÖĞRENCİSİ"
    if gano >= 2.0:
        return "GEÇER NOT"
    if gano >= 1.0:
        return "BÜTÜNLEMEYE KALDIN"
    return "DERSTEN KALDIN"


def draw_hud(img, game: Game, u: float):
    """Sol ust: IYTE rozeti, animasyonlu GANO ve dereceli ilerleme cubugu. Sag ust: oyuncu."""
    W = img.shape[1]
    x1, y1, x2, y2 = int(14 * u), int(14 * u), int(354 * u), int(128 * u)
    pulse = game.hud_pulse
    rrect(img, x1, y1, x2, y2, int(14 * u), DARK, 0.62,
          GOLD if pulse > 0 else MAROON_LIGHT, max(1, int((2 + 2 * pulse) * u)))
    rrect(img, 24 * u, 24 * u, 96 * u, 56 * u, 8 * u, MAROON, 1.0, GOLD, max(1, int(1.5 * u)))
    txt(img, "İYTE", 60 * u, 48 * u, 0.7, WHITE, 2, u, outline=False)
    txt(img, "GANO", 108 * u, 40 * u, 0.55, GOLD, 1, u, "l")
    xe = txt(img, f"{game.disp_gano:.2f}", 108 * u, 78 * u, 1.35, WHITE, 3, u, "l")
    txt(img, f"/ {MAX_GANO:.2f}", xe + 8 * u, 78 * u, 0.55, CREAM, 1, u, "l")
    # Ilerleme cubugu (bordo -> altin gradyan)
    bx1, bx2, by1, by2 = int(24 * u), int(344 * u), int(90 * u), int(102 * u)
    rrect(img, bx1, by1, bx2, by2, int(6 * u), (50, 40, 45), 1.0)
    fill_w = int((bx2 - bx1 - 4) * min(1.0, game.disp_gano / MAX_GANO))
    if fill_w > 2:
        grad = np.linspace(MAROON_LIGHT, GOLD, fill_w).astype(np.uint8)[None, :, :]
        img[by1 + 2:by2 - 1, bx1 + 2:bx1 + 2 + fill_w] = grad
    for q in (1, 2, 3):                                        # 1.00 / 2.00 / 3.00 isaretleri
        xx = bx1 + int((bx2 - bx1) * q / MAX_GANO)
        cv2.line(img, (xx, by1), (xx, by2), (200, 200, 200), 1, cv2.LINE_AA)
    txt(img, f"Kaçılan: {game.dodged}   En iyi: {max(game.best_gano, game.gano):.2f}",
        24 * u, 120 * u, 0.5, CREAM, 1, u, "l", outline=False)
    if game.name:
        draw_pill(img, game.name, W - int(80 * u), int(30 * u), 0.55, u, MAROON, WHITE, 0.85, GOLD)
