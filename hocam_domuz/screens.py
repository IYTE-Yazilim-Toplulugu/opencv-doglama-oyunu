"""Tam ekran menuler: intro, isim, bekleme, geri sayim, oyun sonu."""

import cv2
import math
import numpy as np

from .config import (
    BLACK,
    CREAM,
    FONT,
    GAME_SUBTITLE,
    GAME_TITLE,
    GOLD,
    GREEN,
    LEADERBOARD_SHOWN,
    MAROON,
    MAROON_LIGHT,
    MAX_NAME_LEN,
    ORANGE,
    RED,
    WHITE,
)
from .drawing import blend_rect, draw_pill, ease_out, rrect, text_width, txt
from .entities import grade_letter
from .game import Game
from .hud import grade_title
from .sprites import compose_sprite, sprite_stamp
from .world import draw_cap


AMBIENT_FORMULAS = ("E=mc2", "F=ma", "dy/dx", "a2+b2=c2", "4.00", "sin x", "AA", "y''+y=0", "GANO")
AMBIENT_CODE = ("</>", "def", "{ }", "git push", "python", "OpenCV", "import", "for i in", "#include")


def draw_ambient(img, t: float, u: float, words=AMBIENT_FORMULAS, color=GOLD):
    """Yavasca yukari suzulen yazilar (menu atmosferi): formuller ya da kod parcalari."""
    H, W = img.shape[:2]
    for i, f in enumerate(words):
        x = (i * 0.117 + 0.04) * W + math.sin(t * 0.8 + i) * 14 * u
        y = H - ((t * (22 + 8 * (i % 4)) + i * 83) % (H + 60))
        cv2.putText(img, f, (int(x), int(y)), FONT, 0.6 * u, color, max(1, int(2 * u)), cv2.LINE_AA)


def _card_layer(frame, appear: float):
    """Giris animasyonu: ekran 1'den kucukse bir katmana cizilip sonra karistirilir."""
    return frame if appear >= 1.0 else frame.copy()


def _merge_layer(frame, layer, appear: float):
    if layer is not frame:
        cv2.addWeighted(layer, appear, frame, 1 - appear, 0, dst=frame)


def draw_intro_screen(frame, game: Game, u: float):
    """Acilis: IYTE Yazilim Toplulugu tanitimi (site temasi: turuncu baslik, pastel isik lekeleri,
    {/} logosu) ve 'bir tusa bas' daveti."""
    H, W = frame.shape[:2]
    cx, cy, t = W // 2, H // 2, game.anim_t
    e = ease_out(t / 0.9)
    # Arka plan: kamera karartilir; mavi/pembe/turuncu yumusak lekeler suzulur
    sw, sh = max(8, W // 8), max(8, H // 8)
    small = np.zeros((sh, sw, 3), np.uint8)
    for i, col in enumerate(((250, 160, 90), (200, 100, 240), (70, 130, 250))):
        bx = int(sw * (0.5 + 0.32 * math.sin(t * 0.45 + i * 2.1)))
        by = int(sh * (0.5 + 0.28 * math.cos(t * 0.38 + i * 1.7)))
        cv2.circle(small, (bx, by), int(sw * 0.28), col, -1, cv2.LINE_AA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), sw * 0.09), (W, H), interpolation=cv2.INTER_LINEAR)
    frame[:] = cv2.addWeighted(frame, 0.16, glow, 0.5, 0)
    draw_ambient(frame, t, u, AMBIENT_CODE, (150, 160, 200))
    layer = _card_layer(frame, e)
    # {/} logosu
    ly = cy - 95 * u
    txt(layer, "{", cx - 58 * u, ly + 40 * u, 4.2, ORANGE, 5, u, outline=False)
    txt(layer, "}", cx + 58 * u, ly + 40 * u, 4.2, ORANGE, 5, u, outline=False)
    cv2.line(layer, (int(cx + 17 * u), int(ly - 38 * u)), (int(cx - 17 * u), int(ly + 34 * u)), ORANGE,
             max(3, int(9 * u)), cv2.LINE_AA)
    pulse = 1.0 + 0.02 * math.sin(t * 3)
    txt(layer, "İYTE Yazılım Topluluğu", cx, cy + 5 * u, 1.55 * pulse, ORANGE, 4, u)
    txt(layer, "Software for Everyone", cx, cy + 40 * u, 0.75, CREAM, 1, u, outline=False)
    # daktilo: "sunar" + oyun adi
    txt(layer, "sunar", cx, cy + 82 * u, 0.6, (190, 200, 220), 1, u, outline=False)
    shown = GAME_TITLE[:int(max(0.0, t - 1.0) * 12)]
    cursor = "_" if int(t * 2.5) % 2 == 0 else " "
    txt(layer, shown + cursor, cx, cy + 128 * u, 1.5, GOLD, 4, u)
    if t > 2.4:
        k = 0.55 + 0.45 * math.sin(t * 4)
        draw_pill(layer, "Başlamak için bir tuşa bas", cx, int(cy + 195 * u), 0.8, u, MAROON,
                  tuple(int(c * (0.7 + 0.3 * k)) for c in WHITE), 0.92, GOLD, 2)
    txt(layer, "yazilimiyte.com   ·   @iyte_yazilim   ·   github.com/IYTE-Yazilim-Toplulugu", cx, H - 18 * u,
        0.45, (200, 210, 230), 1, u, outline=False)
    _merge_layer(frame, layer, e)


def draw_name_screen(frame, game: Game, u: float):
    """Acilis ekrani: mezuniyet kepli baslik karti ve isim kutusu."""
    H, W = frame.shape[:2]
    cx, cy, t = W // 2, H // 2, game.anim_t
    draw_ambient(frame, t, u)                              # karartmadan once: soluk gorunur
    blend_rect(frame, 0, 0, W, H, (20, 8, 14), 0.74)
    e = ease_out(t / 0.7)
    layer = _card_layer(frame, e)
    oy = int((1 - e) * 40 * u)
    cw, ch = int(560 * u), int(330 * u)
    x1, y1 = cx - cw // 2, cy - ch // 2 + oy
    x2, y2 = x1 + cw, y1 + ch
    draw_cap(layer, cx, y1 - 34 * u + math.sin(t * 2.5) * 6 * u, 100 * u, t)
    rrect(layer, x1, y1, x2, y2, 20 * u, (34, 26, 32), 0.93, MAROON_LIGHT, max(2, 3 * u))
    rrect(layer, x1 + 2, y1 + 2, x2 - 2, y1 + 56 * u, 18 * u, MAROON, 1.0)
    cv2.rectangle(layer, (x1 + 2, int(y1 + 30 * u)), (x2 - 2, int(y1 + 56 * u)), MAROON, -1)
    txt(layer, "İZMİR YÜKSEK TEKNOLOJİ ENSTİTÜSÜ", cx, y1 + 38 * u, 0.62, WHITE, 2, u, outline=False)
    pulse = 1 + 0.03 * math.sin(t * 3)
    fit = min(2.1, 500.0 / max(1, text_width(GAME_TITLE, 1.0, 5, 1.0)))   # uzun adlar karta sigsin
    txt(layer, GAME_TITLE, cx, y1 + 128 * u, fit * pulse, GOLD, 5, u)
    txt(layer, GAME_SUBTITLE, cx, y1 + 162 * u, min(0.62, 520.0 / max(1, text_width(GAME_SUBTITLE, 1.0, 1, 1.0))),
        CREAM, 1, u, outline=False)
    bx1, bx2, by1, by2 = cx - 200 * u, cx + 200 * u, y1 + 184 * u, y1 + 238 * u
    blink = int(t * 2) % 2 == 0
    rrect(layer, bx1, by1, bx2, by2, 14 * u, (15, 12, 16), 0.9, GOLD if blink else MAROON_LIGHT, max(2, 2.5 * u))
    if game.name or blink:
        label, col = (game.name + ("_" if blink else " ")), WHITE
    else:
        label, col = "", WHITE
    if not game.name:
        txt(layer, "Adını yaz...", cx, (by1 + by2) / 2 + 9 * u,
            0.85, (150, 150, 160), 1, u, outline=False)
    else:
        txt(layer, label, cx, (by1 + by2) / 2 + 12 * u, 1.2, col, 2, u, outline=False)
    txt(layer, "ENTER: başla      ESC: çık", cx, y1 + 272 * u, 0.55, GOLD, 1, u, outline=False)
    txt(layer, "Kafanı sağa sola oynatarak engellerden kaç!", cx, y1 + 300 * u, 0.5, CREAM, 1, u, outline=False)
    _merge_layer(frame, layer, e)


def draw_wait_screen(frame, game: Game, u: float):
    """Yuz bekleme: nabiz atan yuz kilavuzu ve mesaj."""
    H, W = frame.shape[:2]
    cx, cy, t = W // 2, H // 2, game.anim_t
    k = 0.5 + 0.5 * math.sin(t * 4)
    col = tuple(int(c * (0.55 + 0.45 * k)) for c in GOLD)
    axes = (int(95 * u), int(125 * u))
    for a in range(0, 360, 24):                                 # kesikli oval
        cv2.ellipse(frame, (cx, int(cy + 40 * u)), axes, 0, a, a + 12, col, max(2, int(3 * u)), cv2.LINE_AA)
    draw_pill(frame, "Yüzünü kameraya göster", cx, int(cy - 110 * u), 0.9, u, MAROON, WHITE, 0.9, GOLD, 2)


def draw_countdown(frame, game: Game, u: float):
    """Geri sayim: her saniye buyuk baslayip kuculen sayi + yayilan halka."""
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    frac = game.countdown - math.floor(game.countdown)
    if frac == 0:
        frac = 1.0
    scale = 1.0 + 0.9 * frac ** 2
    n = str(max(1, math.ceil(game.countdown)))
    cv2.circle(frame, (cx, cy), max(1, int((70 + 150 * (1 - frac)) * u)), GOLD, max(1, int(4 * frac * u) + 1), cv2.LINE_AA)
    txt(frame, n, cx, cy + 62 * u * scale, 4.5 * scale, GOLD, 10, u)
    draw_pill(frame, "Kafanı sağa sola oynatarak kaç!", cx, int(cy + 150 * u), 0.8, u, MAROON, WHITE, 0.9, GOLD, 2)


def draw_over_screen(frame, game: Game, u: float):
    """Oyun sonu: yukari kayarak gelen 'TRANSKRIPT' karti ve leaderboard."""
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    e = ease_out(game.over_t / 0.6)
    blend_rect(frame, 0, 0, W, H, (15, 6, 10), 0.68 * e)
    layer = _card_layer(frame, e)
    oy = int((1 - e) * 60 * u)
    bounce = 1.0 + 0.12 * max(0.0, 1.0 - game.over_t * 2.5)
    txt(layer, game.death_msg, cx, cy - 212 * u + oy, min(1.7, 760.0 / max(1, 18 * len(game.death_msg))) * bounce,
        RED, 5, u)
    txt(layer, f"{game.name}  -  Final GANO: {game.gano:.2f}", cx, cy - 168 * u + oy, 0.95, WHITE, 2, u)
    draw_pill(layer, grade_title(game.gano), cx, int(cy - 132 * u + oy), 0.62, u, MAROON, GOLD, 0.95, GOLD, 2)

    rows = [(i + 1, en) for i, en in enumerate(game.board[:LEADERBOARD_SHOWN])]
    extra = LEADERBOARD_SHOWN < game.rank <= len(game.board)
    if extra:
        rows.append((game.rank, game.board[game.rank - 1]))
    rh = 36
    top = -100
    n = len(rows)
    bottom = top + 34 + rh * n + (12 if extra else 0) + 10
    x1, x2 = cx - 270 * u, cx + 270 * u
    rrect(layer, x1, cy + top * u + oy, x2, cy + bottom * u + oy, 16 * u, (34, 26, 32), 0.9, MAROON_LIGHT, max(2, 2.5 * u))
    rrect(layer, x1 + 2, cy + top * u + oy + 2, x2 - 2, cy + (top + 32) * u + oy, 14 * u, MAROON, 1.0)
    cv2.rectangle(layer, (int(x1 + 2), int(cy + (top + 18) * u + oy)), (int(x2 - 2), int(cy + (top + 32) * u + oy)), MAROON, -1)
    txt(layer, f"TRANSKRİPT  -  İLK {LEADERBOARD_SHOWN}", cx, cy + (top + 23) * u + oy, 0.62, WHITE, 2, u, outline=False)
    medal = {1: GOLD, 2: (205, 205, 210), 3: (50, 127, 205)}
    for i, (rank, en) in enumerate(rows):
        ry = top + 34 + rh * i + (12 if (extra and i == n - 1) else 0)
        me = rank == game.rank
        if me:
            rrect(layer, x1 + 8 * u, cy + ry * u + oy + 2, x2 - 8 * u, cy + (ry + rh - 2) * u + oy, 10 * u,
                  (60, 120, 50), 0.55, GREEN, max(1, 1.5 * u))
        elif i % 2 == 0:
            rrect(layer, x1 + 8 * u, cy + ry * u + oy + 2, x2 - 8 * u, cy + (ry + rh - 2) * u + oy, 10 * u,
                  (255, 255, 255), 0.07)
        mid = cy + (ry + rh / 2) * u + oy
        cv2.circle(layer, (int(cx - 230 * u), int(mid)), int(13 * u), medal.get(rank, (90, 70, 80)), -1, cv2.LINE_AA)
        txt(layer, str(rank), cx - 230 * u, mid + 6 * u, 0.55, BLACK if rank <= 3 else WHITE, 2, u, outline=False)
        txt(layer, en["name"][:MAX_NAME_LEN], cx - 200 * u, mid + 8 * u, 0.8,
            GREEN if me else WHITE, 2, u, "l", outline=False)
        txt(layer, f"{en['gano']:.2f}", cx + 240 * u, mid + 8 * u, 0.85, GOLD if rank == 1 else (GREEN if me else WHITE),
            2, u, "r", outline=False)
    txt(layer, "R: Yeniden      N: Yeni oyuncu      Q: Çıkış", cx, cy + 250 * u + oy, 0.7, CREAM, 2, u)
    st = game.over_t - 0.55                                # harf notu damgasi karta "cakilir"
    if st > 0:
        k = ease_out(st / 0.25)
        letter = grade_letter(game.gano)
        color = GOLD if letter in ("AA", "BA", "BB") else (40, 40, 225)
        stamp = sprite_stamp(letter, max(8, int(120 * u)), color)
        compose_sprite(layer, stamp, cx + 215 * u, cy - 150 * u + oy, -14, 1.0 + 1.6 * (1 - k))
        if st < 0.12 and game.shake < 0.3:                 # damga carpinca hafif sarsinti
            game.shake = 0.35
    _merge_layer(frame, layer, e)
