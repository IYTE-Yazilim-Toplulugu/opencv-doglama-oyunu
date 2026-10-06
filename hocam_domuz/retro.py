"""8-bit arcade gorunumu: pixel dunya, avatar, HUD ve ekranlar (320x180 tuvale cizilip 3x buyutulur)."""

import math
import random

import cv2
import numpy as np

from .config import (
    KIND_KALEM, KIND_PROJ, LEADERBOARD_SHOWN, MAX_GANO, MAX_NAME_LEN, GAME_TITLE,
    STATE_COUNTDOWN, STATE_INTRO, STATE_NAME, STATE_OVER, STATE_PLAY, STATE_WAIT,
)
from .drawing import ease_out
from .game import Game
from .hud import grade_title
from .entities import grade_letter
from .pixel import (
    BAYER, C, LH, LW, SCALE, blit, box, dim, draw_text, pixel_line, pixelize, quantize, rect, shade,
    text_width, upper_tr,
)
from .retro_bg import background
from .sprites import SPRITES, _canvas, boy_sprite, sprite_boss

ORANGE_PX = C["orange"]
# Piksel ekranda dar yer oldugu icin kisa etiketler
SHORT_LABELS = {"domuz": "DOMUZ", "ruzgar": "RÜZGAR", "vize": "FİZ 101", "kimya": "KİMYA", "devre": "DEVRE",
                "termo": "TERMO", "kalem": "KALEM"}
_sprite_cache = {}
_color_cache = {}


def pal_color(bgr) -> tuple:
    """Rastgele bir BGR rengini en yakin palet rengine cevirir (onbellekli)."""
    key = tuple(int(v) for v in bgr)
    c = _color_cache.get(key)
    if c is None:
        c = tuple(int(v) for v in quantize(np.array([[key]], np.uint8))[0, 0])
        _color_cache[key] = c
    return c


def _cached(key, make):
    spr = _sprite_cache.get(key)
    if spr is None:
        if len(_sprite_cache) > 500:
            _sprite_cache.clear()
        spr = _sprite_cache[key] = make()
    return spr


# --------------------------------------------------------------------------
# AVATAR: domuz burunlu, kepli ogrenci (carpisma kutusunun merkezi = burun)
# --------------------------------------------------------------------------
SNOUT_DY = 0.17          # sprite merkezinden burun ucuna dikey uzaklik (yukseklik orani)


def sprite_avatar(w: int, h: int, step: int, blink: bool, dizzy: bool, squash: float) -> np.ndarray:
    pad = int(max(w, h) * 0.35)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + int(h * 0.55)
    ax, ay = int(w * 0.40 * (1 + squash)), int(h * 0.40 * (1 - squash * 0.6))
    pink, pink_d, skin = (185, 170, 255, 255), (105, 90, 205, 255), (150, 200, 240, 255)
    for sx in (-1, 1):                                          # domuz kulaklari
        wig = int(math.sin(step * 0.9 + sx) * h * 0.012)
        ear = np.array([[cx + sx * int(ax * 0.55), cy - int(ay * 0.75)], [cx + sx * int(ax * 1.12), cy - int(ay * 1.15) + wig],
                        [cx + sx * int(ax * 1.0), cy - int(ay * 0.2)]], np.int32)
        cv2.fillPoly(cv, [ear], pink, cv2.LINE_AA)
        cv2.polylines(cv, [ear], True, pink_d, 2, cv2.LINE_AA)
    cv2.ellipse(cv, (cx, cy), (ax, ay), 0, 0, 360, skin, -1, cv2.LINE_AA)           # bas
    ey, ex = cy - int(ay * 0.10), int(ax * 0.45)
    r = max(3, int(w * 0.06))
    for sx in (-1, 1):                                          # gozler
        if dizzy:
            cv2.line(cv, (cx + sx * ex - r, ey - r), (cx + sx * ex + r, ey + r), (30, 20, 40, 255), 3, cv2.LINE_AA)
            cv2.line(cv, (cx + sx * ex - r, ey + r), (cx + sx * ex + r, ey - r), (30, 20, 40, 255), 3, cv2.LINE_AA)
        elif blink:
            cv2.line(cv, (cx + sx * ex - r, ey), (cx + sx * ex + r, ey), (30, 20, 40, 255), 3, cv2.LINE_AA)
        else:
            cv2.circle(cv, (cx + sx * ex, ey), r, (30, 20, 40, 255), -1, cv2.LINE_AA)
            cv2.circle(cv, (cx + sx * ex - r // 3, ey - r // 3), max(1, r // 3), (255, 255, 255, 255), -1, cv2.LINE_AA)
    for sx in (-1, 1):                                          # allik
        cv2.ellipse(cv, (cx + sx * int(ax * 0.66), cy + int(ay * 0.28)), (int(ax * 0.2), int(ay * 0.13)), 0, 0, 360,
                    (170, 130, 255, 255), -1, cv2.LINE_AA)
    sy = cy + int(ay * 0.30)                                    # domuz burnu
    cv2.ellipse(cv, (cx, sy), (int(ax * 0.40), int(ay * 0.27)), 0, 0, 360, pink_d, -1, cv2.LINE_AA)
    cv2.ellipse(cv, (cx, sy), (int(ax * 0.37), int(ay * 0.24)), 0, 0, 360, pink, -1, cv2.LINE_AA)
    for sx in (-1, 1):
        cv2.ellipse(cv, (cx + sx * int(ax * 0.15), sy), (max(2, int(ax * 0.06)), max(3, int(ay * 0.12))), 0, 0, 360,
                    (70, 50, 140, 255), -1, cv2.LINE_AA)
    top = cy - int(ay * 1.0)                                    # mezuniyet kepi
    skull = np.array([[cx - int(ax * 0.7), top + int(ay * 0.15)], [cx + int(ax * 0.7), top + int(ay * 0.15)],
                      [cx + int(ax * 0.6), top + int(ay * 0.5)], [cx - int(ax * 0.6), top + int(ay * 0.5)]], np.int32)
    cv2.fillPoly(cv, [skull], (38, 32, 30, 255), cv2.LINE_AA)
    board = np.array([[cx - int(w * 0.56), top], [cx, top - int(h * 0.11)], [cx + int(w * 0.56), top],
                      [cx, top + int(h * 0.11)]], np.int32)
    cv2.fillPoly(cv, [board], (60, 52, 50, 255), cv2.LINE_AA)
    cv2.polylines(cv, [board], True, (40, 28, 125, 255), 3, cv2.LINE_AA)
    sway = int(math.sin(step * 0.8) * w * 0.04)
    p1 = (cx + int(w * 0.50), top + int(h * 0.02))
    cv2.line(cv, (cx, top), p1, (70, 190, 245, 255), 3, cv2.LINE_AA)
    cv2.line(cv, p1, (p1[0] + sway, p1[1] + int(h * 0.20)), (70, 190, 245, 255), 5, cv2.LINE_AA)
    cv2.circle(cv, (cx, top), max(3, int(w * 0.04)), (55, 45, 185, 255), -1, cv2.LINE_AA)
    return cv


def draw_avatar(cv, game: Game):
    if game.px is None:
        return
    st = int(game.anim_t * 8)
    w, h = int(game.pbox_w), int(game.pbox_h)
    blink = (st % 24) in (0, 1)
    dizzy = game.state == STATE_OVER
    squash = min(2, int(abs(game.nvx) / 5)) * 0.08
    sway_idx = st % 8
    spr = _cached(("avatar", w, h, sway_idx, blink, dizzy, squash),
                  lambda: pixelize(sprite_avatar(w, h, sway_idx, blink, dizzy, squash)))
    cx, cy = game.px / SCALE, (game.py - SNOUT_DY * h) / SCALE
    bob = 1 if (st % 8) in (2, 3) else 0
    blit(cv, spr, cx, cy - bob)
    if game.state == STATE_PLAY:                                  # carpisma cekirdegi (burun): kucuk beyaz nokta
        cv[int(game.py / SCALE), int(game.px / SCALE)] = C["white"]
    if dizzy:                                                     # basin ustunde donen yildizlar
        for k in range(3):
            a = game.anim_t * 5 + k * 2.1
            sx, sy = int(cx + math.cos(a) * 14), int(cy - h * 0.62 / SCALE + math.sin(a) * 3)
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= sx + dx < LW and 0 <= sy + dy < LH:
                    cv[sy + dy, sx + dx] = C["yellow"]


# --------------------------------------------------------------------------
# DUNYA: engeller, boss, kalem cocuk, parcaciklar
# --------------------------------------------------------------------------
def draw_obstacles(cv, game: Game):
    st = int(game.anim_t * 8)
    for ob in game.obstacles:
        w, h = int(ob.w), int(ob.h)
        key = ("ob", ob.kind, w, h, st % 8, round(ob.seed, 1) if ob.kind != KIND_PROJ else int(ob.seed * 10) % 4)
        spr = _cached(key, lambda: pixelize(SPRITES[ob.kind](w, h, (st % 8) / 8.0, ob.seed),
                                            thr=130 if ob.kind == KIND_KALEM else 96))
        bob = 1 if (st + int(ob.seed * 3)) % 2 else 0
        cx, cy = (ob.x + ob.w / 2) / SCALE, (ob.y + ob.h / 2) / SCALE + bob
        blit(cv, spr, cx, cy)
        if ob.kind != KIND_PROJ:
            draw_text(cv, SHORT_LABELS[ob.kind], cx, cy + h / (2 * SCALE) + 3, C["gold"] if ob.kind == KIND_KALEM else C["white"],
                      anchor="c", outline=True)


def draw_boss_px(cv, game: Game):
    b = game.boss
    if b is None:
        return
    st = int(b.t * 8)
    look = 0 if game.px is None else int(max(-1.0, min(1.0, (game.px - b.x) / (game.W * 0.4))) * 2)
    hit = b.hit_flash > 0 and int(b.hit_flash * 12) % 2 == 0
    charge = b.charge > 0
    w, h = int(b.w), int(b.h)
    spr = _cached(("boss", w, h, st % 16, look, hit, charge),
                  lambda: pixelize(sprite_boss(w, h, (st % 16) / 8.0, look / 2, 1.0 if hit else 0.0, 1.0 if charge else 0.0,
                                               equations=False)))
    jx = int(math.sin(b.t * 60) * b.hit_flash * 2) + (int(math.sin(b.t * 80) * 3) if b.state == "dying" else 0)
    alpha = 1.0 if b.state != "dying" else max(0.0, 1.0 - b.dying_t / 2.3)
    cx, cy = b.x / SCALE + jx, b.y / SCALE
    blit(cv, spr, cx, cy, alpha=alpha)
    if b.state != "dying":
        for i, eq in enumerate(("Y''+4Y=0", "DY/DX", "L(F(T))", "E^(ST)")):      # etrafinda donen denklemler
            a = b.t * 0.9 + i * math.pi / 2
            draw_text(cv, eq, cx + math.cos(a) * w * 0.5 / SCALE, cy + math.sin(a) * h * 0.62 / SCALE,
                      C["pink"], anchor="c", outline=True)


def draw_helper_px(cv, game: Game):
    hs = game.helper_state()
    if hs is None:
        return
    x, y, alpha, slash = hs
    boy = _cached(("boy",), lambda: pixelize(boy_sprite(216), factor=6, thr=120))
    blit(cv, boy, x / SCALE, y / SCALE, alpha=alpha)
    b = game.boss
    if slash is not None and b is not None:
        p = max(0.0, min(1.0, slash))
        x0, y0 = (b.x - b.w * 0.42) / SCALE, (b.y - b.h * 0.50) / SCALE
        x1, y1 = x0 + p * b.w * 0.84 / SCALE, y0 + p * b.h / SCALE
        pixel_line(cv, (x0, y0), (x1, y1), C["white"], 3)
        pixel_line(cv, (x0, y0), (x1, y1), C["gold"], 1)


def draw_particles_px(cv, game: Game):
    for p in game.particles:
        frac = max(0.0, p.life / p.max_life)
        x, y = int(p.x / SCALE), int(p.y / SCALE)
        col = pal_color(p.color)
        r = max(1, int(p.size * (0.3 + 0.7 * frac) / SCALE))
        if p.spark:
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < LW and 0 <= y + dy < LH:
                    cv[y + dy, x + dx] = col
        else:
            rect(cv, x, y, x + r, y + r, col)
    for f in game.floats:
        draw_text(cv, f.text, f.x / SCALE, f.y / SCALE, C["gold"] if f.color == (70, 190, 245) else C["white"],
                  anchor="c", outline=True)


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------
BAR_COLORS = [C["maroon"], C["red"], C["orange"], C["gold"]]


def draw_hud_px(cv, game: Game):
    box(cv, 3, 3, 112, 36, C["ink"], C["gold"] if game.hud_pulse > 0 else C["maroon"], shadow=False)
    draw_text(cv, "GANO", 7, 6, C["gold"], shadow=None)
    num = f"{game.disp_gano:.2f}"
    draw_text(cv, num, 7, 14, C["white"], scale=2)
    draw_text(cv, f"/{MAX_GANO:.2f}", 7 + text_width(num, 2) + 4, 21, C["gray1"], shadow=None)
    n = 25
    fill = int(round(n * min(1.0, game.disp_gano / MAX_GANO)))
    for i in range(n):                                            # 25 bloklu ilerleme cubugu
        col = BAR_COLORS[min(3, i * 4 // n)] if i < fill else C["gray3"]
        rect(cv, 7 + i * 4, 30, 7 + i * 4 + 3, 33, col)
    for q in (1, 2, 3):                                           # 1.00 / 2.00 / 3.00 isaretleri
        rect(cv, 7 + q * n // 4 * 4 - 1, 29, 7 + q * n // 4 * 4, 34, C["white"])
    draw_text(cv, f"KAÇILAN {game.dodged}  EN İYİ {max(game.best_gano, game.gano):.2f}", 5, 40, C["white"], outline=True)
    if game.name:
        nm = upper_tr(game.name)
        w = text_width(nm) + 8
        box(cv, LW - w - 4, 3, LW - 3, 16, C["maroon"], C["gold"], shadow=False)
        draw_text(cv, nm, LW - w // 2 - 3, 6, C["white"], anchor="c", shadow=None)
    b = game.boss
    if b is not None:                                             # boss can seridi (ust serit)
        rect(cv, 122, 1, 258, 13, C["ink"])
        draw_text(cv, "MATH 255", 126, 4, C["pink"], shadow=None)
        seg = 14
        shown = max(0.0, b.hp - (b.dying_t / 0.3 if b.state == "dying" else 0.0))
        for i in range(b.max_hp):
            col = C["red"] if i < shown else C["gray3"]
            rect(cv, 178 + i * (seg + 1), 4, 178 + i * (seg + 1) + seg, 10, col)


def draw_banner_px(cv, game: Game):
    if game.banner_t <= 0:
        return
    el = game.banner_dur - game.banner_t
    scale = 3 if el < 0.25 else 2
    col = pal_color(game.banner_color)
    draw_text(cv, game.banner, LW // 2, 74, col, scale=scale, anchor="c", outline=True)
    draw_text(cv, game.banner_sub, LW // 2, 74 + 8 * scale + 6, C["white"], anchor="c", outline=True)


# --------------------------------------------------------------------------
# KAMERA ONIZLEMESI (Game Boy yesili, titresimli)
# --------------------------------------------------------------------------
GB = np.array([C["ink"], C["green0"], C["green2"], C["green3"]], np.uint8)


def gb_image(cam: np.ndarray, w: int, h: int) -> np.ndarray:
    small = cv2.resize(cam, (w, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = np.clip((g - 0.15) / 0.7, 0, 1)                               # kontrast ger
    idx = np.clip(np.floor(g * 3 + BAYER[:h, :w]).astype(int), 0, 3)
    return GB[idx]


def draw_cam_pip(cv, cam, w, h, cx, cy):
    if cam is None:
        return
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    rect(cv, x1 - 1, y1 - 1, x1 + w + 1, y1 + h + 1, C["white"])
    cv[y1:y1 + h, x1:x1 + w] = gb_image(cam, w, h)


# --------------------------------------------------------------------------
# EKRANLAR
# --------------------------------------------------------------------------
LBRACE = ["...##", "..#..", "..#..", "..#..", ".#...", "#....", ".#...", "..#..", "..#..", "..#..", "...##"]
RBRACE = [r[::-1] for r in LBRACE]
SLASH = ["....#", "....#", "...#.", "...#.", "..#..", "..#..", ".#...", ".#...", "#....", "#....", "#...."]


def draw_bitmap(cv, rows, x, y, color, scale=1, shadow=True):
    m = np.array([[c == "#" for c in r] for r in rows], bool)
    m = np.kron(m, np.ones((scale, scale), bool))
    if shadow:
        _x, _y = x + scale, y + scale
        sub = cv[_y:_y + m.shape[0], _x:_x + m.shape[1]]
        sub[m[:sub.shape[0], :sub.shape[1]]] = C["ink"]
    sub = cv[y:y + m.shape[0], x:x + m.shape[1]]
    sub[m[:sub.shape[0], :sub.shape[1]]] = color


def draw_intro_px(cv, game: Game):
    t = game.anim_t
    shade(cv, 0.8)
    bob = int(round(math.sin(t * 3)))
    lx, ly = LW // 2 - 18, 10 + bob                                # {/} logosu
    draw_bitmap(cv, LBRACE, lx, ly, ORANGE_PX, 2)
    draw_bitmap(cv, SLASH, lx + 13, ly, ORANGE_PX, 2)
    draw_bitmap(cv, RBRACE, lx + 26, ly, ORANGE_PX, 2)
    draw_text(cv, "İYTE YAZILIM TOPLULUĞU", LW // 2, 38, ORANGE_PX, scale=2, anchor="c", outline=True)
    draw_text(cv, "SOFTWARE FOR EVERYONE", LW // 2, 57, C["white"], anchor="c", outline=True)
    draw_text(cv, "SUNAR", LW // 2, 72, C["gray1"], anchor="c", outline=True)
    shown = GAME_TITLE[:int(max(0.0, t - 1.0) * 12)]
    cursor = "_" if int(t * 2.5) % 2 == 0 else " "
    draw_text(cv, shown + cursor, LW // 2, 86, C["gold"], scale=3, anchor="c", grad=(C["yellow"], C["orange"]))
    if t > 2.4 and int(t * 2) % 2 == 0:
        draw_text(cv, "> BİR TUŞA BAS <", LW // 2, 124, C["white"], scale=1, anchor="c", outline=True)
    draw_text(cv, "YAZILIMIYTE.COM   @IYTE_YAZILIM", LW // 2, 168, C["gray1"], anchor="c", outline=True)


def draw_name_px(cv, game: Game):
    t = game.anim_t
    shade(cv, 0.5)
    e = ease_out(t / 0.5)
    oy = int((1 - e) * 20)
    box(cv, 52, 34 + oy, 268, 142 + oy, C["navy"], C["gold"])
    draw_text(cv, "ADINI YAZ!", LW // 2, 44 + oy, C["gold"], scale=2, anchor="c")
    draw_text(cv, "İYTE ÖĞRENCİSİ", LW // 2, 64 + oy, C["pink"], anchor="c", shadow=None)
    name = upper_tr(game.name)
    blink = int(t * 2) % 2 == 0
    x0 = LW // 2 - MAX_NAME_LEN * 14 // 2
    for i in range(MAX_NAME_LEN):                                  # harf kutulari
        x = x0 + i * 14
        active = i == len(name)
        rect(cv, x, 98 + oy, x + 12, 100 + oy, C["gold"] if (active and blink) else C["gray2"])
        if i < len(name):
            draw_text(cv, name[i], x, 82 + oy, C["white"], scale=2)
    draw_text(cv, "ENTER: BAŞLA    ESC: ÇIK", LW // 2, 118 + oy, C["gold"], anchor="c", shadow=None)
    draw_text(cv, "KAFANI SAĞA SOLA OYNATARAK KAÇ!", LW // 2, 130 + oy, C["gray1"], anchor="c", shadow=None)


def draw_wait_px(cv, game: Game, cam):
    shade(cv, 0.7)
    k = int(game.anim_t * 4) % 2
    box(cv, 60, 20, 260, 44, C["maroon"], C["gold"] if k else C["white"])
    draw_text(cv, "YÜZÜNÜ KAMERAYA GÖSTER", LW // 2, 28, C["white"], anchor="c", shadow=None)


def draw_countdown_px(cv, game: Game):
    frac = game.countdown - math.floor(game.countdown)
    frac = 1.0 if frac == 0 else frac
    n = str(max(1, math.ceil(game.countdown)))
    scale = 10 if frac > 0.8 else 8
    sh = int(random.uniform(-1, 1) * (frac > 0.9) * 2)
    draw_text(cv, n, LW // 2 + sh, 58, C["gold"], scale=scale, anchor="c", outline=True, grad=(C["yellow"], C["orange"]))
    draw_text(cv, "KAFANI SAĞA SOLA OYNAT!", LW // 2, 142, C["white"], anchor="c", outline=True)


def draw_over_px(cv, game: Game):
    e = ease_out(game.over_t / 0.6)
    if e > 0.5:
        shade(cv, 0.5)
    oy = int((1 - e) * 40)
    box(cv, 22, 6 + oy, 298, 172 + oy, C["navy"], C["maroon"])
    msg = upper_tr(game.death_msg)
    scale = 2 if text_width(msg, 2) <= 250 else 1
    flash = game.over_t < 0.6 and int(game.over_t * 14) % 2 == 0
    draw_text(cv, msg, LW // 2, 14 + oy, C["white"] if flash else C["red"], scale=scale, anchor="c", outline=True)
    draw_text(cv, f"{game.name} - GANO {game.gano:.2f}", LW // 2, 38 + oy, C["white"], anchor="c", shadow=None)
    draw_text(cv, grade_title(game.gano), LW // 2, 50 + oy, C["gold"], anchor="c", shadow=None)
    letter = grade_letter(game.gano)                               # harf notu damgasi
    st = game.over_t - 0.55
    if st > 0:
        col = C["gold"] if letter in ("AA", "BA", "BB") else C["red"]
        box(cv, 246, 33 + oy, 288, 58 + oy, C["ink"], col)
        draw_text(cv, letter, 267, 38 + oy, col, scale=3, anchor="c", shadow=None)
    rect(cv, 32, 62 + oy, 288, 72 + oy, C["maroon"])
    draw_text(cv, f"TRANSKRİPT - İLK {LEADERBOARD_SHOWN}", LW // 2, 64 + oy, C["white"], anchor="c", shadow=None)
    rows = [(i + 1, en) for i, en in enumerate(game.board[:LEADERBOARD_SHOWN])]
    if LEADERBOARD_SHOWN < game.rank <= len(game.board):
        rows.append((game.rank, game.board[game.rank - 1]))
    medal = {1: C["gold"], 2: C["gray1"], 3: C["brown2"]}
    for i, (rank, en) in enumerate(rows):
        y = 77 + i * 11 + (4 if (rank > LEADERBOARD_SHOWN) else 0) + oy
        me = rank == game.rank
        if me:
            rect(cv, 32, y - 2, 288, y + 9, C["green0"])
        col = C["green3"] if me else C["white"]
        draw_text(cv, f"{rank}.", 38, y, medal.get(rank, C["gray2"]), shadow=None)
        draw_text(cv, upper_tr(en["name"])[:MAX_NAME_LEN], 56, y, col, shadow=None)
        sc = f"{en['gano']:.2f}"
        draw_text(cv, sc, 280, y, C["gold"] if rank == 1 else col, anchor="r", shadow=None)
        if 0 <= y + 6 < LH:                                       # (acilis animasyonunda tuval disina tasabilir)
            for dx in range(56 + text_width(upper_tr(en["name"])[:MAX_NAME_LEN]) + 4, 280 - text_width(sc) - 4, 4):
                cv[y + 6, dx] = C["gray3"]
    draw_text(cv, "R: YENİDEN   N: YENİ OYUNCU   Q: ÇIKIŞ", LW // 2, 158 + oy, C["gray1"], anchor="c", shadow=None)


# --------------------------------------------------------------------------
# ANA CIZIM
# --------------------------------------------------------------------------
def render_retro(game: Game, cam=None, cam_mode: int = 0) -> np.ndarray:
    """Oyunu 320x180 piksel tuvale cizer. cam_mode: 0 kucuk onizleme, 1 gizli, 2 arka plan (Game Boy kamerasi)."""
    st = game.state
    t = game.anim_t
    playing = st in (STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER)
    if cam_mode == 2 and cam is not None and playing:
        cv = gb_image(cam, LW, LH)
    else:
        cv = background(t)
        if st in (STATE_COUNTDOWN, STATE_PLAY):
            shade(cv, 0.8)                                          # oyun nesneleri daha okunakli
    if st == STATE_INTRO:
        draw_intro_px(cv, game)
    elif st == STATE_NAME:
        draw_name_px(cv, game)
    else:
        draw_boss_px(cv, game)
        draw_obstacles(cv, game)
        draw_avatar(cv, game)
        draw_helper_px(cv, game)
        draw_particles_px(cv, game)
        if st in (STATE_PLAY, STATE_COUNTDOWN):
            draw_hud_px(cv, game)
        if game.flash > 0:
            dim(cv, min(0.6, game.flash * 0.6), C["red"])
        if st == STATE_WAIT:
            draw_wait_px(cv, game, cam)
        elif st == STATE_COUNTDOWN:
            draw_countdown_px(cv, game)
        elif st == STATE_PLAY:
            if not game.face_visible:
                box(cv, 80, 76, 240, 98, C["maroon"], C["yellow"])
                draw_text(cv, "YÜZ KAYBOLDU", LW // 2, 83, C["yellow"], anchor="c", shadow=None)
            elif game.survival < 0.9:
                draw_text(cv, "BAŞLA!", LW // 2, 70 - int(game.survival * 20), C["gold"], scale=3, anchor="c", outline=True)
            draw_banner_px(cv, game)
        elif st == STATE_OVER:
            draw_over_px(cv, game)
        if cam is not None and st != STATE_OVER:
            if st == STATE_WAIT:
                draw_cam_pip(cv, cam, 112, 63, LW // 2, 100)
            elif cam_mode == 0:
                draw_cam_pip(cv, cam, 48, 27, LW - 29, LH - 17)
    if game.shake > 0.05:                                           # ekran sarsintisi (piksel kaydirma)
        m = int(round(game.shake * 3))
        cv = np.roll(cv, (random.randint(-m, m), random.randint(-m, m)), axis=(0, 1))
    return cv


def to_screen(cv: np.ndarray, scanlines: bool = True) -> np.ndarray:
    """320x180 tuvali 960x540'a en-yakin-komsu ile buyutur; istege bagli hafif CRT tarama cizgileri."""
    up = cv2.resize(cv, (LW * SCALE, LH * SCALE), interpolation=cv2.INTER_NEAREST)
    if scanlines:
        up[SCALE - 1::SCALE] = (up[SCALE - 1::SCALE].astype(np.uint16) * 215 // 256).astype(np.uint8)
    return up
