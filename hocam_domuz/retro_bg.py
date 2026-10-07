"""Pixel-art IYTE kampusu arka plani: gunbatimi, Ege, Gulbahce'nin yel degirmenleri, 'IYTE' binasi, domuzlar."""

import math

import cv2
import numpy as np

from .pixel import BAYER, C, LH, LW, blit, draw_text, pixel_circle, pixel_line, pixelize, rect
from .sprites import sprite_domuz

SKY = [C["navy"], C["purple"], C["violet"], C["magenta"], C["pink2"], C["orange"], C["gold"]]
HORIZON = 110
SEA_TOP, SEA_BOTTOM = 108, 132
GROUND = 134
SUN = (238, 100)
TURBINES = [(46, 14), (92, 12), (140, 15), (205, 11)]       # (x, kule yuksekligi)
MAIN = (192, 104, 296, GROUND)                               # ana bina (x1, y1, x2, y2)
WINDOWS = [(x, y) for y in (110, 119, 128 - 0) for x in range(198, 290, 8) if not (y == 128 and 238 <= x <= 252)]
STARS = [(int(x), int(y), int(p)) for x, y, p in zip(
    np.random.RandomState(3).randint(4, LW - 4, 28), np.random.RandomState(4).randint(3, 52, 28),
    np.random.RandomState(5).randint(0, 6, 28))]

_cache = {}


def _far_hill(x):
    return 92 + 7 * math.sin(x * 0.028 + 0.5) + 4 * math.sin(x * 0.071 + 1.7)


def _near_hill(x):
    return 124 + 4 * math.sin(x * 0.04 + 3) + 3 * math.sin(x * 0.1)


def _tree(cv, x, base):
    rect(cv, x - 1, base - 7, x + 1, base, C["brown1"])
    pixel_circle(cv, (x, base - 11), 7, C["green0"])
    pixel_circle(cv, (x - 1, base - 12), 5, C["green1"])
    pixel_circle(cv, (x - 2, base - 13), 2, C["green2"])


def _build_static() -> np.ndarray:
    cv = np.zeros((LH, LW, 3), np.uint8)
    # gokyuzu: 7 renk bandi, bandlar arasi siralı titresim
    n = len(SKY)
    pal = np.array(SKY, np.uint8)
    pos = np.arange(HORIZON)[:, None] / (HORIZON - 1) * (n - 1)
    i = np.clip(np.floor(pos).astype(int), 0, n - 2)
    cv[:HORIZON] = pal[i + (BAYER[:HORIZON] < (pos - i))]
    # gunes ve hale
    yy, xx = np.mgrid[0:HORIZON, 0:LW]
    d = np.sqrt((xx - SUN[0]) ** 2 + (yy - SUN[1]) ** 2)
    halo = (d > 15) & (d < 30) & (BAYER[:HORIZON] < 0.55 * (1 - (d - 15) / 15))
    cv[:HORIZON][halo] = C["yellow"]
    cv[:HORIZON][d <= 15] = C["yellow"]
    cv[:HORIZON][(d <= 12) & (d > 8)] = C["white"]
    # uzak tepeler
    for x in range(LW):
        cv[int(_far_hill(x)):SEA_TOP + 4, x] = C["purple"]
    # Ege: yukarıdan asagiya sea3 -> sea2 -> sea1, titresimli
    sea = [C["sea3"], C["sea2"], C["sea1"]]
    for y in range(SEA_TOP, SEA_BOTTOM):
        pos2 = (y - SEA_TOP) / (SEA_BOTTOM - SEA_TOP - 1) * 2
        k = min(1, int(pos2))
        row = np.where(BAYER[y] < (pos2 - k), 1, 0) + k
        cv[y] = np.array(sea, np.uint8)[row]
    # gunes yansimasi (titresimli sutun)
    for y in range(SEA_TOP, SEA_BOTTOM):
        w = 3 + (y - SEA_TOP) // 3
        m = (np.abs(np.arange(LW) - SUN[0]) < w) & (BAYER[y] < 0.5)
        cv[y][m] = C["yellow"]
    # yakin tepeler (koyu yesil) ve zemin
    for x in range(LW):
        cv[int(_near_hill(x)):GROUND + 2, x] = C["green0"]
    cv[GROUND:] = C["green1"]
    cv[GROUND:][BAYER[GROUND:] < 0.2] = C["green2"]
    cv[GROUND + 30:][BAYER[GROUND + 30:] > 0.86] = C["green0"]
    path = np.array([[158, GROUND], [172, GROUND], [216, LH], [118, LH]], np.int32)
    cv2.fillPoly(cv, [path], C["tan"], cv2.LINE_8)
    cv2.polylines(cv, [path], True, C["brown2"], 1, cv2.LINE_8)
    rs = np.random.RandomState(11)
    for _ in range(34):                                   # cicekler
        x, y = rs.randint(2, LW - 2), rs.randint(GROUND + 4, LH - 2)
        if cv[y, x].tolist() in (list(C["green1"]), list(C["green2"]), list(C["green0"])):
            cv[y, x] = [C["pink"], C["white"], C["yellow"], C["red"]][rs.randint(0, 4)]
    # ana bina + tabela
    x1, y1, x2, y2 = MAIN
    rect(cv, x1, y1, x2, y2, C["tan"])
    rect(cv, x1, y1, x2, y1 + 4, C["maroon"])
    rect(cv, x1, y1 + 4, x2, y1 + 5, C["brown1"])
    rect(cv, x1, y2 - 3, x2, y2, C["brown2"])
    rect(cv, x1 - 1, y1, x1, y2, C["ink"])
    rect(cv, x2, y1, x2 + 1, y2, C["ink"])
    rect(cv, x1 - 1, y1 - 1, x2 + 1, y1, C["ink"])
    rect(cv, 240, 124, 252, y2, C["brown0"])
    rect(cv, 245, 124, 246, y2, C["brown1"])
    cv[130, 250] = C["gold"]
    rect(cv, 226, 91, 264, 104, C["ink"])
    rect(cv, 227, 92, 263, 103, C["maroon"])
    rect(cv, 227, 92, 263, 93, C["gold"])
    draw_text(cv, "İYTE", 245, 96, C["white"], anchor="c", shadow=None)
    # saat kulesi
    rect(cv, 171, 92, 187, GROUND, C["ink"])
    rect(cv, 172, 93, 186, GROUND, C["gray1"])
    rect(cv, 181, 93, 186, GROUND, C["gray2"])
    cv2.fillPoly(cv, [np.array([[170, 92], [188, 92], [179, 80]], np.int32)], C["maroon"], cv2.LINE_8)
    pixel_circle(cv, (179, 100), 5, C["ink"])
    pixel_circle(cv, (179, 100), 4, C["white"])
    # laboratuvar binasi
    rect(cv, 21, 111, 87, GROUND, C["ink"])
    rect(cv, 22, 112, 86, GROUND, C["gray1"])
    rect(cv, 22, 112, 86, 116, C["maroon"])
    rect(cv, 22, GROUND - 3, 86, GROUND, C["gray2"])
    for x in range(27, 82, 9):
        rect(cv, x, 120, x + 5, 126, C["sea3"])
        rect(cv, x, 120, x + 5, 121, C["white"])
    pixel_line(cv, (40, 111), (40, 99), C["gray3"])
    # agaçlar
    for tx in (10, 104, 142, 306):
        _tree(cv, tx, GROUND + 8)
    sea_cols = np.array([C["sea1"], C["sea2"], C["sea3"], C["yellow"]], np.uint8)
    region = cv[SEA_TOP:SEA_BOTTOM]
    sea_mask = np.zeros((SEA_BOTTOM - SEA_TOP, LW), bool)
    for col in sea_cols:                                   # sadece gercekten deniz gorunen pikseller parildar
        sea_mask |= (region == col).all(axis=2)
    _cache["sea_mask"] = sea_mask
    return cv


def _cloud(w: int, h: int) -> np.ndarray:
    """Pikselli bulut: ust kenari beyaz, alt kenari koyu pembe."""
    m = np.zeros((h, w), np.uint8)
    for cx, cy, r in ((w * 0.25, h * 0.62, h * 0.36), (w * 0.5, h * 0.42, h * 0.46), (w * 0.75, h * 0.62, h * 0.34)):
        cv2.circle(m, (int(cx), int(cy)), int(r), 1, -1, cv2.LINE_8)
    m[int(h * 0.62):int(h * 0.9), int(w * 0.18):int(w * 0.82)] = 1
    m = m.astype(bool)
    spr = np.zeros((h, w, 4), np.uint8)
    spr[m] = (*C["pink"], 255)
    spr[m & ~np.roll(m, 1, axis=0), :3] = C["white"]
    spr[m & ~np.roll(m, -1, axis=0), :3] = C["pink2"]
    return spr


def _boar_sprite() -> np.ndarray:
    return pixelize(sprite_domuz(78, 44, 0.0, 1.0, speed_lines=False))


def background(t: float) -> np.ndarray:
    """t anindaki animasyonlu arka plan (320x180 BGR). Statik katman onbellekli, hareketliler uste cizilir."""
    if "static" not in _cache:
        _cache["static"] = _build_static()   # (ayrica sea_mask'i onbellege yazar)
        _cache["clouds"] = [(_cloud(34, 11), 22, 5.0), (_cloud(26, 9), 40, 3.0), (_cloud(40, 12), 14, 7.0)]
        _cache["boar"] = _boar_sprite()
    cv = _cache["static"].copy()
    step = int(t * 8)
    for x, y, ph in STARS:                                # yildizlar goz kirpar
        if y < 46:
            cv[y, x] = C["white"] if (step + ph) % 7 < 4 else C["gold"]
    for k, (spr, y, speed) in enumerate(_cache["clouds"]):  # bulutlar
        cx = (LW + 60) - ((t * speed + k * 97) % (LW + 60)) - 30
        blit(cv, spr, cx, y)
    for k in range(3):                                    # martilar
        bx = (LW + 40) - ((t * (14 + 5 * k) + k * 83) % (LW + 40)) - 20
        by = 30 + k * 11 + int(2 * math.sin(t * 2 + k))
        up = (step + k) % 2 == 0
        for dx, dy in ((-2, 1 if up else 0), (-1, 0 if up else 1), (0, 1), (1, 0 if up else 1), (2, 1 if up else 0)):
            if 0 <= int(bx) + dx < LW:
                cv[by + dy, int(bx) + dx] = C["ink"]
    for x, h in TURBINES:                                 # yel degirmenleri
        base = int(_far_hill(x))
        pixel_line(cv, (x, base), (x, base - h), C["gray1"], 1)
        hub = (x, base - h)
        for b in range(3):
            a = t * 1.3 + b * 2.094
            pixel_line(cv, hub, (hub[0] + math.cos(a) * 7, hub[1] + math.sin(a) * 7), C["white"], 1)
        cv[hub[1], hub[0]] = C["red"]
    # Ege parıltilari
    sea = cv[SEA_TOP:SEA_BOTTOM]
    yy, xx = np.mgrid[SEA_TOP:SEA_BOTTOM, 0:LW]
    h = ((xx * 73856093) ^ (yy * 19349663) ^ (step * 83492791)) % 61
    # (statik katman olusurken sea_mask kaydedildi: parilti sadece denizde, binalarda degil)
    sea_mask = _cache["sea_mask"]
    sea[(h == 0) & sea_mask] = C["white"]
    sea[(((xx + step * 2) % 24 == 0) & ((yy % 5) == 0)) & sea_mask] = C["sea3"]
    # bina pencereleri yanip soner
    for i, (wx, wy) in enumerate(WINDOWS):
        lit = ((i * 7919 + int(t * 0.4) * 104729) % 5) != 0
        rect(cv, wx, wy, wx + 3, wy + 4, C["yellow"] if lit else C["navy"])
    ang = t * 0.5                                         # saat kulesinin akrebi/yelkovani
    pixel_line(cv, (179, 100), (179 + math.cos(ang) * 3, 100 + math.sin(ang) * 3), C["ink"])
    pixel_line(cv, (179, 100), (179 + math.cos(ang / 12) * 2, 100 + math.sin(ang / 12) * 2), C["ink"])
    if step % 6 < 3:                                      # laboratuvar anteni isigi
        cv[98, 40] = C["red"]
    boar = _cache["boar"]                                 # yaban domuzlari yuruyor
    for k, (y, speed) in enumerate(((158, 9.0), (170, -6.0))):
        span = LW + 60
        px = ((t * abs(speed) + k * 130) % span) - 30
        x = px if speed > 0 else LW - px
        blit(cv, boar, x, y + (1 if (step + k) % 2 else 0), flip=speed < 0)
    return cv
