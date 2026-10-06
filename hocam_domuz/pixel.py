"""8-bit piksel motoru: palet, sprite piksellendirme, bitmap font, piksel tuval yardimcilari."""

import cv2
import numpy as np

LW, LH = 320, 180        # dusuk cozunurluklu oyun tuvali (3x buyutulup 960x540 olur)
SCALE = 3                # mantiksal (960x540) piksel / tuval pikseli

# --------------------------------------------------------------------------
# PALET (BGR) - sinirli renk = 8-bit gorunum
# --------------------------------------------------------------------------
PALETTE = {
    "ink": (24, 16, 32), "white": (246, 246, 240), "gray1": (200, 196, 190), "gray2": (140, 134, 130),
    "gray3": (84, 78, 84), "navy": (96, 48, 40), "blue": (200, 120, 50), "sea1": (170, 100, 40),
    "sea2": (220, 150, 70), "sea3": (245, 200, 130), "sky": (250, 205, 150), "cyan": (240, 230, 120),
    "purple": (130, 50, 110), "violet": (170, 70, 150), "magenta": (190, 90, 220), "pink": (190, 160, 255),
    "pink2": (150, 110, 235), "maroon": (40, 28, 125), "red": (60, 60, 230), "orange": (50, 125, 240),
    "gold": (70, 190, 245), "yellow": (110, 235, 250), "green0": (40, 90, 40), "green1": (60, 150, 60),
    "green2": (90, 205, 90), "green3": (140, 235, 150), "brown0": (30, 50, 95), "brown1": (45, 85, 140),
    "brown2": (80, 125, 180), "tan": (120, 170, 205), "skin": (150, 200, 240), "skin2": (110, 160, 225),
}
C = PALETTE
_PAL = np.array(list(PALETTE.values()), np.float32)


def quantize(rgb: np.ndarray) -> np.ndarray:
    """Goruntunun her pikselini en yakin palet rengine cevirir."""
    flat = rgb.reshape(-1, 1, 3).astype(np.float32)
    idx = ((flat - _PAL[None]) ** 2).sum(axis=2).argmin(axis=1)
    return _PAL[idx].reshape(rgb.shape).astype(np.uint8)


# Palet tabanli karartma: her palet rengi bir tonu koyu palet rengine gider (gurultusuz, 8-bit tonlu)
_PAL_U8 = np.array(list(PALETTE.values()), np.uint8)


def _key(a: np.ndarray) -> np.ndarray:
    return ((a[..., 0] >> 3).astype(np.uint16) << 10) | ((a[..., 1] >> 3).astype(np.uint16) << 5) | (a[..., 2] >> 3)


_LUT = np.zeros(32768, np.uint8)
_pal_keys = _key(_PAL_U8[None])[0]
assert len(set(_pal_keys.tolist())) == len(_pal_keys), "palet renkleri 15-bit anahtarda cakisiyor"
_LUT[_pal_keys] = np.arange(len(_pal_keys))
_SHADE = {}


def shade(canvas: np.ndarray, factor: float):
    """Tuvali factor (<1) kadar koyulastirir; sonuc yine palet renkleridir."""
    if factor >= 1.0:
        return
    tbl = _SHADE.get(factor)
    if tbl is None:
        tbl = _SHADE[factor] = quantize((_PAL_U8.astype(np.float32) * factor).astype(np.uint8)[None])[0]
    canvas[:] = tbl[_LUT[_key(canvas)]]


# Siralı titresim (Bayer 4x4): kademeli gecis/karartma icin
_BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)
BAYER = np.tile((_BAYER + 0.5) / 16.0, (LH // 4 + 1, LW // 4 + 1))[:LH, :LW]


def dither_mask(level: float, y1=0, y2=LH, x1=0, x2=LW) -> np.ndarray:
    """Bolgede piksellerin 'level' (0-1) orani True olan titresimli maske."""
    return BAYER[y1:y2, x1:x2] < level


def dim(canvas: np.ndarray, level: float, color=C["ink"]):
    """Tuvali siralı titresimle karartir (level: karartilan piksel orani)."""
    if level <= 0:
        return
    m = dither_mask(level)[: canvas.shape[0], : canvas.shape[1]]
    canvas[m] = color


# --------------------------------------------------------------------------
# SPRITE PIKSELLENDIRME
# --------------------------------------------------------------------------
def pixelize(sprite: np.ndarray, factor: int = SCALE, thr: int = 96, outline: bool = True) -> np.ndarray:
    """Yuksek cozunurluklu BGRA sprite'i piksel-art'a cevirir: kucult, palete indirge,
    alfayi sert esikle, 1 piksel koyu dis cizgi ekle. Cikti 2 piksel buyuk (cizgi payi)."""
    h, w = sprite.shape[:2]
    nw, nh = max(1, round(w / factor)), max(1, round(h / factor))
    f = sprite.astype(np.float32)
    a = f[..., 3:4] / 255.0
    pm = np.concatenate([f[..., :3] * a, a], axis=2)              # alfa-carpimli: kenarlar kararmasin
    small = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_AREA)
    sa = small[..., 3]
    rgb = np.clip(small[..., :3] / np.maximum(sa[..., None], 1e-3), 0, 255)
    mask = sa * 255 >= thr
    out = np.zeros((nh + 2, nw + 2, 4), np.uint8)
    out[1:-1, 1:-1, :3] = quantize(rgb)
    out[1:-1, 1:-1, 3] = mask * 255
    if outline:
        m = (out[..., 3] > 0).astype(np.uint8)
        ring = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))) > 0) & (m == 0)
        out[ring, :3] = C["ink"]
        out[ring, 3] = 255
    out[out[..., 3] == 0, :3] = 0
    return out


def blit(canvas: np.ndarray, spr: np.ndarray, cx: float, cy: float, alpha: float = 1.0, flip: bool = False):
    """Piksel sprite'i merkezi (cx, cy) olacak sekilde tuvale yapistirir. alpha<1: titresimli solma."""
    if flip:
        spr = spr[:, ::-1]
    sh, sw = spr.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
    sx1, sy1, sx2, sy2 = max(0, -x0), max(0, -y0), min(sw, W - x0), min(sh, H - y0)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    sub = spr[sy1:sy2, sx1:sx2]
    m = sub[..., 3] > 0
    if alpha < 1.0:
        m &= BAYER[y0 + sy1:y0 + sy2, x0 + sx1:x0 + sx2] < alpha
    region = canvas[y0 + sy1:y0 + sy2, x0 + sx1:x0 + sx2]
    region[m] = sub[..., :3][m]


# --------------------------------------------------------------------------
# BITMAP FONT (5x7, buyuk harf; Turkce karakterli)
# --------------------------------------------------------------------------
_GLYPHS = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.####", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.", "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|##..#|#.#.#|#..##|#...#|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|##.##|#...#", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "####.|....#|....#|.###.|....#|....#|####.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    " ": ".....|.....|.....|.....|.....|.....|.....", ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ",": ".....|.....|.....|.....|.##..|..#..|.#...", ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "!": "..#..|..#..|..#..|..#..|..#..|.....|..#..", "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "-": ".....|.....|.....|#####|.....|.....|.....", "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "/": "....#|....#|...#.|..#..|.#...|#....|#....", "'": "..#..|..#..|.#...|.....|.....|.....|.....",
    "(": "...#.|..#..|.#...|.#...|.#...|..#..|...#.", ")": ".#...|..#..|...#.|...#.|...#.|..#..|.#...",
    "=": ".....|.....|#####|.....|#####|.....|.....", "%": "##..#|##..#|...#.|..#..|.#...|#..##|#..##",
    "_": ".....|.....|.....|.....|.....|.....|#####", "*": ".....|#.#.#|.###.|#####|.###.|#.#.#|.....",
    "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.", "<": "...#.|..#..|.#...|#....|.#...|..#..|...#.",
    ">": ".#...|..#..|...#.|....#|...#.|..#..|.#...", "@": ".###.|#...#|#.###|#.#.#|#.##.|#....|.###.",
    "&": ".##..|#..#.|#.#..|.#...|#.#.#|#..#.|.##.#", "^": "..#..|.#.#.|#...#|.....|.....|.....|.....",
}
# Turkce harfler: (taban harf, aksan)
_ACCENTS = {"Ç": ("C", "cedilla"), "Ğ": ("G", "breve"), "İ": ("I", "dot"), "Ö": ("O", "umlaut"),
            "Ş": ("S", "cedilla"), "Ü": ("U", "umlaut"), "Â": ("A", "circ")}
_ACC_ROWS = {"dot": {0: ".....", 1: "..#.."}, "umlaut": {0: ".....", 1: ".#.#."},
             "breve": {0: "#...#", 1: ".###."}, "circ": {0: "..#..", 1: ".#.#."}}
GLYPH_H = 10             # 2 satir ust aksan + 7 harf + 1 satir alt aksan
BASE_ROW = 2             # harfin ilk satirinin maskedeki yeri

_TR_UPPER = str.maketrans({"i": "İ", "ı": "I", "â": "Â", "ç": "Ç", "ğ": "Ğ", "ö": "Ö", "ş": "Ş", "ü": "Ü"})


def upper_tr(text: str) -> str:
    """Turkce kurallarina uygun buyuk harf (i -> İ, ı -> I)."""
    return text.translate(_TR_UPPER).upper()


MISSING = set()          # fontta olmadigi icin "?" ile cizilen karakterler (testler kontrol eder)


def _glyph_mask(ch: str) -> np.ndarray:
    m = np.zeros((GLYPH_H, 5), bool)
    base, acc = (_ACCENTS[ch] if ch in _ACCENTS else (ch, None))
    if base not in _GLYPHS:
        MISSING.add(ch)
    rows = _GLYPHS.get(base, _GLYPHS["?"]).split("|")
    for r, row in enumerate(rows):
        m[BASE_ROW + r] = [c == "#" for c in row]
    if acc == "cedilla":
        m[BASE_ROW + 7, 2] = True
    elif acc:
        for r, row in _ACC_ROWS[acc].items():
            m[r] = [c == "#" for c in row]
            if acc == "dot":
                m[r] = [c == "#" for c in row]
    return m


_TEXT_CACHE = {}


def text_mask(text: str) -> np.ndarray:
    """Metnin boolean maskesi (GLYPH_H x genislik); onbellekli."""
    m = _TEXT_CACHE.get(text)
    if m is None:
        cols = [np.pad(_glyph_mask(ch), ((0, 0), (0, 1))) for ch in text]
        m = np.concatenate(cols, axis=1)[:, :-1] if cols else np.zeros((GLYPH_H, 0), bool)
        if len(_TEXT_CACHE) > 600:
            _TEXT_CACHE.clear()
        _TEXT_CACHE[text] = m
    return m


def text_width(text: str, scale: int = 1) -> int:
    return max(0, len(text) * 6 - 1) * scale


def missing_glyphs(text: str) -> set:
    """Fontta olmayan karakterler (testler icin)."""
    return {ch for ch in text if ch not in _GLYPHS and ch not in _ACCENTS}


def _paint(canvas, mask, x, y, color):
    H, W = canvas.shape[:2]
    h, w = mask.shape
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub = mask[y1 - y:y2 - y, x1 - x:x2 - x]
    canvas[y1:y2, x1:x2][sub] = color


def draw_text(canvas, text, x, y, color, scale=1, anchor="l", shadow=C["ink"], outline=False, grad=None):
    """Piksel yazi. (x, y) = harflerin ust-sol kosesi (anchor 'c' ortalar, 'r' saga yaslar).
    shadow: gölge rengi (None = yok); outline: etrafina ink cerceve; grad=(ust, alt): iki tonlu."""
    text = upper_tr(text)
    mask = text_mask(text)
    if scale > 1:
        mask = np.kron(mask, np.ones((scale, scale), bool))
    w = text_width(text, scale)
    x0 = int(x - w / 2) if anchor == "c" else int(x - w) if anchor == "r" else int(x)
    y0 = int(y) - BASE_ROW * scale
    if outline:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            _paint(canvas, mask, x0 + dx * max(1, scale // 2 + (scale > 1)), y0 + dy * max(1, scale // 2 + (scale > 1)),
                   C["ink"])
    elif shadow is not None:
        _paint(canvas, mask, x0 + scale, y0 + scale, shadow)
    if grad:
        mid = y0 + (BASE_ROW + 3) * scale + scale // 2
        top = np.zeros_like(mask)
        top[:max(0, mid - y0)] = mask[:max(0, mid - y0)]
        _paint(canvas, top, x0, y0, grad[0])
        _paint(canvas, mask & ~top, x0, y0, grad[1])
    else:
        _paint(canvas, mask, x0, y0, color)


# --------------------------------------------------------------------------
# PIKSEL TUVAL CIZIMLERI
# --------------------------------------------------------------------------
def rect(canvas, x1, y1, x2, y2, color):
    """Dolu dikdortgen [x1,x2) x [y1,y2)."""
    H, W = canvas.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    if x2 > x1 and y2 > y1:
        canvas[y1:y2, x1:x2] = color


def box(canvas, x1, y1, x2, y2, fill, border, shadow=True):
    """8-bit tarzi cerceveli panel (kose pikselleri kirik, istege bagli gölge)."""
    if shadow:
        rect(canvas, x1 + 2, y1 + 2, x2 + 2, y2 + 2, C["ink"])
    rect(canvas, x1, y1, x2, y2, border)
    rect(canvas, x1 + 1, y1 + 1, x2 - 1, y2 - 1, fill)
    for cx, cy in ((x1, y1), (x2 - 1, y1), (x1, y2 - 1), (x2 - 1, y2 - 1)):      # kose kirpma
        rect(canvas, cx, cy, cx + 1, cy + 1, fill if shadow else fill)


def pixel_line(canvas, p0, p1, color, thick=1):
    cv2.line(canvas, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, thick, cv2.LINE_8)


def pixel_circle(canvas, c, r, color, fill=True):
    cv2.circle(canvas, (int(c[0]), int(c[1])), int(r), color, -1 if fill else 1, cv2.LINE_8)
