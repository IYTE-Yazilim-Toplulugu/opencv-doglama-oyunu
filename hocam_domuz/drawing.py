"""Cizim yardimcilari: yazi (Turkce yedekli), yuvarlak kutu, vinyet, kucuk sekiller."""

import cv2
import numpy as np

from .config import BLACK, CREAM, DARK, FONT


_TR_MAP = str.maketrans("çğıöşüâÇĞİÖŞÜÂ", "cgiosuaCGIOSUA")


def _unicode_text_ok() -> bool:
    """OpenCV Turkce harf cizebiliyor mu? (OpenCV 5+ evet; 4.x Hershey fontu '?' cizer.)"""
    def draw_char(ch):
        im = np.zeros((48, 80, 3), np.uint8)
        cv2.putText(im, ch, (5, 36), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return im
    a = draw_char("Ş")
    return not (np.array_equal(a, draw_char("?")) or np.array_equal(a, draw_char("??")))


UNICODE_OK = _unicode_text_ok()


def tr(text: str) -> str:
    """Turkce karakter desteklenmiyorsa ASCII karsiligina cevirir (Ş->S, İ->I ...)."""
    return text if UNICODE_OK else text.translate(_TR_MAP)


def txt(img, text, x, y, scale, color, th=2, u=1.0, anchor="c", outline=True):
    """Kenarlikli yazi. Olcekler 960x540 tasarim birimindedir, u ile ekrana uyarlanir.
    anchor: 'c' ortala, 'l' sola, 'r' saga yasla. Yazinin bittigi x'i dondurur."""
    text = tr(text)
    s, t = scale * u, max(1, int(round(th * u)))
    (tw, _), _ = cv2.getTextSize(text, FONT, s, t)
    x0 = int(x - tw / 2) if anchor == "c" else int(x - tw) if anchor == "r" else int(x)
    y = int(y)
    if outline:
        # Kenarlik: ayni kalinlikta, 8 yone kaydirilmis siyah kopyalar. (Farkli kalinlik
        # kullanmak yeni OpenCV surumlerinde karakter genisligini degistirip yaziyi bozar.)
        d = max(1, int(round(s * 1.6)))
        for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d), (-d, -d), (d, -d), (-d, d), (d, d)):
            cv2.putText(img, text, (x0 + dx, y + dy), FONT, s, BLACK, t, cv2.LINE_AA)
    cv2.putText(img, text, (x0, y), FONT, s, color, t, cv2.LINE_AA)
    return x0 + tw


def text_width(text, scale, th, u) -> int:
    text = tr(text)
    (tw, _), _ = cv2.getTextSize(text, FONT, scale * u, max(1, int(round(th * u))))
    return tw


def blend_rect(img, x1, y1, x2, y2, color, alpha):
    """Dikdortgeni yari saydam doldurur (sadece ilgili bolgeyi isler)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    colored = np.full_like(roi, color)
    cv2.addWeighted(colored, alpha, roi, 1 - alpha, 0, dst=roi)


def _fill_rr(img, x1, y1, x2, y2, r, color):
    """Yuvarlatilmis dolu dikdortgen."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _outline_rr(img, x1, y1, x2, y2, r, color, th):
    """Yuvarlatilmis dikdortgen cercevesi."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, th, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, th, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, th, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, th, cv2.LINE_AA)
    for (cx, cy), a0 in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                         ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, th, cv2.LINE_AA)


def rrect(img, x1, y1, x2, y2, r, color, alpha=1.0, border=None, bth=2):
    """Yari saydam yuvarlatilmis kutu (+ istege bagli cerceve)."""
    x1, y1, x2, y2, r = int(x1), int(y1), int(x2), int(y2), int(r)
    if alpha >= 1.0:
        _fill_rr(img, x1, y1, x2, y2, r, color)
    elif alpha > 0:
        h, w = img.shape[:2]
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2 + 1), min(h, y2 + 1)
        if cx2 > cx1 and cy2 > cy1:
            roi = img[cy1:cy2, cx1:cx2]
            overlay = roi.copy()
            _fill_rr(overlay, x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1, r, color)
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)
    if border is not None:
        _outline_rr(img, x1, y1, x2, y2, r, border, max(1, int(bth)))


def ease_out(t: float) -> float:
    """0..1 arasi yavaslayarak biten gecis (animasyonlar icin)."""
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


_VIGNETTE_CACHE = {}


def apply_vignette(frame: np.ndarray) -> np.ndarray:
    """Kenarlari hafifce karartip genel goruntuyu sinematik yapar (maske onbellekli)."""
    h, w = frame.shape[:2]
    mask = _VIGNETTE_CACHE.get((w, h))
    if mask is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
        m = np.clip(1.08 - 0.32 * d ** 2, 0.55, 1.0).astype(np.float32)
        mask = np.ascontiguousarray(np.repeat(m[..., None], 3, axis=2))
        _VIGNETTE_CACHE[(w, h)] = mask
    return cv2.multiply(frame, mask, dtype=cv2.CV_8U)


def draw_pill(img, text, cx, cy, scale, u, bg=DARK, fg=CREAM, alpha=0.7, border=None, th=1):
    """Ortalanmis yuvarlak etiket (pill)."""
    tw = text_width(text, scale, th, u)
    ph, pw = int(14 * u * scale / 0.5) , int(12 * u)
    rrect(img, cx - tw / 2 - pw, cy - ph, cx + tw / 2 + pw, cy + ph, ph, bg, alpha, border, max(1, 2 * u))
    txt(img, text, cx, cy + int(5 * u * scale / 0.5), scale, fg, th, u, outline=False)


def blend_ellipse(img, center, axes, color, alpha):
    """Yari saydam dolu elips (allik gibi yumusak lekeler icin)."""
    (x, y), (ax, ay) = center, axes
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x - ax - 1), max(0, y - ay - 1), min(w, x + ax + 2), min(h, y + ay + 2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.ellipse(overlay, (x - x1, y - y1), (ax, ay), 0, 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_star(img, x, y, r, color, th=2):
    """Kucuk arti seklinde pirilti."""
    cv2.line(img, (int(x - r), int(y)), (int(x + r), int(y)), color, th, cv2.LINE_AA)
    cv2.line(img, (int(x), int(y - r)), (int(x), int(y + r)), color, th, cv2.LINE_AA)
