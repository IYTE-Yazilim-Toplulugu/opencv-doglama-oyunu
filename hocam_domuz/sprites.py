"""Engel/boss/kalem cocuk sprite'lari (saydam BGRA katmanda cizilip yerlestirilir)."""

import cv2
import math
import numpy as np
import os

from .config import (
    FONT,
    KIND_DEVRE,
    KIND_DOMUZ,
    KIND_KALEM,
    KIND_KIMYA,
    KIND_PROJ,
    KIND_RUZGAR,
    KIND_TERMO,
    KIND_VIZE,
    ROOT,
)


def _canvas(w: int, h: int, pad: int) -> np.ndarray:
    """Seffaf (BGRA) cizim tuvali: engel kutusunun etrafinda pad kadar bosluk."""
    return np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)


def sprite_domuz(w: int, h: int, t: float, seed: float, speed_lines: bool = True) -> np.ndarray:
    """Kizgin yaban domuzu (onden gorunum). Hiz cizgileri t ile kayar."""
    pad = int(max(w, h) * 0.4)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    # Hiz cizgileri: domuz hizla asagi geliyor
    for i in (-1, 0, 1) if speed_lines else ():
        x = cx + int(i * w * 0.28)
        off = int((t * 420 + i * 57 + seed * 13) % 36)
        y1 = pad - int(h * 0.06) - off // 2
        cv2.line(cv, (x, y1), (x, y1 - int(h * 0.38)), (255, 255, 255, 150),
                 max(2, w // 38), cv2.LINE_AA)
    # Kulaklar
    for sx in (-1, 1):
        pts = np.array([[cx + sx * w // 5, cy - h // 3], [cx + sx * int(w * 0.50), cy - int(h * 0.62)],
                        [cx + sx * int(w * 0.36), cy - h // 12]], np.int32)
        cv2.fillPoly(cv, [pts], (30, 55, 100, 255), cv2.LINE_AA)
        inner = np.array([[cx + sx * int(w * 0.27), cy - int(h * 0.28)],
                          [cx + sx * int(w * 0.43), cy - int(h * 0.50)],
                          [cx + sx * int(w * 0.34), cy - int(h * 0.15)]], np.int32)
        cv2.fillPoly(cv, [inner], (110, 120, 200, 255), cv2.LINE_AA)
    # Toynaklar: govdenin altinda sirayla kalkip iner (tiris)
    for k, sx in enumerate((-1, 1)):
        hy = cy + h // 2 - 1 + int(math.sin(t * 16 + k * math.pi) * h * 0.07)
        cv2.ellipse(cv, (cx + sx * int(w * 0.22), hy), (max(3, w // 11), max(3, h // 9)), 0, 0, 360,
                    (20, 30, 50, 255), -1, cv2.LINE_AA)
    # Govde
    cv2.ellipse(cv, (cx, cy), (w // 2, h // 2), 0, 0, 360, (40, 75, 125, 255), -1, cv2.LINE_AA)
    cv2.ellipse(cv, (cx, cy), (w // 2, h // 2), 0, 0, 360, (15, 25, 45, 255), max(2, w // 45), cv2.LINE_AA)
    # Yaldiz/tuy seridi
    cv2.ellipse(cv, (cx, cy - h // 6), (int(w * 0.30), h // 4), 0, 200, 340, (25, 45, 85, 255),
                max(2, w // 40), cv2.LINE_AA)
    # Burun
    cv2.ellipse(cv, (cx, cy + h // 6), (int(w * 0.24), int(h * 0.22)), 0, 0, 360,
                (135, 155, 215, 255), -1, cv2.LINE_AA)
    for sx in (-1, 1):
        cv2.circle(cv, (cx + sx * w // 14, cy + h // 6), max(2, w // 36), (40, 40, 80, 255), -1, cv2.LINE_AA)
        # Gozler ve kizgin kaslar
        ex, ey = cx + sx * int(w * 0.22), cy - int(h * 0.10)
        cv2.circle(cv, (ex, ey), max(3, w // 22), (255, 255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(cv, (ex, ey + 1), max(2, w // 45), (0, 0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(cv, (ex + sx * w // 10, ey - h // 5), (ex - sx * w // 14, ey - h // 11),
                 (10, 15, 30, 255), max(2, w // 28), cv2.LINE_AA)
        # Dis
        cv2.line(cv, (cx + sx * int(w * 0.20), cy + int(h * 0.36)),
                 (cx + sx * int(w * 0.26), cy + int(h * 0.12)), (250, 250, 245, 255),
                 max(3, w // 26), cv2.LINE_AA)
    return cv


def sprite_vize(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """Fizik 101 vize kagidi: kivrik kose, bordo baslik ve daire icine alinmis dusuk not."""
    pad = int(max(w, h) * 0.3)
    cv = _canvas(w, h, pad)
    x1, y1, x2, y2 = pad, pad, pad + w, pad + h
    fold = int(w * 0.28)
    paper = np.array([[x1, y1], [x2 - fold, y1], [x2, y1 + fold], [x2, y2], [x1, y2]], np.int32)
    cv2.fillPoly(cv, [paper], (246, 248, 250, 255), cv2.LINE_AA)
    cv2.polylines(cv, [paper], True, (35, 35, 40, 255), max(2, w // 45), cv2.LINE_AA)
    tri = np.array([[x2 - fold, y1], [x2 - fold, y1 + fold], [x2, y1 + fold]], np.int32)
    cv2.fillPoly(cv, [tri], (200, 205, 212, 255), cv2.LINE_AA)
    cv2.polylines(cv, [tri], True, (35, 35, 40, 255), max(1, w // 60), cv2.LINE_AA)
    s = w / 230.0
    cv2.putText(cv, "FIZ 101", (x1 + int(w * 0.08), y1 + int(h * 0.20)), FONT, 0.62 * s * 1.0,
                (40, 30, 150, 255), max(1, int(s * 2)), cv2.LINE_AA)
    cv2.putText(cv, "VIZE", (x1 + int(w * 0.08), y1 + int(h * 0.34)), FONT, 0.5 * s,
                (90, 90, 95, 255), 1, cv2.LINE_AA)
    for i, formula in enumerate(("F=ma", "E=mc2")):             # vizenin formulleri
        cv2.putText(cv, formula, (x1 + int(w * 0.08), y1 + int(h * (0.50 + 0.13 * i))), FONT,
                    max(0.3, 0.42 * s), (110, 90, 80, 255), 1, cv2.LINE_AA)
    gc = (pad + w // 2, y1 + int(h * 0.75))
    cv2.ellipse(cv, gc, (int(w * 0.32), int(h * 0.18)), -8, 0, 360, (40, 40, 220, 255),
                max(2, w // 40), cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("12", FONT, 0.95 * s, max(2, int(s * 3)))
    cv2.putText(cv, "12", (gc[0] - tw // 2, gc[1] + int(h * 0.075)), FONT, 0.95 * s,
                (40, 40, 220, 255), max(2, int(s * 3)), cv2.LINE_AA)
    return cv


def sprite_ruzgar(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """Gulbahce ruzgari: yari saydam bulut, akan dalgali cizgiler ve ucusan yapraklar."""
    pad = int(h * 0.5)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    cv2.ellipse(cv, (cx, cy), (w // 2, h // 2), 0, 0, 360, (255, 215, 130, 105), -1, cv2.LINE_AA)
    cv2.ellipse(cv, (cx, cy), (w // 2, h // 2), 0, 0, 360, (255, 240, 190, 210), 2, cv2.LINE_AA)
    for i in range(3):
        yb = pad + h * (0.28 + 0.22 * i)
        pts = [(pad + x, int(yb + math.sin(x * 0.045 - t * 7 + i * 1.7 + seed) * h * 0.09))
               for x in range(int(w * 0.10), int(w * 0.90), 6)]
        cv2.polylines(cv, [np.array(pts, np.int32)], False, (255, 255, 255, 235),
                      max(2, int(h * 0.045)), cv2.LINE_AA)
    for k in range(4):                                          # ucusan yapraklar
        lx = pad + int((t * 170 * (1 + 0.25 * k) + k * 97 + seed * 31) % w)
        ly = int(pad + h * (0.2 + 0.2 * k) + math.sin(t * 5 + k) * h * 0.12)
        cv2.ellipse(cv, (lx, ly), (6, 3), int(t * 120 + k * 50), 0, 360, (70, 170, 60, 255), -1, cv2.LINE_AA)
    return cv


def sprite_kimya(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """CHEM 101 Kimya Lab Raporu: icinde yesil sivi kaynayan erlen."""
    pad = int(max(w, h) * 0.3)
    cv = _canvas(w, h, pad)
    cx = pad + w // 2
    y1, y2, x1, x2 = pad, pad + h, pad, pad + w
    nw = max(3, int(w * 0.13))                      # boyun yari genisligi
    yn = y1 + int(h * 0.38)                         # boyun bitisi
    glass = np.array([[cx - nw, y1 + int(h * 0.08)], [cx + nw, y1 + int(h * 0.08)],
                      [cx + nw, yn], [x2, y2], [x1, y2], [cx - nw, yn]], np.int32)
    cv2.fillPoly(cv, [glass], (235, 240, 225, 150), cv2.LINE_AA)

    def edge(y):                                    # govdenin y'deki yari genisligi
        return nw + (w / 2 - nw) * (y - yn) / max(1, (y2 - yn))
    ytop = y1 + int(h * 0.62 + math.sin(t * 6 + seed) * h * 0.02)
    ew = edge(ytop)
    liquid = np.array([[cx - ew, ytop], [cx + ew, ytop], [x2, y2], [x1, y2]], np.int32)
    cv2.fillPoly(cv, [liquid], (90, 225, 100, 235), cv2.LINE_AA)
    for k in range(3):                              # yukari cikan kabarciklar
        by = y2 - int(((t * 55 + k * 23 + seed * 9) % max(1, y2 - ytop)))
        bx = cx + int(math.sin(t * 3 + k * 2 + seed) * w * 0.18)
        cv2.circle(cv, (bx, by), max(2, w // 18), (220, 255, 225, 255), -1, cv2.LINE_AA)
    cv2.polylines(cv, [glass], True, (255, 255, 255, 255), max(2, w // 28), cv2.LINE_AA)
    cv2.rectangle(cv, (cx - nw - 2, y1), (cx + nw + 2, y1 + int(h * 0.09)), (60, 100, 150, 255), -1)
    s = w / 78.0
    cv2.putText(cv, "CHEM", (cx - int(w * 0.28), y2 - int(h * 0.1)), FONT, max(0.28, 0.32 * s),
                (30, 70, 40, 255), 1, cv2.LINE_AA)
    return cv


def sprite_devre(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """EEE 201 Devre Analizi: parlayan elektrik simsegi."""
    pad = int(max(w, h) * 0.45)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    glow = 0.8 + 0.2 * math.sin(t * 30 + seed)
    cv2.ellipse(cv, (cx, cy), (int(w * 1.0 * glow), int(h * 0.62 * glow)), 0, 0, 360,
                (255, 190, 70, 80), -1, cv2.LINE_AA)
    pts = [(0.62, 0.0), (0.12, 0.56), (0.46, 0.56), (0.28, 1.0), (0.90, 0.38), (0.56, 0.38), (0.84, 0.0)]
    bolt = np.array([[pad + int(px * w), pad + int(py * h)] for px, py in pts], np.int32)
    cv2.fillPoly(cv, [bolt], (45, 230, 255, 255), cv2.LINE_AA)
    cv2.polylines(cv, [bolt], True, (255, 255, 255, 255), max(2, w // 18), cv2.LINE_AA)
    for k in range(3):                              # kucuk yan kivilcimlar
        a = t * 9 + k * 2.1 + seed
        x0, y0 = cx + int(math.cos(a) * w * 0.4), cy + int(math.sin(a * 1.3) * h * 0.4)
        cv2.line(cv, (x0, y0), (x0 + int(math.cos(a * 3) * w * 0.35), y0 + int(math.sin(a * 3) * w * 0.35)),
                 (255, 230, 120, 255), 2, cv2.LINE_AA)
    return cv


def sprite_termo(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """ME 301 Termodinamik: alev alev yanan ates topu (alevler yukari dogru)."""
    pad = int(max(w, h) * 0.3)
    cv = _canvas(w, h, pad)
    cx = pad + w // 2
    r = int(min(w, h * 0.78) * 0.5)
    cy = pad + h - r - 2
    for i in range(7):                              # alev dilleri
        a = (i - 3) * 0.33
        flick = 0.75 + 0.25 * math.sin(t * 13 + i * 1.7 + seed)
        tip = (cx + int(math.sin(a) * r * 1.5), cy - int(r * (1.7 + 0.5 * math.cos(a)) * flick))
        tongue = np.array([[cx + int(math.sin(a) * r * 0.5) - int(r * 0.45), cy - int(r * 0.4)],
                           [cx + int(math.sin(a) * r * 0.5) + int(r * 0.45), cy - int(r * 0.4)], tip], np.int32)
        cv2.fillPoly(cv, [tongue], (35, 120, 250, 235), cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), r, (30, 110, 245, 255), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), int(r * 0.72), (60, 205, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), int(r * 0.4), (230, 250, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), r, (20, 40, 120, 255), max(2, w // 40), cv2.LINE_AA)
    return cv


def sprite_proj(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """Boss mermisi: icinde turev ifadesi olan mor, parlayan kure."""
    pad = int(w * 0.5)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    pulse = 1.0 + 0.12 * math.sin(t * 12 + seed)
    cv2.circle(cv, (cx, cy), int(w * 0.85 * pulse), (210, 90, 220, 70), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), w // 2, (120, 30, 150, 255), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), w // 2, (255, 190, 255, 255), max(2, w // 16), cv2.LINE_AA)
    label = ("y'", "dx", "dt", "y''")[int(seed * 10) % 4]
    sc = max(0.3, w / 70.0)
    (tw, th), _ = cv2.getTextSize(label, FONT, sc, 1)
    cv2.putText(cv, label, (cx - tw // 2, cy + th // 2), FONT, sc, (255, 255, 255, 255), 1, cv2.LINE_AA)
    return cv


def sprite_kalem(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """Toplanacak altin kalem: etrafinda nabiz atan halka (kalem cocugu cagirir)."""
    pad = int(max(w, h) * 0.45)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    ring = 0.85 + 0.15 * math.sin(t * 7)
    cv2.circle(cv, (cx, cy), int(max(w, h) * 0.62 * ring), (70, 190, 245, 110), -1, cv2.LINE_AA)
    cv2.circle(cv, (cx, cy), int(max(w, h) * 0.62 * ring), (70, 190, 245, 255), 2, cv2.LINE_AA)
    bw = max(6, int(w * 0.30))
    top, bot = pad + int(h * 0.12), pad + int(h * 0.76)
    cv2.rectangle(cv, (cx - bw, top), (cx + bw, bot), (40, 200, 250, 255), -1)              # govde (sari)
    cv2.rectangle(cv, (cx - bw, top), (cx + bw, bot), (20, 90, 140, 255), 2, cv2.LINE_AA)
    cv2.line(cv, (cx, top), (cx, bot), (20, 150, 200, 255), 2, cv2.LINE_AA)
    tip = np.array([[cx - bw, bot], [cx + bw, bot], [cx, pad + h - 2]], np.int32)
    cv2.fillPoly(cv, [tip], (150, 200, 235, 255), cv2.LINE_AA)
    cv2.fillPoly(cv, [np.array([[cx - bw // 3, pad + int(h * 0.92)], [cx + bw // 3, pad + int(h * 0.92)],
                                [cx, pad + h - 2]], np.int32)], (60, 60, 60, 255), cv2.LINE_AA)
    cv2.rectangle(cv, (cx - bw, pad + int(h * 0.03)), (cx + bw, top), (170, 150, 240, 255), -1)   # silgi
    cv2.rectangle(cv, (cx - bw, pad + int(h * 0.03)), (cx + bw, top), (20, 40, 100, 255), 2, cv2.LINE_AA)
    return cv


def sprite_boss(w: int, h: int, t: float, look: float, hit: float, charge: float, equations: bool = True) -> np.ndarray:
    """MATH 255 - Diferansiyel Denklemler boss'u: boynuzlu, kirmizi gozlu, dislek bir yaratik.
    Etrafinda denklemler doner, altinda dalgalanan tentaculler var. look: -1..1 (gozler oyuncuya bakar),
    hit: vurus beyazlamasi (0..1), charge: saldiri hazirligi (kirmizi aura)."""
    pv, ph = int(h * 0.6), int(w * 0.2)
    cv = np.zeros((h + 2 * pv, w + 2 * ph, 4), np.uint8)
    cx, cy = ph + w // 2, pv + h // 2
    s = w / 330.0
    # Aura
    aura = (60, 60, 240, 120) if charge > 0 else (190, 70, 180, 90)
    k = 1.0 + 0.05 * math.sin(t * (14 if charge > 0 else 3))
    cv2.ellipse(cv, (cx, cy), (int(w * 0.52 * k), int(h * 0.68 * k)), 0, 0, 360, aura, -1, cv2.LINE_AA)
    # Tentaculler (integral isareti gibi kivrilan, ucunda kanca olan)
    for i in range(6):
        bx = cx + int((i - 2.5) * w * 0.115)
        pts = []
        for j in range(12):
            yy = cy + int(h * 0.38) + int(j * h * 0.06)
            xx = bx + int(math.sin(t * 3 + i * 1.1 + j * 0.55) * w * 0.035 * (1 + j * 0.25))
            pts.append((xx, yy))
        for j in range(len(pts) - 1):
            th = max(2, int((9 - j * 0.6) * s * 1.4))
            cv2.line(cv, pts[j], pts[j + 1], (70, 25, 85, 255), th, cv2.LINE_AA)
        cv2.circle(cv, pts[-1], max(3, int(5 * s * 1.4)), (255, 150, 255, 255), -1, cv2.LINE_AA)
    # Boynuzlar
    for sx in (-1, 1):
        horn = np.array([[cx + sx * int(w * 0.14), cy - int(h * 0.36)],
                         [cx + sx * int(w * 0.30), cy - int(h * 0.78)],
                         [cx + sx * int(w * 0.30), cy - int(h * 0.30)]], np.int32)
        cv2.fillPoly(cv, [horn], (40, 15, 55, 255), cv2.LINE_AA)
        cv2.polylines(cv, [horn], True, (255, 150, 255, 255), max(1, int(2 * s * 1.4)), cv2.LINE_AA)
    # Govde
    body_col = (255, 255, 255, 255) if (hit > 0 and int(hit * 12) % 2 == 0) else (50, 18, 62, 255)
    cv2.ellipse(cv, (cx, cy), (int(w * 0.38), int(h * 0.5)), 0, 0, 360, body_col, -1, cv2.LINE_AA)
    cv2.ellipse(cv, (cx, cy), (int(w * 0.38), int(h * 0.5)), 0, 0, 360, (220, 110, 235, 255),
                max(2, int(4 * s * 1.4)), cv2.LINE_AA)
    # Gozler (oyuncuya bakar) ve kizgin kaslar
    for sx in (-1, 1):
        ex, ey = cx + sx * int(w * 0.14), cy - int(h * 0.10)
        cv2.ellipse(cv, (ex, ey), (int(w * 0.085), int(h * 0.15)), 0, 0, 360, (30, 30, 235, 255), -1, cv2.LINE_AA)
        cv2.ellipse(cv, (ex, ey), (int(w * 0.05), int(h * 0.10)), 0, 0, 360, (40, 170, 255, 255), -1, cv2.LINE_AA)
        px = ex + int(look * w * 0.025)
        cv2.ellipse(cv, (px, ey), (max(2, int(w * 0.012)), int(h * 0.10)), 0, 0, 360, (0, 0, 0, 255), -1, cv2.LINE_AA)
        cv2.line(cv, (ex + sx * int(w * 0.11), ey - int(h * 0.20)), (ex - sx * int(w * 0.07), ey - int(h * 0.10)),
                 (240, 140, 245, 255), max(2, int(5 * s * 1.4)), cv2.LINE_AA)
    # Agiz ve disler
    my = cy + int(h * 0.20)
    mw, mh = int(w * 0.22), int(h * 0.16)
    cv2.ellipse(cv, (cx, my), (mw, mh), 0, 0, 360, (10, 5, 20, 255), -1, cv2.LINE_AA)
    for i in range(7):
        tx = cx - mw + int((i + 0.5) * 2 * mw / 7)
        cv2.fillPoly(cv, [np.array([[tx - int(mw / 8), my - int(mh * 0.55)], [tx + int(mw / 8), my - int(mh * 0.55)],
                                    [tx, my + int(mh * 0.1)]], np.int32)], (245, 245, 245, 255), cv2.LINE_AA)
    # Etrafinda donen denklemler
    for i, eq in enumerate(("y''+4y=0", "dy/dx", "L{f(t)}", "e^(st)") if equations else ()):
        a = t * 0.9 + i * math.pi / 2
        tx, ty = cx + int(math.cos(a) * w * 0.50) - int(w * 0.07), cy + int(math.sin(a) * h * 0.62)
        cv2.putText(cv, eq, (tx, ty), FONT, max(0.32, 0.5 * s), (255, 200, 255, 255), max(1, int(s * 1.6)), cv2.LINE_AA)
    return cv


_BOY_RAW = {}


def boy_sprite(height: int) -> np.ndarray:
    """Kalem cocuk (assets/kalem_cocuk.png); dosya yoksa basit bir cizim kullanilir."""
    if "raw" not in _BOY_RAW:
        path = os.path.join(ROOT, "assets", "kalem_cocuk.png")
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
            raw = np.zeros((175, 120, 4), np.uint8)               # yedek: yesil kalemli yuvarlak kafa
            cv2.circle(raw, (80, 50), 34, (130, 190, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(raw, (20, 20), (44, 150), (60, 200, 80, 255), -1)
            cv2.fillPoly(raw, [np.array([[20, 150], [44, 150], [32, 172]], np.int32)], (150, 200, 235, 255))
        raw[raw[..., 3] == 0, :3] = 0                               # kenar halesi koyu olsun
        _BOY_RAW["raw"] = raw
    raw = _BOY_RAW["raw"]
    key = ("sz", height)
    if key not in _BOY_RAW:
        sc = height / raw.shape[0]
        _BOY_RAW[key] = cv2.resize(raw, (max(1, int(raw.shape[1] * sc)), height), interpolation=cv2.INTER_AREA)
    return _BOY_RAW[key]


SPRITES = {KIND_DOMUZ: sprite_domuz, KIND_RUZGAR: sprite_ruzgar, KIND_VIZE: sprite_vize,
           KIND_KIMYA: sprite_kimya, KIND_DEVRE: sprite_devre, KIND_TERMO: sprite_termo,
           KIND_PROJ: sprite_proj, KIND_KALEM: sprite_kalem}
LABELS = {KIND_DOMUZ: "Domuz", KIND_RUZGAR: "Gülbahçe Rüzgârı", KIND_VIZE: "Fizik 101 Vizesi",
          KIND_KIMYA: "Kimya Lab Raporu", KIND_DEVRE: "Devre Analizi", KIND_TERMO: "Termodinamik",
          KIND_KALEM: "KALEM"}


def compose_sprite(img, sprite, cx, cy, angle=0.0, scale=1.0, alpha=1.0):
    """BGRA sprite'i dondurup/olcekleyip merkezi (cx, cy) olacak sekilde karenin uzerine koyar."""
    sh, sw = sprite.shape[:2]
    if angle == 0.0 and scale == 1.0:               # hizli yol: donusturme yok
        warped, ow, oh = sprite, sw, sh
    else:
        ow = oh = int(math.hypot(sw, sh) * scale) + 4
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), angle, scale)
        M[0, 2] += ow / 2 - sw / 2
        M[1, 2] += oh / 2 - sh / 2
        warped = cv2.warpAffine(sprite, M, (ow, oh), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    H, W = img.shape[:2]
    x0, y0 = int(cx - ow / 2), int(cy - oh / 2)
    sx1, sy1, sx2, sy2 = max(0, -x0), max(0, -y0), min(ow, W - x0), min(oh, H - y0)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    part = warped[sy1:sy2, sx1:sx2]
    roi = img[y0 + sy1:y0 + sy2, x0 + sx1:x0 + sx2]
    a = part[..., 3]
    if alpha < 1.0:
        a = (a * max(0.0, alpha)).astype(np.uint8)
    a3 = cv2.merge([a, a, a])
    fg = cv2.multiply(np.ascontiguousarray(part[..., :3]), a3, scale=1 / 255.0)
    bg = cv2.multiply(roi, 255 - a3, scale=1 / 255.0)
    roi[:] = cv2.add(fg, bg)


def sprite_stamp(letter: str, size: int, color) -> np.ndarray:
    """Transkript damgasi: daire icinde harf notu (BGRA)."""
    cv = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    col = (*color, 255)
    cv2.circle(cv, (c, c), int(size * 0.44), col, max(3, size // 16), cv2.LINE_AA)
    cv2.circle(cv, (c, c), int(size * 0.35), col, max(1, size // 55), cv2.LINE_AA)
    sc, th = size / 62.0, max(2, size // 18)
    (tw, tht), _ = cv2.getTextSize(letter, FONT, sc, th)
    cv2.putText(cv, letter, (c - tw // 2, c + tht // 2), FONT, sc, col, th, cv2.LINE_AA)
    return cv
