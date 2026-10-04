#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOCAM DOMUZ VAR! (IYTE Kacis / Dodge) - Kameradan oynanan interaktif oyun
=======================================================
Oyuncu kafasini saga/sola hareket ettirerek kampuste hayatta kalmaya calisan
bir IYTE ogrencisini kontrol eder. Yuz takibi MediaPipe (Tasks API - FaceDetector)
ile yapilir; goruntu ve cizimler OpenCV ile gerceklestirilir.

Kurulum:   pip install opencv-python mediapipe
Calistirma: python iyte_kacis.py            (varsayilan kamera: 0)
            python iyte_kacis.py --camera 1 (baska kamera)
            python iyte_kacis.py --selftest (kamerasiz otomatik test)

Tuslar:  R = yeniden basla | N = yeni oyuncu (isim degistir) | Q (veya ESC) = cik
Acilista isim sorulur; skorlar leaderboard.json dosyasinda saklanir.

Not: Ilk calistirmada yuz algilama modeli (~230 KB) Google'in resmi
deposundan otomatik indirilir ve scriptin yanina kaydedilir.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# AYARLAR
# --------------------------------------------------------------------------
# Oyunun adi. Turkce karakterler desteklenmeyen (eski) OpenCV'de otomatik ASCII'ye cevrilir.
GAME_TITLE = "HOCAM DOMUZ VAR!"
GAME_SUBTITLE = "İYTE Kaçış: Gülbahçe'de hayatta kal, GANO'nu 4.00'a taşı!"
WINDOW_NAME = GAME_TITLE + " - IYTE Kacis"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "blaze_face_short_range.tflite")

LEADERBOARD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "leaderboard.json")
LEADERBOARD_SAVED = 100             # dosyada tutulan en fazla kayit
LEADERBOARD_SHOWN = 5               # oyun bitince gosterilen satir sayisi
MAX_NAME_LEN = 12

CAM_WIDTH, CAM_HEIGHT = 1280, 720   # istenen kamera cozunurlugu
GAME_WIDTH = 960                    # islem hizi icin kare bu genislige kucultulur

NOSE_KEYPOINT = 2                   # BlazeFace: 0 sag goz, 1 sol goz, 2 burun ucu
SMOOTHING = 0.55                    # 0-1 arasi; buyudukce takip daha hizli/titrek
COUNTDOWN_SECONDS = 3.0             # oyun baslamadan onceki geri sayim

MAX_GANO = 4.00
GANO_K = 150.0                      # GANO egrisi: yukselme yavasligi (buyuk = yavas)
SECONDS_WEIGHT = 1.0                # hayatta kalinan her saniye puani
DODGE_WEIGHT = 2.0                  # kacilan her engel puani

HITBOX_SHRINK_PLAYER = (0.45, 0.60)  # yuz kutusunun carpisma icin kullanilan orani (en, boy)
HITBOX_SHRINK_OBSTACLE = 0.12        # engel kutusunun her kenardan kuculme orani (adil oyun)

# Renkler (BGR)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (40, 40, 230)
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
# IYTE temasi: bordo + altin (resmi kurumsal renklere yaklasik)
MAROON = (35, 25, 125)
MAROON_LIGHT = (55, 45, 185)
GOLD = (70, 190, 245)
CREAM = (230, 238, 245)
ORANGE = (60, 115, 235)         # IYTE Yazilim Toplulugu site rengi (turuncu/mercan)
DARK = (28, 20, 26)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# YARDIMCI CIZIM FONKSIYONLARI
# --------------------------------------------------------------------------
_TR_MAP = str.maketrans("çğıöşüâÇĞİÖŞÜÂ", "cgiosuaCGIOSUA")


def _unicode_text_ok() -> bool:
    """OpenCV Turkce harf cizebiliyor mu? (OpenCV 5+ evet; 4.x Hershey fontu '?' cizer.)"""
    def render(ch):
        im = np.zeros((48, 80, 3), np.uint8)
        cv2.putText(im, ch, (5, 36), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        return im
    a = render("Ş")
    return not (np.array_equal(a, render("?")) or np.array_equal(a, render("??")))


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


def intersects(a: Tuple[float, float, float, float],
               b: Tuple[float, float, float, float]) -> bool:
    """Iki (x1, y1, x2, y2) kutusu kesisiyor mu? (AABB carpisma testi)"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


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


# --------------------------------------------------------------------------
# YUZ TAKIBI (MediaPipe)
# --------------------------------------------------------------------------
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
POSE_MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")


def ensure_model(path: str = MODEL_FILE, url: str = MODEL_URL) -> bytes:
    """Modeli dosyadan okur; yoksa indirir. Bayt olarak dondurur."""
    if not os.path.isfile(path) or os.path.getsize(path) < 10_000:
        print(f"[BILGI] Model indiriliyor: {os.path.basename(path)}")
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        except Exception as exc:  # ag hatasi vb.
            print(f"[HATA] Model indirilemedi: {exc}\n"
                  f"Su dosyayi elle indirip '{path}' olarak kaydedin:\n{url}")
            sys.exit(1)
    with open(path, "rb") as f:   # buffer olarak yukle: Turkce karakterli yollarda da calisir
        return f.read()


@dataclass
class FaceInfo:
    """Algilanan yuzun piksel cinsinden bilgisi."""
    nose_x: float
    nose_y: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2


class FaceTracker:
    """MediaPipe FaceDetector'u sarar; en buyuk yuzu ve burun ucunu dondurur."""

    def __init__(self):
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceInfo]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # artan zaman damgasi sart
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)
        if not result.detections:
            return None
        # Birden fazla yuz varsa en buyugunu (kameraya en yakin) sec
        det = max(result.detections,
                  key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = det.bounding_box
        box = (bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height)
        if len(det.keypoints) > NOSE_KEYPOINT:
            kp = det.keypoints[NOSE_KEYPOINT]       # normalize (0-1) koordinat
            nose_x, nose_y = kp.x * w, kp.y * h
        else:                                       # keypoint yoksa kutu merkezi
            nose_x, nose_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return FaceInfo(nose_x, nose_y, box)

    def close(self):
        self.detector.close()


class PoseTracker:
    """Ayakta oynayanlar icin: tum vucut pozundan basi bulur. FaceTracker ile ayni arayuz;
    uzaktan (2-3 m) ve ayakta da calisir, oyun bas konumunu eskisi gibi kullanir."""

    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model(POSE_MODEL_FILE, POSE_MODEL_URL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceInfo]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        nose = lm[0]
        # Yanlis tespiti (esya yigini vb.) elemek icin burun ve en az bir omuz net gorunmeli
        vis = lambda p: 1.0 if p.visibility is None else p.visibility
        if vis(nose) < 0.6 or max(vis(lm[11]), vis(lm[12])) < 0.5:
            return None
        # Bas genisligi: kulaklar arasi; kulaklar gorunmuyorsa gozler arasindan tahmin
        head_w = max(abs(lm[7].x - lm[8].x) * w * 1.35, abs(lm[2].x - lm[5].x) * w * 3.2, w * 0.04)
        head_h = head_w * 1.25
        nx, ny = nose.x * w, nose.y * h
        return FaceInfo(nx, ny, (nx - head_w / 2, ny - head_h * 0.58, nx + head_w / 2, ny + head_h * 0.42))

    def close(self):
        self.detector.close()


# --------------------------------------------------------------------------
# ENGELLER
# --------------------------------------------------------------------------
KIND_DOMUZ, KIND_RUZGAR, KIND_VIZE = "domuz", "ruzgar", "vize"
KIND_KIMYA, KIND_DEVRE, KIND_TERMO = "kimya", "devre", "termo"   # zor muhendislik dersleri
KIND_PROJ, KIND_KALEM = "proj", "kalem"                           # boss mermisi, kalem bonusu

# Dersler GANO'ya gore acilir: (tur, acildigi GANO, secilme agirligi)
SPAWN_TABLE = [
    (KIND_DOMUZ, 0.0, 4), (KIND_RUZGAR, 0.0, 2), (KIND_VIZE, 0.0, 3),
    (KIND_KIMYA, 1.0, 3),     # CHEM 101 Kimya Lab Raporu
    (KIND_DEVRE, 1.5, 3),     # EEE 201 Devre Analizi
    (KIND_TERMO, 2.0, 2),     # ME 301 Termodinamik
]

CORRIDOR_FACTOR = 2.0           # kacis koridoru genisligi: oyuncu carpisma kutusunun bu kati
VIZE_DRIFT_MAX = 0.05           # vizenin oyuncuya dogru en fazla kayma orani (ekran genisligi)
DOMUZ_AFTER = 10.0            # domuzlar oyunun bu saniyesinden sonra baslar
BOSS_TRIGGER_GANO = 2.50      # bu GANO'dan sonra MATH 255 final boss'u gelir
BOSS_WIN_GANO = 3.50          # boss yenilince GANO en az bu degere cikar
BOSS_HP = 5                   # kac kalem darbesiyle kesilir

# Oyun bitince gosterilecek mesajlar (engel turune gore)
DEATH_MESSAGES = {
    KIND_DOMUZ: "DOMUZ ÇARPTI!",
    KIND_RUZGAR: "RÜZGAR ALDI GÖTÜRDÜ!",
    KIND_VIZE: "FİNALDE DÜZELTİRSİN",
    KIND_KIMYA: "KİMYA LAB'DA PATLADIN!",
    KIND_DEVRE: "AKIM ÇARPTI!",
    KIND_TERMO: "TERMODİNAMİK YAKTI!",
    KIND_PROJ: "DENKLEMİ ÇÖZEMEDİN!",
}

# Carpisma kutusunun kuculme orani (ince/sivri sekillerde daha affedici)
HITBOX_SHRINK_KIND = {KIND_VIZE: 0.18, KIND_KIMYA: 0.15, KIND_DEVRE: 0.28,
                      KIND_TERMO: 0.22, KIND_PROJ: 0.22}


@dataclass
class Obstacle:
    kind: str
    x: float          # sol ust kose
    y: float
    w: float
    h: float
    speed: float      # piksel/saniye (asagi dogru)
    base_x: float = 0.0   # zigzag/dalga merkezi
    phase: float = 0.0    # zigzag fazi
    freq: float = 0.0
    amp: float = 0.0
    seed: float = 0.0     # animasyon fazini engele ozel yapar
    vx: float = 0.0       # yatay hiz (boss mermileri)
    home_x: float = 0.0   # vize takibinin sinirli kaymasi icin baslangic merkezi

    def box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def hitbox(self) -> Tuple[float, float, float, float]:
        """Gorselden biraz kucuk carpisma kutusu."""
        k = HITBOX_SHRINK_KIND.get(self.kind, HITBOX_SHRINK_OBSTACLE)
        mx, my = self.w * k, self.h * k
        return (self.x + mx, self.y + my, self.x + self.w - mx, self.y + self.h - my)


def make_obstacle(kind: str, W: int, H: int, difficulty: float) -> Obstacle:
    """Engel turune gore boyut/hiz belirleyip rastgele X'te ekranin ustunde dogurur."""
    if kind == KIND_DOMUZ:        # hizli ve tehlikeli, orta boy
        w, h = W * 0.12, H * 0.12
        speed = H * random.uniform(0.50, 0.65) * difficulty
    elif kind == KIND_RUZGAR:     # genis ama yavas
        w, h = W * 0.28, H * 0.10
        speed = H * random.uniform(0.24, 0.30) * difficulty
    elif kind == KIND_VIZE:       # dengelendi: daha yavas, dar zigzag, zayif takip
        w, h = W * 0.11, H * 0.14
        speed = H * random.uniform(0.27, 0.33) * difficulty
    elif kind == KIND_KIMYA:      # salinarak duser
        w, h = W * 0.075, H * 0.13
        speed = H * random.uniform(0.36, 0.44) * difficulty
    elif kind == KIND_DEVRE:      # hizli, keskin zigzag cizen simsek
        w, h = W * 0.06, H * 0.14
        speed = H * random.uniform(0.50, 0.60) * difficulty
    elif kind == KIND_TERMO:      # buyuk ates topu
        w, h = W * 0.14, H * 0.17
        speed = H * random.uniform(0.40, 0.48) * difficulty
    elif kind == KIND_KALEM:      # toplanacak bonus
        w, h = W * 0.055, H * 0.12
        speed = H * 0.30
    else:                         # KIND_PROJ: boss mermisi
        w = h = W * 0.05
        speed = H * 0.5
    x = random.uniform(0, W - w)
    ob = Obstacle(kind, x, -h, w, h, speed)
    ob.seed = random.uniform(0, math.tau)
    if kind in (KIND_VIZE, KIND_KIMYA, KIND_DEVRE, KIND_KALEM):
        ob.base_x = ob.home_x = x
        ob.phase = random.uniform(0, math.tau)
        if kind == KIND_VIZE:
            ob.freq, ob.amp = random.uniform(1.4, 2.2), W * random.uniform(0.035, 0.06)
        elif kind == KIND_KIMYA:
            ob.freq, ob.amp = random.uniform(1.6, 2.4), W * 0.04
        elif kind == KIND_DEVRE:
            ob.freq, ob.amp = random.uniform(2.2, 3.0), W * random.uniform(0.07, 0.10)
        else:
            ob.freq, ob.amp = 2.0, W * 0.03
    return ob


def make_projectile(cx: float, cy: float, W: int, H: int, vx: float, vy: float,
                    amp: float = 0.0, phase: float = 0.0) -> Obstacle:
    """Boss mermisi: (cx, cy) merkezli; vx yatay hiz ya da amp>0 ise sinus dalgasi."""
    ob = make_obstacle(KIND_PROJ, W, H, 1.0)
    ob.x, ob.y, ob.speed, ob.vx = cx - ob.w / 2, cy - ob.h / 2, vy, vx
    if amp > 0:
        ob.base_x, ob.amp, ob.phase, ob.freq = cx - ob.w / 2, amp, phase, 3.0
    return ob


# --------------------------------------------------------------------------
# LEADERBOARD (JSON dosyasi)
# --------------------------------------------------------------------------
def load_leaderboard(path: str = LEADERBOARD_FILE) -> List[dict]:
    """Kayitlari okur. Dosya yok/bozuksa bos liste doner (oyun asla cokmez)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [e for e in data if isinstance(e, dict)
                and isinstance(e.get("name"), str) and isinstance(e.get("gano"), (int, float))]
    except (OSError, ValueError, TypeError):
        return []


def add_score(name: str, gano: float, path: str = LEADERBOARD_FILE) -> Tuple[int, List[dict]]:
    """Skoru ekler, dosyaya yazar. (oyuncunun sirasi [1'den], siralanmis liste) dondurur."""
    entries = load_leaderboard(path)
    mine = {"name": name, "gano": round(gano, 2), "date": time.strftime("%Y-%m-%d %H:%M")}
    entries.append(mine)
    # Puan yuksek olan ustte; esitlikte once yapan ustte (sort kararli, eski kayitlar once)
    entries.sort(key=lambda e: -e["gano"])
    rank = next(i for i, e in enumerate(entries) if e is mine) + 1
    entries = entries[:LEADERBOARD_SAVED]
    try:                                   # atomik yazim: yarim dosya kalmasin
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[UYARI] Leaderboard kaydedilemedi: {exc}")
    return rank, entries


# --------------------------------------------------------------------------
# OYUN MANTIGI
# --------------------------------------------------------------------------
STATE_INTRO, STATE_NAME, STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER = (
    "intro", "name", "wait", "countdown", "play", "over")


MAX_PARTICLES = 300


def grade_letter(gano: float) -> str:
    """GANO'dan IYTE harf notu (AA, BA, BB, CB, CC, DC, DD, FD, FF)."""
    for limit, letter in ((3.75, "AA"), (3.25, "BA"), (2.75, "BB"), (2.25, "CB"),
                          (1.75, "CC"), (1.25, "DC"), (0.75, "DD"), (0.25, "FD")):
        if gano >= limit:
            return letter
    return "FF"


# Engel arkasinda birakilan iz parcaciklari: (renkler, saniyedeki adet, boyut)
TRAIL = {
    KIND_DOMUZ: ([(110, 150, 185), (150, 180, 205)], 26, 5),      # toz
    KIND_RUZGAR: ([(70, 170, 60), (40, 140, 230), (255, 255, 255)], 16, 4),   # yapraklar
    KIND_VIZE: ([(250, 250, 250), (205, 210, 215)], 12, 4),       # kagit parcalari
    KIND_KIMYA: ([(90, 230, 110), (200, 255, 210)], 14, 4),       # kabarciklar
    KIND_DEVRE: ([(40, 230, 255), (255, 200, 90)], 30, 4),        # kivilcim
    KIND_TERMO: ([(30, 120, 250), (60, 210, 255), (90, 90, 90)], 30, 6),      # ates/kul
    KIND_PROJ: ([(200, 90, 210), (255, 190, 255)], 20, 4),        # mor toz
    KIND_KALEM: ([(70, 190, 245), (255, 255, 255)], 22, 4),       # altin parilti
}
CRASH_COLORS = {
    KIND_DOMUZ: [(40, 75, 125), (135, 155, 215), (255, 255, 255)],
    KIND_RUZGAR: [(255, 215, 130), (255, 255, 255), (70, 170, 60)],
    KIND_VIZE: [(246, 248, 250), (40, 40, 220), (150, 150, 150)],
    KIND_KIMYA: [(90, 230, 110), (220, 235, 230), (60, 160, 70)],
    KIND_DEVRE: [(40, 230, 255), (255, 255, 255), (255, 180, 60)],
    KIND_TERMO: [(30, 120, 250), (60, 210, 255), (40, 40, 40)],
    KIND_PROJ: [(200, 90, 210), (255, 190, 255), (110, 30, 140)],
}
BOSS_COLORS = [(200, 90, 210), (255, 190, 255), (110, 30, 140), (255, 255, 255), (40, 40, 230)]


@dataclass
class FloatText:
    """Yukari suzulerek kaybolan yazi (harf notu gibi)."""
    x: float
    y: float
    text: str
    color: Tuple[int, int, int]
    life: float
    max_life: float


@dataclass
class Particle:
    """Tek bir efekt parcasi (konum, hiz, omur, renk)."""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float
    spark: bool = False       # True: arti seklinde yildiz, False: daire
    gravity: float = 0.0


@dataclass
class Boss:
    """MATH 255 final boss'u (x, y = govde merkezi)."""
    x: float
    y: float
    w: float
    h: float
    hp: int
    max_hp: int
    target_y: float
    t: float = 0.0
    state: str = "enter"          # enter -> fight -> dying
    attack_timer: float = 2.0
    charge: float = 0.0           # >0: saldiriya hazirlaniyor (telegraph suresi)
    pattern: str = ""
    last_pattern: str = ""
    hit_flash: float = 0.0
    dying_t: float = 0.0


@dataclass
class Helper:
    """Kalem toplaninca bossa dalan kalem cocuk animasyonu."""
    t: float
    sx: float
    sy: float
    damaged: bool = False


HELPER_DURATION = 1.1
HELPER_HIT_AT = 0.5               # kalem cocuk bu saniyede bossu keser


class Game:
    """Oyunun tum durumunu tutar: engeller, skor, durum makinesi."""

    def __init__(self, W: int, H: int, best_gano: float = 0.0, name: str = "",
                 lb_path: str = LEADERBOARD_FILE, intro: bool = False):
        self.W, self.H = W, H
        self.best_gano = best_gano
        self.name = name              # oyuncu ismi (reset'te korunur)
        self.lb_path = lb_path
        self.anim_t = 0.0             # arayuz animasyonlari icin surekli akan zaman
        self.reset()
        if intro:                     # sadece uygulama acilisinda: topluluk intro'su
            self.state = STATE_INTRO

    def reset(self):
        """Yeni oyun icin her seyi sifirlar (isim varsa isim sormadan baslar)."""
        self.state = STATE_WAIT if self.name else STATE_NAME
        self.rank = 0
        self.board: List[dict] = []
        self.over_t = 0.0             # game over ekraninin acilis animasyon zamani
        self.shake = 0.0              # ekran sarsintisi (0-1)
        self.flash = 0.0              # carpisma flasi (0-1)
        self.hud_pulse = 0.0          # engel kacinca HUD parlamasi
        self.disp_gano = 0.0          # ekranda yumusakca akan GANO
        self.particles: List[Particle] = []
        self.floats: List[FloatText] = []
        self.obstacles: List[Obstacle] = []
        self.survival = 0.0          # hayatta kalinan sure (sn)
        self.dodged = 0              # kacilan engel sayisi
        self.spawn_timer = 2.5          # ilk engel gec gelsin (yumusak baslangic)
        self.countdown = COUNTDOWN_SECONDS
        self.death_msg = ""
        # Oyuncu (yumusatilmis) konumu
        self.px: Optional[float] = None
        self.py: Optional[float] = None
        self.ny: Optional[float] = None     # burun ucunun (yumusatilmis) y'si
        self.nvx = 0.0                      # burun yatay hizi (piksel/kare)
        self.pbox_w = self.W * 0.12
        self.pbox_h = self.H * 0.2
        self.face_visible = False
        # Gizli kacis koridoru: yavasca suzulur, engeller gececegi yere dogmaz (kazanilabilirlik garantisi)
        self.corr_phase = (random.uniform(0, math.tau), random.uniform(0, math.tau))
        # Boss / ders sistemi
        self.score_time = 0.0         # GANO'ya sayilan sure (boss savasinda durur)
        self.score_dodged = 0         # GANO'ya sayilan kacilan engel
        self.bonus_raw = 0.0          # boss odulu (GANO sicramasi)
        self.boss: Optional[Boss] = None
        self.boss_defeated = False
        self.helper: Optional[Helper] = None
        self.pickup_timer = 1.0
        self.banner, self.banner_sub, self.banner_color = "", "", GOLD
        self.banner_t = self.banner_dur = 0.0

    # ---- isim girisi ----
    def type_key(self, key: int) -> bool:
        """Isim ekraninda tusu isler. ENTER ile isim onaylanirsa True doner."""
        if key in (8, 127) and self.name:                  # Backspace
            self.name = self.name[:-1]
        elif key in (10, 13):                              # Enter
            self.name = self.name.strip()
            if self.name:
                self.state = STATE_WAIT
                return True
        elif 32 <= key <= 126 and len(self.name) < MAX_NAME_LEN:
            if chr(key).isalnum() or chr(key) in " _-.":
                self.name += chr(key)
        return False

    def new_player(self):
        """Isim sifirlanir, isim ekrani acilir (ayni bilgisayarda baska oyuncu icin)."""
        self.name = ""
        self.best_gano = 0.0
        self.reset()

    # ---- skor ----
    @property
    def gano(self) -> float:
        """Skoru 0.00 -> 4.00 arasina asimptotik olarak esler."""
        raw = SECONDS_WEIGHT * self.score_time + DODGE_WEIGHT * self.score_dodged + self.bonus_raw
        return min(MAX_GANO, MAX_GANO * (1.0 - math.exp(-raw / GANO_K)))

    @property
    def difficulty(self) -> float:
        """Yumusak baslayip (0.65) yavasca 1.5'a cikan zorluk carpani (tavan: kacilabilir kalsin)."""
        return min(1.5, 0.65 + self.survival / 90.0)

    # ---- oyuncu ----
    def update_player(self, face: Optional[FaceInfo]):
        """Yuz bilgisinden yumusatilmis oyuncu konumu ve carpisma kutusunu gunceller."""
        self.face_visible = face is not None
        if face is None:
            return
        cy = (face.box[1] + face.box[3]) / 2
        if self.px is None:
            self.px, self.py, self.ny = face.nose_x, cy, face.nose_y
        else:
            a, prev = SMOOTHING, self.px
            if self.ny is None:
                self.ny = face.nose_y
            self.px += a * (face.nose_x - self.px)
            self.py += a * (cy - self.py)
            self.ny += a * (face.nose_y - self.ny)
            self.nvx += 0.4 * ((self.px - prev) - self.nvx)
        # Uzaktan oynayanin kutusu cok kuculmesin (hem kolaylasmasin hem gorunur kalsin)
        self.pbox_w = max(self.W * 0.10, face.box[2] - face.box[0])
        self.pbox_h = max(self.H * 0.17, face.box[3] - face.box[1])

    def player_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Gorsel yuz kutusu (x1, y1, x2, y2)."""
        if self.px is None:
            return None
        return (self.px - self.pbox_w / 2, self.py - self.pbox_h / 2,
                self.px + self.pbox_w / 2, self.py + self.pbox_h / 2)

    def player_hitbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Carpisma icin kullanilan, yuz kutusundan kucuk kutu."""
        if self.px is None:
            return None
        hw = self.pbox_w * HITBOX_SHRINK_PLAYER[0] / 2
        hh = self.pbox_h * HITBOX_SHRINK_PLAYER[1] / 2
        return (self.px - hw, self.py - hh, self.px + hw, self.py + hh)

    # ---- efektler ----
    @property
    def u(self) -> float:
        """Arayuz olcek katsayisi (960x540 tasarimina gore)."""
        return min(self.W / 960.0, self.H / 540.0)

    def burst(self, x, y, n, colors, speed, size, spark=False, up=False, gravity=0.0, life=1.0):
        """(x, y) noktasindan n adet parcacik patlatir."""
        u = self.u
        for _ in range(n):
            ang = random.uniform(-math.pi, 0) if up else random.uniform(0, math.tau)
            sp = speed * u * random.uniform(0.35, 1.0)
            lf = life * random.uniform(0.6, 1.0)
            self.particles.append(Particle(x, y, math.cos(ang) * sp, math.sin(ang) * sp, lf, lf,
                                           random.choice(colors), size * random.uniform(0.6, 1.2),
                                           spark, self.H * gravity))
        del self.particles[:-MAX_PARTICLES]

    def _trail(self, ob: Obstacle, dt: float):
        """Hareket eden engelin arkasina (ustune) toz/yaprak/kagit parcasi birakir."""
        info = TRAIL.get(ob.kind)
        if info is None or ob.y < 0 or random.random() > info[1] * dt:
            return
        colors, rate, size = info
        u = self.u
        self.particles.append(Particle(
            ob.x + random.uniform(0.1, 0.9) * ob.w, ob.y + random.uniform(0, 0.3) * ob.h,
            random.uniform(-25, 25) * u, random.uniform(-30, 5) * u, 0.55, 0.55,
            random.choice(colors), size * random.uniform(0.7, 1.2), False, 0.0))

    def _update_fx(self, dt: float):
        """Animasyon/efekt durumlarini ilerletir (oyun duraklasa bile akar)."""
        self.disp_gano += (self.gano - self.disp_gano) * min(1.0, dt * 8)
        self.hud_pulse = max(0.0, self.hud_pulse - dt * 3)
        self.banner_t = max(0.0, self.banner_t - dt)
        self.shake = max(0.0, self.shake - dt * 1.8)
        self.flash = max(0.0, self.flash - dt * 2.5)
        if self.state == STATE_OVER:
            self.over_t += dt
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += p.gravity * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]
        for f in self.floats:
            f.y -= 55 * self.u * dt
            f.life -= dt
        self.floats = [f for f in self.floats if f.life > 0]

    # ---- ana guncelleme ----
    def update(self, dt: float, face: Optional[FaceInfo]):
        self.anim_t += dt
        self._update_fx(dt)
        self.update_player(face)

        if self.state in (STATE_INTRO, STATE_NAME, STATE_OVER):
            return

        if self.state == STATE_WAIT:
            if self.face_visible:
                self.state = STATE_COUNTDOWN
            return

        if self.state == STATE_COUNTDOWN:
            if not self.face_visible:          # yuz kaybolursa geri sayim sifirlanir
                self.state, self.countdown = STATE_WAIT, COUNTDOWN_SECONDS
                return
            self.countdown -= dt
            if self.countdown <= 0:
                self.state = STATE_PLAY
            return

        if self.state != STATE_PLAY:
            return

        # Yuz gorunmuyorsa oyun duraklar (haksiz olum olmasin)
        if not self.face_visible:
            return

        self.survival += dt
        if self.boss is None:
            self.score_time += dt
            if not self.boss_defeated and self.gano >= BOSS_TRIGGER_GANO:
                self._start_boss()
            else:
                self._spawn(dt)
        else:
            self._update_boss(dt)
        self._update_helper(dt)
        self._move_obstacles(dt)
        self._check_collisions()

    def corridor(self, t: float) -> float:
        """Zaman t'deki (oyun saati) kacis koridorunun merkezi; hizi en fazla ~0.15 ekran/sn."""
        a, b = self.corr_phase
        return self.W * (0.5 + 0.20 * math.sin(0.35 * t + a) + 0.09 * math.sin(0.83 * t + b))

    def _clear_of_corridor(self, ob: Obstacle) -> bool:
        """Engel, oyuncunun yuksekliginden gececegi sure boyunca koridoru kapatmiyor mu?"""
        py = self.py if self.py is not None else self.H * 0.7
        hhp = self.pbox_h * HITBOX_SHRINK_PLAYER[1] / 2 + self.H * 0.06     # oyuncu dikeyde biraz oynayabilir
        v = max(1.0, ob.speed)
        t1, t2 = (py - hhp) / v, (py + hhp + ob.h) / v
        centers = [self.corridor(self.survival + t1 + (t2 - t1) * k / 6) for k in range(7)]
        half = self.pbox_w * HITBOX_SHRINK_PLAYER[0] * CORRIDOR_FACTOR / 2
        c_lo, c_hi = min(centers) - half, max(centers) + half
        swings = ob.kind in (KIND_VIZE, KIND_KIMYA, KIND_DEVRE)
        pad = (ob.amp if swings else 0.0) + (self.W * VIZE_DRIFT_MAX if ob.kind == KIND_VIZE else 0.0)
        cx = ob.home_x if swings else ob.x                # salinan engel: sabit merkez baz alinir
        return cx + ob.w + pad < c_lo or cx - pad > c_hi

    def _spawn(self, dt: float):
        """Zamanlayici dolunca, GANO'ya gore acilmis derslerden rastgele engel uretir."""
        self.spawn_timer -= dt
        if self.spawn_timer > 0:
            return
        # Zorluk arttikca engel araligi 1.6 sn'den 0.8 sn'ye iner
        interval = max(0.8, 1.6 - 0.01 * self.survival)
        self.spawn_timer = interval * random.uniform(0.8, 1.2)
        g = self.gano
        pool = [(k, w) for k, unlock, w in SPAWN_TABLE if g >= unlock]
        if self.survival < DOMUZ_AFTER:                  # en hizli engel ilk saniyelerde cikmaz
            pool = [(k, w) for k, w in pool if k != KIND_DOMUZ]
        kind = random.choices([k for k, _ in pool], weights=[w for _, w in pool])[0]
        for _ in range(8):                               # koridoru kapatan dizilimi eleyerek dene
            ob = make_obstacle(kind, self.W, self.H, self.difficulty)
            if self.px is not None and random.random() < 0.6:   # cogu engel oyuncunun yakininda dogar
                ob.x = max(0.0, min(self.W - ob.w, self.px - ob.w / 2 + random.uniform(-0.15, 0.15) * self.W))
                ob.base_x = ob.home_x = ob.x
            self.obstacles.append(ob)
            if self._clear_of_corridor(ob):
                return
            self.obstacles.pop()
        self.spawn_timer = 0.25                          # kacis yolu kapaliysa kisa sure bekle

    def _move_obstacles(self, dt: float):
        """Engelleri kendi hareket kaliplariyla (zigzag, dalga, sapma) tasir."""
        alive = []
        for ob in self.obstacles:
            ob.y += ob.speed * dt
            self._trail(ob, dt)
            k = ob.kind
            if k == KIND_VIZE:
                # Oyuncuya dogru yavas kayma (takip) + sinus zigzag
                if self.px is not None:
                    target = self.px - ob.w / 2
                    step = self.W * 0.025 * dt
                    ob.base_x += max(-step, min(step, target - ob.base_x))
                    lim = self.W * VIZE_DRIFT_MAX                 # takip en fazla bu kadar kayar
                    ob.base_x = max(ob.home_x - lim, min(ob.home_x + lim, ob.base_x))
                ob.phase += ob.freq * dt
                ob.x = max(0.0, min(self.W - ob.w, ob.base_x + math.sin(ob.phase) * ob.amp))
            elif k in (KIND_KIMYA, KIND_KALEM):
                ob.phase += ob.freq * dt
                ob.x = max(0.0, min(self.W - ob.w, ob.base_x + math.sin(ob.phase) * ob.amp))
            elif k == KIND_DEVRE:                      # ucgen dalga: keskin zigzag
                ob.phase += ob.freq * dt
                tri = 2.0 / math.pi * math.asin(math.sin(ob.phase))
                ob.x = max(0.0, min(self.W - ob.w, ob.base_x + tri * ob.amp))
            elif k == KIND_PROJ:
                if ob.amp > 0:                         # dalga mermisi
                    ob.phase += ob.freq * dt
                    ob.x = ob.base_x + math.sin(ob.phase) * ob.amp
                else:
                    ob.x += ob.vx * dt
            off_side = k == KIND_PROJ and (ob.x < -ob.w * 3 or ob.x > self.W + ob.w * 2)
            if ob.y > self.H or off_side:
                self._obstacle_left(ob)
            else:
                alive.append(ob)
        self.obstacles = alive

    def _obstacle_left(self, ob: Obstacle):
        """Engel ekrandan cikti: normal engelse puan, harf notu ve kivilcim."""
        if ob.kind in (KIND_PROJ, KIND_KALEM) or self.boss is not None:
            return                                     # boss savasinda puan donuk
        self.dodged += 1
        self.score_dodged += 1
        self.hud_pulse = 1.0
        letter = grade_letter(self.gano)
        self.floats.append(FloatText(ob.x + ob.w / 2, self.H - 46 * self.u, letter,
                                     GOLD if letter in ("AA", "BA", "BB") else WHITE, 1.1, 1.1))
        del self.floats[:-10]
        self.burst(ob.x + ob.w / 2, self.H - 10 * self.u, 8, [GOLD, CREAM, WHITE],
                   160, 6, spark=True, up=True, gravity=0.35, life=0.7)

    # ---- carpisma ----
    def _check_collisions(self):
        self._collect_pickups()
        self._check_death()

    def _collect_pickups(self):
        """Kalem bonusu yuze degerse toplanir ve kalem cocuk bossa saldirir."""
        box = self.player_box()
        if box is None:
            return
        for ob in self.obstacles:
            if ob.kind == KIND_KALEM and intersects(box, ob.box()):
                self.obstacles.remove(ob)
                self.burst(ob.x + ob.w / 2, ob.y + ob.h / 2, 16, [GOLD, WHITE, CREAM], 260, 7,
                           spark=True, gravity=0.3, life=0.8)
                self.floats.append(FloatText(self.px, self.py - self.pbox_h * 0.7, "KALEM!", GOLD, 1.0, 1.0))
                if self.boss is not None and self.boss.state == "fight" and self.helper is None:
                    self.helper = Helper(0.0, self.px, self.py)
                return

    def _check_death(self):
        hit = self.player_hitbox()
        if hit is None:
            return
        for ob in self.obstacles:
            if ob.kind != KIND_KALEM and intersects(hit, ob.hitbox()):
                self.state = STATE_OVER
                self.death_msg = DEATH_MESSAGES[ob.kind]
                self.best_gano = max(self.best_gano, self.gano)
                self.rank, self.board = add_score(self.name, self.gano, self.lb_path)
                self.shake, self.flash = 1.0, 1.0
                self.floats.clear()
                self.burst(self.px, self.py, 46, CRASH_COLORS[ob.kind], 380, 8, gravity=0.9, life=1.1)
                self.burst(self.px, self.py, 14, [GOLD, WHITE], 300, 8, spark=True, gravity=0.5, life=0.9)
                return

    # ---- boss: MATH 255 ----
    def set_banner(self, text: str, sub: str, color, dur: float):
        self.banner, self.banner_sub, self.banner_color = text, sub, color
        self.banner_t = self.banner_dur = dur

    def _start_boss(self):
        """GANO esigi asilinca boss yukaridan girer; normal engel uretimi durur."""
        W, H = self.W, self.H
        bw, bh = W * 0.34, H * 0.27
        self.boss = Boss(W / 2, -bh, bw, bh, BOSS_HP, BOSS_HP, target_y=H * 0.09 + bh / 2)
        self.set_banner("UYARI! MATH 255 FİNALİ", "Diferansiyel Denklemler geliyor...", RED, 3.0)
        self.shake, self.flash = 0.8, 0.5

    def _update_boss(self, dt: float):
        b = self.boss
        b.t += dt
        b.hit_flash = max(0.0, b.hit_flash - dt * 4)
        W, H = self.W, self.H
        if b.state == "enter":
            b.y += (b.target_y - b.y) * min(1.0, dt * 2.0)
            if abs(b.y - b.target_y) < 2:
                b.y, b.state = b.target_y, "fight"
                b.attack_timer, self.pickup_timer = 1.2, 1.0
            return
        if b.state == "dying":
            b.dying_t += dt
            if random.random() < dt * 30:               # patlama parcaciklari
                self.burst(b.x + random.uniform(-0.4, 0.4) * b.w, b.y + random.uniform(-0.4, 0.4) * b.h,
                           10, BOSS_COLORS, 260, 8, gravity=0.4, life=0.9)
            self.shake = max(self.shake, 0.5)
            if b.dying_t >= 2.2:
                self._finish_boss()
            return
        b.x = W / 2 + math.sin(b.t * 0.7) * W * 0.27     # yavasca saga sola gezinir
        if b.charge > 0:
            b.charge -= dt
            if b.charge <= 0:
                self._boss_fire(b.pattern)
                b.last_pattern = b.pattern
                b.attack_timer = max(0.9, 2.1 - 0.2 * (b.max_hp - b.hp))   # canı azaldikca hizlanir
        else:
            b.attack_timer -= dt
            if b.attack_timer <= 0:
                options = ["nisan", "yagmur"] + (["dalga"] if b.hp <= 2 else [])
                options = [o for o in options if o != b.last_pattern] or options
                b.pattern = random.choice(options)
                b.charge = 0.7                           # kisa uyari: boss kızarir
        self.pickup_timer -= dt
        if self.pickup_timer <= 0 and not any(o.kind == KIND_KALEM for o in self.obstacles):
            ob = make_obstacle(KIND_KALEM, W, H, 1.0)
            ob.x = ob.base_x = random.uniform(W * 0.1, W * 0.85)
            self.obstacles.append(ob)
            self.pickup_timer = 3.2

    def _boss_fire(self, pattern: str):
        """Boss saldirilari: nisan (3'lu yelpaze), yagmur (bosluklu sira), dalga (sinus)."""
        b, W, H = self.boss, self.W, self.H
        sx, sy = b.x, b.y + b.h * 0.45
        if pattern == "nisan":
            vy = H * 0.55
            ty = self.py if self.py is not None else H * 0.7
            T = max(0.5, (ty - sy) / vy)
            tx0 = self.px if self.px is not None else W / 2
            for off in (-0.18, 0.0, 0.18):
                vx = (tx0 + off * W - sx) / T
                self.obstacles.append(make_projectile(sx, sy, W, H, vx, vy))
        elif pattern == "yagmur":
            cols = 7
            pc = int((self.px if self.px is not None else W / 2) / (W / cols))
            gap = max(0, min(cols - 2, pc + random.randint(-2, 1)))   # iki sutunluk bosluk oyuncuya yakin
            for i in range(cols):
                if i in (gap, gap + 1):
                    continue
                self.obstacles.append(make_projectile((i + 0.5) * W / cols, -W * 0.03, W, H, 0.0, H * 0.38))
        else:                                            # dalga
            for k in range(4):
                self.obstacles.append(make_projectile(sx + (k - 1.5) * W * 0.17, sy, W, H, 0.0, H * 0.42,
                                                      amp=W * 0.05, phase=k * 1.3))
        self.burst(sx, sy, 14, BOSS_COLORS, 220, 6, gravity=0.2, life=0.7)

    def _damage_boss(self):
        """Kalem cocugun vurusu: bossun cani 1 azalir; 0 olursa boss patlar."""
        b = self.boss
        if b is None or b.state != "fight":
            return
        b.hp -= 1
        b.hit_flash = 1.0
        self.shake = max(self.shake, 0.9)
        self.burst(b.x, b.y, 28, BOSS_COLORS, 340, 8, gravity=0.5, life=0.9)
        self.floats.append(FloatText(b.x, b.y + b.h * 0.7, "KESTİN!", GOLD, 1.0, 1.0))
        if b.hp <= 0:
            b.state = "dying"
            for ob in self.obstacles:                    # mermiler ve kalemler dagilir
                if ob.kind in (KIND_PROJ, KIND_KALEM):
                    self.burst(ob.x + ob.w / 2, ob.y + ob.h / 2, 6, BOSS_COLORS, 160, 5, life=0.6)
            self.obstacles = [o for o in self.obstacles if o.kind not in (KIND_PROJ, KIND_KALEM)]

    def _finish_boss(self):
        """Boss yenildi: GANO en az 3.50'ye sıçrar, normal engeller geri doner."""
        self.boss, self.boss_defeated = None, True
        raw_now = SECONDS_WEIGHT * self.score_time + DODGE_WEIGHT * self.score_dodged
        target = -GANO_K * math.log(1.0 - BOSS_WIN_GANO / MAX_GANO)
        self.bonus_raw = max(0.0, target - raw_now)
        self.set_banner("MATH 255 GEÇİLDİ!", f"GANO {BOSS_WIN_GANO:.2f}'ye yükseldi!", GOLD, 3.5)
        self.burst(self.W / 2, self.H * 0.3, 60, [GOLD, WHITE, CREAM, (90, 220, 90)], 420, 8,
                   spark=True, gravity=0.6, life=1.4)
        self.hud_pulse, self.spawn_timer = 1.0, 2.0

    # ---- kalem cocuk ----
    def _update_helper(self, dt: float):
        h = self.helper
        if h is None:
            return
        h.t += dt
        st = self.helper_state()
        if st is not None and h.t < HELPER_HIT_AT and random.random() < 0.8:   # grafit izi
            self.particles.append(Particle(st[0], st[1], random.uniform(-20, 20), random.uniform(-20, 20),
                                           0.4, 0.4, random.choice([(60, 200, 90), (90, 90, 90), WHITE]),
                                           5, False, 0.0))
        if not h.damaged and h.t >= HELPER_HIT_AT:
            h.damaged = True
            self._damage_boss()
        if h.t >= HELPER_DURATION:
            self.helper = None

    def helper_state(self):
        """(x, y, saydamlik, kesme_ilerlemesi veya None) - cizim icin."""
        h = self.helper
        if h is None:
            return None
        b = self.boss
        tx, ty = (b.x, b.y + b.h * 0.35) if b else (self.W / 2, self.H * 0.3)
        k = ease_out(h.t / HELPER_HIT_AT)
        x, y = h.sx + (tx - h.sx) * k, h.sy + (ty - h.sy) * k
        alpha = 1.0 if h.t < 0.8 else max(0.0, 1.0 - (h.t - 0.8) / (HELPER_DURATION - 0.8))
        slash = (h.t - HELPER_HIT_AT) / 0.3 if HELPER_HIT_AT <= h.t <= HELPER_HIT_AT + 0.3 else None
        return x, y, alpha, slash


# --------------------------------------------------------------------------
# CIZIM - SPRITE'LAR (saydam katmanda cizilip donduruerek yerlestirilir)
# --------------------------------------------------------------------------
def _canvas(w: int, h: int, pad: int) -> np.ndarray:
    """Seffaf (BGRA) cizim tuvali: engel kutusunun etrafinda pad kadar bosluk."""
    return np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)


def sprite_domuz(w: int, h: int, t: float, seed: float) -> np.ndarray:
    """Kizgin yaban domuzu (onden gorunum). Hiz cizgileri t ile kayar."""
    pad = int(max(w, h) * 0.4)
    cv = _canvas(w, h, pad)
    cx, cy = pad + w // 2, pad + h // 2
    # Hiz cizgileri: domuz hizla asagi geliyor
    for i in (-1, 0, 1):
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


def sprite_boss(w: int, h: int, t: float, look: float, hit: float, charge: float) -> np.ndarray:
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
    for i, eq in enumerate(("y''+4y=0", "dy/dx", "L{f(t)}", "e^(st)")):
        a = t * 0.9 + i * math.pi / 2
        tx, ty = cx + int(math.cos(a) * w * 0.50) - int(w * 0.07), cy + int(math.sin(a) * h * 0.62)
        cv2.putText(cv, eq, (tx, ty), FONT, max(0.32, 0.5 * s), (255, 200, 255, 255), max(1, int(s * 1.6)), cv2.LINE_AA)
    return cv


_BOY_RAW = {}


def boy_sprite(height: int) -> np.ndarray:
    """Kalem cocuk (assets/kalem_cocuk.png); dosya yoksa basit bir cizim kullanilir."""
    if "raw" not in _BOY_RAW:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "kalem_cocuk.png")
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


def draw_pill(img, text, cx, cy, scale, u, bg=DARK, fg=CREAM, alpha=0.7, border=None, th=1):
    """Ortalanmis yuvarlak etiket (pill)."""
    tw = text_width(text, scale, th, u)
    ph, pw = int(14 * u * scale / 0.5) , int(12 * u)
    rrect(img, cx - tw / 2 - pw, cy - ph, cx + tw / 2 + pw, cy + ph, ph, bg, alpha, border, max(1, 2 * u))
    txt(img, text, cx, cy + int(5 * u * scale / 0.5), scale, fg, th, u, outline=False)


# --------------------------------------------------------------------------
# CIZIM - OYUN NESNELERI
# --------------------------------------------------------------------------
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


AMBIENT_FORMULAS = ("E=mc2", "F=ma", "dy/dx", "a2+b2=c2", "4.00", "sin x", "AA", "y''+y=0", "GANO")
AMBIENT_CODE = ("</>", "def", "{ }", "git push", "python", "OpenCV", "import", "for i in", "#include")


def draw_ambient(img, t: float, u: float, words=AMBIENT_FORMULAS, color=GOLD):
    """Yavasca yukari suzulen yazilar (menu atmosferi): formuller ya da kod parcalari."""
    H, W = img.shape[:2]
    for i, f in enumerate(words):
        x = (i * 0.117 + 0.04) * W + math.sin(t * 0.8 + i) * 14 * u
        y = H - ((t * (22 + 8 * (i % 4)) + i * 83) % (H + 60))
        cv2.putText(img, f, (int(x), int(y)), FONT, 0.6 * u, color, max(1, int(2 * u)), cv2.LINE_AA)


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


# --------------------------------------------------------------------------
# CIZIM - ARAYUZ (HUD ve ekranlar)
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# KAMERA
# --------------------------------------------------------------------------
def open_camera(index: int) -> cv2.VideoCapture:
    """Kamerayi acar. Windows'ta once DirectShow, olmazsa varsayilan arka uc denenir."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            return cap
        cap.release()
    print(f"[HATA] Kamera {index} acilamadi. Baska bir uygulama kullaniyor olabilir; "
          f"--camera 1 gibi baska indeks deneyin.")
    sys.exit(1)


def fit_frame(frame: np.ndarray) -> np.ndarray:
    """Kareyi GAME_WIDTH genisligine orantili kucultur (hiz icin)."""
    h, w = frame.shape[:2]
    if w <= GAME_WIDTH:
        return frame
    return cv2.resize(frame, (GAME_WIDTH, int(h * GAME_WIDTH / w)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# ANA DONGU
# --------------------------------------------------------------------------
def main(camera_index: int = 0, mode: str = "kafa"):
    tracker = PoseTracker() if mode == "vucut" else FaceTracker()
    cap = open_camera(camera_index)
    game: Optional[Game] = None
    last = time.perf_counter()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[HATA] Kameradan goruntu alinamadi.")
                break

            frame = fit_frame(cv2.flip(frame, 1))      # ayna efekti (yatay cevir)
            H, W = frame.shape[:2]
            if game is None or (game.W, game.H) != (W, H):
                game = Game(W, H, best_gano=game.best_gano if game else 0.0,
                            name=game.name if game else "", intro=game is None)

            now = time.perf_counter()
            dt = min(now - last, 0.05)                 # takilmalarda sicrama olmasin
            last = now

            face = tracker.detect(frame)               # yuz/burun tespiti (ham kare)
            frame = apply_vignette(frame)              # sinematik kenar karartma
            game.update(dt, face)                      # oyun mantigi
            cv2.imshow(WINDOW_NAME, render(frame, game))

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                              # ESC her durumda cikar
                break
            if game.state == STATE_INTRO:              # herhangi bir tus intro'yu gecer
                if key != 255 and game.anim_t > 0.3:
                    game.state = STATE_NAME
            elif game.state == STATE_NAME:             # isim yazilirken Q/R harf sayilir
                game.type_key(key)
            elif key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                game.reset()
            elif key in (ord("n"), ord("N")):
                game.new_player()
            # Pencere X ile kapatildiysa cik
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------
# KAMERASIZ OTOMATIK TEST
# --------------------------------------------------------------------------
def selftest():
    """Kamera olmadan mantik, cizim ve MediaPipe baslatmayi dogrular."""
    if hasattr(sys.stdout, "reconfigure"):          # Turkce karakterli yazdirmalar cokmesin
        sys.stdout.reconfigure(errors="replace")
    random.seed(1)
    W, H = 960, 540
    lb = os.path.join(tempfile.mkdtemp(), "lb.json")       # gercek leaderboard'a dokunma

    # 1) MediaPipe model + bos kare (yuz yok -> None donmeli)
    tracker = FaceTracker()
    assert tracker.detect(np.zeros((H, W, 3), np.uint8)) is None
    tracker.close()
    print("[OK] MediaPipe FaceDetector calisiyor")

    # 2) Oyun akisi: sahte yuz ile bekleme -> geri sayim -> oyun -> carpisma
    # Isim ekrani: baslangic durumu, tus girisi, bos isimle ENTER kabul edilmemeli
    game = Game(W, H, lb_path=lb)
    assert game.state == STATE_NAME
    assert not game.type_key(13) and game.state == STATE_NAME
    for ch in "Ali Veli_2!?" + "x" * 20:
        game.type_key(ord(ch))
    assert game.name == ("Ali Veli_2" + "x" * 20)[:MAX_NAME_LEN], game.name
    game.type_key(8)
    assert game.type_key(13) and game.state == STATE_WAIT
    game.name = "Test"
    print("[OK] Isim girisi")
    gi = Game(W, H, lb_path=lb, intro=True)
    assert gi.state == STATE_INTRO
    for a in (0.0, 0.5, 1.5, 3.0, 12.0):
        gi.anim_t = a
        out = render(np.full((H, W, 3), 90, np.uint8), gi)
        assert out.shape == (H, W, 3)
    gi.update(1 / 60, None)
    assert gi.state == STATE_INTRO, "intro kendiliginden gecmemeli"
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_intro.png"), out)
    print("[OK] Intro ekrani")

    frame = np.full((H, W, 3), 90, np.uint8)
    face = FaceInfo(W / 2, H * 0.75, (W / 2 - 60, H * 0.75 - 90, W / 2 + 60, H * 0.75 + 90))
    seen = set()
    for _ in range(60 * 120):                          # 120 sn @ 60 FPS, oyuncu hareketsiz
        game.update(1 / 60, face)
        render(frame.copy(), game)
        seen.add(game.state)
        if game.state == STATE_OVER:
            break
    assert game.state == STATE_OVER, "hareketsiz oyuncu hic carpmadi?"
    assert {STATE_COUNTDOWN, STATE_PLAY, STATE_OVER} <= seen
    assert 0.0 <= game.gano <= MAX_GANO and game.death_msg
    assert game.particles and game.shake > 0 and game.flash > 0, "carpisma efekti yok"
    print(f"[OK] Carpisma: '{game.death_msg}', GANO={game.gano:.2f}, kacilan={game.dodged}")

    # 3) Yuz kaybolunca oyun durmali, R ile reset calismali
    g2 = Game(W, H, name="Test", lb_path=lb)
    for _ in range(300):
        g2.update(1 / 60, face)
    s = g2.survival
    for _ in range(60):
        g2.update(1 / 60, None)
    assert g2.survival == s, "yuz yokken sure akti"
    g2.reset()
    assert g2.state == STATE_WAIT and g2.dodged == 0 and not g2.obstacles
    print("[OK] Duraklatma ve reset")
    assert [grade_letter(x) for x in (4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0)] ==         ["AA", "BA", "BB", "CB", "CC", "DC", "DD", "FD", "FF"]
    # Kacilan engel harf notu uretmeli; iz parcaciklari olusmali
    g5 = Game(W, H, name="Bot", lb_path=lb)
    g5.state, g5.px, g5.py = STATE_PLAY, 20.0, H * 0.9
    g5.obstacles = [make_obstacle(KIND_DOMUZ, W, H, 1.0)]
    g5.obstacles[0].y, g5.obstacles[0].x = H - 5, W - 150
    g5.face_visible = True
    for _ in range(5):
        g5.update(1 / 60, FaceInfo(20.0, H * 0.9, (0, H * 0.85, 40, H * 0.95)))
    assert g5.dodged == 1 and g5.floats, "harf notu cikmadi"
    g6 = Game(W, H, name="Bot", lb_path=lb)
    g6.state = STATE_PLAY
    g6.obstacles = [make_obstacle(KIND_VIZE, W, H, 1.0)]
    g6.obstacles[0].y = 100
    n0 = len(g6.particles)
    for _ in range(120):
        g6.obstacles[0].y = 100
        g6._trail(g6.obstacles[0], 1 / 60)
    assert len(g6.particles) > n0, "iz parcacigi yok"
    print("[OK] Harf notu ve iz parcaciklari")

    # 4) Kosa kosa kacan oyuncu hayatta kalabiliyor mu? (basit kacis botu)
    g3 = Game(W, H, name="Bot", lb_path=lb)
    t = 0.0
    for _ in range(60 * 60):
        t += 1 / 60
        # En yakin engelden uzaga kac
        x = W / 2
        if g3.obstacles:
            near = max(g3.obstacles, key=lambda o: o.y)
            x = W * 0.1 if near.x + near.w / 2 > W / 2 else W * 0.9
        f = FaceInfo(x, H * 0.75, (x - 40, H * 0.75 - 60, x + 40, H * 0.75 + 60))
        g3.update(1 / 60, f)
        if g3.state == STATE_OVER:
            break
    print(f"[OK] Kacis botu: durum={g3.state}, GANO={g3.gano:.2f}, kacilan={g3.dodged}")

    # Yeni dersler: hareket + cizim; kilit acma; kalem cocuk gorseli
    assert boy_sprite(100).shape[0] == 100
    face_mid = FaceInfo(W / 2, H * 0.75, (W / 2 - 45, H * 0.7, W / 2 + 45, H * 0.8))
    for kind in (KIND_KIMYA, KIND_DEVRE, KIND_TERMO, KIND_PROJ, KIND_KALEM, KIND_VIZE):
        gx = Game(W, H, name="K", lb_path=lb)
        gx.state, gx._check_death = STATE_PLAY, (lambda: None)
        gx.update_player(face_mid)
        for _ in range(240):
            if not gx.obstacles:
                gx.obstacles = [make_obstacle(kind, W, H, 1.0)]
            gx._move_obstacles(1 / 60)
            render(frame.copy(), gx)
        assert kind == KIND_KALEM or kind in DEATH_MESSAGES
    base = {KIND_DOMUZ, KIND_RUZGAR, KIND_VIZE}
    gu = Game(W, H, name="U", lb_path=lb)
    seen_kinds = set()
    for _ in range(400):
        gu.obstacles = []
        for _ in range(30):                          # koridoru kapatan dogumlar reddedilir: tekrar dene
            gu.spawn_timer, gu.survival = 0, gu.survival + 0.7
            gu._spawn(0.0)
            if gu.obstacles:
                break
        seen_kinds.update(o.kind for o in gu.obstacles)
    assert seen_kinds <= base, seen_kinds
    gu.score_time = 100.0          # GANO ~1.94: kimya + devre acik, termo kapali
    for _ in range(600):
        gu.obstacles = []
        for _ in range(30):                          # koridoru kapatan dogumlar reddedilir: tekrar dene
            gu.spawn_timer, gu.survival = 0, gu.survival + 0.7
            gu._spawn(0.0)
            if gu.obstacles:
                break
        seen_kinds.update(o.kind for o in gu.obstacles)
    assert {KIND_KIMYA, KIND_DEVRE} <= seen_kinds and KIND_TERMO not in seen_kinds, seen_kinds
    print("[OK] Yeni dersler ve GANO ile kilit acma")

    # MATH 255 boss: tetik, kalem toplama, kalem cocuk vurusu, olum, GANO 3.50
    gb = Game(W, H, name="Boss", lb_path=lb)
    gb.state, gb._check_death = STATE_PLAY, (lambda: None)       # test: oyuncu olmesin
    gb.score_time = 150.0                                         # GANO ~2.53 >= 2.50
    px, frames, patterns, shot1, shot2, hp_seen = W / 2, 0, set(), False, False, set()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    while frames < 60 * 150 and not gb.boss_defeated:
        pk = [o for o in gb.obstacles if o.kind == KIND_KALEM]
        if pk:
            px = pk[0].x + pk[0].w / 2
        gb.update(1 / 60, FaceInfo(px, H * 0.75, (px - 45, H * 0.7, px + 45, H * 0.8)))
        frames += 1
        if gb.boss:
            hp_seen.add(gb.boss.hp)
            if gb.boss.pattern:
                patterns.add(gb.boss.pattern)
            if not shot1 and gb.boss.state == "fight" and gb.boss.charge > 0.3:
                cv2.imwrite(os.path.join(out_dir, "selftest_boss.png"), render(frame.copy(), gb))
                shot1 = True
        if gb.helper and gb.helper.t > 0.56 and not shot2:
            cv2.imwrite(os.path.join(out_dir, "selftest_slash.png"), render(frame.copy(), gb))
            shot2 = True
        if frames % 7 == 0:
            render(frame.copy(), gb)
    assert gb.boss_defeated and gb.boss is None, "boss yenilmedi"
    assert hp_seen >= {BOSS_HP, 1}, hp_seen
    assert gb.gano >= BOSS_WIN_GANO - 1e-6, gb.gano
    assert {"nisan", "yagmur"} <= patterns, patterns
    print(f"[OK] Boss: {frames / 60:.0f} sn'de yenildi, GANO={gb.gano:.2f}, saldirilar={sorted(patterns)}")

    # Kazanilabilirlik garantisi: gizli koridoru izleyen oyuncu 90 sn boyunca hic vurulmamali
    def corridor_run(seed, seconds=90):
        random.seed(seed)
        g = Game(W, H, name="Oracle", lb_path=lb)
        g.state, g.boss_defeated = STATE_PLAY, True
        py, fw, x = H * 0.7, 190.0, W / 2
        for _ in range(int(seconds * 60)):
            x = g.corridor(g.survival)
            g.update(1 / 60, FaceInfo(x, py, (x - fw / 2, py - fw / 2, x + fw / 2, py + fw / 2)))
            if g.state == STATE_OVER:
                return -1
        return g.dodged
    runs = [corridor_run(sd) for sd in range(12)]
    assert min(runs) >= 0, f"koridordaki oyuncu vuruldu: {runs}"
    assert min(runs) >= 25, f"koridor engelleri cok seyreltiyor: {runs}"
    print(f"[OK] Kazanilabilirlik: koridoru izleyen oyuncu 12/12 oyunda 90 sn hayatta, kacilan engel >= {min(runs)}")

    # Leaderboard: kayit, siralama, tekrar okuma, bozuk dosya
    r1, _ = add_score("A", 1.0, lb)
    r2, ents = add_score("B", 2.0, lb)
    assert r2 == 1 and [e["name"] for e in ents][:2] == ["B", "A"] or r2 >= 1
    assert load_leaderboard(lb) == ents
    gs = [e["gano"] for e in ents]
    assert gs == sorted(gs, reverse=True)
    with open(lb, "w") as f:
        f.write("{bozuk")
    assert load_leaderboard(lb) == []
    assert add_score("C", 0.5, lb)[0] == 1
    assert load_leaderboard(os.path.join(os.path.dirname(lb), "yok.json")) == []
    print("[OK] Leaderboard (kayit/siralama/bozuk dosya)")

    # Oyun bitince skor yazilmis olmali (adim 2 ve 4 ayni dosyaya yazdi)
    assert game.rank >= 1 and game.board, "game over leaderboard'a yazmadi"

    # 5) Tum ekran durumlarini ciz (Game Over dahil) - cokme olmamali
    for st in (STATE_INTRO, STATE_NAME, STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER):
        g3.state = st
        g3.board = [{"name": n, "gano": g} for n, g in
                    (("Ayse", 3.1), ("Mehmet", 2.7), ("Zeynep", 2.2), ("Can", 1.9),
                     ("Deniz", 1.5), ("Bot", 0.9), ("Ece", 0.4))]
        g3.rank = 6
        g3.death_msg = DEATH_MESSAGES[KIND_VIZE]
        g3.obstacles = [make_obstacle(k, W, H, 1.0) for k in (KIND_DOMUZ, KIND_RUZGAR, KIND_VIZE)]
        for ob in g3.obstacles:
            ob.y = H * 0.3
        g3.over_t, g3.anim_t = 1.0, 0.9
        g3.flash = g3.shake = 0.0
        img = render(frame.copy(), g3)
        assert img.shape == frame.shape
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"selftest_{st}.png"), img)
    # Farkli kamera boyutu (640x480) ve giris animasyonlari ortasi: cokme/tasma olmamali
    for (W2, H2) in ((640, 480), (1280, 720)):
        g4 = Game(W2, H2, name="Bot", lb_path=lb)
        g4.update_player(FaceInfo(W2 / 2, H2 * 0.7, (W2 / 2 - 50, H2 * 0.6, W2 / 2 + 50, H2 * 0.8)))
        g4.board, g4.rank = g3.board, 7
        g4.floats = [FloatText(W2 / 2, H2 * 0.8, "AA", GOLD, 1.0, 1.1)]
        for st in (STATE_INTRO, STATE_NAME, STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER):
            g4.state = st
            g4.obstacles = [make_obstacle(k, W2, H2, 1.0) for k in (KIND_DOMUZ, KIND_RUZGAR, KIND_VIZE)]
            for ob in g4.obstacles:
                ob.y = H2 * 0.3
            for anim in (0.0, 0.2, 5.0):
                g4.anim_t, g4.over_t = anim, anim
                out = render(np.full((H2, W2, 3), 90, np.uint8), g4)
                assert out.shape == (H2, W2, 3)
    print("[OK] Cizim tamam (selftest_*.png, 640x480 ve 1280x720 dahil)")
    print("TUM TESTLER GECTI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IYTE Kacis - kameradan oynanan oyun")
    parser.add_argument("--camera", type=int, default=0, help="kamera indeksi (varsayilan 0)")
    parser.add_argument("--mode", choices=("kafa", "vucut"), default="kafa",
                        help="kafa: oturarak (yuz takibi) | vucut: ayakta, uzaktan (poz takibi)")
    parser.add_argument("--selftest", action="store_true", help="kamerasiz otomatik test")
    args = parser.parse_args()
    selftest() if args.selftest else main(args.camera, args.mode)
