"""Kamera ve takip: model indirme, yuz (FaceTracker) ve vucut (PoseTracker) takibi."""

import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from typing import Optional, Tuple

from .config import (
    CAM_HEIGHT,
    CAM_WIDTH,
    DOWNLOAD_TIMEOUT,
    GAME_WIDTH,
    MODEL_FILE,
    MODEL_MIN_BYTES,
    MODEL_URL,
    NOSE_KEYPOINT,
    POSE_MODEL_FILE,
    POSE_MODEL_URL,
)


def _model_ok(path: str) -> bool:
    """Dosya var, yeterince buyuk ve HTML hata sayfasi degil mi?"""
    try:
        if os.path.getsize(path) < MODEL_MIN_BYTES:
            return False
        with open(path, "rb") as f:
            return not f.read(512).lstrip().startswith(b"<")
    except OSError:
        return False


def ensure_model(path: str = MODEL_FILE, url: str = MODEL_URL) -> bytes:
    """Modeli dosyadan okur; yoksa/bozuksa indirir. Her hata icin ayri, acik bir mesaj verir."""
    def fail(msg: str):
        print(f"[HATA] {msg}\nModeli elle indirip '{path}' olarak kaydedin:\n{url}")
        sys.exit(1)

    if not _model_ok(path):
        print(f"[BILGI] Model indiriliyor: {os.path.basename(path)}")
        tmp = path + ".part"
        try:
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
        except OSError as exc:                       # ag hatasi, zaman asimi ya da gecici dosya yazilamadi
            fail(f"Model indirilemedi (ag baglantisi ya da diske yazma): {exc}")
        if not _model_ok(tmp):
            os.remove(tmp)
            fail("Indirilen dosya gecersiz (cok kucuk ya da bir hata sayfasi).")
        try:
            os.replace(tmp, path)
        except OSError as exc:
            fail(f"Model '{path}' konumuna kaydedilemedi (izin?): {exc}")
    try:
        with open(path, "rb") as f:   # buffer olarak yukle: Turkce karakterli yollarda da calisir
            return f.read()
    except OSError as exc:
        fail(f"Model dosyasi okunamadi: {exc}")


@dataclass
class FaceInfo:
    """Algilanan yuzun piksel cinsinden bilgisi."""
    nose_x: float
    nose_y: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2


class FaceTracker:
    """MediaPipe FaceDetector'u sarar; kalabalikta tek yuz secer (ortadaki, sonra ona kilitlenir)."""

    BACKGROUND_RATIO = 0.6      # en buyuk yuzun genisliginin bunun altindakiler arka plan sayilir
    LOCK_LOSS_SECONDS = 0.6     # kilitli yuz bu kadar kayboluysa yeniden ortadakini sec

    def __init__(self):
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self._last_ts = -1
        self._lock = None           # kilitli yuzun son (merkez x, merkez y, genislik)
        self._lock_seen = 0.0

    def _pick(self, detections, frame_w: int):
        """Arka plan yuzlerini eler; kilitli yuz varsa en yakinini, yoksa en ortadakini secer.
        Yalnizca 'en buyuk' secilseydi yan yana iki kisi arasinda karakter sicrardi."""
        widest = max(d.bounding_box.width for d in detections)
        near = [d for d in detections if d.bounding_box.width >= widest * self.BACKGROUND_RATIO]

        def center(d):
            b = d.bounding_box
            return b.origin_x + b.width / 2, b.origin_y + b.height / 2

        now = time.monotonic()
        if self._lock is not None and now - self._lock_seen <= self.LOCK_LOSS_SECONDS:
            lx, ly, lw = self._lock
            best = min(near, key=lambda d: (center(d)[0] - lx) ** 2 + (center(d)[1] - ly) ** 2)
            cx, cy = center(best)
            if (cx - lx) ** 2 + (cy - ly) ** 2 <= (1.5 * lw) ** 2:   # ayni kisi: hizli kafa hareketini de kapsar
                det = best
            else:
                det = None
        else:
            det = None
        if det is None:                                   # kilit yok/kayip: ekranin ortasina en yakin yuz
            det = min(near, key=lambda d: abs(center(d)[0] - frame_w / 2))
        cx, cy = center(det)
        self._lock, self._lock_seen = (cx, cy, det.bounding_box.width), now
        return det

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceInfo]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # artan zaman damgasi sart
        self._last_ts = ts
        result = self.detector.detect_for_video(mp_image, ts)
        if not result.detections:
            return None
        det = self._pick(result.detections, w)
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


def scale_face(face: Optional[FaceInfo], sx: float, sy: float) -> Optional[FaceInfo]:
    """Yuz bilgisini baska bir cozunurluge (ornegin 960x540 oyun uzayina) olcekler."""
    if face is None:
        return None
    x1, y1, x2, y2 = face.box
    return FaceInfo(face.nose_x * sx, face.nose_y * sy, (x1 * sx, y1 * sy, x2 * sx, y2 * sy))


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
