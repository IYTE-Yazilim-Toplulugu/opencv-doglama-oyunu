"""Ana dongu ve komut satiri (kamera, takip, oyun, ekran)."""

import argparse
import cv2
import time
from typing import Optional

from .config import STATE_INTRO, STATE_NAME, WINDOW_NAME
from .drawing import apply_vignette
from .game import Game
from .render import render
from .selftest import selftest
from .tracking import FaceTracker, PoseTracker, fit_frame, open_camera


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
            if game is None:
                game = Game(W, H, intro=True)
            elif (game.W, game.H) != (W, H):             # kamera boyutu degisirse oyun durumu korunur
                frame = cv2.resize(frame, (game.W, game.H))

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


def cli():
    parser = argparse.ArgumentParser(description="IYTE Kacis - kameradan oynanan oyun")
    parser.add_argument("--camera", type=int, default=0, help="kamera indeksi (varsayilan 0)")
    parser.add_argument("--mode", choices=("kafa", "vucut"), default="kafa",
                        help="kafa: oturarak (yuz takibi) | vucut: ayakta, uzaktan (poz takibi)")
    parser.add_argument("--selftest", action="store_true", help="kamerasiz otomatik test")
    args = parser.parse_args()
    selftest() if args.selftest else main(args.camera, args.mode)
