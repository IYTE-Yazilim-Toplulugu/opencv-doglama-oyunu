"""Ana dongu ve komut satiri (kamera, takip, oyun, ekran)."""

import argparse
import cv2
import time
from typing import Optional

from .config import STATE_INTRO, STATE_NAME, WINDOW_NAME
from .drawing import apply_vignette
from .game import Game
from .render import render
from .retro import render_retro, to_screen
from .selftest import selftest
from .tracking import FaceTracker, PoseTracker, fit_frame, open_camera, scale_face


ARCADE_W, ARCADE_H = 960, 540          # arcade modunda oyun uzayi sabittir (kamera boyutundan bagimsiz)


def main(camera_index: int = 0, mode: str = "kafa", stil: str = "arcade"):
    tracker = PoseTracker() if mode == "vucut" else FaceTracker()
    cap = open_camera(camera_index)
    arcade = stil == "arcade"
    # arcade: oyuncu kutusu yuz boyutundan bagimsiz sabit (uzaktan/yakindan oynayan icin adil)
    game: Optional[Game] = (Game(ARCADE_W, ARCADE_H, intro=True, fixed_box=(ARCADE_W * 0.13, ARCADE_H * 0.24))
                            if arcade else None)
    cam_mode = 0                       # C tusu: 0 kucuk onizleme, 1 gizli, 2 arka plan
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
            elif not arcade and (game.W, game.H) != (W, H):   # kamera boyutu degisirse oyun durumu korunur
                frame = cv2.resize(frame, (game.W, game.H))

            now = time.perf_counter()
            dt = min(now - last, 0.05)                 # takilmalarda sicrama olmasin
            last = now

            face = tracker.detect(frame)               # yuz/burun tespiti (ham kare)
            if arcade:
                game.update(dt, scale_face(face, ARCADE_W / W, ARCADE_H / H))
                cv2.imshow(WINDOW_NAME, to_screen(render_retro(game, frame, cam_mode)))
            else:
                frame = apply_vignette(frame)          # sinematik kenar karartma
                game.update(dt, face)                  # oyun mantigi
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
            elif key in (ord("c"), ord("C")) and arcade:
                cam_mode = (cam_mode + 1) % 3
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
    parser.add_argument("--stil", choices=("arcade", "klasik"), default="arcade",
                        help="arcade: 8-bit pixel gorunum (varsayilan) | klasik: kamera goruntusu uzerinde")
    parser.add_argument("--selftest", action="store_true", help="kamerasiz otomatik test")
    args = parser.parse_args()
    selftest() if args.selftest else main(args.camera, args.mode, args.stil)
