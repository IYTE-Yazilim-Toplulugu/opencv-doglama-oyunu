"""Oyun sabitleri: ayarlar, renkler, durumlar, engel/boss tablolari."""

import cv2
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # proje koku

# Oyunun adi. Turkce karakterler desteklenmeyen (eski) OpenCV'de otomatik ASCII'ye cevrilir.
GAME_TITLE = "HOCAM DOMUZ VAR!"
GAME_SUBTITLE = "İYTE Kaçış: Gülbahçe'de hayatta kal, GANO'nu 4.00'a taşı!"
WINDOW_NAME = GAME_TITLE + " - IYTE Kacis"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_FILE = os.path.join(ROOT,
                          "blaze_face_short_range.tflite")

LEADERBOARD_FILE = os.path.join(ROOT,
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


POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
POSE_MODEL_FILE = os.path.join(ROOT, "pose_landmarker_lite.task")


MODEL_MIN_BYTES = 10_000
DOWNLOAD_TIMEOUT = 30          # saniye


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


STATE_INTRO, STATE_NAME, STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER = (
    "intro", "name", "wait", "countdown", "play", "over")


MAX_PARTICLES = 300


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


HELPER_DURATION = 1.1
HELPER_HIT_AT = 0.5               # kalem cocuk bu saniyede bossu keser
