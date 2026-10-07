"""Oyun nesneleri: engeller, boss, kalem cocuk, parcaciklar, harf notu."""

import math
import random
from dataclasses import dataclass
from typing import Tuple

from .config import (
    HITBOX_SHRINK_KIND,
    HITBOX_SHRINK_OBSTACLE,
    KIND_DEVRE,
    KIND_DOMUZ,
    KIND_KALEM,
    KIND_KIMYA,
    KIND_PROJ,
    KIND_RUZGAR,
    KIND_TERMO,
    KIND_VIZE,
)


def intersects(a: Tuple[float, float, float, float],
               b: Tuple[float, float, float, float]) -> bool:
    """Iki (x1, y1, x2, y2) kutusu kesisiyor mu? (AABB carpisma testi)"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


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
    elif kind == KIND_PROJ:       # boss mermisi
        w = h = W * 0.05
        speed = H * 0.5
    else:
        raise ValueError(f"bilinmeyen engel turu: {kind!r}")
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


def grade_letter(gano: float) -> str:
    """GANO'dan IYTE harf notu (AA, BA, BB, CB, CC, DC, DD, FD, FF)."""
    for limit, letter in ((3.75, "AA"), (3.25, "BA"), (2.75, "BB"), (2.25, "CB"),
                          (1.75, "CC"), (1.25, "DC"), (0.75, "DD"), (0.25, "FD")):
        if gano >= limit:
            return letter
    return "FF"


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
