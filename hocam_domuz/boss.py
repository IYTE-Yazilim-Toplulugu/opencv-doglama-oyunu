"""MATH 255 final boss'u ve kalem cocuk (Game'e karisan mixin)."""

import math
import random

from .config import (
    BOSS_COLORS,
    BOSS_HP,
    BOSS_WIN_GANO,
    CRASH_COLORS,
    CREAM,
    DODGE_WEIGHT,
    GANO_K,
    GOLD,
    HELPER_DURATION,
    HELPER_HIT_AT,
    KIND_KALEM,
    KIND_PROJ,
    MAX_GANO,
    RED,
    SECONDS_WEIGHT,
    WHITE,
)
from .drawing import ease_out
from .entities import Boss, FloatText, Particle, make_obstacle, make_projectile


class BossMixin:
    """MATH 255 boss'u ve kalem cocuk mantigi (Game icine karisir)."""


    def _start_boss(self):
        """GANO esigi asilinca boss yukaridan girer; normal engel uretimi durur."""
        W, H = self.W, self.H
        bw, bh = W * 0.34, H * 0.27
        self.boss = Boss(W / 2, -bh, bw, bh, BOSS_HP, BOSS_HP, target_y=H * 0.09 + bh / 2)
        self.set_banner("UYARI! MATH 255 FİNALİ", "Diferansiyel Denklemler geliyor...", RED, 3.0)
        self.shake, self.flash = 0.8, 0.5
        for ob in self.obstacles:                    # normal engeller boss girerken dagilir
            self.burst(ob.x + ob.w / 2, ob.y + ob.h / 2, 6, CRASH_COLORS.get(ob.kind, BOSS_COLORS), 160, 5, life=0.6)
        self.obstacles.clear()

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
        elif pattern == "dalga":
            for k in range(4):
                self.obstacles.append(make_projectile(sx + (k - 1.5) * W * 0.17, sy, W, H, 0.0, H * 0.42,
                                                      amp=W * 0.05, phase=k * 1.3))
        else:
            raise ValueError(f"bilinmeyen boss saldirisi: {pattern!r}")
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
