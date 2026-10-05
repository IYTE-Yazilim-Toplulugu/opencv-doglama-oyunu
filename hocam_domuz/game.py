"""Oyunun durum makinesi: oyuncu, engel uretimi, carpisma, skor, efektler."""

import math
import random
from typing import List, Optional, Tuple

from .boss import BossMixin
from .config import (
    BOSS_TRIGGER_GANO,
    CORRIDOR_FACTOR,
    COUNTDOWN_SECONDS,
    CRASH_COLORS,
    CREAM,
    DEATH_MESSAGES,
    DODGE_WEIGHT,
    DOMUZ_AFTER,
    GANO_K,
    GOLD,
    HITBOX_SHRINK_PLAYER,
    KIND_DEVRE,
    KIND_DOMUZ,
    KIND_KALEM,
    KIND_KIMYA,
    KIND_PROJ,
    KIND_VIZE,
    LEADERBOARD_FILE,
    MAX_GANO,
    MAX_NAME_LEN,
    MAX_PARTICLES,
    SECONDS_WEIGHT,
    SMOOTHING,
    SPAWN_TABLE,
    STATE_COUNTDOWN,
    STATE_INTRO,
    STATE_NAME,
    STATE_OVER,
    STATE_PLAY,
    STATE_WAIT,
    TRAIL,
    VIZE_DRIFT_MAX,
    WHITE,
)
from .entities import Boss, FloatText, Helper, Obstacle, Particle, grade_letter, intersects, make_obstacle
from .leaderboard import add_score
from .tracking import FaceInfo


class Game(BossMixin):
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
