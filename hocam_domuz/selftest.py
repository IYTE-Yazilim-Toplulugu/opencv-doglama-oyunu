"""Kamerasiz otomatik test (python iyte_kacis.py --selftest)."""

import cv2
import numpy as np
import os
import random
import sys
import tempfile
import time

from .config import (
    BOSS_HP,
    BOSS_WIN_GANO,
    DEATH_MESSAGES,
    GOLD,
    KIND_DEVRE,
    KIND_DOMUZ,
    KIND_KALEM,
    KIND_KIMYA,
    KIND_PROJ,
    KIND_RUZGAR,
    KIND_TERMO,
    KIND_VIZE,
    MAX_GANO,
    MAX_NAME_LEN,
    STATE_COUNTDOWN,
    STATE_INTRO,
    STATE_NAME,
    STATE_OVER,
    STATE_PLAY,
    STATE_WAIT,
)
from .entities import FloatText, Helper, grade_letter, make_obstacle, make_projectile
from .game import Game
from .leaderboard import add_score, load_leaderboard
from .pixel import LH, LW, MISSING, PALETTE
from .render import render
from .retro import render_retro, to_screen
from .sprites import boy_sprite
from .tracking import FaceInfo, FaceTracker, scale_face

OUT_DIR = os.path.join(tempfile.gettempdir(), "hocam_domuz_selftest")   # test goruntuleri buraya


def selftest():
    """Kamera olmadan mantik, cizim ve MediaPipe baslatmayi dogrular."""
    if hasattr(sys.stdout, "reconfigure"):          # Turkce karakterli yazdirmalar cokmesin
        sys.stdout.reconfigure(errors="replace")
    os.makedirs(OUT_DIR, exist_ok=True)
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
    cv2.imwrite(os.path.join(OUT_DIR, "selftest_intro.png"), out)
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
    assert [grade_letter(x) for x in (4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0)] == ["AA", "BA", "BB", "CB", "CC", "DC", "DD", "FD", "FF"]
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
    assert g3.survival > 0, "bot hic oynamadi"
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
    out_dir = OUT_DIR
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

    # Leaderboard: kayit, siralama, tekrar okuma, bozuk dosya yedegi
    lb2 = os.path.join(os.path.dirname(lb), "rank.json")
    r1, _ = add_score("A", 1.0, lb2)
    r2, ents = add_score("B", 2.0, lb2)
    assert r1 == 1 and r2 == 1 and [e["name"] for e in ents] == ["B", "A"], (r1, r2, ents)
    assert load_leaderboard(lb2) == ents
    assert add_score("C", 1.5, lb2)[0] == 2
    ents = load_leaderboard(lb2)
    gs = [e["gano"] for e in ents]
    assert gs == sorted(gs, reverse=True) and len(ents) == 3
    with open(lb2, "w") as f:
        f.write("{bozuk")
    assert load_leaderboard(lb2) == []
    assert not os.path.exists(lb2), "bozuk dosya yerinde kaldi"
    with open(lb2 + ".bak") as f:
        assert f.read() == "{bozuk", "bozuk dosya yedeklenmedi"
    assert add_score("D", 0.5, lb2)[0] == 1
    assert load_leaderboard(os.path.join(os.path.dirname(lb), "yok.json")) == []
    # Bilinmeyen tur/saldiri sessizce gecmemeli
    gz = Game(W, H, name="X", lb_path=lb)
    gz._start_boss()
    for bad in (lambda: make_obstacle("yok", W, H, 1.0), lambda: gz._boss_fire("yok")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bilinmeyen tur kabul edildi")
    print("[OK] Leaderboard (kayit/siralama/bozuk dosya yedegi)")

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
        cv2.imwrite(os.path.join(OUT_DIR, f"selftest_{st}.png"), img)
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
    # 8-bit arcade modu: her ekran/kamera modu cokmeden cizilir, SADECE palet renkleri kullanilir
    # (8-bit gorunumun garantisi), fontta eksik karakter yok, sprite'lar ve ekranlar makul surede cizilir
    pal = {tuple(c) for c in PALETTE.values()}
    cam = np.random.RandomState(1).randint(30, 220, (540, 960, 3)).astype(np.uint8)
    gr = Game(960, 540, name="Rüzgâr", lb_path=lb, fixed_box=(960 * 0.13, 540 * 0.24))
    gr.update_player(FaceInfo(500, 330, (440, 270, 560, 390)))
    kinds = (KIND_DOMUZ, KIND_RUZGAR, KIND_VIZE, KIND_KIMYA, KIND_DEVRE, KIND_TERMO, KIND_KALEM)
    gr.obstacles = [make_obstacle(k, 960, 540, 1.0) for k in kinds] + [make_projectile(300, 250, 960, 540, 0, 100)]
    for i, ob in enumerate(gr.obstacles):
        ob.x, ob.y = 30 + i * 105, 90 + (i % 2) * 150
    gr.floats = [FloatText(300, 380, "BB", GOLD, 1.0, 1.1)]
    gr.board = [{"name": n, "gano": 3.0 - i * 0.4} for i, n in enumerate(("AYŞE", "MEHMET", "ZEYNEP", "CAN", "DENİZ", "ECE"))]
    gr.rank, gr.state = 6, STATE_PLAY
    gr._start_boss()
    gr.boss.y, gr.boss.state, gr.boss.charge, gr.helper = gr.boss.target_y, "fight", 0.4, Helper(0.55, 500, 330)
    t0 = time.perf_counter()
    frames = 0
    for st in (STATE_INTRO, STATE_NAME, STATE_WAIT, STATE_COUNTDOWN, STATE_PLAY, STATE_OVER):
        for cam_mode in (0, 1, 2):
            for anim in (0.0, 1.0, 5.0):
                gr.state, gr.anim_t, gr.over_t, gr.countdown = st, anim, anim, 2.5
                msg = list(DEATH_MESSAGES.values())[frames % len(DEATH_MESSAGES)]
                gr.death_msg, gr.banner, gr.banner_sub, gr.banner_t, gr.banner_dur = msg, "MATH 255 GEÇİLDİ!", "GANO 3.50'ye yükseldi!", 1.0, 3.0
                cvs = render_retro(gr, cam, cam_mode)
                frames += 1
                assert cvs.shape == (LH, LW, 3) and cvs.dtype == np.uint8
                used = {tuple(c) for c in np.unique(cvs.reshape(-1, 3), axis=0).tolist()}
                assert used <= pal, f"palet disi renk ({st}, kamera {cam_mode}): {sorted(used - pal)[:3]}"
                if cam_mode == 0 and anim == 5.0:
                    cv2.imwrite(os.path.join(OUT_DIR, f"retro_{st}.png"), to_screen(cvs))
    assert not MISSING, f"fontta olmayan karakterler: {MISSING}"
    ms = (time.perf_counter() - t0) / frames * 1000
    assert ms < 60, f"arcade cizimi cok yavas: {ms:.0f} ms/kare"
    sf = scale_face(FaceInfo(10, 20, (0, 0, 20, 40)), 2, 3)
    assert (sf.nose_x, sf.nose_y, sf.box) == (20, 60, (0, 0, 40, 120)) and scale_face(None, 2, 3) is None
    print(f"[OK] Arcade (8-bit) modu: {frames} kare, yalniz palet renkleri, {ms:.0f} ms/kare")
    print("TUM TESTLER GECTI")
