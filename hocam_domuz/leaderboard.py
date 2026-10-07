"""Skor tablosu (JSON dosyasi)."""

import json
import os
import tempfile
import time
from typing import List, Tuple

from .config import LEADERBOARD_FILE, LEADERBOARD_SAVED


def load_leaderboard(path: str = LEADERBOARD_FILE) -> List[dict]:
    """Kayitlari okur. Dosya yoksa (ilk calistirma) bos liste. Dosya bozuksa uyarir, '.bak'a
    tasir (eski skorlar kaybolmasin) ve bos listeyle devam eder. Okunamiyorsa OSError firlatir."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("liste degil")
        return [e for e in data if isinstance(e, dict)
                and isinstance(e.get("name"), str) and isinstance(e.get("gano"), (int, float))]
    except (ValueError, TypeError) as exc:
        backup = path + ".bak"
        print(f"[UYARI] Leaderboard dosyasi bozuk ({exc}); yedegi: {backup}")
        try:
            os.replace(path, backup)
        except OSError as err:
            print(f"[UYARI] Yedek alinamadi: {err}")
        return []


def add_score(name: str, gano: float, path: str = LEADERBOARD_FILE) -> Tuple[int, List[dict]]:
    """Skoru ekler, dosyaya yazar. (oyuncunun sirasi [1'den], siralanmis liste) dondurur."""
    writable = True
    try:
        entries = load_leaderboard(path)
    except OSError as exc:                 # okunamayan dosyanin ustune yazip skorlari silme
        print(f"[UYARI] Leaderboard okunamadi, kaydedilmeyecek: {exc}")
        entries, writable = [], False
    mine = {"name": name, "gano": round(gano, 2), "date": time.strftime("%Y-%m-%d %H:%M")}
    entries.append(mine)
    # Puan yuksek olan ustte; esitlikte once yapan ustte (sort kararli, eski kayitlar once)
    entries.sort(key=lambda e: -e["gano"])
    rank = next(i for i, e in enumerate(entries) if e is mine) + 1
    entries = entries[:LEADERBOARD_SAVED]
    if not writable:
        return rank, entries
    try:                                   # atomik yazim: yarim dosya kalmasin
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[UYARI] Leaderboard kaydedilemedi: {exc}")
    return rank, entries
