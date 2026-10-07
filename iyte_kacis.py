#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOCAM DOMUZ VAR! (İYTE Kaçış / Dodge) - Kameradan oynanan interaktif oyun
=========================================================================
Oyuncu kafasını sağa/sola hareket ettirerek kampüste hayatta kalmaya çalışan
bir İYTE öğrencisini kontrol eder. Yüz takibi MediaPipe (Tasks API) ile,
görüntü ve çizimler OpenCV ile yapılır. Kod `hocam_domuz/` paketindedir.

Kurulum:    pip install -r requirements.txt
Çalıştırma: python iyte_kacis.py                  (varsayılan kamera: 0)
            python iyte_kacis.py --camera 1       (başka kamera)
            python iyte_kacis.py --mode vucut     (ayakta oynama, deneysel)
            python iyte_kacis.py --selftest       (kamerasız otomatik test)

Tuşlar: R = yeniden başla | N = yeni oyuncu | Q (veya ESC) = çık
"""

from hocam_domuz.app import cli

if __name__ == "__main__":
    cli()
