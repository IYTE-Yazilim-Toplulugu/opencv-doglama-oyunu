# Hocam Domuz Var! (IYTE Kaçış · OpenCV + MediaPipe)

Kameradan oynanan "dodge" oyunu. Kafanı sağa-sola hareket ettirerek İYTE kampüsündeki
engellerden (Domuz, Gülbahçe Rüzgârı, Fizik 101 Vizesi) kaç. Skor: **GANO** (0.00 → 4.00).

## Kurulum ve çalıştırma
```bash
pip install opencv-python mediapipe
python iyte_kacis.py
```
- İlk çalıştırmada yüz algılama modeli (~230 KB) otomatik indirilir.
- Başka kamera: `python iyte_kacis.py --camera 1`
- **Ayakta oynamak için (deneysel):** `python iyte_kacis.py --mode vucut` — tüm vücut pozundan başı bulur, 2-3 m uzaktan çalışır. Ilk çalıştırmada poz modeli (~6 MB) iner. Yüz takibine göre daha yavaş olabilir.
- Kamerasız otomatik test: `python iyte_kacis.py --selftest`

## Tuşlar
`R` yeniden başla · `N` yeni oyuncu (isim değiştir) · `Q` / `ESC` çık

Açılışta isim sorulur (ENTER ile onayla). Oyun bitince final GANO'n ve ilk 5 sıralama gösterilir; skorlar `leaderboard.json` dosyasında saklanır (git'e girmez).

## Akış
İYTE Yazılım Topluluğu intro ekranı → "bir tuşa bas" → isim → oyun → transkript/leaderboard.

## Notlar
- Yüz kaybolursa oyun durur (haksız ölüm olmaz).
- **Adil oyun:** Gizli bir kaçış koridoru ekran boyunca yavaşça süzülür; engeller o koridorun geçeceği yere doğmaz. Yani her an ulaşılabilir bir yol vardır. Otomatik test koridoru izleyen oyuncunun hiç ölmediğini doğrular.
- Zorluk tavanı düşük tutulur (engel hızı en fazla 1.5x), uzaktan oynayanın çarpışma kutusu çok küçülmez.
- Zorluk ve skor ayarları dosyanın başındaki sabitlerden değiştirilebilir.
- Yüzüne sevimli bir **domuz burnu**, domuz kulakları ve allık eklenir; burun kafanı oynattıkça ezilir, çarpınca başında yıldızlar döner.

## Dersler ve Boss (MATH 255)
GANO yükseldikçe yeni dersler açılır:

| GANO | Yeni engel |
|---|---|
| 0.00 | Domuz, Gülbahçe Rüzgârı, Fizik 101 Vizesi |
| 1.00 | CHEM 101 Kimya Lab Raporu (sallanan erlen) |
| 1.50 | EEE 201 Devre Analizi (keskin zigzaglı şimşek) |
| 2.00 | ME 301 Termodinamik (büyük ateş topu) |
| 2.50 | **MATH 255 – Diferansiyel Denklemler (final boss)** |

**Boss savaşı:** Normal engeller durur, GANO donar. Boss üç saldırı yapar (nişan yelpazesi, boşluklu denklem yağmuru, sinüs dalgası; her saldırıdan önce kızarıp haber verir). Yukarıdan düşen **altın kalemi yüzünle topla**: Kalem Çocuk bossa dalıp keser. 5 kesikte boss yenilir ve GANO en az **3.50**'ye sıçrar.

`assets/kalem_cocuk.png` Kalem Çocuk görselidir (dosya yoksa basit bir çizim kullanılır).
