# Hocam Domuz Var! (IYTE Kaçış · OpenCV + MediaPipe)

Kameradan oynanan "dodge" oyunu. Kafanı sağa-sola hareket ettirerek İYTE kampüsündeki
engellerden (Domuz, Gülbahçe Rüzgârı, Fizik 101 Vizesi) kaç. Skor: **GANO** (0.00 → 4.00).

## Kurulum ve çalıştırma
```bash
pip install -r requirements.txt
python iyte_kacis.py
```
Test edilen ortam: Python 3.13, MediaPipe 1.0.1, OpenCV 5.0 (Windows). MediaPipe'ın yeni Tasks API'sini (`mediapipe.tasks.python.vision`) kullanır; daha eski sürümler denenmedi.
- İlk çalıştırmada yüz algılama modeli (~230 KB) otomatik indirilir.
- Başka kamera: `python iyte_kacis.py --camera 1`
- Görünüm: varsayılan **8-bit arcade** (pixel-art İYTE kampüsü, kamera köşede küçük önizleme). Kamera görüntüsü üzerinde oynamak için `--stil klasik`.
- **Ayakta oynamak için (deneysel):** `python iyte_kacis.py --mode vucut` — tüm vücut pozundan başı bulur, 2-3 m uzaktan çalışır. İlk çalıştırmada poz modeli (~6 MB) iner. Yüz takibine göre daha yavaş olabilir.
- Kamerasız otomatik test: `python iyte_kacis.py --selftest`

## Tuşlar
`R` yeniden başla · `N` yeni oyuncu (isim değiştir) · `Q` / `ESC` çık · `C` (arcade) kamera önizlemesi: küçük → gizli → arka plan (Game Boy yeşili)

İsim yazarken `Q`, `R` ve `N` harf olarak sayılır; bu ekranda yalnızca `ESC` çıkar. İsimde yalnızca ASCII harf/rakam kullanılabilir (`cv2.waitKey` Türkçe karakterleri okuyamaz; "Sule", "Caglar" gibi yazın).

Açılışta isim sorulur (ENTER ile onayla). Oyun bitince final GANO'n ve ilk 5 sıralama gösterilir; skorlar `leaderboard.json` dosyasında saklanır (git'e girmez).

## Akış
İYTE Yazılım Topluluğu intro ekranı → "bir tuşa bas" → isim → oyun → transkript/leaderboard.

## Notlar
- Yüz kaybolursa oyun durur (haksız ölüm olmaz).
- **Adil oyun:** Gizli bir kaçış koridoru ekran boyunca yavaşça süzülür; engeller o koridorun geçeceği yere doğmaz. Yani her an ulaşılabilir bir yol vardır. Otomatik test koridoru izleyen oyuncunun hiç ölmediğini doğrular.
  Bu garanti **normal engeller** içindir; boss saldırıları ayrı kurallıdır (nişan yelpazesi oyuncunun iki yanından geçer, yağmurdaki boşluk oyuncuya yakın açılır, her saldırı öncesi uyarı verilir) ve bunlar için ayrı bir kazanılabilirlik testi yoktur.
- Zorluk tavanı düşük tutulur (engel hızı en fazla 1.5x), uzaktan oynayanın çarpışma kutusu çok küçülmez.
- Zorluk ve skor ayarları dosyanın başındaki sabitlerden değiştirilebilir.
- Yüzüne sevimli bir **domuz burnu**, domuz kulakları ve allık eklenir; burun kafanı oynattıkça ezilir, çarpınca başında yıldızlar döner.

## Proje yapısı
`iyte_kacis.py` yalnızca başlatıcıdır; kod `hocam_domuz/` paketindedir:

| Dosya | Sorumluluk |
|---|---|
| `config.py` | Tüm sabitler: ayarlar, renkler, durumlar, engel/boss tabloları |
| `tracking.py` | Kamera, model indirme, yüz (`FaceTracker`) ve vücut (`PoseTracker`) takibi |
| `entities.py` | Engeller, boss, kalem çocuk, parçacıklar, harf notu |
| `game.py` | Oyun durum makinesi: oyuncu, engel üretimi, kaçış koridoru, çarpışma, skor |
| `boss.py` | MATH 255 boss'u ve kalem çocuk (`Game`'e karışan mixin) |
| `leaderboard.py` | Skor tablosu (JSON) |
| `sprites.py` | Engel/boss/kalem çocuk sprite çizimleri (yüksek çözünürlük; arcade modunda piksel-art'a çevrilir) |
| `pixel.py` | 8-bit motoru: 32 renkli palet, sprite piksellendirme, 5x7 bitmap font (Türkçe karakterli) |
| `retro_bg.py`, `retro.py` | Pixel-art İYTE kampüsü arka planı; arcade avatar, HUD ve ekranlar |
| `drawing.py` | Yazı (Türkçe yedekli), yuvarlak kutu, vinyet gibi çizim yardımcıları |
| `world.py`, `hud.py`, `screens.py`, `render.py` | Klasik görünüm: oyun dünyası, arayüz, tam ekran menüler ve ana çizim |
| `app.py` | Ana döngü ve komut satırı |
| `selftest.py` | Kamerasız otomatik test (görüntüleri geçici klasöre yazar) |

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
