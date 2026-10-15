# TRAINING_PLAN — Okanka Mini

Tarih: 2026-10-16. Tüm sayılar bu makinede ölçülen değerlere veya açıkça yazılmış varsayımlara dayanır.
Kaggle'daki ilk oturum gerçek `tok/sec` ve `eta` değerlerini log'a basar; plan ona göre güncellenmelidir.

## 1. Seçilen model: **d16** (nanochat `--depth=16`)

| | d12 (yedek plan) | **d16 (seçilen)** |
|---|---:|---:|
| n_embd / heads / katman | 768 / 6 / 12 | **1024 / 8 / 16** |
| Transformer matrisleri | 84.9M | **201.3M** |
| lm_head | 25.2M | **33.6M** |
| "Scaling" parametreleri (matrisler + lm_head; nanochat'in token bütçesi bunu kullanır) | 110.1M | **234.9M** |
| wte (girdi embedding) | 25.2M | 33.6M |
| value embeddings (ResFormer; ucuz, sadece tablo okuma) | 151.0M | 268.4M |
| Toplam parametre (diskteki) | 286M | **537M** |
| FLOPs / token (T=1024) | 0.774 GFLOP | **1.611 GFLOP** |
| Context | 1024 | 1024 (eğitim; T4 belleği ve hız için) |
| Vocab | 32,768 | 32,768 |

Neden d16: "birkaç haftalık ücretsiz kota" için gerçekçi en büyük boyut. d18+ hem T4 belleğine sığmakta zorlanır
(fp32 master ağırlık + Adam durumları + value embedding tabloları) hem de 2-4 haftayı aşar. d12 ~1 haftada biter
ama belirgin şekilde daha zayıf olur; ilk Kaggle oturumunda d16'nın ETA'sı > 80 saat çıkarsa `DEPTH = 12` yapılır.

### Bellek (T4, 16 GB, 2 GPU)
Yerel ölçüm (RTX 5050, fp16 + GradScaler, compile kapalı): d12 B=2 T=1024 tepe 5.6 GB, B=4 tepe 7.0 GB
(örnek başına ~0.7 GB aktivasyon); d16 B=1 statik 2.2 GB model + optimizer durumları ile tepe ~11 GB (tek GPU).
2xT4'te nanochat optimizer durumlarını rank'lara böldüğü için GPU başına ~6 GB statik + B=4 için ~5 GB aktivasyon
≈ 11-12 GB → **d16 için `DEVICE_BS=4`**, d12 için 8. OOM olursa yarıya indir (grad accumulation toplam batch'i korur).

## 2. Token bütçesi

nanochat'in kendi ölçekleme yasası analizi `--target-param-data-ratio=12` (token : scaling-param) ile compute-optimal
kabul ediyor (Chinchilla'nın ~20'si farklı bir parametre sayımı kullanır). Seçim: **oran 12**.

```
tokens = 12 × 234.9M = 2.82B token        (d16)
toplam FLOPs = 2.82e9 × 1.611e9 ≈ 4.54e18
toplam batch (otomatik, nanochat Power-Lines kuralı) = 524,288 token/adım → ≈ 5,380 adım
mikro-batch = 4 × 1024 × 2 GPU = 8,192 token → grad accumulation 64
```

Veri ihtiyacı (ölçülen): bizim shard'larda **4.55 karakter/token**; nanochat'in BOS-hizalı best-fit paketlemesi
T=1024'te uzun dokümanları kırptığı için token'ların sadece **~%46'sı** eğitime girer (simülasyonla ölçüldü; T=2048'de %58).
Bu bir *hesaplama* kaybı değil, sadece daha fazla ham metin demek:
`2.82e9 × 4.55 / 0.46 × 1.1 ≈ 30.7B karakter` → FineWeb-2 tur_Latn'in 3 dosyası (her biri ~15B karakter; toplam 30 dosya,
~95M doküman, 41.9B kelime var) + Türkçe Vikipedi (873M karakter, 2 epoch'a kadar). ≈ 123 shard × ~115 MB ≈ 14 GB.

## 3. Süre tahmini (Kaggle 2xT4)

Varsayım: T4 fp16 tensor tepe 65 TFLOPS; küçük model + SDPA + Kaggle'daki T4 güç sınırı ile **%18-35 MFU** →
GPU başına 12-23 TFLOPS, 2 GPU'da ~%90 ölçekleme → **21-41 TFLOPS** etkin.

| Senaryo | tok/s (2xT4) | d16 oran 12 (2.82B tok) | d12 oran 12 (1.32B) | d12 oran 20 (2.2B) | d16 oran 20 (4.7B) |
|---|---:|---:|---:|---:|---:|
| iyimser (41 TFLOPS) | 25.5k | **31 saat** | 7 saat | 12 saat | 51 saat |
| orta (29 TFLOPS) | 18k | **44 saat** | 10 saat | 16 saat | 73 saat |
| kötümser (21 TFLOPS) | 13k | **60 saat** | 14 saat | 23 saat | 100 saat |

Oturum başına ~10.75 saat eğitim (`TIME_BUDGET_MIN=645`; kurulum + veri indirme + son kayıt/yükleme için ~1 saat pay).

**d16 toplam: ~31-60 GPU-saat ön eğitim + ~1-1.5 saat SFT + ~0.5 saat değerlendirme/yayın
≈ 4-6 Kaggle oturumu ≈ 30 saat/haftalık kota ile 1.5-2.5 hafta.** (Veri hazırlama notebook'u CPU'da çalışır, GPU kotası yemez.)

Kontrol noktası her 100 adımda (~45-60 dk) alınır: model (2.1 GB fp32) + optimizer parçaları (~3.5 GB) + meta;
HF'deki özel `okanka-mini-checkpoints` deposuna arka planda yüklenir, sadece en son adım tutulur ve git geçmişi
sıkıştırılır (`super_squash_history`) ki depolama büyümesin. Oturum kesilirse en fazla ~1 saatlik iş kaybolur.

## 4. Beklenen kalite (dürüst)

* d16 / 2.8B Türkçe token ≈ GPT-2 small-medium sınıfı bir "gevezelik eden" model.
* **Yapabilecekleri:** dilbilgisi büyük ölçüde doğru, basit Türkçe cümleler ve kısa paragraflar; sohbet formatına
  uyma; kimliğini söyleme ("Ben Okanka Mini'yim, Seozit.com'un ..."); Vikipedi tarzı kısa tanımlar.
* **Yapamayacakları:** güvenilir bilgi (sık sık uydurur — yanlış tarih, yanlış kişi), çok adımlı akıl yürütme,
  matematik (2+2'yi bile şaşırabilir), kod, uzun tutarlı metin, talimatları ince ayrıntısıyla izleme, İngilizce.
* SFT verisi küçük (~15K Türkçe konuşma + 260 kimlik) → cevaplar kısa ve kalıplı olur.
* Kısacası: "çalışan, sıfırdan yerli bir demo"; ürün kalitesinde asistan değil. Arayüzde bunu belirtmek gerekir.

## 5. Ücretli GPU ile ne olur? ($100-$300)

nanochat README'sine göre 8xH100 düğümü ≈ **$24/saat** (~$3/GPU/saat); GPT-2 seviyesi (d24-d26, İngilizce CORE ≈ 0.26)
2026'da ~1.7-3 saatte (~$40-75) eğitiliyor. 8xH100 bf16 + FA3 + (opsiyonel fp8) ile ~%40 MFU ≈ 3.2 PFLOPS:

| Bütçe | Süre (8xH100) | Hesap | Mantıklı model | Beklenen Türkçe kalite |
|---|---|---|---|---|
| ~$10 | ~25 dk | 4.5e18 | bizim d16 planı (2 hafta Kaggle'ın tamamı) | yukarıdaki gibi |
| **$100** | ~4 saat | ~4.6e19 | **d24-d26** (~0.7-0.9B scaling param, ~8-9B token) | GPT-2 XL (1.5B) sınıfı: akıcı paragraflar, daha iyi genel kültür, yine sık uydurma; sohbet ve kimlik sağlam |
| **$300** | ~12 saat | ~1.4e20 | **d30-d32** (~1.4-1.7B scaling param, ~15-17B token) | belirgin daha iyi: basit soruları çoğu zaman doğru, daha tutarlı uzun cevaplar; hâlâ GPT-3.5'in çok altında |

Türkçe veri yeterli: FineWeb-2 tur_Latn ~60-70B token içeriyor (d32 için ~17B gerekir). Kod değişikliği gerekmez:
H100'de `NANOCHAT_DTYPE` otomatik bf16, FA3 otomatik açılır (`NANOCHAT_DISABLE_FA3`'ü set etme), `--window-pattern SSSL`
kullanılabilir, `runs/speedrun.sh` mantığıyla tek komut: `torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=26
--target-param-data-ratio=10 --device-batch-size=16 --fp8`, ardından `scripts.chat_sft --mixture=tr`. Tokenizer aynı kalır.
(Bu kurulum para gerektirir; şu an yapılmadı.)

## 6. Adım adım çalıştırma sırası

1. `kaggle/01_tokenizer_and_data` — **CPU** oturumu, bir kez (~1-1.5 saat): 3 FineWeb-2 dosyası + Vikipedi →
   ~123 shard → HF `<kullanıcı>/okanka-mini-data` (özel). SFT/kimlik jsonl'leri de aynı depoya.
2. `kaggle/02_pretrain` — **GPU T4 x2**, bitene kadar her seferinde "Save & Run All" (4-6 kez). Her oturum en son
   checkpoint'i çeker, ~10.75 saat eğitir, kaydeder, yükler, kendini kapatır. GPU tipini ve ayarları değiştirme.
3. `kaggle/03_midtrain_sft` — GPU T4 x2, bir kez (~1 saat).
4. `kaggle/04_eval_and_export` — Türkçe testler + `<kullanıcı>/okanka-mini` model deposuna yayın.
5. `YAYINLAMA.md` — Space + site.
