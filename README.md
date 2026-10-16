# Okanka Mini

**Okanka Mini**, Okanka ekibinin **tamamen sıfırdan** (rastgele ağırlıklarla başlayıp, hiçbir hazır modelin
ağırlıklarını kullanmadan) eğittiği deneysel, küçük bir **Türkçe** sohbet modelidir.

Kod tabanı, Andrej Karpathy'nin açık kaynak [nanochat](https://github.com/karpathy/nanochat) projesinin
(MIT lisanslı) bir çatalıdır (fork). Upstream git geçmişi korunmuştur; `upstream` remote'u karpathy/nanochat'i
gösterir. Orijinal nanochat README'si: [README.nanochat.md](README.nanochat.md). Atıf: [NOTICE](NOTICE).

## Ne değişti?

| Konu | nanochat (orijinal) | Okanka Mini |
|---|---|---|
| Ön eğitim verisi | ClimbMix (İngilizce) | FineWeb-2 `tur_Latn` + Türkçe Vikipedi |
| Tokenizer | İngilizce 32K BPE | **Türkçe 32K BPE**, bu makinede sıfırdan eğitildi, `tokenizer/` klasöründe |
| SFT verisi | SmolTalk, MMLU, GSM8K | Aya (Türkçe, insan yazımı), OASST2 (tr), Vikipedi şablon S-C, kimlik konuşmaları |
| Donanım | 8xH100, bf16, FA3 | Kaggle ücretsiz **2xT4**, fp16 + GradScaler, SDPA |
| Oturumlar | tek seferde | 12 saatlik Kaggle oturumlarına bölünmüş, checkpoint'ler HF Hub'a itiliyor/çekiliyor |
| Sunum | yerel web UI | Hugging Face **Gradio + ZeroGPU** Space, `chat` API'si |

Ayrıntılar: [PATCHES.md](PATCHES.md).

## Klasörler

```
okanka/             Türkçe veri hazırlama, SFT/kimlik verisi, HF senkronizasyonu, dışa aktarma
kaggle/             Kaggle not defterleri (01 veri, 02 ön eğitim, 03 SFT, 04 değerlendirme+yayın) + kernel-metadata.json
space/              Hugging Face Space (Gradio, ZeroGPU): app.py, API.md, deploy_space.py
tokenizer/          Eğitilmiş Türkçe tokenizer (tokenizer.pkl, token_bytes.pt) + sıkıştırma raporu
data/identity_tr.jsonl   Şablonlarla üretilmiş 260 kimlik konuşması
data/kanka_tr.jsonl      (isteğe bağlı) insan yazımı kanka tarzı sohbetler, bkz. KANKA_VERISI.md
nanochat/, scripts/, tasks/   nanochat çekirdeği (küçük yamalarla)
```

## Belgeler

* [TRAINING_PLAN.md](TRAINING_PLAN.md) — model boyutu, token bütçesi, Kaggle saat/hafta tahmini, beklenen kalite, ücretli GPU karşılaştırması
* [DATA_LICENSES.md](DATA_LICENSES.md) — tüm veri setleri ve lisansları, reddedilen setler ve nedenleri
* [PATCHES.md](PATCHES.md) — nanochat'e yapılan değişiklikler
* [YAYINLAMA.md](YAYINLAMA.md) — modeli HF'ye koyma, Space açma, siteye bağlama (adım adım)
* [space/API.md](space/API.md) — sitenin çağıracağı API sözleşmesi
* [tokenizer/README.md](tokenizer/README.md) — tokenizer eğitimi ve karşılaştırma

## Hızlı başlangıç (yerel)

```bash
python -m venv .venv && .venv/Scripts/activate          # Windows (Linux: source .venv/bin/activate)
pip install torch tiktoken rustbpe pyarrow filelock huggingface_hub requests
export NANOCHAT_BASE_DIR=$PWD/data/nanochat_base

python -m okanka.prepare_pretrain_data --target-chars 300000000 --fw2-files 0   # küçük örnek veri
python -m okanka.make_identity
python -m okanka.prepare_sft_data
# küçük bir duman testi (GPU varsa):
NANOCHAT_DTYPE=float16 python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=4 \
   --total-batch-size=8192 --num-iterations=50 --window-pattern=L --core-metric-every=-1 --eval-every=-1 --sample-every=-1 --no-compile
```

Gerçek eğitim Kaggle'da yapılır: [TRAINING_PLAN.md](TRAINING_PLAN.md) ve `kaggle/` klasörü.

## Lisans

Kod: MIT (nanochat © 2025 Andrej Karpathy; Okanka eklemeleri © 2026 Okanka). Veri lisansları için
[DATA_LICENSES.md](DATA_LICENSES.md).
