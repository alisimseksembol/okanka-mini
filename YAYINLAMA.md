# YAYINLAMA — Okanka Mini'yi eğitimden siteye kadar yayınlama

Bu belge, eğitim bittikten sonra modeli Hugging Face'e koymayı, Space'i açmayı ve okanka web sitesine
bağlamayı adım adım anlatır. (Eğitimin kendisi için: `TRAINING_PLAN.md` ve `kaggle/` not defterleri.)

## 0. Hesaplar ve anahtarlar (bir kere)

1. **Hugging Face hesabı** aç: https://huggingface.co/join (ücretsiz).
2. **Yazma (write) token'ı** oluştur: https://huggingface.co/settings/tokens -> *Create new token* ->
   tür **Write** -> adı örn. `okanka-kaggle`. Token'ı kimseyle paylaşma, git'e koyma.
3. **Kaggle hesabı** aç: https://www.kaggle.com -> *Settings* -> **Phone verification** (GPU ve internet
   için şart).
4. Kaggle'da her not defterinde: *Add-ons -> Secrets* ->
   - `HF_TOKEN` = yukarıdaki HF write token
   - `HF_USERNAME` = HF kullanıcı adın
   - (kod deposu özelse) `GITHUB_TOKEN` = sadece okuma yetkili GitHub token'ı
   ve her secret'ın yanındaki kutucuğu o not defteri için **işaretle**.

## 1. Eğitim (özet)

Sırasıyla `kaggle/01_tokenizer_and_data` (CPU), `kaggle/02_pretrain` (GPU T4 x2, bitene kadar tekrar tekrar),
`kaggle/03_midtrain_sft` (GPU) çalıştırılır. Her not defterinin en üstündeki `REPO_URL` satırına bu kodun
GitHub adresini yaz.

## 2. Modeli Hugging Face'e "okanka-mini" olarak yayınla

`kaggle/04_eval_and_export` not defterini çalıştır (GPU T4 veya CPU, internet açık):

* Türkçe test soruları sorulur, çıktılar not defterinde görünür (kaliteyi kendi gözünle kontrol et).
* `okanka.export_model` şunları `https://huggingface.co/<HF_USERNAME>/okanka-mini` deposuna yükler:
  - `chatsft_checkpoints/d16/model_XXXXXX.pt` + `meta_XXXXXX.json` (sadece ağırlıklar, optimizer yok)
  - `tokenizer/tokenizer.pkl`, `tokenizer/token_bytes.pt`
  - `LICENSE` (MIT), `NOTICE` (nanochat atfı), `DATA_LICENSES.md`
  - `README.md` = Türkçe + İngilizce model kartı (`okanka/model_card.md` şablonundan)
* `MAKE_PUBLIC = True` ise depo **herkese açık** olur (Space token'sız okuyabilsin diye önerilir).
  Gizli tutmak istersen `False` yap ve 3. adımda Space'e `HF_TOKEN` (read) secret'ı ekle.

Elle yapmak istersen (bilgisayarda):

```bash
export HF_TOKEN=hf_xxx
NANOCHAT_BASE_DIR=/yol/nanochat python -m okanka.export_model --source sft --model-tag d16 --out export/okanka-mini --push <HF_USERNAME>/okanka-mini
```

## 3. Space'i oluştur (Gradio + ZeroGPU, ücretsiz hesap)

> Not (2026-10): Hugging Face, yeni **Docker** ve **CPU basic Gradio** Space'leri için artık ücretli plan istiyor.
> Ücretsiz kişisel hesaplar en fazla **2 adet ZeroGPU Gradio Space** açabilir ve bunlar **herkese açık (public)**
> olmak zorundadır. Bu yüzden Space'imiz **Gradio SDK + ZeroGPU** olarak tasarlandı.

1. https://huggingface.co/new-space ->
   - Owner: kendi kullanıcı adın, Space name: `okanka-mini-chat`
   - License: MIT
   - **SDK: Gradio**
   - **Hardware: ZeroGPU** (Nvidia, ücretsiz)
   - Visibility: **Public**
   - *Create Space*.
2. Kodu yükle (bilgisayarda, bu repo kökünde):
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_xxx      # write token
   python space/deploy_space.py --space <HF_USERNAME>/okanka-mini-chat --model-repo <HF_USERNAME>/okanka-mini
   ```
   Bu, `space/app.py`, `requirements.txt`, `README.md`, `API.md` ve `nanochat/` paketini Space'e yükler.
   (İstersen Claude Code'a "Space'i deploy et" demen yeterli.)
3. Space -> *Settings* -> **Variables and secrets**:
   - Variable `MODEL_REPO` = `<HF_USERNAME>/okanka-mini` (deploy script'i zaten ayarlamaya çalışır)
   - Secret `OKANKA_MINI_KEY` = uzun rastgele bir parola (opsiyonel ama önerilir; siteyle paylaşılan sır)
   - Secret `HF_TOKEN` = read token (sadece model deposu gizliyse)
4. Space'in *Logs* sekmesinde "Running on ..." görünene kadar bekle. Sonra test et:
   ```bash
   SPACE=https://<HF_USERNAME>-okanka-mini-chat.hf.space
   curl $SPACE/health
   EID=$(curl -s -X POST $SPACE/gradio_api/call/chat -H "Content-Type: application/json" \
     -d '{"data":["[{\"role\":\"user\",\"content\":\"Merhaba, sen kimsin?\"}]",128,"<OKANKA_MINI_KEY>"]}' | python -c "import sys,json;print(json.load(sys.stdin)['event_id'])")
   curl -N $SPACE/gradio_api/call/chat/$EID
   ```
   API'nin tam tarifi: `space/API.md`.

**Space URL'si**: `https://<HF_USERNAME>-okanka-mini-chat.hf.space` (huggingface.co/spaces/... adresi DEĞİL).

## 4. okanka web sitesine bağla (Vercel)

Vercel -> proje -> *Settings* -> **Environment Variables** (Production + Preview):

| Değişken | Değer |
|---|---|
| `OKANKA_MINI_URL` | `https://<HF_USERNAME>-okanka-mini-chat.hf.space` (sonunda `/` olmadan) |
| `OKANKA_MINI_KEY` | Space'teki `OKANKA_MINI_KEY` ile **aynı** değer (Space'te yoksa boş bırak) |
| `NEXT_PUBLIC_OKANKA_MINI_ENABLED` | `true` (model seçicide "Okanka Mini" seçeneğini açar) |

Sonra *Deployments* -> son deployment -> **Redeploy** (NEXT_PUBLIC_ değişkenleri build sırasında okunur,
yeniden build şart). Site, sunucu tarafında `POST {OKANKA_MINI_URL}/gradio_api/call/chat` + SSE ile
konuşur; anahtar tarayıcıya hiç gitmez.

Kapatmak için `NEXT_PUBLIC_OKANKA_MINI_ENABLED` değerini `false` yapıp yeniden deploy etmen yeterli.

## 5. Bilmen gerekenler

* ZeroGPU Space'i bir süre kullanılmazsa uyur; ilk istek 30-120 sn sürebilir. Site bunu "model uyanıyor" diye göstermeli.
* ZeroGPU'nun günlük ücretsiz GPU dakika kotası vardır (çağıran kullanıcıya göre). Kota dolarsa API `error`
  olayı döner; site bu durumda Okanka Mini'yi geçici olarak devre dışı gösterebilir.
* Model küçüktür: hata yapar, uydurur, internete erişimi yoktur. Arayüzde bunu belirten kısa bir not göster.
