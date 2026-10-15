---
title: Okanka Mini
emoji: 💬
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
python_version: "3.12"
pinned: false
license: mit
short_description: Sıfırdan eğitilmiş deneysel küçük Türkçe sohbet modeli
---

# Okanka Mini — Space

Gradio Space serving the Okanka Mini chat model (Seozit.com, trained from scratch with a nanochat fork).

* **Hardware:** create the Space with the **Gradio SDK** and select **ZeroGPU** hardware (the only free
  GPU option for personal accounts; a free account can host up to 2 ZeroGPU Spaces, which must be **public**).
  Docker / CPU-basic Spaces require a paid plan for new Spaces as of 2026-10.
* Without ZeroGPU (local run, or no `spaces` package / no CUDA) the app falls back to CPU automatically.
* API contract: see [API.md](API.md).

## Environment variables / secrets (Space Settings)

| name | required | meaning |
|---|---|---|
| `MODEL_REPO` | yes | HF model repo of the exported model, e.g. `<hf-user>/okanka-mini` |
| `OKANKA_MINI_KEY` | optional (secret) | shared secret; when set, every call must pass the same value as the 3rd input `key` |
| `HF_TOKEN` | only if `MODEL_REPO` is private (secret) | read token |
| `OKANKA_TEMPERATURE`, `OKANKA_TOP_K`, `OKANKA_MAX_TOKENS_CAP` | optional | sampling defaults 0.7 / 50 / 1024 |

## Deploy

The Space needs the `nanochat/` package next to `app.py`. `deploy_space.py` uploads both:

```bash
export HF_TOKEN=hf_xxx   # write token
python space/deploy_space.py --space <hf-user>/okanka-mini-chat
```

## Local test

```bash
pip install -r space/requirements.txt gradio
MODEL_DIR=export/okanka-mini python space/app.py      # or MODEL_REPO=<hf-user>/okanka-mini
curl http://localhost:7860/health
```
