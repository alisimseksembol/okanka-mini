# PATCHES — changes to upstream nanochat

Upstream: https://github.com/karpathy/nanochat (MIT), forked at commit `92d63d4` ("clean up fragile code").
Git history is kept; `upstream` remote points to karpathy/nanochat, so `git fetch upstream && git merge upstream/master`
works (expect small conflicts in the files below). Every in-code change is tagged `[okanka patch]`.

## What upstream already supports (no patch needed)

* **fp16 + GradScaler** in `scripts/base_train.py` and `scripts/chat_sft.py` (`NANOCHAT_DTYPE=float16`). fp16 mode keeps
  embeddings in fp32 so GradScaler can unscale them. Upstream auto-detection picks **fp32** on pre-Ampere GPUs (T4),
  so our notebooks set `NANOCHAT_DTYPE=float16` explicitly on T4/V100 (and `bfloat16` on Ampere+).
* **SDPA fallback** when Flash Attention 3 is not available (`nanochat/flash_attention.py`). FA3 is Hopper/Ampere/Ada only;
  T4 (sm75) always uses PyTorch `scaled_dot_product_attention`. SDPA has no efficient sliding window, so we train with
  `--window-pattern L` (full context in every layer) as upstream recommends.
* **Resumable checkpoints**: `--resume-from-step`, optimizer state per rank, dataloader position (`pq_idx`, `rg_idx`, `epoch`)
  and loop state are saved in `meta_*.json`. We build on that.
* `torch.compile` can be disabled globally with the standard `TORCHDYNAMO_DISABLE=1` (used for P100, which Triton does not support).

## Patched upstream files

| File | Change | Why |
|---|---|---|
| `nanochat/dataset.py` | `DATA_DIR` = `$NANOCHAT_DATA_DIR` or `$NANOCHAT_BASE_DIR/base_data_tr` | Turkish shards instead of ClimbMix. Same parquet format (`text` column, last file = val). |
| `nanochat/tokenizer.py` | `get_tokenizer()` / `get_token_bytes()` look in `$NANOCHAT_TOKENIZER_DIR` -> `$NANOCHAT_BASE_DIR/tokenizer` -> `<repo>/tokenizer/` | The Turkish tokenizer trained on this machine is committed in `tokenizer/`, so Kaggle/Space need no tokenizer step. |
| `nanochat/tokenizer.py` (split pattern) | `SPLIT_PATTERN` drops the English contraction alternative `'(?i:[sdmt]\|ll\|ve\|re)`; upstream kept as `SPLIT_PATTERN_UPSTREAM` | Turkish apostrophe suffixes (`Ankara'da`, `1920'de`) were split as `'d`+`a`. Measured +0.1-2.8% compression, cleaner tokens. Only affects training a tokenizer; the committed tokenizer stores its own pattern. |
| `nanochat/flash_attention.py` | `NANOCHAT_DISABLE_FA3=1` skips the FA3 probe entirely | Avoids touching CUDA / downloading kernels at import (needed on HF ZeroGPU; harmless on T4). |
| `nanochat/common.py` | peak-FLOPS table: T4 65e12, P100 18.7e12, V100 125e12 | Correct MFU print on Kaggle GPUs. |
| `nanochat/engine.py` | calculator timeout uses `SIGALRM` only if it exists | Windows has no `SIGALRM` (local testing). |
| `nanochat/checkpoint_manager.py` | on CPU also upcast fp16 tensors to fp32 | fp16 checkpoints load on CPU. |
| `scripts/base_train.py` | new flags `--auto-resume`, `--time-budget-minutes`, `--hf-repo`, `--keep-local`, `--no-compile`; GradScaler state saved in meta and restored; after each save: barrier, prune old local checkpoints, push to HF in a background thread (joined before exit / next prune); time budget -> save + clean exit (rank-0 clock broadcast to all ranks); Turkish sample prompts | Kaggle sessions are capped at 12 h and disk is small; training must chain across sessions via the HF hub. |
| `scripts/chat_sft.py` | `--mixture tr` (default) = Turkish SFT jsonl x`--sft-epochs` + identity x`--identity-epochs` via the new `CustomJSON` task, val = Turkish val jsonl; ChatCORE (English benchmarks) disabled for `tr`; `--no-compile`. `--mixture upstream` keeps the original SmolTalk/MMLU/GSM8K mixture. | Turkish chat data + Okanka identity. |

## New files

| Path | Purpose |
|---|---|
| `tasks/customjson.py` | JSONL conversations task (like the identity task in older nanochat versions). |
| `okanka/prepare_pretrain_data.py` | FineWeb-2 tur_Latn + Turkish Wikipedia -> nanochat parquet shards (+ optional upload to an HF dataset repo). |
| `okanka/prepare_sft_data.py` | Aya (Turkish) + OASST2 (tr) + Wikipedia template QA -> `data/sft/*.jsonl`. |
| `okanka/make_identity.py` | Template-generated identity conversations -> `data/identity_tr.jsonl` (committed). |
| `okanka/hf_sync.py` | Pull/push checkpoints to a private HF model repo (keeps only the latest step, squashes history). |
| `okanka/tok_compare.py` | Compression comparison vs GPT-2 / cl100k / o200k / Turkish tokenizers. |
| `okanka/export_model.py`, `okanka/model_card.md` | Final export to the HF model repo `okanka-mini`. |
| `okanka/build_notebooks.py`, `kaggle/*` | Kaggle notebooks + `kernel-metadata.json`. |
| `space/*` | Gradio (ZeroGPU) Space with the `chat` API + `GET /health`. |
| `tokenizer/*` | The trained Turkish tokenizer (`tokenizer.pkl`, `token_bytes.pt`) + report. |

## Known caveats

* Resuming requires the **same number of GPUs** (optimizer state is sharded per rank) and the same
  `--depth/--max-seq-len/--device-batch-size/--target-param-data-ratio` (they determine the schedule).
* fp16 training: if loss spikes/NaNs appear, GradScaler skips those steps automatically; if it persists, rerun with
  `NANOCHAT_DTYPE=float32` (about 3-4x slower on T4).
* `scripts/chat_sft.py --num-iterations N` (upstream behaviour) counts *micro-batches*, not optimizer steps, and the LR
  multiplier can go negative past the end; leave it at the default (-1 = one pass over the mixture), as the notebooks do.
* The upstream CORE/ChatCORE evaluations are English; they are disabled for the Turkish runs (`--core-metric-every=-1`).
