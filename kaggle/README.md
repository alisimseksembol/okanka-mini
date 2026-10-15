# Kaggle notebooks

| Notebook | Accelerator | Runs | Does |
|---|---|---|---|
| `01_tokenizer_and_data` | **None (CPU)** — no GPU quota used | once (~1-1.5 h) | checks tokenizer, builds ~123 Turkish shards + SFT jsonl, uploads to `<HF_USERNAME>/okanka-mini-data` |
| `02_pretrain` | **GPU T4 x2** | 4-6 times, until done | resumable pretraining, checkpoints <-> `<HF_USERNAME>/okanka-mini-checkpoints` |
| `03_midtrain_sft` | GPU T4 x2 | once (~1 h) | Turkish SFT + identity |
| `04_eval_and_export` | GPU T4 (or CPU) | once | Turkish chat test, val bpb, publish `<HF_USERNAME>/okanka-mini` |

Before the first run: set `REPO_URL` in the first code cell of each notebook (or in `okanka/build_notebooks.py`
and regenerate with `python -m okanka.build_notebooks`). Kaggle secrets: `HF_TOKEN` (write), `HF_USERNAME`,
optional `GITHUB_TOKEN` (private repo). Internet must be ON (requires phone-verified Kaggle account).

Web UI: *Create -> New Notebook -> File -> Import Notebook* (upload the .ipynb), then *Settings*: accelerator,
internet on, attach secrets, then *Save Version -> Save & Run All (Commit)* (runs in the background, up to 12 h).

CLI alternative (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`): replace `KAGGLE_USERNAME` in each
`kernel-metadata.json`, then `kaggle kernels push -p kaggle/02_pretrain`. Accelerator type (T4 x2) and secrets still
have to be selected once in the web UI for each kernel.
