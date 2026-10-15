# Okanka Turkish tokenizer (32K, trained from scratch)

Files (loaded by `nanochat.tokenizer.get_tokenizer()` automatically when no other tokenizer dir is configured):

| file | what |
|---|---|
| `tokenizer.pkl` | pickled `tiktoken.Encoding` (nanochat's native format): 32,759 BPE ranks + 9 special tokens = 32,768 |
| `token_bytes.pt` | byte length of every token id (for bits-per-byte evaluation) |
| `okanka32k.tiktoken` + `tokenizer_config.json` | the same tokenizer in plain tiktoken text format (base64 ranks) + pattern + special tokens, version-independent backup |
| `compression_report.md/.json` | comparison against other tokenizers |

## How it was trained (2026-10-16, on this Windows laptop)

* Library: nanochat's own **rustbpe** (`pip install rustbpe` has Windows wheels, so the HF `tokenizers` fallback was
  not needed) + tiktoken for inference — i.e. exactly upstream `scripts/tok_train.py`.
* Data: the first **2.0B characters** of our prepared shards: FineWeb-2 `tur_Latn` train file `000_00000` (~94.6% of
  chars) + Turkish Wikipedia 20231101.tr (~5.4%), docs capped at 10K chars. Validation shard (FineWeb-2 test split) excluded.
* Vocab: **32,768** (2^15), training time **177 s**.
* Special tokens (same as nanochat): `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`,
  `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>` (ids 32759-32767).
* Split pattern: nanochat's GPT-4-style pattern **without the English contraction rule** `'(?i:[sdmt]|ll|ve|re)`
  (see `nanochat/tokenizer.py`). With the English rule, Turkish suffixes after apostrophes were cut:
  `1920'de` → `'d` + `e`, `Ankara'da` → `'d` + `a`. Now `'de`, `'da`, `'daki`, `'nın` are single tokens.

## Compression on held-out Turkish (chars per token, higher = better)

| tokenizer | FineWeb-2 test (web) | Wikipedia sample | ğşıİçöü sentences |
|---|---:|---:|---:|
| **Okanka-32K (ours, Turkish pattern)** | **4.845** | **3.885** | **4.757** |
| Okanka-65K (same data/pattern, for comparison) | 5.259 | 4.186 | 5.247 |
| Okanka-32K with upstream English pattern | 4.839 | 3.856 | 4.627 |
| ytu-ce-cosmos/turkish-gpt2 (Turkish, ~50K vocab) | 5.016 | 3.879 | 4.942 |
| tiktoken o200k_base (GPT-4o) | 3.455 | 3.180 | 2.977 |
| tiktoken cl100k_base (GPT-4) | 2.575 | 2.550 | 2.223 |
| tiktoken gpt2 | 2.029 | 2.120 | 1.780 |

Ours needs **2.4x fewer tokens than GPT-2's tokenizer and 1.4x fewer than GPT-4o's** on Turkish web text.

## Why 32K and not 65K?

65K compresses ~8.5% better, but in nanochat the vocabulary size multiplies three tables: `wte`, `lm_head`
and the per-layer **value embeddings** (8 tables for d16). For d16 that is ~336M vocab-dependent parameters at 32K
vs ~671M at 65K: +1.3 GB fp32 weights, +1.3 GB grads, +2.7 GB Adam state — it would not fit a 16 GB T4 with a useful
batch, and the bigger softmax costs compute every token. +8.5% compression is not worth ~2x the memory on free GPUs.
nanochat also tuned its split pattern (`\p{N}{1,2}`) for 32K. For a paid 8xH100 run of d26+, 65K would be reasonable.

Reproduce:
```bash
export NANOCHAT_BASE_DIR=$PWD/data/nanochat_base
python -m okanka.prepare_pretrain_data --target-chars 2200000000 --fw2-files 0
python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768
python -m okanka.tok_compare
```
