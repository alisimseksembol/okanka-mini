"""
Okanka Mini - Turkish pretraining data preparation.

Builds nanochat-compatible pretraining shards from:
  * HuggingFaceFW/fineweb-2, config tur_Latn  (ODC-By 1.0 + Common Crawl ToU)
  * wikimedia/wikipedia, config 20231101.tr   (CC BY-SA 3.0/4.0 + GFDL)

Output format (exactly what nanochat/dataloader.py expects):
  $NANOCHAT_BASE_DIR/base_data_tr/shard_00000.parquet ... shard_NNNNN.parquet  (train)
  $NANOCHAT_BASE_DIR/base_data_tr/shard_99999.parquet                          (val, always sorted last)
  - one string column named "text"
  - row groups of --row-group-size documents (nanochat iterates/DDP-shards by row group)
  - zstd compression
  - manifest.json describing the shards (sources, chars, licenses)

Usage examples:
  # small local sample (tokenizer training), using already-downloaded local files
  python -m okanka.prepare_pretrain_data --target-chars 1_500_000_000 \
      --fw2-local data/raw/fw2_tur_train_000_00000.parquet \
      --wiki-local data/raw/wiki_tr_0.parquet data/raw/wiki_tr_1.parquet \
      --val-local data/raw/fw2_tur_test.parquet

  # full Kaggle run (downloads from HF over HTTPS, no token needed), uploads shards to an HF dataset repo
  python -m okanka.prepare_pretrain_data --target-chars 18_000_000_000 --fw2-files 0 1 \
      --upload-repo <user>/okanka-mini-data --delete-after-upload
"""

import os
import json
import time
import random
import argparse

import requests
import pyarrow as pa
import pyarrow.parquet as pq

FW2_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/tur_Latn/train/000_{idx:05d}.parquet"
FW2_TEST_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/tur_Latn/test/000_00000.parquet"
WIKI_URLS = [
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.tr/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.tr/train-00001-of-00002.parquet",
]
VAL_SHARD_NAME = "shard_99999.parquet"


def default_out_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanochat"))
    return os.path.join(base, "base_data_tr")


def download(url, dest):
    """Download url -> dest with resume-friendly temp file and retries."""
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    for attempt in range(1, 6):
        try:
            print(f"[download] {url} -> {dest} (attempt {attempt})", flush=True)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"[download] failed: {e}", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}")


def clean(text):
    # light normalization only; FineWeb-2 is already filtered + deduplicated
    text = text.replace("\r\n", "\n").replace("\x00", "").strip()
    return text


def iter_parquet_texts(path, kind, min_chars):
    pf = pq.ParquetFile(path)
    cols = ["text"] + (["title"] if kind == "wiki" else [])
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        texts = t.column("text").to_pylist()
        titles = t.column("title").to_pylist() if kind == "wiki" else [None] * len(texts)
        for title, text in zip(titles, texts):
            if not text:
                continue
            text = clean(text)
            if kind == "wiki" and title:
                text = f"{title}\n\n{text}"
            if len(text) < min_chars:
                continue
            yield text


class ShardWriter:
    def __init__(self, out_dir, chars_per_shard, row_group_size, uploader=None, start_idx=None):
        self.out_dir = out_dir
        self.chars_per_shard = chars_per_shard
        self.row_group_size = row_group_size
        self.uploader = uploader
        self.buf, self.buf_chars = [], 0
        os.makedirs(out_dir, exist_ok=True)
        existing = sorted(f for f in os.listdir(out_dir) if f.startswith("shard_") and f.endswith(".parquet") and f != VAL_SHARD_NAME)
        self.shard_idx = len(existing) if start_idx is None else start_idx
        self.written = []

    def add(self, text):
        self.buf.append(text)
        self.buf_chars += len(text)
        if self.buf_chars >= self.chars_per_shard:
            self.flush()

    def flush(self, name=None):
        if not self.buf:
            return
        random.shuffle(self.buf)  # local shuffle (mixes wiki + web inside a shard)
        name = name or f"shard_{self.shard_idx:05d}.parquet"
        path = os.path.join(self.out_dir, name)
        table = pa.Table.from_pydict({"text": self.buf})
        pq.write_table(table, path + ".tmp", row_group_size=self.row_group_size, compression="zstd")
        os.replace(path + ".tmp", path)
        print(f"[shard] wrote {name}: {len(self.buf):,} docs, {self.buf_chars:,} chars", flush=True)
        self.written.append({"name": name, "docs": len(self.buf), "chars": self.buf_chars})
        if name != VAL_SHARD_NAME:
            self.shard_idx += 1
        self.buf, self.buf_chars = [], 0
        if self.uploader is not None:
            self.uploader(path)


def make_uploader(repo_id, delete_after):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    def _up(path):
        api.upload_file(path_or_fileobj=path, path_in_repo=os.path.basename(path), repo_id=repo_id, repo_type="dataset",
                        commit_message=f"add {os.path.basename(path)}")
        print(f"[upload] {os.path.basename(path)} -> {repo_id}", flush=True)
        if delete_after:
            os.remove(path)
    return _up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=default_out_dir())
    p.add_argument("--target-chars", type=int, default=18_000_000_000, help="total train chars to write (~4.6 chars/token for Turkish with our 32K tokenizer)")
    p.add_argument("--chars-per-shard", type=int, default=250_000_000)
    p.add_argument("--row-group-size", type=int, default=1024)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--wiki-ratio", type=float, default=0.08, help="approx. fraction of docs drawn from Wikipedia while it lasts")
    p.add_argument("--wiki-epochs", type=int, default=2, help="how many times Wikipedia may be repeated (high quality -> upweight)")
    p.add_argument("--fw2-files", type=int, nargs="*", default=[0, 1], help="FineWeb-2 tur_Latn train file indices (0..29), each ~4.8GB / ~15B chars")
    p.add_argument("--fw2-local", nargs="*", default=None, help="use these local FineWeb-2 parquet files instead of downloading")
    p.add_argument("--wiki-local", nargs="*", default=None)
    p.add_argument("--val-local", default=None, help="local FineWeb-2 tur_Latn test parquet")
    p.add_argument("--val-chars", type=int, default=60_000_000)
    p.add_argument("--download-dir", default=os.path.join("data", "raw"))
    p.add_argument("--delete-raw", action="store_true", help="delete each downloaded raw FineWeb-2 file after use (saves disk on Kaggle)")
    p.add_argument("--upload-repo", default=None, help="HF dataset repo to upload shards to (needs HF_TOKEN)")
    p.add_argument("--delete-after-upload", action="store_true")
    p.add_argument("--start-shard-idx", type=int, default=None, help="first train shard index (default: count of existing local shards)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    random.seed(args.seed)

    uploader = make_uploader(args.upload_repo, args.delete_after_upload) if args.upload_repo else None
    writer = ShardWriter(args.out_dir, args.chars_per_shard, args.row_group_size, uploader, args.start_shard_idx)

    # --- validation shard (FineWeb-2 official test split) -------------------------
    val_path = args.val_local or download(FW2_TEST_URL, os.path.join(args.download_dir, "fw2_tur_test.parquet"))
    if not os.path.exists(os.path.join(args.out_dir, VAL_SHARD_NAME)):
        n = 0
        for t in iter_parquet_texts(val_path, "fw2", args.min_chars):
            writer.buf.append(t); writer.buf_chars += len(t); n += len(t)
            if n >= args.val_chars:
                break
        writer.flush(name=VAL_SHARD_NAME)

    # --- wikipedia (held in memory, it is only ~1B chars) ---------------------------
    wiki_paths = args.wiki_local or [download(u, os.path.join(args.download_dir, f"wiki_tr_{i}.parquet")) for i, u in enumerate(WIKI_URLS)]
    wiki_docs = [t for wp in wiki_paths for t in iter_parquet_texts(wp, "wiki", args.min_chars)]
    print(f"[wiki] {len(wiki_docs):,} articles, {sum(map(len, wiki_docs)):,} chars", flush=True)
    wiki_pool = []
    for _ in range(args.wiki_epochs):
        ep = wiki_docs[:]
        random.shuffle(ep)
        wiki_pool.extend(ep)
    del wiki_docs
    wiki_pool.reverse()  # pop() from the end

    # --- fineweb-2 interleaved with wikipedia --------------------------------------
    if args.fw2_local:
        fw2_sources = [(pth, False) for pth in args.fw2_local]
    else:
        fw2_sources = [(FW2_URL.format(idx=i), True) for i in args.fw2_files]
    total = 0
    done = False
    stats = {"fw2_docs": 0, "fw2_chars": 0, "wiki_docs": 0, "wiki_chars": 0}
    for src, remote in fw2_sources:
        path = download(src, os.path.join(args.download_dir, os.path.basename(src).replace("000_", "fw2_tur_train_000_"))) if remote else src
        for t in iter_parquet_texts(path, "fw2", args.min_chars):
            writer.add(t); total += len(t)
            stats["fw2_docs"] += 1; stats["fw2_chars"] += len(t)
            if wiki_pool and random.random() < args.wiki_ratio / (1 - args.wiki_ratio):
                w = wiki_pool.pop()
                writer.add(w); total += len(w)
                stats["wiki_docs"] += 1; stats["wiki_chars"] += len(w)
            if total >= args.target_chars:
                done = True
                break
        if remote and args.delete_raw and os.path.exists(path):
            os.remove(path)
        if done:
            break
    writer.flush()

    manifest = {
        "format": "nanochat parquet shards, column 'text'; last file in sorted order (shard_99999) is validation",
        "sources": {
            "HuggingFaceFW/fineweb-2 (tur_Latn)": "ODC-By 1.0; also subject to Common Crawl Terms of Use",
            "wikimedia/wikipedia (20231101.tr)": "CC BY-SA (3.0/4.0) + GFDL; attribution: Wikipedia contributors",
        },
        "stats": stats,
        "train_chars_written_this_run": total,
        "shards": writer.written,
    }
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if uploader is not None:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=mpath, path_in_repo="manifest.json", repo_id=args.upload_repo, repo_type="dataset")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
