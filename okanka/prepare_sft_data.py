"""
Build the Turkish SFT (chat) data for Okanka Mini in nanochat's CustomJSON format
(one JSON array of {"role","content"} messages per line).

Sources (both human-written, both Apache-2.0, commercial use allowed):
  * CohereLabs/aya_dataset (formerly CohereForAI/aya_dataset), language == "Turkish"
      human-annotated prompt/completion pairs (~4K Turkish rows)
  * OpenAssistant/oasst2, lang == "tr"
      human-written conversation trees; we keep the best-ranked path of each tree

Outputs:
  data/sft/sft_tr_train.jsonl, data/sft/sft_tr_val.jsonl   (not committed; regenerate with this script)

Run: python -m okanka.prepare_sft_data [--raw-dir data/raw/sft] [--out-dir data/sft]
"""
import os
import json
import random
import argparse
from collections import defaultdict

import requests
import pyarrow.parquet as pq

AYA_URL = "https://huggingface.co/api/datasets/CohereLabs/aya_dataset/parquet/default/train/0.parquet"
OASST2_URLS = {
    "oasst2_train.parquet": "https://huggingface.co/api/datasets/OpenAssistant/oasst2/parquet/default/train/0.parquet",
    "oasst2_val.parquet": "https://huggingface.co/api/datasets/OpenAssistant/oasst2/parquet/default/validation/0.parquet",
}


def fetch(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {url}")
    r = requests.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def ok_text(t, min_len=2, max_len=6000):
    return isinstance(t, str) and min_len <= len(t.strip()) <= max_len


def load_aya(path):
    t = pq.read_table(path, columns=["inputs", "targets", "language"])
    convs = []
    for inp, tgt, lang in zip(*(t.column(c).to_pylist() for c in ["inputs", "targets", "language"])):
        if lang != "Turkish" or not ok_text(inp) or not ok_text(tgt):
            continue
        convs.append([{"role": "user", "content": inp.strip()}, {"role": "assistant", "content": tgt.strip()}])
    return convs


def load_oasst2(paths):
    rows = []
    for p in paths:
        t = pq.read_table(p, columns=["message_id", "parent_id", "text", "role", "lang", "rank", "deleted", "synthetic", "review_result"])
        d = t.to_pydict()
        for i in range(t.num_rows):
            if d["lang"][i] != "tr" or d["deleted"][i] or d["synthetic"][i]:
                continue
            if d["review_result"][i] is False:
                continue
            rows.append({k: d[k][i] for k in d})
    by_id = {r["message_id"]: r for r in rows}
    children = defaultdict(list)
    for r in rows:
        if r["parent_id"] is not None:
            children[r["parent_id"]].append(r)
    roots = [r for r in rows if r["parent_id"] is None and r["role"] == "prompter"]
    convs = []

    def best_path(node):
        path = [node]
        while children.get(node["message_id"]):
            kids = sorted(children[node["message_id"]], key=lambda c: (c["rank"] if c["rank"] is not None else 99))
            node = kids[0]
            path.append(node)
        return path

    for root in roots:
        path = best_path(root)
        msgs = []
        for m in path:
            role = "user" if m["role"] == "prompter" else "assistant"
            if msgs and msgs[-1]["role"] == role:
                break
            if not ok_text(m["text"]):
                break
            msgs.append({"role": role, "content": m["text"].strip()})
        if msgs and msgs[-1]["role"] == "user":
            msgs = msgs[:-1]
        if len(msgs) >= 2:
            convs.append(msgs)
    _ = by_id
    return convs


WIKI_URLS = [
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.tr/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.tr/train-00001-of-00002.parquet",
]
WIKI_Q = ["{t} nedir?", "{t} hakkında bilgi verir misin?", "{t} hakkında kısa bilgi ver.", "Bana {t} hakkında bir şeyler anlat.",
          "{t} ile ilgili ne biliyorsun?", "{t} konusunu kısaca açıklar mısın?", "{t} hakkında ne söyleyebilirsin?"]


def make_wiki_qa(paths, n, rng, max_answer_chars=700):
    """Template QA from Turkish Wikipedia lead paragraphs (CC BY-SA). Question = template(title), answer = lead sentences."""
    import re
    out = []
    for p in paths:
        t = pq.read_table(p, columns=["title", "text"])
        titles, texts = t.column("title").to_pylist(), t.column("text").to_pylist()
        idx = list(range(len(titles)))
        rng.shuffle(idx)
        for i in idx:
            title, text = titles[i], texts[i]
            if not title or not text or "(anlam ayrımı)" in title or len(text) < 300:
                continue
            lead = text.strip().split("\n")[0].strip()
            if len(lead) < 120 or lead.endswith(":"):
                continue
            sents = re.split(r"(?<=[.!?])\s+", lead)
            ans = ""
            for s_ in sents:
                if len(ans) + len(s_) > max_answer_chars:
                    break
                ans = (ans + " " + s_).strip()
            if len(ans) < 80:
                continue
            q = rng.choice(WIKI_Q).format(t=title)
            out.append([{"role": "user", "content": q}, {"role": "assistant", "content": ans}])
            if len(out) >= n // len(paths) * (paths.index(p) + 1):
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join("data", "raw", "sft"))
    ap.add_argument("--out-dir", default=os.path.join("data", "sft"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wiki-qa", type=int, default=12000, help="number of template QA conversations from Turkish Wikipedia leads (0 = off)")
    ap.add_argument("--wiki-dir", default=os.path.join("data", "raw"))
    args = ap.parse_args()

    aya = load_aya(fetch(AYA_URL, os.path.join(args.raw_dir, "aya_train.parquet")))
    oas = load_oasst2([fetch(u, os.path.join(args.raw_dir, n)) for n, u in OASST2_URLS.items()])
    print(f"aya Turkish: {len(aya)} conversations | oasst2 Turkish: {len(oas)} conversations")

    rng = random.Random(args.seed)
    wqa = []
    if args.wiki_qa > 0:
        wpaths = [fetch(u, os.path.join(args.wiki_dir, f"wiki_tr_{i}.parquet")) for i, u in enumerate(WIKI_URLS)]
        wqa = make_wiki_qa(wpaths, args.wiki_qa, rng)
        print(f"wikipedia template QA: {len(wqa)} conversations")
    allc = aya + oas + wqa
    rng.shuffle(allc)
    n_val = max(50, int(len(allc) * args.val_frac))
    val, train = allc[:n_val], allc[n_val:]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in [("sft_tr_train.jsonl", train), ("sft_tr_val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for c in data:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"wrote {len(data)} -> {os.path.join(args.out_dir, name)}")


if __name__ == "__main__":
    main()
