"""
Compare the Okanka Turkish tokenizer against other tokenizers on held-out Turkish text.

Metrics: chars/token and bytes/token (higher = better compression), and tokens for a fixed
set of Turkish sentences full of ğ ş ı İ ç ö ü.

Run: python -m okanka.tok_compare [--val-parquet $NANOCHAT_BASE_DIR/base_data_tr/shard_99999.parquet] [--out tokenizer/compression_report.md]
"""
import os
import json
import argparse

import pyarrow.parquet as pq

SENTENCES = [
    "Çağrı, İstanbul'daki ağaçların gölgesinde ılık bir öğle yemeği yedi.",
    "Şişli'de oturan öğretmenimiz, öğrencilerine güzel bir öykü okudu.",
    "Ilgaz Dağı'nın eteklerinde yağmur yağınca ırmaklar coşkuyla akar.",
    "Türkiye Büyük Millet Meclisi 23 Nisan 1920'de Ankara'da açıldı.",
    "Görüşebileceğimizi düşünüyordum ama işlerim çok yoğunlaştı.",
    "Afyonkarahisarlılaştıramadıklarımızdanmışsınızcasına konuşuyorsunuz.",
    "Kütüphanedeki kitapların çoğunu okumuş olmalısın, değil mi?",
    "Yapay zekâ modelleri, büyük miktarda metin verisiyle eğitilir.",
]


def load_texts(path, n_docs, cap):
    pf = pq.ParquetFile(path)
    out = []
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            out.append(t[:cap])
            if len(out) >= n_docs:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    base = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nanochat"))
    ap.add_argument("--val-parquet", default=os.path.join(base, "base_data_tr", "shard_99999.parquet"))
    ap.add_argument("--wiki-parquet", default=os.path.join("data", "raw", "wiki_tr_1.parquet"))
    ap.add_argument("--n-docs", type=int, default=3000)
    ap.add_argument("--out", default=os.path.join("tokenizer", "compression_report.md"))
    ap.add_argument("--extra-okanka", nargs="*", default=[], help="extra tokenizer dirs (tokenizer.pkl) to compare, label=path")
    args = ap.parse_args()

    corpora = {"FineWeb-2 tur test (held-out web)": load_texts(args.val_parquet, args.n_docs, 20000)}
    if os.path.exists(args.wiki_parquet):
        corpora["Türkçe Vikipedi (sample)"] = load_texts(args.wiki_parquet, args.n_docs // 3, 20000)
    corpora["ğşıİçöü sentences"] = SENTENCES

    import tiktoken
    from nanochat.tokenizer import RustBPETokenizer
    toks = {}
    toks["Okanka-32K (ours)"] = RustBPETokenizer.from_directory("tokenizer").enc.encode_ordinary
    for spec in args.extra_okanka:
        label, path = spec.split("=", 1)
        toks[label] = RustBPETokenizer.from_directory(path).enc.encode_ordinary
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        try:
            toks[f"tiktoken {name}"] = tiktoken.get_encoding(name).encode_ordinary
        except Exception as e:
            print(f"skip {name}: {e}")
    try:
        from tokenizers import Tokenizer
        for repo in ["dbmdz/bert-base-turkish-cased", "ytu-ce-cosmos/turkish-gpt2"]:
            try:
                tk = Tokenizer.from_pretrained(repo)  # downloads tokenizer.json only, no weights
                toks[f"HF {repo}"] = (lambda t: (lambda s: t.encode(s, add_special_tokens=False).ids))(tk)
            except Exception as e:
                print(f"skip {repo}: {e}")
    except ImportError:
        pass

    rows = []
    for cname, texts in corpora.items():
        nchars = sum(len(t) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        for tname, enc in toks.items():
            ntok = sum(len(enc(t)) for t in texts)
            rows.append({"corpus": cname, "tokenizer": tname, "tokens": ntok,
                         "chars_per_token": nchars / ntok, "bytes_per_token": nbytes / ntok})
    lines = ["# Tokenizer compression on Turkish text", "",
             "Higher chars/token = fewer tokens for the same Turkish text = cheaper training and longer effective context.", ""]
    for cname in corpora:
        lines += [f"## {cname}", "", "| tokenizer | tokens | chars/token | bytes/token |", "|---|---:|---:|---:|"]
        for r in [r for r in rows if r["corpus"] == cname]:
            lines.append(f"| {r['tokenizer']} | {r['tokens']:,} | {r['chars_per_token']:.3f} | {r['bytes_per_token']:.3f} |")
        lines.append("")
    ours = toks["Okanka-32K (ours)"]
    tt = __import__("nanochat.tokenizer", fromlist=["x"]).RustBPETokenizer.from_directory("tokenizer")
    lines += ["## Example segmentation (ours)", ""]
    for s in SENTENCES[:4]:
        ids = ours(s)
        lines.append(f"- `{s}` -> {len(ids)} tokens: " + " | ".join(tt.decode([i]) for i in ids))
    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
