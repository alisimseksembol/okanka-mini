"""
Export the final Okanka Mini chat model (SFT checkpoint) into a self-contained folder and
(optionally) push it to the Hugging Face Hub as the public model repo "okanka-mini".

Folder layout (this is what space/app.py loads):
    README.md                      model card (TR + EN)
    LICENSE                        MIT (code: nanochat + Okanka patches); see README for data notes
    NOTICE                         attribution
    tokenizer/tokenizer.pkl        tiktoken Encoding pickle (nanochat format)
    tokenizer/token_bytes.pt
    chatsft_checkpoints/<tag>/model_XXXXXX.pt   weights only (no optimizer), fp32
    chatsft_checkpoints/<tag>/meta_XXXXXX.json

Run:
    python -m okanka.export_model --out export/okanka-mini [--push <user>/okanka-mini] [--source sft]
"""
import os
import json
import shutil
import argparse

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sft", choices=["sft", "base"])
    ap.add_argument("--model-tag", default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--out", default=os.path.join("export", "okanka-mini"))
    ap.add_argument("--push", default="", help="HF repo id to upload to, e.g. user/okanka-mini (needs HF_TOKEN)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from nanochat.common import get_base_dir
    from nanochat.checkpoint_manager import find_largest_model, find_last_step
    from nanochat.tokenizer import _resolve_tokenizer_dir

    src_root = os.path.join(get_base_dir(), {"sft": "chatsft_checkpoints", "base": "base_checkpoints"}[args.source])
    tag = args.model_tag or find_largest_model(src_root)
    src_dir = os.path.join(src_root, tag)
    step = args.step if args.step is not None else find_last_step(src_dir)

    dst_dir = os.path.join(args.out, "chatsft_checkpoints", tag)  # always exported as "sft" so the Space loads it
    os.makedirs(dst_dir, exist_ok=True)
    sd = torch.load(os.path.join(src_dir, f"model_{step:06d}.pt"), map_location="cpu")
    sd = {k.removeprefix("_orig_mod."): (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save(sd, os.path.join(dst_dir, f"model_{step:06d}.pt"))
    with open(os.path.join(src_dir, f"meta_{step:06d}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    keep = {k: meta[k] for k in ("step", "val_bpb", "model_config") if k in meta}
    with open(os.path.join(dst_dir, f"meta_{step:06d}.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, indent=2)
    n_params = sum(v.numel() for v in sd.values())

    tok_dst = os.path.join(args.out, "tokenizer")
    os.makedirs(tok_dst, exist_ok=True)
    for fn in ("tokenizer.pkl", "token_bytes.pt"):
        shutil.copy2(os.path.join(_resolve_tokenizer_dir(), fn), os.path.join(tok_dst, fn))
    for fn in ("LICENSE", "NOTICE", "DATA_LICENSES.md"):
        shutil.copy2(os.path.join(REPO_ROOT, fn), os.path.join(args.out, fn))

    cfg = keep["model_config"]
    card_tpl = open(os.path.join(REPO_ROOT, "okanka", "model_card.md"), encoding="utf-8").read()
    card = (card_tpl.replace("{{N_LAYER}}", str(cfg["n_layer"])).replace("{{N_EMBD}}", str(cfg["n_embd"]))
            .replace("{{N_HEAD}}", str(cfg["n_head"])).replace("{{VOCAB}}", str(cfg["vocab_size"]))
            .replace("{{SEQ}}", str(cfg["sequence_len"])).replace("{{PARAMS}}", f"{n_params/1e6:.0f}M")
            .replace("{{STEP}}", str(step)).replace("{{TAG}}", tag))
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(f"exported {tag} step {step} ({n_params/1e6:.1f}M params) -> {args.out}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.push, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="model", commit_message=f"Okanka Mini {tag} step {step}")
        print(f"pushed to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
