"""
Okanka Mini - Hugging Face Space (Gradio SDK, ZeroGPU hardware on a free account).

API (see API.md):  Gradio endpoint "chat"  ->  POST {SPACE}/gradio_api/call/chat
  inputs : messages_json (str, JSON array of {"role","content"}), max_tokens (int), key (str)
  output : reply text (str); the function is a generator, so the SSE stream yields the growing text.

Env vars:
  MODEL_REPO        HF model repo with the exported model (default: "okanka-mini"; usually "<user>/okanka-mini")
  MODEL_DIR         optional local directory with the export (skips the hub download; for local testing)
  OKANKA_MINI_KEY   optional shared secret; when set, the 3rd input `key` must match it
  HF_TOKEN          only needed if MODEL_REPO is private
  OKANKA_TEMPERATURE / OKANKA_TOP_K   sampling defaults (0.7 / 50)
"""
import os
import sys
import json
import hmac
import threading

# --- runtime selection: ZeroGPU (spaces + CUDA) or plain CPU fallback -----------------------------
try:
    import spaces  # available on HF Spaces; must be imported before torch touches CUDA
    HAS_SPACES = True
except Exception:  # local run / CPU fallback
    spaces = None
    HAS_SPACES = False

# The model is built in fp32 (small model; avoids dtype juggling between the CPU build and the GPU run).
os.environ.setdefault("NANOCHAT_DTYPE", "float32")
os.environ.setdefault("NANOCHAT_DISABLE_FA3", "1")  # never probe CUDA / download kernels at import time

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.dirname(HERE)):  # Space: ./nanochat is uploaded next to app.py; local: repo root
    if os.path.isdir(os.path.join(cand, "nanochat")) and cand not in sys.path:
        sys.path.insert(0, cand)

import torch  # noqa: E402
import gradio as gr  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "okanka-mini")
MODEL_DIR = os.environ.get("MODEL_DIR", "")
API_KEY = os.environ.get("OKANKA_MINI_KEY", "")
TEMPERATURE = float(os.environ.get("OKANKA_TEMPERATURE", "0.7"))
TOP_K = int(os.environ.get("OKANKA_TOP_K", "50"))
MAX_TOKENS_CAP = int(os.environ.get("OKANKA_MAX_TOKENS_CAP", "1024"))
MAX_PROMPT_TOKENS = 1536


def _model_dir():
    if MODEL_DIR:
        return MODEL_DIR
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_REPO, repo_type="model", token=os.environ.get("HF_TOKEN") or None)


# Export layout (written by okanka/export_model.py):
#   <dir>/tokenizer/tokenizer.pkl, token_bytes.pt
#   <dir>/chatsft_checkpoints/<tag>/model_XXXXXX.pt, meta_XXXXXX.json
LOCAL_DIR = _model_dir()
os.environ["NANOCHAT_BASE_DIR"] = LOCAL_DIR
os.environ["NANOCHAT_TOKENIZER_DIR"] = os.path.join(LOCAL_DIR, "tokenizer")

from nanochat.checkpoint_manager import load_model  # noqa: E402
from nanochat.engine import Engine  # noqa: E402

model, tokenizer, meta = load_model("sft", torch.device("cpu"), phase="eval")
GPU_MODE = HAS_SPACES and torch.cuda.is_available()
if GPU_MODE:
    model.to("cuda")  # ZeroGPU: allowed at startup, the GPU is attached inside @spaces.GPU calls
engine = Engine(model, tokenizer)
_cpu_lock = threading.Lock()  # CPU fallback: one generation at a time

BOS = tokenizer.get_bos_token_id()
U0, U1 = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
A0, A1 = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
SEQ_LEN = model.config.sequence_len


def build_prompt(messages):
    """messages: list of {"role": "system"|"user"|"assistant", "content": str} -> token ids ending in <|assistant_start|>."""
    msgs = [m for m in messages if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")]
    system = "\n\n".join(str(m["content"]) for m in msgs if m["role"] == "system")
    turns = [m for m in msgs if m["role"] != "system"]
    # merge consecutive same-role messages, drop leading assistant turns
    merged = []
    for m in turns:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + str(m["content"])
        else:
            merged.append({"role": m["role"], "content": str(m["content"])})
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    if not merged or merged[-1]["role"] != "user":
        raise ValueError("the last message must come from the user")
    if system:
        merged[0]["content"] = system + "\n\n" + merged[0]["content"]
    # encode turns, then keep only the most recent ones that fit
    blocks = []
    for m in merged:
        if m["role"] == "user":
            blocks.append([U0] + tokenizer.encode(m["content"]) + [U1])
        else:
            blocks.append([A0] + tokenizer.encode(m["content"]) + [A1])
    budget = min(MAX_PROMPT_TOKENS, SEQ_LEN - 64)
    kept, total = [], 1
    for b in reversed(blocks):
        if total + len(b) > budget:
            if not kept:  # the last user message alone is too long: keep its tail
                b = [U0] + b[-(budget - 3):]
                kept.insert(0, b)
            break
        kept.insert(0, b)
        total += len(b)
    ids = [BOS]
    for b in kept:
        ids.extend(b)
    ids.append(A0)
    return ids


def _stream(ids, max_tokens):
    out_tokens = []
    text = ""
    for token_column, _ in engine.generate(ids, num_samples=1, max_tokens=max_tokens, temperature=TEMPERATURE, top_k=TOP_K):
        tok = token_column[0]
        if tok == A1 or tok == BOS:
            break
        out_tokens.append(tok)
        new_text = tokenizer.decode(out_tokens)
        if new_text.endswith("�"):  # incomplete UTF-8 sequence (e.g. half of 'ş'), wait for the next token
            continue
        text = new_text
        yield text
    final = tokenizer.decode(out_tokens).replace("�", "")
    if final != text:
        yield final


if GPU_MODE:
    @spaces.GPU(duration=60)
    def _generate(ids, max_tokens):
        yield from _stream(ids, max_tokens)
else:
    def _generate(ids, max_tokens):
        with _cpu_lock:
            yield from _stream(ids, max_tokens)


def chat(messages_json, max_tokens, key):
    """Gradio API 'chat': yields the growing reply text."""
    if API_KEY and not hmac.compare_digest(str(key or ""), API_KEY):
        raise gr.Error("unauthorized: invalid key")
    try:
        messages = json.loads(messages_json) if isinstance(messages_json, str) else messages_json
        if isinstance(messages, dict):
            messages = messages.get("messages", [])
        assert isinstance(messages, list) and messages
        ids = build_prompt(messages)
    except Exception as e:
        raise gr.Error(f"bad request: messages_json must be a JSON array of {{role, content}} ending with a user message ({e})")
    try:
        max_tokens = int(max_tokens or 256)
    except Exception:
        max_tokens = 256
    max_tokens = max(1, min(max_tokens, MAX_TOKENS_CAP, SEQ_LEN - len(ids)))
    yield ""
    yield from _generate(ids, max_tokens)


demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="messages_json", value='[{"role": "user", "content": "Merhaba! Sen kimsin?"}]', lines=4),
        gr.Number(label="max_tokens", value=256, precision=0),
        gr.Textbox(label="key", value="", type="password"),
    ],
    outputs=gr.Textbox(label="reply"),
    title="Okanka Mini",
    description="Seozit.com'un sıfırdan eğittiği deneysel, küçük Türkçe dil modeli. Hata yapabilir; internete erişimi yoktur.",
    api_name="chat",
    flagging_mode="never",
)


def health():
    return {"status": "ok", "model_repo": MODEL_REPO, "device": "cuda(zerogpu)" if GPU_MODE else "cpu",
            "n_layer": model.config.n_layer, "n_embd": model.config.n_embd, "vocab_size": model.config.vocab_size,
            "sequence_len": SEQ_LEN, "key_required": bool(API_KEY)}


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2 if GPU_MODE else 1, max_size=32)
    # GET /health is registered on the FastAPI app Gradio creates (passed in before Gradio adds its own routes)
    from fastapi.routing import APIRoute
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                app_kwargs={"routes": [APIRoute("/health", health, methods=["GET"])]})
