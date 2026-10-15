"""
Generates the Kaggle notebooks in kaggle/ (run: python -m okanka.build_notebooks).
Editing the notebooks by hand is fine too; this script is just the single source of truth.
"""
import os
import json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kaggle")


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


CONFIG = r'''
# ======================= OKANKA MINI CONFIG (edit if needed) =======================
REPO_URL   = "https://github.com/CHANGE_ME/okanka-mini.git"   # GitHub repo with this code (public, or add Kaggle secret GITHUB_TOKEN)
BRANCH     = "main"
DEPTH      = 16          # model depth (d16 ~ 235M "scaling" params). Use 12 for a faster/cheaper run.
SEQ_LEN    = 1024        # context length used for pretraining on T4
DEVICE_BS  = 4 if DEPTH >= 14 else 8   # per-GPU micro batch (power of 2; halve it if CUDA OOM)
DATA_RATIO = 12          # tokens : scaling-params ratio (nanochat compute-optimal ~10-12; Chinchilla 20)
# Hugging Face repos (created automatically, private):
#   <HF_USERNAME>/okanka-mini-data         (dataset: pretraining shards + SFT jsonl)
#   <HF_USERNAME>/okanka-mini-checkpoints  (model: resumable training checkpoints)
#   <HF_USERNAME>/okanka-mini              (model: final public export, notebook 04)
# ===================================================================================
'''

SETUP = r'''
import os, sys, subprocess, shutil
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
def secret(name, default=None):
    try:
        return secrets.get_secret(name)
    except Exception:
        return default

os.environ["HF_TOKEN"] = secret("HF_TOKEN") or ""
assert os.environ["HF_TOKEN"], "Add a Kaggle secret named HF_TOKEN (HF write token): Add-ons -> Secrets"
HF_USERNAME = secret("HF_USERNAME")
if not HF_USERNAME:
    from huggingface_hub import whoami
    HF_USERNAME = whoami(token=os.environ["HF_TOKEN"])["name"]
DATA_REPO = f"{HF_USERNAME}/okanka-mini-data"
CKPT_REPO = f"{HF_USERNAME}/okanka-mini-checkpoints"
FINAL_REPO = f"{HF_USERNAME}/okanka-mini"
print("HF user:", HF_USERNAME)

# --- code ---
gh = secret("GITHUB_TOKEN")
url = REPO_URL.replace("https://", f"https://{gh}@") if gh else REPO_URL
CODE = "/kaggle/working/okanka-mini"
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, CODE], check=True)
else:
    subprocess.run(["git", "-C", CODE, "pull"], check=False)
os.chdir(CODE)
sys.path.insert(0, CODE)

# --- python deps (torch comes with the Kaggle image) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken>=0.11.0", "rustbpe>=0.1.0", "pyarrow>=21.0.0",
                "filelock", "psutil", "huggingface_hub>=0.34.0", "wandb"], check=True)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
if (major, minor) < (2, 6):
    # nanochat needs a recent torch (SDPA enable_gqa, F.rms_norm, compile fixes). cu126 wheels still support T4 (sm75) and P100 (sm60).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.9.1", "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
    print("Upgraded torch -> restart the kernel (Run > Restart) and run all cells again."); raise SystemExit

# --- nanochat runtime env ---
os.environ["NANOCHAT_BASE_DIR"] = "/tmp/nanochat"           # big ephemeral disk; persistent state lives on the HF hub
os.environ["NANOCHAT_DISABLE_FA3"] = "1"                    # FA3 is Hopper/Ampere-only; T4 uses PyTorch SDPA
os.environ["OMP_NUM_THREADS"] = "1"
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    if cap >= (8, 0):
        os.environ["NANOCHAT_DTYPE"] = "bfloat16"
    elif cap >= (7, 0):
        os.environ["NANOCHAT_DTYPE"] = "float16"            # T4/V100: fp16 tensor cores + GradScaler (nanochat built-in)
    else:
        os.environ["NANOCHAT_DTYPE"] = "float32"            # P100: no tensor cores, triton unsupported -> no compile
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
NGPU = max(1, torch.cuda.device_count())
print("NANOCHAT_DTYPE =", os.environ.get("NANOCHAT_DTYPE"), "| GPUs:", NGPU)
os.makedirs(os.environ["NANOCHAT_BASE_DIR"], exist_ok=True)

def run(cmd):
    """Run a shell command, streaming its output into the notebook; raise on failure."""
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
'''

PULL_DATA = r'''
# Download pretraining shards from the private HF dataset repo (fast: HF -> Kaggle is usually >100 MB/s)
from huggingface_hub import snapshot_download
DATA_DIR = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_data_tr")
snapshot_download(DATA_REPO, repo_type="dataset", local_dir=DATA_DIR, allow_patterns=["shard_*.parquet", "manifest.json"],
                  token=os.environ["HF_TOKEN"], max_workers=8)
shards = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
print(len(shards), "shards:", shards[:3], "...", shards[-2:])
'''

# ----------------------------------------------------------------------------------------------
nb01 = nb([
    md(r'''
# Okanka Mini — 01: tokenizer check + data preparation (CPU notebook)

Run this **once**, on a **CPU** session (Settings: Accelerator = None, Internet = On). CPU sessions do **not**
use your weekly GPU quota.

What it does:
1. Verifies the committed Turkish tokenizer (`tokenizer/`) loads and reports its compression.
2. Downloads FineWeb-2 `tur_Latn` + Turkish Wikipedia over HTTPS, writes nanochat parquet shards
   (`text` column, row groups of 1024 docs), and uploads them to the private HF dataset repo
   `<HF_USERNAME>/okanka-mini-data` (deleting local copies to stay under the disk limit).
3. Builds the Turkish SFT jsonl (Aya + OASST2 + Wikipedia template QA) + identity data and uploads it to the same repo.

Needed Kaggle secrets: `HF_TOKEN` (HF **write** token), optional `HF_USERNAME`, optional `GITHUB_TOKEN` (if the code repo is private).

Time: ~1.5-3 h (mostly parquet processing + upload). If the session dies, just re-run: finished shards are
already on the hub; set `START_SHARD` to the number of shards already uploaded and `SKIP_DOCS` accordingly — or simply delete the dataset repo and start over.
'''),
    code(CONFIG),
    code(SETUP),
    code(r'''
# 1) tokenizer sanity check (the tokenizer was trained from scratch and committed to the repo)
from nanochat.tokenizer import get_tokenizer
tok = get_tokenizer()
s = "Çağrı, İstanbul'daki ağaçların gölgesinde ılık bir öğle yemeği yedi."
ids = tok.encode(s)
print(tok.get_vocab_size(), len(s), "chars ->", len(ids), "tokens:", [tok.decode([i]) for i in ids])
assert tok.decode(ids) == s
'''),
    code(r'''
# 2) How much text do we need?  tokens = DATA_RATIO * scaling_params(DEPTH); chars ≈ tokens * chars_per_token / (1 - crop_loss)
import math, json
SCALING_PARAMS = {12: 110_100_912, 14: 164_233_804, 16: 234_881_792, 18: 324_404_172, 20: 435_160_240}  # transformer matrices + lm_head, vocab 32768
CHARS_PER_TOKEN = 4.55     # measured on our shards incl. full docs (tokenizer/compression_report.md: 4.85 on FineWeb-2 test)
CROP_LOSS = 0.54           # measured: best-fit packing at T=1024 keeps only ~46% of tokens (long docs get cropped)
tokens_needed = DATA_RATIO * SCALING_PARAMS[DEPTH]
TARGET_CHARS = int(tokens_needed * CHARS_PER_TOKEN / (1 - CROP_LOSS) * 1.1)
FW2_FILES = list(range(max(1, math.ceil(TARGET_CHARS / 14e9))))   # each FineWeb-2 tur_Latn train file holds ~15B chars (30 files total)
print(f"tokens needed ≈ {tokens_needed/1e9:.2f}B -> writing ≈ {TARGET_CHARS/1e9:.1f}B chars from FineWeb-2 files {FW2_FILES}")
START_SHARD = 0
'''),
    code(r'''
# 3) build + upload the shards (raw downloads go to /tmp, shards are deleted locally after upload)
run(f"python -m okanka.prepare_pretrain_data --target-chars {TARGET_CHARS} --fw2-files {' '.join(map(str, FW2_FILES))} "
    f"--download-dir /tmp/raw --delete-raw --upload-repo {DATA_REPO} --delete-after-upload --start-shard-idx {START_SHARD}")
'''),
    code(r'''
# 4) SFT + identity data -> data repo (sft/ folder)
run("python -m okanka.make_identity --n 260 --out data/identity_tr.jsonl")
run("python -m okanka.prepare_sft_data --raw-dir /tmp/raw/sft --wiki-dir /tmp/raw --out-dir data/sft --wiki-qa 12000")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for f in ["data/sft/sft_tr_train.jsonl", "data/sft/sft_tr_val.jsonl", "data/identity_tr.jsonl"]:
    api.upload_file(path_or_fileobj=f, path_in_repo="sft/" + os.path.basename(f), repo_id=DATA_REPO, repo_type="dataset")
print([f for f in api.list_repo_files(DATA_REPO, repo_type="dataset")][:10], "...")
'''),
])

# ----------------------------------------------------------------------------------------------
nb02 = nb([
    md(r'''
# Okanka Mini — 02: pretraining (resumable, run repeatedly)

Settings: **Accelerator = GPU T4 x2**, Internet = On, Persistence: none needed. Secrets: `HF_TOKEN`, `HF_USERNAME`.

Every session (≤12 h):
1. pulls the latest checkpoint from `<HF_USERNAME>/okanka-mini-checkpoints` (if any),
2. trains with `torchrun` on both T4s in fp16 (+GradScaler), SDPA attention, full-context windows,
3. saves every `SAVE_EVERY` steps (model + optimizer shards + dataloader position + GradScaler state) and
   pushes each save to the hub in a background thread,
4. stops itself after `TIME_BUDGET_MIN` minutes with a final save+push, so the session never hits the 12 h kill.

Just click **Save Version -> Save & Run All (Commit)** again each time a run ends, until the log says the final step is reached
(the last cell prints progress). Always keep the same DEPTH / SEQ_LEN / DEVICE_BS / GPU type (T4 x2) across sessions.
'''),
    code(CONFIG),
    code(SETUP),
    code(PULL_DATA),
    code(r'''
# Pull the newest checkpoint (if any) so base_train --auto-resume continues from it
from okanka import hf_sync
SUBDIR = f"base_checkpoints/d{DEPTH}"
step = hf_sync.pull(CKPT_REPO, SUBDIR, os.environ["NANOCHAT_BASE_DIR"])
print("resuming from step:", step)
'''),
    code(r'''
TIME_BUDGET_MIN = 645   # Kaggle hard limit is 12h = 720 min; keep ~1h for setup, final save and upload
SAVE_EVERY = 100 if DEPTH >= 14 else 200   # steps; ≈ every 45-60 min on 2xT4 (d16: ~25-35 s/step at 524K tokens/step)
compile_flag = "--no-compile" if os.environ.get("TORCHDYNAMO_DISABLE") == "1" else ""
run(f"torchrun --standalone --nproc_per_node={NGPU} -m scripts.base_train -- "
    f"--depth={DEPTH} --max-seq-len={SEQ_LEN} --window-pattern=L --device-batch-size={DEVICE_BS} "
    f"--target-param-data-ratio={DATA_RATIO} --save-every={SAVE_EVERY} --eval-every={SAVE_EVERY} --eval-tokens={20*SEQ_LEN*DEVICE_BS*NGPU} "
    f"--core-metric-every=-1 --sample-every={SAVE_EVERY} --auto-resume=1 --time-budget-minutes={TIME_BUDGET_MIN} "
    f"--hf-repo={CKPT_REPO} --keep-local=1 {compile_flag}")
'''),
    code(r'''
# Progress report
import json, glob
metas = sorted(glob.glob(os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_checkpoints", f"d{DEPTH}", "meta_*.json")))
if metas:
    m = json.load(open(metas[-1]))
    print("last saved step:", m["step"], "| val_bpb:", m.get("val_bpb"), "| train time so far (min):", round(m["loop_state"]["total_training_time"]/60, 1))
print("Remote checkpoint steps:", hf_sync.remote_steps(CKPT_REPO, f"base_checkpoints/d{DEPTH}"))
'''),
])

# ----------------------------------------------------------------------------------------------
nb03 = nb([
    md(r'''
# Okanka Mini — 03: SFT (chat format + Turkish instructions + identity)

Run after notebook 02 has reached its final step. Settings: GPU T4 x2, Internet On. ~0.5-1.5 h.

nanochat's current version has no separate mid-training stage (it was merged into SFT); this notebook runs
`scripts.chat_sft` with the Turkish mixture (`--mixture tr`): Aya-Turkish + OASST2-Turkish + Wikipedia template QA
(x2 epochs) + identity conversations (x4). The result is pushed to `<HF_USERNAME>/okanka-mini-checkpoints/chatsft_checkpoints/d<DEPTH>`.
'''),
    code(CONFIG),
    code(SETUP),
    code(r'''
from okanka import hf_sync
from huggingface_hub import hf_hub_download
step = hf_sync.pull(CKPT_REPO, f"base_checkpoints/d{DEPTH}", os.environ["NANOCHAT_BASE_DIR"])
assert step is not None, "no pretraining checkpoint found - run notebook 02 first"
print("base step:", step)
os.makedirs("data/sft", exist_ok=True)
for f in ["sft/sft_tr_train.jsonl", "sft/sft_tr_val.jsonl", "sft/identity_tr.jsonl"]:
    p = hf_hub_download(DATA_REPO, f, repo_type="dataset", token=os.environ["HF_TOKEN"])
    shutil.copy(p, "data/" + (f if "identity" not in f else "identity_tr.jsonl"))
!wc -l data/sft/*.jsonl data/identity_tr.jsonl
'''),
    code(r'''
compile_flag = "--no-compile" if os.environ.get("TORCHDYNAMO_DISABLE") == "1" else ""
run(f"torchrun --standalone --nproc_per_node={NGPU} -m scripts.chat_sft -- --mixture=tr "
    f"--device-batch-size={DEVICE_BS} --total-batch-size={DEVICE_BS*SEQ_LEN*NGPU*4} "
    f"--eval-every=100 --eval-tokens={10*SEQ_LEN*DEVICE_BS*NGPU} --sft-epochs=2 --identity-epochs=4 {compile_flag}")
'''),
    code(r'''
from nanochat.checkpoint_manager import find_last_step
sft_dir = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "chatsft_checkpoints", f"d{DEPTH}")
s = find_last_step(sft_dir)
hf_sync.push(CKPT_REPO, f"chatsft_checkpoints/d{DEPTH}", s, os.environ["NANOCHAT_BASE_DIR"])
print("SFT step", s, "pushed")
'''),
    code(r'''
# quick smoke test
run('python -m scripts.chat_cli -i sft -p "Merhaba! Sen kimsin?" -t 0.6')
'''),
])

# ----------------------------------------------------------------------------------------------
nb04 = nb([
    md(r'''
# Okanka Mini — 04: evaluation + export to the Hub ("okanka-mini")

Settings: GPU T4 (x1 is enough) or even CPU, Internet On.
1. pulls the SFT checkpoint, 2. runs a set of Turkish chat prompts, 3. computes validation bpb on the held-out
FineWeb-2 test shard, 4. exports weights + tokenizer + LICENSE/NOTICE + model card to `<HF_USERNAME>/okanka-mini`.

`MAKE_PUBLIC = True` makes the final model repo public (needed if the Space should load it without a token).
'''),
    code(CONFIG),
    code(SETUP + "\nMAKE_PUBLIC = True\n"),
    code(r'''
from okanka import hf_sync
s = hf_sync.pull(CKPT_REPO, f"chatsft_checkpoints/d{DEPTH}", os.environ["NANOCHAT_BASE_DIR"])
assert s is not None, "run notebook 03 first"
import torch
from nanochat.common import autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=f"d{DEPTH}")
engine = Engine(model, tokenizer)
bos = tokenizer.get_bos_token_id()
U0, U1, A0, A1 = (tokenizer.encode_special(t) for t in ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>"])
def ask(q, max_tokens=200, temperature=0.6):
    ids = [bos, U0] + tokenizer.encode(q) + [U1, A0]
    out, _ = engine.generate_batch(ids, num_samples=1, max_tokens=max_tokens, temperature=temperature, top_k=50)
    return tokenizer.decode(out[0][len(ids):])
for q in ["Merhaba! Sen kimsin?", "Seni kim yaptı?", "ChatGPT misin?", "Türkiye'nin başkenti neresidir?",
          "İstanbul hakkında kısa bilgi verir misin?", "Bana kısa bir şiir yaz.", "Fotosentez nedir?",
          "İnternete erişimin var mı?", "2 + 2 kaç eder?", "Sağlıklı beslenme için üç öneri ver."]:
    print("USER:", q); print("OKANKA:", ask(q)); print("-" * 80)
'''),
    code(r'''
# validation bits-per-byte on the held-out FineWeb-2 test shard (lower is better; tokenizer-independent)
from huggingface_hub import hf_hub_download
DATA_DIR = os.path.join(os.environ["NANOCHAT_BASE_DIR"], "base_data_tr"); os.makedirs(DATA_DIR, exist_ok=True)
for f in ["shard_00000.parquet", "shard_99999.parquet"]:
    shutil.copy(hf_hub_download(DATA_REPO, f, repo_type="dataset", token=os.environ["HF_TOKEN"]), os.path.join(DATA_DIR, f))
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes
B, T = 4, model.config.sequence_len
loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, B, T, "val", device=device.type)
print("val bpb:", evaluate_bpb(model, loader, 50, get_token_bytes(device=device)))
'''),
    code(r'''
run(f"python -m okanka.export_model --source sft --model-tag d{DEPTH} --out /tmp/export/okanka-mini --push {FINAL_REPO}" + ("" if MAKE_PUBLIC else " --private"))
print(f"https://huggingface.co/{FINAL_REPO}")
'''),
])


def kmeta(slug, title, nbfile, gpu):
    return {"id": f"KAGGLE_USERNAME/{slug}", "title": title, "code_file": nbfile, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main():
    items = [
        ("01_tokenizer_and_data", "okanka-mini-01-data", "Okanka Mini 01 Data", nb01, False),
        ("02_pretrain", "okanka-mini-02-pretrain", "Okanka Mini 02 Pretrain", nb02, True),
        ("03_midtrain_sft", "okanka-mini-03-sft", "Okanka Mini 03 SFT", nb03, True),
        ("04_eval_and_export", "okanka-mini-04-export", "Okanka Mini 04 Export", nb04, True),
    ]
    for name, slug, title, notebook, gpu in items:
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.ipynb"), "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(kmeta(slug, title, f"{name}.ipynb", gpu), f, indent=2)
        print("wrote", d)


if __name__ == "__main__":
    main()
