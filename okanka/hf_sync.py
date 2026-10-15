"""
Checkpoint sync with a (private) Hugging Face model repo, so that Kaggle sessions (<12h each)
can be chained: pull latest checkpoint at session start, push after every save.

Remote layout (repo_type="model", private):
    <subdir>/model_XXXXXX.pt
    <subdir>/meta_XXXXXX.json
    <subdir>/optim_XXXXXX_rank{r}.pt
where <subdir> mirrors the local nanochat layout, e.g. "base_checkpoints/d16".

Only the latest step is kept remotely; old files are deleted and (optionally) the git history is
squashed so that LFS storage does not grow with every save.

Token: read from the HF_TOKEN environment variable (on Kaggle: Add-ons -> Secrets -> HF_TOKEN).
CLI:
    python -m okanka.hf_sync pull --repo user/okanka-mini-checkpoints --subdir base_checkpoints/d16
    python -m okanka.hf_sync push --repo user/okanka-mini-checkpoints --subdir base_checkpoints/d16 --step 1200
    python -m okanka.hf_sync latest --repo ... --subdir ...
"""
import os
import re
import argparse
import threading

STEP_RE = re.compile(r"model_(\d{6})\.pt$")


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def _base_dir():
    from nanochat.common import get_base_dir
    return get_base_dir()


def remote_steps(repo, subdir):
    api = _api()
    try:
        files = api.list_repo_files(repo, repo_type="model")
    except Exception as e:  # repo does not exist yet
        print(f"[hf_sync] cannot list {repo}: {e}")
        return []
    steps = set()
    for f in files:
        if f.startswith(subdir.rstrip("/") + "/"):
            m = STEP_RE.search(f)
            if m:
                steps.add(int(m.group(1)))
    return sorted(steps)


def pull(repo, subdir, local_root=None, step=None):
    """Download the latest (or given) checkpoint step into local_root/subdir. Returns the step or None."""
    from huggingface_hub import hf_hub_download
    local_root = local_root or _base_dir()
    steps = remote_steps(repo, subdir)
    if not steps:
        print(f"[hf_sync] no remote checkpoint under {repo}/{subdir}")
        return None
    step = steps[-1] if step is None else step
    api = _api()
    files = [f for f in api.list_repo_files(repo, repo_type="model")
             if f.startswith(subdir.rstrip("/") + "/") and f"_{step:06d}" in f]
    for f in files:
        print(f"[hf_sync] downloading {f}")
        hf_hub_download(repo, f, repo_type="model", local_dir=local_root, token=os.environ.get("HF_TOKEN"))
    return step


def push(repo, subdir, step, local_root=None, keep_last=1, squash=True, private=True):
    """Upload checkpoint files of `step` and delete older remote steps."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    local_root = local_root or _base_dir()
    local_dir = os.path.join(local_root, subdir)
    tag = f"_{step:06d}"
    ops = [CommitOperationAdd(path_in_repo=f"{subdir}/{f}", path_or_fileobj=os.path.join(local_dir, f))
           for f in sorted(os.listdir(local_dir)) if tag in f]
    old = [s for s in remote_steps(repo, subdir) if s != step]
    old = old[:-keep_last + 1] if keep_last > 1 else old
    if old:
        files = api.list_repo_files(repo, repo_type="model")
        for s in old:
            ops += [CommitOperationDelete(path_in_repo=f) for f in files
                    if f.startswith(subdir + "/") and f"_{s:06d}" in f]
    api.create_commit(repo, operations=ops, repo_type="model", commit_message=f"checkpoint {subdir} step {step}")
    print(f"[hf_sync] pushed {subdir} step {step} to {repo} (removed old steps: {old})")
    if squash:
        try:
            api.super_squash_history(repo_id=repo, repo_type="model")
        except Exception as e:  # not fatal
            print(f"[hf_sync] squash failed (ignored): {e}")


def push_async(*args, **kwargs):
    """Push in a background thread so training continues while uploading. Returns the thread."""
    def _run():
        try:
            push(*args, **kwargs)
        except Exception as e:
            print(f"[hf_sync] PUSH FAILED: {e}")
    t = threading.Thread(target=_run, daemon=False)
    t.start()
    return t


def prune_local(checkpoint_dir, keep_last=1):
    """Delete all but the newest `keep_last` checkpoint steps from a local dir (disk is small on Kaggle)."""
    if not os.path.isdir(checkpoint_dir):
        return
    steps = sorted({int(m.group(1)) for f in os.listdir(checkpoint_dir) for m in [STEP_RE.search(f)] if m})
    for s in steps[:-keep_last]:
        for f in os.listdir(checkpoint_dir):
            if f"_{s:06d}" in f:
                os.remove(os.path.join(checkpoint_dir, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "push", "latest"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--local-root", default=None)
    args = ap.parse_args()
    if args.cmd == "pull":
        print(pull(args.repo, args.subdir, args.local_root, args.step))
    elif args.cmd == "latest":
        s = remote_steps(args.repo, args.subdir)
        print(s[-1] if s else -1)
    else:
        push(args.repo, args.subdir, args.step, args.local_root)


if __name__ == "__main__":
    main()
