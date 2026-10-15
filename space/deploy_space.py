"""
Upload the Okanka Mini Space (Gradio SDK, ZeroGPU) to the Hugging Face Hub.

    HF_TOKEN=hf_xxx python space/deploy_space.py --space <hf-user>/okanka-mini-chat [--model-repo <hf-user>/okanka-mini]

Uploads space/{app.py,requirements.txt,README.md,API.md} and the repo's nanochat/ package.
Hardware (ZeroGPU) and secrets are set in the Space settings UI (see YAYINLAMA.md); this script tries to
set the MODEL_REPO variable for you.
"""
import os
import argparse
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--space", required=True)
ap.add_argument("--model-repo", default="")
args = ap.parse_args()

api = HfApi(token=os.environ.get("HF_TOKEN"))
api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
for fn in ("app.py", "requirements.txt", "README.md", "API.md"):
    api.upload_file(path_or_fileobj=os.path.join(HERE, fn), path_in_repo=fn, repo_id=args.space, repo_type="space")
api.upload_folder(folder_path=os.path.join(ROOT, "nanochat"), path_in_repo="nanochat", repo_id=args.space,
                  repo_type="space", allow_patterns=["*.py"], commit_message="nanochat package (Okanka fork)")
if args.model_repo:
    try:
        api.add_space_variable(args.space, "MODEL_REPO", args.model_repo)
    except Exception as e:
        print(f"could not set MODEL_REPO automatically ({e}); set it in the Space settings")
print(f"https://huggingface.co/spaces/{args.space}")
