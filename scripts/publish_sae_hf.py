"""
Publishes the 5000-step contrastive SAE to HuggingFace as a public model repo.

Usage:
    python scripts/publish_sae_hf.py [--repo Solshine/biorefusalaudit-sae-gemma2-2b-l12-5000steps]

Requires HF_TOKEN with write access in environment (or .env file).
Run only after explicit user approval — per CLAUDE.md HF push policy.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "Solshine/biorefusalaudit-sae-gemma2-2b-l12-5000steps"
SAE_DIR = Path("runs/sae-training-gemma2-5000steps")
FILES = ["sae_weights.pt", "training_log.jsonl", "analysis.txt", "README.md"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"')

    # Fall back to cached huggingface-cli credentials (from `huggingface-cli login`)
    api = HfApi(token=token if token else None)

    print(f"Creating repo: {args.repo} (private={args.private})")
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"  https://huggingface.co/{args.repo}")

    for fname in FILES:
        fpath = SAE_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fpath}")
            continue
        print(f"  Uploading {fpath} ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"    done: {fname}")

    print(f"\nDone. Model card at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
