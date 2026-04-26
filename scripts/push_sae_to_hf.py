"""Push trained SAE checkpoints to HuggingFace.

REQUIRES EXPLICIT USER APPROVAL before running. Per CLAUDE.md:
  "HF pushes require explicit per-repo approval + external review."

Usage (after approval):
  source .venv/Scripts/activate
  python scripts/push_sae_to_hf.py --repo wmdp
  python scripts/push_sae_to_hf.py --repo pairwise
  python scripts/push_sae_to_hf.py --repo gemma4   # after Gemma 4 E2B training completes

Repos created (private by default, can be made public after review):
  SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-wmdp
  SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-pairwise
  Solshine/gemma4-e2b-bio-sae-v1
"""
import argparse, os, sys
from pathlib import Path

def push(repo_id: str, local_dir: Path, files: list[str], readme: Path, private: bool = True):
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Creating repo {repo_id} (private={private})...")
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)

    print(f"Uploading README...")
    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    for f in files:
        src = local_dir / f
        if src.exists():
            print(f"Uploading {f} ({src.stat().st_size / 1e6:.1f} MB)...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f,
                repo_id=repo_id,
            )
        else:
            print(f"WARNING: {src} not found — skipping")

    print(f"Done: https://huggingface.co/{repo_id}")


CONFIGS = {
    "wmdp": {
        "repo_id": "SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-wmdp",
        "local_dir": Path("runs/sae-training-wmdp-222-5000steps"),
        "files": ["sae_weights.pt", "training_log.jsonl"],
        "readme": Path("hf_assets/gemma2-2b-bio-sae-wmdp/README.md"),
    },
    "pairwise": {
        "repo_id": "SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-pairwise",
        "local_dir": Path("runs/sae-training-gemma2-10ksteps-pairwise"),
        "files": [
            "sae_weights.pt",
            "checkpoint_01000.pt",
            "checkpoint_02000.pt",
            "checkpoint_03000.pt",
            "checkpoint_04000.pt",
            "training_log.jsonl",
        ],
        "readme": Path("hf_assets/gemma2-2b-bio-sae-pairwise/README.md"),
    },
    "gemma4": {
        "repo_id": "Solshine/gemma4-e2b-bio-sae-v1",
        "local_dir": Path("runs/sae-training-gemma4-e2b-5000steps"),
        "files": [
            "sae_weights.pt",
            "checkpoint_01000.pt",
            "checkpoint_02000.pt",
            "checkpoint_03000.pt",
            "checkpoint_04000.pt",
            "training_log.jsonl",
        ],
        "readme": Path("hf_assets/gemma4-e2b-bio-sae-v1/README.md"),
    },
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, choices=["wmdp", "pairwise", "gemma4"])
    p.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    args = p.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not set. Add to .env and run: source .env")
        sys.exit(1)

    cfg = CONFIGS[args.repo]
    push(
        repo_id=cfg["repo_id"],
        local_dir=cfg["local_dir"],
        files=cfg["files"],
        readme=cfg["readme"],
        private=not args.public,
    )
