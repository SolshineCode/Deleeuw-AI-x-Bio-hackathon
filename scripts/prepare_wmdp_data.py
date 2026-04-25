"""Download and prepare WMDP bio corpus for local SAE training.

Downloads cais/wmdp-corpora bio subsets from HuggingFace and saves them as
JSONL with BioRefusalAudit tier labels, ready for use with train_sae_local.py
or any pipeline component that takes --eval-set.

Tier mapping:
  bio_forget_corpus  -> hazard_adjacent_category  (hazard-adjacent biology)
  bio_retain_corpus  -> benign_bio                (safe biology)

Output schema per row (same as eval_set_public_v1.jsonl):
  {
    "id": "wmdp_bio_<split>_<idx>",
    "tier": "hazard_adjacent_category" | "benign_bio",
    "framing": "direct",
    "prompt": "<text>",
    "source": "cais/wmdp-corpora"
  }

Usage:
    python scripts/prepare_wmdp_data.py --out data/wmdp --max-per-tier 5000
    python scripts/prepare_wmdp_data.py --out data/wmdp --max-per-tier 500 --seed 123
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "wmdp")
    ap.add_argument("--max-per-tier", type=int, default=5000,
                    help="Max samples per tier. Use small value for quick test.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="Keep original corpus order instead of shuffling.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits = [
        ("bio_forget_corpus", "hazard_adjacent_category", "forget"),
        ("bio_retain_corpus", "benign_bio", "retain"),
    ]

    combined: list[dict] = []
    stats: dict[str, int] = {}

    for hf_config, tier, split_tag in splits:
        print(f"[wmdp] Loading cais/wmdp-corpora/{hf_config}...")
        try:
            ds = load_dataset("cais/wmdp-corpora", hf_config, split="train")
        except Exception as e:
            print(f"[wmdp] ERROR loading {hf_config}: {e}", file=sys.stderr)
            return 1

        rows = [{"text": r["text"]} for r in ds if r.get("text", "").strip()]
        print(f"[wmdp]   Raw rows: {len(rows)}")

        if not args.no_shuffle:
            rng.shuffle(rows)
        rows = rows[: args.max_per_tier]

        jsonl_rows = [
            {
                "id": f"wmdp_bio_{split_tag}_{i:05d}",
                "tier": tier,
                "framing": "direct",
                "prompt": r["text"],
                "source": f"cais/wmdp-corpora/{hf_config}",
            }
            for i, r in enumerate(rows)
        ]

        # Per-split file for flexibility
        split_path = args.out / f"wmdp_bio_{split_tag}_{len(jsonl_rows)}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[wmdp]   Saved {len(jsonl_rows)} rows → {split_path}")
        stats[tier] = len(jsonl_rows)
        combined.extend(jsonl_rows)

    # Shuffle combined and save
    rng.shuffle(combined)
    combined_path = args.out / f"wmdp_bio_combined_{len(combined)}.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for row in combined:
            f.write(json.dumps(row) + "\n")

    # Write a manifest
    manifest = {
        "source": "cais/wmdp-corpora",
        "citation": "Li et al. 2024, arXiv:2403.03218",
        "tier_mapping": {
            "hazard_adjacent_category": "bio_forget_corpus",
            "benign_bio": "bio_retain_corpus",
        },
        "counts": stats,
        "combined_total": len(combined),
        "combined_file": combined_path.name,
        "seed": args.seed,
        "max_per_tier": args.max_per_tier,
        "note": (
            "For SAE contrastive training: use combined file as --eval-set with "
            "scripts/train_sae_local.py. The bio_forget_corpus (hazard-adjacent) "
            "and bio_retain_corpus (benign) are the paired corpora from WMDP "
            "machine-unlearning research, ideal for contrastive SAE fine-tuning."
        ),
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n[wmdp] Done. Combined: {len(combined)} rows ({stats})")
    print(f"[wmdp] Combined JSONL: {combined_path}")
    print(f"[wmdp] Manifest: {manifest_path}")
    print(f"\n[wmdp] Next step — SAE training on WMDP corpus:")
    print(f"  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/train_sae_local.py \\")
    print(f"    --eval-set {combined_path} --out runs/sae-training-wmdp --steps 5000 --batch-size 4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
