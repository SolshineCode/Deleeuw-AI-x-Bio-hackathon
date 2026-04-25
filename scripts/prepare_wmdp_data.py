"""Download and prepare WMDP bio corpus for local SAE training.

Downloads accessible cais/wmdp-corpora subsets from HuggingFace and saves them
as JSONL with BioRefusalAudit tier labels, ready for use with train_sae_local.py.

IMPORTANT FINDING (2026-04-25): cais/wmdp-corpora/bio_forget_corpus is NOT publicly
available on HuggingFace. Available public configs are:
  bio-retain-corpus   (benign biology, ~3.7K docs)
  cyber-forget-corpus (hazardous cyber, ~2.0K docs)
  cyber-retain-corpus (benign cyber, ~2.5K docs)

This confirms the institutional data gap described in §8 of the BioRefusalAudit paper:
genuine bio-hazard activation corpora require gated institutional access. The benign
biology corpus can be used for the negative class; for the positive (hazard) class,
this script falls back to the BioRefusalAudit eval set's hazard_adjacent tier (22 prompts).
This is severely imbalanced (5000:22) and SAE contrastive convergence is not expected —
which directly demonstrates the binding constraint.

Output schema per row (same as eval_set_public_v1.jsonl):
  {
    "id": "wmdp_bio_<split>_<idx>",
    "tier": "hazard_adjacent_category" | "benign_bio",
    "framing": "direct",
    "prompt": "<text>",
    "source": "cais/wmdp-corpora/bio-retain-corpus" | "eval_set_public_v1"
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

    combined: list[dict] = []
    stats: dict[str, int] = {}

    # --- Benign class: bio-retain-corpus (publicly available, note: HYPHENS not underscores) ---
    print("[wmdp] Loading cais/wmdp-corpora/bio-retain-corpus (benign biology)...")
    try:
        ds_benign = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
        rows_benign = [{"text": r["text"]} for r in ds_benign if r.get("text", "").strip()]
        print(f"[wmdp]   bio-retain-corpus: {len(rows_benign)} raw rows")
        if not args.no_shuffle:
            rng.shuffle(rows_benign)
        rows_benign = rows_benign[: args.max_per_tier]
        benign_rows = [
            {
                "id": f"wmdp_bio_retain_{i:05d}",
                "tier": "benign_bio",
                "framing": "direct",
                "prompt": r["text"],
                "source": "cais/wmdp-corpora/bio-retain-corpus",
            }
            for i, r in enumerate(rows_benign)
        ]
        split_path = args.out / f"wmdp_bio_retain_{len(benign_rows)}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for row in benign_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[wmdp]   Saved {len(benign_rows)} benign rows -> {split_path}")
        stats["benign_bio"] = len(benign_rows)
        combined.extend(benign_rows)
    except Exception as e:
        print(f"[wmdp] ERROR loading bio-retain-corpus: {e}", file=sys.stderr)
        return 1

    # --- Hazard class: bio_forget_corpus IS NOT publicly available ---
    # Confirmed 2026-04-25: cais/wmdp-corpora does not expose bio_forget_corpus publicly.
    # Available public configs: bio-retain-corpus, cyber-forget-corpus, cyber-retain-corpus.
    # This directly confirms the institutional data bottleneck from BioRefusalAudit §8.
    # Fallback: use the eval set's hazard_adjacent prompts (22 prompts — severely imbalanced).
    print("[wmdp] NOTE: bio_forget_corpus is NOT publicly available on cais/wmdp-corpora.")
    print("[wmdp]   This confirms the institutional data gap from BioRefusalAudit §8.")
    print("[wmdp]   Falling back to eval_set hazard_adjacent tier (22 prompts).")
    eval_set_path = REPO / "data" / "eval_set_public" / "eval_set_public_v1.jsonl"
    try:
        with open(eval_set_path, encoding="utf-8") as f:
            eval_rows = [json.loads(line) for line in f if line.strip()]
        hazard_rows_raw = [r for r in eval_rows if r.get("tier") == "hazard_adjacent_category"]
        hazard_jsonl = [
            {
                "id": r["id"],
                "tier": "hazard_adjacent_category",
                "framing": r.get("framing", "direct"),
                "prompt": r["prompt"],
                "source": "eval_set_public_v1 (fallback — bio_forget_corpus unavailable)",
            }
            for r in hazard_rows_raw
        ]
        split_path = args.out / f"wmdp_bio_hazard_fallback_{len(hazard_jsonl)}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for row in hazard_jsonl:
                f.write(json.dumps(row) + "\n")
        print(f"[wmdp]   Saved {len(hazard_jsonl)} hazard rows (fallback) -> {split_path}")
        print(f"[wmdp]   WARNING: {len(benign_rows)}:{len(hazard_jsonl)} class imbalance.")
        print(f"[wmdp]   Contrastive convergence is not expected with this corpus.")
        stats["hazard_adjacent_category"] = len(hazard_jsonl)
        combined.extend(hazard_jsonl)
    except Exception as e:
        print(f"[wmdp] ERROR loading eval set fallback: {e}", file=sys.stderr)
        return 1

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
