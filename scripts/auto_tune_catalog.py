"""Replace a stub feature catalog with empirically-firing features per category.

Per specialist review (2026-04-23): "The feature catalogs are explicitly
described as plausible-but-unvalidated stubs, and the local smoke test
produced zero feature activation projection on bio prompts. If the indexed
features are not the features actually firing, the divergence score can
look neat while pointing at the wrong mechanism. Treat catalog validation
as the top blocker."

This script consumes the `activations.npz` dumped by a `cli run --dump-activations`
pass and produces a data-driven feature catalog: for each target category
({bio_content, hazard_adjacent, refusal_circuitry, hedging, deception_correlate}),
identify the SAE features that fire with maximum Cohen's-d effect size between
the category's positive-signal prompts and a neutral baseline drawn from the
eval set.

The default mapping of tier → category:
  - benign_bio prompts → positive signal for `bio_content`
  - dual_use_bio prompts → positive signal for `hazard_adjacent`
  - hazard_adjacent_category → positive signal for `refusal_circuitry`
  - (hedge-labeled completions, if report.json is passed in) → `hedging`

The `deception_correlate` category is left as a small hand-curated stub
because deception-correlated features don't have a clean positive-control
source in the bio-safety eval set (they need a separate calibration).

Usage:
    python scripts/auto_tune_catalog.py \
        --activations runs/gemma-2-2b-it-L12-activations/activations.npz \
        --existing data/feature_catalog/gemma-2-2b-it.json \
        --out data/feature_catalog/gemma-2-2b-it.json \
        --top-per-category 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.features.feature_profiler import FeatureCatalog
from biorefusalaudit.features.feature_validator import validate_catalog


CATEGORY_TO_POSITIVE_TIER = {
    "bio_content": "benign_bio",
    "hazard_adjacent": "dual_use_bio",
    "refusal_circuitry": "hazard_adjacent_category",
}


def cohens_d_per_feature(pos: np.ndarray, neg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-feature Cohen's-d effect size: |mean(pos) - mean(neg)| / pooled_std."""
    mu_p = pos.mean(axis=0)
    mu_n = neg.mean(axis=0)
    var_p = pos.var(axis=0)
    var_n = neg.var(axis=0)
    pooled_std = np.sqrt((var_p + var_n) / 2 + eps)
    return np.abs(mu_p - mu_n) / pooled_std


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True, type=Path,
                    help="Path to activations.npz from an eval run")
    ap.add_argument("--existing", required=True, type=Path,
                    help="Existing catalog JSON (its note field + model_name + sae_source are preserved)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-per-category", default=20, type=int)
    ap.add_argument("--min-effect-size", default=0.1, type=float,
                    help="Minimum Cohen's-d to include a feature; below this is noise")
    args = ap.parse_args()

    print(f"[auto_tune] Loading {args.activations}")
    npz = np.load(args.activations, allow_pickle=True)
    ids = npz["ids"]
    tiers = npz["tiers"]
    feats = npz["feature_acts"]  # (N, d_sae)
    print(f"[auto_tune] {feats.shape[0]} prompts, d_sae = {feats.shape[1]}")

    existing_catalog = FeatureCatalog.load(args.existing)
    new_categories: dict[str, list[int]] = {}

    for category, pos_tier in CATEGORY_TO_POSITIVE_TIER.items():
        pos_mask = tiers == pos_tier
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos < 3 or n_neg < 3:
            print(f"[auto_tune] {category}: skipping — insufficient pos/neg ({n_pos}/{n_neg})")
            new_categories[category] = existing_catalog.categories.get(category, [])
            continue

        effect = cohens_d_per_feature(feats[pos_mask], feats[neg_mask])
        # Keep only features that (a) exceed min-effect-size and (b) have nonzero
        # mean activation on the positive side (avoids picking features that only
        # differ because the negative side is noisy-nonzero while positive is dead).
        pos_mean = feats[pos_mask].mean(axis=0)
        valid = (effect > args.min_effect_size) & (pos_mean > 0)
        effect_masked = np.where(valid, effect, -np.inf)
        top_idx = np.argsort(-effect_masked)[: args.top_per_category]
        top_idx = [int(i) for i in top_idx if effect_masked[i] > -np.inf]
        new_categories[category] = top_idx
        print(f"[auto_tune] {category}: picked {len(top_idx)} features; top effect = {effect[top_idx[0]]:.2f}" if top_idx else f"[auto_tune] {category}: no features passed filter")

    # Preserve hedging + deception_correlate from existing catalog — those
    # positive-control sources aren't cleanly available in the eval set.
    for category in ("hedging", "deception_correlate"):
        if category in existing_catalog.categories:
            new_categories[category] = existing_catalog.categories[category]
            print(f"[auto_tune] {category}: preserved from existing catalog ({len(new_categories[category])} features)")

    out_obj = {
        "model_name": existing_catalog.model_name,
        "sae_source": existing_catalog.sae_source,
        "catalog_version": "0.2-auto-tuned",
        "catalog_note": (
            "Categories bio_content / hazard_adjacent / refusal_circuitry auto-tuned via "
            "Cohen's-d on tier-labeled activation dumps from the eval set (see "
            f"source_activations below). Categories hedging / deception_correlate "
            "preserved from v0.1 stub — these lack clean positive controls in the "
            "eval set and need separate calibration."
        ),
        "source_activations": str(args.activations),
        "top_per_category": args.top_per_category,
        "min_effect_size": args.min_effect_size,
        "categories": new_categories,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[auto_tune] Wrote {args.out}")

    # Validate
    updated = FeatureCatalog.load(args.out)
    report = validate_catalog(updated, d_sae=feats.shape[1])
    if report.problems:
        print(f"[auto_tune] Validator problems: {report.problems}")
        return 1
    print(f"[auto_tune] Validator clean. Stats: {report.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
