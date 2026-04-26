"""Re-score a completed report.json using the calibration T_prior.

Needed when pass1 had no catalog features (all f_vec=zero), causing fit_calibration.py
to produce a degenerate all-zero T. Re-scoring avoids rerunning 45-minute inference.

Usage:
    python scripts/rescore_with_prior_T.py \\
        --report runs/gemma-4-E2B-it-our-sae-v1-pass2/report.json \\
        --calibration configs/calibration_gemma4_oursae_v1.yaml \\
        --out runs/gemma-4-E2B-it-our-sae-v1-pass2-rescored/report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from biorefusalaudit.scoring.divergence import divergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cal = yaml.safe_load(Path(args.calibration).read_text())
    T = np.array(cal["T"], dtype=np.float64)

    # Confirm T is not degenerate
    test_s = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    test_expected = T.T @ test_s
    print(f"T sanity — T^T @ [comply-ish s]: {test_expected.round(4)}")
    if np.linalg.norm(test_expected) < 1e-10:
        raise RuntimeError("T is still degenerate (all zeros). Fix calibration YAML first.")

    data = json.loads(Path(args.report).read_text())
    records = data["records"] if isinstance(data, dict) else data
    print(f"Re-scoring {len(records)} records with T from {args.calibration}")

    SURFACE_ORDER = ("refuse", "comply", "partial", "hedge", "evade")
    per_tier = defaultdict(list)
    corrected = []
    for rec in records:
        ss = rec["surface_soft"]
        if isinstance(ss, dict):
            s = np.array([ss.get(k, 0.0) for k in SURFACE_ORDER], dtype=np.float64)
        else:
            s = np.array(ss, dtype=np.float64)
        f = np.array(rec["feature_vec"], dtype=np.float64)
        new_d = divergence(s, f, T)
        rec = dict(rec)
        rec["divergence_original"] = rec.get("divergence")
        rec["divergence"] = round(new_d, 6)
        rec["divergence_note"] = "recomputed with T_prior; original T was degenerate (fitted on zero-feature pass1 data)"
        corrected.append(rec)
        per_tier[rec.get("tier", "unknown")].append(new_d)

    # Preserve top-level metadata, replace records with rescored list
    if isinstance(data, dict):
        data["records"] = corrected
        data["rescore_note"] = "D recomputed with T_prior; original T was degenerate (zero-matrix from pass1 with empty catalog)"
        output = data
    else:
        output = corrected

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Written to {out_path}")

    print("\n=== Per-tier mean D (T_prior) ===")
    for tier in sorted(per_tier):
        vals = per_tier[tier]
        print(f"  {tier}: n={len(vals)} mean={np.mean(vals):.4f} ± {np.std(vals):.4f}  "
              f"[{np.min(vals):.4f}, {np.max(vals):.4f}]")

    all_vals = [r["divergence"] for r in corrected]
    print(f"\n  overall: n={len(all_vals)} mean={np.mean(all_vals):.4f} ± {np.std(all_vals):.4f}")

    # Surface label distribution by tier
    print("\n=== Surface label distribution by tier ===")
    tier_labels = defaultdict(lambda: defaultdict(int))
    for rec in corrected:
        tier_labels[rec.get("tier", "unknown")][rec.get("surface_label", "??")] += 1
    for tier in sorted(tier_labels):
        counts = dict(tier_labels[tier])
        total = sum(counts.values())
        print(f"  {tier} (n={total}): " +
              "  ".join(f"{k}={v} ({100*v/total:.0f}%)" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
