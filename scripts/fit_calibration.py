"""Fit the alignment matrix T on real (surface_soft, feature_vec) pairs.

Per specialist review (2026-04-23): "The calibration matrix T is prior-only
and has not yet been fit on real activation data. Your scalar scores are
still partially hand-shaped rather than empirically aligned to observed
activations. Fit T only after catalog tuning, then rerun the full report
and show pre/post calibration deltas."

This script consumes a `report.json` from a `cli run` pass and its
`prompt_records.surface_soft` + `feature_vec` fields to assemble an (N, 5)
pair matrix. The fit is ridge-regularized least squares via
`fit_alignment_matrix` in `biorefusalaudit/scoring/calibration.py`.

The existing prior-T in `configs/calibration_gemma2_2b.yaml` is retained as
a historical block labeled `fit_method: prior` so reproducibility of prior
divergence scores is preserved. The new fit is written as a new block
appended to the same config with the current date.

Usage:
    python scripts/fit_calibration.py \\
        --report runs/gemma-2-2b-it-L12-activations/report.json \\
        --config configs/calibration_gemma2_2b.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.scoring.calibration import fit_alignment_matrix
from biorefusalaudit.scoring.divergence import (
    FEATURE_CATEGORIES,
    SURFACE_CLASSES,
    divergence,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--reg-lambda", default=0.1, type=float)
    ap.add_argument("--max-condition", default=1e6, type=float)
    args = ap.parse_args()

    print(f"[fit_cal] Loading {args.report}")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    records = report["records"]

    # Build (N, 5) surface + feature arrays (redacted tier-3 still retains the
    # numeric vectors; only prompt/completion text are redacted).
    surf = np.zeros((len(records), 5), dtype=np.float64)
    feat = np.zeros((len(records), 5), dtype=np.float64)
    for i, r in enumerate(records):
        s = r["surface_soft"]
        for j, c in enumerate(SURFACE_CLASSES):
            surf[i, j] = float(s.get(c, 0.0))
        feat[i] = np.asarray(r["feature_vec"], dtype=np.float64)

    print(f"[fit_cal] {len(records)} pairs; fitting T...")
    T_new, diag = fit_alignment_matrix(
        surf, feat, reg_lambda=args.reg_lambda, max_condition=args.max_condition
    )
    print(f"[fit_cal] residual MSE={diag['residual_mse']:.4f} cond={diag['condition_number']:.2e}")

    # Compute pre/post divergence deltas using the prior T from config.
    prior_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prior_T = np.array(prior_cfg["T"], dtype=np.float64)
    deltas_per_row = []
    for i in range(len(records)):
        d_prior = divergence(surf[i], feat[i], prior_T)
        d_new = divergence(surf[i], feat[i], T_new)
        deltas_per_row.append({
            "id": records[i]["id"],
            "tier": records[i]["tier"],
            "d_prior": float(d_prior),
            "d_new": float(d_new),
            "delta": float(d_new - d_prior),
        })
    abs_deltas = np.abs([d["delta"] for d in deltas_per_row])
    print(f"[fit_cal] Divergence shift: mean |Delta|={abs_deltas.mean():.3f} max |Delta|={abs_deltas.max():.3f}")

    # Append new fit block to config, preserve existing.
    updated = dict(prior_cfg)
    updated["T_prior"] = prior_T.tolist()
    updated["T_prior_note"] = prior_cfg.get("fit_method", "identity-biased prior") + " (was T)"
    updated["T"] = T_new.tolist()
    updated["fit_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    updated["fit_samples"] = int(diag["n_samples"])
    updated["fit_method"] = f"ridge least-squares (lambda={args.reg_lambda}) on {args.report.name}"
    updated["fit_residual_mse"] = float(diag["residual_mse"])
    updated["fit_condition_number"] = float(diag["condition_number"])

    args.config.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    print(f"[fit_cal] Updated {args.config}")

    # Write delta report for traceability.
    delta_path = args.report.parent / "calibration_delta.json"
    delta_path.write_text(json.dumps({
        "report": str(args.report),
        "config": str(args.config),
        "per_row": deltas_per_row,
        "summary": {
            "mean_abs_delta": float(abs_deltas.mean()),
            "max_abs_delta": float(abs_deltas.max()),
            "n_rows": len(records),
        }
    }, indent=2), encoding="utf-8")
    print(f"[fit_cal] Wrote {delta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
