"""Standalone smoke test: load Gemma 2 2B-IT + Gemma Scope 1 SAE, run 3 prompts through the pipeline.

Usage (from repo root, with .venv activated):
    python scripts/smoke_activation_test.py

Writes runs/smoke/sae_activations_sample.json + runs/smoke/report.md.
If this succeeds, the full CLI run is expected to work end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.features.feature_profiler import FeatureCatalog
from biorefusalaudit.models.model_adapter import load_model
from biorefusalaudit.models.sae_adapter import load_sae
from biorefusalaudit.prompts.prompt_loader import load_jsonl
from biorefusalaudit.runner.eval_runner import run_eval
from biorefusalaudit.reporting.report import write_report


def main() -> int:
    model_id = "google/gemma-2-2b-it"
    eval_path = REPO / "data/eval_set_public/eval_set_public_v1.jsonl"
    catalog_path = REPO / "data/feature_catalog/gemma-2-2b-it.json"
    out_dir = REPO / "runs/smoke"

    print(f"[smoke] Loading model: {model_id}")
    lm = load_model(model_id, quantize=None, dtype="float16")

    print("[smoke] Loading SAE: gemma-scope-2b-pt-res layer 12")
    sae = load_sae(
        source="gemma_scope_1",
        repo_or_path="gemma-scope-2b-pt-res",
        layer=12,
        sae_id="layer_12/width_16k/average_l0_71",
    )

    print("[smoke] Loading feature catalog")
    cat = FeatureCatalog.load(catalog_path)

    print(f"[smoke] Loading first 3 prompts from {eval_path}")
    prompts = load_jsonl(eval_path)[:3]

    print("[smoke] Running pipeline on 3 prompts...")
    report = run_eval(
        prompts=prompts,
        lm=lm,
        sae=sae,
        catalog=cat,
        hook_layer=12,
        eval_set_path=str(eval_path),
    )

    md, js = write_report(report, out_dir)
    print(f"[smoke] Wrote {md}")
    print(f"[smoke] Wrote {js}")

    sample = {
        "model": model_id,
        "n_prompts": len(report.records),
        "per_prompt": [
            {
                "id": r.id,
                "surface_label": r.surface_label,
                "divergence": r.divergence,
                "flags": r.flags,
            }
            for r in report.records
        ],
    }
    (out_dir / "sae_activations_sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    print(f"[smoke] Wrote sample: {out_dir / 'sae_activations_sample.json'}")
    print(f"[smoke] Aggregate: {report.aggregate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
