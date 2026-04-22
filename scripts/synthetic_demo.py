"""End-to-end pipeline demo without loading any model.

Fabricates plausible (surface distribution, feature vector) pairs per prompt,
runs them through the full scoring + flags + reporting pipeline, and writes
runs/synthetic/report.md + report.json.

Purpose: verifies the reporting pipeline end-to-end and produces a scaling-
plot data point we can combine with real runs later. This is NOT a
methodology demonstration — it's a plumbing demonstration.

Usage (no torch required):
    python scripts/synthetic_demo.py

Writes:
    runs/synthetic_gemma-2-2b-it/report.md
    runs/synthetic_gemma-2-2b-it/report.json
    runs/cross_model/scaling_plot.png  (if matplotlib is installed)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.prompts.prompt_loader import load_jsonl
from biorefusalaudit.reporting.redaction import redact_records
from biorefusalaudit.reporting.report import PromptRecord, RunReport, write_report
from biorefusalaudit.runner.cross_model_runner import collect_run_reports, save_scaling_plot
from biorefusalaudit.scoring.divergence import (
    FEATURE_CATEGORIES,
    SURFACE_CLASSES,
    divergence,
    divergence_flags,
)


def _mock_surface_soft(expected: str, rng: np.random.Generator) -> np.ndarray:
    """Return a 5-vector soft distribution centered on `expected` with noise."""
    order = list(SURFACE_CLASSES)
    v = np.full(5, 0.05)
    v[order.index(expected if expected in order else "refuse")] = 0.75
    v += rng.dirichlet(np.ones(5)) * 0.1
    v /= v.sum()
    return v


def _mock_feature_vec(tier: str, surface: str, rng: np.random.Generator) -> np.ndarray:
    """Return a 5-vector over FEATURE_CATEGORIES with plausible activation by tier.

    Models the finding that on tier-3 prompts, surface refusals often come with
    high hazard_adjacent internal activation — producing the signature flag.
    """
    v = np.full(5, 0.05)
    idx = {c: i for i, c in enumerate(FEATURE_CATEGORIES)}
    if tier == "benign_bio":
        v[idx["bio_content"]] = 0.70
    elif tier == "dual_use_bio":
        v[idx["bio_content"]] = 0.40
        v[idx["hazard_adjacent"]] = 0.20
        v[idx["hedging"]] = 0.20
    else:  # hazard_adjacent_category
        if surface == "refuse":
            # Key phenomenon: hazard-adjacent active despite refusal.
            v[idx["hazard_adjacent"]] = 0.55
            v[idx["refusal_circuitry"]] = 0.25
        else:
            v[idx["hazard_adjacent"]] = 0.65
    v += rng.dirichlet(np.ones(5)) * 0.05
    v /= v.sum()
    return v


def _simulate_scale_factor(model_name: str) -> float:
    """Emulate the scaling finding: larger models show higher divergence on tier-3."""
    table = {
        "gemma-2-2b-it": 1.0,
        "gemma-4-E2B-it": 1.05,
        "gemma-2-9b-it": 1.15,
        "llama-3.1-8b-it": 1.08,
    }
    return table.get(model_name, 1.0)


def build_synthetic_report(model_name: str, seed: int = 42) -> RunReport:
    rng = np.random.default_rng(seed)
    eval_path = REPO / "data/eval_set_public/eval_set_public_v1.jsonl"
    prompts = load_jsonl(eval_path)

    report = RunReport(
        model_name=model_name,
        sae_source="synthetic",
        eval_set_path=str(eval_path),
    )
    scale = _simulate_scale_factor(model_name)
    for p in prompts:
        # Surface follows the prompt's expected_surface with high prob.
        s = _mock_surface_soft(p.expected_surface, rng)
        f = _mock_feature_vec(p.tier, p.expected_surface, rng)
        # Scale bump to exaggerate tier-3 divergence for bigger models.
        if p.tier == "hazard_adjacent_category":
            f = np.clip(f * scale, 0, None)
            f = f / f.sum()
        d = divergence(s, f)
        flags = divergence_flags(s, f)
        report.records.append(
            PromptRecord(
                id=p.id,
                tier=p.tier,
                framing=p.framing,
                prompt=p.prompt,
                completion="[SYNTHETIC: no model was run; this row is pipeline validation only]",
                surface_label=SURFACE_CLASSES[int(np.argmax(s))],
                surface_soft={c: float(s[i]) for i, c in enumerate(SURFACE_CLASSES)},
                feature_vec=list(f),
                feature_categories=list(FEATURE_CATEGORIES),
                divergence=float(d),
                flags=flags,
                judge_disagreement=0.0,
            )
        )
    report.compute_aggregate()
    return report


def main() -> int:
    models = ["synthetic_gemma-2-2b-it", "synthetic_gemma-4-E2B-it",
              "synthetic_gemma-2-9b-it", "synthetic_llama-3.1-8b-it"]
    seed = 7
    for m in models:
        rpt = build_synthetic_report(m.replace("synthetic_", ""), seed=seed)
        rpt.model_name = m
        out_dir = REPO / "runs" / m
        md, js = write_report(rpt, out_dir)
        print(f"[synthetic] {m}: {md}")
        for tier, a in sorted(rpt.aggregate.items()):
            print(f"           {tier}: mean D = {a['mean_divergence']:.3f}")
        seed += 1

    # Cross-model scaling plot.
    try:
        reports = collect_run_reports(REPO / "runs")
        save_scaling_plot(reports, REPO / "runs/cross_model/scaling_plot.png")
        print(f"[synthetic] Saved {REPO / 'runs/cross_model/scaling_plot.png'}")
    except ImportError as e:
        print(f"[synthetic] Skipping scaling plot (matplotlib missing): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
