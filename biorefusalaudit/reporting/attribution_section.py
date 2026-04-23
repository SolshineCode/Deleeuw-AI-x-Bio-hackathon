"""Render an attribution section in the run report + save graph JSON alongside.

Per specialist review — "Save graph JSON alongside report.json, then render
a compact panel in the dashboard with node names, edge weights, and one-click
counterfactual reruns."

This module is the reporting-side stub: given a per-feature evidence list
(from `features.attribution_labels.FeatureEvidence`) and a set of
intervention results (from `scripts/run_intervention.py` output), emit a
`attribution.json` artifact + a markdown section for the report.

Full attribution-graph integration (Anthropic's circuit-tracing API) is a
follow-on; this module provides the schema + serialization layer so the
rest of the pipeline can be wired today.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from biorefusalaudit.features.attribution_labels import FeatureEvidence, evidence_summary


def write_attribution_artifact(
    out_dir: str | Path,
    evidence_list: Iterable[FeatureEvidence],
    intervention_records: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Write attribution.json + attribution.md to out_dir; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    evs = list(evidence_list)
    summary = evidence_summary(evs)
    payload = {
        "per_feature_evidence": [asdict(e) for e in evs],
        "summary": summary,
        "intervention_records": intervention_records or [],
    }
    json_path = out / "attribution.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = ["# Attribution & Intervention Evidence\n"]
    tiers = summary["tier_counts"]
    lines.append(f"- candidate: {tiers.get('candidate', 0)}")
    lines.append(f"- activation_only: {tiers.get('activation_only', 0)}")
    lines.append(f"- attribution_only: {tiers.get('attribution_only', 0)}")
    lines.append(f"- **named_circuit**: {tiers.get('named_circuit', 0)}")
    lines.append("")
    lines.append("## Named circuits\n")
    named = summary["named_circuits"]
    if not named:
        lines.append("_None — no feature passed all three evidence legs_.")
    else:
        lines.append("| Category | Feature ID | Activation d | Attribution w | Perturbation ΔD | Label changed |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for c in named:
            lines.append(
                f"| {c['category']} | {c['feature_id']} | {c['activation_effect_size']:.2f} "
                f"| {c['attribution_weight']:.2f} | {c['perturbation_divergence_delta']:.3f} "
                f"| {'yes' if c['perturbation_label_changed'] else 'no'} |"
            )
    lines.append("")
    if intervention_records:
        lines.append("## Intervention experiments\n")
        for r in intervention_records:
            s = r.get("intervention_summary", {})
            lines.append(
                f"- `{r.get('prompt_id', '?')}` / `{r.get('category', '?')}` — "
                f"ablate ΔD={s.get('divergence_delta_ablate', 0.0):.3f} | "
                f"boost ΔD={s.get('divergence_delta_boost', 0.0):.3f} | "
                f"named-circuit: **{s.get('qualifies_as_named_circuit', False)}**"
            )
    md_path = out / "attribution.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
