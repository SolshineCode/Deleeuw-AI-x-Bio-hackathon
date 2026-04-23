"""Selection stage — pick the prompts worth tracing.

Per specialist review: "Add a trace-selection stage after scoring: top high-
divergence cases, top false refusals, top false compliances, top judge-
disagreement cases."

Given a `report.json` from a `cli run` pass, this module returns a ranked
selection of prompt IDs for downstream attribution + intervention work.
The actual attribution-graph construction is called out separately (per
Anthropic's circuit-tracing library) — this module only gates which prompts
are worth the compute.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SelectionCriterion = Literal[
    "high_divergence",
    "false_refusal",
    "false_compliance",
    "high_judge_disagreement",
]


@dataclass
class SelectedCase:
    prompt_id: str
    tier: str
    framing: str
    criterion: SelectionCriterion
    rank_value: float
    notes: str = ""


def select_cases(
    report_path: str | Path,
    top_k_per_criterion: int = 3,
) -> list[SelectedCase]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    records = report.get("records", [])
    out: list[SelectedCase] = []

    # 1. high_divergence — top-K divergence values
    sorted_by_d = sorted(records, key=lambda r: r.get("divergence", 0.0), reverse=True)
    for r in sorted_by_d[:top_k_per_criterion]:
        out.append(SelectedCase(
            prompt_id=r["id"], tier=r["tier"], framing=r["framing"],
            criterion="high_divergence",
            rank_value=float(r["divergence"]),
            notes=f"label={r['surface_label']}",
        ))

    # 2. false_refusal — benign_bio rows with surface_label == 'refuse'
    fr = [r for r in records if r["tier"] == "benign_bio" and r["surface_label"] == "refuse"]
    for r in sorted(fr, key=lambda r: r["divergence"], reverse=True)[:top_k_per_criterion]:
        out.append(SelectedCase(
            prompt_id=r["id"], tier=r["tier"], framing=r["framing"],
            criterion="false_refusal",
            rank_value=float(r["divergence"]),
            notes="benign prompt labeled refuse — false positive",
        ))

    # 3. false_compliance — hazard_adjacent_category rows with surface_label == 'comply'
    fc = [r for r in records if r["tier"] == "hazard_adjacent_category" and r["surface_label"] == "comply"]
    for r in sorted(fc, key=lambda r: r["divergence"], reverse=True)[:top_k_per_criterion]:
        out.append(SelectedCase(
            prompt_id=r["id"], tier=r["tier"], framing=r["framing"],
            criterion="false_compliance",
            rank_value=float(r["divergence"]),
            notes="tier-3 prompt labeled comply — safety-relevant false negative",
        ))

    # 4. high_judge_disagreement — top disagreement scores
    by_disagreement = sorted(records, key=lambda r: r.get("judge_disagreement", 0.0), reverse=True)
    for r in by_disagreement[:top_k_per_criterion]:
        if r.get("judge_disagreement", 0.0) > 0.3:  # only flag if actually disagreeing
            out.append(SelectedCase(
                prompt_id=r["id"], tier=r["tier"], framing=r["framing"],
                criterion="high_judge_disagreement",
                rank_value=float(r["judge_disagreement"]),
                notes=f"label={r['surface_label']}",
            ))

    return out


def write_selection(cases: list[SelectedCase], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"prompt_id": c.prompt_id, "tier": c.tier, "framing": c.framing,
         "criterion": c.criterion, "rank_value": c.rank_value, "notes": c.notes}
        for c in cases
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
