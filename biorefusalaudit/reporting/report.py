"""Markdown + JSON report generation for a single-model run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

from biorefusalaudit.reporting.redaction import redact_records


@dataclass
class PromptRecord:
    id: str
    tier: str
    framing: str
    prompt: str
    completion: str
    surface_label: str
    surface_soft: dict[str, float]
    feature_vec: list[float]
    feature_categories: list[str]
    divergence: float
    flags: dict[str, bool]
    judge_disagreement: float


@dataclass
class RunReport:
    model_name: str
    sae_source: str
    eval_set_path: str
    records: list[PromptRecord] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)

    def compute_aggregate(self) -> None:
        """Populate self.aggregate with per-tier means and flag counts."""
        by_tier: dict[str, list[PromptRecord]] = {}
        for r in self.records:
            by_tier.setdefault(r.tier, []).append(r)

        agg: dict[str, dict] = {}
        for tier, rs in by_tier.items():
            divs = [r.divergence for r in rs]
            mean_d = mean(divs) if divs else 0.0
            std_d = stdev(divs) if len(divs) > 1 else 0.0
            flag_totals: dict[str, int] = {}
            for r in rs:
                for flag, v in r.flags.items():
                    flag_totals[flag] = flag_totals.get(flag, 0) + int(bool(v))
            agg[tier] = {
                "n": len(rs),
                "mean_divergence": mean_d,
                "std_divergence": std_d,
                "flag_counts": flag_totals,
            }
        self.aggregate = agg


def _render_markdown(report: RunReport) -> str:
    lines: list[str] = []
    lines.append(f"# BioRefusalAudit Run: `{report.model_name}`")
    lines.append("")
    lines.append(f"- Model: `{report.model_name}`")
    lines.append(f"- SAE source: `{report.sae_source}`")
    lines.append(f"- Eval set: `{report.eval_set_path}`")
    lines.append(f"- Total prompts: {len(report.records)}")
    lines.append("")
    lines.append("## Aggregate by tier")
    lines.append("")
    lines.append("| Tier | n | Mean divergence | Std | Hazard-active-on-refuse | Refusal-active-on-comply | Deception-correlate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tier, a in sorted(report.aggregate.items()):
        flags = a.get("flag_counts", {})
        lines.append(
            f"| `{tier}` | {a['n']} | {a['mean_divergence']:.3f} | {a['std_divergence']:.3f} "
            f"| {flags.get('hazard_features_active_despite_refusal', 0)} "
            f"| {flags.get('refusal_features_active_despite_compliance', 0)} "
            f"| {flags.get('deception_correlate_active', 0)} |"
        )
    lines.append("")
    lines.append("## Per-prompt records")
    lines.append("")
    lines.append("See `report.json` for machine-readable records (tier-3 completions redacted).")
    return "\n".join(lines) + "\n"


def write_report(report: RunReport, out_dir: str | Path) -> tuple[Path, Path]:
    """Write report.md + report.json under out_dir. Returns (md_path, json_path)."""
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    report.compute_aggregate()

    md_path = outp / "report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    json_records = redact_records([asdict(r) for r in report.records])
    payload = {
        "model_name": report.model_name,
        "sae_source": report.sae_source,
        "eval_set_path": report.eval_set_path,
        "aggregate": report.aggregate,
        "records": json_records,
    }
    json_path = outp / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path
