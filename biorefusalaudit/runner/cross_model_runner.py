"""Iterate eval_runner over a list of (model, SAE) targets from models.yaml.

Produces a combined comparison table and a matplotlib scaling plot across
models and tiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


def load_models_yaml(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("models", [])


def collect_run_reports(runs_root: str | Path) -> list[tuple[str, dict]]:
    """Return [(model_name, report_json_payload)] for every runs/*/report.json."""
    root = Path(runs_root)
    out: list[tuple[str, dict]] = []
    for report_path in sorted(root.glob("*/report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((report_path.parent.name, payload))
    return out


def build_comparison_table(reports: Sequence[tuple[str, dict]]) -> str:
    """Render a markdown comparison table across the collected reports."""
    tiers = ("benign_bio", "dual_use_bio", "hazard_adjacent_category")
    lines: list[str] = []
    lines.append("| Model | " + " | ".join(f"{t} mean D" for t in tiers) + " |")
    lines.append("|---|" + "|".join("---:" for _ in tiers) + "|")
    for name, payload in reports:
        agg = payload.get("aggregate", {})
        row = [name]
        for t in tiers:
            row.append(f"{agg.get(t, {}).get('mean_divergence', 0.0):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_scaling_plot(reports: Sequence[tuple[str, dict]], out_path: str | Path) -> None:
    """Save a matplotlib bar chart comparing tiers across models to out_path (png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = ("benign_bio", "dual_use_bio", "hazard_adjacent_category")
    models = [name for name, _ in reports]
    data = {t: [] for t in tiers}
    for _, payload in reports:
        agg = payload.get("aggregate", {})
        for t in tiers:
            data[t].append(agg.get(t, {}).get("mean_divergence", 0.0))

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(tiers):
        ax.bar(x + (i - 1) * w, data[t], w, label=t)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Mean divergence D")
    ax.set_title("BioRefusalAudit — divergence by tier × model")
    ax.legend()
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
