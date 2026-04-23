"""Rebuild the cross-model scaling plot from any runs/*/report.json present.

Usage:
    python scripts/build_scaling_plot.py [--out demo/scaling_plot.png]

Skips synthetic_* runs unless --include-synthetic is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.runner.cross_model_runner import (
    build_comparison_table,
    collect_run_reports,
    save_scaling_plot,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--out", default=str(REPO / "demo" / "scaling_plot.png"))
    ap.add_argument("--include-synthetic", action="store_true")
    args = ap.parse_args()

    reports = collect_run_reports(args.runs_dir)
    if not args.include_synthetic:
        reports = [(n, p) for n, p in reports if not n.startswith("synthetic_")]

    if not reports:
        print("[scaling] No non-synthetic reports found. Run the CLI first, or pass --include-synthetic.")
        return 1

    print(f"[scaling] Found {len(reports)} reports:")
    for name, _ in reports:
        print(f"           - {name}")

    save_scaling_plot(reports, args.out)
    print(f"[scaling] Saved {args.out}")

    md_table = build_comparison_table(reports)
    table_path = Path(args.out).with_suffix(".md")
    table_path.write_text("# Cross-model comparison\n\n" + md_table + "\n", encoding="utf-8")
    print(f"[scaling] Saved {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
