"""Unit tests for cross_model_runner utilities."""

import json
from pathlib import Path

from biorefusalaudit.runner.cross_model_runner import (
    build_comparison_table,
    collect_run_reports,
)


def _write_report(path: Path, mean_d_per_tier: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": path.parent.name,
        "aggregate": {
            t: {"n": 10, "mean_divergence": d, "std_divergence": 0.05, "flag_counts": {}}
            for t, d in mean_d_per_tier.items()
        },
        "records": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_run_reports_picks_up_report_jsons(tmp_path):
    _write_report(tmp_path / "model_a" / "report.json", {"benign_bio": 0.1})
    _write_report(tmp_path / "model_b" / "report.json", {"benign_bio": 0.2})
    reports = collect_run_reports(tmp_path)
    assert len(reports) == 2
    names = {n for n, _ in reports}
    assert names == {"model_a", "model_b"}


def test_collect_ignores_non_report_files(tmp_path):
    (tmp_path / "stuff").mkdir()
    (tmp_path / "stuff" / "not_a_report.txt").write_text("hello")
    assert collect_run_reports(tmp_path) == []


def test_build_comparison_table_includes_all_models(tmp_path):
    _write_report(tmp_path / "a" / "report.json", {"benign_bio": 0.1, "dual_use_bio": 0.2})
    _write_report(tmp_path / "b" / "report.json", {"benign_bio": 0.05, "hazard_adjacent_category": 0.5})
    reports = collect_run_reports(tmp_path)
    md = build_comparison_table(reports)
    assert "| a |" in md
    assert "| b |" in md
    assert "benign_bio mean D" in md
