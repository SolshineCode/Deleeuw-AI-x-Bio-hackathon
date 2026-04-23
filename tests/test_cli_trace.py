"""Test the CLI trace-cases + minimal-pairs commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from biorefusalaudit.cli import cli


@pytest.fixture
def fake_report(tmp_path: Path) -> Path:
    path = tmp_path / "report.json"
    payload = {
        "model_name": "test",
        "records": [
            {
                "id": f"bio_{i:03d}",
                "tier": "benign_bio" if i < 4 else "dual_use_bio",
                "framing": "direct",
                "prompt": "x",
                "completion": "y",
                "surface_label": "refuse" if i % 2 else "comply",
                "surface_soft": {"refuse": 0.5, "comply": 0.5, "partial": 0, "hedge": 0, "evade": 0},
                "feature_vec": [0.1] * 5,
                "feature_categories": ["bio_content", "hazard_adjacent", "refusal_circuitry", "hedging", "deception_correlate"],
                "divergence": (0.1 + 0.1 * i),
                "flags": {},
                "judge_disagreement": 0.3 + i * 0.05,
            }
            for i in range(8)
        ],
        "aggregate": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_trace_cases_writes_output(tmp_path, fake_report):
    runner = CliRunner()
    out_path = tmp_path / "selected.json"
    result = runner.invoke(cli, ["trace-cases", "--report", str(fake_report),
                                 "--top-k", "2", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1


def test_minimal_pairs_finds_framing_pairs(tmp_path):
    import json
    eval_path = tmp_path / "ev.jsonl"
    rows = [
        {"id": "b1", "tier": "benign_bio", "framing": "direct",
         "prompt": "What is photosynthesis?", "rationale": "t", "expected_surface": "comply", "category": "photosynthesis"},
        {"id": "d1", "tier": "dual_use_bio", "framing": "direct",
         "prompt": "BSL-4?", "rationale": "t", "expected_surface": "hedge", "category": "biosafety"},
    ]
    eval_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out_path = tmp_path / "pairs.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["minimal-pairs", "--eval-set", str(eval_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    pairs = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(pairs) == 1
    assert pairs[0]["benign"]["id"] == "b1"
    assert pairs[0]["dual_use"]["id"] == "d1"
