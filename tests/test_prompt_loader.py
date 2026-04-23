"""Unit tests for DualUsePrompt + JSONL loader."""

import json
from pathlib import Path

import pytest

from biorefusalaudit.prompts.prompt_loader import DualUsePrompt, load_jsonl


def test_loads_public_eval_set_53_rows():
    """The eval set file authored for the hackathon submission must load cleanly."""
    path = (
        Path(__file__).parent.parent
        / "data"
        / "eval_set_public"
        / "eval_set_public_v1.jsonl"
    )
    if not path.exists():
        pytest.skip(f"eval set not present at {path}")
    rows = load_jsonl(path)
    assert len(rows) >= 53  # 23 benign + 30 dual_use (tier-3 may be appended later)
    tiers = {r.tier for r in rows}
    assert {"benign_bio", "dual_use_bio"} <= tiers


def test_malformed_row_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "tier": "NOT_A_TIER"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_empty_prompt_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    obj = {
        "id": "x",
        "tier": "benign_bio",
        "framing": "direct",
        "prompt": "   ",
        "rationale": "test",
        "expected_surface": "comply",
        "category": "test",
    }
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_valid_row_roundtrip(tmp_path):
    p = tmp_path / "ok.jsonl"
    obj = {
        "id": "b1",
        "tier": "benign_bio",
        "framing": "direct",
        "prompt": "Explain photosynthesis.",
        "rationale": "textbook",
        "expected_surface": "comply",
        "category": "photosynthesis",
        "sampled_from": "Alberts",
    }
    p.write_text(json.dumps(obj) + "\n", encoding="utf-8")
    rows = load_jsonl(p)
    assert len(rows) == 1
    assert rows[0].id == "b1"
    assert rows[0].sampled_from == "Alberts"


def test_default_sampled_from():
    p = DualUsePrompt(
        id="x",
        tier="benign_bio",
        framing="direct",
        prompt="ok",
        rationale="r",
        expected_surface="comply",
        category="c",
    )
    assert p.sampled_from == "original"
