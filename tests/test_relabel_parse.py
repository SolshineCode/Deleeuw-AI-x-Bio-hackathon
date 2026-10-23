"""Parsing and safety guards of scripts/relabel_openrouter_free.py (2026-10-22)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import relabel_openrouter_free as R  # noqa: E402


def test_parse_takes_last_json_list_after_thinking():
    labels, st = R.parse_labels('thinking [x] then [{"item":1,"label":"comply"},{"item":2,"label":"hedge"}]', 2)
    assert st == "ok" and labels == ["comply", "hedge"]


def test_unclear_and_unknown_labels_are_none():
    labels, _ = R.parse_labels('[{"item":1,"label":"unclear"},{"item":2,"label":"maybe"}]', 2)
    assert labels == [None, None]


def test_judge_refusal_is_not_a_label():
    labels, st = R.parse_labels("I'm sorry, I can't help with that.", 3)
    assert labels == [None, None, None] and st == "judge_refused"


def test_missing_items_are_none():
    labels, _ = R.parse_labels('[{"item":2,"label":"refuse"}]', 3)
    assert labels == [None, "refuse", None]


def test_non_free_model_is_refused():
    with pytest.raises(AssertionError):
        R.call_judge("k", "openai/gpt-4o", [("p", "c")])


def test_default_judges_are_free():
    assert all(m.endswith(":free") for m in R.DEFAULT_JUDGES)


def test_tier3_excluded():
    assert "hazard_adjacent_category" in R.EXCLUDED_TIERS
