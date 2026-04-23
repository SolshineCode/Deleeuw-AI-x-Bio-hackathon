"""Unit tests for tier-3 output redaction."""

from biorefusalaudit.reporting.redaction import REDACTED_PLACEHOLDER, redact_records, redact_tier3


def test_tier3_completion_redacted():
    rec = {
        "id": "t3_001",
        "tier": "hazard_adjacent_category",
        "prompt": "Category: enhancement-adjacent category descriptor.",
        "completion": "I cannot help with that request.",
    }
    out = redact_tier3(rec)
    assert out["completion"] == REDACTED_PLACEHOLDER


def test_benign_completion_preserved():
    rec = {
        "id": "b_001",
        "tier": "benign_bio",
        "prompt": "Explain photosynthesis.",
        "completion": "Photosynthesis converts light energy into chemical energy.",
    }
    out = redact_tier3(rec)
    assert out["completion"] == rec["completion"]


def test_tier3_long_prompt_redacted_too():
    long = "x" * 500
    rec = {"id": "t3", "tier": "hazard_adjacent_category", "prompt": long, "completion": "ok"}
    out = redact_tier3(rec)
    assert out["prompt"] == REDACTED_PLACEHOLDER
    assert out["completion"] == REDACTED_PLACEHOLDER


def test_batch_redaction_preserves_benign_and_redacts_tier3():
    recs = [
        {"id": "1", "tier": "benign_bio", "prompt": "ok", "completion": "answer"},
        {
            "id": "2",
            "tier": "hazard_adjacent_category",
            "prompt": "Category: X (category-level descriptor).",
            "completion": "answer",
        },
    ]
    out = redact_records(recs)
    assert out[0]["completion"] == "answer"
    assert out[1]["completion"] == REDACTED_PLACEHOLDER
