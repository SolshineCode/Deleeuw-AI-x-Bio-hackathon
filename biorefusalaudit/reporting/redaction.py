"""Redaction for tier-3 outputs before serialization.

Any per-prompt record whose prompt is tier-3 has its completion text replaced
by a category-label placeholder. This prevents accidental leakage of
hazardous content through model completions in publicly-shipped reports.
"""

from __future__ import annotations

REDACTED_PLACEHOLDER = "[REDACTED — tier-3 hazard_adjacent_category completion]"


def redact_tier3(record: dict) -> dict:
    """Return a copy of record with completion redacted if tier == hazard_adjacent_category."""
    r = dict(record)
    if r.get("tier") == "hazard_adjacent_category":
        if "completion" in r:
            r["completion"] = REDACTED_PLACEHOLDER
        if "prompt" in r:
            # Tier-3 public rows already carry category descriptors only,
            # but double-redact to be safe if a runner somehow wrote raw content.
            if len(r["prompt"]) > 400 or "Category:" not in r["prompt"]:
                r["prompt"] = REDACTED_PLACEHOLDER
    return r


def redact_records(records: list[dict]) -> list[dict]:
    return [redact_tier3(r) for r in records]
