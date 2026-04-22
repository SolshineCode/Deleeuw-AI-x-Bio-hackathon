"""Pre-publication safety hygiene checks on eval-set files.

Primary check: tier-3 rows in the public file must be category-level
descriptors, not hazardous prompt bodies. If a public JSONL carries
hazardous content, block the file before commit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from biorefusalaudit.prompts.prompt_loader import DualUsePrompt, load_jsonl

# Category-descriptor markers that are expected in tier-3 public rows.
# Tier-3 public prompts must read like "Category: <name>. <framing scaffold>."
# rather than a concrete how-to question.
_CATEGORY_MARKERS = (
    "category:",
    "category-level",
    "(category-level descriptor)",
)

# A small-is-fine signal: tier-3 public rows should be short descriptor strings.
# Anything above this length is suspicious and gets flagged.
_MAX_TIER3_PUBLIC_LEN = 400

# Conservative regex for prompts that look like actionable how-to questions.
# Catches imperative-mood verbs paired with method/protocol/procedure nouns.
_HOWTO_PATTERN = re.compile(
    r"\b(how to|give me the steps|write a protocol|provide a procedure|"
    r"describe the method for|list the reagents)\b",
    re.IGNORECASE,
)


def check_no_hazard_bodies(path: str | Path) -> list[str]:
    """Return a list of problem descriptions; empty list means the file is clean.

    Check is non-destructive (never modifies the file). Callers decide whether
    to fail the build, redact, or manually review.
    """
    problems: list[str] = []
    prompts = load_jsonl(path)
    for p in prompts:
        if p.tier != "hazard_adjacent_category":
            continue
        body = p.prompt
        lowered = body.lower()
        has_marker = any(m in lowered for m in _CATEGORY_MARKERS)
        if not has_marker:
            problems.append(
                f"{p.id}: tier-3 public prompt missing category marker "
                f"(expected one of {_CATEGORY_MARKERS}); content starts with: {body[:60]!r}"
            )
        if len(body) > _MAX_TIER3_PUBLIC_LEN:
            problems.append(
                f"{p.id}: tier-3 public prompt too long ({len(body)} chars > {_MAX_TIER3_PUBLIC_LEN})"
            )
        if _HOWTO_PATTERN.search(body):
            problems.append(
                f"{p.id}: tier-3 public prompt contains actionable how-to phrasing "
                f"(likely hazardous body leaked into public file)"
            )
    return problems


def assert_clean(path: str | Path) -> None:
    """Raise if the file has any tier-3 hygiene problems."""
    problems = check_no_hazard_bodies(path)
    if problems:
        raise AssertionError(
            "Safety review failed for {}:\n  - {}".format(path, "\n  - ".join(problems))
        )
