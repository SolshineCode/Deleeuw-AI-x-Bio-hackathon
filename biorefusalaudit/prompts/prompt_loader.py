"""DualUsePrompt dataclass + JSONL loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

VALID_TIERS = {
    "benign_bio", "dual_use_bio", "hazard_adjacent_category",
    "benign_legal", "dual_use_legal", "hazard_adjacent_legal",
}
VALID_FRAMINGS = {"direct", "educational", "roleplay", "obfuscated"}
VALID_EXPECTED_SURFACE = {
    "comply", "refuse", "hedge", "partial", "evade",
    "comply_or_hedge", "refuse_or_hedge",
    # Note: bio tiers use original naming (hazard_adjacent_category), legal tiers use *_legal suffix.
    # refuse_or_hedge is canonical; hedge_or_refuse is a legacy alias — normalize in JSONL files.
}


@dataclass(frozen=True)
class DualUsePrompt:
    id: str
    tier: str
    framing: str
    prompt: str
    rationale: str
    expected_surface: str
    category: str
    sampled_from: str = "original"
    domain: str = ""

    def __post_init__(self):
        if self.tier not in VALID_TIERS:
            raise ValueError(f"tier {self.tier!r} not in {sorted(VALID_TIERS)}")
        if self.framing not in VALID_FRAMINGS:
            raise ValueError(f"framing {self.framing!r} not in {sorted(VALID_FRAMINGS)}")
        if self.expected_surface not in VALID_EXPECTED_SURFACE:
            raise ValueError(
                f"expected_surface {self.expected_surface!r} not in {sorted(VALID_EXPECTED_SURFACE)}"
            )
        if not self.prompt.strip():
            raise ValueError(f"prompt must be non-empty (id={self.id})")


def load_jsonl(path: str | Path) -> list[DualUsePrompt]:
    """Load prompts from a JSONL file. Raises on any malformed or invalid row."""
    p = Path(path)
    prompts: list[DualUsePrompt] = []
    with p.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{i}: invalid JSON: {e}") from e
            try:
                prompts.append(DualUsePrompt(**obj))
            except TypeError as e:
                raise ValueError(f"{p}:{i}: schema mismatch: {e}") from e
    return prompts


def iter_jsonl(path: str | Path) -> Iterator[DualUsePrompt]:
    """Iterator variant for large files."""
    for p in load_jsonl(path):
        yield p
