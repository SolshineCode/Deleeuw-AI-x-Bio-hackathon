"""Stratified sampling over (tier, framing) grid."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

from biorefusalaudit.prompts.prompt_loader import DualUsePrompt


def stratified_sample(
    prompts: Sequence[DualUsePrompt],
    n_per_cell: int,
    seed: int | None = 42,
) -> list[DualUsePrompt]:
    """Return n_per_cell prompts sampled without replacement from each (tier, framing) cell.

    If a cell has fewer than n_per_cell prompts, all of them are returned.
    """
    rng = random.Random(seed)
    bins: dict[tuple[str, str], list[DualUsePrompt]] = defaultdict(list)
    for p in prompts:
        bins[(p.tier, p.framing)].append(p)
    out: list[DualUsePrompt] = []
    for key in sorted(bins.keys()):
        pool = bins[key]
        rng.shuffle(pool)
        out.extend(pool[:n_per_cell])
    return out


def marginal_counts(prompts: Sequence[DualUsePrompt]) -> dict:
    """Return {'tier': Counter, 'framing': Counter, 'cell': Counter} for diagnostics."""
    from collections import Counter

    tiers = Counter(p.tier for p in prompts)
    framings = Counter(p.framing for p in prompts)
    cells = Counter((p.tier, p.framing) for p in prompts)
    return {"tier": dict(tiers), "framing": dict(framings), "cell": dict(cells)}
