"""Sanity-validate a feature catalog JSON before use.

Checks:
  - All required categories present
  - All feature indices are unique across categories (no double-counting)
  - Minimum count per category per PRD thresholds
  - Indices within [0, d_sae) for the given d_sae
  - Optional: positive-vs-negative activation check given prompt activations
    (compare mean activation on bio vs neutral prompts; flag features that
    don't differentiate meaningfully)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from biorefusalaudit.features.feature_profiler import FeatureCatalog
from biorefusalaudit.scoring.divergence import FEATURE_CATEGORIES

# PRD minimums per category (Phase 1 "Target: 30-60 validated features per model").
_MIN_PER_CATEGORY = {
    "bio_content": 20,
    "hazard_adjacent": 10,
    "refusal_circuitry": 10,
    "hedging": 5,
    "deception_correlate": 5,
}


@dataclass
class ValidationReport:
    catalog_path: str
    problems: list[str]
    warnings: list[str]
    stats: dict

    def is_clean(self) -> bool:
        return not self.problems


def validate_catalog(catalog: FeatureCatalog, d_sae: int) -> ValidationReport:
    problems: list[str] = []
    warnings: list[str] = []
    stats: dict = {"category_counts": {}, "total_unique_features": 0}

    # Structural checks
    missing = [c for c in FEATURE_CATEGORIES if c not in catalog.categories]
    if missing:
        problems.append(f"missing categories: {missing}")

    seen: set[int] = set()
    dupes: set[int] = set()
    for cat in FEATURE_CATEGORIES:
        ids = catalog.categories.get(cat, [])
        stats["category_counts"][cat] = len(ids)
        for i in ids:
            if not isinstance(i, int):
                problems.append(f"{cat}: non-int feature id {i!r}")
                continue
            if i < 0 or i >= d_sae:
                problems.append(f"{cat}: feature id {i} out of [0, {d_sae})")
            if i in seen:
                dupes.add(i)
            seen.add(i)

        min_req = _MIN_PER_CATEGORY.get(cat, 1)
        if len(ids) < min_req:
            warnings.append(
                f"{cat}: {len(ids)} features < PRD minimum {min_req} "
                f"(acceptable for v0.1 stub; re-validate on Neuronpedia before publication)"
            )

    stats["total_unique_features"] = len(seen)
    if dupes:
        warnings.append(f"{len(dupes)} features appear in multiple categories: {sorted(dupes)[:20]}")

    return ValidationReport(
        catalog_path=str(getattr(catalog, "_path", "unknown")),
        problems=problems,
        warnings=warnings,
        stats=stats,
    )


def differentiation_check(
    catalog: FeatureCatalog,
    bio_activations: np.ndarray,
    neutral_activations: np.ndarray,
    min_effect_size: float = 0.1,
) -> dict[str, list[int]]:
    """For each category, return the subset of features that meaningfully differ
    between bio-content prompts and neutral prompts.

    bio_activations: (N_bio, d_sae). neutral_activations: (N_neutral, d_sae).
    Returns a filtered catalog (dict of category -> list of feature ids) keeping
    only features with |mean(bio) - mean(neutral)| > min_effect_size * pooled_std.
    """
    bio = np.asarray(bio_activations)
    neu = np.asarray(neutral_activations)
    assert bio.shape[1] == neu.shape[1], "d_sae mismatch between activation sets"

    mean_bio = bio.mean(axis=0)
    mean_neu = neu.mean(axis=0)
    pooled_std = np.sqrt((bio.var(axis=0) + neu.var(axis=0)) / 2 + 1e-8)
    effect = np.abs(mean_bio - mean_neu) / pooled_std

    filtered: dict[str, list[int]] = {}
    for cat, ids in catalog.categories.items():
        filtered[cat] = [i for i in ids if i < effect.size and effect[i] > min_effect_size]
    return filtered
