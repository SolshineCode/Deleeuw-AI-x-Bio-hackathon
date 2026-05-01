"""Top-k feature extraction + categorization via hand-validated feature catalog.

Given a per-prompt SAE feature activation vector (d_sae,), returns:
- top-k feature indices (the strongest-firing features)
- a 5-category normalized vector f ∈ R^5 over FEATURE_CATEGORIES, computed
  by summing activations within each category (as defined by the catalog)
  and L1-normalizing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biorefusalaudit.scoring.divergence import FEATURE_CATEGORIES


@dataclass
class FeatureCatalog:
    model_name: str
    sae_source: str
    # Map category name → list of feature indices
    categories: dict[str, list[int]]

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCatalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_name=data["model_name"],
            sae_source=data["sae_source"],
            categories={k: list(v) for k, v in data["categories"].items()},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "sae_source": self.sae_source,
                    "categories": self.categories,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def top_k_features(activations: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
    """Return list of (feature_idx, activation_value) sorted by magnitude desc."""
    flat = np.asarray(activations).reshape(-1)
    order = np.argsort(-np.abs(flat))[:k]
    return [(int(i), float(flat[i])) for i in order]


def categorize(
    activations: np.ndarray,
    catalog: FeatureCatalog,
    projection_W: np.ndarray | None = None,
) -> np.ndarray:
    """Project d_sae activations onto the 5-category feature vector.

    If projection_W (shape 5 × d_sae) is provided, uses the learned Track-B
    projection: f = L1_norm(relu(W @ acts)). Otherwise falls back to the
    hand-coded catalog index selection.

    Returns a (5,) np.ndarray, L1-normalized, in FEATURE_CATEGORIES order.
    """
    flat = np.asarray(activations, dtype=np.float64).reshape(-1)
    if projection_W is not None:
        W = np.asarray(projection_W, dtype=np.float64)
        vec = W @ flat  # (5,)
        vec = np.maximum(vec, 0.0)  # relu — activations should be non-negative
        total = vec.sum()
        if total > 0:
            vec = vec / total
        return vec

    # Catalog index selection (original path)
    vec = np.zeros(5, dtype=np.float64)
    for i, cat in enumerate(FEATURE_CATEGORIES):
        idxs = catalog.categories.get(cat, [])
        if not idxs:
            continue
        idxs_arr = np.asarray([j for j in idxs if 0 <= j < flat.size], dtype=np.int64)
        if idxs_arr.size == 0:
            continue
        vec[i] = float(np.abs(flat[idxs_arr]).sum())
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec
