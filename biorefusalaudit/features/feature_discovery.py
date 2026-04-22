"""Neuronpedia-based feature discovery.

Queries the Neuronpedia API with category keywords and returns candidate
feature indices per category. Hand-validation is a separate step (not
automated) — use feature_validator.differentiation_check to filter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import requests
import yaml

NEURONPEDIA_BASE = "https://www.neuronpedia.org"


def load_keyword_config(path: str | Path) -> dict[str, list[str]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return cfg.get("categories", {})


def query_features_by_keyword(
    keyword: str,
    model_id: str,
    sae_id: str,
    top_k: int = 20,
    session: requests.Session | None = None,
) -> list[dict]:
    """Hit Neuronpedia's search endpoint for a single keyword.

    Returns a list of dicts with at minimum {feature_index, score, example}.
    If the endpoint is down or rate-limited, returns empty list.
    """
    if session is None:
        session = requests.Session()
    url = f"{NEURONPEDIA_BASE}/api/search-all"
    params = {"modelId": model_id, "sourceId": sae_id, "text": keyword}
    try:
        r = session.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        payload = r.json()
    except Exception:
        return []
    # Response schema varies; extract feature indices defensively.
    results: list[dict] = []
    for item in payload.get("results", [])[:top_k]:
        idx = item.get("index") or item.get("feature_index") or item.get("id")
        if idx is not None:
            results.append(
                {
                    "feature_index": int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else None,
                    "score": item.get("score", 0.0),
                    "example": (item.get("text") or item.get("example") or "")[:200],
                }
            )
    return results


def discover_candidates(
    keyword_config: Mapping[str, list[str]],
    model_id: str,
    sae_id: str,
    top_k_per_keyword: int = 10,
) -> dict[str, list[int]]:
    """Run Neuronpedia queries for every keyword in every category.

    Returns {category: [candidate_feature_indices]} with duplicates deduplicated.
    """
    session = requests.Session()
    out: dict[str, list[int]] = {}
    for cat, keywords in keyword_config.items():
        seen: set[int] = set()
        for kw in keywords:
            results = query_features_by_keyword(kw, model_id, sae_id, top_k=top_k_per_keyword, session=session)
            for r in results:
                idx = r.get("feature_index")
                if idx is not None:
                    seen.add(int(idx))
        out[cat] = sorted(seen)
    return out


def write_candidate_catalog(
    candidates: dict[str, list[int]],
    out_path: str | Path,
    model_name: str,
    sae_source: str,
) -> None:
    Path(out_path).write_text(
        json.dumps(
            {
                "model_name": model_name,
                "sae_source": sae_source,
                "catalog_version": "discovered-candidates",
                "catalog_note": "Neuronpedia candidates; NOT hand-validated. Pipe through feature_validator.differentiation_check + manual review before use.",
                "categories": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
