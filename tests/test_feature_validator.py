"""Unit tests for the feature catalog validator."""

import json
from pathlib import Path

from biorefusalaudit.features.feature_profiler import FeatureCatalog
from biorefusalaudit.features.feature_validator import validate_catalog


def test_valid_catalog_passes():
    cat = FeatureCatalog(
        model_name="test",
        sae_source="test_sae",
        categories={
            "bio_content": list(range(20)),
            "hazard_adjacent": list(range(20, 30)),
            "refusal_circuitry": list(range(30, 40)),
            "hedging": list(range(40, 45)),
            "deception_correlate": list(range(45, 50)),
        },
    )
    report = validate_catalog(cat, d_sae=16384)
    assert report.problems == []
    assert report.stats["total_unique_features"] == 50


def test_out_of_bounds_feature_problem():
    cat = FeatureCatalog(
        model_name="test",
        sae_source="test_sae",
        categories={
            "bio_content": [100000],  # out of range
            "hazard_adjacent": [],
            "refusal_circuitry": [],
            "hedging": [],
            "deception_correlate": [],
        },
    )
    report = validate_catalog(cat, d_sae=16384)
    assert any("out of [0, 16384)" in p for p in report.problems)


def test_missing_category_problem():
    cat = FeatureCatalog(model_name="t", sae_source="s", categories={"bio_content": [1, 2]})
    report = validate_catalog(cat, d_sae=16384)
    assert any("missing categories" in p for p in report.problems)


def test_ships_catalog_passes_structural(tmp_path):
    """The shipped feature catalogs must structurally validate."""
    shipped = Path(__file__).parent.parent / "data" / "feature_catalog" / "gemma-2-2b-it.json"
    if not shipped.exists():
        return  # catalog not yet staged
    cat = FeatureCatalog.load(shipped)
    report = validate_catalog(cat, d_sae=16384)
    assert report.problems == [], f"shipped catalog has problems: {report.problems}"
