"""Unit tests for the divergence metric."""

import numpy as np
import pytest

from biorefusalaudit.scoring.divergence import (
    FEATURE_CATEGORIES,
    SURFACE_CLASSES,
    divergence,
    divergence_flags,
    refusal_depth,
)


def test_divergence_aligned_vectors_near_zero():
    s = np.array([1.0, 0, 0, 0, 0])  # pure refuse
    f = np.array([0.0, 0.5, 0.5, 0.0, 0.0])
    # With T = identity, T^T @ s = s = [1,0,0,0,0], which is bio_content-aligned.
    # f is hazard_adjacent + refusal_circuitry. Divergence should be high (≈1).
    d = divergence(s, f)
    assert 0.9 < d < 1.1

    # Align s and f perfectly (both in the same coordinate direction).
    s2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    f2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    d2 = divergence(s2, f2)
    assert d2 < 1e-6


def test_divergence_orthogonal_vectors_near_one():
    s = np.array([1.0, 0, 0, 0, 0])
    f = np.array([0, 0, 0, 0, 1.0])
    d = divergence(s, f)
    assert 0.99 <= d <= 1.01


def test_divergence_antipodal_vectors_near_two():
    # Cannot construct perfectly antipodal with nonneg inputs; so we flip sign
    # of one of them to verify the (0,2) range covers the negative-cosine case.
    s = np.array([1.0, 0, 0, 0, 0])
    f = np.array([-1.0, 0, 0, 0, 0])
    d = divergence(s, f)
    assert d > 1.9


def test_divergence_rejects_bad_shape():
    with pytest.raises(ValueError):
        divergence(np.ones(4), np.ones(5))
    with pytest.raises(ValueError):
        divergence(np.ones(5), np.ones(5), T=np.ones((4, 5)))


def test_divergence_accepts_identity_T_default():
    s = np.ones(5) / 5
    f = np.ones(5) / 5
    d = divergence(s, f)
    assert d < 1e-6


def test_refusal_depth_only_when_refuse_dominates():
    s_refuse = np.array([1.0, 0, 0, 0, 0])
    f = np.array([1.0, 0, 0, 0, 0])
    depth = refusal_depth(s_refuse, f)
    assert depth > 0.99

    s_comply = np.array([0, 1.0, 0, 0, 0])
    with pytest.raises(ValueError):
        refusal_depth(s_comply, f)


def test_divergence_flags_hazard_on_refuse():
    s_refuse = np.array([1.0, 0, 0, 0, 0])
    f_high_hazard = np.array([0.1, 0.8, 0.05, 0.025, 0.025])
    flags = divergence_flags(s_refuse, f_high_hazard)
    assert flags["hazard_features_active_despite_refusal"] is True
    assert flags["refusal_features_active_despite_compliance"] is False


def test_divergence_flags_refusal_on_comply():
    s_comply = np.array([0, 1.0, 0, 0, 0])
    f_refusal_active = np.array([0.05, 0.05, 0.8, 0.05, 0.05])
    flags = divergence_flags(s_comply, f_refusal_active)
    assert flags["refusal_features_active_despite_compliance"] is True


def test_surface_and_feature_ordering_stable():
    assert SURFACE_CLASSES == ("refuse", "comply", "partial", "hedge", "evade")
    assert FEATURE_CATEGORIES == (
        "bio_content",
        "hazard_adjacent",
        "refusal_circuitry",
        "hedging",
        "deception_correlate",
    )
