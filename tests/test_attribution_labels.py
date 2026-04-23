"""Tests for the attribution evidence-tier classifier."""

from biorefusalaudit.features.attribution_labels import (
    FeatureEvidence,
    classify_tier,
    evidence_summary,
)


def test_candidate_tier_when_nothing_clears():
    ev = FeatureEvidence(feature_id=1, category="bio_content")
    assert classify_tier(ev) == "candidate"


def test_activation_only_tier():
    ev = FeatureEvidence(
        feature_id=2, category="bio_content",
        activation_effect_size=0.5,
    )
    assert classify_tier(ev) == "activation_only"


def test_attribution_only_tier():
    ev = FeatureEvidence(
        feature_id=3, category="hazard_adjacent",
        attribution_weight=0.3,
    )
    assert classify_tier(ev) == "attribution_only"


def test_named_circuit_requires_all_three_legs():
    ev = FeatureEvidence(
        feature_id=4, category="refusal_circuitry",
        activation_effect_size=0.5,
        attribution_weight=0.3,
        perturbation_divergence_delta=0.3,
    )
    assert classify_tier(ev) == "named_circuit"


def test_label_change_alone_qualifies_as_named_circuit():
    ev = FeatureEvidence(
        feature_id=5, category="refusal_circuitry",
        perturbation_label_changed=True,
    )
    assert classify_tier(ev) == "named_circuit"


def test_evidence_summary_aggregates():
    evs = [
        FeatureEvidence(feature_id=1, category="bio_content", tier="candidate"),
        FeatureEvidence(feature_id=2, category="bio_content",
                        activation_effect_size=0.5, tier="activation_only"),
        FeatureEvidence(feature_id=3, category="refusal_circuitry",
                        activation_effect_size=0.5, attribution_weight=0.3,
                        perturbation_divergence_delta=0.3, tier="named_circuit"),
    ]
    summary = evidence_summary(evs)
    assert summary["tier_counts"]["candidate"] == 1
    assert summary["tier_counts"]["activation_only"] == 1
    assert summary["tier_counts"]["named_circuit"] == 1
    assert len(summary["named_circuits"]) == 1
