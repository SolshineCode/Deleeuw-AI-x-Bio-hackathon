"""Attribution labels for per-feature evidence tiers (per specialist review).

A feature crosses from "catalog entry" to "named circuit" only when all three
evidence sources align: activation (it fires on the relevant prompts), attribution
(the attribution graph pins it to the decisive token span), and perturbation
(ablating / boosting it changes the downstream behavior).

This module defines the evidence-tier schema + labeling utilities; the actual
attribution-graph integration lives in `runner/trace_selected_cases.py` and
calls out to the intervention script for the perturbation leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

EvidenceTier = Literal["candidate", "activation_only", "attribution_only", "named_circuit"]


@dataclass
class FeatureEvidence:
    feature_id: int
    category: str
    # Activation leg
    activation_effect_size: float = 0.0  # Cohen's-d vs neutral
    activation_n_samples: int = 0
    # Attribution leg
    attribution_weight: float = 0.0  # edge weight from the attribution graph
    attribution_span: str = ""  # token span this feature attributes to
    # Perturbation leg
    perturbation_label_changed: bool = False
    perturbation_divergence_delta: float = 0.0
    # Derived
    tier: EvidenceTier = "candidate"
    notes: str = ""


def classify_tier(
    ev: FeatureEvidence,
    tau_activation: float = 0.2,
    tau_attribution: float = 0.1,
    tau_perturbation: float = 0.2,
) -> EvidenceTier:
    """Derive the evidence tier from the three legs.

    - `candidate`: no leg above threshold
    - `activation_only`: activation leg clears tau but attribution / perturbation don't
    - `attribution_only`: attribution leg clears tau but activation / perturbation don't
    - `named_circuit`: all three legs clear their thresholds, OR perturbation
      changed the surface label outright
    """
    has_act = ev.activation_effect_size >= tau_activation
    has_attr = ev.attribution_weight >= tau_attribution
    has_perturb = (
        ev.perturbation_label_changed
        or abs(ev.perturbation_divergence_delta) >= tau_perturbation
    )
    if has_act and has_attr and has_perturb:
        return "named_circuit"
    if ev.perturbation_label_changed:
        # Surface-label flip is strong enough on its own to graduate.
        return "named_circuit"
    if has_act and has_attr:
        return "activation_only"  # attribution alone doesn't license causal claim
    if has_act:
        return "activation_only"
    if has_attr:
        return "attribution_only"
    return "candidate"


def set_tier(ev: FeatureEvidence) -> None:
    ev.tier = classify_tier(ev)


def evidence_summary(evs: list[FeatureEvidence]) -> dict:
    tier_counts: dict[str, int] = {
        "candidate": 0,
        "activation_only": 0,
        "attribution_only": 0,
        "named_circuit": 0,
    }
    for ev in evs:
        tier_counts[ev.tier] = tier_counts.get(ev.tier, 0) + 1
    named = [asdict(ev) for ev in evs if ev.tier == "named_circuit"]
    return {
        "tier_counts": tier_counts,
        "named_circuits": named,
    }
