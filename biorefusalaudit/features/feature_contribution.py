"""Per-feature contribution to the residual direction — mini attribution graph.

Given the SAE-encoded representation of a residual at a specific token position,
this module computes each feature's signed contribution `z_i * ||W_dec[i]||` to
the residual's magnitude. The top-K features by contribution magnitude form the
nodes of a mini attribution graph; directed edges represent the feature's
decoder-direction cosine similarity to the residual (positive = this feature
points in the residual's direction; negative = against).

This is a lightweight proxy for Anthropic's circuit-tracing output. It doesn't
capture cross-layer interactions or cross-feature causal edges — those require
the full circuit-tracer library. But it gives us a reporting-ready per-prompt
attribution JSON that the dashboard can render as the "feature contribution
panel" the specialist review asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureContribution:
    feature_id: int
    activation: float  # z_i pre-decode
    decoder_magnitude: float  # ||W_dec[i]||
    contribution: float  # signed
    cosine_with_residual: float  # in [-1, 1]


def compute_contributions(
    residual: torch.Tensor,  # (d_model,) — one token's residual vector
    z: torch.Tensor,  # (d_sae,) — SAE-encoded representation at this token
    W_dec: torch.Tensor,  # (d_sae, d_model) — SAE decoder
    top_k: int = 10,
) -> list[FeatureContribution]:
    """Return top-K features by |contribution| with directional annotations."""
    assert residual.ndim == 1 and z.ndim == 1
    # Pre-compute decoder row magnitudes (constant across prompts for the same SAE)
    dec_mag = torch.linalg.vector_norm(W_dec, dim=-1)  # (d_sae,)
    # Signed contribution of feature i to residual magnitude:
    # z_i * ||W_dec[i]|| * cos(W_dec[i], residual) summed = ||residual|| approximately
    resid_unit = residual / (torch.linalg.vector_norm(residual) + 1e-8)
    # cos between each decoder row and the residual unit-vector
    W_unit = W_dec / (dec_mag.unsqueeze(-1) + 1e-8)  # (d_sae, d_model)
    cos = W_unit @ resid_unit  # (d_sae,)
    contribution = z * dec_mag * cos  # signed
    idx = torch.argsort(torch.abs(contribution), descending=True)[:top_k]
    out: list[FeatureContribution] = []
    for i in idx.tolist():
        out.append(FeatureContribution(
            feature_id=int(i),
            activation=float(z[i].item()),
            decoder_magnitude=float(dec_mag[i].item()),
            contribution=float(contribution[i].item()),
            cosine_with_residual=float(cos[i].item()),
        ))
    return out


def contributions_to_graph(contribs: Sequence[FeatureContribution]) -> dict:
    """Convert contribution list to an attribution-graph JSON (Anthropic-style schema)."""
    nodes = [
        {
            "id": f"feat_{c.feature_id}",
            "feature_id": c.feature_id,
            "activation": c.activation,
            "contribution": c.contribution,
        }
        for c in contribs
    ]
    edges = [
        {
            "source": f"feat_{c.feature_id}",
            "target": "residual",
            "weight": c.contribution,
            "cosine": c.cosine_with_residual,
        }
        for c in contribs
    ]
    return {"nodes": nodes + [{"id": "residual", "kind": "output"}], "edges": edges}
