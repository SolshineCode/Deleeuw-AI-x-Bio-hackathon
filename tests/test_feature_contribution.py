"""Tests for per-feature contribution calculation."""

import torch

from biorefusalaudit.features.feature_contribution import (
    compute_contributions,
    contributions_to_graph,
)


def test_top_k_shape_and_order():
    d_model = 8
    d_sae = 32
    torch.manual_seed(0)
    W_dec = torch.randn(d_sae, d_model)
    residual = torch.randn(d_model)
    z = torch.zeros(d_sae)
    # Set a few features with known activations
    z[0] = 1.0
    z[5] = 2.0
    z[15] = -1.5

    contribs = compute_contributions(residual, z, W_dec, top_k=5)
    assert len(contribs) == 5
    # Magnitude-descending order
    mags = [abs(c.contribution) for c in contribs]
    assert mags == sorted(mags, reverse=True)


def test_zero_z_gives_zero_contribution():
    d_model, d_sae = 4, 8
    W_dec = torch.randn(d_sae, d_model)
    residual = torch.randn(d_model)
    z = torch.zeros(d_sae)
    contribs = compute_contributions(residual, z, W_dec, top_k=3)
    for c in contribs:
        assert abs(c.contribution) < 1e-8


def test_graph_structure():
    d_model, d_sae = 4, 8
    W_dec = torch.randn(d_sae, d_model)
    residual = torch.randn(d_model)
    z = torch.randn(d_sae)
    contribs = compute_contributions(residual, z, W_dec, top_k=3)
    graph = contributions_to_graph(contribs)
    assert "nodes" in graph
    assert "edges" in graph
    # One residual node + top_k feature nodes
    assert len(graph["nodes"]) == 4
    assert len(graph["edges"]) == 3
    for edge in graph["edges"]:
        assert edge["target"] == "residual"
