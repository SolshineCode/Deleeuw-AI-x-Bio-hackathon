"""Unit tests for the alignment-matrix calibration."""

import numpy as np
import pytest

from biorefusalaudit.scoring.calibration import fit_alignment_matrix


def test_identity_recovery_on_aligned_pairs():
    rng = np.random.default_rng(0)
    n = 50
    s = rng.dirichlet(np.ones(5), size=n)  # 50 simplex-distributed surfaces
    f = s.copy()  # perfectly aligned
    T, diag = fit_alignment_matrix(s, f, reg_lambda=0.01)
    # T should be near identity since f = s.
    assert np.allclose(T, np.eye(5), atol=0.2)
    assert diag["n_samples"] == 50
    assert diag["residual_mse"] < 0.05


def test_fit_reduces_residual_vs_zero_T():
    rng = np.random.default_rng(1)
    n = 40
    s = rng.dirichlet(np.ones(5), size=n)
    # Construct a non-trivial true mapping: permutation.
    perm = np.array([1, 2, 3, 4, 0])
    f = s[:, perm] + 0.01 * rng.standard_normal((n, 5))
    T, diag = fit_alignment_matrix(s, f)
    # Residual should be small — we recovered the permutation within noise.
    assert diag["residual_mse"] < 0.01


def test_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        fit_alignment_matrix(np.zeros((10, 5)), np.zeros((9, 5)))
    with pytest.raises(ValueError):
        fit_alignment_matrix(np.zeros((10, 4)), np.zeros((10, 5)))


def test_raises_on_ill_conditioned_data():
    # All surface vectors identical -> S S^T is rank 1; condition will be huge
    # even after regularization if we turn regularization off.
    s = np.tile([1.0, 0, 0, 0, 0], (20, 1))
    f = np.random.default_rng(2).standard_normal((20, 5))
    with pytest.raises(ValueError):
        fit_alignment_matrix(s, f, reg_lambda=1e-10, max_condition=1e3)
