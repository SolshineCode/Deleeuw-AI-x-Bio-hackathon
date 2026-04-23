"""Fit the 5x5 alignment matrix T on held-out positive controls.

Given paired (s_i, f_i) observations from prompts where the surface behavior
and internal activation are known to agree (by human validation or ground-
truth construction), T^T maps s to the *expected* internal activation.

We solve a ridge-regularized least squares:
    T^T = argmin_W sum_i ||W s_i - f_i||^2 + lambda ||W||_F^2

which has closed form W = F S^T (S S^T + lambda I)^{-1} where S is the 5xN
stack of surface vectors and F is the 5xN stack of feature vectors.

A condition-number check on S S^T + lambda I guards against ill-conditioned
calibration sets (too few positive controls per surface class).
"""

from __future__ import annotations

import numpy as np

from biorefusalaudit.scoring.divergence import (
    FEATURE_CATEGORIES,
    SURFACE_CLASSES,
)

_DEFAULT_LAMBDA = 0.1
_MAX_CONDITION = 1.0e6


def fit_alignment_matrix(
    surface_vecs: np.ndarray,
    feature_vecs: np.ndarray,
    reg_lambda: float = _DEFAULT_LAMBDA,
    max_condition: float = _MAX_CONDITION,
) -> tuple[np.ndarray, dict]:
    """Fit T such that T^T @ s approximates f on the training pairs.

    Args:
        surface_vecs: array of shape (N, 5) with row = surface distribution.
        feature_vecs: array of shape (N, 5) with row = normalized feature vector.
        reg_lambda: L2 regularization strength.
        max_condition: raise if the regularized Gram matrix condition > this.

    Returns:
        (T, diagnostics) where T has shape (5, 5) and diagnostics is a dict
        with keys: residual_mse, condition_number, n_samples.
    """
    s = np.asarray(surface_vecs, dtype=np.float64)
    f = np.asarray(feature_vecs, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 5:
        raise ValueError(f"surface_vecs must be (N, 5), got {s.shape}")
    if f.ndim != 2 or f.shape[1] != 5:
        raise ValueError(f"feature_vecs must be (N, 5), got {f.shape}")
    if s.shape[0] != f.shape[0]:
        raise ValueError(
            f"row count mismatch: {s.shape[0]} surface vs {f.shape[0]} feature"
        )

    S = s.T  # (5, N)
    F = f.T  # (5, N)
    gram = S @ S.T + reg_lambda * np.eye(5)
    cond = float(np.linalg.cond(gram))
    if cond > max_condition:
        raise ValueError(
            f"Calibration Gram matrix is ill-conditioned (cond={cond:.2e} > {max_condition:.2e}). "
            "Likely cause: too few positive-control prompts per surface class. "
            "Fix: add more calibration samples before fitting T."
        )

    W = F @ S.T @ np.linalg.inv(gram)  # (5, 5)
    T = W.T

    predicted = (T.T @ S).T  # (N, 5)
    residual = predicted - f
    mse = float(np.mean(residual ** 2))
    return T, {
        "residual_mse": mse,
        "condition_number": cond,
        "n_samples": int(s.shape[0]),
        "surface_classes": list(SURFACE_CLASSES),
        "feature_categories": list(FEATURE_CATEGORIES),
    }
