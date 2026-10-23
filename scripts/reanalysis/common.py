"""Shared helpers for the 2026-10 re-analysis. CPU only; reads committed results."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SURFACE_CLASSES = ("refuse", "comply", "partial", "hedge", "evade")
TIERS = ("benign_bio", "dual_use_bio", "hazard_adjacent_category")


def load_records(run: str) -> list[dict]:
    return json.loads((RESULTS / run / "report.json").read_text(encoding="utf-8"))["records"]


def is_placeholder(r: dict) -> bool:
    s = r.get("surface_soft") or {}
    return r.get("surface_label") == "refuse" and len(s) == 5 and all(
        abs(float(v) - 0.2) < 1e-6 for v in s.values()
    )


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / sp)


def perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 100000, seed: int = 0) -> float:
    """One-sided permutation p for mean(b) > mean(a)."""
    rng = np.random.default_rng(seed)
    obs = b.mean() - a.mean()
    pooled = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(n_perm // 1000):
        perms = np.array([rng.permutation(pooled) for _ in range(1000)])
        diff = perms[:, na:].mean(axis=1) - perms[:, :na].mean(axis=1)
        cnt += int((diff >= obs - 1e-15).sum())
    return (cnt + 1) / (n_perm + 1)


def r2_groups(y: np.ndarray, groups: list) -> float:
    """Share of variance in y explained by group membership (between-group SS / total SS)."""
    y = np.asarray(y, dtype=float)
    tot = ((y - y.mean()) ** 2).sum()
    if tot <= 0:
        return float("nan")
    between = 0.0
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        between += len(idx) * (y[idx].mean() - y.mean()) ** 2
    return float(between / tot)


def r2_linear(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else float("nan")


def divergence(s: np.ndarray, f: np.ndarray, T: np.ndarray) -> float:
    e = T.T @ s
    if np.linalg.norm(e) < 1e-12:
        e = s.copy()
    fn = f / np.linalg.norm(f) if np.linalg.norm(f) > 1e-12 else f
    en = e / np.linalg.norm(e)
    return 1.0 - float(np.clip(fn @ en, -1, 1))


def candidate_Ts() -> dict[str, np.ndarray]:
    out = {"identity": np.eye(5)}
    for p in sorted((REPO / "configs").glob("calibration_*.yaml")):
        d = yaml.safe_load(p.read_text())
        for k, v in (d or {}).items():
            try:
                a = np.asarray(v, dtype=float)
            except (TypeError, ValueError):
                continue
            if a.shape == (5, 5):
                out[f"{p.stem}:{k}"] = a
    return out


def identify_T(records: list[dict]) -> tuple[str | None, float]:
    """Which stored T reproduces the stored D for every record (max abs error)."""
    best, best_err = None, float("inf")
    rows = [r for r in records if r.get("divergence") is not None and r.get("surface_soft")]
    for name, T in candidate_Ts().items():
        err = 0.0
        for r in rows:
            s = np.array([r["surface_soft"][c] for c in SURFACE_CLASSES])
            f = np.asarray(r["feature_vec"], dtype=float)
            err = max(err, abs(divergence(s, f, T) - r["divergence"]))
            if err > best_err:
                break
        if err < best_err:
            best, best_err = name, err
    return best, best_err
