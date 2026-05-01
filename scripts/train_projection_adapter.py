"""Track B projection adapter — learns W ∈ ℝ^{5 × d_sae}.

Freezes the Gemma Scope SAE entirely and learns a small projection matrix W
that maps SAE activation vectors to the 5 bio-safety feature categories,
replacing the hand-coded catalog index selection.

Training objective:
    L = L_contrastive + λ_recon * L_recon + λ_l2 * ||W||_F²

    L_contrastive: margin ranking loss on (hazard, benign) pairs:
        max(0, margin - D_w(hazard) + D_w(benign))
        where D_w computes divergence using f = L1_norm(relu(W @ acts))
        Goal: hazard-tier prompts should have higher D than benign-tier.

    L_recon: MSE between W-projected f and catalog-computed f (soft supervision).
        Regularizes W toward the existing catalog prior.

Outputs:
    configs/projection_adapter_gemma2_2b.pt   -- W state_dict
    configs/projection_adapter_gemma2_2b.yaml -- hyperparams + fit_date + val_loss

Usage — initial training on 75-prompt corpus:
    python scripts/train_projection_adapter.py \\
        --activations runs/gemma-2-2b-it-L12-activations/activations.npz \\
        --report runs/gemma-2-2b-it-L12-activations/report.json \\
        --calibration configs/calibration_gemma2_2b.yaml \\
        --out-pt configs/projection_adapter_gemma2_2b.pt \\
        --out-yaml configs/projection_adapter_gemma2_2b.yaml

    # After Phase 1 explicit-prompt evals complete:
    python scripts/train_projection_adapter.py \\
        --activations \\
            runs/gemma-2-2b-it-L12-activations/activations.npz \\
            runs/gemma-2-2b-it-explicit-tier3-qwen3-100/activations.npz \\
            runs/gemma-2-2b-it-explicit-benign-qwen3-100/activations.npz \\
            runs/gemma-2-2b-it-explicit-dualuse-qwen3-100/activations.npz \\
        --report \\
            runs/gemma-2-2b-it-L12-activations/report.json \\
            runs/gemma-2-2b-it-explicit-tier3-qwen3-100/report.json \\
            runs/gemma-2-2b-it-explicit-benign-qwen3-100/report.json \\
            runs/gemma-2-2b-it-explicit-dualuse-qwen3-100/report.json \\
        --calibration configs/calibration_gemma2_2b.yaml \\
        --out-pt configs/projection_adapter_gemma2_2b.pt \\
        --out-yaml configs/projection_adapter_gemma2_2b.yaml

Changelog:
    2026-04-30: Initial implementation (Track B projection adapter)
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

FEATURE_CATEGORIES = (
    "bio_content",
    "hazard_adjacent",
    "refusal_circuitry",
    "hedging",
    "deception_correlate",
)
SURFACE_CLASSES = ("refuse", "comply", "partial", "hedge", "evade")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_activations_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "ids": [str(x) for x in d["ids"]],
        "tiers": [str(x) for x in d["tiers"]],
        "framings": [str(x) for x in d["framings"]],
        "feature_acts": d["feature_acts"].astype(np.float32),
    }


def load_report_records(path: str) -> dict[str, dict]:
    """Return {id: record} dict from report.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["records"]}


def build_dataset(
    activations_paths: list[str],
    report_paths: list[str],
    calibration_path: str,
) -> tuple[dict, np.ndarray]:
    """Merge activations + report records into a single dataset dict.

    Returns (dataset, T_matrix).
    dataset keys: ids, tiers, framings, feature_acts (n × d_sae),
                  surface_soft (n × 5), feature_vec (n × 5).
    """
    all_ids, all_tiers, all_framings, all_acts = [], [], [], []
    all_surf, all_fvec = [], []

    report_by_id: dict[str, dict] = {}
    for rp in report_paths:
        if rp and Path(rp).exists():
            report_by_id.update(load_report_records(rp))

    for ap in activations_paths:
        d = load_activations_npz(ap)
        for idx, pid in enumerate(d["ids"]):
            acts = d["feature_acts"][idx]
            tier = d["tiers"][idx]
            framing = d["framings"][idx]
            all_ids.append(pid)
            all_tiers.append(tier)
            all_framings.append(framing)
            all_acts.append(acts)

            if pid in report_by_id:
                rec = report_by_id[pid]
                ss = rec.get("surface_soft", {})
                surf = np.array([ss.get(c, 0.0) for c in SURFACE_CLASSES], dtype=np.float32)
                fvec_raw = rec.get("feature_vec")
                if fvec_raw is not None and len(fvec_raw) == 5:
                    fvec = np.array(fvec_raw, dtype=np.float32)
                else:
                    fvec = np.zeros(5, dtype=np.float32)
            else:
                # No report record: use tier-based prior for surface_soft
                surf = _tier_prior_surface(tier)
                fvec = np.zeros(5, dtype=np.float32)
            all_surf.append(surf)
            all_fvec.append(fvec)

    cfg = yaml.safe_load(Path(calibration_path).read_text(encoding="utf-8"))
    T = np.array(cfg["T"], dtype=np.float32)

    return {
        "ids": all_ids,
        "tiers": np.array(all_tiers),
        "framings": np.array(all_framings),
        "feature_acts": np.stack(all_acts, axis=0),  # (n, d_sae)
        "surface_soft": np.stack(all_surf, axis=0),  # (n, 5)
        "feature_vec": np.stack(all_fvec, axis=0),   # (n, 5)
    }, T


def _tier_prior_surface(tier: str) -> np.ndarray:
    """Fallback surface_soft prior when no report record is available."""
    priors = {
        "hazard_adjacent_category": [0.70, 0.05, 0.10, 0.10, 0.05],  # mostly refuse
        "dual_use_bio": [0.30, 0.25, 0.25, 0.15, 0.05],
        "benign_bio": [0.05, 0.60, 0.10, 0.20, 0.05],  # mostly comply
    }
    p = priors.get(tier, [0.2, 0.2, 0.2, 0.2, 0.2])
    return np.array(p, dtype=np.float32)


def stratified_split(
    n: int, tiers: np.ndarray, val_frac: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, val_indices) with per-tier stratification."""
    rng = np.random.default_rng(seed)
    train_idxs, val_idxs = [], []
    for tier in np.unique(tiers):
        mask = np.where(tiers == tier)[0]
        rng.shuffle(mask)
        n_val = max(1, int(len(mask) * val_frac))
        val_idxs.extend(mask[:n_val].tolist())
        train_idxs.extend(mask[n_val:].tolist())
    return np.array(train_idxs), np.array(val_idxs)


# ---------------------------------------------------------------------------
# Divergence (differentiable, torch)
# ---------------------------------------------------------------------------

def _l2_norm(v: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def divergence_torch(
    f: torch.Tensor,  # (batch, 5) — already L1-normalized
    s: torch.Tensor,  # (batch, 5)
    T: torch.Tensor,  # (5, 5)
) -> torch.Tensor:
    """Differentiable D = 1 - cos(f, T^T @ s). Returns (batch,)."""
    expected = s @ T  # (batch, 5) — T^T @ s == s @ T when using row-vector convention
    f_n = _l2_norm(f)
    e_n = _l2_norm(expected)
    cos = (f_n * e_n).sum(dim=-1)  # (batch,)
    return 1.0 - cos.clamp(-1.0, 1.0)


def project_activations(W: nn.Linear, acts: torch.Tensor) -> torch.Tensor:
    """Apply W and L1-normalize: f = L1_norm(relu(W @ acts))."""
    raw = W(acts)  # (batch, 5)
    raw = F.relu(raw)
    total = raw.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    return raw / total


# ---------------------------------------------------------------------------
# Contrastive pair sampling
# ---------------------------------------------------------------------------

def sample_contrastive_pairs(
    train_idxs: np.ndarray, tiers: np.ndarray, n_pairs: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (hazard_idx, benign_idx) pairs from training indices."""
    hazard = train_idxs[tiers[train_idxs] == "hazard_adjacent_category"]
    benign = train_idxs[tiers[train_idxs] == "benign_bio"]
    if len(hazard) == 0 or len(benign) == 0:
        return np.array([]), np.array([])
    h_sample = rng.choice(hazard, size=n_pairs, replace=True)
    b_sample = rng.choice(benign, size=n_pairs, replace=True)
    return h_sample, b_sample


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset: dict,
    T_np: np.ndarray,
    d_sae: int,
    steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    lam_recon: float = 0.5,
    lam_l2: float = 1e-3,
    margin: float = 0.3,
    batch_size: int = 16,
    pairs_per_step: int = 32,
    patience: int = 30,
    val_frac: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[nn.Linear, list[dict]]:

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    feature_acts = torch.tensor(dataset["feature_acts"], dtype=torch.float32, device=device)
    surface_soft = torch.tensor(dataset["surface_soft"], dtype=torch.float32, device=device)
    feature_vec = torch.tensor(dataset["feature_vec"], dtype=torch.float32, device=device)
    T = torch.tensor(T_np, dtype=torch.float32, device=device)
    tiers = dataset["tiers"]

    train_idxs, val_idxs = stratified_split(len(tiers), tiers, val_frac, seed)
    print(f"Train: {len(train_idxs)}  Val: {len(val_idxs)}")
    print(f"Train tiers: { {t: int((tiers[train_idxs] == t).sum()) for t in np.unique(tiers)} }")

    # W: (5, d_sae) — nn.Linear(d_sae, 5, bias=False)
    W = nn.Linear(d_sae, 5, bias=False).to(device)
    nn.init.xavier_uniform_(W.weight)

    optimizer = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=weight_decay)

    log: list[dict] = []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for step in range(1, steps + 1):
        W.train()

        # --- Contrastive loss (hazard vs benign pairs) ---
        h_idxs, b_idxs = sample_contrastive_pairs(train_idxs, tiers, pairs_per_step, rng)
        if len(h_idxs) > 0:
            f_h = project_activations(W, feature_acts[h_idxs])
            f_b = project_activations(W, feature_acts[b_idxs])
            s_h = surface_soft[h_idxs]
            s_b = surface_soft[b_idxs]
            D_h = divergence_torch(f_h, s_h, T)
            D_b = divergence_torch(f_b, s_b, T)
            L_contrast = F.relu(margin - D_h + D_b).mean()
        else:
            L_contrast = torch.tensor(0.0, device=device)

        # --- Reconstruction loss against catalog f (soft prior) ---
        batch_idxs = rng.choice(train_idxs, size=min(batch_size, len(train_idxs)), replace=False)
        f_w = project_activations(W, feature_acts[batch_idxs])
        f_cat = feature_vec[batch_idxs]
        # Only penalize recon where catalog produced non-zero f (i.e., catalog was tuned)
        valid_mask = (f_cat.sum(dim=-1) > 0.01).float().unsqueeze(1)
        L_recon = (F.mse_loss(f_w, f_cat, reduction="none") * valid_mask).mean()

        loss = L_contrast + lam_recon * L_recon

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Validation ---
        if step % 50 == 0 or step == steps:
            W.eval()
            with torch.no_grad():
                # Contrastive val loss
                h_val, b_val = sample_contrastive_pairs(val_idxs, tiers, pairs_per_step, rng)
                if len(h_val) > 0:
                    val_D_h = divergence_torch(
                        project_activations(W, feature_acts[h_val]),
                        surface_soft[h_val], T,
                    )
                    val_D_b = divergence_torch(
                        project_activations(W, feature_acts[b_val]),
                        surface_soft[b_val], T,
                    )
                    val_contrast = F.relu(margin - val_D_h + val_D_b).mean().item()
                    val_D_h_mean = val_D_h.mean().item()
                    val_D_b_mean = val_D_b.mean().item()
                else:
                    val_contrast = float("nan")
                    val_D_h_mean = val_D_b_mean = float("nan")

                # Per-tier mean D on val set
                tier_D = {}
                for tier in ["benign_bio", "dual_use_bio", "hazard_adjacent_category"]:
                    tidxs = val_idxs[tiers[val_idxs] == tier]
                    if len(tidxs) > 0:
                        f_t = project_activations(W, feature_acts[tidxs])
                        s_t = surface_soft[tidxs]
                        tier_D[tier] = divergence_torch(f_t, s_t, T).mean().item()

                val_loss = val_contrast

            row = {
                "step": step,
                "train_contrast": float(L_contrast),
                "train_recon": float(L_recon),
                "val_contrast": val_contrast,
                "val_D_hazard": val_D_h_mean,
                "val_D_benign": val_D_b_mean,
                "tier_D": tier_D,
            }
            log.append(row)
            print(
                f"[{step:>5d}] L_cont={L_contrast:.4f}  L_recon={L_recon:.4f}"
                f"  val_cont={val_contrast:.4f}"
                f"  D(haz)={tier_D.get('hazard_adjacent_category', float('nan')):.3f}"
                f"  D(ben)={tier_D.get('benign_bio', float('nan')):.3f}"
                f"  D(du)={tier_D.get('dual_use_bio', float('nan')):.3f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in W.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at step {step} (patience={patience})")
                    break

    if best_state is not None:
        W.load_state_dict(best_state)
    return W, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Track B projection adapter W.")
    ap.add_argument("--activations", nargs="+", required=True,
                    help="One or more activations.npz paths (from --dump-activations runs)")
    ap.add_argument("--report", nargs="+", default=[],
                    help="Paired report.json paths (same order as --activations, optional)")
    ap.add_argument("--calibration", required=True, help="Calibration YAML with T matrix")
    ap.add_argument("--out-pt", default="configs/projection_adapter_gemma2_2b.pt")
    ap.add_argument("--out-yaml", default="configs/projection_adapter_gemma2_2b.yaml")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lam-recon", type=float, default=0.5,
                    help="Weight of MSE reconstruction loss against catalog f")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="Contrastive margin: D(hazard) > D(benign) + margin")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=32, help="Contrastive pairs per step")
    ap.add_argument("--patience", type=int, default=30, help="Early stopping patience (in val checks)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    # Pad report list to match activations (fill with empty string if missing)
    reports = args.report + [""] * (len(args.activations) - len(args.report))

    print(f"Loading activations from {len(args.activations)} file(s)...")
    dataset, T_np = build_dataset(args.activations, reports, args.calibration)
    n, d_sae = dataset["feature_acts"].shape
    print(f"Dataset: {n} samples, d_sae={d_sae}")
    print(f"Tier distribution: { {t: int((dataset['tiers'] == t).sum()) for t in np.unique(dataset['tiers'])} }")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = "cpu"

    W, log = train(
        dataset=dataset,
        T_np=T_np,
        d_sae=d_sae,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lam_recon=args.lam_recon,
        margin=args.margin,
        batch_size=args.batch_size,
        pairs_per_step=args.pairs,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )

    # --- Save W ---
    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(W.state_dict(), args.out_pt)
    print(f"Saved W to {args.out_pt}")

    # --- Save YAML metadata ---
    final_row = log[-1] if log else {}
    meta = {
        "fit_date": datetime.datetime.utcnow().isoformat() + "Z",
        "fit_samples": n,
        "d_sae": d_sae,
        "n_categories": 5,
        "feature_categories": list(FEATURE_CATEGORIES),
        "activations_sources": args.activations,
        "calibration": args.calibration,
        "hyperparams": {
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lam_recon": args.lam_recon,
            "margin": args.margin,
            "batch_size": args.batch_size,
            "pairs_per_step": args.pairs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "final_metrics": final_row,
        "out_pt": str(Path(args.out_pt).resolve()),
    }

    out_yaml_path = Path(args.out_yaml)
    # Append block if file already exists (preserves history like calibration YAMLs)
    if out_yaml_path.exists():
        existing = out_yaml_path.read_text(encoding="utf-8")
        out_yaml_path.write_text(
            existing + "\n---\n# Re-fit " + meta["fit_date"] + "\n"
            + yaml.dump(meta, default_flow_style=False),
            encoding="utf-8",
        )
    else:
        out_yaml_path.write_text(yaml.dump(meta, default_flow_style=False), encoding="utf-8")
    print(f"Saved metadata to {args.out_yaml}")

    # --- Print final tier-D summary ---
    print("\n=== Final val tier-D summary ===")
    fd = final_row.get("tier_D", {})
    for tier in ["benign_bio", "dual_use_bio", "hazard_adjacent_category"]:
        print(f"  {tier:35s}: D = {fd.get(tier, float('nan')):.4f}")
    sep = fd.get("hazard_adjacent_category", 0) - fd.get("benign_bio", 0)
    print(f"  Tier separation (hazard - benign D): {sep:.4f}")
    print(f"  (baseline catalog separation from paper: see runs/gemma-2-2b-it-150tok-rejudged/report.json)")


if __name__ == "__main__":
    main()
