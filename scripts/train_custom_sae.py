"""Train a custom TopK SAE on Gemma 2 2B-IT residual-stream activations.

Implements the methodology documented in papers/sae_layer_selection/methodology.md
(inherited from the author's prior L12 SAE training pipeline on Qwen2.5-0.5B /
Llama-3.2-1B): TopK with k=32, 4× expansion, activation-at-all-positions, Adam
at 3e-4, 200 epochs with early stopping (patience 15), per-batch decoder
normalization.

Purpose: a custom SAE that can be compared against the pre-trained Gemma Scope 1
release on the same bio-safety eval set. Provides the "methodology portability"
story for the paper.

Usage:
    python scripts/train_custom_sae.py \\
        --model google/gemma-2-2b-it \\
        --layer 12 \\
        --k 32 \\
        --expansion 4 \\
        --n-prompts 200 \\
        --epochs 200 \\
        --out data/custom_sae/gemma-2-2b-it-L12-topk-k32.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.models.model_adapter import load_model, residual_stream_hook
from biorefusalaudit.models.sae_adapter import TopKSAE
from biorefusalaudit.prompts.prompt_loader import load_jsonl


def collect_activations(lm, prompts, layer, max_tokens_per_prompt):
    """Forward-pass each prompt; capture residual stream at layer; return stacked activations."""
    all_acts = []
    with torch.no_grad():
        with residual_stream_hook(lm.model, layer) as get_resid:
            for p in prompts:
                enc = lm.tokenizer(p.prompt, return_tensors="pt", truncation=True,
                                    max_length=max_tokens_per_prompt).to(
                    next(lm.model.parameters()).device
                )
                lm.model(**enc)
                resid = get_resid()  # (batch=1, seq, d_model)
                if resid is not None:
                    all_acts.append(resid.reshape(-1, resid.shape[-1]).cpu().float())
    return torch.cat(all_acts, dim=0) if all_acts else torch.empty(0, 0)


def train_topk_sae(acts, k, expansion, epochs, batch_size, lr, patience, d_sae=None):
    """Train a TopKSAE on the (N, d_model) activation tensor."""
    d_model = acts.shape[1]
    if d_sae is None:
        d_sae = d_model * expansion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = TopKSAE(d_model, d_sae, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    acts = acts.to(device)
    n = acts.shape[0]
    best_loss = float("inf")
    patience_left = patience
    losses_history = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        total_batches = 0
        for i in range(0, n, batch_size):
            batch = acts[perm[i:i + batch_size]]
            opt.zero_grad()
            z, x_hat = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, dim=-1)
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        losses_history.append(avg_loss)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_left = patience
        else:
            patience_left -= 1
        if epoch % 20 == 0 or patience_left <= 0:
            print(f"[train_sae] epoch {epoch+1}/{epochs} loss={avg_loss:.5f} best={best_loss:.5f} patience={patience_left}")
        if patience_left <= 0:
            print(f"[train_sae] early stop at epoch {epoch+1}")
            break

    # Final eval: MSE, EV, L0, alive-feature count on training set
    with torch.no_grad():
        z, x_hat = sae(acts)
        mse = F.mse_loss(x_hat, acts).item()
        total_var = acts.var(dim=0).sum().item()
        residual_var = (acts - x_hat).var(dim=0).sum().item()
        explained_var = 1.0 - residual_var / total_var
        l0 = (z != 0).float().sum(dim=-1).mean().item()
        alive = (z.sum(dim=0) > 0).sum().item()

    return sae, {
        "mse_loss": mse,
        "explained_variance": explained_var,
        "l0": l0,
        "alive_features": int(alive),
        "d_model": d_model,
        "d_sae": d_sae,
        "k": k,
        "expansion": expansion,
        "epochs_trained": len(losses_history),
        "best_loss": best_loss,
        "loss_history": losses_history,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--layer", default=12, type=int)
    ap.add_argument("--eval-set", type=Path,
                    default=REPO / "data/eval_set_public/eval_set_public_v1.jsonl")
    ap.add_argument("--n-prompts", default=75, type=int)
    ap.add_argument("--max-tokens-per-prompt", default=64, type=int)
    ap.add_argument("--k", default=32, type=int)
    ap.add_argument("--expansion", default=4, type=int)
    ap.add_argument("--epochs", default=200, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--d-sae", type=int, help="Force d_sae to a specific value")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    print(f"[train_sae] Loading model: {args.model}")
    lm = load_model(args.model)

    prompts = load_jsonl(args.eval_set)[: args.n_prompts]
    print(f"[train_sae] Collecting activations at layer {args.layer} on {len(prompts)} prompts...")
    t0 = time.time()
    acts = collect_activations(lm, prompts, args.layer, args.max_tokens_per_prompt)
    print(f"[train_sae] Collected {acts.shape[0]} activation vectors of dim {acts.shape[1]} in {time.time()-t0:.1f}s")

    print(f"[train_sae] Training TopKSAE k={args.k} expansion={args.expansion} epochs={args.epochs}")
    t0 = time.time()
    sae, metrics = train_topk_sae(
        acts, k=args.k, expansion=args.expansion, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        d_sae=args.d_sae
    )
    metrics["train_time_s"] = time.time() - t0
    metrics["n_training_activations"] = int(acts.shape[0])
    metrics["model"] = args.model
    metrics["layer"] = args.layer
    metrics["architecture"] = "topk"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Save weights
    torch.save(sae.state_dict(), args.out)
    # Save metrics
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train_sae] Wrote {args.out} + {meta_path}")
    print(f"[train_sae] Final metrics: MSE={metrics['mse_loss']:.4f} "
          f"EV={metrics['explained_variance']:.4f} L0={metrics['l0']:.1f} "
          f"alive={metrics['alive_features']}/{metrics['d_sae']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
