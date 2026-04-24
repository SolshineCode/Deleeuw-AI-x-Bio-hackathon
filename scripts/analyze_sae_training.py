"""Analyze SAE training log from train_sae_local.py.

Prints: final metrics, convergence summary, loss decomposition.
Saves: demo/sae_training_curves.png (optional, requires matplotlib).

Usage:
    python scripts/analyze_sae_training.py \
        --log runs/sae-training-local/training_log.jsonl \
        [--plot]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_log(log_path: str) -> list[dict]:
    records = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/sae-training-local/training_log.jsonl")
    ap.add_argument("--plot", action="store_true", help="Save loss curves to demo/sae_training_curves.png")
    args = ap.parse_args()

    records = load_log(args.log)
    if not records:
        print("No records found.")
        return

    first, last = records[0], records[-1]
    n = len(records)

    print(f"\n{'='*70}")
    print(f"SAE TRAINING ANALYSIS — {args.log}")
    print(f"{'='*70}")
    print(f"Steps logged:    {n}  (step {first['step']} → step {last['step']})")

    print(f"\n{'Metric':22s} {'Initial':>10s} {'Final':>10s} {'Delta':>10s}")
    print("-" * 56)
    for key, label in [
        ("total_loss",   "Total loss"),
        ("l_recon",      "L_recon"),
        ("l_sparsity",   "L_sparsity"),
        ("l_contrastive","L_contrastive"),
        ("l0",           "L0 (active features)"),
    ]:
        if key in first and key in last:
            v0, v1 = first[key], last[key]
            print(f"  {label:20s} {v0:10.4f} {v1:10.4f} {v1-v0:+10.4f}")

    # Convergence: check last 20% of steps
    tail = records[int(0.8 * n):]
    if len(tail) >= 2:
        tail_losses = [r["total_loss"] for r in tail]
        tail_range = max(tail_losses) - min(tail_losses)
        tail_mean = sum(tail_losses) / len(tail_losses)
        cv = tail_range / tail_mean if tail_mean > 0 else 0
        print(f"\nConvergence (last 20% of steps):")
        print(f"  Total-loss range: {tail_range:.4f}  (CV={cv:.3f})")
        if cv < 0.05:
            print(f"  Status: CONVERGED (CV < 0.05)")
        elif cv < 0.15:
            print(f"  Status: NEAR-CONVERGED (CV < 0.15)")
        else:
            print(f"  Status: NOT CONVERGED (CV={cv:.3f} >= 0.15) — consider more steps")

    # Contrastive loss direction
    l_c_last = last.get("l_contrastive", float("nan"))
    print(f"\nContrastive loss (final): {l_c_last:.4f}")
    if l_c_last < 0.5:
        print("  > Low: SAE is successfully separating hazard/benign feature spaces")
    elif l_c_last < 0.8:
        print("  > Moderate: partial hazard/benign separation")
    else:
        print("  > High (>0.8): cosine sim near 1 — hazard/benign means still aligned (not separated)")

    print(f"\nFinal L0 (active features/token): {last.get('l0', '?'):.1f}")
    print(f"{'='*70}\n")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [r["step"] for r in records]
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            fig.suptitle("SAE Training Curves (BioRefusalAudit local SAE)")

            axes[0, 0].plot(steps, [r["total_loss"] for r in records], label="Total loss")
            axes[0, 0].set_title("Total Loss"); axes[0, 0].set_xlabel("Step"); axes[0, 0].legend()

            axes[0, 1].plot(steps, [r["l_recon"] for r in records], color="steelblue", label="L_recon")
            axes[0, 1].set_title("Reconstruction Loss"); axes[0, 1].set_xlabel("Step"); axes[0, 1].legend()

            axes[1, 0].plot(steps, [r["l_contrastive"] for r in records], color="firebrick", label="L_contrast")
            axes[1, 0].set_title("Contrastive Loss (lower = better separation)"); axes[1, 0].set_xlabel("Step"); axes[1, 0].legend()

            axes[1, 1].plot(steps, [r["l0"] for r in records], color="seagreen", label="L0")
            axes[1, 1].set_title("L0 (Active Features / Token)"); axes[1, 1].set_xlabel("Step"); axes[1, 1].legend()

            plt.tight_layout()
            out_path = Path("demo/sae_training_curves.png")
            out_path.parent.mkdir(exist_ok=True)
            plt.savefig(out_path, dpi=150)
            print(f"Training curves saved: {out_path}")
        except ImportError:
            print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
