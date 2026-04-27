"""
SAE internal-representation confound analysis.

Compares Gemma Scope feature activations across three content groups:
  - hazard_adjacent  (genuine biosecurity content, from main eval)
  - psilocybin       (Schedule I, biologically benign / proneurogenic)
  - cannabis         (Schedule I, culturally normalized)

For each of the 16,384 SAE features, computes mean activation per group
and Cohen's d between pairs. Top differential features are printed with
Neuronpedia URLs for manual inspection.

Usage (after legality-gemma2-sae-can-psi run completes):
    python scripts/analyze_sae_confound.py

Or pass custom paths:
    python scripts/analyze_sae_confound.py \
        --main-acts runs/gemma-2-2b-it-L12-tuned/activations.npz \
        --confound-acts runs/legality-gemma2-sae-can-psi/activations.npz \
        --confound-report runs/legality-gemma2-sae-can-psi/report.json \
        --top-n 20
"""
import argparse
import json
from pathlib import Path

import numpy as np


NEURONPEDIA_BASE = "https://neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k"
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_LAYER = 12


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d: positive means a > b."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2 + 1e-9)
    return float((a.mean() - b.mean()) / pooled_std)


def load_main_acts(path: Path) -> dict[str, np.ndarray]:
    """Load main eval activations grouped by tier."""
    data = np.load(path, allow_pickle=True)
    tiers = data["tiers"]
    acts = data["feature_acts"]
    groups: dict[str, list] = {}
    for tier, row in zip(tiers, acts):
        groups.setdefault(tier, []).append(row)
    return {k: np.stack(v) for k, v in groups.items()}


def load_confound_acts(acts_path: Path, report_path: Path) -> dict[str, np.ndarray]:
    """Load confound-run activations grouped by compound (from JSONL category join)."""
    # Load JSONL category lookup
    jsonl_cat: dict[str, str] = {}
    for jsonl in [
        Path("data/eval_set_public/eval_set_legality_confound_v1.jsonl"),
        Path("data/eval_set_public/eval_set_legality_can_psi_v1.jsonl"),
    ]:
        if jsonl.exists():
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("id") and obj.get("category"):
                        jsonl_cat[obj["id"]] = obj["category"]

    data = np.load(acts_path, allow_pickle=True)
    ids = data["ids"]
    acts = data["feature_acts"]

    groups: dict[str, list] = {}
    for pid, row in zip(ids, acts):
        cat = jsonl_cat.get(pid, "")
        compound = "unknown"
        for c in ("cannabis", "psilocybin", "lsd", "mescaline"):
            if cat.startswith(c):
                compound = c
                break
        groups.setdefault(compound, []).append(row)

    return {k: np.stack(v) for k, v in groups.items()}


def top_features(
    group_a: np.ndarray,
    group_b: np.ndarray,
    label_a: str,
    label_b: str,
    top_n: int = 20,
) -> list[dict]:
    """Return top_n features where group_a >> group_b (by Cohen's d)."""
    d_sae = group_a.shape[1]
    results = []
    for feat_idx in range(d_sae):
        a_vals = group_a[:, feat_idx]
        b_vals = group_b[:, feat_idx]
        d = cohen_d(a_vals, b_vals)
        mean_a = float(a_vals.mean())
        mean_b = float(b_vals.mean())
        if mean_a < 0.01 and mean_b < 0.01:
            continue  # skip near-zero in both
        results.append({
            "feature_idx": feat_idx,
            "cohen_d": d,
            f"mean_{label_a}": mean_a,
            f"mean_{label_b}": mean_b,
            "neuronpedia": f"{NEURONPEDIA_BASE}/{feat_idx}",
        })
    results.sort(key=lambda x: -x["cohen_d"])
    return results[:top_n]


def print_table(rows: list[dict], label_a: str, label_b: str, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"  (positive d = {label_a} > {label_b})")
    print(f"{'='*70}")
    print(f"  {'feat':>6}  {'d':>7}  {'mean_'+label_a:>14}  {'mean_'+label_b:>14}  neuronpedia")
    print(f"  {'-'*65}")
    for r in rows:
        print(
            f"  {r['feature_idx']:>6}  {r['cohen_d']:>7.3f}"
            f"  {r[f'mean_{label_a}']:>14.4f}  {r[f'mean_{label_b}']:>14.4f}"
            f"  {r['neuronpedia']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-acts",
                        default="runs/gemma-2-2b-it-L12-tuned/activations.npz")
    parser.add_argument("--confound-acts",
                        default="runs/legality-gemma2-sae-can-psi/activations.npz")
    parser.add_argument("--confound-report",
                        default="runs/legality-gemma2-sae-can-psi/report.json")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    main_path = Path(args.main_acts)
    confound_path = Path(args.confound_acts)

    if not main_path.exists():
        print(f"ERROR: main activations not found: {main_path}")
        return
    if not confound_path.exists():
        print(f"ERROR: confound activations not found: {confound_path}")
        print("  (run is still in progress — re-run this script when complete)")
        return

    print(f"Loading main eval activations: {main_path}")
    main_groups = load_main_acts(main_path)
    for k, v in main_groups.items():
        print(f"  {k}: {len(v)} prompts, {v.shape[1]} features")

    print(f"\nLoading confound activations: {confound_path}")
    confound_groups = load_confound_acts(confound_path, Path(args.confound_report))
    for k, v in confound_groups.items():
        print(f"  {k}: {len(v)} prompts")

    hazard = main_groups.get("hazard_adjacent_category", np.zeros((0, 16384)))
    psilocybin = confound_groups.get("psilocybin", np.zeros((0, 16384)))
    cannabis = confound_groups.get("cannabis", np.zeros((0, 16384)))

    print(f"\nGroup sizes: hazard_adjacent={len(hazard)}, psilocybin={len(psilocybin)}, cannabis={len(cannabis)}")

    if len(hazard) == 0 or len(psilocybin) == 0 or len(cannabis) == 0:
        print("ERROR: one or more groups empty — cannot compare")
        return

    n_feat = hazard.shape[1]
    print(f"SAE dimension: {n_feat} features")

    # --- Key comparisons ---

    # 1. Psilocybin ≈ Hazard >> Cannabis → internal confound (psi conflated with biosecurity)
    rows_psi_vs_can = top_features(psilocybin, cannabis, "psilocybin", "cannabis",
                                   top_n=args.top_n)
    print_table(rows_psi_vs_can, "psilocybin", "cannabis",
                "Features: PSILOCYBIN > CANNABIS (psilocybin-specific signal)")

    # 2. Hazard >> Psilocybin → model internally distinguishes hazard from psilocybin
    rows_haz_vs_psi = top_features(hazard, psilocybin, "hazard", "psilocybin",
                                   top_n=args.top_n)
    print_table(rows_haz_vs_psi, "hazard", "psilocybin",
                "Features: HAZARD > PSILOCYBIN (hazard-specific, not psilocybin)")

    # 3. Hazard >> Cannabis → hazard vs cannabis comparison
    rows_haz_vs_can = top_features(hazard, cannabis, "hazard", "cannabis",
                                   top_n=args.top_n)
    print_table(rows_haz_vs_can, "hazard", "cannabis",
                "Features: HAZARD > CANNABIS")

    # 4. Check overlap: features that fire on BOTH hazard AND psilocybin but not cannabis
    print(f"\n{'='*70}")
    print("  OVERLAP ANALYSIS: features active in hazard AND psilocybin but not cannabis")
    print(f"{'='*70}")

    mean_haz = hazard.mean(axis=0)
    mean_psi = psilocybin.mean(axis=0)
    mean_can = cannabis.mean(axis=0)

    threshold = 0.5  # minimum mean activation to be considered "active"
    both_haz_psi = np.where(
        (mean_haz > threshold) & (mean_psi > threshold) & (mean_can < threshold)
    )[0]

    print(f"  Features active in hazard+psilocybin but not cannabis (threshold={threshold}):")
    print(f"  Count: {len(both_haz_psi)}")
    if len(both_haz_psi) > 0:
        top_overlap = sorted(both_haz_psi,
                             key=lambda i: -(mean_haz[i] + mean_psi[i]))[:10]
        print(f"  {'feat':>6}  {'mean_haz':>10}  {'mean_psi':>10}  {'mean_can':>10}  neuronpedia")
        print(f"  {'-'*60}")
        for i in top_overlap:
            print(
                f"  {i:>6}  {mean_haz[i]:>10.4f}  {mean_psi[i]:>10.4f}"
                f"  {mean_can[i]:>10.4f}  {NEURONPEDIA_BASE}/{i}"
            )

    # Summary
    print(f"\n{'='*70}")
    print("  INTERPRETATION SUMMARY")
    print(f"{'='*70}")

    # Count features where psilocybin > cannabis by more than 0.5 mean activation
    psi_gt_can = np.sum((mean_psi - mean_can) > 0.5)
    haz_gt_psi = np.sum((mean_haz - mean_psi) > 0.5)
    haz_gt_can = np.sum((mean_haz - mean_can) > 0.5)
    confound_feats = len(both_haz_psi)

    print(f"  Features where psilocybin >> cannabis (Δ>0.5):  {psi_gt_can}")
    print(f"  Features where hazard >> psilocybin (Δ>0.5):    {haz_gt_psi}")
    print(f"  Features where hazard >> cannabis (Δ>0.5):      {haz_gt_can}")
    print(f"  Shared hazard+psilocybin features (not cannabis): {confound_feats}")
    print()
    if confound_feats > 0 and haz_gt_psi < confound_feats:
        print("  >> INTERNAL CONFOUND LIKELY: psilocybin shares feature space with hazard content")
        print("     and those shared features are suppressed in cannabis.")
    elif haz_gt_psi > psi_gt_can:
        print("  >> INTERNAL DISTINCTION LIKELY: hazard and psilocybin occupy different feature space")
        print("     Refusal of psilocybin may be circuit-level (downstream of features), not representation-level.")
    else:
        print("  >> AMBIGUOUS: check neuronpedia links for top features manually.")


if __name__ == "__main__":
    main()
