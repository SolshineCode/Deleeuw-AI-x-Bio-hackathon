"""Re-analysis with relabelled surface labels (2026-10). CPU only.

Inputs: committed results/*/report.json, results/relabel_2026-10/{manifest,votes}.jsonl.
Runs on whatever subset has been relabelled; the output header states the coverage and
marks the result PILOT unless every manifest item has both judges.

Analyses:
  A. Label rates per run and tier: original vs relabelled, over the same records, with
     Wilson 95% CIs on judged records and the unjudged count shown.
  B. Fate of the original placeholder refusals (what the new judges call them).
  C. Variance decomposition of stored D per run: share explained by the original label
     (between-label SS / total SS), by the 5-dim feature vector f (OLS R^2), and the
     within-label (activation-driven) share. Uses all records with stored D, independent
     of the relabel.
  D. D recomputed with relabelled one-hot surface vectors, using the calibration matrix T
     identified in Step 0, on relabelled records.
  E. Finding 7 test (Gemma 4 + author SAE, 150 tok): (i) original comply/refuse separation;
     (ii) whether it survives once the label is removed: does f itself differ between the
     relabelled classes (permutation test on the distance between class means of
     normalised f)?

Tier-3 (hazard_adjacent_category) records are not relabelled (completions redacted); they
are excluded from A, B, D and reported as 'not checkable'.

Output: results/reanalysis_2026-10/reanalysis_<tag>.{json,md}
Usage:  python scripts/reanalysis/reanalyze.py --tag pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RESULTS, SURFACE_CLASSES, candidate_Ts, divergence, identify_T, is_placeholder, r2_groups,
    r2_linear, wilson,
)
import relabel_openrouter_free as RL  # noqa: E402

D_RUNS = (
    "gemma-2-2b-it-L12-tuned", "gemma-2-2b-it-150tok-rejudge", "gemma-2-2b-it-80tok-rejudge",
    "gemma-2-2b-it-explicit-benign-qwen3-100", "gemma-2-2b-it-explicit-dualuse-qwen3-100",
    "gemma-2-2b-it-explicit-tier3-qwen3-100", "gemma-4-E2B-it-our-sae-v1-150tok",
    "gemma-4-E2B-it-L17-pass5", "control-legal-financial-gemma2", "gemma-2-2b-it-bio-sae-v1-150tok",
    "gemma-4-E2B-it-bio-sae-v1-150tok",
)
RATE_RUNS = (
    "gemma-2-2b-it-L12-tuned", "gemma-4-E2B-it-L17-pass5", "gemma-4-E2B-it-our-sae-v1-150tok",
    "llama-3.2-1b-it-cross-arch", "qwen25-1b5-it-cross-arch", "phi3-mini-4k-it-cross-arch",
    "gemma-2-2b-it-explicit-benign-qwen3-100", "gemma-2-2b-it-explicit-dualuse-qwen3-100",
    "psilocybin-gemma2-sae-80tok", "psilocybin-gemma2-sae-150tok", "psilocybin-qwen25-1b5-80tok",
    "psilocybin-llama32-1b-80tok", "legality-gemma2-80tok", "legality-gemma2-sae-can-psi",
    "control-legal-financial-gemma2",
)


def new_label_index(relabel_dir: Path, min_judges: int = 2) -> tuple[dict, dict, int, int]:
    """(run, record id) -> final relabel verdict; plus coverage numbers."""
    manifest = [json.loads(x) for x in (relabel_dir / "manifest.jsonl").read_text().splitlines()]
    final = RL.consolidate(relabel_dir) if (relabel_dir / "votes.jsonl").exists() else {}
    idx = {}
    for m in manifest:
        f = final.get(m["item_id"])
        if not f or f["n_judges"] < min_judges:
            continue
        for o in m["occurrences"]:
            idx[(o["run"], o["id"], o.get("condition"))] = f
    complete = sum(1 for f in final.values() if f["n_judges"] >= 2)
    return idx, final, complete, len(manifest)


def fmt_rate(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]"


def perm_test_f(F: np.ndarray, groups: list[str], n_perm: int = 5000, seed: int = 0) -> float | None:
    labs = sorted(set(groups))
    if len(labs) < 2 or min(Counter(groups).values()) < 3:
        return None
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    g = np.array(groups)

    def stat(gg):
        a, b = Fn[gg == labs[0]], Fn[gg == labs[1]]
        return float(np.linalg.norm(a.mean(0) - b.mean(0)))

    obs = stat(g)
    rng = np.random.default_rng(seed)
    cnt = sum(stat(rng.permutation(g)) >= obs - 1e-15 for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel-dir", default=str(RESULTS / "relabel_2026-10"))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--min-judges", type=int, default=2,
                    help="items need votes from this many judges (a failed vote still counts as a vote)")
    args = ap.parse_args()
    idx, final, complete, total = new_label_index(Path(args.relabel_dir), args.min_judges)
    is_pilot = complete < total
    n_used = len({f["item_id"] for f in idx.values()})
    agree_pool = [f for f in final.values() if f["n_usable"] == 2]
    out = {"coverage": {"items_with_both_judges": complete, "manifest_items": total, "pilot": is_pilot,
                        "items_used": n_used, "min_judges": args.min_judges,
                        "both_answered": len(agree_pool), "both_agree": sum(f["agree"] for f in agree_pool)}}
    md = [f"# Re-analysis {'(PILOT, descriptive only)' if is_pilot else '(full relabel)'}: generated", "",
          f"Items used: {n_used} of {total} unique non-tier-3 completions (min judges {args.min_judges});",
          f"both judges answered on {len(agree_pool)} and agreed on {sum(f['agree'] for f in agree_pool)}.",
          "An item with one usable vote takes that vote's label; an exact tie between judges is unjudged.",
          "Rates below cover only records whose completion has been relabelled, so per-run n is",
          "smaller than the published n and the subset is a stratified pilot sample, not the full run." if is_pilot else "",
          "Tier-3 records are not relabelled (completions redacted in every committed run).", ""]

    # A + B
    md += ["## A. Label rates, original vs relabelled (same records)", "",
           "| run | tier | n | original refuse | original comply | relabelled refuse | relabelled comply | relabelled hedge+partial | unjudged |",
           "|---|---|---|---|---|---|---|---|---|"]
    fate = Counter()
    A = []
    for run in RATE_RUNS:
        p = RESULTS / run / "report.json"
        if not p.exists():
            continue
        recs = json.loads(p.read_text())["records"]
        by_t = defaultdict(list)
        for r in recs:
            f = idx.get((run, r["id"], r.get("condition")))
            if f and r["tier"] != "hazard_adjacent_category":
                by_t[r["tier"]].append((r, f))
                if is_placeholder(r):
                    fate[f["label"] or "unjudged:" + f["unjudged_reason"].split(":")[0]] += 1
        for t, pairs in sorted(by_t.items()):
            n = len(pairs)
            new = [f["label"] for _, f in pairs]
            judged = [x for x in new if x]
            nj = len(judged)
            row = {
                "run": run, "tier": t, "n": n,
                "orig_refuse": sum(r["surface_label"] == "refuse" for r, _ in pairs),
                "orig_comply": sum(r["surface_label"] == "comply" for r, _ in pairs),
                "new_counts": dict(Counter(judged)), "n_judged": nj, "n_unjudged": n - nj,
            }
            A.append(row)
            md.append(
                f"| {run} | {t} | {n} | {row['orig_refuse']} | {row['orig_comply']} | "
                f"{fmt_rate(judged.count('refuse'), nj)} | {fmt_rate(judged.count('comply'), nj)} | "
                f"{fmt_rate(judged.count('hedge') + judged.count('partial'), nj)} | {n - nj} |"
            )
    out["A_rates"] = A
    out["B_placeholder_fate"] = dict(fate)
    md += ["", "## B. What the relabel says about original placeholder refusals", "",
           "| relabelled as | records |", "|---|---|"] + [f"| {k} | {v} |" for k, v in fate.most_common()]

    # C variance decomposition (no relabel needed)
    md += ["", "## C. Variance decomposition of stored D (all records with D)", "",
           "Between-label share = variance of D explained by the original surface label. f R^2 = OLS",
           "of D on the 5-dim feature vector. Within-label share = 1 - between-label share.", "",
           "| run | n | between-label share | f R^2 | within-label (activation) share | placeholder records |",
           "|---|---|---|---|---|---|"]
    C = []
    for run in D_RUNS:
        p = RESULTS / run / "report.json"
        if not p.exists():
            continue
        recs = [r for r in json.loads(p.read_text())["records"] if r.get("divergence") is not None]
        y = np.array([r["divergence"] for r in recs])
        labs = [r["surface_label"] for r in recs]
        F = np.array([r["feature_vec"] for r in recs], dtype=float)
        rb = r2_groups(y, labs)
        rf = r2_linear(y, F)
        row = {"run": run, "n": len(recs), "r2_label": rb, "r2_f": rf, "within": 1 - rb,
               "n_placeholder": sum(map(is_placeholder, recs))}
        C.append(row)
        md.append(f"| {run} | {len(recs)} | {rb:.3f} | {rf:.3f} | {1-rb:.3f} | {row['n_placeholder']} |")
    out["C_variance"] = C

    # D recompute with new labels
    md += ["", "## D. D recomputed with relabelled one-hot surface vectors (relabelled records only)", "",
           "| run | T used | tier | n | mean D (stored) | mean D (relabelled) |", "|---|---|---|---|---|---|"]
    Ts = candidate_Ts()
    Dres = []
    for run in ("gemma-2-2b-it-L12-tuned", "gemma-2-2b-it-explicit-benign-qwen3-100",
                "gemma-2-2b-it-explicit-dualuse-qwen3-100", "gemma-4-E2B-it-our-sae-v1-150tok",
                "control-legal-financial-gemma2"):
        recs = json.loads((RESULTS / run / "report.json").read_text())["records"]
        tname, err = identify_T(recs)
        if tname is None or err > 1e-6:
            md.append(f"| {run} | not identified (max err {err:.2g}) | | | | |")
            continue
        T = Ts[tname]
        by_t = defaultdict(lambda: ([], []))
        for r in recs:
            f = idx.get((run, r["id"], r.get("condition")))
            if not f or not f["label"] or r["tier"] == "hazard_adjacent_category":
                continue
            s = np.array([1.0 if c == f["label"] else 0.0 for c in SURFACE_CLASSES])
            by_t[r["tier"]][0].append(r["divergence"])
            by_t[r["tier"]][1].append(divergence(s, np.asarray(r["feature_vec"], float), T))
        for t, (old, new) in sorted(by_t.items()):
            Dres.append({"run": run, "T": tname, "tier": t, "n": len(old),
                         "mean_D_stored": float(np.mean(old)), "mean_D_relabelled": float(np.mean(new))})
            md.append(f"| {run} | {tname} | {t} | {len(old)} | {np.mean(old):.3f} | {np.mean(new):.3f} |")
    out["D_recomputed"] = Dres

    # E Finding 7
    run = "gemma-4-E2B-it-our-sae-v1-150tok"
    recs = json.loads((RESULTS / run / "report.json").read_text())["records"]
    F = np.array([r["feature_vec"] for r in recs], dtype=float)
    orig = [r["surface_label"] for r in recs]
    y = np.array([r["divergence"] for r in recs])
    new = [(idx.get((run, r["id"], None)) or {}).get("label") for r in recs]
    sel = [i for i, x in enumerate(new) if x]
    newc = Counter(new[i] for i in sel)
    groups = ["comply" if new[i] == "comply" else "not_comply" for i in sel]
    p_new = perm_test_f(F[sel], groups) if sel else None
    p_orig = perm_test_f(F, ["comply" if x == "comply" else "not_comply" for x in orig])
    E = {
        "n": len(recs), "orig_counts": dict(Counter(orig)),
        "orig_placeholder_refuse": sum(map(is_placeholder, recs)),
        "r2_D_on_orig_label": r2_groups(y, orig),
        "f_component_sd": [float(x) for x in F.std(0)],
        "relabelled_n": len(sel), "relabelled_counts": dict(newc),
        "perm_p_f_by_orig_label": p_orig, "perm_p_f_by_relabel": p_new,
    }
    out["E_finding7"] = E
    md += ["", "## E. Finding 7 test (Gemma 4 E2B-IT + author SAE, 150 tokens)", "",
           f"- Original labels: {E['orig_counts']}; placeholder refusals: {E['orig_placeholder_refuse']}.",
           f"- Share of D variance explained by the original label: {E['r2_D_on_orig_label']:.5f}.",
           f"- SD of each feature-vector component: {[round(x, 4) for x in E['f_component_sd']]}.",
           f"- Permutation test, does normalised f differ between original comply and non-comply: p = {p_orig}.",
           f"- Relabelled records: {len(sel)} with labels {dict(newc)}.",
           f"- Same test with relabelled classes: p = {p_new if p_new is not None else 'not computable (a class has fewer than 3 records)'}.",
           "", "Reading: if D's separation were activation-driven, f would differ between the label",
           "groups. A large p-value together with near-zero f variance means the separation comes",
           "from the label, not from the activations."]

    o = RESULTS / "reanalysis_2026-10"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"reanalysis_{args.tag}.json").write_text(json.dumps(out, indent=1, default=str))
    (o / f"reanalysis_{args.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
