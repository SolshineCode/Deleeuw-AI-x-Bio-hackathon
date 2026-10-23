"""Step 0: reproduce the published (arXiv v1) numbers from committed results/.

Every later re-analysis compares against these. A check passes when the recomputed value is
within tolerance of the printed value. Failing checks are reported, not hidden: some are
expected to fail (e.g. Table 2's dual-use cell, see docs/UPGRADE_2026-10.md P-F6-2).

Output: results/reanalysis_2026-10/step0_reproduce.{json,md}
Usage:  python scripts/reanalysis/step0_reproduce.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    RESULTS, TIERS, bootstrap_ci, cohens_d, identify_T, is_placeholder, load_records, perm_p,
)

OUT = RESULTS / "reanalysis_2026-10"
checks: list[dict] = []


def check(name, paper, got, tol, source, note=""):
    ok = abs(paper - got) <= tol if isinstance(paper, (int, float)) else paper == got
    checks.append({"check": name, "paper": paper, "recomputed": round(got, 4) if isinstance(got, float) else got,
                   "tol": tol, "pass": bool(ok), "source": source, "note": note})


def by_tier(recs):
    out = {t: [] for t in TIERS}
    for r in recs:
        if r["tier"] in out:
            out[r["tier"]].append(r)
    return out


def pct(recs, label):
    return 100.0 * sum(r["surface_label"] == label for r in recs) / len(recs)


def main() -> int:
    # Table 1 / Finding 1
    run = "gemma-2-2b-it-L12-tuned-rejudged"
    bt = by_tier(load_records(run))
    paper = {"benign_bio": (0.467, 0.190, (0.392, 0.548), 70, 30, 0),
             "dual_use_bio": (0.655, 0.119, (0.612, 0.697), 80, 20, 0),
             "hazard_adjacent_category": (0.669, 0.113, (0.619, 0.715), 0, 100, 0)}
    for t, (m, sd, ci, c, h, rf) in paper.items():
        d = np.array([r["divergence"] for r in bt[t]])
        check(f"T1 {t} mean D", m, float(d.mean()), 0.0006, run)
        check(f"T1 {t} SD", sd, float(d.std(ddof=0)), 0.0006, run, "population SD (ddof=0); sample SD is larger by ~2%")
        lo, hi = bootstrap_ci(d)
        check(f"T1 {t} CI low", ci[0], lo, 0.01, run, "percentile bootstrap, 10k, seed 0")
        check(f"T1 {t} CI high", ci[1], hi, 0.01, run, "percentile bootstrap, 10k, seed 0")
        check(f"T1 {t} comply %", c, pct(bt[t], "comply"), 0.6, run)
        check(f"T1 {t} hedge %", h, pct(bt[t], "hedge"), 0.6, run)
        check(f"T1 {t} refuse %", rf, pct(bt[t], "refuse"), 0.6, run)
    a = np.array([r["divergence"] for r in bt["benign_bio"]])
    b = np.array([r["divergence"] for r in bt["hazard_adjacent_category"]])
    check("T1 Cohen's d benign vs hazard", 1.29, cohens_d(a, b), 0.015, run, "pooled sample SD gives 1.25; population SD gives 1.28; neither gives 1.29")
    check("T1 permutation p (one-sided)", 0.0001, perm_p(a, b), 0.0002, run, "100k permutations")
    haz_hedge_on_redacted = sum(
        r["surface_label"] == "hedge" and r["completion"].startswith("[REDACTED") for r in bt["hazard_adjacent_category"]
    )
    check("T1 hazard hedges computed from redacted text (R3)", 0, haz_hedge_on_redacted, 0, run,
          "expected to fail: every hazard-tier hedge was labelled from the redaction string")

    # Table 2 / Finding 6
    run = "gemma-2-2b-it-150tok-rejudged"
    bt = by_tier(load_records(run))
    for t, m in (("benign_bio", 0.478), ("dual_use_bio", 0.655), ("hazard_adjacent_category", 0.760)):
        d = np.array([r["divergence"] for r in bt[t]])
        check(f"T2 descriptor {t} mean D", m, float(d.mean()), 0.0006, run,
              "dual-use expected to fail: paper used 0.655 from a different run (P-F6-2)" if t == "dual_use_bio" else "")
    for run, t, m, c, rf in (("gemma-2-2b-it-explicit-benign-qwen3-100", "benign_bio", 0.473, 84, 16),
                             ("gemma-2-2b-it-explicit-dualuse-qwen3-100", "dual_use_bio", 0.675, 76, 24),
                             ("gemma-2-2b-it-explicit-tier3-qwen3-100", "hazard_adjacent_category", 0.714, 33, 67)):
        recs = load_records(run)
        d = np.array([r["divergence"] for r in recs])
        check(f"T2 explicit {t} mean D", m, float(d.mean()), 0.0006, run)
        check(f"T2 explicit {t} comply %", c, pct(recs, "comply"), 0.6, run)
        check(f"T2 explicit {t} refuse %", rf, pct(recs, "refuse"), 0.6, run)
        check(f"T2 explicit {t} refuse labels that are placeholders", 0, sum(map(is_placeholder, recs)), 0, run,
              "expected to fail (R1)")
        # Finding 3 flags
        hf = {"benign_bio": 5, "dual_use_bio": 19, "hazard_adjacent_category": 39}[t]
        rc = {"benign_bio": 82, "dual_use_bio": 76, "hazard_adjacent_category": 33}[t]
        check(f"F3 {t} hazard_features_active_despite_refusal", hf,
              sum(r["flags"].get("hazard_features_active_despite_refusal", False) for r in recs), 0, run)
        check(f"F3 {t} refusal_features_active_despite_compliance", rc,
              sum(r["flags"].get("refusal_features_active_despite_compliance", False) for r in recs), 0, run)
    recs = load_records("gemma-2-2b-it-explicit-tier3-qwen3-100")
    for fr, m, n in (("educational", 0.733, 27), ("roleplay", 0.717, 23), ("direct", 0.707, 27), ("obfuscated", 0.698, 23)):
        d = np.array([r["divergence"] for r in recs if r["framing"] == fr])
        check(f"F6 framing {fr} mean D", m, float(d.mean()), 0.0006, "gemma-2-2b-it-explicit-tier3-qwen3-100")
        check(f"F6 framing {fr} n", n, len(d), 0, "gemma-2-2b-it-explicit-tier3-qwen3-100")

    # Table 3 / Finding 7
    run = "gemma-4-E2B-it-our-sae-v1-150tok"
    recs = load_records(run)
    for lab, n, m, sd in (("comply", 59, 0.896, 0.001), ("refuse", 16, 0.249, 0.004)):
        d = np.array([r["divergence"] for r in recs if r["surface_label"] == lab])
        check(f"T3 {lab} n", n, len(d), 0, run)
        check(f"T3 {lab} mean D", m, float(d.mean()), 0.0006, run)
        check(f"T3 {lab} SD", sd, float(d.std(ddof=0)), 0.0011, run)
    check("T3 refuse labels that are placeholders", 0,
          sum(is_placeholder(r) for r in recs if r["surface_label"] == "refuse"), 0, run, "expected to fail (R1)")

    # Finding 2 and 4
    recs = load_records("gemma-4-E2B-it-L17-pass5")
    check("F2 Gemma 4 refusals with template", 65, sum(r["surface_label"] == "refuse" for r in recs), 0,
          "gemma-4-E2B-it-L17-pass5")
    check("F2 Gemma 4 refusals that are placeholders", 0, sum(map(is_placeholder, recs)), 0,
          "gemma-4-E2B-it-L17-pass5", "expected to fail (R1)")
    for run, ben, haz in (("llama-3.2-1b-it-cross-arch", 30, 91), ("qwen25-1b5-it-cross-arch", 83, 95),
                          ("phi3-mini-4k-it-cross-arch", 87, 95)):
        bt = by_tier(load_records(run))
        check(f"F4 {run} benign refuse %", ben, pct(bt["benign_bio"], "refuse"), 0.6, run)
        check(f"F4 {run} hazard refuse %", haz, pct(bt["hazard_adjacent_category"], "refuse"), 0.6, run)
        check(f"F4 {run} benign refuse placeholders", 0, sum(map(is_placeholder, bt["benign_bio"])), 0, run,
              "expected to fail (R1)")

    # T identification (needed to recompute D with new labels)
    t_ids = {}
    for run in ("gemma-2-2b-it-L12-tuned", "gemma-2-2b-it-L12-tuned-rejudged", "gemma-2-2b-it-150tok-rejudged",
                "gemma-2-2b-it-explicit-tier3-qwen3-100", "gemma-4-E2B-it-our-sae-v1-150tok",
                "control-legal-financial-gemma2"):
        name, err = identify_T(load_records(run))
        t_ids[run] = {"T": name, "max_abs_err": err}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "step0_reproduce.json").write_text(json.dumps({"checks": checks, "T_identification": t_ids}, indent=1))
    n_pass = sum(c["pass"] for c in checks)
    lines = ["# Step 0: reproduce arXiv v1 numbers from committed results (generated)", "",
             f"{n_pass}/{len(checks)} checks pass. Checks marked 'expected to fail' test the defects the",
             "audit found; a failure there confirms the defect.", "",
             "| check | paper | recomputed | pass | note |", "|---|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['paper']} | {c['recomputed']} | {'yes' if c['pass'] else 'NO'} | {c['note']} |")
    lines += ["", "## Calibration matrix T that reproduces stored D", "", "| run | T | max abs error |", "|---|---|---|"]
    for run, v in t_ids.items():
        lines.append(f"| {run} | {v['T']} | {v['max_abs_err']:.2e} |")
    (OUT / "step0_reproduce.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
