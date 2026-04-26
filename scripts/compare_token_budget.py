"""
Compare surface label distributions between 200-token (corrected) and 80-token runs.
Shows how token budget interacts with tier and framing on surface classification.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/compare_token_budget.py \
        --report-200 runs/gemma-2-2b-it-L12-tuned-rejudged/report.json \
        --report-80  runs/gemma-2-2b-it-80tok-rejudge/report.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from biorefusalaudit.judges.regex_classifier import regex_classify

UNIFORM = {"refuse": 0.2, "comply": 0.2, "partial": 0.2, "hedge": 0.2, "evade": 0.2}

def is_degenerate(soft):
    return len(soft) == 5 and all(abs(v - 0.2) < 1e-6 for v in soft.values())

def rejudge(records):
    out = []
    for rec in records:
        rec = dict(rec)
        verdict = regex_classify(rec.get("completion", ""))
        if verdict.label is not None:
            rec["surface_label"] = verdict.label
        elif is_degenerate(rec.get("surface_soft", {})):
            rec["surface_label"] = "hedge"  # ambiguous truncated completions default hedge
        out.append(rec)
    return out

def dist(records, key="surface_label"):
    counts = defaultdict(int)
    for r in records:
        counts[r[key]] += 1
    n = len(records)
    return {k: (v, 100*v/n) for k, v in sorted(counts.items(), key=lambda x: -x[1])}

def print_breakdown(records_200, records_80, groupby):
    groups = sorted({r.get(groupby, "?") for r in records_200})
    print(f"\n{'Tier/framing':<30} {'200tok comply%':>14} {'200tok hedge%':>13} {'200tok refuse%':>14} | {'80tok comply%':>13} {'80tok hedge%':>12} {'80tok refuse%':>13}")
    print("-" * 115)
    for g in groups:
        r200 = [r for r in records_200 if r.get(groupby) == g]
        r80 = [r for r in records_80 if r.get(groupby) == g]
        n200, n80 = len(r200), len(r80)
        def pct(recs, label):
            return 100*sum(1 for r in recs if r.get("surface_label") == label)/max(len(recs),1)
        print(f"{g:<30} {pct(r200,'comply'):>14.1f} {pct(r200,'hedge'):>13.1f} {pct(r200,'refuse'):>14.1f} | {pct(r80,'comply'):>13.1f} {pct(r80,'hedge'):>12.1f} {pct(r80,'refuse'):>13.1f}  (n={n200}/{n80})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-200", required=True, help="200-token corrected report")
    ap.add_argument("--report-80", required=True, help="80-token report (may need re-judging)")
    ap.add_argument("--rejudge-80", action="store_true", help="Apply regex re-judge to 80-token report")
    args = ap.parse_args()

    data200 = json.loads(Path(args.report_200).read_text(encoding="utf-8"))
    recs200 = data200.get("records", data200) if isinstance(data200, dict) else data200

    data80 = json.loads(Path(args.report_80).read_text(encoding="utf-8"))
    recs80 = data80.get("records", data80) if isinstance(data80, dict) else data80

    if args.rejudge_80:
        recs80 = rejudge(recs80)

    print(f"Reports: 200-tok n={len(recs200)}, 80-tok n={len(recs80)}")

    print("\n=== OVERALL DISTRIBUTION ===")
    d200 = dist(recs200)
    d80 = dist(recs80)
    labels = sorted(set(list(d200) + list(d80)))
    print(f"{'Label':<12} {'200tok count':>12} {'200tok %':>10} | {'80tok count':>11} {'80tok %':>9}")
    for lbl in labels:
        c200, p200 = d200.get(lbl, (0, 0.0))
        c80, p80 = d80.get(lbl, (0, 0.0))
        print(f"{lbl:<12} {c200:>12} {p200:>10.1f} | {c80:>11} {p80:>9.1f}")

    print("\n=== BY TIER ===")
    print_breakdown(recs200, recs80, "tier")

    print("\n=== BY FRAMING ===")
    print_breakdown(recs200, recs80, "framing")

    # Per-prompt comparison: label changed between 200 and 80 tokens
    id_to_200 = {r["id"]: r for r in recs200 if "id" in r}
    id_to_80 = {r["id"]: r for r in recs80 if "id" in r}
    common = sorted(set(id_to_200) & set(id_to_80))
    if common:
        print(f"\n=== PER-PROMPT LABEL CHANGES (200tok → 80tok, n={len(common)} matched) ===")
        changes = [(pid, id_to_200[pid]["surface_label"], id_to_80[pid]["surface_label"],
                    id_to_200[pid].get("tier","?"), id_to_200[pid].get("framing","?"))
                   for pid in common
                   if id_to_200[pid]["surface_label"] != id_to_80[pid]["surface_label"]]
        if changes:
            print(f"Changed: {len(changes)}/{len(common)}")
            print(f"{'ID':<12} {'200tok':>8} {'80tok':>8} {'tier':<30} {'framing'}")
            for pid, l200, l80, tier, framing in sorted(changes):
                print(f"{pid:<12} {l200:>8} {l80:>8} {tier:<30} {framing}")
        else:
            print("No label changes between 200-tok and 80-tok.")

if __name__ == "__main__":
    main()
