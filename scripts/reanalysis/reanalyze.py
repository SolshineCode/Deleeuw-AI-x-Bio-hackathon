"""Re-analysis with relabelled surface labels (2026-10). CPU only.

Changelog:
  2026-10-22  first version (sections A-E), run on the 300-item pilot.
  2026-10-23  extended for the full relabel: coverage per judge, inter-judge agreement
              (raw, Cohen's kappa, confusion matrix), per run x tier rates for every class
              with k/n and Wilson 95% CIs, per-judge sensitivity table, and a diff table
              against the numbers printed in arXiv v1 (scripts/reanalysis/published_v1.py).
              Partial coverage is explicit: every cell reports how many records are
              labelled, labelled by one judge only, unjudged (tie or no usable vote), not
              yet voted, or not relabellable. Unjudged records are never imputed as any
              class. Output file names carry "_PARTIAL" whenever any manifest item lacks a
              vote from either judge. New --out-dir option. The old section A (original vs
              relabelled refuse/comply only) is replaced by section 4.

Inputs: committed results/*/report.json, results/relabel_2026-10/{manifest,votes}.jsonl,
public eval-set category fields (psilocybin / legality files only). No prompt or
completion text is read except through the relabel manifest's hashes; nothing textual is
written. Output is aggregates only (counts, rates, ids of runs).

Label policy (inherited from relabel_openrouter_free.consolidate): the latest vote per
(item, judge); an item with one usable vote takes that label; two usable votes that
disagree are a tie and the item is UNJUDGED; no usable vote is UNJUDGED.

Sections:
  0. Coverage summary (PARTIAL or COMPLETE)
  1. Coverage per judge
  2. Inter-judge agreement (agreement is not accuracy)
  3. Diff vs arXiv v1 printed numbers
  4. Rates per run x tier, every class, consensus labels
  5. Sensitivity: rates per run x tier from each judge alone
  B. Fate of the original placeholder refusals
  C. Variance decomposition of stored D (independent of the relabel)
  D. D recomputed with relabelled one-hot surface vectors
  E. Finding 7 test (Gemma 4 + author SAE, 150 tok)

Tier-3 (hazard_adjacent_category) records are not relabelled (completions redacted in every
committed run); they appear only as 'not checkable'.

Usage:
  python scripts/reanalysis/reanalyze.py --tag full
  python scripts/reanalysis/reanalyze.py --tag DRYRUN \
      --out-dir results/relabel_2026-10/reanalysis_DRYRUN_2026-10-24
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    REPO, RESULTS, SURFACE_CLASSES, candidate_Ts, divergence, identify_T, is_placeholder, r2_groups,
    r2_linear, wilson,
)
from published_v1 import CATEGORY_FILES, PUBLISHED  # noqa: E402
import relabel_openrouter_free as RL  # noqa: E402

T3 = "hazard_adjacent_category"
D_RUNS = (
    "gemma-2-2b-it-L12-tuned", "gemma-2-2b-it-150tok-rejudge", "gemma-2-2b-it-80tok-rejudge",
    "gemma-2-2b-it-explicit-benign-qwen3-100", "gemma-2-2b-it-explicit-dualuse-qwen3-100",
    "gemma-2-2b-it-explicit-tier3-qwen3-100", "gemma-4-E2B-it-our-sae-v1-150tok",
    "gemma-4-E2B-it-L17-pass5", "control-legal-financial-gemma2", "gemma-2-2b-it-bio-sae-v1-150tok",
    "gemma-4-E2B-it-bio-sae-v1-150tok",
)
# Runs behind the paper's behavioural claims; listed first in section 4.
HEADLINE_RUNS = (
    "gemma-2-2b-it-L12-tuned-rejudged", "gemma-4-E2B-it-L17-pass5", "gemma-4-E2B-it-our-sae-v1-150tok",
    "llama-3.2-1b-it-cross-arch", "qwen25-1b5-it-cross-arch", "phi3-mini-4k-it-cross-arch",
    "gemma-2-2b-it-explicit-benign-qwen3-100", "gemma-2-2b-it-explicit-dualuse-qwen3-100",
    "psilocybin-gemma2-sae-80tok", "psilocybin-gemma2-sae-150tok", "psilocybin-qwen25-1b5-80tok",
    "psilocybin-llama32-1b-80tok", "legality-gemma2-80tok", "legality-gemma2-sae-can-psi",
    "control-legal-financial-gemma2",
)
STATES = ("labelled", "unjudged_tie", "unjudged_failed", "not_yet", "not_relabellable")


# --------------------------------------------------------------------------- loading


def load_relabel(relabel_dir: Path):
    manifest = [json.loads(x) for x in (relabel_dir / "manifest.jsonl").read_text().splitlines()]
    final = RL.consolidate(relabel_dir) if (relabel_dir / "votes.jsonl").exists() else {}
    occ = {}
    for m in manifest:
        for o in m["occurrences"]:
            key = (o["run"], o["id"], o.get("condition"))
            if key in occ and occ[key] != m["item_id"]:
                raise SystemExit(f"occurrence key maps to two items: {key}")
            occ[key] = m["item_id"]
    latest: dict[tuple[str, str], dict] = {}
    for ln in (relabel_dir / "votes.jsonl").read_text().splitlines():
        v = json.loads(ln)
        k = (v["item_id"], v["judge"])
        if k not in latest or v["status"] == "ok" or latest[k]["status"] != "ok":
            latest[k] = v  # same rule as RL.consolidate
    return manifest, final, occ, latest


def item_state(final: dict, iid: str | None, min_judges: int) -> tuple[str, str | None, bool]:
    """(state, label, single_judge) for one manifest item."""
    if iid is None:
        return "not_relabellable", None, False
    f = final.get(iid)
    if not f or f["n_judges"] < min_judges:
        return "not_yet", None, False
    if f["label"]:
        return "labelled", f["label"], f["n_usable"] == 1
    reason = f.get("unjudged_reason") or ""
    return ("unjudged_tie" if reason.startswith("tie") else "unjudged_failed"), None, False


def categories() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in CATEGORY_FILES:
        for ln in (REPO / p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                out.setdefault(r["id"], r.get("category"))
    return out


# --------------------------------------------------------------------------- stats


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]"


def rate_obj(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "p": (k / n if n else None), "wilson95": [lo, hi]}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labs = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in labs) / (n * n)
    return None if pe >= 1 else (po - pe) / (1 - pe)


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


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relabel-dir", default=str(RESULTS / "relabel_2026-10"))
    ap.add_argument("--out-dir", default=str(RESULTS / "reanalysis_2026-10"))
    ap.add_argument("--tag", default="full")
    ap.add_argument("--min-judges", type=int, default=1,
                    help="an item counts only once this many judges have voted on it (a failed vote "
                         "counts as a vote). Default 1: use every label, and report single-judge counts.")
    ap.add_argument("--quiet", action="store_true", help="do not echo the markdown to stdout")
    args = ap.parse_args()

    rel = Path(args.relabel_dir)
    manifest, final, occ, latest = load_relabel(rel)
    judges = sorted({j for _, j in latest})
    n_items = len(manifest)
    n_both_voted = sum(1 for f in final.values() if f["n_judges"] >= 2)
    complete = n_both_voted == n_items and len(judges) >= 2
    status = "COMPLETE" if complete else "PARTIAL"
    gen = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cats = categories()

    # per-item state counts
    st_items = Counter(item_state(final, m["item_id"], args.min_judges)[0] for m in manifest)
    single_items = sum(1 for m in manifest if item_state(final, m["item_id"], args.min_judges)[2])
    final_labels = Counter(item_state(final, m["item_id"], args.min_judges)[1] or "-" for m in manifest)
    retryable = Counter(v["status"] for v in latest.values() if v["status"] != "ok")

    out: dict = {"generated_utc": gen, "status": status, "tag": args.tag, "min_judges": args.min_judges,
                 "judges": judges}
    md = [f"# Re-analysis [{status}] ({args.tag}), generated {gen}", ""]
    if not complete:
        md += ["**PARTIAL COVERAGE. Not for citation.** At least one manifest item lacks a vote from one of",
               "the judges. Every rate below covers only the records labelled so far; per-cell n is",
               "smaller than the published n, and the labelled subset is not a random sample (it is",
               "whatever the free judges have reached). Labels can still change: an item now labelled by",
               "one judge becomes UNJUDGED if the second judge disagrees.", ""]
    md += ["Label policy: latest vote per (item, judge); one usable vote gives that label; two usable votes",
           "that disagree are a tie and the item is UNJUDGED; no usable vote is UNJUDGED. Unjudged records",
           "are never counted as any class. Tier-3 records are not relabelled (completions redacted in",
           "every committed run). Not included: the 53 tier-1/2 completions from the local Qwen3.6-35B",
           "replication (not in results/, so not in the manifest).", ""]

    # 0. coverage summary
    out["coverage"] = {"manifest_items": n_items, "items_with_any_vote": len(final),
                       "items_with_both_judges_voted": n_both_voted, "item_states": dict(st_items),
                       "labelled_by_one_judge_only": single_items, "final_labels": dict(final_labels),
                       "latest_failed_vote_statuses": dict(retryable)}
    md += ["## 0. Coverage summary", "",
           f"- Manifest: {n_items} unique non-tier-3 completions. Items with any vote: {len(final)}; "
           f"with votes from both judges: {n_both_voted}.",
           f"- Item states: " + ", ".join(f"{s} {st_items.get(s, 0)}" for s in STATES[:4]) + ".",
           f"- Labelled items resting on one usable vote only: {single_items}.",
           f"- Final labels: " + ", ".join(f"{k} {v}" for k, v in sorted(final_labels.items())) + " ('-' = no label).",
           f"- Latest vote is a failure, by status (retryable unless noted): {dict(retryable)}.", ""]

    # 1. coverage per judge
    md += ["## 1. Coverage per judge", "",
           "| judge | items voted | usable labels | failed (latest) | label counts |", "|---|---|---|---|---|"]
    J = {}
    for j in judges:
        vs = [v for (iid, jj), v in latest.items() if jj == j]
        ok = [v for v in vs if v["status"] == "ok" and v["label"]]
        lc = Counter(v["label"] for v in ok)
        J[j] = {"voted": len(vs), "usable": len(ok), "failed": len(vs) - len(ok),
                "failed_by_status": dict(Counter(v["status"] for v in vs if not (v["status"] == "ok" and v["label"]))),
                "labels": dict(lc), "of_manifest": n_items}
        md.append(f"| {j} | {len(vs)}/{n_items} | {len(ok)}/{n_items} | {len(vs) - len(ok)} | "
                  + ", ".join(f"{k} {v}" for k, v in lc.most_common()) + " |")
    out["judges_coverage"] = J

    # 2. agreement
    md += ["", "## 2. Inter-judge agreement (agreement is not accuracy)", "",
           "Agreement between two LLM judges says nothing about whether either is right; judge accuracy",
           "comes only from the blind human validation sheet.", ""]
    AG = {}
    if len(judges) >= 2:
        ja, jb = judges[0], judges[1]
        pairs = []
        for m in manifest:
            a, b = latest.get((m["item_id"], ja)), latest.get((m["item_id"], jb))
            if a and b and a["status"] == "ok" and b["status"] == "ok" and a["label"] and b["label"]:
                pairs.append((a["label"], b["label"]))
        k_agree = sum(x == y for x, y in pairs)
        bin_pairs = [("refuse" if x == "refuse" else "other", "refuse" if y == "refuse" else "other") for x, y in pairs]
        cmp_pairs = [("comply" if x == "comply" else "other", "comply" if y == "comply" else "other") for x, y in pairs]
        conf = Counter(pairs)
        AG = {"judge_a": ja, "judge_b": jb, "both_usable": len(pairs), "agree": rate_obj(k_agree, len(pairs)),
              "kappa": cohen_kappa(pairs), "kappa_comply_vs_other": cohen_kappa(cmp_pairs),
              "agree_refuse_vs_other": rate_obj(sum(x == y for x, y in bin_pairs), len(bin_pairs)),
              "confusion": {f"{x}|{y}": c for (x, y), c in sorted(conf.items())}}
        kap = AG["kappa"]
        kc = AG["kappa_comply_vs_other"]
        md += [f"- Judges: A = {ja}, B = {jb}. Both gave a usable label on {len(pairs)} items.",
               f"- Exact agreement: {fmt_rate(k_agree, len(pairs))}. Cohen's kappa (all classes): "
               f"{'n/a' if kap is None else f'{kap:.3f}'}; comply-vs-other kappa: {'n/a' if kc is None else f'{kc:.3f}'}.",
               f"- Refuse-vs-other agreement: {fmt_rate(AG['agree_refuse_vs_other']['k'], len(pairs))}.",
               "", "Confusion matrix (rows A, columns B):", "",
               "| A \\ B | " + " | ".join(SURFACE_CLASSES) + " |", "|---" * (len(SURFACE_CLASSES) + 1) + "|"]
        for x in SURFACE_CLASSES:
            md.append(f"| {x} | " + " | ".join(str(conf.get((x, y), 0)) for y in SURFACE_CLASSES) + " |")
    out["agreement"] = AG

    # per-record table: run, tier -> list of (record, state, label, single, per-judge labels)
    run_model: dict[str, str] = {}
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    runs_in_manifest = sorted({o["run"] for m in manifest for o in m["occurrences"]})
    report_runs = sorted(p.parent.name for p in RESULTS.glob("*/report.json")
                         if not p.parent.name.startswith(RL.SKIP_RUN_PREFIXES))
    for run in report_runs:
        data = json.loads((RESULTS / run / "report.json").read_text(encoding="utf-8"))
        run_model[run] = data.get("model_name") or data.get("model") or ""
        for r in data.get("records", []):
            iid = occ.get((run, r.get("id"), r.get("condition")))
            state, lab, single = item_state(final, iid, args.min_judges)
            jl = {j: (latest.get((iid, j)) or {}).get("label") if iid else None for j in judges}
            cells[(run, r.get("tier", ""))].append({
                "rec": r, "iid": iid, "state": state, "label": lab, "single": single, "judge_labels": jl,
                "category": cats.get(r.get("id")),
            })

    # sanity: every occurrence in the manifest is matched by exactly one record
    n_occ = sum(len(m["occurrences"]) for m in manifest)
    n_matched = sum(1 for v in cells.values() for x in v if x["iid"])
    out["sanity"] = {"manifest_occurrences": n_occ, "records_matched_to_manifest": n_matched,
                     "runs_in_manifest": len(runs_in_manifest)}
    if n_occ != n_matched:
        md += ["", f"**SANITY FAILURE:** {n_occ} manifest occurrences but {n_matched} records matched.", ""]

    def select(run, tier, category=None):
        if tier == "tiers_1_2":
            xs = [x for (rr, t), v in cells.items() if rr == run and t != T3 for x in v]
        else:
            xs = list(cells.get((run, tier), []))
        if category:
            xs = [x for x in xs if x["category"] == category]
        return xs

    # 3. diff vs published
    md += ["", "## 3. Diff vs numbers printed in arXiv v1", "",
           "orig = the run's original labels on the same records (checks the cell reproduces). relab = consensus",
           "relabel over labelled records only, Wilson 95% CI. Delta = relab % minus published % (percentage points),",
           "shown only when at least one record is labelled. Tier-3 cells cannot be checked.", "",
           "| claim | published % | orig (reproduces?) | relab | unjudged | not yet voted | single-judge | delta pp | note |",
           "|---|---|---|---|---|---|---|---|---|"]
    DIFF = []
    for c in PUBLISHED:
        row = dict(c)
        if c["tier"] == T3:
            row.update({"checkable": False})
            DIFF.append(row)
            md.append(f"| {c['claim']} | {c['published']} | - | not checkable (tier-3 completions redacted) | | | | | {c.get('note', '')} |")
            continue
        xs = select(c["run"], c["tier"], c.get("category"))
        n = len(xs)
        ko = sum(x["rec"].get("surface_label") == c["label"] for x in xs)
        lab = [x for x in xs if x["state"] == "labelled"]
        kl = sum(x["label"] == c["label"] for x in lab)
        orig_pct = 100 * ko / n if n else None
        # paper prints whole percentages; same 0.6 pp tolerance as step0_reproduce.py
        repro = orig_pct is not None and abs(orig_pct - c["published"]) <= 0.6
        sc = Counter(x["state"] for x in xs)
        delta = (100 * kl / len(lab) - c["published"]) if lab else None
        row.update({"checkable": True, "n_records": n, "orig": rate_obj(ko, n), "orig_reproduces": bool(repro),
                    "relab": rate_obj(kl, len(lab)), "states": dict(sc),
                    "single_judge": sum(x["single"] for x in lab), "delta_pp": delta})
        DIFF.append(row)
        md.append(f"| {c['claim']} | {c['published']} | {ko}/{n} = {orig_pct:.0f}% ({'yes' if repro else 'NO'}) | "
                  f"{fmt_rate(kl, len(lab))} | {sc.get('unjudged_tie', 0) + sc.get('unjudged_failed', 0)} | "
                  f"{sc.get('not_yet', 0)} | {row['single_judge']} | "
                  f"{'' if delta is None else f'{delta:+.0f}'} | {c.get('note', '')} |")
    out["diff_vs_published"] = DIFF

    # 4. full rate table
    md += ["", "## 4. Rates per run x tier, every class (consensus labels)", "",
           "n = records in the cell; lab = labelled; 1j = of those, labelled by one judge only; unj = unjudged",
           "(tie / no usable vote); wait = not yet voted; n/r = not relabellable (empty or redacted completion).",
           "Class rates are k/lab with Wilson 95% CI. orig refuse = original label count, for reference.", "",
           "| run | model | tier | n | lab | 1j | unj (tie/fail) | wait | n/r | orig refuse | "
           + " | ".join(SURFACE_CLASSES) + " |",
           "|---" * (10 + len(SURFACE_CLASSES)) + "|"]
    order = [r for r in HEADLINE_RUNS if r in report_runs] + [r for r in report_runs if r not in HEADLINE_RUNS]
    RATES = []
    for run in order:
        for (rr, t), xs in sorted(cells.items()):
            if rr != run or t == T3 or not any(x["iid"] for x in xs):
                continue
            lab = [x for x in xs if x["state"] == "labelled"]
            sc = Counter(x["state"] for x in xs)
            lc = Counter(x["label"] for x in lab)
            row = {"run": run, "model": run_model.get(run, ""), "tier": t, "n": len(xs), "labelled": len(lab),
                   "single_judge": sum(x["single"] for x in lab), "states": dict(sc),
                   "orig_refuse": sum(x["rec"].get("surface_label") == "refuse" for x in xs),
                   "orig_placeholder_refuse": sum(is_placeholder(x["rec"]) for x in xs),
                   "rates": {c: rate_obj(lc.get(c, 0), len(lab)) for c in SURFACE_CLASSES}}
            RATES.append(row)
            md.append(f"| {run} | {row['model']} | {t} | {len(xs)} | {len(lab)} | {row['single_judge']} | "
                      f"{sc.get('unjudged_tie', 0)}/{sc.get('unjudged_failed', 0)} | {sc.get('not_yet', 0)} | "
                      f"{sc.get('not_relabellable', 0)} | {row['orig_refuse']} | "
                      + " | ".join(fmt_rate(lc.get(c, 0), len(lab)) for c in SURFACE_CLASSES) + " |")
        t3 = cells.get((run, T3))
        if t3 and run in HEADLINE_RUNS:
            md.append(f"| {run} | {run_model.get(run, '')} | {T3} | {len(t3)} | not checkable (redacted) |"
                      + " |" * (5 + len(SURFACE_CLASSES)))
    out["rates"] = RATES

    # 5. per-judge sensitivity
    md += ["", "## 5. Sensitivity: each judge alone (headline runs)", "",
           "Same cells, labelled by one judge only (its latest usable vote), ignoring the other judge. Shows",
           "how much the consensus rule (ties become unjudged) moves the rates.", "",
           "| run | tier | judge | usable | refuse | comply | partial+hedge+evade |", "|---|---|---|---|---|---|---|"]
    SENS = []
    for run in [r for r in HEADLINE_RUNS if r in report_runs]:
        for (rr, t), xs in sorted(cells.items()):
            if rr != run or t == T3 or not any(x["iid"] for x in xs):
                continue
            for j in judges:
                ls = [x["judge_labels"][j] for x in xs if x["judge_labels"].get(j)]
                lc = Counter(ls)
                other = lc["partial"] + lc["hedge"] + lc["evade"]
                SENS.append({"run": run, "tier": t, "judge": j, "usable": len(ls),
                             "refuse": rate_obj(lc["refuse"], len(ls)), "comply": rate_obj(lc["comply"], len(ls)),
                             "partial_hedge_evade": rate_obj(other, len(ls))})
                md.append(f"| {run} | {t} | {j.split('/')[-1]} | {len(ls)}/{len(xs)} | {fmt_rate(lc['refuse'], len(ls))} | "
                          f"{fmt_rate(lc['comply'], len(ls))} | {fmt_rate(other, len(ls))} |")
    out["sensitivity_per_judge"] = SENS

    # B. placeholder fate
    fate = Counter()
    for xs in cells.values():
        for x in xs:
            if x["iid"] and is_placeholder(x["rec"]):
                fate[x["label"] or x["state"]] += 1
    out["B_placeholder_fate"] = dict(fate)
    md += ["", "## B. What the relabel says about original placeholder refusals (relabellable records)", "",
           "| relabelled as / state | records |", "|---|---|"] + [f"| {k} | {v} |" for k, v in fate.most_common()]

    # C. variance decomposition (no relabel needed)
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
        rb, rf = r2_groups(y, labs), r2_linear(y, F)
        row = {"run": run, "n": len(recs), "r2_label": rb, "r2_f": rf, "within": 1 - rb,
               "n_placeholder": sum(map(is_placeholder, recs))}
        C.append(row)
        md.append(f"| {run} | {len(recs)} | {rb:.3f} | {rf:.3f} | {1-rb:.3f} | {row['n_placeholder']} |")
    out["C_variance"] = C

    # D. recompute D with relabelled labels
    md += ["", "## D. D recomputed with relabelled one-hot surface vectors (labelled records only)", "",
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
        for x in cells.get((run, "benign_bio"), []) + cells.get((run, "dual_use_bio"), []) + [
                y for (rr, t), v in cells.items() if rr == run and t not in ("benign_bio", "dual_use_bio", T3) for y in v]:
            r = x["rec"]
            if x["state"] != "labelled" or r.get("divergence") is None:
                continue
            s = np.array([1.0 if c == x["label"] else 0.0 for c in SURFACE_CLASSES])
            by_t[r["tier"]][0].append(r["divergence"])
            by_t[r["tier"]][1].append(divergence(s, np.asarray(r["feature_vec"], float), T))
        for t, (old, new) in sorted(by_t.items()):
            Dres.append({"run": run, "T": tname, "tier": t, "n": len(old),
                         "mean_D_stored": float(np.mean(old)), "mean_D_relabelled": float(np.mean(new))})
            md.append(f"| {run} | {tname} | {t} | {len(old)} | {np.mean(old):.3f} | {np.mean(new):.3f} |")
    out["D_recomputed"] = Dres

    # E. Finding 7
    run = "gemma-4-E2B-it-our-sae-v1-150tok"
    recs = json.loads((RESULTS / run / "report.json").read_text())["records"]
    F = np.array([r["feature_vec"] for r in recs], dtype=float)
    orig = [r["surface_label"] for r in recs]
    y = np.array([r["divergence"] for r in recs])
    new = [item_state(final, occ.get((run, r["id"], r.get("condition"))), args.min_judges)[1] for r in recs]
    sel = [i for i, x in enumerate(new) if x]
    newc = Counter(new[i] for i in sel)
    groups = ["comply" if new[i] == "comply" else "not_comply" for i in sel]
    p_new = perm_test_f(F[sel], groups) if sel else None
    p_orig = perm_test_f(F, ["comply" if x == "comply" else "not_comply" for x in orig])
    E = {"n": len(recs), "orig_counts": dict(Counter(orig)), "orig_placeholder_refuse": sum(map(is_placeholder, recs)),
         "r2_D_on_orig_label": r2_groups(y, orig), "f_component_sd": [float(x) for x in F.std(0)],
         "relabelled_n": len(sel), "relabelled_counts": dict(newc),
         "perm_p_f_by_orig_label": p_orig, "perm_p_f_by_relabel": p_new}
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

    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    stem = f"reanalysis_{args.tag}" + ("" if complete else "_PARTIAL")
    (o / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str))
    (o / f"{stem}.md").write_text("\n".join(md) + "\n")
    if not args.quiet:
        print("\n".join(md))
    print(f"\nwrote {o / stem}.{{md,json}} [{status}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
