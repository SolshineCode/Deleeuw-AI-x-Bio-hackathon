"""Sampler for the 150-item blind human validation sheet, v2 (added 2026-10-23).

Supersedes the v1 composition in scripts/build_validation_sheet.py (drawn from the pilot,
before any judge votes existed). v2 is stratified by tier AND by how the two relabel judges
voted, so judge accuracy can be estimated separately where the judges agree and where they
disagree, then reweighted to the full manifest with the recorded stratum sizes.

Composition (fixed seed 20261023):
  * 128 relabel-manifest items (tiers 1-2 and controls). Tier quotas as in v1: benign_bio 44,
    dual_use_bio 44, psilocybin_control 14, legality_confound 14, legal control 12.
    Within each tier the quota is split across agreement strata:
      agree     both judges gave a usable label and it is the same        45%
      disagree  both judges gave a usable label and they differ (=tie)    40%
      single    exactly one judge gave a usable label                      15%
    Items with no usable label are not sampled (no judge label to score); their count is
    recorded. Shortfalls in a stratum move to the other strata of the same tier.
    Within a stratum, items are shuffled with the seed, then items where either judge said
    refuse, hedge or evade are moved to the front (rare classes; stable order otherwise),
    and picks rotate across run families.
  * 22 tier-3 (hazard_adjacent_category) records from the local Qwen3.6-35B replication,
    identified by record id only. Their text never enters the sheet: the row says
    VIEW-LOCALLY and is read with scripts/secure_view.py --key <v2 key>.

Outputs:
  --ids-out    JSON of item ids, strata and stratum sizes. No text. Safe to commit (ids are
               sha256 prefixes of tier-1/2 prompt+completion, and tier-3 record ids).
  --build-sheet (optional) also writes the sheet CSV + INSTRUCTIONS.md to --sheet-dir and
               the unblinding key to --key-dir. Refused while relabel coverage is PARTIAL
               (override: --allow-partial), refused if --sheet-dir is not git-ignored or
               outside git, and refused if --key-dir is inside a non-ignored git path. The
               sheet holds prompt/completion text and must never be committed or printed.

The script prints counts only.

Usage:
  python scripts/reanalysis/sample_validation.py --ids-out <json>
  python scripts/reanalysis/sample_validation.py --ids-out <json> --build-sheet \
      --sheet-dir ~/phd-by-publication/biorefusal_validation/v2 \
      --key-dir ~/chem-bench-qwen36-2026-08-12/results/biorefusalaudit/private/validation_2026-10/v2
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import secrets
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import relabel_openrouter_free as RL  # noqa: E402
from build_validation_sheet import INSTRUCTIONS as INSTRUCTIONS_V1  # noqa: E402

SEED = 20261023
TIER_QUOTA = {"benign_bio": 44, "dual_use_bio": 44, "psilocybin_control": 14,
              "legality_confound": 14, "legal": 12}
STRATUM_SHARE = (("agree", 0.45), ("disagree", 0.40), ("single", 0.15))
RARE = {"refuse", "hedge", "evade"}
N_TIER3 = 22
DEFAULT_QWEN36 = Path("/home/darkstar/chem-bench-qwen36-2026-08-12/results/biorefusalaudit/classified.jsonl")


def bucket(tier: str) -> str:
    return "legal" if tier.endswith("_legal") else tier


def latest_votes(relabel_dir: Path) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for ln in (relabel_dir / "votes.jsonl").read_text().splitlines():
        v = json.loads(ln)
        k = (v["item_id"], v["judge"])
        if k not in latest or v["status"] == "ok" or latest[k]["status"] != "ok":
            latest[k] = v  # same rule as RL.consolidate
    return latest


def agreement_class(labels: list[str | None]) -> str:
    usable = [x for x in labels if x]
    if len(usable) >= 2:
        return "agree" if len(set(usable)) == 1 else "disagree"
    return "single" if len(usable) == 1 else "none"


def split_quota(q: int) -> dict[str, int]:
    out = {s: int(round(q * f)) for s, f in STRATUM_SHARE}
    out["agree"] += q - sum(out.values())
    return out


def git_ignored_or_outside(path: Path) -> bool:
    path = path.expanduser().resolve()
    probe = path / "probe.csv"
    d = path
    while not d.exists():
        d = d.parent
    top = subprocess.run(["git", "-C", str(d), "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if top.returncode != 0:
        return True  # not inside any git work tree
    r = subprocess.run(["git", "-C", top.stdout.strip(), "check-ignore", "-q", str(probe)])
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--relabel-dir", default="results/relabel_2026-10")
    ap.add_argument("--qwen36-file", default=str(DEFAULT_QWEN36))
    ap.add_argument("--ids-out", required=True)
    ap.add_argument("--build-sheet", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--sheet-dir")
    ap.add_argument("--key-dir")
    args = ap.parse_args()

    rel = Path(args.relabel_dir)
    manifest = [json.loads(x) for x in (rel / "manifest.jsonl").read_text().splitlines()]
    latest = latest_votes(rel)
    judges = sorted({j for _, j in latest})
    n_both = sum(1 for m in manifest if all((m["item_id"], j) in latest for j in judges))
    complete = len(judges) >= 2 and n_both == len(manifest)
    status = "COMPLETE" if complete else "PARTIAL"

    rng = random.Random(SEED)
    strata: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for m in manifest:
        labs = [(latest.get((m["item_id"], j)) or {}).get("label") for j in judges]
        labs = [x if (latest.get((m["item_id"], j)) or {}).get("status") == "ok" else None
                for x, j in zip(labs, judges)]
        strata[bucket(m["tier"])][agreement_class(labs)].append(
            {"item_id": m["item_id"], "tier": m["tier"], "family": RL.run_family(m["occurrences"][0]["run"]),
             "rare": bool(RARE & {x for x in labs if x})})
    sizes = {b: {s: len(v) for s, v in sorted(d.items())} for b, d in sorted(strata.items())}

    chosen: list[dict] = []
    shortfalls = {}
    for b, q in TIER_QUOTA.items():
        want = split_quota(q)
        pools = {}
        for s, _ in STRATUM_SHARE:
            pool = list(strata[b].get(s, []))
            rng.shuffle(pool)
            pool.sort(key=lambda x: not x["rare"])  # stable: rare first, seeded order otherwise
            # rotate across run families
            fam: dict[str, list[dict]] = defaultdict(list)
            for x in pool:
                fam[x["family"]].append(x)
            order, keys = [], sorted(fam)
            while any(fam[k] for k in keys):
                for k in keys:
                    if fam[k]:
                        order.append(fam[k].pop(0))
            pools[s] = order
        got = {s: pools[s][:want[s]] for s in want}
        missing = q - sum(len(v) for v in got.values())
        for s, _ in STRATUM_SHARE:  # redistribute shortfalls within the tier
            while missing > 0 and len(got[s]) < len(pools[s]):
                got[s].append(pools[s][len(got[s])])
                missing -= 1
        if missing:
            shortfalls[b] = missing
        for s, xs in got.items():
            chosen += [{"item_id": x["item_id"], "tier": x["tier"], "stratum": s, "rare_label": x["rare"]} for x in xs]

    qd = json.loads(Path(args.qwen36_file).read_text())
    t3 = sorted(r["id"] for r in qd["records"] if r.get("tier") == "hazard_adjacent_category")
    rng.shuffle(t3)
    t3 = sorted(t3[:N_TIER3])

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "relabel_coverage": status, "seed": SEED, "judges": judges,
        "tier_quota": TIER_QUOTA, "stratum_share": dict(STRATUM_SHARE),
        "stratum_population": sizes, "shortfalls": shortfalls,
        "n_manifest_items": len(chosen), "n_tier3_view_locally": len(t3),
        "note": ("PARTIAL: agreement strata will change as the relabel finishes; regenerate after coverage "
                 "is complete. This list is a dry run." if not complete else "coverage complete"),
        "items": chosen,
        "tier3_view_locally": [{"record_id": r, "source": "qwen36_local", "file": str(Path(args.qwen36_file))}
                               for r in t3],
    }
    ip = Path(args.ids_out)
    ip.parent.mkdir(parents=True, exist_ok=True)
    ip.write_text(json.dumps(out, indent=1))
    by = Counter((c["tier"], c["stratum"]) for c in chosen)
    print(f"[{status}] sampled {len(chosen)} manifest items + {len(t3)} tier-3 VIEW-LOCALLY ids -> {ip}")
    print("by tier x stratum:", dict(sorted(by.items())))
    print("rare-label items sampled:", sum(c["rare_label"] for c in chosen), "; shortfalls:", shortfalls or "none")

    if not args.build_sheet:
        return 0
    if not complete and not args.allow_partial:
        print("refusing to build the sheet: relabel coverage is PARTIAL (use --allow-partial to override)",
              file=sys.stderr)
        return 3
    if not args.sheet_dir or not args.key_dir:
        print("--sheet-dir and --key-dir are required with --build-sheet", file=sys.stderr)
        return 2
    sd, kd = Path(args.sheet_dir).expanduser(), Path(args.key_dir).expanduser()
    if not git_ignored_or_outside(sd) or not git_ignored_or_outside(kd):
        print("refusing: sheet or key directory is inside a git work tree and not ignored", file=sys.stderr)
        return 4
    texts = RL.load_texts(Path(args.results_dir))
    rows, key = [], {}
    for c in chosen:
        sid = "V" + secrets.token_hex(3).upper()
        p, comp = texts[c["item_id"]]
        rows.append({"sheet_id": sid, "prompt": p, "completion": comp, "human_label": "", "notes": ""})
        key[sid] = {"source": "relabel_manifest", **c}
    for r in t3:
        sid = "V" + secrets.token_hex(3).upper()
        rows.append({"sheet_id": sid, "prompt": "VIEW-LOCALLY", "completion": "VIEW-LOCALLY",
                     "human_label": "", "notes": ""})
        key[sid] = {"source": "qwen36_local", "record_id": r, "tier": "hazard_adjacent_category",
                    "qwen36_file": str(Path(args.qwen36_file).resolve())}
    rng.shuffle(rows)
    sd.mkdir(parents=True, exist_ok=True)
    kd.mkdir(parents=True, exist_ok=True)
    kd.chmod(0o700)
    with (sd / "validation_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet_id", "prompt", "completion", "human_label", "notes"])
        w.writeheader()
        w.writerows(rows)
    kp = kd / "validation_key.json"
    instr = INSTRUCTIONS_V1.replace("python scripts/secure_view.py <sheet_id>",
                                    f"python scripts/secure_view.py <sheet_id> --key {kp}")
    (sd / "INSTRUCTIONS.md").write_text(instr, encoding="utf-8")
    kp.write_text(json.dumps(key, indent=1), encoding="utf-8")
    kp.chmod(0o600)
    print(f"sheet rows: {len(rows)} ({len(t3)} VIEW-LOCALLY) written to {sd}; key entries {len(key)} in {kd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
