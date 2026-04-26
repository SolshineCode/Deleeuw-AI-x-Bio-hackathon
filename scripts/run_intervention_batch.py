"""Batch intervention runner — iterates over pending prompt IDs in tier-balanced order.

Usage:
    python scripts/run_intervention_batch.py [--max N] [--dry-run]

Fixes the --out directory bug from the ad-hoc batch scripts: always passes the
full file path  runs/interventions/<id>_refusal_circuitry.json  to run_intervention.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

REPO = Path(__file__).parent.parent
EVAL_SET = REPO / "data/eval_set_public/eval_set_public_v1.jsonl"
OUT_DIR = REPO / "runs/interventions"
LOG_DIR = OUT_DIR
SCRIPT = REPO / "scripts/run_intervention.py"

DONE = {
    "bio_001","bio_002","bio_004","bio_010","bio_016","bio_021",
    "bio_024","bio_025","bio_026","bio_027","bio_028","bio_029",
    "bio_030","bio_031","bio_032","bio_054","bio_055","bio_060",
    "bio_066","bio_069","bio_074",
}


def load_pending() -> list[dict]:
    rows = [json.loads(l) for l in EVAL_SET.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["id"] not in DONE]


def interleaved_order(pending: list[dict]) -> list[str]:
    tiers: dict[str, list[str]] = {}
    for r in pending:
        tiers.setdefault(r["tier"], []).append(r["id"])
    # hazard_adjacent first (fewest done), then dual_use, then benign
    pools = [
        tiers.get("hazard_adjacent_category", []),
        tiers.get("dual_use_bio", []),
        tiers.get("benign_bio", []),
    ]
    ordered: list[str] = []
    for group in zip_longest(*pools):
        ordered.extend(x for x in group if x)
    return ordered


def run_one(prompt_id: str, dry_run: bool) -> bool:
    out_file = OUT_DIR / f"{prompt_id}_refusal_circuitry.json"
    log_file = LOG_DIR / f"{prompt_id}_refusal_circuitry.log"

    if out_file.exists():
        print(f"[batch] SKIP {prompt_id} — output already exists", flush=True)
        return True

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPT),
        "--model", "google/gemma-2-2b-it",
        "--prompt-id", prompt_id,
        "--eval-set", str(EVAL_SET),
        "--catalog", str(REPO / "data/feature_catalog/gemma-2-2b-it.json"),
        "--category", "refusal_circuitry",
        "--top-k", "5",
        "--out", str(out_file),
    ]

    env_prefix = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONPATH": str(REPO)}
    import os
    env = {**os.environ, **env_prefix}

    print(f"[batch] START {prompt_id} -> {out_file.name}", flush=True)
    if dry_run:
        print(f"[batch] DRY-RUN cmd: {' '.join(cmd)}", flush=True)
        return True

    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if result.returncode == 0 and out_file.exists():
        data = json.loads(out_file.read_text(encoding="utf-8"))
        nc = data["intervention_summary"]["qualifies_as_named_circuit"]
        eff = data["intervention_summary"]["intervention_effect_size"]
        print(f"[batch] DONE  {prompt_id}  NC={nc}  effect={eff:.3f}  ({elapsed:.0f}s)", flush=True)
        return True
    else:
        print(f"[batch] FAIL  {prompt_id}  rc={result.returncode}  ({elapsed:.0f}s)  see {log_file.name}", flush=True)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=None, help="Max prompts to run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pending = load_pending()
    ordered = interleaved_order(pending)
    if args.max:
        ordered = ordered[: args.max]

    print(f"[batch] {len(ordered)} prompts queued", flush=True)

    ok = fail = skip = 0
    for pid in ordered:
        success = run_one(pid, dry_run=args.dry_run)
        if success:
            out_file = OUT_DIR / f"{pid}_refusal_circuitry.json"
            if out_file.exists():
                ok += 1
            else:
                skip += 1
        else:
            fail += 1

    print(f"\n[batch] COMPLETE  ok={ok}  fail={fail}  skip={skip}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
