"""Dose-proportionality runs for top-10 highest-effect prompts.

Re-runs the top-10 prompts by intervention_effect_size at boost=1.5, 2.0, and 4.0
(the main batch already produced 3.0x results). This gives a 4-point dose-response
curve per prompt, enabling proper CMF validation beyond the single-multiplier result.

Output: runs/interventions/<id>_refusal_circuitry_boost<X>.json
        (e.g. bio_014_refusal_circuitry_boost1.5.json)

Usage:
    python scripts/run_dose_proportionality.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
EVAL_SET = REPO / "data/eval_set_public/eval_set_public_v1.jsonl"
OUT_DIR = REPO / "runs/interventions"
SCRIPT = REPO / "scripts/run_intervention.py"

TOP10_IDS = [
    "bio_014", "bio_017", "bio_019", "bio_007", "bio_070",
    "bio_040", "bio_003", "bio_005", "bio_020", "bio_011",
]

BOOST_VALUES = [1.5, 2.0, 4.0]


def run_one(prompt_id: str, boost: float) -> bool:
    boost_tag = f"boost{boost:.1f}".rstrip("0").rstrip(".")
    out_file = OUT_DIR / f"{prompt_id}_refusal_circuitry_{boost_tag}.json"
    log_file = OUT_DIR / f"{prompt_id}_refusal_circuitry_{boost_tag}.log"

    if out_file.exists():
        print(f"[dose] SKIP {prompt_id} boost={boost} — output exists", flush=True)
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
        "--boost", str(boost),
        "--out", str(out_file),
    ]

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONPATH": str(REPO)}

    print(f"[dose] START {prompt_id} boost={boost} -> {out_file.name}", flush=True)
    t0 = time.time()
    with log_file.open("w", encoding="utf-8") as lf:
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    if result.returncode == 0 and out_file.exists():
        data = json.loads(out_file.read_text(encoding="utf-8"))
        nc = data["intervention_summary"]["qualifies_as_named_circuit"]
        eff = data["intervention_summary"]["intervention_effect_size"]
        print(f"[dose] DONE  {prompt_id} boost={boost}  NC={nc}  eff={eff:.3f}  ({elapsed:.0f}s)", flush=True)
        return True
    else:
        print(f"[dose] FAIL  {prompt_id} boost={boost}  rc={result.returncode}  ({elapsed:.0f}s)", flush=True)
        return False


def main() -> int:
    total = len(TOP10_IDS) * len(BOOST_VALUES)
    print(f"[dose] Dose-proportionality: {len(TOP10_IDS)} prompts × {len(BOOST_VALUES)} boosts = {total} runs", flush=True)
    print(f"[dose] Boost values: {BOOST_VALUES}", flush=True)
    print(f"[dose] Prompt IDs: {TOP10_IDS}", flush=True)

    ok = fail = skip = 0
    for prompt_id in TOP10_IDS:
        for boost in BOOST_VALUES:
            success = run_one(prompt_id, boost)
            if success:
                out_tag = f"boost{boost:.1f}".rstrip("0").rstrip(".")
                if (OUT_DIR / f"{prompt_id}_refusal_circuitry_{out_tag}.json").exists():
                    ok += 1
                else:
                    skip += 1
            else:
                fail += 1

    print(f"\n[dose] COMPLETE  ok={ok}  fail={fail}  skip={skip}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
