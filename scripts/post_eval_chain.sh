#!/usr/bin/env bash
# Post-eval chain: when the primary activation-collection eval completes
# (marker: "Processed N/N prompts" in stderr.log), automatically run
# auto_tune_catalog → fit_calibration → re-eval with tuned catalog →
# intervention experiments on top cases.
#
# Usage:
#   bash scripts/post_eval_chain.sh <source_run_dir>
#   e.g., bash scripts/post_eval_chain.sh runs/gemma-2-2b-it-L12-activations

set -u
cd "$(dirname "$0")/.."

SRC="${1:-runs/gemma-2-2b-it-L12-activations}"
LOG="$SRC/stderr.log"
ACTS="$SRC/activations.npz"
REPORT="$SRC/report.json"

echo "[post_eval_chain] waiting for $LOG to show 'Processed' line ..."
while :; do
  if [[ -f "$LOG" ]] && grep -q "^\[biorefusalaudit\] Processed" "$LOG"; then
    break
  fi
  sleep 30
done
echo "[post_eval_chain] source eval complete: $SRC"

if [[ ! -f "$ACTS" ]]; then
  echo "ERROR: no activations.npz at $ACTS — source eval may have been run without --dump-activations" >&2
  exit 2
fi

echo "---- step A: auto_tune_catalog ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/auto_tune_catalog.py \
    --activations "$ACTS" \
    --existing data/feature_catalog/gemma-2-2b-it.json \
    --out data/feature_catalog/gemma-2-2b-it.json \
    --top-per-category 20 \
  2>&1 | tee "$SRC/auto_tune.log"

echo "---- step B: fit_calibration ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/fit_calibration.py \
    --report "$REPORT" \
    --config configs/calibration_gemma2_2b.yaml \
  2>&1 | tee "$SRC/fit_cal.log"

RE_OUT="runs/gemma-2-2b-it-L12-tuned"
mkdir -p "$RE_OUT"
echo "---- step C: re-eval with tuned catalog + fitted T ----"
PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out "$RE_OUT" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --quantize none \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --max-new-tokens 50 \
    --dump-activations \
  2>&1 | tee "$RE_OUT/stderr.log"

mkdir -p runs/interventions
echo "---- step D: intervention on top high-divergence prompts ----"
# Pick three prompts from the tuned report
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -c "
import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
from biorefusalaudit.runner.trace_selected_cases import select_cases
cases = select_cases('$RE_OUT/report.json', top_k_per_criterion=2)
# Print up to 4 unique prompt IDs + their categories-of-interest
seen = set()
for c in cases:
    if c.prompt_id in seen:
        continue
    # Map criterion → category to intervene on
    if c.criterion == 'false_refusal' or c.criterion == 'high_divergence':
        cat = 'refusal_circuitry'
    elif c.criterion == 'false_compliance':
        cat = 'hazard_adjacent'
    else:
        cat = 'refusal_circuitry'
    print(f'{c.prompt_id} {cat} {c.criterion}')
    seen.add(c.prompt_id)
    if len(seen) >= 4:
        break
" > "$RE_OUT/intervention_targets.txt"

while read -r PID CAT CRIT; do
  [[ -z "$PID" ]] && continue
  OUT="runs/interventions/${PID}_${CAT}.json"
  echo ">> intervention: $PID ($CRIT) → category $CAT"
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
    .venv/Scripts/python.exe -u scripts/run_intervention.py \
      --model google/gemma-2-2b-it \
      --prompt-id "$PID" \
      --category "$CAT" \
      --top-k 5 \
      --catalog data/feature_catalog/gemma-2-2b-it.json \
      --calibration configs/calibration_gemma2_2b.yaml \
      --max-new-tokens 60 \
      --out "$OUT" \
    2>&1 | tee "runs/interventions/${PID}_${CAT}.log" || echo "skipped $PID/$CAT"
done < "$RE_OUT/intervention_targets.txt"

echo "---- step E: regenerate scaling plot ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe scripts/build_scaling_plot.py \
    --runs-dir runs \
    --out demo/scaling_plot_real.png \
  2>&1 | tail -5

echo "==== post_eval_chain DONE at $(date -u +'%Y-%m-%dT%H:%M:%SZ') ===="
