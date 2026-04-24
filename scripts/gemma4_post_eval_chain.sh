#!/usr/bin/env bash
# Gemma 4 E2B post-eval chain.
# Waits for the activation-collection run to finish, then:
#   A. auto-tune the Gemma 4 feature catalog from real activations
#   B. fit calibration T
#   C. pass-2 eval with tuned catalog + calibrated T
#
# Usage:
#   bash scripts/gemma4_post_eval_chain.sh [source_run_dir]
#   Default source: runs/gemma-4-E2B-it-L17-activations

set -u
cd "$(dirname "$0")/.."

SAE_PATH="C:\Users\caleb\deception-nanochat-sae-research\experiments\gf1_behavior_sae\saes\gemma4_e2b_gf1_behaviorSAE_topk_k16_L17_bnb4bit.pt"
SRC="${1:-runs/gemma-4-E2B-it-L17-activations}"
LOG="$SRC/stderr.log"
ACTS="$SRC/activations.npz"
REPORT="$SRC/report.json"
CATALOG="data/feature_catalog/gemma-4-E2B-it.json"
CALIBRATION="configs/calibration_gemma4_e2b.yaml"
TUNED_OUT="runs/gemma-4-E2B-it-L17-tuned"

echo "[gemma4_chain] waiting for $LOG to show 'Processed' line ..."
while :; do
  if [[ -f "$LOG" ]] && grep -q "^\[biorefusalaudit\] Processed" "$LOG"; then
    break
  fi
  sleep 30
done
echo "[gemma4_chain] source eval complete."

if [[ ! -f "$ACTS" ]]; then
  echo "ERROR: no activations.npz at $ACTS" >&2; exit 2
fi

echo "---- A: auto_tune_catalog ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/auto_tune_catalog.py \
    --activations "$ACTS" \
    --existing "$CATALOG" \
    --out "$CATALOG" \
    --top-per-category 20 \
  2>&1 | tee "$SRC/auto_tune.log"

echo "---- B: fit_calibration ----"
# Note: fit_calibration.py takes --report and --config only; no --catalog argument.
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/fit_calibration.py \
    --report "$REPORT" \
    --config "$CALIBRATION" \
  2>&1 | tee "$SRC/fit_calibration.log"

echo "---- C: pass-2 eval (tuned catalog + calibrated T, --dump-activations) ----"
# --dump-activations writes activations.npz to TUNED_OUT so a subsequent
# auto_tune_catalog pass can refine the catalog further using 4-bit activations
# (the pass-1 activations come from the CPU/fp16 run and may have distribution
# shift relative to bitsandbytes 4-bit inference).
mkdir -p "$TUNED_OUT"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -m biorefusalaudit.cli run \
    --model google/gemma-4-E2B-it \
    --quantize 4bit \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out "$TUNED_OUT" \
    --sae-source custom \
    --sae-release "$SAE_PATH" \
    --layer 17 \
    --architecture topk \
    --k 16 \
    --d-model 1536 \
    --d-sae 6144 \
    --catalog "$CATALOG" \
    --calibration "$CALIBRATION" \
    --max-new-tokens 200 \
    --dump-activations \
  2>&1 | tee "$TUNED_OUT/stderr.log"

echo "[gemma4_chain] done. Results in $TUNED_OUT/report.md"
