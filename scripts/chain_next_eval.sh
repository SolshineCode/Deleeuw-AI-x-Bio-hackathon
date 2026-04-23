#!/usr/bin/env bash
# Detect when the current GPU eval finishes + kick the next one.
# Used to keep the local GPU continuously busy between runs.

set -u
cd "$(dirname "$0")/.."

CURRENT_LOG="${1:-runs/gemma-2-2b-it-gpu/stderr.log}"
NEXT_NAME="${2:-gemma-2-2b-it-L20-l0_71}"
NEXT_LAYER="${3:-20}"
NEXT_SAE_ID="${4:-layer_20/width_16k/average_l0_71}"

echo "Waiting for $CURRENT_LOG to show 'Processed' line or file to stabilize..."
while :; do
  if [[ -f "$CURRENT_LOG" ]] && grep -q "Processed" "$CURRENT_LOG"; then
    echo "Current eval completed — chaining next."
    break
  fi
  sleep 30
done

NEXT_OUT="runs/$NEXT_NAME"
mkdir -p "$NEXT_OUT"

echo "==== $(date) kicking next eval: $NEXT_NAME ($NEXT_SAE_ID) ===="
PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out "$NEXT_OUT" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "$NEXT_SAE_ID" \
    --layer "$NEXT_LAYER" \
    --quantize none \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
  2>&1 | tee "$NEXT_OUT/stderr.log"

echo "==== $(date) $NEXT_NAME done (exit $?) ===="
