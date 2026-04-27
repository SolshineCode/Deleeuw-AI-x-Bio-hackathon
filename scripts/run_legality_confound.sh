#!/usr/bin/env bash
# Run legality confound suite across 3 models (NullSAE for speed).
# Eval set: data/eval_set_public/eval_set_legality_confound_v1.jsonl (51 prompts)
# Models: Gemma 2 2B-IT, Qwen 2.5 1.5B, Llama 3.2 1B
# NullSAE (--sae-source none) returns zero feature vec; D=1.0 always.
# This run is behavioral-label-only — surface posture across Schedule I compounds.
#
# Run from repo root:
#   KMP_DUPLICATE_LIB_OK=TRUE bash scripts/run_legality_confound.sh 2>&1 | tee runs/legality-confound.log

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
EVAL_SET="data/eval_set_public/eval_set_legality_confound_v1.jsonl"

echo "=== Legality Confound Suite START $(date) ==="
echo "Eval set: $EVAL_SET ($(wc -l < $EVAL_SET) prompts)"

# --- Gemma 2 2B-IT (80-tok, NullSAE) ---
echo ""
echo "--- Gemma 2 2B-IT 80-tok NullSAE $(date) ---"
python -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set "$EVAL_SET" \
    --out runs/legality-gemma2-80tok \
    --sae-source none \
    --max-new-tokens 80

echo "Gemma 2 80-tok done $(date)"

# --- Gemma 2 2B-IT (150-tok, NullSAE) ---
echo ""
echo "--- Gemma 2 2B-IT 150-tok NullSAE $(date) ---"
python -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set "$EVAL_SET" \
    --out runs/legality-gemma2-150tok \
    --sae-source none \
    --max-new-tokens 150

echo "Gemma 2 150-tok done $(date)"

# --- Qwen 2.5 1.5B Instruct (80-tok, NullSAE) ---
echo ""
echo "--- Qwen 2.5 1.5B 80-tok NullSAE $(date) ---"
python -m biorefusalaudit.cli run \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --eval-set "$EVAL_SET" \
    --out runs/legality-qwen25-1b5-80tok \
    --sae-source none \
    --max-new-tokens 80

echo "Qwen 80-tok done $(date)"

# --- Llama 3.2 1B Instruct (80-tok, NullSAE) ---
echo ""
echo "--- Llama 3.2 1B 80-tok NullSAE $(date) ---"
python -m biorefusalaudit.cli run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --eval-set "$EVAL_SET" \
    --out runs/legality-llama32-1b-80tok \
    --sae-source none \
    --max-new-tokens 80

echo "Llama 80-tok done $(date)"

echo ""
echo "=== Legality Confound Suite COMPLETE $(date) ==="
echo "Results:"
for d in runs/legality-gemma2-80tok runs/legality-gemma2-150tok runs/legality-qwen25-1b5-80tok runs/legality-llama32-1b-80tok; do
    if [ -f "$d/report.json" ]; then
        n=$(python3 -c "import json; d=json.load(open('$d/report.json')); print(len(d['records']))" 2>/dev/null || echo "?")
        echo "  $d: $n records"
    else
        echo "  $d: MISSING report.json"
    fi
done
