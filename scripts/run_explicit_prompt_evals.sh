#!/usr/bin/env bash
# Runs all explicit-prompt eval sets through Gemma 2 2B-IT + Gemma Scope 1.
# Sequential (single GPU, 4GB VRAM). Estimated ~15-25 min per 100-sample file,
# ~5-8 min per 22-sample file.
# Usage: bash scripts/run_explicit_prompt_evals.sh > /tmp/explicit_evals.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
source .venv/Scripts/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

BASE_FLAGS=(
    --model google/gemma-2-2b-it
    --sae-source gemma_scope_1
    --sae-release gemma-scope-2b-pt-res
    --sae-id "layer_12/width_16k/average_l0_82"
    --layer 12
    --catalog data/feature_catalog/gemma-2-2b-it.json
    --calibration configs/calibration_gemma2_2b.yaml
    --max-new-tokens 150
    --no-llm-judges
    --dump-activations
    --skip-safety-check
)

run_eval() {
    local EVAL_SET="$1"
    local OUT_DIR="$2"
    echo "[$(date)] Starting: $EVAL_SET -> $OUT_DIR"
    python -m biorefusalaudit.cli run \
        "${BASE_FLAGS[@]}" \
        --eval-set "$EVAL_SET" \
        --out "$OUT_DIR"
    echo "[$(date)] Done: $OUT_DIR"
}

# --- Primary comparison set (100-sample, all three tiers) ---
run_eval \
    data/eval_set_gated/eval_set_tier3_explicit_qwen3_100sample.jsonl \
    runs/gemma-2-2b-it-explicit-tier3-qwen3-100

run_eval \
    data/eval_set_gated/eval_set_benign_explicit_qwen3_100sample.jsonl \
    runs/gemma-2-2b-it-explicit-benign-qwen3-100

run_eval \
    data/eval_set_gated/eval_set_dualuse_explicit_qwen3_100sample.jsonl \
    runs/gemma-2-2b-it-explicit-dualuse-qwen3-100

# --- 22-prompt v1 sets (cross-generator comparison) ---
run_eval \
    data/eval_set_gated/eval_set_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-tier3-gemma4-v1

run_eval \
    data/eval_set_gated/eval_set_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-tier3-qwen3-v1

# --- Calibration holdout explicit sets ---
run_eval \
    data/eval_set_gated/calibration_holdout_v2_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v2-gemma4

run_eval \
    data/eval_set_gated/calibration_holdout_v2_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v2-qwen3

run_eval \
    data/eval_set_gated/calibration_holdout_v3_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v3-gemma4

run_eval \
    data/eval_set_gated/calibration_holdout_v3_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v3-qwen3

echo "[$(date)] All explicit-prompt evals complete."
