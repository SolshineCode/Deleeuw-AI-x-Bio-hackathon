#!/usr/bin/env bash
# Continuation of run_save_and_continue.sh, picking up from Step 2 (tier3-gemma4-v1).
# Use when the main pipeline crashed after dual-use completed.
#
# Usage: bash scripts/run_continue_from_tier3.sh >> runs/run_save_and_continue.log 2>&1 &

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

save_and_commit() {
    local RUN_NAME="$1"
    local RUN_DIR="runs/$RUN_NAME"
    local RESULTS_DIR="results/$RUN_NAME"

    echo "[$(date)] Saving $RUN_NAME to $RESULTS_DIR ..."
    mkdir -p "$RESULTS_DIR"
    cp "$RUN_DIR/report.json"     "$RESULTS_DIR/" 2>/dev/null || true
    cp "$RUN_DIR/activations.npz" "$RESULTS_DIR/" 2>/dev/null || true
    cp "$RUN_DIR/report.md"       "$RESULTS_DIR/" 2>/dev/null || true

    git add "$RESULTS_DIR/"
    git add -f "$RESULTS_DIR/activations.npz" 2>/dev/null || true
    git commit -m "data: save $RUN_NAME results (data mandate)

$(python -c "
import json, statistics
try:
    r = json.load(open('$RESULTS_DIR/report.json'))
    divs = [rec['divergence'] for rec in r['records']]
    tiers = set(rec['tier'] for rec in r['records'])
    print(f'n={len(r[\"records\"])} tier(s)={list(tiers)}')
    print(f'mean_D={statistics.mean(divs):.4f} median={statistics.median(divs):.4f} std={statistics.stdev(divs) if len(divs)>1 else 0:.4f}')
except: pass
" 2>/dev/null)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    git push origin feature/hf-publish-pipeline
    echo "[$(date)] Committed and pushed $RUN_NAME."
}

run_eval_and_save() {
    local EVAL_SET="$1"
    local OUT_DIR="$2"
    local RUN_NAME
    RUN_NAME=$(basename "$OUT_DIR")

    # Skip if already committed to results/
    if [ -f "results/$RUN_NAME/report.json" ]; then
        echo "[$(date)] $RUN_NAME already in results/, skipping."
        return 0
    fi

    echo "[$(date)] Starting eval: $EVAL_SET -> $OUT_DIR"
    python -m biorefusalaudit.cli run \
        "${BASE_FLAGS[@]}" \
        --eval-set "$EVAL_SET" \
        --out "$OUT_DIR"
    echo "[$(date)] Eval done: $OUT_DIR"

    save_and_commit "$RUN_NAME"
}

# -----------------------------------------------------------------------
# Step 2: 22-prompt v1 sets (cross-generator comparison)
# -----------------------------------------------------------------------
run_eval_and_save \
    data/eval_set_gated/eval_set_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-tier3-gemma4-v1

run_eval_and_save \
    data/eval_set_gated/eval_set_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-tier3-qwen3-v1

# -----------------------------------------------------------------------
# Step 3: Calibration holdout sets
# -----------------------------------------------------------------------
run_eval_and_save \
    data/eval_set_gated/calibration_holdout_v2_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v2-gemma4

run_eval_and_save \
    data/eval_set_gated/calibration_holdout_v2_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v2-qwen3

run_eval_and_save \
    data/eval_set_gated/calibration_holdout_v3_tier3_explicit_gemma4_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v3-gemma4

run_eval_and_save \
    data/eval_set_gated/calibration_holdout_v3_tier3_explicit_qwen3_v1.jsonl \
    runs/gemma-2-2b-it-explicit-cal-v3-qwen3

# -----------------------------------------------------------------------
# Step 4: Retrain Track B projection adapter on all explicit activations
# -----------------------------------------------------------------------
echo "[$(date)] All evals done. Collecting activation paths for Track B retrain..."

ACTIVATION_FILES=()
for npz in results/gemma-2-2b-it-explicit-*/activations.npz; do
    [ -f "$npz" ] && ACTIVATION_FILES+=("$npz")
done
[ -f results/gemma-2-2b-it-L12-activations/activations.npz ] && \
    ACTIVATION_FILES+=(results/gemma-2-2b-it-L12-activations/activations.npz)

echo "[$(date)] Training Track B adapter on ${#ACTIVATION_FILES[@]} activation file(s)..."
# Collect report files for explicit-prompt runs (provides real surface_soft labels)
REPORT_FILES=()
[ -f results/gemma-2-2b-it-150tok-rejudged/report.json ] && \
    REPORT_FILES+=(results/gemma-2-2b-it-150tok-rejudged/report.json)
for rpt in results/gemma-2-2b-it-explicit-*/report.json; do
    [ -f "$rpt" ] && REPORT_FILES+=("$rpt")
done

python scripts/train_projection_adapter.py \
    --activations "${ACTIVATION_FILES[@]}" \
    --report "${REPORT_FILES[@]}" \
    --calibration configs/calibration_gemma2_2b.yaml \
    --out-pt configs/projection_adapter_gemma2_2b.pt \
    --out-yaml configs/projection_adapter_gemma2_2b.yaml

echo "[$(date)] Track B adapter retrain complete."
git add configs/projection_adapter_gemma2_2b.yaml
git commit -m "model: retrain Track B adapter on explicit-prompt activations

Trained on all explicit-prompt activation vectors (n=300 from Wave 3 +
n=75 from original L12 run) after all explicit-prompt evals completed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git push origin feature/hf-publish-pipeline

echo "[$(date)] Pipeline complete. GPU work done."
