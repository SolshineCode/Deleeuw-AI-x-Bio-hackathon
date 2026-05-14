#!/usr/bin/env bash
# Watches for completed eval run dirs and auto-commits+pushes them.
# Run this in background — survives independently of Claude Code session.
# Usage: bash scripts/watch_and_commit.sh >> runs/watch_and_commit.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
source .venv/Scripts/activate
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

save_and_commit() {
    local RUN_NAME="$1"
    local RUN_DIR="runs/$RUN_NAME"
    local RESULTS_DIR="results/$RUN_NAME"
    [ -f "results/$RUN_NAME/report.json" ] && echo "[$(date)] $RUN_NAME already committed, skipping." && return 0
    echo "[$(date)] Saving $RUN_NAME ..."
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
    print(f'n={len(r[chr(34)]records{chr(34)])}  tier(s)={list(tiers)}')
    print(f'mean_D={statistics.mean(divs):.4f}  std={statistics.stdev(divs) if len(divs)>1 else 0:.4f}')
except: pass
" 2>/dev/null)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    git push origin feature/hf-publish-pipeline
    echo "[$(date)] Committed and pushed $RUN_NAME."
}

EVALS=(
    gemma-2-2b-it-explicit-tier3-gemma4-v1
    gemma-2-2b-it-explicit-tier3-qwen3-v1
    gemma-2-2b-it-explicit-cal-v2-gemma4
    gemma-2-2b-it-explicit-cal-v2-qwen3
    gemma-2-2b-it-explicit-cal-v3-gemma4
    gemma-2-2b-it-explicit-cal-v3-qwen3
)

# Watch loop
while true; do
    for name in "${EVALS[@]}"; do
        report="runs/$name/report.json"
        committed="results/$name/report.json"
        if [ -f "$report" ] && [ ! -f "$committed" ]; then
            echo "[$(date)] Detected completed run: $name"
            save_and_commit "$name"
        fi
    done

    # Check if all evals committed — then run Track B and exit
    all_done=true
    for name in "${EVALS[@]}"; do
        [ ! -f "results/$name/report.json" ] && all_done=false && break
    done
    if $all_done; then
        echo "[$(date)] All evals committed. Running Track B retrain..."
        ACTIVATION_FILES=()
        for npz in results/gemma-2-2b-it-explicit-*/activations.npz; do
            [ -f "$npz" ] && ACTIVATION_FILES+=("$npz")
        done
        [ -f results/gemma-2-2b-it-L12-activations/activations.npz ] && \
            ACTIVATION_FILES+=(results/gemma-2-2b-it-L12-activations/activations.npz)
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
        git add configs/projection_adapter_gemma2_2b.yaml
        git commit -m "model: retrain Track B adapter on explicit-prompt activations (n≈375 vectors)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        git push origin feature/hf-publish-pipeline
        echo "[$(date)] Track B complete. Watch loop exiting."
        exit 0
    fi

    sleep 120
done
