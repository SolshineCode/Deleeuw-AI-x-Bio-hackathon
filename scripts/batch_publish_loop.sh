#!/usr/bin/env bash
# Polls the 100-sample output files every INTERVAL_MIN minutes and publishes
# to HuggingFace whenever any file has grown. Run from the repo root:
#   bash scripts/batch_publish_loop.sh > /tmp/batch_publish.log 2>&1 &

set -e
INTERVAL_MIN=${1:-30}
VENV="$(pwd)/.venv/Scripts/activate"
SCRIPT="scripts/publish_hf_datasets.py"

FILES=(
    "data/eval_set_gated/eval_set_tier3_explicit_qwen3_100sample.jsonl"
    "data/eval_set_gated/eval_set_benign_explicit_qwen3_100sample.jsonl"
    "data/eval_set_gated/eval_set_dualuse_explicit_qwen3_100sample.jsonl"
)

declare -A LAST_COUNTS
for f in "${FILES[@]}"; do
    LAST_COUNTS["$f"]=0
done

echo "[$(date)] Batch publish loop started. Interval: ${INTERVAL_MIN}min"

while true; do
    sleep $((INTERVAL_MIN * 60))

    CHANGED=0
    for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
            COUNT=$(wc -l < "$f" 2>/dev/null || echo 0)
            if [ "$COUNT" -gt "${LAST_COUNTS[$f]}" ]; then
                echo "[$(date)] $f grew: ${LAST_COUNTS[$f]} -> $COUNT"
                LAST_COUNTS["$f"]=$COUNT
                CHANGED=1
            fi
        fi
    done

    if [ "$CHANGED" -eq 1 ]; then
        echo "[$(date)] Changes detected -- publishing to HF..."
        source "$VENV" && python "$SCRIPT" --target both
        echo "[$(date)] Publish complete."
    else
        echo "[$(date)] No new rows since last check. Skipping publish."
    fi

    # Stop when all files reach 100 rows
    ALL_DONE=1
    for f in "${FILES[@]}"; do
        COUNT=$(wc -l < "$f" 2>/dev/null || echo 0)
        if [ "$COUNT" -lt 100 ]; then
            ALL_DONE=0
            break
        fi
    done
    if [ "$ALL_DONE" -eq 1 ]; then
        echo "[$(date)] All files at 100+ rows. Final publish and loop exit."
        source "$VENV" && python "$SCRIPT" --target both
        break
    fi
done

echo "[$(date)] Batch publish loop finished."
