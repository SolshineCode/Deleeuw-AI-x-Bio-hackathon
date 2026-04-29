#!/usr/bin/env bash
# Waits for the GPU to have <2000 MiB in use, then runs both generation passes.
# Run from the bio-hackathon project root:
#   bash scripts/run_generation_when_gpu_free.sh > /tmp/gen_waiter.log 2>&1 &

set -e
VENV="/c/Users/caleb/deception-nanochat-sae-research/.venv-gemma4/Scripts/python.exe"
SCRIPT="scripts/generate_tier3_explicit.py"
LOG_GEMMA4="/tmp/gen_gemma4_main.log"
LOG_QWEN3="/tmp/gen_qwen3_main.log"

echo "[$(date)] Waiting for GPU to have <2000 MiB used..."
while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "[$(date)] GPU memory used: ${USED} MiB"
    if [ -n "$USED" ] && [ "$USED" -lt 2000 ]; then
        echo "[$(date)] GPU is free. Starting generation."
        break
    fi
    sleep 60
done

# Run Gemma 4 first (same arch as eval target -- high priority)
echo "[$(date)] Starting Gemma 4 E2B abliterated generation..."
"$VENV" -u "$SCRIPT" \
    --model gemma4-abliterated \
    --out data/eval_set_public/eval_set_tier3_explicit_gemma4_v1.jsonl \
    2>&1 | tee "$LOG_GEMMA4"

# Run Qwen3 via Ollama (GPU now warm)
echo "[$(date)] Starting Qwen3 4B abliterated generation..."
"$VENV" -u "$SCRIPT" \
    --model qwen3-abliterated \
    --out data/eval_set_public/eval_set_tier3_explicit_qwen3_v1.jsonl \
    2>&1 | tee "$LOG_QWEN3"

echo "[$(date)] Both generation passes complete."
echo "[$(date)] Run:  python scripts/publish_hf_datasets.py --target gated"
echo "[$(date)] to update the gated HF dataset with the new explicit-prompt files."
