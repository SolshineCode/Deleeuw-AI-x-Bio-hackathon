#!/usr/bin/env bash
# Overnight cross-architecture eval: Llama 3.1 8B-Instruct + Llama Scope l16r_8x
#
# Fills §4.4 of the paper with real Llama 3.1 8B results.
#
# Hardware: GTX 1650 Ti 4 GB VRAM. Llama 8B at 4-bit NF4 needs ~4.5 GB, so
# --max-gpu-memory 3GiB forces device_map=auto+max_memory to split GPU/CPU.
# Expect ~150-300s/prompt with partial CPU offload. 75 prompts × 200s ≈ 4 hours.
# Three passes (collect → tune → eval) ≈ 10-12 hours total. Run overnight.
#
# Usage (from repo root, venv activated):
#   source .venv/Scripts/activate
#   bash scripts/run_llama31_cross_arch.sh 2>&1 | tee runs/llama31-cross-arch.log
#
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="meta-llama/Llama-3.1-8B-Instruct"
EVAL_SET="data/eval_set_public/eval_set_public_v1.jsonl"
SAE_REPO="fnlp/Llama3_1-8B-Base-LXR-8x"
SAE_LAYER_ID="Llama3_1-8B-Base-L16R-8x"
LAYER=16
PASS1_OUT="runs/llama-3.1-8b-it-L16-pass1"
TUNED_OUT="runs/llama-3.1-8b-it-L16-tuned"
CATALOG="data/feature_catalog/llama-3.1-8b-it.json"
CALIBRATION="configs/calibration_llama31_8b.yaml"

echo "[$(date)] === PASS 1: collect activations (Llama 8B, ~4 hours with CPU offload) ==="
python -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "$PASS1_OUT" \
    --sae-source llama_scope \
    --sae-release "$SAE_REPO" \
    --sae-id "$SAE_LAYER_ID" \
    --layer "$LAYER" \
    --quantize 4bit \
    --max-gpu-memory 3GiB \
    --no-llm-judges \
    --max-new-tokens 200 \
    --dump-activations

echo "[$(date)] === CATALOG AUTO-TUNE ==="
python scripts/auto_tune_catalog.py \
    --activations "$PASS1_OUT/activations.npz" \
    --out "$CATALOG" \
    --model "$MODEL" \
    --sae-source llama_scope \
    --top-k 20

echo "[$(date)] === FIT CALIBRATION T ==="
python scripts/fit_calibration.py \
    --report "$PASS1_OUT/report.json" \
    --out "$CALIBRATION"

echo "[$(date)] === PASS 2: tuned catalog + fitted T (~4 hours) ==="
python -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "$TUNED_OUT" \
    --sae-source llama_scope \
    --sae-release "$SAE_REPO" \
    --sae-id "$SAE_LAYER_ID" \
    --layer "$LAYER" \
    --quantize 4bit \
    --max-gpu-memory 3GiB \
    --no-llm-judges \
    --max-new-tokens 200 \
    --catalog "$CATALOG" \
    --calibration "$CALIBRATION"

echo "[$(date)] === REJUDGE ==="
python scripts/rejudge_stored_completions.py \
    --report "$TUNED_OUT/report.json" \
    --out "runs/llama-3.1-8b-it-L16-rejudged/report.json"

echo "[$(date)] === CONTRASTIVE SAE FINE-TUNE (pairwise, 5000 steps) ==="
python scripts/train_sae_local.py \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "runs/sae-training-llama31-5000steps-pairwise" \
    --layer "$LAYER" \
    --d-sae 32768 \
    --k 32 \
    --steps 5000 \
    --batch-size 4 \
    --lr 3e-4 \
    --lam-sparse 0.04 \
    --lam-contrast 0.5 \
    --contrastive-mode pairwise \
    --checkpoint-every 1000

echo "[$(date)] === REBUILD SCALING PLOT ==="
python scripts/build_scaling_plot.py --out demo/scaling_plot.png

echo "[$(date)] === DONE ==="
echo "Results: $TUNED_OUT/report.json"
echo "Corrected: runs/llama-3.1-8b-it-L16-rejudged/report.json"
echo "Next: update paper §4.4 and run compare_token_budget.py for cross-arch D table"
