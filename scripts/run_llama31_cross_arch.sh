#!/usr/bin/env bash
# Overnight cross-architecture eval: Llama 3.1 8B-Instruct + Llama Scope l16r_8x
#
# Fills §4.4 of the paper with real Llama 3.1 8B results.
#
# HARDWARE LIMITATION (confirmed 2026-04-26):
# GTX 1650 Ti (4 GB VRAM) cannot run Llama 3.1 8B locally. The model needs
# ~4.5 GB at 4-bit NF4. CPU offload via device_map=auto is broken in
# bitsandbytes 0.49.2 + accelerate 1.13 (AlignDevicesHook.pre_forward fails
# dispatching Params4bit — all 75 prompts fail with "meta tensor" error).
# Three runs confirmed: 0/75 each. See TROUBLESHOOTING.md §Bug C.
#
# THE CORRECT ENVIRONMENT FOR THIS SCRIPT IS A COLAB T4 (16 GB VRAM):
#   - The T4 holds the full 4-bit model without any CPU offload
#   - Run notebooks/colab_biorefusalaudit.ipynb for the managed version
#   - Or run this script from inside a Colab session with the repo cloned
#
# Usage (Colab T4 only):
#   source .venv/Scripts/activate   # or pip install -e . in Colab
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

# Create stub catalog if it doesn't exist yet (required by --existing below)
if [ ! -f "$CATALOG" ]; then
    python - << 'PYEOF'
import json
from pathlib import Path
stub = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "sae_source": "llama_scope_direct:fnlp/Llama3_1-8B-Base-LXR-8x/Llama3_1-8B-Base-L16R-8x",
    "catalog_version": "stub-v1",
    "catalog_note": "Stub — pending auto-tune from pass-1 activations.",
    "source_activations": "PENDING",
    "top_per_category": 20,
    "min_effect_size": 0.3,
    "categories": {"bio_content": [], "hazard_adjacent": [], "refusal_circuitry": [], "hedging": [], "deception_correlate": []}
}
Path("data/feature_catalog").mkdir(parents=True, exist_ok=True)
Path("data/feature_catalog/llama-3.1-8b-it.json").write_text(json.dumps(stub, indent=2))
print("Created stub catalog: data/feature_catalog/llama-3.1-8b-it.json")
PYEOF
fi

echo "[$(date)] === PASS 1: collect activations (Llama 8B, ~4 hours on T4) ==="
# No --max-gpu-memory: T4 has 16 GB, the full 4-bit model fits without CPU offload.
# If running locally on a <8GB card, this will OOM or fail (see header comment).
python -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "$PASS1_OUT" \
    --sae-source llama_scope \
    --sae-release "$SAE_REPO" \
    --sae-id "$SAE_LAYER_ID" \
    --layer "$LAYER" \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 200 \
    --dump-activations

echo "[$(date)] === CATALOG AUTO-TUNE ==="
python scripts/auto_tune_catalog.py \
    --activations "$PASS1_OUT/activations.npz" \
    --existing "$CATALOG" \
    --out "$CATALOG" \
    --top-per-category 20

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
