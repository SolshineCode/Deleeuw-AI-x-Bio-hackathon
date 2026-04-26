#!/usr/bin/env bash
# Gemma 2 2B-IT + author-trained bio SAE v1 eval pipeline
# Mirrors the Gemma 4 E2B bio SAE pipeline for cross-model comparison.
#
# SAE: runs/sae-training-gemma2-5000steps/sae_weights.pt
#   TopK(k=32), d_model=2304, d_sae=6144, layer 12 (mean contrastive, 5000 steps)
#
# Usage:
#   bash scripts/run_gemma2_biosae_pipeline.sh 2>&1 | tee runs/gemma2-biosae-v1-pipeline.log
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.
source .venv/Scripts/activate

MODEL="google/gemma-2-2b-it"
SAE_CKPT="runs/sae-training-gemma2-5000steps/sae_weights.pt"
EVAL_SET="data/eval_set_public/eval_set_public_v1.jsonl"
PASS1_OUT="runs/gemma-2-2b-it-bio-sae-v1-pass1"
PASS2_OUT="runs/gemma-2-2b-it-bio-sae-v1-pass2-catalog"
CATALOG="data/feature_catalog/gemma-2-2b-it-bio-sae-v1.json"

echo "[$(date)] === PASS 1: collect activations (Gemma 2 2B, ~45 min) ==="
python -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "$PASS1_OUT" \
    --sae-source custom \
    --sae-release "$SAE_CKPT" \
    --k 32 \
    --d-model 2304 \
    --d-sae 6144 \
    --architecture topk \
    --layer 12 \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 80 \
    --dump-activations \
    2>&1 | tee runs/gemma2-biosae-v1-pass1.log

echo "[$(date)] === CATALOG AUTO-TUNE ==="
python -c "
import json
from pathlib import Path
stub = {
    'model_name': 'google/gemma-2-2b-it',
    'sae_source': 'local:runs/sae-training-gemma2-5000steps/sae_weights.pt',
    'categories': {
        'bio_content': [], 'hazard_adjacent': [], 'refusal_circuitry': [],
        'hedging': [], 'deception_correlate': []
    }
}
Path('$CATALOG').parent.mkdir(parents=True, exist_ok=True)
Path('$CATALOG').write_text(json.dumps(stub, indent=2))
print('Created stub catalog')
"
python scripts/auto_tune_catalog.py \
    --activations "$PASS1_OUT/activations.npz" \
    --existing "$CATALOG" \
    --out "$CATALOG" \
    --top-per-category 20 \
    2>&1 | tee runs/gemma2-biosae-v1-catalog-tune.log

echo "[$(date)] === PASS 2: tuned catalog (~45 min) ==="
python -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set "$EVAL_SET" \
    --out "$PASS2_OUT" \
    --sae-source custom \
    --sae-release "$SAE_CKPT" \
    --k 32 \
    --d-model 2304 \
    --d-sae 6144 \
    --architecture topk \
    --layer 12 \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 80 \
    --catalog "$CATALOG" \
    2>&1 | tee runs/gemma2-biosae-v1-pass2.log

echo "[$(date)] === DONE ==="
echo "Pass2 report: $PASS2_OUT/report.json"
