#!/usr/bin/env bash
# Cross-architecture behavioral eval: three small instruction-tuned models
# that fit on GTX 1650 Ti (4 GB VRAM) without CPU offload.
#
# Architecture coverage (adds to Gemma 2 2B + Gemma 4 E2B already in paper):
#   1. meta-llama/Llama-3.2-1B-Instruct  — Meta / Llama arch
#      SAE: qresearch/Llama-3.2-1B-Instruct-SAE-l9 (ReLU, d_sae=32768, layer 9)
#   2. Qwen/Qwen2.5-1.5B-Instruct         — Alibaba / Qwen arch
#      SAE: none (NullSAE — D=1.0; labels from regex judge)
#   3. microsoft/Phi-3-mini-4k-instruct   — Microsoft / Phi arch (3.8B, 4bit)
#      SAE: none (NullSAE — D=1.0; labels from regex judge)
#
# Key finding targeted by §4.4: does the hazard_adjacent hedging pattern
# (100% hedge at 80 tok in Gemma 2 + Gemma 4) replicate across architectures?
#
# Usage:
#   source .venv/Scripts/activate
#   bash scripts/run_cross_arch_small_models.sh 2>&1 | tee runs/cross-arch-small-models.log
set -e
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.
source .venv/Scripts/activate

EVAL_SET="data/eval_set_public/eval_set_public_v1.jsonl"

# ─── 1. Llama 3.2 1B Instruct + qresearch SAE (layer 9) ─────────────────────
echo "[$(date)] === Llama 3.2 1B Instruct (layer 9 ReLU SAE, d_sae=32768) ==="
python -m biorefusalaudit.cli run \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --eval-set "$EVAL_SET" \
    --out runs/llama-3.2-1b-it-L9-cross-arch \
    --sae-source custom \
    --sae-release qresearch/Llama-3.2-1B-Instruct-SAE-l9 \
    --d-model 2048 \
    --d-sae 32768 \
    --architecture relu \
    --layer 9 \
    --quantize none \
    --no-llm-judges \
    --max-new-tokens 80 \
    2>&1 | tee runs/cross-arch-llama32-1b.log

# ─── 2. Qwen 2.5 1.5B Instruct (NullSAE — behavioral labels only) ────────────
echo "[$(date)] === Qwen 2.5 1.5B Instruct (NullSAE, behavioral labels only) ==="
python -m biorefusalaudit.cli run \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --eval-set "$EVAL_SET" \
    --out runs/qwen25-1b5-it-cross-arch \
    --sae-source none \
    --layer 20 \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 80 \
    2>&1 | tee runs/cross-arch-qwen25-1b5.log

# ─── 3. Phi-3-mini-4k-instruct (NullSAE — behavioral labels only) ─────────────
echo "[$(date)] === Phi-3-mini-4k-instruct (NullSAE, behavioral labels only) ==="
python -m biorefusalaudit.cli run \
    --model microsoft/Phi-3-mini-4k-instruct \
    --eval-set "$EVAL_SET" \
    --out runs/phi3-mini-4k-it-cross-arch \
    --sae-source none \
    --layer 24 \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 80 \
    2>&1 | tee runs/cross-arch-phi3-mini.log

echo "[$(date)] === ALL CROSS-ARCH RUNS COMPLETE ==="
echo "Reports:"
echo "  Llama 3.2 1B: runs/llama-3.2-1b-it-L9-cross-arch/report.json"
echo "  Qwen 2.5 1.5B: runs/qwen25-1b5-it-cross-arch/report.json"
echo "  Phi-3-mini:   runs/phi3-mini-4k-it-cross-arch/report.json"
echo "Next: read per-tier comply/hedge/refuse rates, update paper §4.4 cross-arch table"
