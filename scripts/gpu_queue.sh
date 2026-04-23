#!/usr/bin/env bash
# GPU queue — keep GPU busy by chaining eval configs.
# Each run writes to runs/<model>-<variant>/ and stderr.log.

set -u
cd "$(dirname "$0")/.."

COMMON=(-m biorefusalaudit.cli run
        --eval-set data/eval_set_public/eval_set_public_v1.jsonl
        --quantize none
        --catalog data/feature_catalog/gemma-2-2b-it.json)

run_one() {
  local name=$1; shift
  local out="runs/$name"
  mkdir -p "$out"
  echo "==== $(date) START $name ===="
  PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
    .venv/Scripts/python.exe -u "${COMMON[@]}" --out "$out" "$@" \
    2>&1 | tee "$out/stderr.log"
  echo "==== $(date) END   $name (exit $?) ===="
}

# Gemma 2 2B-IT layer 12 L0=82 (baseline — redo if needed)
run_one gemma-2-2b-it-L12-l0_82 \
  --model google/gemma-2-2b-it \
  --sae-source gemma_scope_1 \
  --sae-release gemma-scope-2b-pt-res \
  --sae-id "layer_12/width_16k/average_l0_82" \
  --layer 12 \
  --calibration configs/calibration_gemma2_2b.yaml

# Deeper layer (~80% depth in Gemma 2 2B's 26 layers)
run_one gemma-2-2b-it-L20-l0_71 \
  --model google/gemma-2-2b-it \
  --sae-source gemma_scope_1 \
  --sae-release gemma-scope-2b-pt-res \
  --sae-id "layer_20/width_16k/average_l0_71" \
  --layer 20 \
  --calibration configs/calibration_gemma2_2b.yaml

# Shallower layer (~20% depth)
run_one gemma-2-2b-it-L5-l0_68 \
  --model google/gemma-2-2b-it \
  --sae-source gemma_scope_1 \
  --sae-release gemma-scope-2b-pt-res \
  --sae-id "layer_5/width_16k/average_l0_68" \
  --layer 5 \
  --calibration configs/calibration_gemma2_2b.yaml

# Different L0 sparsity at layer 12 (denser features)
run_one gemma-2-2b-it-L12-l0_176 \
  --model google/gemma-2-2b-it \
  --sae-source gemma_scope_1 \
  --sae-release gemma-scope-2b-pt-res \
  --sae-id "layer_12/width_16k/average_l0_176" \
  --layer 12 \
  --calibration configs/calibration_gemma2_2b.yaml

echo "==== GPU queue exhausted at $(date) ===="
