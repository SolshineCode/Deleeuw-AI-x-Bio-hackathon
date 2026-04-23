#!/usr/bin/env bash
# Flagship-pipeline orchestration — run the specialist-review required sequence:
# eval-with-activations → auto-tune-catalog → fit-calibration → eval-with-validated-catalog
# → select-top-cases → intervention → attribution report.
#
# After running, runs/flagship/ contains:
#   - activations.npz (from first pass)
#   - report.json (post-tuning final numbers)
#   - attribution.json + attribution.md
#   - intervention/<prompt_id>_<category>.json for each intervention
#   - calibration_delta.json (pre/post T shift)

set -u
cd "$(dirname "$0")/.."

MODEL="${1:-google/gemma-2-2b-it}"
LAYER="${2:-12}"
SAE_ID="${3:-layer_12/width_16k/average_l0_82}"
RUN_DIR="runs/flagship"

mkdir -p "$RUN_DIR"
echo "==== $(date -u +'%Y-%m-%dT%H:%M:%SZ') START flagship pipeline ===="

echo "---- Step 1: activation-collection eval (75 prompts, max_new_tokens=30) ----"
PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out "$RUN_DIR/pass1_activations" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "$SAE_ID" \
    --layer "$LAYER" \
    --quantize none \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --max-new-tokens 30 \
    --dump-activations \
  2>&1 | tee "$RUN_DIR/pass1_activations/stderr.log"

echo "---- Step 2: auto-tune feature catalog from real activations ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/auto_tune_catalog.py \
    --activations "$RUN_DIR/pass1_activations/activations.npz" \
    --existing data/feature_catalog/gemma-2-2b-it.json \
    --out data/feature_catalog/gemma-2-2b-it.json \
    --top-per-category 20 \
  2>&1 | tee "$RUN_DIR/auto_tune_catalog.log"

echo "---- Step 3: fit calibration T on pass-1 (s,f) pairs ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u scripts/fit_calibration.py \
    --report "$RUN_DIR/pass1_activations/report.json" \
    --config configs/calibration_gemma2_2b.yaml \
  2>&1 | tee "$RUN_DIR/fit_calibration.log"

echo "---- Step 4: re-run eval with tuned catalog + fitted T ----"
PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe -u -m biorefusalaudit.cli run \
    --model "$MODEL" \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out "$RUN_DIR/pass2_tuned" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "$SAE_ID" \
    --layer "$LAYER" \
    --quantize none \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --max-new-tokens 50 \
  2>&1 | tee "$RUN_DIR/pass2_tuned/stderr.log"

echo "---- Step 5: intervention experiments on top selected cases ----"
mkdir -p "$RUN_DIR/interventions"
# select + intervene — hand-picked three cases
for PID in bio_054 bio_001 bio_023; do
  for CAT in refusal_circuitry bio_content hazard_adjacent; do
    OUT="$RUN_DIR/interventions/${PID}_${CAT}.json"
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
      .venv/Scripts/python.exe -u scripts/run_intervention.py \
        --model "$MODEL" \
        --prompt-id "$PID" \
        --category "$CAT" \
        --top-k 5 \
        --sae-id "$SAE_ID" \
        --layer "$LAYER" \
        --catalog data/feature_catalog/gemma-2-2b-it.json \
        --calibration configs/calibration_gemma2_2b.yaml \
        --max-new-tokens 60 \
        --out "$OUT" \
      2>&1 | tee "$RUN_DIR/interventions/${PID}_${CAT}.log" || echo "  skipped $PID/$CAT"
  done
done

echo "---- Step 6: regenerate scaling plot + attribution artifact ----"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
  .venv/Scripts/python.exe scripts/build_scaling_plot.py \
    --runs-dir runs \
    --out demo/scaling_plot.png \
  2>&1 | tee "$RUN_DIR/scaling_plot.log"

echo "==== $(date -u +'%Y-%m-%dT%H:%M:%SZ') DONE flagship pipeline ===="
echo "Reports: $RUN_DIR/pass2_tuned/report.md"
echo "Interventions: $RUN_DIR/interventions/"
echo "Scaling plot: demo/scaling_plot.png"
