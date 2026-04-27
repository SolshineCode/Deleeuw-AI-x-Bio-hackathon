#!/usr/bin/env bash
# Extended psilocybin legality-confound test suite
# Runs all model variants sequentially. Start after Phi-3-mini frees the GPU (~18:50 local).
#
# Models / configs:
#   1. Gemma 2 2B-IT + Gemma Scope 1 SAE, 80-tok  — primary (D-values + labels)
#   2. Gemma 2 2B-IT + Gemma Scope 1 SAE, 150-tok  — token-budget comparison on psilocybin
#   3. Qwen 2.5 1.5B + NullSAE, 80-tok             — cross-arch (Qwen refused 83% benign bio)
#   4. Llama 3.2 1B + NullSAE, 80-tok              — cross-arch (Llama refused 30% benign bio)
#
# Key comparison:
#   psilocybin_cultivation (Schedule I, illegal) vs psilocybin_pharmacology (legal to study)
#   vs psilocybin_clinical (legal in research) vs main eval hazard_adjacent tier
#   If cultivation ≈ hazard_adjacent refuse rate: legality confound confirmed
#   If pharmacology ≈ benign_bio comply rate: bio vocabulary alone is not the trigger

set -euo pipefail

cd "$(dirname "$0")/.."
if [ -d ".venv/Scripts" ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

EVALSET="data/eval_set_public/eval_set_psilocybin_control_v1.jsonl"

run_model() {
    local LABEL="$1"
    local OUTDIR="$2"
    local LOGFILE="$3"
    shift 3
    echo ""
    echo "========================================"
    echo "[psilocybin] Starting: $LABEL at $(date)"
    echo "========================================"
    python -m biorefusalaudit.cli run \
        --eval-set "$EVALSET" \
        --no-llm-judges \
        "$@" \
        2>&1 | tee -a "$LOGFILE"
    echo "[psilocybin] Completed: $LABEL at $(date)"
}

summarize() {
    local OUTDIR="$1"
    local LABEL="$2"
    echo ""
    echo "--- $LABEL summary ---"
    python -c "
import json, collections
with open('$OUTDIR/report.json') as f:
    data = json.load(f)
records = data['records']
cats = ['psilocybin_pharmacology', 'psilocybin_cultivation', 'psilocybin_clinical', 'psilocybin_legal']
for cat in cats:
    rows = [r for r in records if r.get('category','') == cat]
    if not rows: continue
    counts = collections.Counter(r['surface_label'] for r in rows)
    n = len(rows)
    mean_d = sum(r['divergence'] for r in rows)/n
    print(f'  {cat} (n={n}): comply={counts.get(\"comply\",0)/n*100:.0f}%  hedge={counts.get(\"hedge\",0)/n*100:.0f}%  refuse={counts.get(\"refuse\",0)/n*100:.0f}%  D={mean_d:.3f}')
" 2>&1
}

echo "[psilocybin-extended] Suite start: $(date)"
echo "[psilocybin-extended] Eval set: $EVALSET (15 prompts)"
echo "[psilocybin-extended] 4 model runs queued — estimated ~60 min total"

# --- Run 1: Gemma 2 + Gemma Scope 1, 80-tok ---
OUTDIR1="runs/psilocybin-gemma2-sae-80tok"
LOG1="runs/psilocybin-gemma2-sae-80tok.log"
run_model "Gemma 2 2B-IT + Gemma Scope 1 SAE (80-tok)" "$OUTDIR1" "$LOG1" \
    --model google/gemma-2-2b-it \
    --out "$OUTDIR1" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --max-new-tokens 80
summarize "$OUTDIR1" "Gemma 2 + SAE 80-tok"

# --- Run 2: Gemma 2 + Gemma Scope 1, 150-tok ---
OUTDIR2="runs/psilocybin-gemma2-sae-150tok"
LOG2="runs/psilocybin-gemma2-sae-150tok.log"
run_model "Gemma 2 2B-IT + Gemma Scope 1 SAE (150-tok)" "$OUTDIR2" "$LOG2" \
    --model google/gemma-2-2b-it \
    --out "$OUTDIR2" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --max-new-tokens 150
summarize "$OUTDIR2" "Gemma 2 + SAE 150-tok"

# --- Run 3: Qwen 2.5 1.5B NullSAE, 80-tok ---
OUTDIR3="runs/psilocybin-qwen25-1b5-80tok"
LOG3="runs/psilocybin-qwen25-1b5-80tok.log"
run_model "Qwen 2.5 1.5B + NullSAE (80-tok)" "$OUTDIR3" "$LOG3" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --out "$OUTDIR3" \
    --sae-source none \
    --layer 20 \
    --quantize 4bit \
    --max-new-tokens 80
summarize "$OUTDIR3" "Qwen 2.5 1.5B NullSAE 80-tok"

# --- Run 4: Llama 3.2 1B NullSAE, 80-tok ---
OUTDIR4="runs/psilocybin-llama32-1b-80tok"
LOG4="runs/psilocybin-llama32-1b-80tok.log"
run_model "Llama 3.2 1B + NullSAE (80-tok)" "$OUTDIR4" "$LOG4" \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --out "$OUTDIR4" \
    --sae-source none \
    --layer 9 \
    --quantize none \
    --max-new-tokens 80
summarize "$OUTDIR4" "Llama 3.2 1B NullSAE 80-tok"

echo ""
echo "========================================"
echo "[psilocybin-extended] All runs complete: $(date)"
echo "========================================"
echo ""
echo "Cross-model psilocybin comparison:"
echo "(compare cultivation vs. pharmacology within each model;"
echo " compare each model's cultivation refuse% vs. its hazard_adjacent refuse% from cross-arch runs)"
echo ""
for OUTDIR in "$OUTDIR1" "$OUTDIR2" "$OUTDIR3" "$OUTDIR4"; do
    label=$(basename "$OUTDIR")
    python -c "
import json, collections
try:
    with open('$OUTDIR/report.json') as f:
        data = json.load(f)
    records = data['records']
    cats = ['psilocybin_pharmacology', 'psilocybin_cultivation', 'psilocybin_clinical', 'psilocybin_legal']
    print('$label')
    for cat in cats:
        rows = [r for r in records if r.get('category','') == cat]
        if not rows: continue
        counts = collections.Counter(r['surface_label'] for r in rows)
        n = len(rows)
        print(f'  {cat}: refuse={counts.get(\"refuse\",0)/n*100:.0f}%  hedge={counts.get(\"hedge\",0)/n*100:.0f}%  comply={counts.get(\"comply\",0)/n*100:.0f}%')
except FileNotFoundError:
    print('$label: report.json not found')
" 2>&1
done

echo ""
echo "[psilocybin-extended] === SUITE COMPLETE ==="
