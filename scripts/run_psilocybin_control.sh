#!/usr/bin/env bash
# Psilocybin legality-vs-hazard control experiment
# Tests whether refusal circuit responds to Schedule I legal status rather than genuine CBRN risk.
# Psilocybin is: biologically benign, not a CBRN agent, US Schedule I (illegal), clinically studied.
# Comparison: if refusal rates on psilocybin_cultivation tier match hazard_adjacent tier,
# legality (not bio-hazard) is the primary driver of the refusal circuit.
#
# Run AFTER cross-arch runs free the GPU (Phi-3-mini expected ~18:50 local time).
# Expected runtime: ~20 min on GTX 1650 Ti (15 prompts, Gemma 2 2B-IT, 80-tok, regex judge only)

set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/Scripts/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

OUTDIR="runs/psilocybin-control-gemma2"
LOGFILE="runs/psilocybin-control-gemma2.log"
EVALSET="data/eval_set_public/eval_set_psilocybin_control_v1.jsonl"

echo "[psilocybin-control] Starting run at $(date)" | tee "$LOGFILE"
echo "[psilocybin-control] Eval set: $EVALSET (15 prompts)" | tee -a "$LOGFILE"
echo "[psilocybin-control] Model: google/gemma-2-2b-it + Gemma Scope 1 layer 12" | tee -a "$LOGFILE"
echo "[psilocybin-control] Purpose: legality-vs-hazard confound control" | tee -a "$LOGFILE"

python -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set "$EVALSET" \
    --out "$OUTDIR" \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml \
    --no-llm-judges \
    --max-new-tokens 80 \
    2>&1 | tee -a "$LOGFILE"

echo "[psilocybin-control] Run complete at $(date)" | tee -a "$LOGFILE"
echo "[psilocybin-control] Results: $OUTDIR/report.json" | tee -a "$LOGFILE"
echo ""
echo "[psilocybin-control] Quick distribution summary:"
python -c "
import json, collections
with open('$OUTDIR/report.json') as f:
    data = json.load(f)
records = data['records']
cats = ['psilocybin_pharmacology', 'psilocybin_cultivation', 'psilocybin_clinical', 'psilocybin_legal']
for cat in cats:
    rows = [r for r in records if r.get('category','') == cat]
    if not rows:
        continue
    counts = collections.Counter(r['surface_label'] for r in rows)
    n = len(rows)
    print(f'  {cat} (n={n}): comply={counts.get(\"comply\",0)/n*100:.0f}%  hedge={counts.get(\"hedge\",0)/n*100:.0f}%  refuse={counts.get(\"refuse\",0)/n*100:.0f}%')
print()
# Also print mean D per category
print('  Mean D by category:')
for cat in cats:
    rows = [r for r in records if r.get('category','') == cat]
    if not rows:
        continue
    mean_d = sum(r['divergence'] for r in rows) / len(rows)
    print(f'  {cat}: D={mean_d:.3f}')
" 2>&1 | tee -a "$LOGFILE"
