# BioRefusalAudit — end-to-end eval run on Gemma 2 2B-IT.
# Assumes setup.ps1 has been run and .venv activated.

param(
    [string]$Model = "google/gemma-2-2b-it",
    [string]$EvalSet = "data/eval_set_public/eval_set_public_v1.jsonl",
    [string]$Out = "runs/gemma-2-2b-it",
    [switch]$UseLlmJudges
)

$ErrorActionPreference = "Stop"

$judgeFlag = "--no-llm-judges"
if ($UseLlmJudges) {
    $judgeFlag = "--use-llm-judges"
}

Write-Host "[run_eval] Model: $Model"
Write-Host "[run_eval] Eval set: $EvalSet"
Write-Host "[run_eval] Out: $Out"
Write-Host "[run_eval] LLM judges: $UseLlmJudges"

python -m biorefusalaudit.cli run `
    --model $Model `
    --eval-set $EvalSet `
    --out $Out `
    --sae-source gemma_scope_1 `
    --sae-release gemma-scope-2b-pt-res `
    --sae-id "layer_12/width_16k/average_l0_71" `
    --layer 12 `
    --catalog data/feature_catalog/gemma-2-2b-it.json `
    --calibration configs/calibration_gemma2_2b.yaml `
    $judgeFlag

Write-Host "[run_eval] Done. Report at: $Out\report.md"
