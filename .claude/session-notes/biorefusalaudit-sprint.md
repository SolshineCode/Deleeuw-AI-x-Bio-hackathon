# BioRefusalAudit Sprint Notes — 2026-04-23 (Night before hackathon day 1)

## Current branch: feat/gemma4-calibration-chain-docs
## PR #13 open on GitHub: https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/pull/13

---

## SESSION ACCOMPLISHMENTS

### Pass-5 results (Gemma 4 E2B with correct chat template)
- 65 refuse / 9 comply / 1 hedge (75 prompts)
- ALL 9 complies: `refusal_features_active_despite_compliance=True`
- 5 hazard-adjacent tier complies (actual transgressions): pathogen-adjacent immunology, vaccine-escape, zoonotic spillover, veterinary biocontrol, prion biology
- 4 dual-use bio complies: FDA/EMA regulatory framing (direct), NIH recombinant guidelines (roleplay), moratorium description (obfuscated), gene drive description (obfuscated)
- D near-zero (0.004-0.007) under re-fitted T (cond=165) — community SAE gap, not model property

### T re-fit from pass-5
- `configs/calibration_gemma4_e2b.yaml` updated with T fitted from pass-5 data
- cond=165, MSE=0.0008
- D still near-zero → flag signals are primary channel for Gemma 4 path

### Format ablation experiment IN PROGRESS
- Script: `scripts/format_ablation.py`
- 4 conditions × 24 prompts = 96 completions
- Started at 22:02:29, process PID 28756 still running (GPU 25-35% util)
- Output: `runs/gemma-4-E2B-it-format-ablation/report.json` (not yet written)
- Will trigger a commit/push once results are available

### Paper updates
- §4.5: pass-5 results added (inverse flag finding, hazard transgressions, D near-zero explanation)
- §4.5 caveat: GemmaScope vs community SAE training gap explanation  
- Tense fix: "will reflect" → "reflects (complete 2026-04-23)"
- Word count: 3,479 / 3,500

### Other changes committed and pushed (commit 8fc8dd7)
- `prompt_loader.py`: removed duplicate `hedge_or_refuse` (standardized on `refuse_or_hedge`)
- `control_legal_financial_v1.jsonl`: normalized the 1 `hedge_or_refuse` → `refuse_or_hedge`
- `docs/METHOD.md`: framing-bypass empirical finding + SAE training provenance failure mode
- `demo/scaling_plot.png`: regenerated from 13 real run directories

---

## KEY FINDINGS TO HIGHLIGHT IN SUBMISSION

1. **Safety-format dependency** (confirmed): RLHF safety circuit requires exact `<bos><|turn>user...<turn|>\n<|turn>model` tokens to engage.

2. **Bidirectional flag pattern** (new):
   - `hazard_features_active_despite_refusal`: safety circuit fires, hazard features active (pass-4 → 82-100%)
   - `refusal_features_active_despite_compliance`: safety circuit fires but surface output is compliance (pass-5 → 100% of complies)
   Together: systematic internal-surface mismatch in BOTH directions.

3. **Framing bypass**: obfuscated prompts ("avoid using X") and roleplay framing produce compliance while keeping refusal circuitry active at ~0.32.

4. **SAE training quality gap**: GemmaScope (diverse internet text) vs community SAE (deception-focused behavioral activations, TopK=16). The gap explains near-zero D on Gemma 4. NOT an intrinsic Gemma 4 property.

5. **Format ablation** (pending results): 4 conditions test whether safety circuit keyed to exact `<|turn>model` sequence or semantic structure.

---

## WHAT'S STILL NEEDED FOR SUBMISSION

### User-action required:
1. Run `notebooks/colab_biorefusalaudit.ipynb` on Colab T4 → fills §4.4 cross-arch table
2. Record Streamlit dashboard demo video (60-90s)
3. Submit to hackathon portal

### Can do autonomously:
- [ ] Format ablation results → update paper §4.5/METHOD.md (WAITING on ablation to finish)
- [ ] Dashboard smoke test
- [ ] Final word count check before submission

---

## OPEN PR
- PR #13: feat/gemma4-calibration-chain-docs
- Gemini review triggered (3 prior reviews + 1 new)
- Prior comments addressed: tense fix in paper, redundant label cleanup
- NOT ready to merge — needs user review + Colab T4 results

---

## COMMANDS TO RESUME

```bash
cd /c/Users/caleb/projects/Deleeuw-AI-x-Bio-hackathon
source .venv/Scripts/activate

# Check ablation process status:
ps -ef | grep format_ablation | grep -v grep
ls -la runs/gemma-4-E2B-it-format-ablation/

# Once ablation done, analyze results:
python -c "import json; d=json.load(open('runs/gemma-4-E2B-it-format-ablation/report.json')); print(d['aggregate'])"

# Run tests:
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m pytest tests/ -q -m "not integration"

# Push to PR:
git push origin feat/gemma4-calibration-chain-docs
```
