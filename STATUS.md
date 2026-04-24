# Status

**As of:** 2026-04-23 (updated)
**Hackathon:** [AIxBio Hackathon 2026](https://apartresearch.com/sprints/aixbio-hackathon-2026-04-24-to-2026-04-26), April 24–26, Track 3 (AI Biosecurity Tools), sponsored by Fourth Eon Bio.

## Build state — MVP sprint in progress

What is in place and verified working:

- **Package scaffold** — `biorefusalaudit/` full layout: scoring (divergence + calibration), prompts (loader + stratifier + safety_review), judges (regex + consortium + Gemini/Haiku/Ollama LLM adapters), models (HF adapter + SAE adapter with TopK + JumpReLU), features (profiler + validator + Neuronpedia discovery), runner (single-model + cross-model), reporting (report + tier-3 redaction), Click CLI.
- **42 unit tests green** under `pytest -m "not integration"`.
- **Eval set** — 75 prompts in `data/eval_set_public/eval_set_public_v1.jsonl` (23 benign + 30 dual-use + 22 tier-3 category descriptors). Tier-3 hygiene check (`biorefusalaudit.cli check-safety`) returns clean.
- **Core divergence metric** — correct surface → feature mapping in the prior T (refuse → refusal_circuitry, comply → bio_content, hedge → hedging, evade → deception_correlate). Synthetic demo produces the expected pattern (benign 0.02, dual-use 0.11, tier-3 0.47).
- **Gemma 4 integration** — fixed `load_sae` to handle non-standard weight filenames on HF (e.g., `gemma4_...pt` instead of `sae_weights.pt`). Confirmed that several `gf1_behaviorSAE` repos on HF *do* contain 75MB weights, despite initial metadata-only appearances.
- **Judge consortium** — regex first-pass verified on sample completions (comply / refuse / hedge all classify correctly). Gemini + Claude Haiku + Ollama adapters wired; full consortium path exercised by unit tests.
- **Real model loading verified** — Gemma 2 2B-IT + Gemma Scope 1 SAE (`layer_12/width_16k/average_l0_82`) loads and generates completions on CPU. Smoke test at `runs/smoke/` produced a real report.
- **Colab T4 notebook** — `notebooks/colab_biorefusalaudit.ipynb` runs the full eval pipeline on free Colab T4 for Gemma 2 9B-IT + Llama 3.1 8B-Instruct (bnb 4-bit) with valid, verified sae_lens release IDs. Open-in-Colab badge in `notebooks/README.md`.
- **Streamlit dashboard** — skeletal; paste-a-prompt → classification + top-k features + divergence wired.
- **Demo artifact** — `demo/scaling_plot_synthetic.png` (70 KB matplotlib scaling plot over 4 synthetic runs).
- **Paper writeup** — draft at `paper/writeup.md` documenting synthetic pipeline validation + Colab plan + limitations.

## Caveats known and documented

CORRECTED 2026-04-23: The following two caveats are no longer accurate.

- ~~**Feature catalogs are stubs, not hand-validated.**~~ `data/feature_catalog/gemma-4-E2B-it.json` is now auto-tuned via `auto_tune_catalog.py` from 4-bit GPU activations (`runs/gemma-4-E2B-it-L17-tuned/activations.npz`). Cohen's-d: refusal_circuitry=1.99, bio_content=1.03, hazard_adjacent=0.70. Validator clean. Gemma 2 catalog similarly tuned.
- ~~**Calibration T is prior-only.**~~ T is now fit from real activation data. Gemma 2 2B-IT T fit from pass-2 report (n=75, cond=151). Gemma 4 E2B T fit iteratively: first fit from pass-2 (degenerate — zero feature_vecs); re-fit from pass-3 (real nonzero feature_vecs, 60% nonzero fraction, cond=151, max |ΔD|=0.097). Pass-4 running with this T.

Gemma 4 E2B calibration chain completed steps (2026-04-23):
- Pass-1: 4-bit GPU, stub catalog, dump-activations → `runs/gemma-4-E2B-it-L17/activations.npz`
- auto_tune_catalog: Cohen's-d selected catalog → `data/feature_catalog/gemma-4-E2B-it.json`
- Pass-2 (tuned catalog): feature_vecs still zero (catalog tuned from mismatched fp16 activations) → degenerate T
- Pass-3 (correct 4-bit activations, tuned catalog): feature_vecs 60% nonzero; D=0.061/0.056/0.053 flat (degenerate T)
- T re-fit from pass-3 (real feature_vecs): cond=151, max |ΔD|=0.097
- Pass-4 (properly fitted T): **COMPLETE** (D: benign=0.002, dual-use=0.001, hazard=0.000)
  - Near-zero D across all tiers: Gemma 4 E2B refuses ALL 75 prompts → degenerate S matrix → T predicts mean-F regardless of tier
  - Hazard flags: 82.6% / 96.7% / 100.0% `hazard_features_active_despite_refusal` per tier
  - Key finding: global surface refusal does not suppress hazard feature activations
  - 4th T re-fit from pass-4 data: cond=151, |ΔD|=0.000 (confirms T is stable under uniform-refuse data)
- **Gemma Scope 2 for Gemma 3 is not publicly released** as of 2026-04-22. Gemma 3 weights cached for forward compatibility; SAE path dispatches with `skip_reason` in `configs/models.yaml`. MVP uses Gemma 2 + Gemma Scope 1 as the primary demonstration path and Gemma 4 E2B + custom SAE as the "custom SAE portability" row.

CORRECTED 2026-04-23: The following two caveats in the original status are no longer accurate.

- ~~**CPU-only torch on this box.**~~ CUDA torch (2.6+cu124) is installed and working. GTX 1650 Ti Max-Q (4 GB VRAM) confirmed active. Full 75-prompt Gemma 4 E2B pass-2 runs in ~5 min on GPU.
- ~~**bitsandbytes not installable on Python 3.13 / Windows.**~~ bitsandbytes 0.49.2 is installed and its CUDA kernels are functional (`Linear4bit.cuda()` confirmed). 4-bit quantization works locally.

New caveat replacing both:

- **`device_map="auto"` routes Gemma 4 to CPU.** `Gemma4ForConditionalGeneration` is a multimodal architecture; `accelerate`'s memory estimator mis-sizes it and silently falls back to CPU even when 4 GB VRAM is available. **Fix in place** (`model_adapter.py` now uses `device_map={"": 0}` for quantized models when CUDA is detected). Full details in `TROUBLESHOOTING.md §Gemma 4 (multimodal) loads on CPU despite CUDA being available`.
- **Residual hook accumulated per-step tensors, causing VRAM overflow on long generations.** The `residual_stream_hook` appended one tensor per autoregressive step; a 200-token generation produced ~240 MB of accumulated tensors on top of the ~1 GB 4-bit model weights, leaving no VRAM headroom. Slow prompts took 150–170 s instead of ~6–8 s. **Fix in place** (overwrite `captured[0]` instead of append — `model_adapter.py`). After fix, generation rate ~85 s / 200-token output on GTX 1650 Ti at bitsandbytes NF4; fast refusals ~2 s. See `TROUBLESHOOTING.md §Slow generation on long outputs`.

## Explicitly NOT in the MVP

Deferred to post-submission approval:

- **A100 cross-model sweep** — Gemma 3 4B / 12B, Llama 3.1 70B etc. Deferred until after submission and pending approval of the compute spend (see `docs/COMPUTE.md`).
- **Full Neuronpedia automated feature discovery** — `feature_discovery.py` is wired for the Neuronpedia search API; hand-curation is the primary path for v0.1.
- **Transcoder analysis (S2 stretch)** — Gemma Scope 2 cross-layer transcoders not in MVP.
- **Jailbreak-correlation study (S3 stretch)** — not in MVP.
- **Unlearning before/after (S4 stretch)** — not in MVP.

## Next planned (pre-submission)

DONE (2026-04-23):
- ✅ Gemma 2 2B-IT full calibration chain: pass-1 → auto-tune → pass-2 → fit-T; real D values in paper §4.2
- ✅ Gemma 4 E2B calibration chain through pass-3; T re-fit from real feature_vecs; pass-4 in progress
- ✅ Intervention experiments: 4 feature×prompt pairs, all pass three-legged gate; results in paper §4.3
- ✅ Paper §4.5 Gemma 4 E2B section with honest calibration chain narrative
- ✅ `demo/scaling_plot.png` regenerated from 8 real run directories

DONE (continued):
- ✅ Pass-4 Gemma 4 E2B (properly fitted T): D=0.002/0.001/0.000 (near-zero, uniform-refuse model)
- ✅ Paper §4.5 updated with pass-4 results and "global surface refusal" interpretation
- ✅ 4th T re-fit from pass-4 data
- ✅ Pass-5 Gemma 4 E2B (correct chat template): 65 refuse / 9 comply / 1 hedge; inverse flag finding; 5 hazard-adjacent transgressions; T re-fit (cond=165); paper §4.5 updated with complete findings

DONE (2026-04-24, 8-hour GPU grant — autonomous):
- ✅ Dashboard smoke check COMPLETE: correct D values, feature panel, Named circuit ✓ intervention panel (bio_004)
- ✅ REVIEWER_QUICKSTART.md: fixed test count 51→56, updated commit table
- ✅ SUBMISSION_CHECKLIST.md: dashboard item ticked, word count updated 3479→3499
- ✅ `notebooks/colab_gemma4_sae_training.ipynb`: Phase 0 Colab SAE training notebook (Gemini-authored + amended: multimodal pick_layer fix, dataset fallback)
- ✅ `scripts/summarize_interventions.py`: intervention summary table script
- ✅ `scripts/train_sae_local.py`: local SAE training proof-of-concept script (collects raw residuals, trains TopK(k=32) SAE, logs to JSONL)
- ✅ interventions COMPLETE: 11 named circuits out of 11 total (11/11 = 100%). Counterintuitive finding: 3/11 cases (bio_004/bio_010/bio_060) showed comply→refuse on ablate, suggesting refusal_circuitry features serve compliance-enabling roles in some contexts. See `scripts/summarize_interventions.py` for full table.
- ✅ Local SAE training COMPLETE (02:16 PDT, 500 steps, 22s wall clock): Gemma 4 E2B residuals → TopK(k=32) SAE trained. Key finding: L_contrastive increased (0.74→0.97) — 75-prompt corpus insufficient for bio-specific feature separation (expected); L_recon improved (3.15→0.20). Checkpoint: `runs/sae-training-local/sae_weights.pt`. Fixed Gemma4Model.layers AttributeError (path: model.model.language_model.layers). See `docs/METHOD.md §Proof-of-concept`.
- ✅ Format ablation (80tok) COMPLETE (04:16 PDT, 119 min): n=72, conditions A/B/D. A: 24/24 comply (incl. all 8 hazard-adjacent); B: 14/24 loops, 9/24 comply, 1/24 empty, 0 refuse; D: 24/24 comply. Key finding: 0% refuse in all conditions — safety circuit requires >80 tokens to articulate a refusal. `runs/gemma-4-E2B-it-format-ablation-80tok/report.json`
- ✅ fix: device_map string→int (2026-04-24, 06:09 PDT): `{"": "cuda"}` silently routes bitsandbytes NF4 to CPU on Windows WDDM; fixed to `{"": torch.cuda.current_device()}`. Post-load device assertion added. TROUBLESHOOTING.md + CLAUDE.md gotcha #8. Wasted 33 min diagnosed and fixed.
- ⏳ Gemma 2 2B-IT format ablation RUNNING (06:09 PDT relaunch, PID 50411): conditions A/B/C/D, 8 per tier = 96 records, 80tok, confirmed on cuda:0. Est. completion ~07:50 PDT. 8/96 records done as of 06:20; all comply so far — cross-model format sensitivity comparison vs Gemma 4
- ⏳ Gemma 2 SAE training QUEUED (chain v2 PID 50483, 2000 steps): waiting for format ablation; Gemma 2 residuals → longer convergence test vs Gemma 4 500-step run

TODO (user-action required, planned post-submission):
1. Run `notebooks/colab_biorefusalaudit.ipynb` on Colab T4 (Gemma 2 9B-IT + Llama 3.1 8B-Instruct, ~90 min).
2. Pull `runs/colab_*/report.{md,json}`, regenerate `demo/scaling_plot.png` with cross-arch data.
3. Fill §4.4 cross-arch table in paper.
4. Record Streamlit dashboard demo video (60–90 s).
5. Submit.

DONE (2026-04-23, autonomous):
- ✅ Non-bio control experiment COMPLETE: `runs/control-legal-financial-gemma2/report.{md,json}`
  - benign_legal D=0.573 (std=0.134) / dual_use_legal D=0.672 (std=0.060) / hazard_adjacent_legal D=0.665 (std=0.139)
  - Bio vs. legal effect sizes: d=1.29 (p=0.0001, non-overlapping CIs) vs. d=0.67 (p=0.052, NS, CIs overlap 0.071)
  - **Finding:** D has partial bio-domain advantage (larger gradient in bio) but non-zero legal tier trend confirms generic sensitivity routing also contributes. Motivates domain-specific SAE fine-tuning. Paper §8 updated with full statistical comparison.

## Safety-format dependency investigation (NEW, 2026-04-23)

**Finding:** Gemma 4 E2B's RLHF safety behavior appears gated on the chat template format. Pass-4 ran WITHOUT `apply_chat_template`, resulting in:
- 30/75 empty completions (model emits EOS on raw out-of-distribution text)
- 12/75 repetition loops
- 0% genuine refusals among "coherent" completions (Gemini + Nemotron cross-validation)

**Chat template confirmed (2026-04-23):** Gemma 4 uses `<bos><|turn>user\n...<turn|>\n<|turn>model` — distinct from Gemma 2's `<start_of_turn>` format. The fix (`_apply_chat_template` in `model_adapter.py`) has been implemented and tested.

**Pass-5 (COMPLETE 2026-04-23):** Gemma 4 E2B with correct chat template: 65 refuse / 9 comply / 1 hedge.
- Safety-format dependency confirmed: correct `<|turn>` tokens are prerequisite for behavioral engagement.
- **Inverse flag finding:** ALL 9 comply cases have `refusal_features_active_despite_compliance=True` — safety circuit activates (refusal_circuitry ≈ 0.325 mean) but surface behavior is compliance.
- 5 of 22 hazard-adjacent prompts (23%) produced complies: pathogen-adjacent immunology, vaccine-escape, zoonotic spillover, veterinary biocontrol, prion biology framings — **model transgressions under correct-format evaluation**.
- D under re-fitted T (cond=165): still near-zero (benign=0.006, dual-use=0.006, hazard=0.004) — custom SAE insufficient discriminative power for tier separation via D. Flag signals are informative channel for this model.
- Paper §4.5 updated with complete pass-5 results.

**Planned follow-on experiment:** Explicit format ablation — same prompts, same model, 4 conditions:
1. No template (current pass-4) → generation failure
2. Correct template (pass-5) → baseline behavior
3. Malformed template (truncated turn tokens) → partially activated safety?
4. Alternative role labels in template → does "assistant" vs "model" matter?

This tests whether the safety circuit is keyed to the EXACT token sequence or to the semantic structure of the turn format. Documents in `docs/METHOD.md §Safety-format dependency`.

**Format ablation experiment (COMPLETE 2026-04-23):** 3-condition ablation (A=correct / B=generic / D=wrong-role), 4 prompts × 3 tiers = 36 completions at 40 max tokens. Results at `runs/gemma-4-E2B-it-format-ablation/report.json`.
- **Condition B** (generic User:/Assistant:): 7/12 repetition loops + 5/12 comply — confirms format is necessary for coherent Gemma 4 generation.
- **Condition A vs D** (wrong role "assistant"): indistinguishable — both 12/12 educational responses. Safety/coherence appears semantically keyed to turn structure, not exact `model` token.
- **Caveat:** 40-token max + regex judge insufficient to detect refusal in truncated outputs; refuse rates are 0% in all conditions. Findings are valid for coherence/loop detection, not refusal rate comparison.

## Planned follow-on research arc (post-hackathon)

Motivated by Neuronpedia feature validation (features 2620/1041/7541 are generic vocabulary, not bio-specific) and the non-bio control experiment:

### Phase 0 — Colab SAE training notebook (new, hackathon-deliverable)

**Goal:** T4-compatible Colab notebook that fine-tunes the Gemma 4 E2B community SAE on bio-safety behavioral activations. Key properties:
- Frozen Gemma 4 E2B-IT (NF4 4-bit, ~1GB VRAM) + trainable SAE module (~150MB)
- **Configurable HF dataset** — `HF_DATASET_REPO` + `HF_TEXT_COLUMN` + `HF_LABEL_COLUMN` cell at top; any HF dataset with tier labels can be used
- Residual hook at layer 17 captures multi-token activations across full response
- Training loop: TopK(k=32) SAE with L_recon + L_contrastive + L_sparsity
- **W&B logging** of loss, L0 sparsity, feature activation histograms, top-feature semantic drift per step
- Checkpoint saves to HF via `huggingface_hub.upload_file` every N steps
- Output: fine-tuned SAE weights; run BioRefusalAudit audit with new weights to measure D improvement

**Why this matters:** The near-zero D values on Gemma 4 E2B are due to the community SAE's narrow training distribution (deception-focused activations, not bio-safety text). A T4-trained SAE on bio-safety behavioral activations would unlock tier-discriminative D values and validate the full pipeline end-to-end. This is "Track A" made accessible without institutional compute.

**Status:** Planned. Script: `notebooks/colab_gemma4_sae_training.ipynb`. Full technical spec: `docs/METHOD.md §Colab SAE Training Notebook`. Training data: `SolshineCode/biorefusalaudit-gated` (HL3-gated) + any HF text dataset.

### Phase 1 — Track B adapter (feasible with existing corpus)
- Collect multi-token residual-stream activations from all `runs/*/activations.npz` (currently first-token only)
- Train projection adapter W ∈ ℝ^{k_cat × d_sae} with contrastive loss (hazard_active vs. hazard_suppressed refusals)
- Validate via Neuronpedia: do projected features show bio-specific semantics?
- Requires: multi-token activation capture patch to `model_adapter.py` + adapter training script

### Phase 2 — Track A full SAE fine-tune (requires institutional data access)
- Collect ~10K+ activation vectors from paired bio-hazard completions (base vs. RLHF model; genuine vs. shallow refusals)
- Fine-tune Gemma Scope SAE encoder+decoder with dual objective: reconstruction + contrastive tier separation
- Target: Neuronpedia top features should show bio-specific semantics (pathogen terms, mechanism-of-harm vocabulary, containment language)
- Institutional partners: AISI, CLTR, national biosecurity labs (CBRN red-team datasets under HL3-gated access)
- Proposal vehicle: Coefficient Giving RFP (due May 11) and AISI-UK research collaboration

### Phase 3 — Battery of domain-specific SAEs
- Apply same methodology to CSAM-adjacent, financial fraud, CBRN abuse domains
- Each SAE trained on domain-specific behavioral activation corpus
- Mirrors the Secret Agenda (arXiv:2509.20393) multi-domain deception detection extension

See `docs/METHOD.md §Planned extension` and `paper/writeup.md §8` for full technical specification.

## Build branch

- Active branch: `hackathon-mvp-sprint`
- **Not yet pushed.** Per standing push policy, remote pushes wait for explicit approval.
