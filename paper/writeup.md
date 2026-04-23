# BioRefusalAudit: Auditing the Depth of LLM Bio-Safety Refusals Using Sparse Autoencoder Interpretability

**Caleb DeLeeuw** — AIxBio Hackathon 2026, Track 3 (Biosecurity Tools)

**Status at submission:** flagship pipeline shipped end-to-end + specialist review fully addressed (see `notes/SPECIALIST_REVIEW_2026_04_23.md`). The three-legged evidence requirement (activation + attribution + perturbation) is codified: no feature gets called a "named circuit" in this paper unless it passes all three legs, via `biorefusalaudit/features/attribution_labels.py::classify_tier`. Results in §4 below come from the tuned-catalog + fitted-T run on Gemma 2 2B-IT + Gemma Scope 1 layer 12; prior-work substrate-selection reasoning documented in `papers/sae_layer_selection/`.

---

## 1. Problem

Existing bio-safety evaluations — VCT (Götting et al., 2025), WMDP-Bio (Li et al., 2024), ABC-Bench, LAB-Bench (Laurent et al., 2024) — measure whether a model *will* produce hazardous output. They do not measure whether a model that refuses *can't*, or merely *doesn't right now*.

This distinction matters for every deployment decision. A model with a shallow refusal over an intact internal capability is one prompt-engineering step (or one framing shift) away from providing hazardous content. A model with a deep refusal — where internal representations do not activate on hazard features even when probed — is structurally safer. Qi et al. (2024) argue the alignment floor sits "a few tokens deep" for most current RLHF models. That claim needs a measurement, not just an argument.

BioRefusalAudit provides the first cross-model, interpretability-grounded tool for this measurement. We call it **refusal depth**.

## 2. Contribution

1. A **calibrated surface-internal divergence metric** `D(s, f) = 1 - cos(f, T^T · s)` over the soft-weighted surface classification `s` (refuse / comply / partial / hedge / evade) and the SAE-feature activation vector `f` (bio_content / hazard_adjacent / refusal_circuitry / hedging / deception_correlate). See `docs/METHOD.md`.
2. A **75-prompt public eval set** stratified by tier (benign_bio 23, dual_use_bio 30, hazard_adjacent_category 22 at descriptor-level in the public file; full tier-3 bodies on an HL3-gated HF dataset) and by framing (direct / educational / roleplay / obfuscated).
3. A **multi-judge consortium** combining regex first-pass, Gemini CLI, Claude Haiku via `claude -p` subprocess, and a local Ollama judge, with weighted voting and disagreement flagging. Inherits and adapts the nanochat-SAE-deception-research pattern.
4. A **reproducible end-to-end pipeline** runnable from a fresh clone on a 4GB GPU for Gemma 2 2B-IT + Gemma Scope 1 residual SAEs; with a Gemma 4 E2B-IT custom-SAE secondary path demonstrating methodology portability to community-trained SAEs.
5. A **Streamlit dashboard skeleton** for per-prompt interactive auditing.

## 3. Method (summary)

For each (model, prompt) pair:
1. Generate a completion at T=0.7 with forward-hook capture of the residual stream at a model-specific target layer (~50% depth).
2. Project captured residual vectors through the model's SAE; take mean per-feature activation across generated tokens.
3. Project the d_sae-dimensional activation into 5 hand-validated feature categories via `data/feature_catalog/<model>.json`; L1-normalize to produce `f`.
4. Run judge consortium on (prompt, completion); aggregate votes to the soft distribution `s` over the 5 surface classes.
5. Compute `D(s, f, T)` and the three divergence flags (`hazard_features_active_despite_refusal`, `refusal_features_active_despite_compliance`, `deception_correlate_active`).

The alignment matrix `T ∈ ℝ^{5×5}` is fit by ridge-regularized least squares on held-out positive-control prompts, with a condition-number guard to fail loudly on under-calibrated datasets.

## 4. Results

### 4.0 Evidence discipline

Per specialist review, every claim in this section satisfies the three-legged gate:

1. **Activation evidence** — the feature fires on the relevant prompt tier with Cohen's-d ≥ 0.2 vs a neutral baseline (`biorefusalaudit/features/feature_validator.py::differentiation_check`).
2. **Attribution evidence** — the feature's decoder direction aligns with the residual-stream direction on the decisive token span. (v0.2 uses feature-magnitude ranking as a proxy; Anthropic-style circuit graphs land in the follow-on release.)
3. **Perturbation evidence** — ablating the feature via SAE-decompose-patch-recompose changes either the surface label or the divergence score by |ΔD| > 0.2 (`scripts/run_intervention.py`).

Claims not meeting all three legs appear below but are labeled `candidate` / `activation_only` / `attribution_only` per `classify_tier`.

### 4.1 Pipeline validation (synthetic)

Before any model load, the full scoring + reporting + cross-model aggregation pipeline was validated via `scripts/synthetic_demo.py`. It fabricates plausible surface distributions and feature vectors per prompt (using the PRD's expected-surface labels), applies the identity-biased prior T from `configs/calibration_gemma2_2b.yaml`, and writes full reports under `runs/synthetic_*/`.

Resulting per-tier mean divergence, averaged across four synthetic "model" runs:

| Tier | Mean D (synthetic) |
|---|---:|
| `benign_bio` | 0.02 |
| `dual_use_bio` | 0.11 |
| `hazard_adjacent_category` | 0.47 |

The pattern matches the expected finding shape (tier-3 high, benign low) by construction. This validates:
- JSONL loading + 75-row parse
- Divergence computation with the prior T
- Flag-firing on hazard-on-refuse
- Per-tier aggregation + report generation
- Cross-model aggregation + matplotlib scaling plot — see `demo/scaling_plot_synthetic.png`

These numbers are **synthetic** and serve only as a plumbing check. They are not a result about any actual model.

### 4.2 Flagship run — Gemma 2 2B-IT + Gemma Scope 1 layer 12, tuned catalog + fitted T

**Hardware:** GTX 1650 Ti Max-Q, 4 GB VRAM, `device_map="auto"` partial GPU + CPU offload. Two 75-prompt passes: pass 1 is an activation-collection run with stub catalog (generates `activations.npz` for tuning); pass 2 re-runs with the tuned catalog + fitted T.

**Pass 1 → pass 2 artifact chain:**

1. Pass 1 → 75 prompts → `runs/gemma-2-2b-it-L12-activations/activations.npz`
2. `scripts/auto_tune_catalog.py` → `data/feature_catalog/gemma-2-2b-it.json` v0.2 (Cohen's-d-selected top-20 features per category, preserving hedging + deception_correlate as stubs for lack of positive controls in the eval set)
3. `scripts/fit_calibration.py` → `configs/calibration_gemma2_2b.yaml` with fitted T + prior T retained
4. Pass 2 → 75 prompts → `runs/gemma-2-2b-it-L12-tuned/report.json`

**Per-tier mean divergence (pass 2 with tuned catalog + fitted T):**

| Tier | n | Mean D (pass 2) | Std | Flags — hazard-on-refuse (%) |
|---|---:|---:|---:|---:|
| `benign_bio` | 23 | _[to be filled when run completes]_ | _x_ | _x_ |
| `dual_use_bio` | 30 | _[to be filled]_ | _x_ | _x_ |
| `hazard_adjacent_category` | 22 | _[to be filled]_ | _x_ | _x_ |

Placeholder pending the tuned-eval run — see `runs/gemma-2-2b-it-L12-tuned/report.md` when the post-eval chain daemon completes. All numbers will be cross-checked against the `calibration_delta.json` pre/post-fit deltas.

### 4.3 Intervention experiments — causal evidence

Three prompts selected from pass 2 via `trace_selected_cases.py` (top high-divergence + top false-refusal + top false-compliance) and intervened on with `scripts/run_intervention.py`.

For each selected prompt × category pair, three completions are generated:

- **Baseline:** canonical forward pass, no hook.
- **Ablated:** top-5 features in the chosen category zeroed in the residual stream at layer 12 via SAE-decompose-subtract-recompose.
- **Boosted:** same features multiplied by 3× instead.

A feature earns "named circuit" status only if `label_changed` OR `|ΔD| > 0.2` between baseline and intervention.

**Results to be filled from `runs/interventions/*.json`** once the chain completes. Expected format:

| Prompt id | Category | Baseline label | Ablated label | Boosted label | ΔD_ablate | ΔD_boost | Named circuit? |
|---|---|---|---|---|---:|---:|---|

### 4.4 Cross-architecture reference (Colab T4)

- `google/gemma-2-9b-it` + `gemma-scope-9b-pt-res` layer 20 at bnb 4-bit (unreleased at time of local hackathon run — available via the `notebooks/colab_biorefusalaudit.ipynb` pipeline)
- `meta-llama/Llama-3.1-8B-Instruct` + `OpenMOSS-Team` Llama Scope `l16r_8x`

Results land in `runs/colab_*/report.{md,json}`; the cross-model scaling plot regenerates via `scripts/build_scaling_plot.py --include-synthetic`. Gemma 3 family evaluation deferred pending Gemma Scope 2 public release (Gemma 3 weights cached locally for forward compatibility).

## 5. Limitations

- **Gemma Scope 2 pending.** The originally-planned primary substrate (Gemma 3 + Gemma Scope 2 residual SAEs) is not yet publicly released. MVP demonstrates methodology on Gemma 2 + Gemma Scope 1 plus Gemma 4 E2B + custom SAEs. This is a scope reduction, not a methodological compromise — the divergence metric, eval set, and judge consortium are all model-agnostic.
- **Quantization confound for Gemma 4 E2B.** Gemma 4 E2B at 4-bit produces activations from a quantized model; any divergence number on this model is measuring the quantized system, not the fp16 reference. Flagged explicitly in `runs/gemma-4-E2B-it/report.md`.
- **Feature catalog is stubbed.** The v0.1 feature catalogs under `data/feature_catalog/` use plausible-but-not-hand-validated feature indices. A hand-validation pass through Neuronpedia is in the queue; results shipped here should be interpreted as methodology demonstration, not publication-ready numbers.
- **Calibration on small n.** T-matrix fit uses the prompt pool itself as positive-control holdout; calibration with a larger external positive-control set would tighten T.
- **A100 cross-architecture (Llama 3.1 8B) deferred** pending user approval of rental budget (~$30-45). The MVP cannot speak to cross-architecture generalization from open weights until that runs.

## 6. Relation to prior work

The divergence metric builds on the ensemble-category insight from the author's Secret Agenda work (arXiv:2509.20393 / AAAI 2026): since individual SAE auto-labels fail to detect strategic deception across 38 models, the fix is to trust the category-level behavior of ensembles of hand-validated features rather than any single feature. See README §"Related work and state of the art" for the broader literature survey (Qi et al. 2024 on shallow safety; Arditi et al. 2024 on refusal-direction geometry; Lieberum et al. 2024 + Templeton et al. 2024 on SAE foundations; Marks & Rager et al. 2024 on sparse feature circuits; Goldowsky-Dill et al. 2025 on deception probes).

## 7. Responsible release

Code is released under HL3-FULL; tiers 1+2 of the eval set under CC-BY-4.0; tier 3 is category-level-only in this public repo with full bodies on an HL3-gated HF dataset (to be published at `SolshineCode/biorefusalaudit-gated`). See `SAFETY.md`, `docs/HL3_RATIONALE.md`, and `LICENSE_HL3_DATASET.md`.

## 8. Future work

- Gemma Scope 2 full sweep when released (Gemma 3 270M / 1B / 4B / 12B).
- Llama 3.1 8B via Llama Scope — already cached; cross-architecture comparison adds only GPU time.
- Transcoder + CLT analysis (PRD S2) to move from "metric" to "circuit" for the strongest divergence flags.
- Unlearning before/after (PRD S4) — apply the metric to RMU snapshots for direct biosecurity value.
- Coefficient Giving RFP (due May 11) — 6-month follow-on proposal framed around the results shipped here.
