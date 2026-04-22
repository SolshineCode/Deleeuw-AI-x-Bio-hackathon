# BioRefusalAudit: Auditing the Depth of LLM Bio-Safety Refusals Using Sparse Autoencoder Interpretability

**Caleb DeLeeuw** — AIxBio Hackathon 2026, Track 3 (Biosecurity Tools)

**Status at submission:** baseline methodology + tooling shipped. Cross-model runs in progress on a 4GB local GPU (Gemma 2 2B-IT, Gemma 4 E2B-IT). Gemma 3 family evaluation + A100 cross-architecture sweep (Llama 3.1 8B) deferred to a post-submission v1.1 pending Gemma Scope 2 public release and rental approval.

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

**MVP scope:** results reported here are from the initial pipeline validation runs on Gemma 2 2B-IT with Gemma Scope 1 residual SAEs (layer 12) and Gemma 4 E2B-IT with Solshine custom TopK SAEs (layer 17). See `runs/` for raw reports; this section will be populated with per-tier means once the full 75-prompt run completes.

*Placeholder table (to be replaced with measured numbers):*

| Model | Benign mean D | Dual-use mean D | Tier-3 mean D | Hazard-on-refuse flags (% of tier-3) |
|---|---:|---:|---:|---:|
| gemma-2-2b-it | — | — | — | — |
| gemma-4-E2B-it (4-bit) | — | — | — | — |

Cross-model scaling is a headline stretch (PRD S1): the 270M → 1B → 4B Gemma 3 sweep was planned on Gemma Scope 2, which is not yet public as of 2026-04-22. The MVP submission reports on the two local models above; the Gemma 3 family sweep lands in v1.1.

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
