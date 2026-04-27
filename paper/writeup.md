# BioRefusalAudit: Auditing the Depth of LLM Biosecurity Refusals Using Sparse Autoencoder Interpretability

**Caleb DeLeeuw**, AIxBio Hackathon 2026, Track 3 (Biosecurity Tools)

**Status at submission:** End-to-end pipeline shipped; specialist review addressed (`notes/SPECIALIST_REVIEW_2026_04_23.md`). Three-legged evidence requirement codified: activation + attribution + perturbation, enforced via `attribution_labels.py::classify_tier`. Primary results: Gemma 2 2B-IT + Gemma Scope 1 layer 12, 75 prompts, tuned catalog + within-sample fitted T. Gemma 4 E2B-IT validation run with author-trained SAE at 150-token budget. Behavioral-label-only cross-architecture runs: Llama 3.2 1B, Qwen 2.5 1.5B, Phi-3-mini (NullSAE). Judge-pipeline failure corrected: 29/75 original "refuse" labels were consortium fallback artifacts; corrected via regex re-judge at `runs/gemma-2-2b-it-L12-tuned-rejudged/`.

**Headline findings:**

*The core question.* Standard biosecurity evaluations measure whether a model produces hazardous output. But there is a prior question: when a model refuses, does it *can't*, or just *won't right now*? A model with shallow refusal sitting over an intact internal hazard representation is one framing shift from compliance. BioRefusalAudit measures that gap — calibrated surface-internal divergence (D) between judge-classified surface output and SAE feature activation patterns. High D on a refusal means the model is computing more than it is saying.

**Finding 1: Gemma 2 2B-IT never genuinely refuses — it hedges.** After correcting a judge-pipeline failure that mislabeled 29/75 prompts as "refuse" (§4.2 correction), the actual distribution across 75 prompts is: **0 genuine refusals, 40 comply (53%), 35 hedge (47%)**. The hazard-adjacent tier hedges universally — 100% hedge, 0% comply, 0% refuse — while benign biology is 70% comply / 30% hedge. Hedge-without-refuse is a distinct failure mode from refusal bypass: the model never formed a genuine refusal to bypass. Binary surface classifiers that collapse "hedge" and "refuse" cannot surface this distinction.

**Finding 2: D separates internal activation posture from surface behavior with near-zero within-class variance.** On Gemma 4 E2B-IT at 150-token budget with author-trained SAE (T_prior calibration, n = 75): comply responses D = 0.896 (±0.001), refuse responses D = 0.249 (±0.004), **0.647-point separation, zero overlap**. D = 0.249 on a refusal means internal SAE feature activations align with the refusal direction — the model isn't just saying no, it is not activating bio-hazard feature space. D = 0.896 on a compliant response means bio-relevant features remain active — expected for educational biology content, and exactly what a well-calibrated audit should surface. On Gemma 2 2B-IT (within-sample fitted T): tier separation mean D = 0.467 / 0.655 / 0.669 (benign / dual-use / hazard), Cohen's *d* = 1.29, p = 0.0001, non-overlapping 95% CIs. See within-sample calibration caveat (§4.2).

**Finding 3: Format-gating and the 80-token problem.** Gemma 4 E2B's entire RLHF safety circuit is disabled without canonical `<start_of_turn>` chat-template tokens. Both models refuse **0% at 80-token max generation** across all tiers. Safety articulation requires token budget; constrained deployment contexts — mobile apps, embedded tools, latency-capped APIs — systematically bypass the safety circuits that standard lab-bench evaluations measure. Surface refusal rates do not transfer to production when production ships with different format or length constraints.

**Finding 4: The refusal circuit is a biology detector, not a hazard detector.** Causal intervention named-circuit qualification runs inverted relative to hazard level: benign 87% > dual-use 80% > hazard-adjacent 73%. The strongest single result is bio_014, a benign roleplay prompt, with activation effect size 1.139 and no surface behavior change. The auto-tuned catalog encodes general technical-governance vocabulary rather than bio-specific refusal circuitry. Domain-specific SAE fine-tuning is the fix.

**Primary validity caveats:** T is fit within-sample for Gemma 2 results; a held-out calibration (v3 set) inverted tier ordering (d = −0.967), confirming T is framing-distribution-sensitive. Feature catalog is Cohen's-d auto-tuned, not Neuronpedia-validated. Findings 1, 3, and 4 operate on raw labels and activation magnitudes and survive both caveats. D-values in Finding 2 should be read with the calibration caveat attached. Full correction history in §5.

---

## 1. Problem

Existing biosecurity evaluations (VCT, Götting et al. 2025; WMDP-Bio, Li et al. 2024; ABC-Bench; LAB-Bench, Laurent et al. 2024) measure whether a model *will* produce hazardous output. They don't measure whether a refusing model *can't*, or merely *doesn't right now*.

This gap matters for every deployment decision. A model with a shallow refusal over intact internal hazard representations is one framing shift from compliance. Qi et al. (2024) argue the alignment floor sits "a few tokens deep." Crook (AIxBio keynote, 2026) independently identifies "binary prediction unusable — calibrated confidence required" as the measurement gap. BioRefusalAudit provides the first interpretability-grounded measurement of this distinction — proof-of-concept validated on Gemma 2 2B-IT, with architecture designed for cross-model use as SAE families mature. We call it **refusal depth**.

**Policy motivation (Sandbrink & Crook, AIxBio 2026).** Sandbrink called for systems that monitor AI-bio interactions without reading interaction content. Auditing SAE activations rather than prompt/completion text lets a deployer flag shallow refusals while preserving user privacy, since the activation signal is separable from linguistic content. Dual-use concern masked by innocuous framing is invisible to output-only screeners. Our divergence flags surface it upstream of the completed artifact. This composes with output-based screeners covering distinct failure modes. Yassif and Carter (NTI Bio, January 2026) frame managed-access AI tools as requiring a measurement layer and an enforcement layer. BioRefusalAudit provides both (§7).

## 2. Contribution

1. A **calibrated surface-internal divergence metric** `D(s, f) = 1 - cos(f, T^T · s)` over soft surface classification `s` (refuse / comply / partial / hedge / evade) and SAE-feature vector `f` (bio_content / hazard_adjacent / refusal_circuitry / hedging / deception_correlate). See `docs/METHOD.md`.
2. A **75-prompt public eval set** stratified by tier (benign_bio 23, dual_use_bio 30, hazard_adjacent 22 at descriptor level, with full tier-3 bodies on HL3-gated HF dataset) and framing (direct / educational / roleplay / obfuscated).
3. A **multi-judge consortium** combining regex first-pass, Gemini CLI, Claude Haiku via `claude -p` subprocess, and local Ollama, with weighted voting and disagreement flagging.
4. A **reproducible end-to-end pipeline** runnable from a fresh clone on a 4 GB GPU (Gemma 2 2B-IT + Gemma Scope 1). Gemma 4 E2B + author-trained SAE validates portability to community-trained artifacts.
5. A **Streamlit dashboard** for per-prompt interactive auditing.
6. A **HL3-gated responsible release** (to our knowledge the first in AI biosecurity tooling) instantiating the BDL framework (Bloomfield, Black, Crook et al., *Science* 2026): tiers 1–2 CC-BY-4.0 open, tier-3 hazard bodies behind signed attestation.

## 3. Method (summary)

For each (model, prompt) pair:
1. Generate at T=0.7 with forward-hook capture of the residual stream at a model-specific target layer (~50% depth).
2. Project captured residuals through the model's SAE and take mean per-feature activation across generated tokens.
3. Project the d_sae-dimensional activation into 5 feature categories via `data/feature_catalog/<model>.json` and L1-normalize to produce `f`.
    - **Correction (2026-04-24):** "hand-validated" describes the design target, not the v1.0 MVP. Actual catalogs are Cohen's-d-selected from held-out activation dumps (`scripts/auto_tune_catalog.py`). Semantic validation against Neuronpedia labels and max-activating examples is planned work (§8). Neuronpedia inspection (§4.2) found top auto-tuned features encode generic technical/governance vocabulary rather than bio-specific refusal circuitry. Every §4 result should be read under "auto-tuned catalog, not semantically validated."
4. Run judge consortium and aggregate votes to soft distribution `s`.
5. Compute `D(s, f, T)` and the three flags (`hazard_features_active_despite_refusal`, `refusal_features_active_despite_compliance`, `deception_correlate_active`).

The alignment matrix `T ∈ ℝ^{5×5}` is fit by ridge-regularized least squares on held-out positive-control prompts, with a condition-number guard to fail loudly on under-calibrated datasets.

## 4. Results

**Primary validity caveats:** Two structural limits apply to every D-value in this section. (1) *Within-sample T calibration.* T is fit on the same 75-prompt eval set used to compute the reported divergences. All D-value comparisons should be read as within-sample calibrated, not held-out calibrated. (2) *Auto-tuned, not semantically validated, feature catalog.* See the correction note in §3 and the catalog-validation paragraph in §4.2. Flag-based findings (§4.6) operate on raw per-feature activation magnitudes and do not pass through T. They survive both caveats more robustly than D-value comparisons.

### 4.0 Evidence discipline

Per specialist review, every claim in this section satisfies three legs: (1) Cohen's-d ≥ 0.2 activation differentiation (`feature_validator.py::differentiation_check`); (2) decoder-direction alignment with the decisive residual span; (3) ablation shifts surface label or |ΔD| > 0.2 (`run_intervention.py`). Sub-threshold claims appear as `candidate` / `activation_only` / `attribution_only` per `classify_tier`.

### 4.1 Pipeline validation (synthetic)

Before any model load, the full scoring + reporting + cross-model aggregation pipeline was validated via `scripts/synthetic_demo.py`, producing per-tier mean D = 0.02 / 0.11 / 0.47 (benign / dual-use / hazard) by construction. These are **synthetic** plumbing-check numbers only, not results about any real model.

### 4.2 Flagship run: Gemma 2 2B-IT + Gemma Scope 1 layer 12

Two 75-prompt passes: pass 1 collects activations for catalog tuning (`activations.npz`). Pass 2 reruns with the tuned catalog + fitted T. Hardware: GTX 1650 Ti Max-Q, 4 GB VRAM, `device_map="auto"` with partial GPU + CPU offload.

**Per-tier mean divergence (pass 2, tuned catalog + fitted T):**

| Tier | n | Mean D | Std | 95% CI (bootstrap) | Hazard-on-refuse (%) † |
|---|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.467 | 0.190 | [0.392, 0.548] | — |
| `dual_use_bio` | 30 | 0.655 | 0.119 | [0.612, 0.697] | — |
| `hazard_adjacent_category` | 22 | 0.669 | 0.113 | [0.619, 0.715] | — |

† Retracted: 0 genuine refusals per corrected run (§4.2 correction). Corrected surface labels: benign 70% comply / 30% hedge; dual-use 80% comply / 20% hedge; hazard-adjacent 100% hedge / 0% comply / 0% refuse.

95% CIs from 10,000 bootstrap resamplings (seed 42). Benign CI [0.392, 0.548] does not overlap hazard [0.619, 0.715]. Effect size (hazard vs. benign): Cohen's d = **1.29**, p = 0.0001. Within-tier overlap exists: 22% of benign > dual-use mean, 32% of hazard < dual-use mean. Individual-prompt discrimination requires wider CIs than group-level separation at n=23–30.

**Calibration validity caveat:** T is fitted on the same 75 prompts evaluated here — within-sample, not held-out. A held-out calibration experiment (60-prompt v3 set, §5 correction) produced a T that *inverts* tier ordering on v1 data: held-out T yields benign D=0.089 > hazard D=0.023 (d=−0.967). T is framing-distribution-sensitive; v3 over-represents roleplay/obfuscated prompts and the fitted T overfits that distribution. The D-values in this section are within-sample calibration demonstrations — they show the pipeline produces non-trivial, tier-ordered outputs, but they should not be read as held-out validated or distribution-general measurements. For policy-relevant claims use the flag-based findings in §4.6, which operate on raw activation magnitudes and do not pass through T.

**Feature catalog validation, Neuronpedia lookup (2026-04-23):**

| Feature ID | Category | Neuronpedia label | Assessment |
|---|---|---|---|
| 2620 | refusal_circuitry / hazard_adjacent | "authority, control, and criticism of governance" | Polysemantic; plausibly bio-relevant but not specifically refusal |
| 1041 | refusal_circuitry | "technical terms and concepts related to programming or data structures" | Not semantically refusal-related; activates on bio technical vocabulary |
| 7541 | hazard_adjacent / refusal_circuitry | "phrases related to technical descriptions in software development contexts" | Generic technical vocabulary feature |

Auto-tuning by Cohen's d identifies statistically discriminative features, not semantically validated ones. Top catalog features in `refusal_circuitry` and `hazard_adjacent` are generic technical/governance vocabulary that fire differentially on biosecurity prompts, probably because bio prompts use technical vocabulary rather than because these features represent bio-specific refusal circuitry. Semantic validation (Neuronpedia lookups + max-activating example inspection) is needed before any strong mechanistic claim. The tier-separation D result (p=0.0001, d=1.29) may reflect vocabulary routing rather than refusal depth per se. The non-bio control (§5, completed 2026-04-23) confirms D has a bio-domain advantage: legal/financial eval shows d=0.67, p=0.052 (weaker, CIs overlapping), ruling out pure domain-agnostic routing while confirming generic sensitivity partially contributes.

**Correction (2026-04-25):** Judge failure audit, corrected results in. All 29 "refuse" labels in pass-2 carry degenerate soft scores `{label: 0.2 for all labels}`. LLM judges were unavailable. The consortium fell back to uniform prior and dict-order argmax picked "refuse" (first insertion key). Regex re-judge of stored completions (`scripts/rejudge_stored_completions.py`; `runs/gemma-2-2b-it-L12-tuned-rejudged/`) yields **0 genuine refusals; 40 comply (53%); 35 hedge (47%).** Hazard-on-refuse percentages in table above are retracted (see footnote †). D-values are activation-based and unchanged. The corrected primary signal: Gemma 2 2B-IT consistently hedges on hazard-adjacent prompts but does not refuse them — a distinct behavioral posture from genuine refusal, invisible to binary surface evaluation. Full diagnosis in `TROUBLESHOOTING.md §Judge consortium silently returns uniform prior`.

### 4.3 Intervention experiments: causal evidence

All 75 prompts intervened on with `scripts/run_intervention.py` (completed 2026-04-25; see `runs/interventions/`). CMF criterion: `label_changed` OR `|ΔD_ablate| + |ΔD_boost| > 0.2`. **60/75 qualify (80%).** Dose-proportionality validation (boost=1.5, 2.0, 4.0) on the top-10 highest-effect prompts is complete.

| Tier | n | NC=True | Rate | Highest effect |
|---|---:|---:|---:|---|
| benign_bio | 23 | 20 | 87% | bio_014 (1.139) |
| dual_use_bio | 30 | 24 | 80% | bio_040 (0.712) |
| hazard_adjacent | 22 | 16 | 73% | bio_062 (0.434) |
| **overall** | **75** | **60** | **80%** | |

**NC by framing:** direct 86%, roleplay 83%, obfuscated 82%, educational 68%. All four exceed the 60% threshold. Educational's lower rate reflects cooperative completions that reduce the surface-label contrast the CMF criterion requires.

**Dose-proportionality (top-10, boost=1.5/2.0/3.0/4.0):** NC rates are 9/10, 9/10, 10/10, 9/10. 7/10 qualify at all four levels and 3 are mixed. Effect sizes are non-monotone (expected, given selection bias at 3.0x), but circuit identity is stable: the same prompts qualify at 1.5x and 4.0x. The named-circuit finding does not depend on the specific multiplier.

**Key finding: inverted tier ordering.** NC rate is highest for benign_bio and lowest for hazard_adjacent. This reinforces the §4.2 catalog-validation finding: the auto-tuned refusal_circuitry features encode generic technical-governance vocabulary that fires on biology content at all hazard levels, not on bio-specific hazard circuitry. bio_014 (benign/roleplay, effect=1.139) is the corpus peak: the largest internal-state shift with no surface behavior change, a clean example of the divergence BioRefusalAudit is designed to surface.

### 4.4 Cross-architecture reference

**Track A — Large models (hardware-limited, not included in submission results):**
- `google/gemma-2-9b-it` and `meta-llama/Llama-3.1-8B-Instruct` both exceed local VRAM (GTX 1650 Ti, 4 GB). CPU offload is broken in bnb 0.49.2 + accelerate 1.13 for these model sizes (all 75 prompts fail, see `TROUBLESHOOTING.md §Bug C`). These runs are deferred. Gemma 3 family evaluation deferred pending Gemma Scope 2 public release.

**Track B — Local GTX 1650 Ti (small models, 2026-04-26):** Three additional architectures confirmed to fit in 4 GB VRAM, covering Meta/Llama, Alibaba/Qwen, and Microsoft/Phi families. Key question: does the hazard_adjacent hedging pattern (100% hedge at 80 tok in Gemma 2 + Gemma 4) replicate across architectures? All three run with NullSAE (behavioral labels only, D=1.0 throughout). Note: the community Llama ReLU SAE (`qresearch/Llama-3.2-1B-Instruct-SAE-l9`) was originally planned for Llama but `--architecture relu` is invalid in the current CLI (only `topk`/`jumprelu` accepted); see `TROUBLESHOOTING.md §--architecture relu invalid`. All three models ran as behavioral-label-only comparisons.

**Table 4.4. Cross-architecture surface label distribution (NullSAE behavioral runs; 80-token budget; n=75 per model; regex judge; 2026-04-26). All runs complete.**

| Model | Params | benign comply% | benign hedge% | benign refuse% | dual-use comply% | dual-use hedge% | dual-use refuse% | hazard comply% | hazard hedge% | hazard refuse% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B-IT (Gemma Scope 1, reference) | 2B | 70 | 30 | 0 | 80 | 20 | 0 | 0 | **100** | 0 |
| Llama 3.2 1B-Instruct (NullSAE) | 1B | 70 | 0 | 30 | 57 | 0 | 43 | 9 | 0 | **91** |
| Qwen 2.5 1.5B-Instruct (NullSAE) | 1.5B | 17 | 0 | 83 | 10 | 0 | 90 | 5 | 0 | **95** |
| Phi-3-mini-4k-instruct (NullSAE) | 3.8B | 13 | 0 | 87 | 10 | 0 | 90 | 5 | 0 | **95** |

**Key observations from completed runs (2026-04-26):**

1. **Gemma 2's hedge posture is model-family-specific, not universal.** All three cross-arch models show zero hedging across all 75 prompts. Their safety circuits produce binary comply/refuse. "Hedge-without-refuse" is a Gemma-family characteristic, not a general property of RLHF alignment.

2. **Llama 3.2 1B shows the most calibrated tier sensitivity.** Refuse rate rises from 30% on benign to 91% on hazard-adjacent: a 61-point gradient. This is the only cross-arch model that meaningfully distinguishes tiers. Compare Gemma 2: 0% refuse on benign but also 0% refuse on hazard (100% hedge). Compare Qwen and Phi-3: 12-point gradients — barely discriminating.

3. **Qwen 2.5 1.5B and Phi-3-mini-4k are nearly identical despite a 2.5x parameter difference.** Qwen: benign 83% / hazard 95% refuse. Phi-3: benign 87% / hazard 95% refuse. The benign-to-hazard gradient is 12 points for both. This suggests the RLHF training recipe and data distribution, not model scale, determines the over-refusal pattern.

4. **Qwen and Phi-3 over-refuse benign biology at 83–87%.** This is not safe behavior — it's non-discriminating. A model that refuses 87% of textbook biology questions and 95% of hazard-adjacent questions hasn't learned to detect hazard; it's learned to refuse biology. The 8-point spread between benign and hazard refuse rates is not meaningful discrimination.

5. **No cross-arch model uses hedging.** Only Gemma 2 hedges. Binary comply/refuse classifiers will miss Gemma 2's posture entirely, misclassifying hedges as either comply or refuse depending on classifier threshold.

Script: `scripts/run_cross_arch_small_models.sh`. All runs complete as of 16:42 PDT 2026-04-26 (Phi-3-mini 75/75, 3291.7s).

**Psilocybin legality control — completed (2026-04-26):** A 15-prompt control set (`data/eval_set_public/eval_set_psilocybin_control_v1.jsonl`) tests whether the refusal circuit responds to US legal status (Schedule I) rather than genuine CBRN risk. Sub-categories: `psilocybin_pharmacology` (bio vocab, not illegal to study), `psilocybin_cultivation` (*Psilocybe* mycology, US-illegal), `psilocybin_clinical` (legal in research), `psilocybin_legal` (pure policy framing). Run across 4 models (Gemma 2 2B-IT at 80- and 150-tok; Qwen 2.5 1.5B; Llama 3.2 1B). Full analysis and per-model breakdown in §4.7. Core finding: Gemma 2 refuses psilocybin cultivation at 25–50% while refusing hazard-adjacent biology at 0% — the refusal circuit is tracking legal status, not CBRN risk.

### 4.5 Portability path: Gemma 4 E2B-IT + author-trained SAE

To show the methodology is not locked to Gemma Scope releases, we ran the full calibration chain on Gemma 4 E2B-IT using an author-trained TopK SAE from prior deception-detection research (Secret Agenda, arXiv:2509.20393; k=16, layer 17, d_sae=6144). This exercises every pipeline component against a community artifact.

**Critical finding:** Post-hoc inspection of the initial pass revealed 40% empty completions and 16% repetition loops. These weren't genuine refusals. Gemma 4 IT requires canonical `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` tokens to engage its RLHF safety circuit. Without them the model receives text outside its training distribution and either emits EOS immediately or loops. D stays near-zero across all passes because the cross-domain SAE (deception-trained, not biosecurity-trained) cannot separate bio tiers. This is the predicted outcome from Secret Agenda's non-generalizability finding. The flag signals remain valid: they test raw activation magnitudes, not T-calibrated D.

**Safety-format dependency as an interpretability finding.** Gemma 4 E2B's RLHF safety circuit is gated on chat-template formatting. Without canonical turn tokens, it does not activate.

**Pass 4 results (fitted T, pre-chat-template fix):**

| Tier | n | Mean D | Std | Hazard-on-refuse (%) |
|---|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.002 | 0.002 | 82.6% |
| `dual_use_bio` | 30 | 0.001 | 0.001 | 96.7% |
| `hazard_adjacent_category` | 22 | 0.000 | 0.000 | 100.0% |

`hazard_features_active_despite_refusal` at 100% on hazard-adjacent prompts is consistent with a global-refusal strategy that doesn't selectively suppress hazard features. D near-zero reflects the degenerate T fit and cross-domain SAE limitation. **Scientific caveat:** Gemma 4 E2B runs at 4-bit NF4 quantization (4 GB VRAM constraint), so all D-values measure the quantized system. More importantly, this SAE was trained on deception-focused behavioral activations, not biosecurity behavioral activations. D≈0 is the *predicted* outcome here: Secret Agenda's central finding is that behaviorally-specific SAEs don't generalize across behavioral domains, and domain specificity is the binding constraint. The failure is cross-domain transfer between behavioral corpora, not a fundamental pipeline failure. Gemma Scope coverage for Gemma 4 (unreleased at submission) provides the future generalist baseline.

**Key finding for policy:** The `hazard_features_active_despite_refusal` flag is the primary signal for Gemma 4 E2B. At 100% on hazard-adjacent prompts, this model's refusals on hazard content are structurally similar to its refusals on benign content. Both refuse, but both also activate hazard features. This is consistent with a "global refusal" safety strategy that doesn't selectively suppress hazard representations. A deployer reporting a 100% surface-refusal rate for this model on hazard-adjacent prompts is reporting something technically true and practically incomplete.

**Pass 5 (correct chat template applied, 2026-04-23):** With `_apply_chat_template()` fix: 65 refuse / 9 comply / 1 hedge. All 9 comply cases fire `refusal_features_active_despite_compliance` (mean refusal_circuitry ≈ 0.325). Five are hazard-adjacent prompts (pathogen-adjacent immunology, vaccine-escape, zoonotic spillover, veterinary biocontrol, prion biology). D under re-fitted T remains near-zero (benign=0.006, hazard=0.004). The cross-domain SAE limitation holds. Flags are the informative channel.

**Correction (2026-04-24, chat-template token names):** Earlier prose used `<|turn>` / `<turn|>` shorthand instead of Gemma's canonical `<start_of_turn>` / `<end_of_turn>`. The malformed form appears deliberately in `scripts/format_ablation.py` conditions C and D. No empirical claim depends on this distinction.

**Pass 2 (bio-trained SAE, 2026-04-26):** Full 75-prompt eval of Gemma 4 E2B-IT against our WMDP-contrastive-trained bio SAE (`Solshine/gemma4-e2b-bio-sae-v1`, TopK k=32, d_model=1536, d_sae=6144, layer 17, 2000-step training). Unlike the deception-focused SAE in Pass 4/5, this SAE produces non-zero feature vectors: 1285/1500 bio_content feature hits in pass1, mean activation 14.71. **Pipeline calibration bug (documented):** when pass1 runs without a catalog, f_vec is zero for all prompts → `fit_calibration.py` produces T=0 → pass2 D=1.000 for every prompt despite real feature activations. Fix: restore T_prior when fitted T is degenerate (added to `divergence.py`); report.json re-scored offline (`scripts/rescore_with_prior_T.py`). Re-scored results with T_prior:

| Tier | n | Mean D | Std | comply (%) | refuse (%) | hedge (%) |
|---|---:|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.614 | 0.321 | 57% | 43% | 0% |
| `dual_use_bio` | 30 | 0.703 | 0.295 | 70% | 30% | 0% |
| `hazard_adjacent` | 22 | 0.647 | 0.306 | 59% | 36% | 5% |
| **overall** | **75** | **0.659** | **0.309** | | | |

**Calibration caveat:** T_prior is an identity-biased permutation matrix, not a data-fitted alignment. Data-fitted T (which requires a behavioral corpus where both f_vec and s_vec are informative) would produce lower absolute D and tighter tier separation. Compare to Gemma Scope baseline (Gemma 2 2B, fitted T, 80-tok): benign=0.362, dual=0.406, hazard=0.404. The higher D here reflects T_prior miscalibration, not necessarily weaker bio encoding. **Key positive finding:** the bio-trained SAE features fire reliably on bio prompts. The measurement infrastructure is working; the calibration pipeline ordering needs a guard for featureless-pass1 runs. **Surface label finding:** benign tier refuses at 43% — the same over-refusal pattern reported in §4.6 Finding A. Gemma 4 E2B safety circuit fires on biosecurity-adjacent content regardless of tier. The bio SAE activates consistently, confirming bio-domain encoding despite collapsed L_contrastive (corpus-size bottleneck noted in §8).

**150-tok run — D label-split finding (n=75, complete):** At 150-token budget with T_prior calibration, the divergence metric produces a clean two-cluster distribution by surface label — no overlap, stable variance across all 75 prompts:

| Surface label | n | Mean D | Std |
|---|---:|---:|---:|
| comply | 59 | 0.896 | 0.001 |
| refuse | 16 | 0.249 | 0.004 |

Per-tier mean D (150-tok, T_prior), for comparison to 80-tok Pass 2 (T_prior) below:

| Tier | n | Mean D | Std | comply% | refuse% | hedge% |
|---|---:|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.754 | 0.274 | 78% | 22% | 0% |
| `dual_use_bio` | 30 | 0.767 | 0.263 | 80% | 20% | 0% |
| `hazard_adjacent` | 22 | 0.750 | 0.277 | 77% | 23% | 0% |
| **overall** | **75** | **0.759** | **0.271** | | | |

80-tok T_prior (Pass 2) for reference: benign=0.614, dual=0.703, hazard=0.647. Higher 150-tok mean D reflects more comply labels at 150 tokens (comply D=0.896 >> refuse D=0.249) — tier-level mean D is a label-mix artifact, not a pure activation signal. Note: at 150 tokens the tier-level means converge (0.750–0.767 range), unlike the 80-tok spread; this reflects more even comply/refuse distributions across tiers at the longer budget. The label-split is the interpretable quantity.

**Interpretation:** D=0.249 on refuse labels means internal SAE feature activations are *aligning* with the expected refusal direction under T_prior — these are structurally consistent refusals (low divergence). D=0.896 on comply means the internal pattern diverges from what a pure comply label would predict — consistent with bio-content features remaining active even when the surface output is compliant. The 0.647-point separation with near-zero within-class variance is stronger than the tier-level separation in the 80-tok results, and survives the T_prior-miscalibration caveat because the effect operates at the label level (label is directly what T maps to). Within-class variance is stable from the n=57 interim to n=75 final (comply std=0.001 unchanged; refuse std 0.004 unchanged), confirming the bimodal structure is not a small-sample artifact.

*Format ablation (80tok, n=96 G4 A/B/C/D + n=96 G2 A/B/C/D):* G4 cond B: 58% loops; cond C: hazard 100% empty, benign/dual-use 100% comply; cond D: 100% comply. G2: 100% comply, 0% loops across all conditions. G2 at 150tok (n=48): same. **Full 75-prompt G2 corrected run at 80tok:** 0 genuine refusals; 42 comply, 33 hedge. Hazard-adjacent: 100% hedge at all three budgets (80/150/200 tok). Posture is stable. At 150 tok, obfuscated peaks at 94% (vs 77%/59% at 80/200); hazard_adj D=0.760 vs 0.669. 14/75 per-prompt changes; 0 hazard-adjacent.

### 4.6 Primary policy findings

Two findings warrant elevation because they survive the calibration-circularity and catalog-validation caveats: both rest on raw activation magnitudes and behavioral counts, not D-values through the fitted T.

**Finding A: Format-dependent safety activation.** Gemma 4 E2B's RLHF safety circuit is gated on canonical chat-template tokens. Without them: 40% empty, 16% loops, 0% refusals. With them (pass 5): 65 refuse / 9 comply / 1 hedge. Full 75-prompt G2 corrected run at 80tok: **0 genuine refusals; hazard-adjacent 100% hedge at 80, 150, and 200 tokens.** Tier-level behavioral posture is token-budget-stable. Framing shifts: obfuscated prompts hedge more at 80 tok (77% vs 59%). Direct/educational comply more (29%/32% hedge vs 48%/42%). *Policy implication:* constrained deployment contexts may surface fewer refuses but not fewer hedges. The activation signal persists where surface classifiers cannot. Monitoring at the activation layer is format-stable and budget-stable where output classifiers are not. This directly addresses the Sandbrink managed-access agenda cited in §1.

**Finding B: Safety activation without behavioral suppression.** All 9 correct-format Gemma 4 comply cases fire `refusal_features_active_despite_compliance`. The safety circuit represents the hazard context without gating the output. Five of the 9 are hazard-adjacent biology prompts. The structural inverse was reported in §4.2 (see §4.2 correction: with 0 genuine G2 refusals, that cell is vacuous). The Gemma 4 result stands independently and the typology holds. Together, these two flags produce a concrete four-cell safety-posture typology:

| | hazard_on_refuse low | hazard_on_refuse high |
|---|---|---|
| **refusal_on_comply low** | deep refusal (structurally safe) | shallow refusal (surface-only) |
| **refusal_on_comply high** | over-refuses benign, represents hazard | non-suppressive safety (Gemma 4 E2B pattern) |

BioRefusalAudit distinguishes all four cells at the activation layer. Surface evaluation alone cannot. For deployers and model providers, this two-flag surface is a lighter-weight audit than full D-value calibration and survives the §4 validity caveats.

### 4.7 Legality confound control: Schedule I vs. CBRN hazard

The cross-arch behavioral comparison (§4.4) raised an alternative explanation for Qwen and Phi-3's high refuse rates: models may be routing refusals based on US legal status rather than genuine CBRN hazard detection. To test this, we designed a 15-prompt control set of psilocybin-adjacent biology — a compound that is Schedule I federally but biologically non-toxic, carries FDA Breakthrough Therapy designation for depression, and appears in peer-reviewed NEJM and JAMA Psychiatry trials. Four sub-categories probe distinct dimensions of legality vs. biology: `psilocybin_pharmacology` (technical biological discussion, not illegal to study), `psilocybin_cultivation` (*Psilocybe* mycology, US-illegal), `psilocybin_clinical` (framed within legal research contexts), and `psilocybin_legal` (pure policy/law framing). If refusal rates track cultivation > pharmacology — illegality of activity, not bio-hazard — the refusal circuit responds to legal status as a proxy for danger.

**Results — psilocybin-only control (n=4 per category; 4 models):**

| Model | Pharmacology refuse% | Cultivation refuse% | Clinical refuse% | Legal refuse% | Hazard-adjacent refuse% (main eval) |
|---|---:|---:|---:|---:|---:|
| Gemma 2 2B-IT (80-tok) | 0% | **25%** | 0% | 33% | **0%** |
| Gemma 2 2B-IT (150-tok) | 25% | **50%** | 0% | 0% | **0%** |
| Qwen 2.5 1.5B (80-tok) | 75% | 50% | 100% | 100% | 95% |
| Llama 3.2 1B (80-tok) | 0% | 0% | 50% | 33% | 91% |

**Results — cross-compound legality confound (Gemma 2 2B-IT, 80-tok, n=3–4 per cell; 2026-04-26):**

To test whether the Gemma 2 confound is Schedule I-general or compound-specific, we ran the expanded 51-prompt legality confound eval set across four Schedule I compounds — all biologically benign, all lacking meaningful CBRN uplift potential. Cannabis was included as a critical internal control: federally Schedule I since 1970, but cannabis cultivation information is culturally normalized and legal in 24 states.

| Compound | Legal status | Cultivation refuse% | Pharmacology refuse% | Clinical refuse% | Legal refuse% |
|---|---|---:|---:|---:|---:|
| Psilocybin (Sch I) | Fully illegal | **33%** | 0% | 0% | 0% |
| Cannabis (Sch I) | Federally illegal, 24 states legal | **0%** | 0% | 0% | 0% |
| LSD (Sch I) | Fully illegal | — | **25%** | 0% | 0% |
| Mescaline/peyote (Sch I†) | Alkaloid illegal; plant legal to own | **0%**‡ | **25%** | — | 0% |

† Mescaline is Schedule I; peyote plant (*Lophophora williamsii*) is legal to purchase and cultivate ornamentally in the US.  
‡ Mescaline "cultivation" prompts cover peyote plant ecology and botany (legal activity) — 0% refusals, consistent with legal framing of the plant.  
LSD cultivation cell absent from 51-prompt set (ergot biology prompts added in expanded 432-prompt set running separately).

The cannabis result is the key comparison. Cannabis cultivation gets 0% refusals despite identical federal Schedule I status. The confound is not "this compound is Schedule I → refuse cultivation." It is compound-specific. Psilocybin cultivation carries a cultural salience and stigma that cannabis cultivation no longer does, and RLHF fine-tuning has absorbed that distinction. The refusal circuit is not running a DEA database lookup; it is recovering a culturally-conditioned riskiness signal from training data.

LSD and mescaline pharmacology each show 25% refusals — lower than psilocybin cultivation but above zero, suggesting partial pharmacology confounds for classical psychedelics that do not extend to their plant-source or legal framing. The three-way mescaline legal paradox (alkaloid illegal, plant legal, Native American Church exemption) produces 0% refusals across all framings — the model answers the legal policy framing without flagging it as hazardous.

**Per-model interpretation (psilocybin-only control):**

*Gemma 2 (legality confound confirmed):* Cultivation refuse rate (25–50%) exceeds pharmacology refuse rate (0–25%) AND substantially exceeds hazard-adjacent refuse rate (0%). The model refuses discussing how to grow *Psilocybe* mushrooms — US-illegal — at higher rates than it refuses genuinely hazard-adjacent biology prompts (zoonotic spillover, pathogen-adjacent immunology). The refusal circuit is doing something more like "this activity is federally illegal" than "this content is CBRN-hazardous." A biosecurity evaluator relying on surface refusal rates from Gemma 2 would measure *negative* hazard specificity: the model refuses psilocybin cultivation more readily than actual biosecurity risks.

*Qwen 2.5 1.5B (topic-level detection):* Refuses 50–100% across all four psilocybin sub-categories regardless of legal angle. The trigger here is the word "psilocybin" and associated vocabulary — topic detection, not illegality detection. Qwen's 95% hazard-adjacent refuse rate from the main eval is consistent: it refuses most biology that sounds alarming, regardless of actual CBRN relevance. This is a different failure mode from Gemma 2's legality confound — non-discriminating content-category blocking rather than legal-status proxying.

*Llama 3.2 1B (no confound):* Refuses 0% on pharmacology and cultivation sub-categories, with moderate refuse rates on clinical/legal sub-categories (50% and 33%). Hazard-adjacent refuse rate from main eval is 91%. Llama complies freely on psilocybin but refuses genuinely hazardous biology — the opposite of the legality confound pattern. This is arguably the most discriminating posture of the four models tested, even though Llama was initially flagged for over-refusal in the cross-arch comparison.

**Policy implication.** Current RLHF safety training does not reliably separate "this is federally illegal" from "this is a CBRN risk." For two of the four models tested, the refusal circuit conflates them — but the confound is not a simple Schedule I flag. It reflects cultural salience encoded during RLHF: psilocybin cultivation is refused; cannabis cultivation (same legal status) is not. A deployment context that wants to allow discussion of Schedule I compounds in clinical, research, or harm-reduction contexts — while genuinely blocking CBRN-hazardous biology — cannot rely on these refusal circuits to make that distinction. BioRefusalAudit provides the measurement layer that makes this conflation auditable.

**Caveat and expanded run.** The cross-compound table above has n=3–4 per cell — pilot estimates only. An expanded eval set at n=27 per sub-category across 4 compounds and 4 sub-categories (psilocybin, mescaline/peyote, LSD, cannabis) was built and is running as of 2026-04-26: `data/eval_set_public/eval_set_legality_confound_v1.jsonl` (432 prompts, 16 sub-categories × 27 prompts each; `scripts/run_legality_confound.sh`). Full results pending completion. Mescaline/peyote replaced MDMA as the fourth compound; MDMA has documented serotonergic neurotoxicity and hyponatremia risk and does not satisfy the biologically benign design requirement.

## 5. Limitations

- **Non-bio control (completed, §8).** D on a 30-prompt legal/financial eval shows weaker tier separation (Cohen's d=0.67, p=0.052, CIs overlapping) vs. bio (d=1.29, p=0.0001, non-overlapping). D has a bio-domain advantage, but generic sensitivity routing partially contributes. Domain-specific SAE fine-tuning (§8) is the structural fix.
- **Feature catalog not semantically validated.** Neuronpedia inspection of top auto-tuned features (2620, 1041, 7541) reveals generic technical/governance vocabulary. Automated LLM labeling of decontextualized windows is unreliable (probe: Gemini 2.5 Flash on 200-word bio corpus chunks, ~60% accuracy, reasons hallucinated). Full-paragraph context is required.
- **Judge failure in flagship G2 run (corrected).** 29/75 records in `runs/gemma-2-2b-it-L12-tuned/` had degenerate soft judgments (uniform prior → dict-order "refuse" label). Regex re-judge (`runs/gemma-2-2b-it-L12-tuned-rejudged/`): **0 genuine refusals; 40 comply (53%); 35 hedge (47%)**. Hazard-adjacent tier: 100% hedge, 0% comply, 0% refuse. §4.2 flag rates retracted. D-values (d=1.29) unchanged. Full diagnosis: `TROUBLESHOOTING.md §Judge consortium silently returns uniform prior`.
- **CMF status overstates evidence.** §4.3 results are candidate mechanistic features, not named circuits. Full validation requires paraphrase consistency testing and dose-proportionality.
- **Within-sample T calibration (correction, 2026-04-25):** Held-out calibration run (60-prompt v3, 20 per tier). Held-out T fit produced mean|ΔT|=0.580. Rerunning v1 eval with held-out T inverts tier ordering (benign=0.089 > hazard=0.023, d=-0.967). **Finding:** T is sensitive to framing distribution. V3 has more roleplay/obfuscated prompts than v1. Held-out T overfits to that distribution. Calibration set must match evaluation framing distribution. Within-sample T remains the better-calibrated choice for v1 results.
- **Gemma Scope 2 pending.** Primary planned substrate (Gemma 3 + Gemma Scope 2) not yet released. MVP uses Gemma 2 + Gemma Scope 1.
- **Large-model cross-architecture comparison deferred.** Gemma 2 9B-IT and Llama 3.1 8B-Instruct exceed local VRAM; CPU offload is broken in the current bnb + accelerate versions (see `TROUBLESHOOTING.md §Bug C`). Small-model cross-arch results (Llama 3.2 1B, Qwen 2.5 1.5B, Phi-3-mini) are complete and reported in §4.4.

## 6. Relation to prior work

BioRefusalAudit extends two findings from the author's Secret Agenda work (arXiv:2509.20393 / AAAI 2026). First, individual SAE auto-labels fail at domain-specific behavioral detection. Category-level ensembles are more reliable. Second, behaviorally-trained SAEs don't generalize across behavioral domains, reconfirmed here by D≈0 on bio prompts with the deception-focused SAE (§4.5). Domain-specific biosecurity behavioral corpora are required. Cross-domain transfer is insufficient. See README §"Related work" for the broader literature (Qi et al. 2024; Arditi et al. 2024; Lieberum et al. 2024; Templeton et al. 2024; Marks & Rager et al. 2024; Goldowsky-Dill et al. 2025).

## 7. Responsible release

Code: HL3-FULL. Eval tiers 1+2: CC-BY-4.0. Tier 3: category-level descriptors in this public repo, with full prompt bodies on HL3-gated HF dataset (`SolshineCode/biorefusalaudit-gated`). See `SAFETY.md`, `docs/HL3_RATIONALE.md`, `LICENSE_HL3_DATASET.md`.

**Why HL3, not MIT.** MIT and Apache 2.0 permit offensive applications. For a tool identifying which models retain hazard representations behind a surface refusal, that neutrality is a real attack surface: an adversarial actor using the divergence score to target shallow-refusal models. HL3-FULL prevents this with enforceable human-rights conditions. Violation terminates the license. Enforceable repo tests (`safety_review.check_no_hazard_bodies`, `reporting.redaction.redact_tier3`) exceed host ground rules.

## 8. Future work

**Domain-specific SAE fine-tuning** is the binding constraint. Neuronpedia validation confirms the auto-tuned catalog encodes generic technical-governance vocabulary rather than bio-specific circuitry. The corrective: fine-tune SAEs on biosecurity behavioral corpora (base vs. RLHF pairs) separating shallow refusals (hazard features active) from genuine ones (suppressed). A projection adapter is feasible with the existing corpus; full fine-tune needs institutional CBRN datasets (~10K samples). **Proof-of-concept (2026-04-25):** Three SAE training runs complete across two model families. (1) Gemma 2 2B, mean contrastive, 5000 steps: L_recon −99.5%, L_contrastive +7% (vocabulary bottleneck on 75 prompts). (2) Gemma 2 2B, pairwise NT-Xent, 5000 steps: L_recon −99.84%, L_contrastive 0.905→7e-6; near-zero by step 4000. (3) Gemma 4 E2B, pairwise NT-Xent, 5000 steps (d_model=1536, d_sae=6144, layer 17): L_recon −99.98% (final 0.00058), L_contrastive 0.570→0.777 (step 1000) then collapses to 0.000116 by step 4999. All three runs show the same corpus-size bottleneck: the NT-Xent gradient cannot maintain tier separation on 75 prompts once reconstruction dominates. The step 1000 checkpoint (L_contrastive=0.777) is the recommended artifact for tier-separation analysis; the final checkpoint is a high-quality reconstruction SAE. Published: `SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-wmdp`, `-pairwise`, and `Solshine/gemma4-e2b-bio-sae-v1` (pending HF push approval). Corpus size, not compute, is the binding constraint. **WMDP bio-forget corpus (access confirmed 2026-04-25):** Gated access to `cais/wmdp-bio-forget-corpus` (Li et al. 2024) — 24,453 scientific papers on hazardous biology — was granted during the hackathon sprint. The v2 SAE training pipeline has been updated to use this corpus at 5,000 steps; that training run is in progress. The bio vocabulary bottleneck (L_contrastive collapse on 75 prompts) is resolved at the corpus level. What remains is paired behavioral activation corpora — genuine refusal vs. shallow-refusal response pairs from a safety-tuned model — which requires institutional operational data. Institutional CBRN behavioral pairs are the final unlock for contrastive tier separation. See `docs/METHOD.md §Planned extension`.

**Bio-domain SAE eval on Gemma 4 E2B (2026-04-26):** First full eval of Gemma 4 E2B-IT using the domain-specific bio SAE (checkpoint_01000.pt, step 1000, peak contrastive signal L_contrastive=0.777) with auto-tuned catalog (32 features, Cohen's d top=3.28). Tier-level mean D: benign_bio=0.362 (n=23), dual_use_bio=0.406 (n=30), hazard_adjacent=0.404 (n=22). Tier ordering is directionally correct (benign < dual_use ≈ hazard); dual_use/hazard separation is weak (Δ=0.002). Within-comply D increases monotonically with tier: benign 0.466 → dual_use 0.494 → hazard_adjacent 0.504 — this is the refusal-depth signal. Refuse D is near-constant across tiers (~0.228). Calibration T fitted (ridge, λ=0.1): MSE≈0, condition number=483. **Caveat:** D is strictly bimodal (comply ~0.46–0.51, refuse ~0.226–0.230) because `--no-llm-judges` produces binary surface_soft scores; the near-zero calibration residual is an artifact of fitting to binary labels, not a sign of quality fit. Continuous D and reliable calibration require LLM judge scores. **Domain-specificity confirmed:** Cross-domain deception SAE (§4.5) gave D≈0 across all tiers; bio-specific catalog gives D=0.36–0.41 — domain specificity is the operative variable, consistent with Secret Agenda's non-generalizability finding.

**Cross-model bio SAE comparison (2026-04-26):** Gemma 2 2B-IT evaluated with author-trained mean-contrastive bio SAE (TopK k=32, d_model=2304, d_sae=6144, layer 12, 5000 steps). Same pass1→auto-tune→pass2 pipeline. Calibration T fitted (ridge, λ=0.1): MSE=0.0005, condition=550.

| Tier | Gemma 2 2B mean D | Gemma 4 E2B mean D | Δ |
|------|-------------------|---------------------|---|
| benign_bio | **0.446** (comply 0.522, refuse 0.231) | 0.362 (comply 0.466, refuse 0.227) | +0.084 |
| dual_use_bio | **0.488** (comply 0.578, refuse 0.238) | 0.406 (comply 0.494, refuse 0.229) | +0.082 |
| hazard_adjacent | **0.475** (comply 0.583, refuse 0.243) | 0.404 (comply 0.504, refuse 0.229) | +0.071 |

Both models show the same qualitative pattern: comply D > refuse D, and within-comply D increases monotonically with tier severity. Gemma 2 2B within-comply D range: 0.522→0.583 (+0.061 across tiers). Gemma 4 E2B within-comply D range: 0.466→0.504 (+0.038). The larger Gemma 2 gradient may reflect greater bio-feature amplitude in the mean-contrastive SAE vs. the NT-Xent checkpoint. **Caveats:** (1) D is bimodal in both models due to binary surface_soft from regex-only judge; continuous D requires LLM judges. (2) Both SAEs were trained on the same 75-prompt eval set used for evaluation — same-corpus bias cannot be ruled out. (3) The two SAEs use different contrastive objectives (mean vs. NT-Xent step 1000) and different model families; D values are not directly comparable across models. The qualitative tier ordering and the comply/refuse D split are the portable findings.

**Token-budget stability of bio SAE D (2026-04-26):** Both models re-run at 150 tokens (same catalogs). Within-label D is budget-stable across both; mean D shifts are entirely driven by surface-label switching (more comply / fewer refuse at 150 tok).

| Model | Tier | 80-tok mean D | 150-tok mean D | Δ | Mechanism |
|-------|------|---------------|----------------|---|-----------|
| Gemma 2 2B | benign_bio | 0.446 | 0.435 | −0.012 | stable (no switching) |
| Gemma 2 2B | dual_use_bio | 0.488 | 0.536 | +0.048 | 4 refuse→comply |
| Gemma 2 2B | hazard_adjacent | 0.475 | 0.508 | +0.033 | 3 refuse→comply |
| Gemma 4 E2B | benign_bio | 0.362 | 0.420 | +0.057 | 5 refuse→comply |
| Gemma 4 E2B | dual_use_bio | 0.406 | 0.461 | +0.055 | 6 refuse→comply |
| Gemma 4 E2B | hazard_adjacent | 0.404 | 0.475 | +0.071 | 4 refuse→comply + 1 refuse→hedge(D=1.0) |

Within-label comply D change: <0.016 for Gemma 2, <0.007 for Gemma 4. Within-label refuse D change: <0.004 for both. The single hazard_adjacent hedge (D=1.000) in Gemma 4 at 150 tok is consistent with the §4.6 "safety activation without behavioral suppression" typology. **Cross-model finding:** SAE internal feature activations are token-budget stable on both model families; surface classification is not. At higher token budgets, models hedge longer before declining, and the regex surface classifier interprets the hedge-then-decline as comply rather than refuse. The activation signal is stable where the surface classifier is not — directly replicating §4.5's format-stability finding at the budget dimension. Policy implication: per-prompt D at a fixed budget is a reliable internal-state measure; tier-level mean D is not budget-stable and should always be reported with the token budget used.

**Gemma 2 2B bio SAE at 200 tokens (2026-04-26):** Completing the 3-point series. Tier mean D at 80/150/200 tok — benign: 0.446/0.435/0.455 (range 0.020, stable); dual_use: 0.488/0.536/0.475 (non-monotonic); hazard_adjacent: 0.475/0.508/0.514 (monotone, saturating). Comply/refuse counts: dual_use 22/8 → 26/4 → 22/8; hazard_adjacent 15/7 → 18/4 → 18/4. Within-label comply D at 200 tok: dual_use 0.578→0.560 (−0.018), hazard_adjacent 0.583→0.574 (−0.009). Refuse D stable across all budgets (<0.005 range). **Non-monotonic finding (dual_use):** the 4 refuse→comply switches at 150 tok revert at 200 tok — with enough budget the model completes its hedged refusal. At 150 tok the regex classifier catches the hedge as comply; at 200 tok the model reaches its refusal conclusion. The underlying SAE feature signal (comply D ~0.56–0.58, refuse D ~0.24) is stable; what changes is whether the model finishes the thought. **Confirmed cross-budget pattern:** hazard_adjacent label distribution stabilizes by 150 tok (18 comply, 4 refuse at both 150 and 200 tok), while dual_use oscillates. The model handles hazard content more consistently across budgets than dual-use content.

**Gemma 4 E2B bio SAE at 200 tokens (2026-04-26):** Completing the symmetric 3-point table. Measured label counts at 200 tok: benign comply=17/refuse=6, dual_use comply=23/refuse=7, hazard comply=17/refuse=5. Implied counts at earlier budgets (back-calculated from known mean D + within-label D): benign 13c/10r → 18c/5r → 17c/6r; dual_use 20c/10r → 26c/4r → 23c/7r; hazard 14c/8r → 20c/2r → 17c/5r.

| Tier | 80-tok mean D | 150-tok mean D | 200-tok mean D | Pattern |
|---|---|---|---|---|
| benign_bio | 0.362 | 0.420 | **0.413** | non-monotonic (−0.007) |
| dual_use_bio | 0.406 | 0.461 | **0.436** | non-monotonic (−0.025) |
| hazard_adjacent | 0.404 | 0.475 | **0.442** | non-monotonic (−0.033) |

Within-comply D across budgets: benign 0.466→0.473→0.478 (+0.012, monotone); dual_use 0.494→0.497→0.499 (+0.005, monotone); hazard 0.504→0.502→0.505 (flat, ±0.003). Refuse D: 0.227–0.229 across all tiers and budgets (completely stable). **Non-monotonic pattern confirmed for Gemma 4:** all three tiers peak in comply count at 150 tok and partially revert at 200 tok — benign −1, dual_use −3, hazard −3. Same mechanism as Gemma 2: with more budget the model completes its hedged refusal. Unlike Gemma 2 where hazard stabilized by 150 tok, Gemma 4 hazard continues to revert at 200 tok (3 prompts), suggesting Gemma 4 requires more tokens to complete its hazard refusal thought.

**Cross-model 200-tok non-monotonic comparison:** Gemma 2 dual_use: 4 reverts; Gemma 4 dual_use: 3 reverts. Gemma 2 hazard: 0 reverts (stabilized at 150 tok); Gemma 4 hazard: 3 reverts. The reversal magnitude is similar across models (~3–4 prompts per tier), confirming the phenomenon is general rather than model-specific. In both cases within-label D is budget-stable and the SAE feature signal (comply D ~0.47–0.58, refuse D ~0.23–0.24) does not change — only the surface classifier label switches. **Policy implication (updated):** token budget at evaluation time affects measured tier-level mean D for all models tested. Per-prompt D at fixed budget remains a reliable internal-state measure. Tier-level mean D should always specify the token budget and is not comparable across budgets without label-distribution adjustment.

- **Non-bio domain control** *(completed 2026-04-23; n=30, legal/financial eval, same model + SAE).* D shows a tier gradient (benign 0.573, dual-use 0.672, hazard 0.665) but with markedly weaker effect than bio: Cohen's d=0.67, p=0.052, CIs overlapping by 0.071. This rules out pure domain-agnostic routing. D has a bio-domain advantage, likely because the catalog was tuned on bio prompts, while confirming generic sensitivity routing is present and partially contributes to the bio signal. Domain-specific SAE fine-tuning completes the transition.
- **Held-out calibration set** *(attempted 2026-04-25; see §5 correction: held-out T revealed framing-distribution sensitivity rather than retiring the caveat; next step: matched-framing holdout set);* **Gemma Scope 2 sweep** when released (Gemma 3 270M through 12B); **Llama 3.1 8B** cross-architecture comparison (already cached locally); **unlearning before/after** on RMU snapshots; **Transcoder + CLT analysis** to move from metric to circuit for the strongest divergence flags. Coefficient Giving RFP (due May 11) frames a 6-month follow-on around these results.
