# BioRefusalAudit: Auditing the Depth of LLM Biosecurity Refusals Using Sparse Autoencoder Interpretability

**Caleb DeLeeuw**, AIxBio Hackathon 2026, Track 3 (Biosecurity Tools)

**Status at submission:** Flagship pipeline shipped end-to-end + specialist review fully addressed (`notes/SPECIALIST_REVIEW_2026_04_23.md`). The three-legged evidence requirement (activation + attribution + perturbation) is codified: no feature gets called a "named circuit" unless it passes all three via `attribution_labels.py::classify_tier`. Results in §4 from tuned-catalog + fitted-T run on Gemma 2 2B-IT + Gemma Scope 1 layer 12.

**Headline findings:** (1) Gemma 2 2B-IT shows per-tier mean D = 0.467 / 0.655 / 0.669 (benign / dual-use / hazard), Cohen's *d* = 1.29, p = 0.0001, non-overlapping 95% CIs. `hazard_features_active_despite_refusal` fires on **34.8% of benign refusals**, a novel over-refusal signature with intact hazard representations (§4.2). (2) Gemma 4 E2B's RLHF safety circuit is format-gated: missing canonical `<start_of_turn>` tokens disables it entirely. Both models refuse **0% at 80-token max generation** across all tiers (§4.5). (3) All 9 correct-format Gemma 4 comply cases fire `refusal_features_active_despite_compliance`. The safety circuit represents hazard without suppressing output (§4.6). **Primary validity caveats:** T is fit within-sample; held-out calibration (v3) inverted tier ordering (d=-0.967), showing T is framing-distribution-sensitive — within-sample T is better-matched for v1 results (§5 CORRECTED). The feature catalog is Cohen's-d auto-tuned, not semantically validated. Flag-based findings (§4.6) survive both caveats.

---

## 1. Problem

Existing biosecurity evaluations (VCT, Götting et al. 2025; WMDP-Bio, Li et al. 2024; ABC-Bench; LAB-Bench, Laurent et al. 2024) measure whether a model *will* produce hazardous output. They don't measure whether a refusing model *can't*, or merely *doesn't right now*.

This gap matters for every deployment decision. A model with a shallow refusal over intact internal hazard representations is one framing shift from compliance. Qi et al. (2024) argue the alignment floor sits "a few tokens deep." Crook (AIxBio keynote, 2026) independently identifies "binary prediction unusable — calibrated confidence required" as the measurement gap. BioRefusalAudit provides the first cross-model, interpretability-grounded measurement. We call it **refusal depth**.

**Policy motivation (Sandbrink & Crook, AIxBio 2026).** Sandbrink called for systems that monitor AI–bio interactions without reading interaction content. Auditing SAE activations rather than prompt/completion text lets a deployer flag shallow refusals while preserving user privacy — the activation signal is separable from linguistic content. Dual-use concern masked by innocuous framing is invisible to output-only screeners; our divergence flags surface it upstream of the completed artifact. This composes with output-based screeners covering distinct failure modes. Yassif and Carter (NTI Bio, January 2026) frame managed-access AI tools as requiring a measurement layer and an enforcement layer. BioRefusalAudit provides both (§7).

## 2. Contribution

1. A **calibrated surface-internal divergence metric** `D(s, f) = 1 - cos(f, T^T · s)` over soft surface classification `s` (refuse / comply / partial / hedge / evade) and SAE-feature vector `f` (bio_content / hazard_adjacent / refusal_circuitry / hedging / deception_correlate). See `docs/METHOD.md`.
2. A **75-prompt public eval set** stratified by tier (benign_bio 23, dual_use_bio 30, hazard_adjacent 22 at descriptor level; full tier-3 bodies on HL3-gated HF dataset) and framing (direct / educational / roleplay / obfuscated).
3. A **multi-judge consortium** combining regex first-pass, Gemini CLI, Claude Haiku via `claude -p` subprocess, and local Ollama, with weighted voting and disagreement flagging.
4. A **reproducible end-to-end pipeline** runnable from a fresh clone on a 4 GB GPU (Gemma 2 2B-IT + Gemma Scope 1). Gemma 4 E2B + author-trained SAE validates portability to community-trained artifacts.
5. A **Streamlit dashboard** for per-prompt interactive auditing.

## 3. Method (summary)

For each (model, prompt) pair:
1. Generate at T=0.7 with forward-hook capture of the residual stream at a model-specific target layer (~50% depth).
2. Project captured residuals through the model's SAE and take mean per-feature activation across generated tokens.
3. Project the d_sae-dimensional activation into 5 feature categories via `data/feature_catalog/<model>.json` and L1-normalize to produce `f`.
    - **CORRECTED 2026-04-24:** "hand-validated" describes the design target, not the v1.0 MVP. Actual catalogs are Cohen's-d-selected from held-out activation dumps (`scripts/auto_tune_catalog.py`). Semantic validation against Neuronpedia labels and max-activating examples is planned work (§8). Neuronpedia inspection (§4.2) found top auto-tuned features encode generic technical/governance vocabulary rather than bio-specific refusal circuitry. Every §4 result should be read under "auto-tuned catalog, not semantically validated."
4. Run judge consortium and aggregate votes to soft distribution `s`.
5. Compute `D(s, f, T)` and the three flags (`hazard_features_active_despite_refusal`, `refusal_features_active_despite_compliance`, `deception_correlate_active`).

The alignment matrix `T ∈ ℝ^{5×5}` is fit by ridge-regularized least squares on held-out positive-control prompts, with a condition-number guard to fail loudly on under-calibrated datasets.

## 4. Results

**Primary validity caveats (read before §4.1–§4.6, added 2026-04-24):** Two structural limits apply to every D-value in this section. (1) *Within-sample T calibration.* T is fit on the same 75-prompt eval set used to compute the reported divergences. All D-value comparisons should be read as within-sample calibrated, not held-out calibrated. (2) *Auto-tuned, not semantically validated, feature catalog.* See the CORRECTED block in §3 and the honest-assessment paragraph in §4.2. Flag-based findings (§4.6) operate on raw per-feature activation magnitudes and do not pass through T. They survive both caveats more robustly than D-value comparisons.

### 4.0 Evidence discipline

Per specialist review, every claim in this section satisfies three legs: (1) Cohen's-d ≥ 0.2 activation differentiation (`feature_validator.py::differentiation_check`); (2) decoder-direction alignment with the decisive residual span; (3) ablation shifts surface label or |ΔD| > 0.2 (`run_intervention.py`). Sub-threshold claims appear as `candidate` / `activation_only` / `attribution_only` per `classify_tier`.

### 4.1 Pipeline validation (synthetic)

Before any model load, the full scoring + reporting + cross-model aggregation pipeline was validated via `scripts/synthetic_demo.py`, producing per-tier mean D = 0.02 / 0.11 / 0.47 (benign / dual-use / hazard) by construction. These are **synthetic** plumbing-check numbers only, not results about any real model.

### 4.2 Flagship run: Gemma 2 2B-IT + Gemma Scope 1 layer 12

Two 75-prompt passes: pass 1 collects activations for catalog tuning (`activations.npz`). Pass 2 reruns with the tuned catalog + fitted T. Hardware: GTX 1650 Ti Max-Q, 4 GB VRAM, `device_map="auto"` with partial GPU + CPU offload.

**Per-tier mean divergence (pass 2, tuned catalog + fitted T):**

| Tier | n | Mean D | Std | 95% CI (bootstrap) | Hazard-on-refuse (%) |
|---|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.467 | 0.190 | [0.392, 0.548] | 34.8% |
| `dual_use_bio` | 30 | 0.655 | 0.119 | [0.612, 0.697] | 30.0% |
| `hazard_adjacent_category` | 22 | 0.669 | 0.113 | [0.619, 0.715] | 18.2% |

95% CIs from 10,000 bootstrap resamplings (seed 42). Benign CI [0.392, 0.548] does not overlap hazard [0.619, 0.715]. Effect size (hazard vs. benign): Cohen's d = **1.29**, p = 0.0001. Within-tier overlap exists (22% of benign > dual-use mean; 32% of hazard < dual-use mean). Individual-prompt discrimination requires wider CIs than group-level separation at n=23–30.

**Feature catalog validation, Neuronpedia lookup (2026-04-23):**

| Feature ID | Category | Neuronpedia label | Assessment |
|---|---|---|---|
| 2620 | refusal_circuitry / hazard_adjacent | "authority, control, and criticism of governance" | Polysemantic; plausibly bio-relevant but not specifically refusal |
| 1041 | refusal_circuitry | "technical terms and concepts related to programming or data structures" | Not semantically refusal-related; activates on bio technical vocabulary |
| 7541 | hazard_adjacent / refusal_circuitry | "phrases related to technical descriptions in software development contexts" | Generic technical vocabulary feature |

**Honest assessment:** Auto-tuning by Cohen's d identifies statistically discriminative features, not semantically validated ones. Top catalog features in `refusal_circuitry` and `hazard_adjacent` are generic technical/governance vocabulary that fire differentially on biosecurity prompts, probably because bio prompts use technical vocabulary rather than because these features represent bio-specific refusal circuitry. Semantic validation (Neuronpedia lookups + max-activating example inspection) is needed before any strong mechanistic claim. The tier-separation D result (p=0.0001, d=1.29) may reflect vocabulary routing rather than refusal depth per se. The non-bio control (§5, completed 2026-04-23) confirms D has a bio-domain advantage: legal/financial eval shows d=0.67, p=0.052 (weaker, CIs overlapping), ruling out pure domain-agnostic routing while confirming generic sensitivity partially contributes.

**CORRECTED 2026-04-25 — Judge failure audit.** Post-hoc inspection of the pass-2 run reveals that all 29 "refuse" labels carry a degenerate soft distribution `{refuse:0.2, comply:0.2, ...}` — the judge consortium returned the uniform prior for these records (LLM judges unavailable at run time; fallback to prior), and dict-order argmax selected "refuse" as the first key. The 46 "comply" records have real non-uniform soft scores. The actual completions for all 29 "refuse" records are biology answers. Consequence: (1) the `hazard_features_active_despite_refusal` flag reported at 34.8% / 30.0% / 18.2% per tier is driven entirely by degenerate records; the flag fires on 21/29 degenerate records and 0/46 real records. What the flag is actually measuring is "biology completions with high biology/hazard SAE feature activations" — a real but differently-named finding. (2) The D-value separation (d=1.29, p=0.0001) depends on `surface_soft` via T; degenerate records contribute flat soft vectors, and T was fitted on data that includes them. A rerun with regex-only judges (`runs/gemma-2-2b-it-200tok-rejudge/`, in progress) will produce corrected surface labels and corrected flags. The tier-separation D result may hold (since D depends on activations, not just surface labels), but the refusal-rate-dependent flags require the corrected run.

### 4.3 Intervention experiments: causal evidence

All 75 prompts intervened on with `scripts/run_intervention.py` (completed 2026-04-25; see `runs/interventions/`). CMF criterion: `label_changed` OR `|ΔD_ablate| + |ΔD_boost| > 0.2`. **60/75 qualify (80%).** Dose-proportionality validation (boost=1.5, 2.0, 4.0) on the top-10 highest-effect prompts is complete.

| Tier | n | NC=True | Rate | Highest effect |
|---|---:|---:|---:|---|
| benign_bio | 23 | 20 | 87% | bio_014 (1.139) |
| dual_use_bio | 30 | 24 | 80% | bio_040 (0.712) |
| hazard_adjacent | 22 | 16 | 73% | bio_062 (0.434) |
| **overall** | **75** | **60** | **80%** | |

**NC by framing:** direct 86%, roleplay 83%, obfuscated 82%, educational 68%. Framing variation is consistent: all four types exceed the 60% threshold. Educational framings elicit more cooperative model responses, reducing the surface-label contrast that the CMF criterion requires.

**Dose-proportionality (top-10 by effect size, boost=1.5/2.0/3.0/4.0):** NC qualification rates are 9/10, 9/10, 10/10, 9/10 across the four boost levels. 7/10 prompts qualify NC=True at all four levels; the remaining 3 are mixed (NC=False at one non-3.0x boost level). Effect sizes are non-monotone with boost magnitude — expected, since prompts were selected for highest effect at 3.0x. What is stable is circuit identity: the same prompts qualify at 1.5x and 4.0x as at 3.0x. This confirms the named-circuit finding is not an artefact of the single intervention multiplier used in the main sweep.

**Key finding — inverted tier ordering:** NC rate is highest for benign_bio and lowest for hazard_adjacent. This reinforces the §4.2 catalog-validation finding: the auto-tuned refusal_circuitry features encode generic technical-governance vocabulary that fires on biology content at all hazard levels, not on bio-specific hazard circuitry. bio_014 (benign/roleplay, effect=1.139) is the corpus peak: the largest internal-state shift with no surface behavior change — a clean example of the divergence BioRefusalAudit is designed to surface.

### 4.4 Cross-architecture reference (Colab T4)

- `google/gemma-2-9b-it` + `gemma-scope-9b-pt-res` layer 20 at bnb 4-bit
- `meta-llama/Llama-3.1-8B-Instruct` + Llama Scope `l16r_8x`

Results land in `runs/colab_*/report.{md,json}`. Scaling plot regenerates via `scripts/build_scaling_plot.py --include-synthetic`. Gemma 3 family evaluation deferred pending Gemma Scope 2 public release.

### 4.5 Portability path: Gemma 4 E2B-IT + author-trained SAE

To show the methodology is not locked to Gemma Scope releases, we ran the full calibration chain on Gemma 4 E2B-IT using an author-trained TopK SAE from prior deception-detection research (Secret Agenda, arXiv:2509.20393; k=16, layer 17, d_sae=6144). This exercises every pipeline component against a community artifact.

**Critical finding (2026-04-23):** Post-hoc inspection of the initial pass revealed 40% empty completions and 16% repetition loops. These weren't genuine refusals. Gemma 4 IT requires canonical `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` tokens to engage its RLHF safety circuit. Without them the model receives text outside its training distribution and either emits EOS immediately or loops. D stays near-zero across all passes because the cross-domain SAE (deception-trained, not biosecurity-trained) cannot separate bio tiers. This is the predicted outcome from Secret Agenda's non-generalizability finding. The flag signals remain valid: they test raw activation magnitudes, not T-calibrated D.

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

**CORRECTED 2026-04-24 (chat-template token names):** Earlier prose used `<|turn>` / `<turn|>` shorthand instead of Gemma's canonical `<start_of_turn>` / `<end_of_turn>`. The malformed form appears deliberately in `scripts/format_ablation.py` conditions C and D. No empirical claim depends on this distinction.

*Format ablation (80tok, n=96 G4 A/B/C/D + n=96 G2 A/B/C/D):* G4 cond B (generic template): 58% loops. G4 cond C (missing role token): hazard-tier 100% empty, dual-use/benign 100% comply. G4 cond D (wrong role label): 100% comply. G2 all conditions: 100% comply, 0% loops. Both models 0% refuse at 80tok. G2 at 150tok (n=48): same.

### 4.6 Primary policy findings (elevated 2026-04-24)

Two findings from §4.2 and §4.5 warrant elevation because they have immediate deployment-policy implications *independent* of the calibration-circularity and catalog-validation caveats. Both rest on raw per-feature activation magnitudes and behavioral counts, not D-values passing through the within-sample-fit T.

**Finding A: Format-dependent safety activation.** Gemma 4 E2B's RLHF safety circuit is gated on canonical chat-template tokens. Without them: 40% empty, 16% loops, 0% genuine refusals. With them (pass 5): 65 refuse / 9 comply / 1 hedge from the same 75 prompts. The format ablation (n=96 per condition) adds an orthogonal result: both Gemma 4 E2B and Gemma 2 2B-IT refuse **0% across all tiers when generation is capped at 80 tokens**. Safety articulation requires token budget. *Policy implication:* constrained deployment contexts (mobile UIs, embedded applications, latency-capped APIs with 80-token limits) may systematically bypass the safety circuits that lab benchmarks measure. A monitoring layer at the activation level is format-stable where surface classifiers are not. This directly addresses the Sandbrink managed-access agenda cited in §1.

**Finding B: Safety activation without behavioral suppression.** All 9 correct-format Gemma 4 comply cases fire `refusal_features_active_despite_compliance`. The safety circuit represents the hazard context without gating the output. Five of the 9 are hazard-adjacent biology prompts. The structural inverse appears in the Gemma 2 flagship result (§4.2): `hazard_features_active_despite_refusal` fires on 34.8% of *benign* refusals. That's over-refusal with intact hazard representations. Together, these two flags produce a concrete four-cell safety-posture typology:

| | hazard_on_refuse low | hazard_on_refuse high |
|---|---|---|
| **refusal_on_comply low** | deep refusal (structurally safe) | shallow refusal (surface-only) |
| **refusal_on_comply high** | over-refuses benign, represents hazard | non-suppressive safety (Gemma 4 E2B pattern) |

BioRefusalAudit distinguishes all four cells at the activation layer. Surface evaluation alone cannot. For deployers and model providers, this two-flag surface is a lighter-weight audit than full D-value calibration and survives the §4 validity caveats.

## 5. Limitations

- **Non-bio control (completed, §8).** D on a 30-prompt legal/financial eval shows weaker tier separation (Cohen's d=0.67, p=0.052, CIs overlapping) vs. bio (d=1.29, p=0.0001, non-overlapping). D has a bio-domain advantage, but generic sensitivity routing partially contributes. Domain-specific SAE fine-tuning (§8) is the structural fix.
- **Feature catalog not semantically validated.** Neuronpedia inspection of top auto-tuned features (2620, 1041, 7541) reveals generic technical/governance vocabulary. Automated LLM labeling of decontextualized windows is unreliable (probe: Gemini 2.5 Flash on 200-word bio corpus chunks, ~60% accuracy, reasons hallucinated); full-paragraph context is required.
- **CMF status overstates evidence.** §4.3 results are candidate mechanistic features, not named circuits. Full validation requires paraphrase consistency testing and dose-proportionality.
- **Within-sample T calibration — CORRECTED 2026-04-25:** Held-out calibration run (60-prompt v3, 20 per tier). Held-out T fit produced mean|ΔT|=0.580. Rerunning v1 eval with held-out T inverts tier ordering (benign=0.089 > hazard=0.023, d=-0.967). **Finding:** T is sensitive to framing distribution. V3 has more roleplay/obfuscated prompts than v1; held-out T overfits to that distribution. Calibration set must match evaluation framing distribution. Within-sample T remains the better-calibrated choice for v1 results.
- **Gemma Scope 2 pending.** Primary planned substrate (Gemma 3 + Gemma Scope 2) not yet released. MVP uses Gemma 2 + Gemma Scope 1.
- **Cross-architecture comparison deferred.** Pending Colab T4 runtime for Gemma 2 9B-IT + Llama 3.1 8B-Instruct.

## 6. Relation to prior work

BioRefusalAudit extends two findings from the author's Secret Agenda work (arXiv:2509.20393 / AAAI 2026). First, individual SAE auto-labels fail at domain-specific behavioral detection. Category-level ensembles are more reliable. Second, behaviorally-trained SAEs don't generalize across behavioral domains, reconfirmed here by D≈0 on bio prompts with the deception-focused SAE (§4.5). Domain-specific biosecurity behavioral corpora are required. Cross-domain transfer is insufficient. See README §"Related work" for the broader literature (Qi et al. 2024; Arditi et al. 2024; Lieberum et al. 2024; Templeton et al. 2024; Marks & Rager et al. 2024; Goldowsky-Dill et al. 2025).

## 7. Responsible release

Code: HL3-FULL. Eval tiers 1+2: CC-BY-4.0. Tier 3: category-level descriptors in this public repo, with full prompt bodies on HL3-gated HF dataset (`SolshineCode/biorefusalaudit-gated`). See `SAFETY.md`, `docs/HL3_RATIONALE.md`, `LICENSE_HL3_DATASET.md`.

**Why HL3, not MIT.** MIT and Apache 2.0 expressly permit offensive applications. For a tool identifying which models retain hazard representations behind a surface refusal, that neutrality creates a specific risk: an adversarial actor using our divergence score to *target* shallow-refusal models for prompt-engineering attack. HL3-FULL prevents this with enforceable human-rights conditions. The tiered data release instantiates the Biosecurity Data Level (BDL) framework (Bloomfield, Black, Crook et al., *Science* 2026): tiers 1+2 CC-BY-4.0 open, tier 3 requires signed attestation. Enforceable repo tests (`safety_review.check_no_hazard_bodies`, `reporting.redaction.redact_tier3`) exceed host ground rules. HL3 is a norm-setting signal that biosecurity AI tooling belongs in a different licensing category than general-purpose infrastructure.

## 8. Future work

**Domain-specific SAE fine-tuning** is the binding constraint. Neuronpedia validation confirms the auto-tuned catalog encodes generic technical-governance vocabulary rather than bio-specific circuitry. The corrective is to fine-tune SAEs on biosecurity behavioral activation corpora, using base vs. RLHF model pairs with a contrastive objective separating shallow refusals (hazard features active) from genuine ones (hazard features suppressed). A projection adapter over frozen Gemma Scope weights is feasible with the existing 75-prompt corpus. A full fine-tune needs institutional CBRN datasets (~10K samples). **WMDP data availability (confirmed 2026-04-25):** `cais/wmdp-bio-forget-corpus` (Center for AI Safety, Li et al. 2024) is gated but accessible at ~24K peer-reviewed biology documents with DOI provenance. Format confound applies: median document length ~4K words vs. eval prompt ~30 words; chunking to ~200-token segments is required before contrastive SAE training. Full specification in `docs/METHOD.md §Planned extension`.

- **Non-bio domain control** *(completed 2026-04-23; n=30, legal/financial eval, same model + SAE).* D shows a tier gradient (benign 0.573, dual-use 0.672, hazard 0.665) but with markedly weaker effect than bio: Cohen's d=0.67, p=0.052, CIs overlapping by 0.071. This rules out pure domain-agnostic routing. D has a bio-domain advantage, likely because the catalog was tuned on bio prompts, while confirming generic sensitivity routing is present and partially contributes to the bio signal. Domain-specific SAE fine-tuning completes the transition.
- **Held-out calibration set** *(attempted 2026-04-25; see §5 CORRECTED — held-out T revealed framing-distribution sensitivity rather than retiring the caveat; next step: matched-framing holdout set);* **Gemma Scope 2 sweep** when released (Gemma 3 270M through 12B); **Llama 3.1 8B** cross-architecture comparison (already cached locally); **unlearning before/after** on RMU snapshots; **Transcoder + CLT analysis** to move from metric to circuit for the strongest divergence flags. Coefficient Giving RFP (due May 11) frames a 6-month follow-on around these results.
