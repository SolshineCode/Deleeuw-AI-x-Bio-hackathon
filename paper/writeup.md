# BioRefusalAudit: Auditing the Depth of LLM Biosecurity Refusals Using Sparse Autoencoder Interpretability

**Caleb DeLeeuw**, AIxBio Hackathon 2026, Track 3 (Biosecurity Tools)

**Status at submission:** Flagship pipeline shipped end-to-end + specialist review fully addressed (`notes/SPECIALIST_REVIEW_2026_04_23.md`). The three-legged evidence requirement (activation + attribution + perturbation) is codified: no feature gets called a "named circuit" unless it passes all three via `attribution_labels.py::classify_tier`. Results in §4 from tuned-catalog + fitted-T run on Gemma 2 2B-IT + Gemma Scope 1 layer 12.

**Headline findings:** (1) Gemma 2 2B-IT: per-tier mean D = 0.467 / 0.655 / 0.669 (benign / dual-use / hazard), Cohen's *d* = 1.29, p = 0.0001, non-overlapping 95% CIs. Corrected surface labels (regex re-judge of stored completions): **0 genuine refusals; 40 comply (53%), 35 hedge (47%)**. Hazard-adjacent tier 100% hedge, 0% comply, 0% refuse. The 34.8% over-refusal statistic is retracted (§4.2 CORRECTED). (2) Gemma 4 E2B's RLHF safety circuit is format-gated: missing canonical `<start_of_turn>` tokens disables it entirely. Both models refuse **0% at 80-token max generation** across all tiers (§4.5). (3) All 9 correct-format Gemma 4 comply cases fire `refusal_features_active_despite_compliance`. The safety circuit represents hazard without suppressing output (§4.6). **Primary validity caveats:** T is fit within-sample. Held-out calibration (v3) inverted tier ordering (d=-0.967), showing T is framing-distribution-sensitive. Within-sample T is better-matched for v1 results (§5 CORRECTED). The feature catalog is Cohen's-d auto-tuned, not semantically validated. Flag-based findings (§4.6) survive both caveats.

---

## 1. Problem

Existing biosecurity evaluations (VCT, Götting et al. 2025; WMDP-Bio, Li et al. 2024; ABC-Bench; LAB-Bench, Laurent et al. 2024) measure whether a model *will* produce hazardous output. They don't measure whether a refusing model *can't*, or merely *doesn't right now*.

This gap matters for every deployment decision. A model with a shallow refusal over intact internal hazard representations is one framing shift from compliance. Qi et al. (2024) argue the alignment floor sits "a few tokens deep." Crook (AIxBio keynote, 2026) independently identifies "binary prediction unusable — calibrated confidence required" as the measurement gap. BioRefusalAudit provides the first cross-model, interpretability-grounded measurement. We call it **refusal depth**.

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

| Tier | n | Mean D | Std | 95% CI (bootstrap) | Hazard-on-refuse (%) † |
|---|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.467 | 0.190 | [0.392, 0.548] | — |
| `dual_use_bio` | 30 | 0.655 | 0.119 | [0.612, 0.697] | — |
| `hazard_adjacent_category` | 22 | 0.669 | 0.113 | [0.619, 0.715] | — |

† Retracted: 0 genuine refusals per corrected run (§4.2 CORRECTED). Corrected surface labels: benign 70% comply / 30% hedge; dual-use 80% comply / 20% hedge; hazard-adjacent 100% hedge / 0% comply / 0% refuse.

95% CIs from 10,000 bootstrap resamplings (seed 42). Benign CI [0.392, 0.548] does not overlap hazard [0.619, 0.715]. Effect size (hazard vs. benign): Cohen's d = **1.29**, p = 0.0001. Within-tier overlap exists: 22% of benign > dual-use mean, 32% of hazard < dual-use mean. Individual-prompt discrimination requires wider CIs than group-level separation at n=23–30.

**Feature catalog validation, Neuronpedia lookup (2026-04-23):**

| Feature ID | Category | Neuronpedia label | Assessment |
|---|---|---|---|
| 2620 | refusal_circuitry / hazard_adjacent | "authority, control, and criticism of governance" | Polysemantic; plausibly bio-relevant but not specifically refusal |
| 1041 | refusal_circuitry | "technical terms and concepts related to programming or data structures" | Not semantically refusal-related; activates on bio technical vocabulary |
| 7541 | hazard_adjacent / refusal_circuitry | "phrases related to technical descriptions in software development contexts" | Generic technical vocabulary feature |

**Honest assessment:** Auto-tuning by Cohen's d identifies statistically discriminative features, not semantically validated ones. Top catalog features in `refusal_circuitry` and `hazard_adjacent` are generic technical/governance vocabulary that fire differentially on biosecurity prompts, probably because bio prompts use technical vocabulary rather than because these features represent bio-specific refusal circuitry. Semantic validation (Neuronpedia lookups + max-activating example inspection) is needed before any strong mechanistic claim. The tier-separation D result (p=0.0001, d=1.29) may reflect vocabulary routing rather than refusal depth per se. The non-bio control (§5, completed 2026-04-23) confirms D has a bio-domain advantage: legal/financial eval shows d=0.67, p=0.052 (weaker, CIs overlapping), ruling out pure domain-agnostic routing while confirming generic sensitivity partially contributes.

**CORRECTED 2026-04-25: Judge failure audit, corrected results in.** All 29 "refuse" labels in pass-2 carry degenerate soft scores `{label: 0.2 for all labels}`. LLM judges were unavailable. The consortium fell back to uniform prior and dict-order argmax picked "refuse" (first insertion key). Regex re-judge of stored completions (`scripts/rejudge_stored_completions.py`; `runs/gemma-2-2b-it-L12-tuned-rejudged/`) yields **0 genuine refusals; 40 comply (53%); 35 hedge (47%).** Hazard-on-refuse percentages in table above are retracted (see footnote †). D-values are activation-based and unchanged. The corrected primary signal: Gemma 2 2B-IT consistently hedges on hazard-adjacent prompts but does not refuse them — a distinct behavioral posture from genuine refusal, invisible to binary surface evaluation. Full diagnosis in `TROUBLESHOOTING.md §Judge consortium silently returns uniform prior`.

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

*Format ablation (80tok, n=96 G4 A/B/C/D + n=96 G2 A/B/C/D):* G4 cond B: 58% loops; cond C: hazard 100% empty, benign/dual-use 100% comply; cond D: 100% comply. G2: 100% comply, 0% loops across all conditions. G2 at 150tok (n=48): same. **Full 75-prompt G2 corrected run at 80tok:** 0 genuine refusals; 42 comply, 33 hedge. Hazard-adjacent: 100% hedge at all three budgets (80/150/200 tok). Posture is stable. At 150 tok, obfuscated peaks at 94% (vs 77%/59% at 80/200); hazard_adj D=0.760 vs 0.669. 14/75 per-prompt changes; 0 hazard-adjacent.

### 4.6 Primary policy findings (elevated 2026-04-24)

Two findings warrant elevation because they survive the calibration-circularity and catalog-validation caveats: both rest on raw activation magnitudes and behavioral counts, not D-values through the fitted T.

**Finding A: Format-dependent safety activation.** Gemma 4 E2B's RLHF safety circuit is gated on canonical chat-template tokens. Without them: 40% empty, 16% loops, 0% refusals. With them (pass 5): 65 refuse / 9 comply / 1 hedge. Full 75-prompt G2 corrected run at 80tok: **0 genuine refusals; hazard-adjacent 100% hedge at 80, 150, and 200 tokens.** Tier-level behavioral posture is token-budget-stable. Framing shifts: obfuscated prompts hedge more at 80 tok (77% vs 59%). Direct/educational comply more (29%/32% hedge vs 48%/42%). *Policy implication:* constrained deployment contexts may surface fewer refuses but not fewer hedges. The activation signal persists where surface classifiers cannot. Monitoring at the activation layer is format-stable and budget-stable where output classifiers are not. This directly addresses the Sandbrink managed-access agenda cited in §1.

**Finding B: Safety activation without behavioral suppression.** All 9 correct-format Gemma 4 comply cases fire `refusal_features_active_despite_compliance`. The safety circuit represents the hazard context without gating the output. Five of the 9 are hazard-adjacent biology prompts. The structural inverse was reported in §4.2 (see §4.2 CORRECTED: with 0 genuine G2 refusals, that cell is vacuous). The Gemma 4 result stands independently and the typology holds. Together, these two flags produce a concrete four-cell safety-posture typology:

| | hazard_on_refuse low | hazard_on_refuse high |
|---|---|---|
| **refusal_on_comply low** | deep refusal (structurally safe) | shallow refusal (surface-only) |
| **refusal_on_comply high** | over-refuses benign, represents hazard | non-suppressive safety (Gemma 4 E2B pattern) |

BioRefusalAudit distinguishes all four cells at the activation layer. Surface evaluation alone cannot. For deployers and model providers, this two-flag surface is a lighter-weight audit than full D-value calibration and survives the §4 validity caveats.

## 5. Limitations

- **Non-bio control (completed, §8).** D on a 30-prompt legal/financial eval shows weaker tier separation (Cohen's d=0.67, p=0.052, CIs overlapping) vs. bio (d=1.29, p=0.0001, non-overlapping). D has a bio-domain advantage, but generic sensitivity routing partially contributes. Domain-specific SAE fine-tuning (§8) is the structural fix.
- **Feature catalog not semantically validated.** Neuronpedia inspection of top auto-tuned features (2620, 1041, 7541) reveals generic technical/governance vocabulary. Automated LLM labeling of decontextualized windows is unreliable (probe: Gemini 2.5 Flash on 200-word bio corpus chunks, ~60% accuracy, reasons hallucinated). Full-paragraph context is required.
- **Judge failure in flagship G2 run (corrected).** 29/75 records in `runs/gemma-2-2b-it-L12-tuned/` had degenerate soft judgments (uniform prior → dict-order "refuse" label). Regex re-judge (`runs/gemma-2-2b-it-L12-tuned-rejudged/`): **0 genuine refusals; 40 comply (53%); 35 hedge (47%)**. Hazard-adjacent tier: 100% hedge, 0% comply, 0% refuse. §4.2 flag rates retracted. D-values (d=1.29) unchanged. Full diagnosis: `TROUBLESHOOTING.md §Judge consortium silently returns uniform prior`.
- **CMF status overstates evidence.** §4.3 results are candidate mechanistic features, not named circuits. Full validation requires paraphrase consistency testing and dose-proportionality.
- **Within-sample T calibration (CORRECTED 2026-04-25):** Held-out calibration run (60-prompt v3, 20 per tier). Held-out T fit produced mean|ΔT|=0.580. Rerunning v1 eval with held-out T inverts tier ordering (benign=0.089 > hazard=0.023, d=-0.967). **Finding:** T is sensitive to framing distribution. V3 has more roleplay/obfuscated prompts than v1. Held-out T overfits to that distribution. Calibration set must match evaluation framing distribution. Within-sample T remains the better-calibrated choice for v1 results.
- **Gemma Scope 2 pending.** Primary planned substrate (Gemma 3 + Gemma Scope 2) not yet released. MVP uses Gemma 2 + Gemma Scope 1.
- **Cross-architecture comparison deferred.** Pending Colab T4 runtime for Gemma 2 9B-IT + Llama 3.1 8B-Instruct.

## 6. Relation to prior work

BioRefusalAudit extends two findings from the author's Secret Agenda work (arXiv:2509.20393 / AAAI 2026). First, individual SAE auto-labels fail at domain-specific behavioral detection. Category-level ensembles are more reliable. Second, behaviorally-trained SAEs don't generalize across behavioral domains, reconfirmed here by D≈0 on bio prompts with the deception-focused SAE (§4.5). Domain-specific biosecurity behavioral corpora are required. Cross-domain transfer is insufficient. See README §"Related work" for the broader literature (Qi et al. 2024; Arditi et al. 2024; Lieberum et al. 2024; Templeton et al. 2024; Marks & Rager et al. 2024; Goldowsky-Dill et al. 2025).

## 7. Responsible release

Code: HL3-FULL. Eval tiers 1+2: CC-BY-4.0. Tier 3: category-level descriptors in this public repo, with full prompt bodies on HL3-gated HF dataset (`SolshineCode/biorefusalaudit-gated`). See `SAFETY.md`, `docs/HL3_RATIONALE.md`, `LICENSE_HL3_DATASET.md`.

**Why HL3, not MIT.** MIT and Apache 2.0 permit offensive applications. For a tool identifying which models retain hazard representations behind a surface refusal, that neutrality is a real attack surface: an adversarial actor using the divergence score to target shallow-refusal models. HL3-FULL prevents this with enforceable human-rights conditions. Violation terminates the license. Enforceable repo tests (`safety_review.check_no_hazard_bodies`, `reporting.redaction.redact_tier3`) exceed host ground rules.

## 8. Future work

**Domain-specific SAE fine-tuning** is the binding constraint. Neuronpedia validation confirms the auto-tuned catalog encodes generic technical-governance vocabulary rather than bio-specific circuitry. The corrective: fine-tune SAEs on biosecurity behavioral corpora (base vs. RLHF pairs) separating shallow refusals (hazard features active) from genuine ones (suppressed). A projection adapter is feasible with the existing corpus; full fine-tune needs institutional CBRN datasets (~10K samples). **Proof-of-concept (2026-04-25):** Three SAE training runs complete across two model families. (1) Gemma 2 2B, mean contrastive, 5000 steps: L_recon −99.5%, L_contrastive +7% (vocabulary bottleneck on 75 prompts). (2) Gemma 2 2B, pairwise NT-Xent, 5000 steps: L_recon −99.84%, L_contrastive 0.905→7e-6; near-zero by step 4000. (3) Gemma 4 E2B, pairwise NT-Xent, 5000 steps (d_model=1536, d_sae=6144, layer 17): L_recon −99.98% (final 0.00058), L_contrastive 0.570→0.777 (step 1000) then collapses to 0.000116 by step 4999. All three runs show the same corpus-size bottleneck: the NT-Xent gradient cannot maintain tier separation on 75 prompts once reconstruction dominates. The step 1000 checkpoint (L_contrastive=0.777) is the recommended artifact for tier-separation analysis; the final checkpoint is a high-quality reconstruction SAE. Published: `SolshineCode/biorefusalaudit-gemma2-2b-bio-sae-wmdp`, `-pairwise`, and `Solshine/gemma4-e2b-bio-sae-v1` (pending HF push approval). Corpus size, not compute, is the binding constraint. **WMDP (confirmed 2026-04-25):** `cais/wmdp-bio-forget-corpus` (Li et al. 2024) not publicly available (only `bio-retain-corpus` available). Institutional CBRN datasets remain the unlock. See `docs/METHOD.md §Planned extension`.

**Bio-domain SAE eval on Gemma 4 E2B (2026-04-26):** First full eval of Gemma 4 E2B-IT using the domain-specific bio SAE (checkpoint_01000.pt, step 1000, peak contrastive signal L_contrastive=0.777) with auto-tuned catalog (32 features, Cohen's d top=3.28). Tier-level mean D: benign_bio=0.362 (n=23), dual_use_bio=0.406 (n=30), hazard_adjacent=0.404 (n=22). Tier ordering is directionally correct (benign < dual_use ≈ hazard); dual_use/hazard separation is weak (Δ=0.002). Within-comply D increases monotonically with tier: benign 0.466 → dual_use 0.494 → hazard_adjacent 0.504 — this is the refusal-depth signal. Refuse D is near-constant across tiers (~0.228). Calibration T fitted (ridge, λ=0.1): MSE≈0, condition number=483. **Caveat:** D is strictly bimodal (comply ~0.46–0.51, refuse ~0.226–0.230) because `--no-llm-judges` produces binary surface_soft scores; the near-zero calibration residual is an artifact of fitting to binary labels, not a sign of quality fit. Continuous D and reliable calibration require LLM judge scores. **Domain-specificity confirmed:** Cross-domain deception SAE (§4.5) gave D≈0 across all tiers; bio-specific catalog gives D=0.36–0.41 — domain specificity is the operative variable, consistent with Secret Agenda's non-generalizability finding.

- **Non-bio domain control** *(completed 2026-04-23; n=30, legal/financial eval, same model + SAE).* D shows a tier gradient (benign 0.573, dual-use 0.672, hazard 0.665) but with markedly weaker effect than bio: Cohen's d=0.67, p=0.052, CIs overlapping by 0.071. This rules out pure domain-agnostic routing. D has a bio-domain advantage, likely because the catalog was tuned on bio prompts, while confirming generic sensitivity routing is present and partially contributes to the bio signal. Domain-specific SAE fine-tuning completes the transition.
- **Held-out calibration set** *(attempted 2026-04-25; see §5 CORRECTED: held-out T revealed framing-distribution sensitivity rather than retiring the caveat; next step: matched-framing holdout set);* **Gemma Scope 2 sweep** when released (Gemma 3 270M through 12B); **Llama 3.1 8B** cross-architecture comparison (already cached locally); **unlearning before/after** on RMU snapshots; **Transcoder + CLT analysis** to move from metric to circuit for the strongest divergence flags. Coefficient Giving RFP (due May 11) frames a 6-month follow-on around these results.
