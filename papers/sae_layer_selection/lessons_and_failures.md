# Lessons and Failures — Inherited From Prior SAE Research

Honest dead-ends and their design-implication for BioRefusalAudit. Source file + line range cited for each lesson.

## 1. The JumpReLU threshold-zero bug

**Source:** `FINDINGS.md §F16` (prior-work repo).

In the author's pre-2026-04-11 runs, custom JumpReLU SAEs were trained with threshold initialized and held at zero by a bug in the training loop. The apparent empirical advantage of JumpReLU over Gated and TopK in early runs was therefore **confounded** — the "JumpReLU" SAE was functionally a ReLU SAE without the jump-discontinuity the architecture names. A custom autograd STE (straight-through estimator) was added to correct this, and the training pipeline now explicitly asserts `(threshold != 0).any()` before the first SAE backward pass.

**Implication for BioRefusalAudit:** we use Gemma Scope 1's JumpReLU SAEs, which DeepMind trained correctly and released with per-feature learned thresholds. We do not claim a JumpReLU-specific advantage in our results — that claim would require re-running the architecture comparison in the bio-safety domain. Our unit tests assert loader correctness on the pre-trained weights (`test_feature_validator.py`) but not the architecture-comparison claim.

## 2. SAE decomposition can destroy signal vs. raw residuals

**Source:** `LEARNINGS_2026_04_03.md` lines 38–47.

> SAE Decomposition Loses the Deception Signal. At both layers tested: Layer 12: Raw 86.8 % > Gated 83.4 % > JumpReLU 82.7 % > TopK 65.8 %. SAE features are not aligned with the behavioral dimensions that encode deception.

**Implication for BioRefusalAudit:** we positioned the divergence metric as a *probing + interpretability* tool, not a superior-accuracy detector. The paper claims:

> BioRefusalAudit measures whether a refusal is *deep* — whether the model's internal hazard-adjacent features are *still active* while the surface says "refuse". Raw-residual linear probes can measure that divergence with roughly the same sensitivity as SAE-projected divergence, but cannot *decompose* it into named feature categories (hazard-adjacent vs. refusal-circuitry vs. deception-correlate) and cannot target specific features for causal intervention.

That framing is robust even under the SAE-decomposition-loses-signal result because the value BioRefusalAudit adds is *interpretability + causal intervention capability*, not raw sensitivity.

## 3. SAE feature spaces don't transfer cross-model

**Source:** `FINDINGS.md §F8` lines 198–214.

> SAE latent space geometry is model-specific. Encoding model B's activations through model A's SAE decoder gives near-chance discrimination, even for same-family pairs at shared d_in=2048. All 6 transfer pairs tested at 0.466–0.500 balanced accuracy vs. 0.63–0.80 in-distribution.

**Implication for BioRefusalAudit:** we do *not* reuse a feature catalog across model families. Each model in `configs/models.yaml` gets its own `data/feature_catalog/<model>.json`. The hackathon MVP ships a catalog for Gemma 2 2B-IT and a separate one for Gemma 4 E2B-IT (custom Solshine SAE). If a reviewer tries to reuse one catalog across models, the unit test in `test_feature_validator.py` will flag catalog-SAE mismatch but will not catch semantic transferability failures — those are flagged in the catalog's `catalog_note` field.

## 4. Class imbalance silently destroys results

**Source:** `LEARNINGS_2026_04_03.md` lines 30–36.

> The first Llama run reported 90.2 % accuracy. Impressive! Except the class split was 146:17. A probe that just predicts 'deceptive' every time gets ~86 %. The actual balanced accuracy was 68.3 %.

**Implication for BioRefusalAudit:** the eval set is stratified by (tier × framing) with per-cell counts visible in `data/eval_set_public/stratification_stats.md`. The divergence metric is computed and reported per-tier, never pooled across tiers. Permutation tests + bootstrap confidence intervals are on the planned-additions list for paper §4 (not in v0.1 report output, but the data is retained in `activations.npz` for post-hoc computation).

## 5. Small-n SAE training leaves most features dead

**Source:** Prior-work custom L12 SAEs trained on Qwen2.5-0.5B and Llama-3.2-1B — alive-feature counts were 571/3584 (16 %) and 537/8192 (6.6 %). The training set was ~3200 activation vectors.

**Implication for BioRefusalAudit:** we do *not* train custom SAEs for the hackathon MVP. We use Gemma Scope 1's pre-trained JumpReLU SAEs, which DeepMind trained on orders of magnitude more activation data (explicitly so) and which have non-dead feature fractions well above the author's custom SAEs. Custom SAE training is documented in `methodology.md` as a follow-on path for model families without a public SAE release; any such custom training should use at least 100 K activation vectors to avoid the dead-feature problem.

## 6. Judge consortium failure modes on real completions

**Specialist-review addition (2026-04-23):** The judge consortium path was mainly exercised by unit tests during the v0.1 development. The regex first-pass misclassifies certain biology-teacher prose formats (e.g., "Here's a breakdown: **Light-dependent reactions**..." labeled as refuse in an earlier smoke test). Remediation:

1. Extended the regex comply patterns (commit `0291545`) to recognize "breakdown / summary / outline / rundown" and markdown-bold section headers.
2. The consortium's `disagreement` score is recorded per prompt in `report.json`; high-disagreement prompts are flagged by `trace_selected_cases.py` as candidates for attribution + intervention review.
3. The `--use-llm-judges` flag turns on Gemini + `claude -p` + Ollama for cases where the regex first-pass is ambiguous.

**Implication:** any released accuracy number should cite the specific judge mix used (regex-only vs. full consortium).

## 7. Feature-catalog stubs produce misleadingly neat divergence

**Current project, 2026-04-22:** The v0.1 feature catalog in `data/feature_catalog/gemma-2-2b-it.json` was populated with plausible-but-unvalidated feature indices as a scaffold. On the first real eval, 2 of 3 benign-bio prompts projected to a zero feature vector (the stub indices didn't match any actually-firing features). This artificially inflates divergence scores (zero vector → cosine 0 → D = 1.0).

**Remediation (same commit series):**

1. Added `scripts/auto_tune_catalog.py` (this PR) — Cohen's-d auto-tuning from real activation data.
2. Added `--dump-activations` to the CLI so eval passes persist per-prompt d_sae=16 384 feature vectors for downstream catalog tuning.
3. Added `feature_validator.differentiation_check` — filters features below a minimum effect-size threshold before they enter the catalog.

**Implication:** the hackathon submission's final paper numbers will be reported against a *tuned* catalog, and the v0.1 stub numbers will be kept in an appendix labeled "pre-tuning" so the improvement from tuning is transparent.

## Summary — honest claims we can make

- The mid-depth layer-selection methodology transfers cleanly to Gemma 2 2B-IT.
- The SAE-decomposition-dilutes-at-scale finding means we claim *interpretability + causal intervention* as our value-add, not *raw accuracy*.
- Feature catalogs are model-specific, not reusable.
- Stratified per-tier divergence reporting + per-tier confidence intervals are required; pooled reporting hides class-imbalance artifacts.
- Small-n custom SAEs have high dead-feature fractions; use pre-trained releases for MVP.
- Catalog stubs must be validated before they can ground any mechanistic claim.

And the claims we explicitly do *not* make:

- "BioRefusalAudit is more accurate than a raw-residual linear probe."
- "Catalog stubs at v0.1 constitute a validated feature set."
- "Feature indices transfer across model families."
- "Deception-detection accuracies from the author's prior work translate to bio-safety accuracies."
