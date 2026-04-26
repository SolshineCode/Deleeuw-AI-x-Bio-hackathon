# Measuring Refusal Depth: A Governance Tool for LLM Bio-Safety Evaluation

**Caleb DeLeeuw**, one-page policy brief for CLTR / AISI / UK AI Safety Institute audiences. Companion to the AIxBio Hackathon 2026 technical submission.

## The gap

Current deployment decisions for biology-adjacent models rely on **capability evaluations** (VCT, WMDP-Bio, ABC-Bench). These answer one question: *will this model produce hazardous output when asked?*

They do not answer the question that matters for a governance decision: *if this model refuses, is its refusal robust?*

A model that refuses a direct request but activates hazard-relevant internal representations is one prompt-engineering step away from compliance. Capability evaluations report it as "safe." Deployment decisions on that basis underweight the most easily reversed form of alignment.

## The measurement

BioRefusalAudit operationalizes **refusal depth** as a calibrated surface-internal divergence metric over ensembled SAE feature categories. The tool ships as:

- A 75-prompt stratified eval set (benign / dual-use / hazard-adjacent × four framings)
- A multi-judge consortium for surface classification
- A divergence score per (model, prompt) grounded in auto-tuned SAE feature categories (Cohen's d selected; semantic validation via Neuronpedia is planned)
- Cross-model and cross-framing aggregation

A model with **low divergence on refusal** has a refusal backed by internal non-activation. A model with **high divergence on refusal** has a refusal papered over an intact internal capability. The second is a deployment risk the first is not. BioRefusalAudit also detects a third posture invisible to binary classifiers: **hedge-without-refuse** — where the model softens its response without committing to refusal, and internal hazard representations remain active. On Gemma 2 2B-IT, this is the actual safety posture for hazard-adjacent prompts: 100% hedge, 0% comply, 0% genuine refuse. The model never refuses — but it also never fully complies. A binary "refusal rate" metric collapses this into a 0% pass rate and reports it as equally unsafe as a model that complies outright.

**Causal validation:** To confirm the metric traces genuine internal circuits rather than surface correlates, we run perturbation experiments (ablate and boost the top-k SAE features identified by the divergence score; re-measure surface label and D). In **60/75 tested prompt-feature pairs** (80%) across all three tiers on Gemma 2 2B-IT, ablation or boost shifted either the surface label or |ΔD| > 0.2 — confirming candidate mechanistic features (CMF). The NC rate runs *inverted* to hazard tier: benign 87% > dual-use 80% > hazard-adjacent 73%. This is consistent with an auto-tuned catalog that encodes general biology vocabulary rather than bio-specific hazard circuitry, and is the structural motivation for domain-specific SAE fine-tuning.

## Why this is a policy artifact, not only a research one

1. **Auditable metric.** Divergence is a number. Governance frameworks (RSPs, AI Act Annex IV, NIST AI RMF GOVERN-5) need numbers they can require a provider to report. "Refusal depth" fits.

2. **Comparable across providers.** Because the metric is model-agnostic and the eval set is public-under-CC-BY, two providers can be compared on the same instrument. This is the missing piece the current benchmark ecosystem cannot provide — VCT tells you which model is more capable of hazardous content; BioRefusalAudit tells you which model's refusal is better *earned*.

3. **Pre-deployment signal, not post-incident.** Unlike jailbreak red-teaming (which depends on finding a working prompt), divergence can be measured on refusing models without ever producing hazardous output. This preserves responsible-disclosure norms while still measuring the safety floor.

4. **Unlearning verification.** The metric is the right instrument for before/after evaluation of RMU or Representation Engineering unlearning interventions. A deployment claim of the form "we unlearned X" is only falsifiable via an interpretability measurement on whether the relevant internal capability is actually attenuated — not via a capability eval which measures surface behavior only.

## Recommendations

- **Include refusal-depth reporting** in RSPs and analogous frameworks as a standard companion to capability evals for models with bio-adjacent deployment.
- **Fund domain-specific SAE fine-tuning** as the next critical public-good infrastructure investment. Current auto-tuned feature catalogs (selected by Cohen's d on general SAE activations) identify features that statistically co-vary with bio-hazard tier but carry generic technical-governance semantics rather than bio-specific meaning — confirmed by Neuronpedia interpretability audits of the Gemma Scope 1 catalog. The fix is to fine-tune SAEs on behavioral activation corpora: residual-stream activations collected specifically during bio-hazard prompt completions, with a contrastive objective that separates shallow refusals (hazard features active) from genuine refusals (hazard features suppressed). A proof-of-concept local training run (GTX 1650 Ti, 75-prompt corpus, 500 steps) confirmed the training infrastructure is consumer-hardware-feasible but revealed the bottleneck: with only a 75-prompt eval set, the contrastive loss does not converge — hazard/benign feature means remain indistinguishable. This directly quantifies the minimum corpus requirement: the 75-prompt eval set is insufficient; institutional CBRN red-team data is the prerequisite. Institutional holders of CBRN red-team datasets (AISI, CLTR, national biosecurity labs) are the natural partners for training data.
- **Fund hand-validated feature catalogs** for major open-weight model families as public-good infrastructure, similar to how Gemma Scope was made available. Without validated feature catalogs per model, divergence metrics cannot be trusted cross-model. Domain-specific fine-tuning (above) produces catalogs as a byproduct and should be prioritized over hand-curation alone.
- **Adopt tiered dataset release practices** (HL3-gated for hazard-adjacent) as standard for biosecurity-adjacent benchmarks. The BDL framework (Bloomfield, Black, Crook et al., Science 2026) provides the structural precedent; BioRefusalAudit adopts it. The same framework applies to CBRN behavioral activation corpora used for SAE fine-tuning, as they carry implicit hazard information in their activation patterns and warrant equivalent access controls. The NTI managed access framework (Yassif & Carter, *A Framework for Managed Access to Biological AI Tools*, NTI Bio January 2026) provides the policy framing: refusal depth scoring is the measurement layer that makes tiered access decisions principled rather than arbitrary. BioRefusalAudit implements both layers: measurement (the divergence score) and enforcement (HL3 licensing).
- **Distinguish refusal depth from refusal rate** in governance reporting. A model with 100% surface refusal on hazard-adjacent prompts and high divergence on those refusals is *less safe* than a weaker model with 95% refusal and low divergence — the former has capability behind a thin policy layer; the latter has lower capability full stop. The Gemma 4 E2B finding (100% refusal rate, yet hazard_features_active_despite_refusal = 100% on hazard-adjacent tier) illustrates this risk concretely: global surface refusal makes standard capability evals report a safe model while the refusal circuit is demonstrably shallow.

## Contact

Caleb DeLeeuw · caleb.deleeuw@gmail.com · [github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)

Sponsor: Fourth Eon Bio (hackathon Track 3). Methodology lineage: nanochat-SAE-deception-research; Secret Agenda (arXiv:2509.20393, AAAI 2026 AI GOV).
