# Measuring Refusal Depth: A Governance Tool for LLM Bio-Safety Evaluation

**Caleb DeLeeuw** — one-page policy brief for CLTR / AISI / UK AI Safety Institute audiences. Companion to the AIxBio Hackathon 2026 technical submission.

## The gap

Current deployment decisions for biology-adjacent models rely on **capability evaluations** (VCT, WMDP-Bio, ABC-Bench). These answer one question: *will this model produce hazardous output when asked?*

They do not answer the question that matters for a governance decision: *if this model refuses, is its refusal robust?*

A model that refuses a direct request but activates hazard-relevant internal representations is one prompt-engineering step away from compliance. Capability evaluations report it as "safe." Deployment decisions on that basis underweight the most easily reversed form of alignment.

## The measurement

BioRefusalAudit operationalizes **refusal depth** as a calibrated surface-internal divergence metric over ensembled SAE feature categories. The tool ships as:

- A 75-prompt stratified eval set (benign / dual-use / hazard-adjacent × four framings)
- A multi-judge consortium for surface classification
- A divergence score per (model, prompt) grounded in hand-validated SAE features
- Cross-model and cross-framing aggregation

A model with **low divergence on refusal** has a refusal backed by internal non-activation. A model with **high divergence on refusal** has a refusal papered over an intact internal capability. The second is a deployment risk the first is not.

**Causal validation:** To confirm the metric traces genuine internal circuits rather than surface correlates, we run perturbation experiments (ablate and boost the top-k SAE features identified by the divergence score; re-measure surface label and D). In 9/9 tested prompt-feature pairs on Gemma 2 2B-IT, ablation or boost shifted either the surface label or D by |ΔD| > 0.2 — confirming these features constitute identifiable mechanistic circuits, not post-hoc correlates. Three cases (bio_004 benign/direct, bio_010 benign/educational, bio_060 hazard-adjacent/roleplay) showed the counterintuitive direction: suppressing refusal-circuitry features caused the model to *refuse* content it was previously complying with, suggesting these features mediate contextual engagement (compliance-enabling) rather than acting as simple refusal toggles. One benign case (bio_001) showed comply→comply with |ΔD|=0.29, consistent with the same compliance-enabling role: the features actively reduce divergence even when the surface label remains compliant.

## Why this is a policy artifact, not only a research one

1. **Auditable metric.** Divergence is a number. Governance frameworks (RSPs, AI Act Annex IV, NIST AI RMF GOVERN-5) need numbers they can require a provider to report. "Refusal depth" fits.

2. **Comparable across providers.** Because the metric is model-agnostic and the eval set is public-under-CC-BY, two providers can be compared on the same instrument. This is the missing piece the current benchmark ecosystem cannot provide — VCT tells you which model is more capable of hazardous content; BioRefusalAudit tells you which model's refusal is better *earned*.

3. **Pre-deployment signal, not post-incident.** Unlike jailbreak red-teaming (which depends on finding a working prompt), divergence can be measured on refusing models without ever producing hazardous output. This preserves responsible-disclosure norms while still measuring the safety floor.

4. **Unlearning verification.** The metric is the right instrument for before/after evaluation of RMU or Representation Engineering unlearning interventions. A deployment claim of the form "we unlearned X" is only falsifiable via an interpretability measurement on whether the relevant internal capability is actually attenuated — not via a capability eval which measures surface behavior only.

## Recommendations

- **Include refusal-depth reporting** in RSPs and analogous frameworks as a standard companion to capability evals for models with bio-adjacent deployment.
- **Fund domain-specific SAE fine-tuning** as the next critical public-good infrastructure investment. Current auto-tuned feature catalogs (selected by Cohen's d on general SAE activations) identify features that statistically co-vary with bio-hazard tier but carry generic technical-governance semantics rather than bio-specific meaning — confirmed by Neuronpedia interpretability audits of the Gemma Scope 1 catalog. The fix is to fine-tune SAEs on behavioral activation corpora: residual-stream activations collected specifically during bio-hazard prompt completions, with a contrastive objective that separates shallow refusals (hazard features active) from genuine refusals (hazard features suppressed). This is the methodological extension that turns BioRefusalAudit from a domain-agnostic sensitivity metric into a bio-specific circuit auditor. Institutional holders of CBRN red-team datasets (AISI, CLTR, national biosecurity labs) are the natural partners for training data.
- **Fund hand-validated feature catalogs** for major open-weight model families as public-good infrastructure, similar to how Gemma Scope was made available. Without validated feature catalogs per model, divergence metrics cannot be trusted cross-model. Domain-specific fine-tuning (above) produces catalogs as a byproduct and should be prioritized over hand-curation alone.
- **Adopt tiered dataset release practices** (HL3-gated for hazard-adjacent) as standard for biosecurity-adjacent benchmarks. The BDL framework (Bloomfield, Black, Crook et al., Science 2026) provides the structural precedent; BioRefusalAudit adopts it. The same framework applies to CBRN behavioral activation corpora used for SAE fine-tuning — they carry implicit hazard information in their activation patterns and warrant equivalent access controls.
- **Distinguish refusal depth from refusal rate** in governance reporting. A model with 100% surface refusal on hazard-adjacent prompts and high divergence on those refusals is *less safe* than a weaker model with 95% refusal and low divergence — the former has capability behind a thin policy layer; the latter has lower capability full stop. The Gemma 4 E2B finding (100% refusal rate, yet hazard_features_active_despite_refusal = 100% on hazard-adjacent tier) illustrates this risk concretely: global surface refusal makes standard capability evals report a safe model while the refusal circuit is demonstrably shallow.

## Contact

Caleb DeLeeuw · caleb.deleeuw@gmail.com · [github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)

Sponsor: Fourth Eon Bio (hackathon Track 3). Methodology lineage: nanochat-SAE-deception-research; Secret Agenda (arXiv:2509.20393, AAAI 2026 AI GOV).
