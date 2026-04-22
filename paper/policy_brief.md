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

## Why this is a policy artifact, not only a research one

1. **Auditable metric.** Divergence is a number. Governance frameworks (RSPs, AI Act Annex IV, NIST AI RMF GOVERN-5) need numbers they can require a provider to report. "Refusal depth" fits.

2. **Comparable across providers.** Because the metric is model-agnostic and the eval set is public-under-CC-BY, two providers can be compared on the same instrument. This is the missing piece the current benchmark ecosystem cannot provide — VCT tells you which model is more capable of hazardous content; BioRefusalAudit tells you which model's refusal is better *earned*.

3. **Pre-deployment signal, not post-incident.** Unlike jailbreak red-teaming (which depends on finding a working prompt), divergence can be measured on refusing models without ever producing hazardous output. This preserves responsible-disclosure norms while still measuring the safety floor.

4. **Unlearning verification.** The metric is the right instrument for before/after evaluation of RMU or Representation Engineering unlearning interventions. A deployment claim of the form "we unlearned X" is only falsifiable via an interpretability measurement on whether the relevant internal capability is actually attenuated — not via a capability eval which measures surface behavior only.

## Recommendations

- **Include refusal-depth reporting** in RSPs and analogous frameworks as a standard companion to capability evals for models with bio-adjacent deployment.
- **Fund hand-validated feature catalogs** for major open-weight model families as public-good infrastructure, similar to how Gemma Scope was made available. Without validated feature catalogs per model, divergence metrics cannot be trusted cross-model.
- **Adopt tiered dataset release practices** (HL3-gated for hazard-adjacent) as standard for biosecurity-adjacent benchmarks. The BDL framework (Bloomfield, Black, Crook et al., Science 2026) provides the structural precedent; BioRefusalAudit adopts it.
- **Distinguish refusal depth from refusal rate** in governance reporting. A model with 100% surface refusal on hazard-adjacent prompts and high divergence on those refusals is *less safe* than a weaker model with 95% refusal and low divergence — the former has capability behind a thin policy layer; the latter has lower capability full stop.

## Contact

Caleb DeLeeuw · caleb.deleeuw@gmail.com · [github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)

Sponsor: Fourth Eon Bio (hackathon Track 3). Methodology lineage: nanochat-SAE-deception-research; Secret Agenda (arXiv:2509.20393, AAAI 2026 AI GOV).
