# SAE Layer Selection — Provenance for the Gemma 2 2B-IT Layer-12 Probe

This folder is the methodological-foundation side of the BioRefusalAudit hackathon submission. It documents **why we probe Gemma 2 2B-IT residual stream at layer 12**, what custom Layer-12 SAEs we have trained elsewhere, and the cross-model scaling work that validates mid-depth residual SAEs as a generalizable probe substrate for behavior measurement.

Short version: layer 12 at width_16k is not a guess. It sits at the midpoint of the ~50 % relative-depth band where behavioral signal is strongest across three architecture families (nanochat, Qwen, Llama) in prior unpublished SAE work by the author. The bio-safety application in the main repo is the first domain-migration of that substrate. Prior work on behavioral-signal SAE detection is cited throughout as methodological foundation, not as a claim of bio-safety results.

## Files

- [`layer_selection_provenance.md`](layer_selection_provenance.md) — the layer-selection story: why mid-depth, why Gemma Scope 1 layer 12 specifically, what alternatives were considered.
- [`methodology.md`](methodology.md) — SAE training methodology: TopK k=32 at 4× expansion, activation-collection-at-all-token-positions, per-batch decoder normalization, 200-epoch Adam 3e-4 with early stopping. Inherited from the author's behavioral-detection SAE pipeline.
- [`cross_model_scaling.md`](cross_model_scaling.md) — the 9-model cross-architecture sweep showing mid-depth is universal and the ~1.7 B boundary where SAE-decomposed features flip from amplifying-vs-diluting a concentrated behavioral signal.
- [`lessons_and_failures.md`](lessons_and_failures.md) — honest dead-ends (JumpReLU threshold-zero bug, SAE features underperforming raw residuals above scale, cross-model feature-space non-transfer, class-imbalance pitfalls). Each lesson informs a design decision in BioRefusalAudit.
- [`data/`](data/) — raw JSON artifacts copied from the author's behavioral-detection research: layer sweeps, cross-model summaries, SAE training configs + meta files.

## Relation to prior work

Much of the substrate reasoning here was developed by the author across two prior research efforts:

- **The Secret Agenda** (arXiv [2509.20393](https://arxiv.org/abs/2509.20393), AAAI 2026 AI GOV) — published negative result: single-feature SAE auto-labels fail to detect strategic deception. BioRefusalAudit's divergence metric is the ensemble-category response to that failure.
- **Behavioral-signal SAE detection** (unpublished, ongoing in `github.com/SolshineCode/deception-nanochat-sae-research`, private drafts) — the numbered findings (§F1–§F25) and the 9-model cross-architecture sweep that grounded mid-depth layer selection. Materials from that body of work are duplicated into this folder for hackathon-submission completeness and clearly cited as their origin. Where a result is deception-specific (for example, the ~1.7 B boundary for honest-vs-deceptive separability), the domain-general form is imported here (in that case: a scale-dependent transition in SAE-decomposability of concentrated behavioral signals) rather than the deception-specific claim.

If any reviewer needs the full traceable provenance back to its source repository, every claim in this folder is cited to its origin file + line range.

## What the hackathon submission uses from all this

1. The layer-selection rationale (mid-depth / ~50 % relative depth) → justifies `--layer 12` as the canonical probe point in `configs/models.yaml`.
2. The TopK training methodology (k=32, 4× expansion, activation-at-all-positions) → drives any custom SAE training we do for Gemma 2 2B-IT if the Gemma Scope 1 release turns out to be unsuitable (backup path).
3. The scale-dependent SAE-decomposability finding → informs why we choose a pre-trained Gemma Scope 1 JumpReLU SAE at Gemma 2's 2 B scale (below the 1.7 B boundary, which is where pre-trained JumpReLU SAEs retain the behavioral signal under sparse decomposition).
4. The class-imbalance + robustness-check battery → built into the divergence-metric unit tests and the report-generation pipeline.
5. The negative results (SAE features don't steer behavior cleanly; cross-model feature spaces don't transfer) → frames BioRefusalAudit as a *probing* tool, not a *steering* tool.
