# SAE Training Methodology — Inherited From Prior Behavioral-Detection Work

BioRefusalAudit's primary SAE substrate is the pre-trained `google/gemma-scope-2b-pt-res` release from Google DeepMind — not a custom SAE we trained. However, the methodology we'd use if we had to train a custom SAE for a new model family is inherited verbatim from the author's prior behavioral-detection SAE pipeline. Two custom Layer-12 SAEs were trained in that work on Qwen2.5-0.5B and Llama-3.2-1B and their configurations + trained metrics are duplicated into `data/` here for transparency.

## Training pipeline

Source: `/c/Users/caleb/deception-nanochat-sae-research/experiments/v4_sandbagging_concealment/new_models_2026_04_18/train_sae_v4.py` lines 1–237.

**Activation collection** — the script collects residual-stream activations from the target transformer block at **every token position** across three prompt templates:

1. A neutral / benign template
2. A contextual / loaded template
3. A stressful / threat template

This produces on the order of 3 000 activation vectors per model × template combination. All-position collection (as opposed to last-token) matters: it is what the Gemma Scope 1 authors also do, and it is what gives the trained SAE coverage of both prompt-encoding and generation-time activations.

**Architecture — TopK.** Not Gated, not vanilla ReLU, not JumpReLU during the training of these specific custom SAEs. TopK with `k=32` is chosen for two reasons:

1. Deterministic sparsity. TopK always keeps exactly the top 32 pre-activation values, so every example has the same active-feature count. This makes downstream feature analysis (differentiation check, per-category projection) directly comparable across prompts without having to normalize by variable active-count.
2. Compatibility with the downstream `feature_validator.differentiation_check` routine shipped in `biorefusalaudit/features/feature_validator.py`, which computes a Cohen's-d-like effect size per feature. The effect-size computation is cleanest when active-count is constant.

Gemma Scope 1 itself uses **JumpReLU**. BioRefusalAudit accommodates both: `biorefusalaudit/models/sae_adapter.py` has `TopKSAE` and `JumpReLUSAE` classes and auto-dispatches on the architecture tag in the catalog config.

**Expansion factor: 4×.** Feature dimensionality is `d_sae = 4 × d_model`. For Qwen2.5-0.5B (d_model=896) this gives d_sae=3584; for Llama-3.2-1B (d_model=2048) this gives d_sae=8192. Gemma Scope 1 width_16k is a 7× expansion over Gemma 2 2B (d_model=2304), meaning more features per activation direction and therefore more monosemanticity at the cost of more dead features.

**Optimizer + schedule.** Adam at learning rate 3e-4, batch size 128, up to 200 epochs with early stopping (patience 15 epochs on reconstruction loss). Decoder columns are L2-normalized per-batch (`F.normalize(sae.W_dec.data, dim=-1)`) — this is a standard SAE training stabilization.

**Seeding.** The author's current training script does not explicitly seed. For the hackathon-visible custom SAEs we would add a seed parameter to enable bit-exact reproducibility. That is an item on the `lessons_and_failures.md` punch-list.

## Results table — prior-work custom Layer-12 SAEs

Copied verbatim from the meta-JSON artifacts in `data/`:

| Model | d_in | d_sae | k | MSE | Explained Variance | L0 | Alive Features | Training Time |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 896 | 3 584 | 32 | 0.0148 | **0.9999** | 32.0 | 571 | 53.5 s |
| Llama-3.2-1B | 2 048 | 8 192 | 32 | 0.0002 | **0.9999** | 32.0 | 537 | 157.2 s |

Source configs: `data/qwen25_0p5b_topk_k32_L12_v4mixed_cfg.json` and `data/llama_topk_k32_L12_v4mixed_cfg.json`. Source training-meta: `data/qwen25_0p5b_topk_k32_L12_v4mixed_meta.json` and `data/llama_topk_k32_L12_v4mixed_meta.json`.

Reading the numbers:

- **Explained variance > 0.9999** on both. The SAE reconstructs the residual stream with >99.99 % fidelity at L0=32. This is at the high end of published SAE explained-variance metrics and is consistent with the author's prior findings about mid-depth layers being cleanly decomposable at TopK=32 for models ≤ 1.3 B params.
- **L0 = 32.0 exactly** on both — deterministic because TopK enforces exactly k active features per input.
- **Alive features** — 571/3 584 (16 %) and 537/8 192 (6.6 %). Most features are dead at this small training set size (~3 200 vectors). The author's notes flag this as a known outcome of the small training set and an area for follow-up (retrain with orders of magnitude more activation samples).
- **Training time** — 53 s and 157 s on the author's GTX 1650 Ti Max-Q, roughly linear in d_sae.

## How this methodology applies to Gemma 2 2B-IT in BioRefusalAudit

The Gemma Scope 1 SAE we use for BioRefusalAudit's primary experiments is DeepMind-trained at much larger scale (billions of activation vectors) than the custom SAEs documented here. We cite the author's custom-SAE methodology not because we use those exact weights, but to show:

1. The same SAE-training pipeline has been validated before on matched-depth / matched-scale targets,
2. Should we need a custom SAE for a model family without a public release (Gemma 3, Gemma 4 E2B, or post-submission A100-scale targets), the pipeline is already proven,
3. The L0=32 operating point chosen for BioRefusalAudit's default (Gemma Scope 1 average_l0_82 is the closest public release to an ideal ~32 with some budget) is consistent with prior-work decisions on sparsity-vs-monosemanticity trade-offs.

## Reproduction instructions

To train an additional custom SAE for BioRefusalAudit on a new model:

```bash
# Copy the canonical training script from the prior-work repo into biorefusalaudit/scripts/train_sae.py
cp /c/Users/caleb/deception-nanochat-sae-research/experiments/v4_sandbagging_concealment/new_models_2026_04_18/train_sae_v4.py \
   scripts/train_sae.py

# Adapt the three prompt templates to bio-safety ones (benign_bio / dual_use_bio / hazard_adjacent_category category-descriptors)
# Adapt the hook layer to the mid-depth choice (e.g., 46% of model depth)
# Run:
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/train_sae.py \
    --model google/gemma-2-2b-it \
    --layer 12 \
    --expansion 4 \
    --k 32 \
    --epochs 200 \
    --out data/custom_sae/gemma-2-2b-it-L12-topk-k32.pt
```

The resulting SAE can be loaded into BioRefusalAudit's pipeline via `load_sae(source="custom", repo_or_path=<path>, architecture="topk", k=32, d_model=2304, d_sae=9216)`.

## What is NOT imported

The prior-work custom SAEs were trained on prompt templates drawn from behavioral-testing research (trivia-question templates with neutral / V0 / V3 framings). Those templates are appropriate for behavioral probing of dishonest-vs-honest answering but not for bio-safety research. Any re-training of a custom SAE for BioRefusalAudit would use the hackathon eval set's prompt mix directly.
