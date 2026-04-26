---
license: hl3-full
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - biosafety
  - biorefusalaudit
  - gemma
  - sae
base_model: google/gemma-2-2b-it
datasets:
  - cais/wmdp-corpora
language:
  - en
---

# BioRefusalAudit: Gemma 2 2B Contrastive Bio-Safety SAE (Pairwise, 10k-step run)

A TopK Sparse Autoencoder (SAE) fine-tuned on biology-domain activations from `google/gemma-2-2b-it` with an NT-Xent pairwise contrastive objective.

Trained as part of the [BioRefusalAudit](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon) project (AIxBio Hackathon 2026, Track 3: Biosecurity Tools).

## Architecture

| Parameter | Value |
|-----------|-------|
| Type | TopK Sparse Autoencoder |
| d_model | 2304 (Gemma 2 2B hidden size) |
| d_sae | 6144 (~2.7x expansion) |
| k (sparsity) | 32 active features |
| Hook layer | Layer 12 (residual stream post-MLP) |
| Base model | google/gemma-2-2b-it |

## Training

- **Dataset**: `cais/wmdp-corpora` bio corpora + BioRefusalAudit 75-prompt eval set (pairwise tiers: hazard/benign/dual-use)
- **Objective**: Reconstruction (MSE) + Sparsity (L1) + NT-Xent pairwise contrastive cosine separation
- **Steps**: 5,000 (run label "10ksteps" refers to planned extended run; 5k completed)
- **Optimizer**: AdamW, lr=3e-4
- **Intermediate checkpoints**: steps 1000, 2000, 3000, 4000 (included)

### Final Training Metrics (step 4999)

| Metric | Initial (step 0) | Final (step 4999) |
|--------|-----------------|-------------------|
| total_loss | 3.16 | 0.019 |
| l_recon | 2.65 | **0.0043** |
| l_sparsity | 1.50 | 0.362 |
| l_contrastive | 0.905 | 7.4e-6 |
| L0 (mean active features) | 32.0 | 32.0 |

**Note on collapsed contrastive signal**: `l_contrastive` drops to ~0 by the final checkpoint, indicating the contrastive term effectively vanished during training. This may reflect that with pairwise NT-Xent and a small corpus, the contrastive signal collapses once reconstruction dominates. The WMDP-trained SAE (`biorefusalaudit-gemma2-2b-bio-sae-wmdp`) maintains `l_contrastive=0.060` at step 4999 and is recommended over this checkpoint for refusal-depth analysis.

The reconstruction quality is excellent (`l_recon=0.004` vs 2.65 at init), making this checkpoint useful as a base reconstruction SAE if contrastive separation is added via probe-based analysis rather than SAE fine-tuning.

## Intermediate Checkpoints

The repo includes checkpoints at steps 1000, 2000, 3000, 4000 — useful for studying the trajectory of contrastive signal collapse.

## Usage

Same as `biorefusalaudit-gemma2-2b-bio-sae-wmdp` — see that model card for the `TopKSAE` class definition.

## Citation

```bibtex
@misc{deleeuw2026biorefusalaudit,
  title={BioRefusalAudit: Measuring Refusal Depth in LLMs via SAE Feature Divergence},
  author={de Leeuw, Caleb},
  year={2026},
  howpublished={AIxBio Hackathon 2026, Track 3: Biosecurity Tools},
  note={https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon}
}
```

## License

Code and weights released under the [Hippocratic License 3.0 (HL3-Full)](https://firstdonoharm.dev/).
