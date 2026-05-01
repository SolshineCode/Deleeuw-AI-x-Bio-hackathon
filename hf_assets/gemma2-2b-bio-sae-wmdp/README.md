---
license: other
license_name: hl3-full
license_link: https://firstdonoharm.dev/version/3/0/full.html
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

# BioRefusalAudit: Gemma 2 2B Contrastive Bio-Safety SAE (WMDP-trained)

A TopK Sparse Autoencoder (SAE) fine-tuned on biology-domain activations from `google/gemma-2-2b-it` with a contrastive objective that separates hazard-adjacent from benign biological feature activations.

Trained as part of the [BioRefusalAudit](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon) project (AIxBio Hackathon 2026, Track 3: Biosecurity Tools), which measures *refusal depth* — the divergence between an LLM's surface refusal behavior and what its SAE feature activations reveal about internal hazard-concept processing.

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

- **Dataset**: [`cais/wmdp-corpora`](https://huggingface.co/datasets/cais/wmdp-corpora) — `bio_forget_corpus` (hazard-adjacent, ~222 samples used) + `bio_retain_corpus` (benign biology)
- **Objective**: Reconstruction (MSE) + Sparsity (L1) + Contrastive cosine tier separation
- **Steps**: 5,000
- **Optimizer**: AdamW, lr=3e-4
- **Contrastive mode**: Pairwise (NT-Xent-style cosine similarity between hazard and benign token-level feature profiles)

### Final Training Metrics (step 4999)

| Metric | Initial (step 0) | Final (step 4999) |
|--------|-----------------|-------------------|
| total_loss | 7.42 | 0.090 |
| l_recon | 7.29 | 0.066 |
| l_sparsity | 1.97 | 0.430 |
| l_contrastive | 0.567 | **0.060** |
| L0 (mean active features) | 32.0 | 32.0 |

The sustained `l_contrastive=0.060` at step 4999 indicates active tier separation is maintained throughout training (hazard vs. benign feature profiles remain distinguishable).

## Usage

```python
import torch
import torch.nn as nn

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        pre_relu = torch.relu(pre)
        topk_vals, topk_idx = torch.topk(pre_relu, self.k, dim=-1)
        out = torch.zeros_like(pre_relu)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z, x @ self.W_enc + self.b_enc

# Load
sae = TopKSAE(d_model=2304, d_sae=6144, k=32)
sae.load_state_dict(torch.load("sae_weights.pt", map_location="cpu"))
sae.eval()

# Hook Gemma 2 2B layer 12 and collect activations
# Then: x_hat, z, pre = sae(activations.float())
# z is the sparse feature vector — use for refusal depth analysis
```

## Methodology

This SAE is part of the BioRefusalAudit divergence metric pipeline:

```
D(s, f, T) = 1 - cos_sim(projected_hazard_features, projected_benign_features)
```

where `f` is extracted from this SAE's encoder output `z` at the hook layer. High D scores with a surface "refuse" label indicate *shallow refusals* — the model says no but its internal feature activations still encode hazard-adjacent concepts.

See the [paper writeup](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/paper/writeup.md) for the full formalization.

## Caveats

- Trained on ~222 WMDP samples per tier — small corpus; feature catalog tuning with the full ~3.9K WMDP hazard documents is ongoing
- The ~2.7x expansion ratio is below the typical 4-8x used in Gemma Scope; wider SAEs may capture more bio-specific features
- Neuronpedia validation not yet run on this checkpoint; feature interpretability unverified

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

Code and weights released under the [Hippocratic License 3.0 (HL3-Full)](https://firstdonoharm.dev/) — use restricted to applications that do not cause harm.
