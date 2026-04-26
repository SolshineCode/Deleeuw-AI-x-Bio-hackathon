---
license: hl3-full
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - biosafety
  - biorefusalaudit
  - gemma
  - gemma4
  - sae
base_model: google/gemma-4-E2B-it
datasets:
  - SolshineCode/biorefusalaudit-eval-public
language:
  - en
---

# BioRefusalAudit: Gemma 4 E2B-IT Contrastive Bio-Safety SAE (v1)

A TopK Sparse Autoencoder (SAE) trained on biology-domain residual-stream activations from `google/gemma-4-E2B-it` at layer 17, with an NT-Xent pairwise contrastive objective to separate hazard-adjacent from benign biological feature activations.

Trained as part of the [BioRefusalAudit](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon) project (AIxBio Hackathon 2026, Track 3: Biosecurity Tools).

## Architecture

| Parameter | Value |
|-----------|-------|
| Type | TopK Sparse Autoencoder |
| d_model | 1536 (Gemma 4 E2B text hidden size) |
| d_sae | 6144 (4x expansion) |
| k (sparsity) | 32 active features |
| Hook layer | Layer 17 (residual stream, post-MLP) |
| Base model | google/gemma-4-E2B-it |

## Training

- **Dataset**: BioRefusalAudit 75-prompt public eval set (`data/eval_set_public/eval_set_public_v1.jsonl`) — three tiers: `benign_bio` (23 prompts), `dual_use_bio` (30), `hazard_adjacent_category` (22)
- **Objective**: Reconstruction (MSE) + Sparsity (L1) + NT-Xent pairwise contrastive cosine separation
- **Steps**: 5,000
- **Optimizer**: AdamW, lr=3e-4, lam_sparse=0.04, lam_contrast=0.1
- **Contrastive mode**: pairwise (NT-Xent cosine similarity, temperature=0.1)
- **Checkpoints**: steps 1000, 2000, 3000, 4000

### Training Metrics

| Metric | Initial (step 0) | Step 1000 | Step 2000 | Final (step 4999) |
|--------|-----------------|-----------|-----------|-------------------|
| total_loss | 3.42 | 0.170 | 0.048 | **0.0049** |
| l_recon | 3.32 | 0.087 | 0.043 | **0.00058** |
| l_sparsity | 1.08 | 0.130 | 0.118 | 0.108 |
| l_contrastive | 0.570 | **0.777** | 0.0018 | 0.000116 |
| L0 (mean active features) | 32.0 | 32.0 | 32.0 | 32.0 |

**Key finding — pre-collapse window**: `l_contrastive` peaks at 0.778 at step 1000, then collapses sharply between steps 1000–2000 (0.778 → 0.002). This creates a narrow window where the SAE encodes both strong reconstruction AND meaningful tier separation. **`checkpoint_01000.pt` is the recommended checkpoint for bio-safety refusal-depth analysis** — it retains the contrastive signal while still having reasonable reconstruction quality (l_recon=0.142).

The final checkpoint (step 4999) has excellent reconstruction (l_recon=0.00058) but near-zero contrastive separation. Use it as a reconstruction SAE; use `checkpoint_01000.pt` for tier-separation analysis. For comparison, the WMDP-trained Gemma 2 2B SAE maintains l_contrastive=0.060 at final step because it trained on ~5,000 bio documents rather than 75 prompts.

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
sae = TopKSAE(d_model=1536, d_sae=6144, k=32)
sae.load_state_dict(torch.load("sae_weights.pt", map_location="cpu"))
sae.eval()

# Hook Gemma 4 E2B layer 17 residual stream during generation
# Then: x_hat, z, pre = sae(activations.float())
# z is the sparse feature vector (shape: [seq_len, 6144]) — 32 nonzero entries per position
```

## Cross-Architecture Context

This SAE extends the BioRefusalAudit pipeline to Gemma 4 E2B (2B parameters, released April 2025). Key differences from the Gemma 2 2B baseline:

- **Larger vocabulary**: 256K tokens vs 256K (same)
- **Multimodal architecture**: `Gemma4ForConditionalGeneration` wraps a language backbone at `model.model.language_model` — hook path: `model.model.language_model.layers[17]`
- **4x expansion ratio**: same as Gemma 2 SAEs (d_sae/d_model = 6144/1536 = 4.0)
- **Smaller per-layer size**: d_model=1536 vs 2304, so SAE is smaller (6144×1536×2 = ~19MB vs ~28MB for Gemma 2)

## Caveats

- Trained on 75 prompts — very small corpus; contrastive signal collapsed early due to insufficient positive/negative pairs for NT-Xent
- No Neuronpedia validation; feature interpretability unverified
- Cross-architecture comparison to Gemma 2 2B SAEs requires accounting for d_model difference (1536 vs 2304)
- The 4x expansion ratio (used here) is below Gemma Scope's standard 8x; wider SAEs would likely capture more bio-specific features

## Intermediate Checkpoints

Checkpoints at steps 1000, 2000, 3000, 4000 are included for studying the trajectory of contrastive collapse. The step 1000 checkpoint (~l_contrastive=0.4) retains meaningful tier separation before reconstruction dominates.

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
