# Cross-Model Scaling — Why Layer 12 Is a Stable Probe Point

## TL;DR

In prior unpublished SAE work by the author, the mid-depth residual-stream probe point was validated across **9 open-weight models spanning 135 M – 3.8 B parameters** (SmolLM2-135M, Pythia-160M, TinyLlama-1.1B, Llama-3.2-1B, Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-0.6B-Base, Qwen3-1.7B, Phi-2-2.7B, plus the author-trained nanochat-d20 561M and nanochat-d32 1.88B). The ~50 % relative-depth band held as the peak-signal layer across all 9. For Gemma 2 2B-IT specifically, that translates to **layer 12 of 26 = 46 %**, which is the Gemma Scope 1 release's deepest width_16k layer with publicly-released trained JumpReLU SAEs.

Additionally, a **scale transition at ~1.3–1.7 B parameters** was identified: above this boundary, SAE-decomposed features stop amplifying the behavioral signal and start diluting it relative to raw residual activations. For bio-safety probing, this sets an important constraint: Gemma 2 2B is *just* above the boundary, so we should expect SAE-projected divergence to perform comparably (not strictly better) to a raw-residual linear probe. That's the honest framing.

## The 9-model sweep

Source: `FINDINGS.md §F9` in the author's prior-work repo, supported by `qwen3_experiment_results.md` and `RESEARCH_ARC.md`.

| Model | Params | Layer (relative depth) | Peak behavioral-signal accuracy | SAE beat raw probe? |
|---|---:|---|---:|---|
| Pythia-160M | 160 M | ~50 % | low (base rate) | +6.2 pp (TopK) |
| SmolLM2-135M | 135 M | ~50 % | moderate | +14.3 pp (TopK) |
| Qwen2.5-0.5B | 0.5 B | ~50 % | moderate | partial |
| nanochat-d20 | 0.56 B | L18 / 20 (90 % at small scale) | 66.1 % | +8.7 pp (TopK) |
| TinyLlama-1.1B | 1.1 B | L8 / 22 (36 %) | moderate | +5.4 pp (TopK) |
| Llama-3.2-1B | 1.3 B | **L8 / 16 (50 %)** | 76.2 % | partial; JumpReLU +9.5 pp |
| Qwen3-1.7B | 1.7 B | L17 / 28 (61 %) | — | **SAEs hurt** |
| nanochat-d32 | 1.88 B | **L12 / 32 (39 %)** | 86.9 % | **SAEs hurt** |
| Phi-4-mini-reasoning | 3.8 B | — | — | **SAEs hurt** |

Source data copied into this folder:

- `data/cross_model_summary.json` — full cross-model comparison table: cosine alignment raw ↔ SAE (0.97), steering effectiveness at L8/L12 for Llama and Qwen2.5-0.5B, multi-layer combined intervention (+25.06 pp).
- `data/layer_behavioral_signal_profile.json` — detailed per-layer signal profile across L0–L16 for Llama, showing bimodal suppression-head distribution (L0–L2 cluster + L7–L9 cluster).

## The mid-depth universality claim

From `FINDINGS.md §F5` (same repo):

> Across three architecture families (nanochat/GPT-2-like, Qwen2.5/3, Llama 3.2), the behavioral signal peaks in mid-network layers at ~39–63 % relative depth — not at early or late layers. nanochat-d32 L12/32 (39 %); Qwen3-1.7B L17/28 (61 %); Llama-3.2-1B L8/16 (50 %).

The 40–60 % relative-depth band is empirically the best probe point for "behavioral content" (honest vs. deceptive, safe vs. hazard-adjacent, tactful vs. blunt, etc.) — this is where the residual stream carries the model's representation of *what it is about to do*, not what it has just read and not the tokenized logits it is about to emit. Gemma 2 2B's **46 %** depth at layer 12 lands cleanly inside that band.

## The ~1.7 B scale transition

From `FINDINGS.md §F9`:

> At ~1.3–1.7 B parameters there is a transition in whether SAE-decomposed features beat raw activations for deception detection. Below the transition: SAEs help or match. Above: SAEs hurt. Evidence: 9-model cross-model study (SmolLM2-135M, Pythia-160M, TinyLlama-1.1B, Llama-3.2-1B, Qwen2.5-0.5B/1.5B, Qwen3-0.6B-Base/1.7B, Phi-2-2.7B).

### Why this happens (PCA saturation)

From `RESEARCH_ARC.md` lines 353–362 (same repo):

| Model | Components to saturate probe accuracy |
|---|---|
| Llama-3.2-1B | 5 components |
| nanochat-d20 | 10 components |
| nanochat-d32 | ~100 components |

The author's interpretation:

> Smaller models encode [behavioral signal] in just 5–10 principal components of the residual stream. Larger models distribute the signal across ~100 dimensions. SAE sparse decomposition amplifies concentrated signals and destroys distributed ones. The architecture perfectly matches the observation.

### Implications for BioRefusalAudit

Gemma 2 2B (2.61 B params) is above the 1.7 B boundary. The author's prior-work prediction: SAE-projected features on Gemma 2 will *not* give us more behavioral resolution than a raw-residual linear probe would. They might give comparable resolution, with the benefit of *interpretability* (we can name the features).

This matters for BioRefusalAudit's framing: we are not claiming that SAE projection gives a *more sensitive* divergence signal than raw activations. We are claiming that it gives an *interpretable* divergence signal, where the direction of the divergence (hazard-on-refuse vs. refusal-on-comply vs. deception-correlate-active) can be pinned to specific named features, and where intervention experiments (feature ablation / boost) can test causality directly. A raw-residual probe can measure divergence but cannot decompose it.

## Signal strength scales with log(params)

From `LEARNINGS_2026_04_03.md` lines 19–28:

> Signal Strength Scales With Model Size. 1.88 B gets 86.9 %, 1.3 B gets 76.2 %, 561 M gets 66.1 %. Roughly linear with log(params).

The implication for the bio-safety application is that larger models (Gemma 2 9B on Colab T4; Gemma 2 27B / Llama 3.1 70B on A100) should show *stronger* divergence signals per prompt than Gemma 2 2B, even though they sit further above the ~1.7 B SAE-dilution boundary. The signal is stronger; the SAE decomposition is noisier. These two effects partially cancel but the net is expected to favor larger models for this measurement.

This is the basis for the `notebooks/colab_biorefusalaudit.ipynb` sweep targeting Gemma 2 9B-IT and Llama 3.1 8B-Instruct as stretch targets.

## What is claimed + what is not

**Claimed:** the layer-selection methodology (mid-depth / ~50 % relative depth) transfers cleanly to Gemma 2 2B-IT + Gemma Scope 1 layer 12.

**Claimed:** the SAE-decomposition-dilutes-at-scale finding informs the expected resolution of divergence measurements on Gemma 2 2B vs. smaller models.

**Not claimed:** the specific deception-detection numbers cited above transfer to bio-safety refusal detection. The tasks are sufficiently different that the numerical accuracies do not import.

**Not claimed:** BioRefusalAudit achieves a specific accuracy on bio-safety. Real accuracy + calibration numbers come from the eval-set runs reported in `paper/writeup.md §4`.
