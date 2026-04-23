# Why Layer 12? — Provenance of the Residual-Stream Probe Point

## Short answer

Layer 12 of Gemma 2 2B-IT sits at **46 % relative depth** (26-layer model), inside the mid-depth band (~40–60 %) where behavioral signal is most robustly present across architectures. BioRefusalAudit uses `google/gemma-scope-2b-pt-res` layer_12 / width_16k / average_l0_82 as the canonical probe SAE for exactly this reason.

The mid-depth-band finding did not originate with this hackathon. It is the output of a multi-model layer sweep in the author's prior behavioral-signal SAE work and is summarized below with source citations.

## The mid-depth pattern

From the author's `FINDINGS.md §F5` (behavior-detection research repo):

> Across three architecture families (nanochat/GPT-2-like, Qwen2.5/3, Llama 3.2), the behavioral signal peaks in mid-network layers at ~39–63 % relative depth — not at early or late layers. nanochat-d32 L12/32 (39 %); Qwen3-1.7B L17/28 (61 %); Llama-3.2-1B L8/16 (50 %).

Source: `/c/Users/caleb/deception-nanochat-sae-research/FINDINGS.md` §F5 lines 118–135.

The layer-sweep source data is copied into this folder verbatim:

- `data/nanochat_balanced_layer_sweep.json` — behavioral-signal probe accuracy at each of layers 0–32 on nanochat-d32, showing the L12 peak.
- `data/v3_llama_1b_layer_sweep.json` — the parallel sweep on Llama-3.2-1B showing the L8 peak at 50 % depth.
- `data/layer_behavioral_signal_profile.json` — the detailed per-layer signal profile across L0–L16 on Llama with the bimodal (L0–L2 + L7–L9) suppression-head distribution.

## Why mid-depth rather than shallow or deep

From `RESEARCH_RESULTS.md` (same repo, lines 293–297):

> The behavioral signal is: **absent in early layers** (L2–L8 near/below chance) — NOT a low-level embedding artifact; **peaks at layer 12** (mid-network), remains strong at L16 — consistent with semantic/behavioral processing; **declines in late layers** (L20–L28) — NOT a next-token prediction artifact; range of 32 pp (39 % to 70 %) — a genuine layer-specific signal, not flat noise. This is exactly the predicted pattern for a real behavioral representation: early layers handle syntax, mid layers handle semantics/behavior, late layers prepare output logits.

Source: `/c/Users/caleb/deception-nanochat-sae-research/RESEARCH_RESULTS.md` lines 293–297.

In plain terms: the residual stream carries different kinds of information at different depths. Very early layers are still untangling tokenization and syntax. Very late layers are already compressing toward next-token logits. The middle is where a model is representing the *content* of what it has read and what it is about to say — which is where any behavioral probe needs to look to find features that are not pure surface or pure logit prediction.

## The specific Gemma 2 2B-IT layer-12 choice

Gemma 2 2B has 26 transformer blocks. Layer 12 sits at **46 % depth** — dead centre of the 40–60 % mid-depth band. The pre-trained Gemma Scope 1 residual-stream SAE at this layer is publicly available from Google DeepMind at width_16k with five L0 sparsity options (average_l0 ∈ {22, 41, 82, 176, 445}).

BioRefusalAudit's default uses `average_l0_82` — a middling sparsity that prior work (`LEARNINGS_2026_04_03.md` lines 49–56) identifies as a reasonable operating point for behavioral probing, where the SAE is sparse enough to be interpretable (≈ 80 features active per token) but dense enough to capture the behavioral signal that gets lost at k < 32.

## Alternative layers considered

`configs/models.yaml` keeps layers 5 and 20 available as sweep targets so that the L12 choice is empirically re-validated for each model family we probe. The provenance note for each entry cites both the mid-depth band rationale (this document) and the cross-model-sweep validation (`cross_model_scaling.md`).

For the hackathon submission we report L12 results as the headline and use L5 / L20 sweeps only as robustness checks in the appendix.

## Signal strength scales with model size

From `LEARNINGS_2026_04_03.md` (same repo, lines 19–28):

> Signal Strength Scales With Model Size. 1.88 B gets 86.9 %, 1.3 B gets 76.2 %, 561 M gets 66.1 %. Roughly linear with log(params). If this holds, larger models (8 B+) should show even stronger behavioral signals. The layer-depth profile also shifts with scale: the 1.88 B model peaks at 39 % depth (mid-network). The two smaller models peak at 95–100 % depth (near output).

This is consequential for the bio-safety application. Models bigger than Gemma 2 2B (which we can run on Colab T4 and in follow-on A100 work) should show the behavioral signal at the same relative depth, *with greater signal strength*. The cross-architecture scaling argument in §4 of the paper leans on this result.

Source: `/c/Users/caleb/deception-nanochat-sae-research/LEARNINGS_2026_04_03.md` lines 19–28.

## Note on framing

The prior-work findings cited here originate from research on detecting *behavioral* signals in LLM activations. BioRefusalAudit applies the same substrate (mid-depth residual SAEs) to the specific behavioral question of *bio-safety refusal depth*. The methodology — layer choice, SAE architecture, sparsity target, activation-collection pipeline — transfers without modification. The specific *findings* about behavioral-signal separability do not transfer automatically and are not claimed here; we re-validate on the 75-prompt eval set.
