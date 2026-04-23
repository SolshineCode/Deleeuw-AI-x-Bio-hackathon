# Compute Requirements

This document spells out the hardware footprint for local development, the A100 rental path for the cross-model sweep (deferred stretch), and reproduction guidance.

## Local dev

Tested floor: **NVIDIA GTX 1650 Ti Max-Q, 4GB VRAM**, Windows 11, Python 3.13. This is what the MVP sprint ran on.

| Model | Precision | Fits on 4GB? | Notes |
|---|---|---|---|
| `google/gemma-3-270m-it` | fp16 | Yes | Primary local target. Full eval fits comfortably. |
| `google/gemma-3-1b-it` | bnb 4-bit | Yes (tight) | Requires `bitsandbytes` wheel — see [TROUBLESHOOTING.md](../TROUBLESHOOTING.md). |
| `google/gemma-4-e2b-it` | bnb 4-bit | Yes (tight) | Same 4-bit path as 1B. |
| `google/gemma-3-4b-it` | bnb 4-bit | No on 4GB — rent A100 | 10GB min even quantized. |
| `google/gemma-3-12b-it` | bnb 4-bit | No — rent A100 | 24GB min. |
| `meta-llama/Llama-3.1-8B-Instruct` | bnb 4-bit | No — rent A100 | 16GB min. |

**Wall clock per model** on 75 prompts (full public eval set plus tier-3 when gated dataset is in place): **approximately 20–40 minutes each** on the 1650 Ti Max-Q, dominated by SAE activation collection during generation. The judge consortium adds 5–10 minutes per model, parallelizable.

## A100 sweep (deferred stretch)

The full cross-model scaling figure in the paper requires the larger Gemma 3 sizes and the Llama 3.1 8B cross-architecture comparison. These need a rented A100 80GB.

- **Providers:** [RunPod](https://runpod.io) and [Lambda](https://lambdalabs.com). RunPod has lower-friction spin-up for short jobs; Lambda is usually a bit cheaper for sustained use.
- **Full 5-model sweep:** 270M, 1B, 4B, E2B, Llama 3.1 8B over the 75-prompt eval set, including judge consortium and calibration fit per model. Estimated wall clock **3–4 hours**. Estimated cost **$30–45** at current A100 80GB on-demand pricing.
- **Setup:** `scripts/setup.ps1` is Windows-first; on the rental, use the Linux bash equivalent in `scripts/setup.sh` (or run the same pip install line manually). The Gemma Scope 2 weights are ~10–30 GB per model depending on layer coverage — factor download time into the wall-clock budget.

This sweep is deferred until post-submission and will run only with explicit approval of the compute spend.

## Why not Colab

The natural alternative for short GPU jobs would be Colab free tier. It does not work well here for three reasons:

1. **Memory.** Loading a quantized Gemma 3 model, its Gemma Scope 2 SAE, and the activation-collection buffers simultaneously is tight even on the T4 (16GB). Free tier occasionally gets throttled or preempted.
2. **Wall clock.** Free T4 sessions time out before a full 5-model sweep completes. Even a single 12B run is marginal.
3. **Network and HF gating.** Gemma 3 is HF-gated; pulling weights into a fresh Colab session every run is slow and re-burns the gated-download path. A persistent A100 rental avoids this.

If you want to prototype feature discovery or dashboard changes on Colab, it is fine — just don't expect the full eval to run there.

## Reproduction

The end-to-end pipeline for the 270M local run is wrapped in a single script:

```powershell
.\scripts\run_eval.ps1
```

This activates `.venv`, sets required env vars, and runs:

```powershell
python -m biorefusalaudit.cli run `
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl `
    --model google/gemma-3-270m-it `
    --out runs/repro
```

Output lands in `runs/repro/`: `report.md`, `report.json`, per-prompt traces, and the flag list. The paper's local-tier numbers should reproduce within calibration noise.
