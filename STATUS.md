# Status

**As of:** 2026-04-22
**Hackathon:** [AIxBio Hackathon 2026](https://apartresearch.com/sprints/aixbio-hackathon-2026-04-24-to-2026-04-26), April 24–26, Track 3 (AI Biosecurity Tools), sponsored by Fourth Eon Bio.

## Build state — MVP sprint in progress

What is in place and verified working:

- **Package scaffold** — `biorefusalaudit/` full layout: scoring (divergence + calibration), prompts (loader + stratifier + safety_review), judges (regex + consortium + Gemini/Haiku/Ollama LLM adapters), models (HF adapter + SAE adapter with TopK + JumpReLU), features (profiler + validator + Neuronpedia discovery), runner (single-model + cross-model), reporting (report + tier-3 redaction), Click CLI.
- **42 unit tests green** under `pytest -m "not integration"`.
- **Eval set** — 75 prompts in `data/eval_set_public/eval_set_public_v1.jsonl` (23 benign + 30 dual-use + 22 tier-3 category descriptors). Tier-3 hygiene check (`biorefusalaudit.cli check-safety`) returns clean.
- **Core divergence metric** — correct surface → feature mapping in the prior T (refuse → refusal_circuitry, comply → bio_content, hedge → hedging, evade → deception_correlate). Synthetic demo produces the expected pattern (benign 0.02, dual-use 0.11, tier-3 0.47).
- **Judge consortium** — regex first-pass verified on sample completions (comply / refuse / hedge all classify correctly). Gemini + Claude Haiku + Ollama adapters wired; full consortium path exercised by unit tests.
- **Real model loading verified** — Gemma 2 2B-IT + Gemma Scope 1 SAE (`layer_12/width_16k/average_l0_82`) loads and generates completions on CPU. Smoke test at `runs/smoke/` produced a real report.
- **Colab T4 notebook** — `notebooks/colab_biorefusalaudit.ipynb` runs the full eval pipeline on free Colab T4 for Gemma 2 9B-IT + Llama 3.1 8B-Instruct (bnb 4-bit) with valid, verified sae_lens release IDs. Open-in-Colab badge in `notebooks/README.md`.
- **Streamlit dashboard** — skeletal; paste-a-prompt → classification + top-k features + divergence wired.
- **Demo artifact** — `demo/scaling_plot_synthetic.png` (70 KB matplotlib scaling plot over 4 synthetic runs).
- **Paper writeup** — draft at `paper/writeup.md` documenting synthetic pipeline validation + Colab plan + limitations.

## Caveats known and documented

- **Feature catalogs are stubs, not hand-validated.** `data/feature_catalog/gemma-2-2b-it.json` + `gemma-4-E2B-it.json` use plausible-but-not-validated indices. The local smoke test produced zero feature activation projection (catalog indices didn't match the SAE's actually-firing features on bio prompts). Real validation requires the Colab pass + `feature_validator.differentiation_check`. Documented in `docs/METHOD.md §Known failure modes` and in the catalog files' `catalog_note` field.
- **Calibration T is prior-only**, not fit on real activation data. `fit_alignment_matrix` is tested + working; fitting runs once Colab outputs arrive.
- **CPU-only torch on this box.** `.venv` has torch 2.11.0+cpu — no CUDA wheel for Python 3.13 on Windows as of 2026-04-22. Local full 75-prompt run takes ~3–5 hours on CPU, so real numbers come from the Colab T4 path.
- **Gemma Scope 2 for Gemma 3 is not publicly released** as of 2026-04-22. Gemma 3 weights cached for forward compatibility; SAE path dispatches with `skip_reason` in `configs/models.yaml`. MVP uses Gemma 2 + Gemma Scope 1 as the primary demonstration path and Gemma 4 E2B + custom SAE as the "custom SAE portability" row.
- **bitsandbytes not installable** on Python 3.13 / Windows — 4-bit quantization path works only on Colab. Documented in `TROUBLESHOOTING.md`.

## Explicitly NOT in the MVP

Deferred to post-submission approval:

- **A100 cross-model sweep** — Gemma 3 4B / 12B, Llama 3.1 70B etc. Deferred until after submission and pending approval of the compute spend (see `docs/COMPUTE.md`).
- **Full Neuronpedia automated feature discovery** — `feature_discovery.py` is wired for the Neuronpedia search API; hand-curation is the primary path for v0.1.
- **Transcoder analysis (S2 stretch)** — Gemma Scope 2 cross-layer transcoders not in MVP.
- **Jailbreak-correlation study (S3 stretch)** — not in MVP.
- **Unlearning before/after (S4 stretch)** — not in MVP.

## Next planned (pre-submission)

1. Open `notebooks/colab_biorefusalaudit.ipynb` on Colab T4, run both Gemma 2 9B-IT and Llama 3.1 8B-Instruct cells (~90 min wall clock).
2. Pull the resulting `runs/colab_*/report.{md,json}` back to the repo, regenerate `demo/scaling_plot.png` via `scripts/build_scaling_plot.py`.
3. Use `feature_validator.differentiation_check` on Colab activation dumps to auto-tune the feature catalog from real data; update `data/feature_catalog/*.json`.
4. Fit calibration T on the tuned catalog's positive-control activations; update `configs/calibration_gemma2_2b.yaml`.
5. Fill in paper numbers in §4.
6. Record 60–90 second demo video of the Streamlit dashboard walkthrough.
7. Submit.

## Build branch

- Active branch: `hackathon-mvp-sprint`
- **Not yet pushed.** Per standing push policy, remote pushes wait for explicit approval.
