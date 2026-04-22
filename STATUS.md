# Status

**As of:** 2026-04-22
**Hackathon:** [AIxBio Hackathon 2026](https://apartresearch.com/sprints/aixbio-hackathon-2026-04-24-to-2026-04-26), April 24–26, Track 3 (AI Biosecurity Tools), sponsored by Fourth Eon Bio.

## Build state — MVP sprint in progress

What is in place:

- **Package scaffold** — `biorefusalaudit/` layout per the README repo structure is laid down; CLI entry point present.
- **Eval set** — 75 prompts authored across tiers 1–3 and four framing axes. Tier 1+2 (53 prompts) are in `data/eval_set_public/`. Tier 3 (22 prompts) is staged for the gated HF dataset.
- **Core divergence metric** — `biorefusalaudit/scoring/divergence.py` implements `D(s, f) = 1 - cos(f, T^T·s)`. Calibration scaffold in `calibration.py`.
- **Judge consortium** — regex first-pass, Gemini, Claude Haiku (via `claude -p` subprocess), and a local-model tertiary judge wired through `judges/consortium.py` with weighted voting and disagreement scoring.
- **3-model local runs queued** — Gemma 3 270M-IT (fp16), Gemma 3 1B-IT (bnb 4-bit), Gemma 4 E2B-IT (bnb 4-bit) on the GTX 1650 Ti Max-Q dev box.
- **Streamlit dashboard** — skeletal; paste-a-prompt path wired, side-by-side comparison and batch mode stubbed.
- **Paper writeup** — in progress at `paper/writeup.md`; structure and methods sections drafted, numbers pending the 3-model run.

## Explicitly NOT in the MVP

Deferred to post-submission approval:

- **A100 cross-model sweep** — the 4B, 12B, and Llama 3.1 8B runs require rented compute. Deferred until after submission and pending approval of the compute spend (see [docs/COMPUTE.md](docs/COMPUTE.md)).
- **Neuronpedia automated feature discovery** — for MVP, feature catalogs are hand-validated. Automated discovery via Neuronpedia API is scoped for a follow-up.
- **Transcoder analysis (S2 stretch)** — Gemma Scope 2 cross-layer transcoders are in `transcoder_adapter.py` as stubs only.
- **Jailbreak-correlation study (S3 stretch)** — correlating divergence with jailbreak susceptibility is not in MVP.
- **Unlearning before/after (S4 stretch)** — measuring whether unlearning reduces divergence is not in MVP.

## Next planned (pre-submission)

1. Complete end-to-end 3-model local run (Gemma 3 270M-IT, Gemma 3 1B-IT, Gemma 4 E2B-IT).
2. Fill in paper numbers: per-tier means, cross-model scaling plot, flag counts, representative case studies.
3. Record 60–90 second demo video of the Streamlit dashboard walkthrough.
4. Submit.

## Build branch

- Active branch: `hackathon-mvp-sprint`
- **Not yet pushed.** Per standing push policy, remote pushes wait for explicit approval.
