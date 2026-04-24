# Submission Checklist — AIxBio Hackathon 2026

Per specialist review, no claim should survive into the README, paper, or demo unless it is backed by a real run + a validated feature + at least one causal intervention. This checklist enforces that gate.

## Pre-flight — what must be true

### Evidence

- [x] Tuned feature catalog committed (`data/feature_catalog/gemma-2-2b-it.json` with `catalog_version: 0.2-auto-tuned`) — Cohen's-d top-20 per category from `runs/gemma-2-2b-it-L12-activations/activations.npz`
- [x] Fitted T in `configs/calibration_gemma2_2b.yaml` with `fit_method: ridge least-squares (lambda=0.1) on report.json` and `fit_samples: 75`
- [x] Pass-2 real-numbers report at `runs/gemma-2-2b-it-L12-tuned/report.{md,json}` with non-zero per-tier mean divergence — benign 0.467, dual-use 0.655, hazard-adj 0.669
- [x] At least one intervention record at `runs/interventions/*.json` where `qualifies_as_named_circuit: true` — 8/8 intervened prompts qualify (bio_004/016/021/027/060/066/069/074); bio_010/001/002 in progress (queue, chain running 2026-04-24)

### Paper

- [x] §4.2 per-tier table filled with real pass-2 numbers — in `paper/writeup.md`
- [x] §4.3 intervention table filled with at least one real row — 4 rows with real ΔD
- [x] Every "named circuit" claim cross-references a specific intervention JSON — cross-refs `runs/interventions/*.json`
- [x] Limitations section states catalog + T origin + what was NOT attempted — §4.5 caveats block
- [ ] §4.4 cross-architecture table filled — **BLOCKING: Colab T4 run not yet executed**

### Artifacts

- [x] `demo/scaling_plot.png` regenerated from real reports — 18 real `runs/*/report.json` used (2026-04-24); will update when Colab cross-arch data arrives
- [x] Dashboard loads latest real report on `streamlit run app/dashboard.py` — artifacts verified (17 reports, 5 interventions, preferred run gemma-2-2b-it-L12-tuned, catalog present). Browser smoke check complete (2026-04-24).
- [x] At least one intervention panel renders in the dashboard — 8 intervention JSON files present at `runs/interventions/` (as of 2026-04-24; 3 more in queue)
- [ ] `REVIEWER_QUICKSTART.md` one-command path verified from a clean clone

### Tests + infrastructure

- [x] `pytest -m "not integration"` passes cleanly — 56 tests green (re-confirmed 2026-04-24)
- [x] `biorefusalaudit check-safety --eval-set data/eval_set_public/eval_set_public_v1.jsonl` returns OK (re-confirmed 2026-04-24)
- [x] `biorefusalaudit trace-cases --report runs/gemma-2-2b-it-L12-tuned/report.json` returns 12 selected cases (re-confirmed 2026-04-24)

### Repo signal

- [x] README opens with: problem / method / evidence (real numbers) / limitations / demo — fabricated Gemma 3 4B numbers replaced with real Gemma 2 2B-IT results 2026-04-23
- [x] Branch state: all work on `main` via squash-merged PRs; no orphan branches
- [x] `paper/writeup.md` word count ≤ 3 500 — 3 492 words (confirmed 2026-04-24; fixed "Three"→"Four" prompts in §4.3)

## Submission gate

If all checkboxes above are green: tag the repo, push the tag, submit.

```bash
git tag -a v1.0-submission -m "AIxBio Hackathon 2026 submission"
git push origin v1.0-submission
```

## What's explicitly out of scope for v1.0

- Anthropic-style attribution-graph rendering (schema in place via `features/attribution_labels.py`; full integration deferred)
- Gemma 3 family evaluation (pending Gemma Scope 2 public release)
- A100 sweep beyond what Colab T4 can host
- Minimal-pair attribution-graph diffs (minimal pairs generated at `data/eval_set_public/minimal_pairs.json`; diff layer is post-submission)
- Blinded-audit rubric on scored outputs (tooling not yet built)
- **Domain-specific SAE fine-tuning** — Track B adapter and Track A full fine-tune (see `docs/METHOD.md §Planned extension`; motivated by control experiment and Neuronpedia validation; Phase 1 feasible post-hackathon with existing corpus)

Each of these gets a one-paragraph "planned" entry in the paper's §8 Future Work, not a claim in the body.

## Stretch: Colab SAE training notebook (new, planned during hackathon)

- [x] `notebooks/colab_gemma4_sae_training.ipynb` — T4 SAE fine-tuning on HF dataset + W&B logging + HF checkpoint save — **COMPLETE 2026-04-24**
  - Configurable: `HF_DATASET_REPO`, `HF_TEXT_COLUMN`, `HF_LABEL_COLUMN`
  - Logs: total loss, L_recon, L_sparsity, L_contrastive, L0, feature density per step to W&B
  - Saves checkpoint to HF every N steps
  - Multimodal architecture fix in place (`pick_layer` handles both CausalLM and ImageTextToText)
  - Public eval set fallback if gated dataset unavailable
  - Full spec: `docs/METHOD.md §Colab SAE Training Notebook`

## Nice-to-have before submission (not blocking)

- [x] Non-bio control experiment results: `runs/control-legal-financial-gemma2/report.{md,json}` — **COMPLETE 2026-04-23.** benign 0.573 / dual-use 0.672 / hazard 0.665. Bio d=1.29 (p=0.0001) vs. legal d=0.67 (p=0.052, NS). Paper §8 updated with full statistical comparison.
- [x] Dashboard loads and renders latest real report (streamlit smoke check) — verified 2026-04-24: loads gemma-2-2b-it-L12-tuned, correct D values, feature panel, intervention panel (bio_004 Named circuit? ✓ YES)
- [ ] Demo video (60–90 s)
