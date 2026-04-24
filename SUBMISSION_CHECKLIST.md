# Submission Checklist — AIxBio Hackathon 2026

Per specialist review, no claim should survive into the README, paper, or demo unless it is backed by a real run + a validated feature + at least one causal intervention. This checklist enforces that gate.

## Pre-flight — what must be true

### Evidence

- [x] Tuned feature catalog committed (`data/feature_catalog/gemma-2-2b-it.json` with `catalog_version: 0.2-auto-tuned`) — Cohen's-d top-20 per category from `runs/gemma-2-2b-it-L12-activations/activations.npz`
- [x] Fitted T in `configs/calibration_gemma2_2b.yaml` with `fit_method: ridge least-squares (lambda=0.1) on report.json` and `fit_samples: 75`
- [x] Pass-2 real-numbers report at `runs/gemma-2-2b-it-L12-tuned/report.{md,json}` with non-zero per-tier mean divergence — benign 0.467, dual-use 0.655, hazard-adj 0.669
- [x] At least one intervention record at `runs/interventions/*.json` where `qualifies_as_named_circuit: true` — all 5 intervened prompts qualify (bio_004/021/027/069/074)

### Paper

- [x] §4.2 per-tier table filled with real pass-2 numbers — in `paper/writeup.md`
- [x] §4.3 intervention table filled with at least one real row — 4 rows with real ΔD
- [x] Every "named circuit" claim cross-references a specific intervention JSON — cross-refs `runs/interventions/*.json`
- [x] Limitations section states catalog + T origin + what was NOT attempted — §4.5 caveats block
- [ ] §4.4 cross-architecture table filled — **BLOCKING: Colab T4 run not yet executed**

### Artifacts

- [ ] `demo/scaling_plot.png` regenerated from real reports (not synthetic) — pending Colab results
- [ ] Dashboard loads latest real report on `streamlit run app/dashboard.py` — needs smoke verification
- [ ] At least one intervention panel renders in the dashboard
- [ ] `REVIEWER_QUICKSTART.md` one-command path verified from a clean clone

### Tests + infrastructure

- [x] `pytest -m "not integration"` passes cleanly — 56 tests green (confirmed 2026-04-23)
- [x] `biorefusalaudit check-safety --eval-set data/eval_set_public/eval_set_public_v1.jsonl` returns OK (confirmed 2026-04-23)
- [x] `biorefusalaudit trace-cases --report runs/gemma-2-2b-it-L12-tuned/report.json` returns 12 selected cases (confirmed 2026-04-23)

### Repo signal

- [x] README opens with: problem / method / evidence (real numbers) / limitations / demo — fabricated Gemma 3 4B numbers replaced with real Gemma 2 2B-IT results 2026-04-23
- [x] Branch state: all work on `main` via squash-merged PRs; no orphan branches
- [x] `paper/writeup.md` word count ≤ 3 500 — 1 772 words (confirmed 2026-04-23)

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

Each of these gets a one-paragraph "planned" entry in the paper's §8 Future Work, not a claim in the body.
