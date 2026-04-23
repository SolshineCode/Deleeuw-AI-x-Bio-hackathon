# Submission Checklist — AIxBio Hackathon 2026

Per specialist review, no claim should survive into the README, paper, or demo unless it is backed by a real run + a validated feature + at least one causal intervention. This checklist enforces that gate.

## Pre-flight — what must be true

### Evidence

- [ ] Tuned feature catalog committed (`data/feature_catalog/gemma-2-2b-it.json` with `catalog_version: 0.2-auto-tuned`)
- [ ] Fitted T in `configs/calibration_gemma2_2b.yaml` with `fit_method: ridge least-squares ...` and `fit_samples > 0`
- [ ] Pass-2 real-numbers report at `runs/gemma-2-2b-it-L12-tuned/report.{md,json}` with non-zero per-tier mean divergence on `benign_bio` (i.e., catalog is actually matching firing features)
- [ ] At least one intervention record at `runs/interventions/*.json` where `qualifies_as_named_circuit: true`

### Paper

- [ ] §4.2 per-tier table filled with real pass-2 numbers
- [ ] §4.3 intervention table filled with at least one real row
- [ ] Every "named circuit" claim cross-references a specific intervention JSON
- [ ] Limitations section states catalog + T origin + what was NOT attempted

### Artifacts

- [ ] `demo/scaling_plot.png` regenerated from real reports (not synthetic)
- [ ] Dashboard loads latest real report on `streamlit run app/dashboard.py`
- [ ] At least one intervention panel renders in the dashboard
- [ ] `REVIEWER_QUICKSTART.md` one-command path verified from a clean clone

### Tests + infrastructure

- [ ] `pytest -m "not integration"` passes cleanly
- [ ] `biorefusalaudit check-safety --eval-set data/eval_set_public/eval_set_public_v1.jsonl` returns OK
- [ ] `biorefusalaudit trace-cases --report <report>` returns selected cases

### Repo signal

- [ ] README opens with: problem / method / evidence / limitations / demo — no intermediate clutter above those five
- [ ] Branch state: one flagship run on `main`, no orphan feature branches
- [ ] `paper/writeup.md` word count ≤ 3 500

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
