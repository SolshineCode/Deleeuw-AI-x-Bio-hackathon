# Specialist Review — 2026-04-23

Received a third-party review of the repo while the Layer-12 eval was running. Recording it here for traceability.

## Review verdict (summary)

> "Credible MVP spine, but not yet submission-hardened because the central mechanistic claim still rests on synthetic behavior, stub feature catalogs, and prior-only calibration rather than validated activation-grounded evidence."

> "The main work now is not a rewrite; it is tightening evidence, causal validation, reproducibility, and the story judges will see first."

Accepted. The review is correct.

## Concerns raised (prioritized response)

| # | Concern | Blocker? | Immediate action |
|---|---|---|---|
| 1 | Feature catalogs are unvalidated stubs; local smoke produced zero feature activation on bio prompts | **Top blocker** | Write `scripts/auto_tune_catalog.py`, collect per-prompt d_sae activations from a fresh eval, replace stub indices with features that actually fire on biology content (Cohen's-d vs neutral baseline). |
| 2 | Calibration T is prior-only | High | Fit T on real (s, f) pairs *after* catalog is tuned. Report pre/post deltas. |
| 3 | Eval numbers in paper are synthetic | High | Section 4 placeholders get replaced with real numbers once the activation-collection run + catalog tune complete. |
| 4 | Judge consortium real-world coverage | Medium | Turn on `--use-llm-judges` for the final run (Gemini + `claude -p` + Ollama) — adds disagreement reporting + confidence bands automatically via `aggregate_votes`. |
| 5 | Compute schedule risk | Medium | Freeze the Colab notebook environment path and make `scripts/run_eval.ps1` produce a one-command reproduction with expected runtime + artifact names. |
| 6 | Cross-model story scope | Low | Keep narrow: one flagship (Gemma 2 2B-IT + Gemma Scope 1 L12) + one portability row (Gemma 4 E2B-IT + Solshine custom SAE). Drop the aspirational Gemma 3 + Llama 3.1 framing in the paper. |
| 7 | Demo is skeletal | Medium | Simplify dashboard to one-prompt-one-verdict-one-feature-panel-one-intervention view. Cut everything else. |
| 8 | Repo signal-to-noise | Low | Consolidate submission narrative into README as problem / method / evidence / limitations / demo. Already close. |
| 9 | Branch on remote | RESOLVED | PR #1 merged to main as `1b7d0c4` on 2026-04-23 01:30 UTC. |

## Circuit-tracing integration (new requirement)

The review asks for Anthropic-style circuit tracing + attribution — not buried inside the scalar divergence metric, but as a parallel artifact pipeline. Specific requirements:

1. **Trace-selection stage** — pick the interesting cases after scoring: top high-divergence, top false refusals, top false compliances, top judge-disagreement.
2. **Token-span tracing** — for refusals, inspect hazard-recognition tokens vs. refusal-planning tokens vs. wording tokens separately, not just the final label.
3. **Minimal-pair graph diffs** — benign vs. dual-use variant, safe vs. tier-3 descriptor, neutral vs. jailbreak phrasing.
4. **Intervention tests as first-class outputs** — damp a claimed refusal feature, boost a claimed bio-content feature, measure the completion change. A feature must pass BOTH activation relevance AND intervention relevance to be called a "circuit" in the report.
5. **Save graph JSON alongside report.json.** Dashboard renders a compact panel.
6. **Circuit-confidence field** in reporting combining graph sparsity + intervention effect size + judge agreement.

**Proposed new modules** (per review's schema):

- `biorefusalaudit/runner/trace_selected_cases.py`
- `biorefusalaudit/features/attribution_labels.py`
- `biorefusalaudit/reporting/attribution_section.py`
- Streamlit pane for graph summary + interventions.

**Design rule from review:** every reported circuit claim has three linked artifacts: activation evidence, attribution graph evidence, perturbation evidence.

## Submission-freeze plan (revised)

From the review's closing:

> "Freeze scope around one flagship result: a real Gemma 2 run with validated feature hits, calibrated divergence, a compact attribution graph for a representative refusal case, and one intervention showing that changing the identified feature changes the behavior. That package is much stronger than a broader but still partly synthetic cross-model story."

Priority order for the remaining GPU budget (~6h):

1. **Activation collection run** — fresh 75-prompt pass that dumps per-prompt d_sae=16384 feature vectors to `runs/<model>/activations.npz`. ETA ~1h GPU.
2. **Catalog auto-tune** — Cohen's-d on `benign_bio` vs `dual_use_bio` vs `hazard_adjacent_category` activations; replace stub indices with empirically-firing features. ETA ~15m CPU.
3. **Calibrate T** — fit `fit_alignment_matrix` on the now-validated (s, f) pairs. ETA ~5m CPU.
4. **Primary eval** — full 75-prompt run with tuned catalog + fit T + LLM judges on (`--use-llm-judges`). ETA ~1.5h GPU + LLM calls.
5. **Intervention experiment** — pick one high-divergence refuse-labeled prompt; ablate the top "refusal" feature; re-generate; compare. Then boost a "bio_content" feature on a benign prompt; re-generate; compare. ETA ~30m GPU + post-analysis.
6. **Attribution module scaffolding** — `trace_selected_cases.py` pulls the top-K cases by selection criteria and invokes intervention + records graph JSON. ETA ~1h code.
7. **Paper + dashboard update** — fill Section 4 with real numbers; fill Section 5 with intervention evidence; simplify dashboard to the one-flagship path. ETA ~1h.
8. **Submit.**

## Constraint discipline

Per review: "No claim should survive into the README, paper, or demo unless it is backed by a real run, a validated feature, and at least one causal intervention."

This is the acceptance gate for the submission. Any claim in README/paper/demo that doesn't have all three gets demoted to an "intended" / "planned" section or removed.

---

Reviewer feedback retained verbatim in the full message to the team; this note is the repo-facing summary + action log.
