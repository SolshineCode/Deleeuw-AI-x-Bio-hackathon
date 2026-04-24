# CLAUDE.md — BioRefusalAudit / AIxBio Hackathon 2026

Operative instructions for any Claude Code session in this repo. Tight by design — additions should replace/consolidate, not accrete. Project-specific detail lives in `STATUS.md`, `paper/writeup.md`, `docs/METHOD.md`, and `TROUBLESHOOTING.md`.

---

## Project context

**v1.0 (hackathon submission):** Tool for measuring **refusal depth** — the divergence between what an LLM's *surface behavior* says ("I refuse") and what its *internal SAE feature activations* show ("hazard features still firing") — to tell structurally-safe refusals from shallow ones on bio-safety prompts. AIxBio Hackathon 2026, Track 3 (Biosecurity Tools), sponsored by Fourth Eon Bio. Methodology lineage: nanochat-SAE-deception-research + Secret Agenda (arXiv:2509.20393).

**Planned follow-on arc:** Neuronpedia validation revealed the auto-tuned Gemma Scope catalog encodes generic technical-governance vocabulary rather than bio-specific features. The fix — motivated by Secret Agenda's behavioral-sampling methodology — is domain-specific SAE fine-tuning on bio-hazard behavioral activation corpora (base vs. RLHF model pairs; genuine vs. shallow refusals; institutional CBRN datasets). Two tracks: Track B (projection adapter, feasible with 75-prompt corpus) and Track A (full SAE fine-tune, requires ~10K samples from institutional partners). Non-bio control experiment (`runs/control-legal-financial-gemma2/`) tests whether D currently measures bio-specific refusal or domain-agnostic sensitivity routing. Full technical spec: `docs/METHOD.md §Planned extension`.

---

## 📘 Canonical docs to read / append to

- `STATUS.md` — current build state, verified capabilities vs acknowledged caveats. Update as reality changes.
- `paper/writeup.md` — ≤6-page submission paper. Fill in real numbers as runs complete; never delete prior rows — add updates / correction blocks.
- `paper/policy_brief.md` — one-page governance framing (CLTR/AISI/AISI-UK audience).
- `docs/METHOD.md` — full formalization of the divergence metric, flag conditions, calibration procedure.
- `TROUBLESHOOTING.md` — known gotchas (KMP OpenMP workaround, sae_lens ID enumeration, Python 3.13 + bnb, Gemma Scope 2 release status). Add to it whenever a new gotcha burns you.
- `SAFETY.md` — tier-gated release policy; tier-3 hygiene rules; disclosure contact.

---

## ⛔ Data-obsessiveness directive (inherited from nanochat-SAE-deception-research)

Save everything. Every completion, every classification decision, every per-judge vote, every seed, every attention-mask warning. If a script generates it, the script persists it.

**Concrete requirements for this repo:**

- **Eval runs** (`runner.eval_runner.run_eval`): emit per-prompt records to `runs/<model>/report.json` with schema `{id, tier, framing, prompt, completion, surface_label, surface_soft, feature_vec, feature_categories, divergence, flags, judge_disagreement}`. Already implemented by `reporting.report.PromptRecord`.
- **Judge consortium passes** (`judges.consortium.aggregate_votes`): record every judge's independent vote on each sample — judge name, label, confidence, raw response — before aggregation collapses them. The `ConsortiumVerdict.votes` field carries this; don't drop it.
- **SAE activation dumps**: when running `feature_validator.differentiation_check` or any catalog-tuning pass, persist the full `(prompt_id, feature_vec[d_sae])` to an `.npz` under `runs/<model>/activations.npz`. The stub-vs-validated-catalog rewrite needs these for reproducibility.
- **stderr preserved** in `runs/<model>/stderr.log` — not grep-filtered. If the CLI dies silently, the log is the forensics.
- **Re-runs with new seed / model / SAE ID**: new directory suffix (`runs/gemma-2-2b-it-seed7/`, `runs/gemma-2-2b-it-l0-82/`). Never overwrite a prior report — divergence numbers anchor the paper.
- **Calibration T updates** (`configs/calibration_<model>.yaml`): on every re-fit, append a block with `fit_date`, `fit_samples`, `fit_method`, and the resulting T. Keep every historical T so re-runs of earlier reports reproduce byte-identically.

Rule of thumb: if a reviewer (or future-you) might want to re-analyze this data with a different calibration matrix, a different catalog, a different judge weighting, a different divergence formula — and the data isn't saved — you have failed the directive. **Generate once, save forever, re-analyze infinitely.**

---

## ⛔ Safety / scope hygiene directive (project-specific)

**Non-negotiable rules for any eval-set or prompt-handling change:**

1. **Tier-3 hazard content never enters the public repo.** The public JSONL (`data/eval_set_public/eval_set_public_v1.jsonl`) carries category-level descriptors only. Full tier-3 prompt bodies live on the HL3-gated HF dataset (`SolshineCode/biorefusalaudit-gated`, not-yet-published). The `safety_review.check_no_hazard_bodies` unit test is the gate; never suppress its failures.
2. **Tier-3 model completions are always redacted before serialization.** `reporting.redaction.redact_tier3` runs on every report write; `test_redaction.py` asserts this; never bypass.
3. **Feature catalog publication**: catalogs (`data/feature_catalog/*.json`) contain only feature indices + category labels — never the hazardous prompts that validated them.
4. **No actionable uplift in any prompt we author.** Eval set authoring adheres to the guardrails in `data/eval_set_public/schema.md`: no specific pathogen names paired with enhancement techniques, no synthesis protocols, no agent-selection strategies. The test "a biology professor could say this aloud in a public lecture" applies.

If an experiment plan looks like it might violate any of the above, stop and escalate to Caleb before running.

---

## Non-negotiable user directives (broader patterns)

1. **Keep the local GPU running work continuously.** CUDA torch install is mandatory (not CPU fallback); long-running jobs dispatch to `run_in_background=true`; queue the next GPU job as one finishes. **Any time the GPU becomes free, immediately start the next relevant GPU job without waiting to be asked.** Priority queue: (a) cross-model format ablations, (b) longer SAE training runs, (c) additional intervention experiments, (d) flagship reruns with new seeds. Releases only when Caleb says stop. See `~/.claude/projects/C--Users-caleb/memory/feedback_gpu_always_running.md`.
2. **No push / PR / gh-comment without explicit approval.** Branch stays local until Caleb says "push it." This supersedes any perceived urgency.
3. **Brutal honesty on confidence.** Flag stub catalogs as stubs, prior-only calibration as prior-only, synthetic numbers as synthetic — before they get confused for real results in the paper.
4. **Consolidate related work into single unified PRs.** Per-module PRs add review overhead; the hackathon submission wants one coherent PR.
5. **New experiments get new subfolders.** `runs/<model>-<modifier>/` — never overwrite a prior run's outputs.
6. **Append-only for `STATUS.md`, `paper/writeup.md`, `demo/README.md`.** Corrections add inline `CORRECTED YYYY-MM-DD` blocks; never silently rewrite history.
7. **Don't touch `~/.claude/handoff.md`.** That belongs to the deception-nanochat-sae-research sprint. Sprint-specific notes go to `~/.claude/session-notes/biorefusalaudit-sprint.md`.
8. **TDD-style plan structure** (red / green / refactor). Failing assertions first, minimum change to pass, enrichment last. Applied to code and to docs.
9. **Full automation.** Don't ask Caleb to "open browser and paste X" mid-session; if a browser action is needed, use `mcp__claude-in-chrome__*` tools to drive it.

---

## Git / PR / push protocol

- Git identity: `SolshineCode`, `caleb.deleeuw@gmail.com`.
- Never commit directly to `main`. Always: feature branch → PR.
- Every PR on approval: `MSYS_NO_PATHCONV=1 gh pr comment <PR#> --body "/gemini review"` — the `MSYS_NO_PATHCONV=1` prefix is required on Windows/Git Bash or the `/gemini` gets rewritten to a Windows path.
- Merge only after Gemini review has landed + Caleb sign-off. Squash-merge.
- HL3-FULL license applies to all code; CC-BY-4.0 to tiers 1–2 eval data; HL3-Gated to tier-3.

---

## File storage policy

- `.pt` weights (SAE checkpoints, fine-tune adapters): HuggingFace only, under `SolshineCode/` or `Solshine/`. **Never on GitHub.**
- HF pushes require **explicit** per-repo approval + external review. Local prep (convert, stage, write README) is fine.
- Run outputs (`runs/`): local + gitignored. Regeneratable from scripts. Committed artifacts go in `demo/` (scaling plots, smoke samples, screen recordings).
- Cached model / SAE weights live under `%USERPROFILE%\.cache\huggingface\` — never track or copy.
- `.venv/`, `.env`, tokens: gitignored. Only `.env.example` is committed.

---

## Quick start

```bash
cd /c/Users/caleb/projects/Deleeuw-AI-x-Bio-hackathon

# Activate venv (Git Bash):
source .venv/Scripts/activate

# Verify CUDA:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Run unit tests (required to be green before any commit):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m pytest tests/ -q -m "not integration"

# Synthetic plumbing check (no GPU needed):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/synthetic_demo.py

# Real smoke (GPU preferred; CPU fallback is documented):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python scripts/smoke_activation_test.py

# Full eval (one model):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out runs/gemma-2-2b-it \
    --sae-source gemma_scope_1 \
    --sae-release gemma-scope-2b-pt-res \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --catalog data/feature_catalog/gemma-2-2b-it.json \
    --calibration configs/calibration_gemma2_2b.yaml

# Rebuild the cross-model scaling plot from any runs/*/report.json:
python scripts/build_scaling_plot.py --out demo/scaling_plot.png
```

**Always use `KMP_DUPLICATE_LIB_OK=TRUE`** on Windows when torch + numpy both load — OpenMP conflict otherwise.

**Enumerate valid sae_lens IDs** before invoking a new release:
```python
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
print([i for i in get_pretrained_saes_directory()['gemma-scope-2b-pt-res'].saes_map.keys() if 'layer_12/width_16k' in i])
```

API keys: `GEMINI_API_KEY` (required), `ANTHROPIC_API_KEY` (optional — otherwise `claude -p` subprocess), `HF_TOKEN` (required for Gemma family). Never commit; `.env.example` is the template.

---

## Code-modification discipline

Peer-review-grade repo. Every code change traceable:
- Never overwrite a previously-run experimental script without a changelog comment block at the file top (date, what changed, why).
- Preserve original scripts in git history.
- Every experiment run: parameters, results, timestamps in the result JSON itself (already done via `RunReport.model_name` + `sae_source` + `eval_set_path`).

---

## Architecture gotchas (keep updated — each one burned us once)

1. **sae_lens segfaults on Python 3.13 + Torch 2.6 (Windows).** Do not use it. Bypass is `_load_gemma_scope_direct` in `sae_adapter.py` — fetches `.npz` weights directly from HF.
2. **Device mismatch.** SAE modules must be cast to the activation's device dynamically. Fixed in `sae_adapter.py` with `to(target_device)` + fp32 cast.
3. **KMP OpenMP conflict.** Always prefix runs with `KMP_DUPLICATE_LIB_OK=TRUE` on Windows. Documented in `TROUBLESHOOTING.md`.
4. **Gemma 4 community SAEs** use non-standard weight filenames (not `sae_weights.pt`). Fixed in `sae_adapter.py` with repo-scan fallback.
5. **`device_map="auto"` silently routes Gemma 4 to CPU.** `Gemma4ForConditionalGeneration` is multimodal; `accelerate`'s memory estimator mis-sizes it and picks CPU even with 4 GB free VRAM. Fix: `device_map={"": 0}` for quantized models when CUDA is available — implemented in `model_adapter.py`. See `TROUBLESHOOTING.md` for full diagnosis and the misleading log message that looks like a capacity error but is actually a bytes-level metadata artifact.
6. **Residual hook accumulates per-step tensors during generation.** The hook fires once per autoregressive token. Appending each capture to a list fills VRAM (~240 MB for a 200-token generation) and causes silent CPU spill on 4 GB cards, making long generations 20× slower. Fix: overwrite `captured[0]` instead of appending — implemented in `model_adapter.py`. Symptom: alternating fast (1–3 s) and slow (150–200 s) prompts in the same run.
7. **Catalog tuning must use activations from the same quantization regime as the final eval.** If you auto-tune on fp16 activations but eval in 4-bit NF4, the selected catalog features won't fire in the 4-bit run → `feature_vec = 0` → `D = 1.0` everywhere (not a real result — it's a silent catalog failure). Always collect `--dump-activations` under the same `--quantize` flag as the final eval. Similarly, calibration T can only be fitted meaningfully once `feature_vec ≠ 0`; fitting T on degenerate zero-feature data produces a placeholder, not a calibration. Full chain: `pass-1 (correct quantize) → auto_tune → pass-2 (dump) → fit_T → pass-3 (final)`. See `docs/METHOD.md §Required chain ordering`.

---

## Per-session reminders

- Re-read the data-obsessiveness + safety-hygiene directives at the start of any session that touches runs, eval sets, or model weights.
- Save GPU time: verify `torch.cuda.is_available()` → True before any run. If CPU-only, stop and fix (see `feedback_gpu_always_running.md`).
- Run `pytest -m "not integration"` before every commit (must be green). Integration tests gate on real-model loads; skip on CPU CI.
- Flag any deviation from `STATUS.md` "What's in place" list in your reply to the user — they need to know when something claimed-working slipped.
- HF pushes and remote pushes are SEPARATELY gated — approval for one doesn't imply approval for the other.

---

*This file is load-bearing context for every session in this repo. Additions should replace or consolidate, not accrete. Detailed historical notes live in `STATUS.md`, `paper/writeup.md`, and the canonical docs linked above.*
