# Overnight Session Plan — 2026-04-24 → 2026-04-25 (AIxBio Hackathon)

**Authored:** 2026-04-24 evening
**Sprint window:** AIxBio Hackathon 2026-04-24 → 2026-04-26 (exact cutoff time TBD; verify on the [Apart sprint page](https://apartresearch.com/sprints/aixbio-hackathon-2026-04-24-to-2026-04-26))
**Resources for tonight:** local GTX 1650 Ti (4 GB) overnight + likely Colab T4 in parallel
**Current PR open:** #15 ([host-rule alignment + advisor reframes](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/pull/15)), Gemini review landed (4 medium-priority comments on chat-template token form — see `notes/PR15_REVIEW_NOTES.md` if/when filed)

---

## Background — what triggered this plan

Advisor letter (Opus simulation, 2026-04-24) graded the submission as **70.75 / 100 — strong finalist, not top prize as submitted**, with the following critique buckets:

1. **Calibration circularity** — T fit on the same 75 prompts used for evaluation. The biggest validity threat. Already given a §4-head structural caveat in PR #15; the **structural fix** is a held-out calibration set.
2. **Auto-tuned catalog ≠ hand-validated catalog** — the methodology section claimed hand-validation, the implementation runs Cohen's-d auto-tuning on activations. PR #15 added a §3 `CORRECTED 2026-04-24` block. The **structural fix** is a Neuronpedia + max-activating-example hand-validation pass.
3. **Cross-arch §4.4 empty** — Gemma 2 9B-IT + Llama 3.1 8B-Instruct results pending. "An empty §4.4 is a wound" per the advisor. Fixable via the `notebooks/colab_biorefusalaudit.ipynb` Colab T4 path.
4. **Format ablation buried** — addressed in PR #15 with new §4.6 elevation; no new work needed.
5. **Policy brief underdeveloped** — `paper/policy_brief.md` should open with Sandbrink managed-access measurement-layer framing.
6. **Demo gap** — no video; Streamlit needs API keys and a GPU. Track 3 ostensibly favors practitioner-runnable tools.

---

## Tonight's priority queue (ranked by judge-impact-per-hour)

### Tier A — pre-GPU prep (CPU work, blocks Tier B)

- [ ] **Author held-out calibration prompt set** (~25 prompts; balanced across tiers + framings; no hazard bodies in the public file; same `schema.md` "biology professor at a public lecture" guardrail).
  - Output: `data/eval_set_public/calibration_holdout_v1.jsonl`
  - Public JSONL carries category-level descriptors only; tier-3 bodies on the HL3-gated HF dataset (or held back entirely if HF push hasn't been approved yet).
  - This unblocks the structural fix to advisor critique #1.
- [ ] **Colab T4 pre-flight checklist** for `notebooks/colab_biorefusalaudit.ipynb`.
  - HF_TOKEN in Colab userdata secrets (Gemma + Llama license-acceptance required).
  - GITHUB_TOKEN optional (push back) vs. download-results path.
  - Integer `device_map={"": 0}` already in `model_adapter.py` — Linux Colab doesn't need the KMP guard.
  - Expected wall clock: ~60–90 min per model on T4.
  - Output landing: `runs/colab_gemma2_9b_it_l20/report.json`, `runs/colab_llama31_8b_l16/report.json`.
  - Wire-up to `scripts/build_scaling_plot.py --include-synthetic` for §4.4 figure regen.

### Tier B — overnight, parallel resources

- [ ] **Local GPU**: held-out calibration pass on Gemma 2 2B-IT.
  - Pass A: collect activations on the held-out set under same `--quantize` flag as flagship (`--dump-activations`).
  - Refit T from held-out activations + held-out judge-aggregated S.
  - Pass B: re-run the original 75-prompt flagship eval with the held-out-fit T → fresh `runs/gemma-2-2b-it-L12-tuned-heldout-T/report.json`.
  - Document in writeup `§4.2` as a `CORRECTED 2026-04-24` block (append-only): new D table + delta from within-sample T.
  - Retires advisor critique #1 from "framing-only" to "structurally-resolved."
- [ ] **Caleb on Colab T4**: Gemma 2 9B-IT first (more familiar SAE substrate), then Llama 3.1 8B-Instruct.
  - Fills empty §4.4.
  - If both complete, regenerate cross-model scaling plot.

### Tier C — CPU work to interleave with the GPU job

- [ ] **Neuronpedia hand-validation pass** on top features in `data/feature_catalog/gemma-2-2b-it.json`.
  - Pull top 5–10 features per category (bio_content, hazard_adjacent, refusal_circuitry).
  - For each: Neuronpedia auto-label + max-activating examples.
  - File per-feature verdict: `validated` / `partial` / `rejected` with rationale.
  - Output: `data/feature_catalog/gemma-2-2b-it_validation_2026-04-24.md` (or extend the JSON with a `validation` block).
  - Update writeup §4.2 with a `CORRECTED 2026-04-24` block summarizing the hand-validation pass.
  - Partially retires advisor critique #2.
- [ ] **Policy brief rewrite** (`paper/policy_brief.md`).
  - Open with Sandbrink managed-access measurement-layer framing (cite April 23 keynote).
  - Independent of the technical appendix — should be readable without reading `writeup.md` first.
  - Lead claim: "SAE feature activations as a privacy-preserving monitoring layer below content inspection — composable with managed-access frameworks (Yassif/Carter NTI)."
  - Then briefly cite the format-dependency finding (§4.6 of writeup) as the urgent operational input.
- [ ] **Demo packaging**.
  - Streamlit screenshots into `demo/` (per-prompt audit view, divergence histogram, flag dashboard).
  - `demo/README.md` polish: no-GPU walkthrough using `scripts/synthetic_demo.py`.
  - Optional: short narrated screen-capture script (Caleb to record if time permits).

---

## Open questions / Caleb decisions

1. **Submission cutoff time** — confirm on Apart page; plan currently assumes Sun 2026-04-26 23:59 PT (= Mon 06:59 UTC).
2. **HF push approval** for the held-out calibration set — required before publishing to `SolshineCode/biorefusalaudit-gated`. Otherwise the held-out set stays local.
3. **PR #15 strategy** for the Gemini review — see "PR #15 Gemini review notes" below. Defer merge until tonight's edits land?
4. **Subscribe to PR #15 activity** for the rest of tonight's session?

---

## PR #15 Gemini review notes (2026-04-24)

Gemini left 4 medium-priority comments, all flagging the same surface issue: my new headline + §4.6 use canonical Gemma chat-template tokens (`<start_of_turn>` / `<end_of_turn>`), while existing §4.5 prose (lines 179, 189) and `scripts/format_ablation.py` use ad-hoc shorthand (`<|turn>` / `<turn|>`).

**My read of the truth:**
- Gemma's actual chat-template tokens (what `tokenizer.apply_chat_template()` produces, what `format_ablation.py` condition A invokes) are `<start_of_turn>` and `<end_of_turn>`. These are the canonical special tokens registered in the Gemma tokenizer.
- The `<|turn>` / `<turn|>` form is not a real token — it's an ad-hoc literal-character string used in `format_ablation.py` conditions C and D as **deliberately malformed** test templates, and it has leaked into descriptive prose in `writeup.md` §4.5 lines 179, 189 + the script's docstring header.
- **My new text is correct; the existing §4.5 prose is wrong** in lines 179 and 189. Gemini's suggestion would propagate the existing bug.

**Action:** add `CORRECTED 2026-04-24` blocks to writeup §4.5 lines 179 and 189 + fix the script docstring (a code comment, not under the append-only paper rule). Reply to Gemini's threads explaining the direction.

---

## Things explicitly out of scope tonight

- Domain-specific SAE fine-tuning (`docs/METHOD.md §Planned extension`) — too large for overnight; stays as future-work in §8.
- Expanding the eval set beyond ~25 held-out prompts — quality bio-prompt authoring is slow and the existing 75 are adequate for the flagship pass.
- Track switching — we stay on Track 3, with the Sandbrink/protocol-screening framing carrying the practitioner-relevance argument.
- Recording the demo video — needs Caleb on camera; defer unless time permits.

---

*This plan supersedes any earlier tonight-plan notes. Update inline as items complete (`- [x]`); add new items as discovered. Append-only narrative updates if direction changes.*
