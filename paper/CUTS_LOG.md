# Paper Cuts Log

Content removed from `paper/writeup.md` during word-count trimming. Each entry records what was lost (not just rephrasing), the commit that removed it, and where the full content lives if documented elsewhere.

---

## Cut 1 — Gemma 4 pass 3 degenerate-T results table + calibration bootstrap detail
**Removed in:** `ab912b9 → 49a7483` (human-writing-check pass 2 + trim to 3388 words)  
**Original section:** §4.5 Portability path (Gemma 4 E2B)  
**What was lost:**

Full numerical results from the degenerate-T pass (pass 3, before T was re-fitted from non-zero feature vecs):

| Tier | n | Mean D (pass 3) | Std |
|---|---:|---:|---:|
| benign_bio | 23 | 0.061 | 0.018 |
| dual_use_bio | 30 | 0.056 | 0.015 |
| hazard_adjacent_category | 22 | 0.053 | 0.004 |

Also cut: the mean SAE component vector for pass 3 (`[0.424, 0.241, 0.335, 0, 0]` for bio_content / hazard_adjacent / refusal_circuitry), T re-fit condition number (151, well-conditioned), and the full 6-pass chain description with per-pass notes.

**Why it matters:** The flat 0.061/0.056/0.053 gradient is the clearest empirical demonstration of the calibration bootstrap problem (D≈0 when T is fit on zero-feature data). It's diagnostic evidence, not noise.

**Lives at:** `docs/METHOD.md §Known failure modes — Calibration bootstrap problem`; run artifacts at `runs/gemma-4-E2B-it-L17-pass3/`.

---

## Cut 2 — Protocol/cloud-lab screening use case paragraph
**Removed in:** `ab912b9 → 49a7483`  
**Original section:** §1 Policy motivation  
**What was lost:**

> "The same activation-signature approach extends naturally to cloud-lab and experimental-protocol screening contexts: when a researcher queries an LLM to help design an experimental workflow, BioRefusalAudit's divergence flags can surface cases where the model's internal hazard features fire even though the generated protocol text reads as benign — dual-use concern masked by innocuous framing. Model-internal auditing composes with output-based screeners (sequence filters, protocol classifiers) that inspect generated artifacts; the two signals cover distinct failure modes. Our contribution is the upstream-of-output layer — what the model is representing while it writes — which remains invisible to any screener that sees only the completed artifact."

**Why it matters:** This is the clearest statement of how BioRefusalAudit composes with (rather than replaces) existing output-only screeners. It's a direct answer to "why not just use a content filter?" and names a specific deployment context (cloud lab protocol assistance) that reviewers from biosecurity policy backgrounds would recognize.

**Lives at:** `paper/policy_brief.md` partially; not fully preserved elsewhere.

---

## Cut 3 — §7 Why-HL3 extended rationale (~160 words trimmed to ~80)
**Removed in:** `15bbc97 → b4d99b4`  
**Original section:** §7 Responsible release  
**What was lost:**

The full version connected HL3 to three specific keynote speakers (Sandbrink, Crook, Yassif/NTI) by name and framed the tool as implementing *both* the measurement layer *and* enforcement layer of Yassif & Carter (NTI Bio, January 2026). Also included the norm-setting argument explicitly:

> "The author has released unrelated work under permissive licenses elsewhere; HL3 was chosen here specifically because this artifact is itself a biosecurity instrument. As interpretability-based biosecurity tools proliferate, the default release posture for this artifact class will be set by early contributors. Choosing HL3 now is an argument that biosecurity AI tooling belongs in a different licensing category than general-purpose AI infrastructure."

**Why it matters:** The norm-setting argument is the strongest *policy* justification for HL3 (not just the legal one). It names the class of artifact, not just this instance.

**Lives at:** `docs/HL3_RATIONALE.md` (full version).

---

## Cut 4 — Synthetic validation table + plumbing-check narrative
**Removed in:** `ab912b9 → 49a7483`  
**Original section:** §4.1 (before flagship run)  
**What was lost:**

Synthetic per-tier D values from `scripts/synthetic_demo.py`:

| Tier | Mean D (synthetic) |
|---|---:|
| benign_bio | 0.02 |
| dual_use_bio | 0.11 |
| hazard_adjacent_category | 0.47 |

Plus the 5-step pipeline validation narrative (JSONL loading, divergence formula, flag-firing, cross-model aggregation, matplotlib plot) and the explicit "these numbers are synthetic and serve only as a plumbing check" disclaimer.

**Why it matters:** The table proved the pipeline produces the expected tier shape before any real model run — that's reproducibility evidence. Also, reviewers looking for "did you verify the code before claiming results" have no equivalent reassurance in the current paper.

**Lives at:** `runs/synthetic_*/report.json`; `scripts/synthetic_demo.py` still runnable.

---

## Cut 5 — CMF gate explicit "loose gate" caveat + dose-proportionality detail
**Removed in:** `15bbc97 → b4d99b4` (partial); `68fbb49` (final 3 words)  
**Original section:** §4.3  
**What was lost (composite of two trims):**

Full original CMF paragraph:

> "A feature cluster qualifies as a candidate mechanistic feature (CMF) if `label_changed` OR `|ΔD| > 0.2` on ablation. Note: this is a loose gate. Label changes may reflect surface noise rather than deep mechanism. Full 'named circuit' status per mechanistic interpretability standards would require paraphrase consistency testing and dose-proportionality **across boost multipliers**. We report CMF status as a weak causal signal warranting further investigation, not as a validated circuit claim."

The trimmed paper version: "Full validation requires paraphrase consistency testing and dose-proportionality."

**What's substantively missing:** "across boost multipliers" specifies *what* the dose-proportionality test means operationally (test multiple ablation/boost magnitudes, not just the single 3× we ran). "Loose gate" + "label changes may reflect surface noise" was the honest qualifier on the criterion's sensitivity.

**Lives at:** `docs/METHOD.md §CMF qualification criterion`.

---

## Cut 6 — §1 Sandbrink privacy-compatible activation auditing paragraph
**Removed in:** `15bbc97 → b4d99b4`  
**Original section:** §1 Policy motivation  
**What was lost:**

> "BioRefusalAudit addresses this directly: auditing SAE feature activations rather than interaction content lets a deployer flag structurally shallow refusals without reading the prompt or completion text. The activation signal is separable from linguistic content, making this privacy-compatible where content-inspection approaches are not."

**Why it matters:** This is the cleanest statement of the privacy-compatibility argument — that activation auditing doesn't require reading user content. It answers the "what's the deployment model?" question that policy audiences care about.

**Lives at:** `paper/policy_brief.md` partially; `README.md` §"Why activation-level".

---

## Cut 7 — Intervention table condensed from tier-explicit to tier-merged
**Removed in:** `b4d99b4` (table expansion commit)  
**Original section:** §4.3  
**What was lost:**

The old table had `Category` column (`refusal_circuitry`) instead of `Tier` column. Original 7-row table used `Category` as the second column (which was always `refusal_circuitry` since that was the only category intervened on). The replacement 12-row table correctly uses `Tier` (benign_bio / dual_use_bio / hazard_adjacent). No content lost — this was a structural improvement, not a cut.

**Status:** Not a content loss. Logged for completeness.
