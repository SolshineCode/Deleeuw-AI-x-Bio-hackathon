#!/usr/bin/env python3
"""Replace all em dashes in README.md with context-appropriate alternatives."""
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"

replacements = [
    # Line 14
    ("2026-04-26) — Track 3:", "2026-04-26), Track 3:"),
    # Line 17
    ("](https://arxiv.org/abs/2509.20393) — \"The Secret Agenda",
     "](https://arxiv.org/abs/2509.20393): \"The Secret Agenda"),
    # Line 25 (abstract blockquote)
    ("*refusal depth* — the divergence between", "*refusal depth*, the divergence between"),
    # Line 31
    ("evaluations — VCT, WMDP-Bio, ABC-Bench, ABLE — measure",
     "evaluations (VCT, WMDP-Bio, ABC-Bench, ABLE) measure"),
    # Line 33
    ("use an educational scaffold — and a model", "use an educational scaffold, and a model"),
    # Line 35
    ("multi-turn scaffolding — the exact attack", "multi-turn scaffolding, the exact attack"),
    # Line 37
    ("metric for refusal depth — they can ask", "metric for refusal depth. They can ask"),
    # Lines 47/49/51 (numbered list items)
    ("**Surface classification** — is the model", "**Surface classification**: is the model"),
    ("**Internal feature activation profile** — which", "**Internal feature activation profile**: which"),
    ("**Surface-internal divergence score** — a calibrated", "**Surface-internal divergence score**: a calibrated"),
    # Line 57
    ("SAE feature activation vectors — internal computational artifacts",
     "SAE feature activation vectors, which are internal computational artifacts"),
    # Line 65
    ("behaviors are structurally deep — they can run behavioral red-teaming",
     "behaviors are structurally deep. They can run behavioral red-teaming"),
    # Lines 150-152 (bullet list)
    ("**Private by construction** — feature", "**Private by construction**: feature"),
    ("**Deployable by model providers** — Gemma", "**Deployable by model providers**: Gemma"),
    ("**Actionable for enterprises** — a divergence", "**Actionable for enterprises**: a divergence"),
    # Line 156 (two em dashes)
    ("structurally consistent refusal — internal feature activations align",
     "structurally consistent refusal, where internal feature activations align"),
    ("deeper review — the model is saying no", "deeper review: the model is saying no"),
    # Lines 260-262 (flag bullet list)
    ("hazard_features_active_despite_refusal` — surface refuses",
     "hazard_features_active_despite_refusal`: surface refuses"),
    ("refusal_features_active_despite_compliance` — surface complies",
     "refusal_features_active_despite_compliance`: surface complies"),
    ("deception_correlate_active` — deception-correlated",
     "deception_correlate_active`: deception-correlated"),
    # Line 264
    ("ensemble category-level divergence — more robust",
     "ensemble category-level divergence, more robust"),
    # Line 274 (table row)
    ("**Primary result model** — `gemma-scope-2b-pt-res",
     "**Primary result model**: `gemma-scope-2b-pt-res"),
    # Lines 337-342 (judge consortium numbered list)
    ("**Regex first-pass** — fast", "**Regex first-pass**: fast"),
    ("**Gemini** (via CLI) — primary", "**Gemini** (via CLI): primary"),
    ("**Claude Haiku** (via Claude Code API) — secondary",
     "**Claude Haiku** (via Claude Code API): secondary"),
    ("**Local model** (Llama 3.1 8B Instruct or Qwen 2.5 7B via Ollama) — tertiary",
     "**Local model** (Llama 3.1 8B Instruct or Qwen 2.5 7B via Ollama): tertiary"),
    ("**Aggregation** — weighted", "**Aggregation**: weighted"),
    ("**Human sampling** — 10-15%", "**Human sampling**: 10-15%"),
    # Line 378 (table row)
    ("**Primary result model** — all paper", "**Primary result model**: all paper"),
    # Line 393
    ("**Refusal depth by tier — Gemma 2", "**Refusal depth by tier, Gemma 2"),
    # Line 401
    ("`paper/writeup.md` — original run", "`paper/writeup.md`: original run"),
    # Line 403
    ("73% — consistent", "73%, consistent"),
    # Line 416
    ("activation dumps — not Neuronpedia", "activation dumps, not Neuronpedia"),
    # Line 506
    ("not novel individually — the contribution",
     "not novel individually. The contribution"),
    # Line 508
    ("not descriptive — safety training", "not descriptive: safety training"),
    # Line 512 (two em dashes around parenthetical)
    ("Gemma Scope 2 — announced by DeepMind in Dec 2025 with explicit focus on refusal mechanisms — is the natural substrate",
     "Gemma Scope 2, announced by DeepMind in Dec 2025 with explicit focus on refusal mechanisms, is the natural substrate"),
]

text = README.read_text(encoding="utf-8")
original = text

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"  replaced: {old[:60]!r}")
    else:
        print(f"  NOT FOUND: {old[:60]!r}")

# Catch any remaining em dashes with a generic fallback
remaining = text.count("—")
if remaining:
    print(f"\nWARNING: {remaining} em dash(es) still remaining!")
    for i, line in enumerate(text.splitlines(), 1):
        if "—" in line:
            print(f"  Line {i}: {line[:100]}")
else:
    print(f"\nAll em dashes removed.")

if text != original:
    README.write_text(text, encoding="utf-8")
    print(f"Written: {README}")
else:
    print("No changes made.")
