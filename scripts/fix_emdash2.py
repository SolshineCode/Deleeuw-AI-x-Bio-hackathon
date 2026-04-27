#!/usr/bin/env python3
"""Second pass: remaining em dashes in README.md."""
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"

replacements = [
    # Line 25 (abstract)
    ("75 prompts — only universal hedging", "75 prompts, only universal hedging"),
    # Line 33
    ("A model with a deep refusal — where internal representations do not activate on hazard features even when probed — is structurally different.",
     "A model with a deep refusal, where internal representations do not activate on hazard features even when probed, is structurally different."),
    # Line 57
    ("without reading interaction content — no content disclosure, no proprietary data exposure, no IP risk.",
     "without reading interaction content: no content disclosure, no proprietary data exposure, no IP risk."),
    # Line 65
    ("on a refusal warrants review — it is saying no while internally representing",
     "on a refusal warrants review: it is saying no while internally representing"),
    # Line 67
    ("HL3 does not — it binds downstream users to enforceable human rights obligations",
     "HL3 does not. It binds downstream users to enforceable human rights obligations"),
    # Line 132
    ("tier hedges universally — the model knows something is sensitive but won't refuse outright.",
     "tier hedges universally: the model knows something is sensitive but won't refuse outright."),
    # Line 134 (two em dashes)
    ("framing-distribution-sensitive — see §4.2 of the paper for the full caveat",
     "framing-distribution-sensitive (see §4.2 of the paper for the full caveat)"),
    ("The flag-based findings (§4.6) — zero genuine refusals despite hazard-tier hedging, and safety-circuit activation without behavioral suppression in Gemma 4 — survive that caveat",
     "The flag-based findings (§4.6), including zero genuine refusals despite hazard-tier hedging and safety-circuit activation without behavioral suppression in Gemma 4, survive that caveat"),
    # Line 144 (section header)
    ("## Policy motivation — why refusal depth matters to deployers",
     "## Policy motivation: why refusal depth matters to deployers"),
    # Line 146
    ("dangerous bio research activity* — while **not breaching user privacy, not disclosing company proprietary information, and not triggering IP concerns**.",
     "dangerous bio research activity*, without breaching user privacy, disclosing company proprietary information, or triggering IP concerns."),
    # Line 148
    ("**internal model representations via SAE feature activations** — a layer of information that is:",
     "**internal model representations via SAE feature activations**, a layer of information that is:"),
    # Line 280
    ("`Solshine/gemma4-e2b-bio-sae-v1` — a 2000-step mean-contrastive SAE",
     "`Solshine/gemma4-e2b-bio-sae-v1`, a 2000-step mean-contrastive SAE"),
    # Line 287 (inside code block comment)
    ("# Gemma Scope 2 (Gemma 2 2B, layer 12, width 16k) — primary eval path",
     "# Gemma Scope 2 (Gemma 2 2B, layer 12, width 16k): primary eval path"),
    # Line 362
    ("framing-sensitive — a model whose refusal depth varies significantly by framing is less reliably safe.",
     "framing-sensitive: a model whose refusal depth varies significantly by framing is less reliably safe."),
    # Line 385
    ("and behavioral surface labels only — useful for cross-architecture",
     "and behavioral surface labels only, useful for cross-architecture"),
    # Line 516 (two em dashes around parenthetical)
    ("motivated by the observation — consistent with the GCG literature and with Arditi et al. — that refusal circuitry",
     "motivated by the observation, consistent with the GCG literature and with Arditi et al., that refusal circuitry"),
    # Line 527
    ("CC-BY-4.0 — fully open.",
     "CC-BY-4.0, fully open."),
    # Line 531 (two em dashes around list)
    ("aligns with how every organization involved in this work — SecureBio, Fourth Eon Bio, Sentinel Bio, CLTR — actually thinks",
     "aligns with how every organization involved in this work, including SecureBio, Fourth Eon Bio, Sentinel Bio, and CLTR, actually thinks"),
    # Line 697
    ("SAE ecosystem — particularly the Llama Scope authors — for making",
     "SAE ecosystem, particularly the Llama Scope authors, for making"),
]

text = README.read_text(encoding="utf-8")
original = text

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"  replaced: {old[:70]!r}")
    else:
        print(f"  NOT FOUND: {old[:70]!r}")

remaining = text.count("—")
if remaining:
    print(f"\nWARNING: {remaining} em dash(es) still remaining!")
    for i, line in enumerate(text.splitlines(), 1):
        if "—" in line:
            print(f"  Line {i}: {line[:120]}")
else:
    print(f"\nAll em dashes removed.")

if text != original:
    README.write_text(text, encoding="utf-8")
    print(f"Written: {README}")
else:
    print("No changes made.")
