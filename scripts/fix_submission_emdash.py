#!/usr/bin/env python3
"""Fix all em dashes in paper/submission.md."""
from pathlib import Path

f = Path(__file__).parent.parent / "paper" / "submission.md"

replacements = [
    # Line 10 abstract (two)
    ("*refusal depth* — the divergence between",
     "*refusal depth*, the divergence between"),
    ("75 prompts — only universal hedging",
     "75 prompts, only universal hedging"),
    # Line 95
    ("consistent with the refusal direction — the model may not be strongly activating",
     "consistent with the refusal direction; the model may not be strongly activating"),
    # Line 116 (two)
    ("Cannabis is also federally Schedule I — illegal since 1970 — yet cannabis cultivation gets",
     "Cannabis is also federally Schedule I (illegal since 1970), yet cannabis cultivation gets"),
    # Line 122 table cell N/A
    ("| LSD | Schedule I | — | **25%** |",
     "| LSD | Schedule I | N/A | **25%** |"),
    # Line 160 conclusion (two)
    ("in the training distribution — state-level legality, commercial availability, cultural salience — rather than CBRN hazard risk",
     "in the training distribution, tracking state-level legality, commercial availability, and cultural salience, rather than CBRN hazard risk"),
    # Line 168 link text
    ("Hippocratic License 3.0 — HL3-BDS-CL-ECO-EXTR-FFD-MEDIA-MIL-MY-SUP-SV-TAL-USTA-XUAR",
     "Hippocratic License 3.0 (HL3-BDS-CL-ECO-EXTR-FFD-MEDIA-MIL-MY-SUP-SV-TAL-USTA-XUAR)"),
    # Line 170 link text
    ("Live dashboard — all 75 prompts, feature activations, intervention results",
     "Live dashboard: all 75 prompts, feature activations, intervention results"),
    # Line 246
    ("prompt optimization signal — iterating phrasings",
     "prompt optimization signal, iterating phrasings"),
    # Line 250
    ("RSP evaluators — not adversaries.",
     "RSP evaluators, not adversaries."),
    # Line 256
    ("identifies a novel exploitable mechanism — for example, a specific SAE feature direction",
     "identifies a novel exploitable mechanism, for example a specific SAE feature direction"),
]

text = f.read_text(encoding="utf-8")
original = text

for old, new in replacements:
    if old in text:
        text = text.replace(old, new)
        print(f"  OK: {old[:60]!r}")
    else:
        print(f"  MISS: {old[:60]!r}")

remaining = [(i+1, line) for i, line in enumerate(text.splitlines()) if "—" in line]
if remaining:
    print(f"\nWARNING: {len(remaining)} line(s) still have em dashes:")
    for n, line in remaining:
        print(f"  L{n}: {line[:100]}")
else:
    print("\nAll em dashes removed.")

if text != original:
    f.write_text(text, encoding="utf-8")
    print(f"Written: {f}")
