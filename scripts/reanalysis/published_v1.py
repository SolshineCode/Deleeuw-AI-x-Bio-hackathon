"""Behavioural-rate numbers printed in arXiv:2605.30162v1, as data (added 2026-10-23).

The re-analysis compares each of these cells against the relabelled surface labels. Each
entry names the committed run the number came from and the records it covers. Before a
cell is compared, `reanalyze.py` recomputes it from the ORIGINAL labels in that run and
reports whether it reproduces; a cell that does not reproduce is flagged, not dropped.

Fields:
  claim      short id used in the diff table (paper section + cell)
  ref        where in paper/submission_final.tex the number is printed
  run        results/<run>/report.json the number was computed from
  tier       record tier filter; "tiers_1_2" = every record that is not tier 3
  category   optional category filter (public eval-set `category` field, by record id)
  label      surface class the percentage counts
  published  the printed percentage
  note       caveats on the comparison

Tier-3 (hazard_adjacent_category) cells are listed so the diff table is complete, but they
cannot be checked: every committed tier-3 completion is redacted (change list R7).
No prompt or completion text is used here; only run names, tiers, categories and labels.
"""

from __future__ import annotations

T1 = "gemma-2-2b-it-L12-tuned-rejudged"
LEG80 = "legality-gemma2-80tok"

PUBLISHED: list[dict] = [
    # Table 1 / Finding 1 (Gemma 2 2B-IT, 75 prompts)
    *[{"claim": f"T1 {t} {lab}", "ref": "Table 1, tex:506-521", "run": T1, "tier": t,
       "label": lab, "published": p,
       **({"note": "v1 'hedge' on tiers 1-2 was the regex no-match fallback; compare with the relabel's partial too (section 4)"}
          if lab == "hedge" and t != "hazard_adjacent_category" else {})}
      for t, lab, p in (("benign_bio", "comply", 70), ("benign_bio", "hedge", 30), ("benign_bio", "refuse", 0),
                        ("dual_use_bio", "comply", 80), ("dual_use_bio", "hedge", 20), ("dual_use_bio", "refuse", 0),
                        ("hazard_adjacent_category", "hedge", 100), ("hazard_adjacent_category", "refuse", 0))],
    # Table 2 / Finding 6, explicit-prompt rows
    {"claim": "T2 explicit benign comply", "ref": "Table 2, tex:748-764", "run": "gemma-2-2b-it-explicit-benign-qwen3-100",
     "tier": "benign_bio", "label": "comply", "published": 84},
    {"claim": "T2 explicit benign refuse", "ref": "Table 2, tex:748-764", "run": "gemma-2-2b-it-explicit-benign-qwen3-100",
     "tier": "benign_bio", "label": "refuse", "published": 16},
    {"claim": "T2 explicit dual-use comply", "ref": "Table 2, tex:748-764", "run": "gemma-2-2b-it-explicit-dualuse-qwen3-100",
     "tier": "dual_use_bio", "label": "comply", "published": 76},
    {"claim": "T2 explicit dual-use refuse", "ref": "Table 2, tex:748-764", "run": "gemma-2-2b-it-explicit-dualuse-qwen3-100",
     "tier": "dual_use_bio", "label": "refuse", "published": 24},
    {"claim": "T2 explicit tier-3 refuse", "ref": "Table 2 / tex:766-774", "run": "gemma-2-2b-it-explicit-tier3-qwen3-100",
     "tier": "hazard_adjacent_category", "label": "refuse", "published": 67},
    # Finding 2 / Finding 4 table, Gemma 4 row (published over all 75 prompts incl. tier 3)
    {"claim": "F2 Gemma 4 refuse (template)", "ref": "Finding 2 + F4 table, tex:542-552, 620-640",
     "run": "gemma-4-E2B-it-L17-pass5", "tier": "tiers_1_2", "label": "refuse", "published": 87,
     "note": "published 65/75 covers all tiers; only the tier-1/2 subset can be checked"},
    # Table 3 / Finding 7 (published over all 75 prompts)
    {"claim": "T3 Gemma 4 our-SAE comply", "ref": "Table 3, tex:800-817", "run": "gemma-4-E2B-it-our-sae-v1-150tok",
     "tier": "tiers_1_2", "label": "comply", "published": 79,
     "note": "published 59/75 covers all tiers; only the tier-1/2 subset can be checked"},
    {"claim": "T3 Gemma 4 our-SAE refuse", "ref": "Table 3, tex:800-817", "run": "gemma-4-E2B-it-our-sae-v1-150tok",
     "tier": "tiers_1_2", "label": "refuse", "published": 21,
     "note": "published 16/75 covers all tiers; only the tier-1/2 subset can be checked"},
    # Finding 4 table (benign refuse %, hazard refuse %)
    {"claim": "F4 Gemma 2 benign refuse", "ref": "F4 table, tex:620-640", "run": T1, "tier": "benign_bio",
     "label": "refuse", "published": 0},
    *[{"claim": f"F4 {name} {short} refuse", "ref": "F4 table, tex:620-640", "run": run, "tier": t,
       "label": "refuse", "published": p}
      for name, run, b, h in (("Llama 3.2 1B", "llama-3.2-1b-it-cross-arch", 30, 91),
                              ("Qwen 2.5 1.5B", "qwen25-1b5-it-cross-arch", 83, 95),
                              ("Phi-3-mini", "phi3-mini-4k-it-cross-arch", 87, 95))
      for short, t, p in (("benign", "benign_bio", b), ("hazard", "hazard_adjacent_category", h))],
    # Finding 5, psilocybin table (refuse % by angle)
    *[{"claim": f"F5 {name} {ang} refuse", "ref": "F5 table 1, tex:674-690", "run": run, "tier": "psilocybin_control",
       "category": f"psilocybin_{ang}", "label": "refuse", "published": p}
      for name, run, ph, cu in (("Gemma 2 80-tok", "psilocybin-gemma2-sae-80tok", 0, 25),
                                ("Gemma 2 150-tok", "psilocybin-gemma2-sae-150tok", 25, 50),
                                ("Qwen 2.5 1.5B", "psilocybin-qwen25-1b5-80tok", 75, 50),
                                ("Llama 3.2 1B", "psilocybin-llama32-1b-80tok", 0, 0))
      for ang, p in (("pharmacology", ph), ("cultivation", cu))],
    # Finding 5, cross-compound table (Gemma 2, 80 tokens)
    *[{"claim": f"F5 cross-compound {cat} refuse", "ref": "F5 table 2, tex:692-703", "run": LEG80,
       "tier": "legality_confound", "category": cat, "label": "refuse", "published": p,
       **({"note": "paper's 'cultivation' column for mescaline maps to category mescaline_cacti"}
          if cat == "mescaline_cacti" else {})}
      for cat, p in (("psilocybin_cultivation", 33), ("psilocybin_pharmacology", 0),
                     ("cannabis_cultivation", 0), ("cannabis_pharmacology", 0),
                     ("lsd_pharmacology", 25), ("mescaline_cacti", 0), ("mescaline_pharmacology", 25))],
]

# Category lookup sources (public eval sets; tier 1-2 / control material only).
CATEGORY_FILES = (
    "data/eval_set_public/eval_set_psilocybin_control_v1.jsonl",
    "data/eval_set_public/eval_set_legality_confound_v1.jsonl",
    "data/eval_set_public/eval_set_legality_can_psi_v1.jsonl",
)
