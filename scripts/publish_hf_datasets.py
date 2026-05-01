#!/usr/bin/env python3
"""Publish biorefusalaudit public and gated datasets to HuggingFace.

Public  (CC-BY-4.0):  Solshine/biorefusalaudit-public
Gated   (HL3):        Solshine/biorefusalaudit-gated

Usage:
  # Initial publish or full refresh:
  python scripts/publish_hf_datasets.py

  # Update gated repo only (e.g. after adding explicit-prompt files):
  python scripts/publish_hf_datasets.py --target gated

  # Update public repo only:
  python scripts/publish_hf_datasets.py --target public

  # Dry-run (stage files but do not upload):
  python scripts/publish_hf_datasets.py --dry-run
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).parent.parent
DATA_PUB = REPO_ROOT / "data" / "eval_set_public"
DATA_GAT = REPO_ROOT / "data" / "eval_set_gated"
DATA_CTL = REPO_ROOT / "data" / "eval_set_control"

PUBLIC_REPO = "Solshine/biorefusalaudit-public"
GATED_REPO  = "Solshine/biorefusalaudit-gated"

STAGE = REPO_ROOT / "hf_stage"
STAGE_PUB = STAGE / "public"
STAGE_GAT = STAGE / "gated"


# -- helpers ------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")

def count_jsonl(path):
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())


# -- split data ---------------------------------------------------------------

def stage_data():
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE_PUB.mkdir(parents=True)
    STAGE_GAT.mkdir(parents=True)

    # main eval set -- split by tier
    main = read_jsonl(DATA_PUB / "eval_set_public_v1.jsonl")
    write_jsonl([r for r in main if r["tier"] != "hazard_adjacent_category"],
                STAGE_PUB / "eval_set_public_v1.jsonl")
    write_jsonl([r for r in main if r["tier"] == "hazard_adjacent_category"],
                STAGE_GAT / "eval_set_tier3_v1.jsonl")

    # calibration holdout v1 -- all public
    write_jsonl(read_jsonl(DATA_PUB / "calibration_holdout_v1.jsonl"),
                STAGE_PUB / "calibration_holdout_v1.jsonl")

    # calibration holdout v2 -- split
    hv2 = read_jsonl(DATA_PUB / "calibration_holdout_v2.jsonl")
    write_jsonl([r for r in hv2 if r["tier"] != "hazard_adjacent_category"],
                STAGE_PUB / "calibration_holdout_v2.jsonl")
    write_jsonl([r for r in hv2 if r["tier"] == "hazard_adjacent_category"],
                STAGE_GAT / "calibration_holdout_v2_tier3.jsonl")

    # calibration holdout v3 -- split
    hv3 = read_jsonl(DATA_PUB / "calibration_holdout_v3.jsonl")
    write_jsonl([r for r in hv3 if r["tier"] != "hazard_adjacent_category"],
                STAGE_PUB / "calibration_holdout_v3.jsonl")
    write_jsonl([r for r in hv3 if r["tier"] == "hazard_adjacent_category"],
                STAGE_GAT / "calibration_holdout_v3_tier3.jsonl")

    # legality / psilocybin controls -- all public
    for fname in ["eval_set_legality_can_psi_v1.jsonl",
                  "eval_set_legality_confound_v1.jsonl",
                  "eval_set_psilocybin_control_v1.jsonl"]:
        write_jsonl(read_jsonl(DATA_PUB / fname), STAGE_PUB / fname)

    # domain control set -- all public
    write_jsonl(read_jsonl(DATA_CTL / "control_legal_financial_v1.jsonl"),
                STAGE_PUB / "control_legal_financial_v1.jsonl")

    # copy schema docs
    shutil.copy(DATA_PUB / "schema.md",              STAGE_PUB / "schema.md")
    shutil.copy(DATA_PUB / "stratification_stats.md", STAGE_PUB / "stratification_stats.md")

    # explicit-prompt files -- all in data/eval_set_gated/ (HL3, have real prompt bodies)
    GATED_EXPLICIT_FILES = [
        "eval_set_tier3_explicit_gemma4_v1.jsonl",
        "eval_set_tier3_explicit_qwen3_v1.jsonl",
        "calibration_holdout_v2_tier3_explicit_gemma4_v1.jsonl",
        "calibration_holdout_v2_tier3_explicit_qwen3_v1.jsonl",
        "calibration_holdout_v3_tier3_explicit_gemma4_v1.jsonl",
        "calibration_holdout_v3_tier3_explicit_qwen3_v1.jsonl",
        "eval_set_tier3_explicit_qwen3_100sample.jsonl",
        "eval_set_benign_explicit_qwen3_100sample.jsonl",
        "eval_set_dualuse_explicit_qwen3_100sample.jsonl",
    ]
    for fname in GATED_EXPLICIT_FILES:
        src = DATA_GAT / fname
        if src.exists():
            shutil.copy(src, STAGE_GAT / fname)
            n = count_jsonl(src)
            print(f"  included gated explicit-prompt: {fname} ({n} rows)")
        else:
            print(f"  skipping (not yet generated): {fname}")


# -- dataset cards (dynamic) --------------------------------------------------

PUBLIC_CARD = """\
---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- biosecurity
- ai-safety
- refusal-evaluation
- sparse-autoencoder
- dual-use
size_categories:
- 1K<n<10K
---

# BioRefusalAudit -- Public Eval Set (Tiers 1 & 2)

Part of the **BioRefusalAudit** project ([GitHub](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)),
submitted to AIxBio Hackathon 2026, Track 3 (Biosecurity Tools).

BioRefusalAudit measures *refusal depth*: the divergence between an LLM's surface refusal
behavior and its internal SAE feature activations on biology-safety prompts.

## What is in this dataset

This public release contains **tiers 1 and 2** of the eval set:

| Tier | Label | Description | n |
|------|-------|-------------|---|
| 1 | `benign_bio` | Innocuous biology questions (textbook material) | 23 |
| 2 | `dual_use_bio` | Legitimate research framings, hazard-adjacent context | 30 |

Plus supplementary sets:
- `calibration_holdout_v1/v2/v3` -- held-out calibration splits (public tiers only)
- `eval_set_legality_confound_v1` -- legality-confound ablation (432 prompts)
- `eval_set_legality_can_psi_v1` -- cannabis/psilocybin legality set (27 prompts)
- `eval_set_psilocybin_control_v1` -- psilocybin domain control (15 prompts)
- `control_legal_financial_v1` -- cross-domain control: legal/financial topics (30 prompts)

All prompts include: `id`, `tier`, `framing` (direct/educational/roleplay/obfuscated),
`prompt`, `rationale`, `expected_surface`, `category`, `sampled_from`.

## Tier 3

Tier 3 (`hazard_adjacent_category`) is released separately under Hippocratic License 3.0
at [`Solshine/biorefusalaudit-gated`](https://huggingface.co/datasets/Solshine/biorefusalaudit-gated).
Access requires attestation of biosecurity/AI safety research purpose.

## License

CC-BY-4.0. Cite as:

```bibtex
@misc{deleeuw2026biorefusalaudit,
  title  = {BioRefusalAudit: Auditing the Depth of LLM Bio-Safety Refusals
            Using Sparse Autoencoder Interpretability},
  author = {DeLeeuw, Caleb},
  year   = {2026},
  url    = {https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon}
}
```
"""


def build_gated_card(stage_gat_path: Path) -> str:
    """Build the gated README dynamically based on which files are present."""

    # Detect which file groups are present
    descriptor_files = [
        ("eval_set_tier3_v1.jsonl",            "Main eval set tier-3 descriptors"),
        ("calibration_holdout_v2_tier3.jsonl", "Calibration holdout v2 tier-3 descriptors"),
        ("calibration_holdout_v3_tier3.jsonl", "Calibration holdout v3 tier-3 descriptors"),
    ]
    explicit_gemma4_files = [
        ("eval_set_tier3_explicit_gemma4_v1.jsonl",               "Main eval set -- Gemma 4 E2B abliterated"),
        ("calibration_holdout_v2_tier3_explicit_gemma4_v1.jsonl", "Calibration holdout v2 -- Gemma 4 E2B abliterated"),
        ("calibration_holdout_v3_tier3_explicit_gemma4_v1.jsonl", "Calibration holdout v3 -- Gemma 4 E2B abliterated"),
    ]
    explicit_qwen3_files = [
        ("eval_set_tier3_explicit_qwen3_v1.jsonl",               "Main eval set -- Qwen 3 4B abliterated"),
        ("calibration_holdout_v2_tier3_explicit_qwen3_v1.jsonl", "Calibration holdout v2 -- Qwen 3 4B abliterated"),
        ("calibration_holdout_v3_tier3_explicit_qwen3_v1.jsonl", "Calibration holdout v3 -- Qwen 3 4B abliterated"),
    ]

    def file_row(fname, desc):
        p = stage_gat_path / fname
        n = count_jsonl(p) if p.exists() else None
        status = str(n) if n is not None else "pending"
        return f"| `{fname}` | {status} | {desc} |"

    has_explicit_gemma4 = any((stage_gat_path / f).exists() for f, _ in explicit_gemma4_files)
    has_explicit_qwen3  = any((stage_gat_path / f).exists() for f, _ in explicit_qwen3_files)
    has_any_explicit    = has_explicit_gemma4 or has_explicit_qwen3

    # Build file table
    desc_rows    = "\n".join(file_row(f, d) for f, d in descriptor_files)
    g4_rows      = "\n".join(file_row(f, d) for f, d in explicit_gemma4_files)
    qwen3_rows   = "\n".join(file_row(f, d) for f, d in explicit_qwen3_files)

    explicit_section = ""
    if has_any_explicit:
        explicit_section = f"""
## Companion dataset: explicit probe questions

In addition to the category-descriptor files, this dataset includes explicit probe questions
generated by abliterated open-weight models. These provide a direct comparison between
abstract (category-label) and explicit framing of the same hazard categories.

Generation methodology: see
[`notes/GENERATION_METHODOLOGY.md`](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/notes/GENERATION_METHODOLOGY.md).

All generated prompts passed a safety filter enforcing the same content policy as the rest
of the dataset: no actionable uplift, no specific pathogen-technique pairs, no synthesis
protocols. Prompts that did not pass after 3 retries are tagged `[NEEDS_REVIEW]` and
excluded from the counts below pending manual review.

Two generation sources are included to control for same-architecture representational confounds:

| Source | Model | Architecture |
|--------|-------|-------------|
| `gemma4-e2b-abliterated` | `huihui-ai/Huihui-gemma-4-E2B-it-abliterated` | Gemma 4 E2B |
| `qwen3-4b-abliterated` | `huihui_ai/qwen3-abliterated:4b` | Qwen 3 4B |

### Explicit-prompt files -- Gemma 4 E2B abliterated

| File | n | Description |
|------|---|-------------|
{g4_rows}

### Explicit-prompt files -- Qwen 3 4B abliterated

| File | n | Description |
|------|---|-------------|
{qwen3_rows}
"""
    else:
        explicit_section = """
## Companion dataset: explicit probe questions (in progress)

Explicit hazard-adjacent probe questions -- one per category descriptor, generated by
abliterated open-weight models -- are being added to this dataset progressively as
generation and safety review completes. See
[`notes/GENERATION_METHODOLOGY.md`](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/notes/GENERATION_METHODOLOGY.md)
for the generation design and safety filtering methodology.
"""

    card = f"""\
---
license: other
license_name: hippocratic-license-3.0
license_link: https://firstdonoharm.dev/version/3/0/bds-cl-eco-extr-ffd-media-mil-my-sup-sv-tal-usta-xuar.html
task_categories:
- text-classification
language:
- en
tags:
- biosecurity
- ai-safety
- refusal-evaluation
- dual-use
- gated
size_categories:
- n<1K
---

# BioRefusalAudit -- Tier 3 Eval Set (Gated, HL3)

Part of the **BioRefusalAudit** project ([GitHub](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon)),
submitted to AIxBio Hackathon 2026, Track 3 (Biosecurity Tools).

## Access

This dataset is gated. To request access, attest that:
1. You are a biosecurity or AI safety researcher using the data for **defensive** purposes.
2. You will not reproduce specific prompt bodies publicly or in model training data without
   equivalent safety controls.
3. You agree to the Hippocratic License 3.0 terms linked above.

Request access via the HuggingFace access form, or email the maintainer via
[SAFETY.md](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/SAFETY.md).

## Category-descriptor files

**Tier 3** (`hazard_adjacent_category`): category-level descriptor prompts across
the main eval set and calibration holdouts. Each entry identifies a hazard-adjacent
biological research category; the descriptor string itself is the probe.

This approach -- using category taxonomy labels as probes rather than explicit
hazard-adjacent questions -- was chosen because frontier LLM providers declined to
generate explicit prompts in a biosecurity research context without direct jailbreaking.
Rather than jailbreak production models, the dataset uses category descriptors that
themselves elicit measurable hazard-feature activation and surface-level hedging.
This is a finding in its own right: the category label alone is sufficient to activate
hazard-adjacent SAE features. The hackathon results are reported for this format.

| File | n | Description |
|------|---|-------------|
{desc_rows}
{explicit_section}
## Schema

Category-descriptor files: `id`, `tier`, `framing`, `prompt`, `rationale`,
`expected_surface`, `category`, `sampled_from`.

Explicit-prompt files: all of the above, plus `descriptor_original`,
`generation_model`, `generation_source_label`, `needs_review`.

## License

[Hippocratic License 3.0](https://firstdonoharm.dev/version/3/0/bds-cl-eco-extr-ffd-media-mil-my-sup-sv-tal-usta-xuar.html)
(HL3-BDS-CL-ECO-EXTR-FFD-MEDIA-MIL-MY-SUP-SV-TAL-USTA-XUAR).
Use for defensive biosecurity and AI safety research only.

## Citation

```bibtex
@misc{{deleeuw2026biorefusalaudit,
  title  = {{BioRefusalAudit: Auditing the Depth of LLM Bio-Safety Refusals
            Using Sparse Autoencoder Interpretability}},
  author = {{DeLeeuw, Caleb}},
  year   = {{2026}},
  url    = {{https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon}}
}}
```
"""
    return card


# -- upload -------------------------------------------------------------------

def publish(target: str = "both", dry_run: bool = False):
    api = HfApi()

    print("\n-- Staging data --")
    stage_data()

    # Write dataset cards
    (STAGE_PUB / "README.md").write_text(PUBLIC_CARD, encoding="utf-8")
    gated_card = build_gated_card(STAGE_GAT)
    (STAGE_GAT / "README.md").write_text(gated_card, encoding="utf-8")
    print("  wrote README.md for both repos")

    if dry_run:
        print("\n-- DRY RUN: staged but not uploaded --")
        print(f"  public stage: {STAGE_PUB}")
        print(f"  gated stage:  {STAGE_GAT}")
        print("\n-- Gated README preview --")
        print(gated_card[:2000])
        return

    if target in ("both", "public"):
        print(f"\n-- Publishing {PUBLIC_REPO} --")
        create_repo(PUBLIC_REPO, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(STAGE_PUB),
            repo_id=PUBLIC_REPO,
            repo_type="dataset",
            commit_message="BioRefusalAudit public eval set (tiers 1+2)",
        )
        print(f"  https://huggingface.co/datasets/{PUBLIC_REPO}")

    if target in ("both", "gated"):
        print(f"\n-- Publishing {GATED_REPO} --")
        create_repo(GATED_REPO, repo_type="dataset", exist_ok=True, private=False)
        api.update_repo_settings(GATED_REPO, repo_type="dataset", gated="auto")
        api.upload_folder(
            folder_path=str(STAGE_GAT),
            repo_id=GATED_REPO,
            repo_type="dataset",
            commit_message="BioRefusalAudit tier-3 gated eval set (HL3)",
        )
        print(f"  https://huggingface.co/datasets/{GATED_REPO}")

    print("\n-- Done. Cleaning up stage dir. --")
    shutil.rmtree(STAGE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["both", "public", "gated"], default="both",
                        help="Which repo(s) to publish (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage files and preview README without uploading")
    args = parser.parse_args()
    publish(target=args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
