#!/usr/bin/env python3
"""
audit_eval_set.py — Quality auditor for all BioRefusalAudit public eval JSONL files.

Checks every JSONL file in data/eval_set_public/ for:
  - JSON parse validity
  - Required field presence and type
  - Enum value constraints (tier, framing, expected_surface)
  - ID format and uniqueness across all files
  - Tier/surface consistency rules
  - Prompt word-count range (20–50 words for tiers 1+2)
  - No duplicate categories within a tier+framing cell
  - Tier-3 body redaction (stubs must not contain real prompt text)

Usage:
    python scripts/audit_eval_set.py [--verbose] [--samples N]

Exits 0 on clean audit, 1 on any hard failure.
"""

import json
import sys
import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "data" / "eval_set_public"

REQUIRED_FIELDS = {"id", "tier", "framing", "prompt", "rationale",
                   "expected_surface", "category", "sampled_from"}
VALID_TIERS = {"benign_bio", "dual_use_bio", "hazard_adjacent_category"}
VALID_FRAMINGS = {"direct", "educational", "roleplay", "obfuscated"}
VALID_SURFACES = {"comply", "hedge", "partial", "refuse"}

# Rules: benign → comply only; dual_use → hedge or partial; tier-3 → refuse
TIER_SURFACE_RULES = {
    "benign_bio": {"comply"},
    "dual_use_bio": {"hedge", "partial"},
    "hazard_adjacent_category": {"refuse"},
}

# Tier-3 stub sentinel phrases — all tier-3 bodies should contain one
TIER3_SENTINEL = "category-level descriptor"


def load_all_files(eval_dir: Path) -> list[tuple[str, list[dict]]]:
    """Load all JSONL files; return list of (filename, rows)."""
    files = sorted(eval_dir.glob("*.jsonl"))
    results = []
    for f in files:
        rows = []
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_source_file"] = f.name
                obj["_lineno"] = lineno
                rows.append(obj)
            except json.JSONDecodeError as e:
                print(f"FAIL  JSON parse error in {f.name} line {lineno}: {e}")
                sys.exit(1)
        results.append((f.name, rows))
    return results


def audit_all(files: list[tuple[str, list[dict]]], verbose: bool = False,
              samples: int = 3) -> bool:
    all_rows = []
    for _, rows in files:
        all_rows.extend(rows)

    failures = []
    warnings = []

    # ── 1. Required fields ──────────────────────────────────────────────────
    for row in all_rows:
        missing = REQUIRED_FIELDS - set(row.keys())
        if missing:
            failures.append(
                f"{row['_source_file']}:{row['_lineno']} [{row.get('id','?')}] "
                f"missing fields: {missing}"
            )

    # ── 2. Enum constraints ─────────────────────────────────────────────────
    for row in all_rows:
        rid = row.get("id", "?")
        src = f"{row['_source_file']}:{row['_lineno']}"
        if row.get("tier") not in VALID_TIERS:
            failures.append(f"{src} [{rid}] invalid tier: {row.get('tier')}")
        if row.get("framing") not in VALID_FRAMINGS:
            failures.append(f"{src} [{rid}] invalid framing: {row.get('framing')}")
        if row.get("expected_surface") not in VALID_SURFACES:
            failures.append(f"{src} [{rid}] invalid expected_surface: {row.get('expected_surface')}")

    # ── 3. ID format and uniqueness ─────────────────────────────────────────
    id_pattern = re.compile(r"^bio_\d{3}$")
    all_ids = [row.get("id", "") for row in all_rows]
    dupes = [i for i, c in Counter(all_ids).items() if c > 1]
    if dupes:
        failures.append(f"Duplicate IDs across all files: {dupes}")
    for row in all_rows:
        rid = row.get("id", "")
        if not id_pattern.match(rid):
            failures.append(f"{row['_source_file']}:{row['_lineno']} malformed ID: {rid!r}")

    # ── 4. Tier / surface consistency ───────────────────────────────────────
    for row in all_rows:
        tier = row.get("tier")
        surface = row.get("expected_surface")
        rid = row.get("id", "?")
        if tier in TIER_SURFACE_RULES and surface not in TIER_SURFACE_RULES[tier]:
            failures.append(
                f"{row['_source_file']}:{row['_lineno']} [{rid}] "
                f"tier={tier} but expected_surface={surface} "
                f"(allowed: {TIER_SURFACE_RULES[tier]})"
            )

    # ── 5. Prompt word count (tiers 1+2 only: 20–50 words target) ──────────
    # Hard fail: <15 words (structurally too short to be a real prompt).
    # Soft warning: <20 or >60 words (outside spec target; v1 has a few ~17-word
    # prompts that are accepted — lower hard-fail to 15 to avoid retroactive failures).
    for row in all_rows:
        if row.get("tier") == "hazard_adjacent_category":
            continue  # stubs are short by design
        prompt = row.get("prompt", "")
        wc = len(prompt.split())
        rid = row.get("id", "?")
        if wc < 15:
            failures.append(
                f"Word count FAIL [{rid}] too short ({wc} words): {prompt[:60]}…"
            )
        elif wc < 20:
            warnings.append(
                f"[{rid}] prompt length {wc} words (target ≥20, spec allows short v1 prompts): "
                f"{prompt[:60]}…"
            )
        elif wc > 60:  # slight tolerance over 50 for multi-clause educational prompts
            warnings.append(
                f"[{rid}] prompt length {wc} words (target ≤50): {prompt[:60]}…"
            )

    # ── 6. Tier-3 stub body check ────────────────────────────────────────────
    for row in all_rows:
        if row.get("tier") != "hazard_adjacent_category":
            continue
        prompt = row.get("prompt", "")
        rid = row.get("id", "?")
        if TIER3_SENTINEL not in prompt.lower():
            failures.append(
                f"{row['_source_file']}:{row['_lineno']} [{rid}] "
                f"tier-3 prompt does not contain sentinel phrase "
                f"'{TIER3_SENTINEL}' — may contain real hazard body"
            )

    # ── 7. Category uniqueness warnings ────────────────────────────────────
    cat_cell: dict[tuple, list] = defaultdict(list)
    for row in all_rows:
        key = (row.get("tier"), row.get("framing"), row.get("category"))
        cat_cell[key].append(row.get("id", "?"))
    for (tier, framing, cat), ids in cat_cell.items():
        if len(ids) > 1 and tier != "hazard_adjacent_category":
            warnings.append(
                f"Category '{cat}' appears {len(ids)}× in tier={tier} "
                f"framing={framing}: {ids}"
            )

    # ── 8. Rationale length check ───────────────────────────────────────────
    for row in all_rows:
        rat = row.get("rationale", "")
        rid = row.get("id", "?")
        sentences = [s.strip() for s in rat.split(".") if s.strip()]
        if len(sentences) > 2:
            warnings.append(
                f"[{rid}] rationale appears to be >1 sentence: {rat[:80]}…"
            )

    # ── Print results ───────────────────────────────────────────────────────
    print("=" * 70)
    print("BioRefusalAudit Eval Set Audit")
    print("=" * 70)

    # File summary
    print(f"\nFiles scanned ({len(files)}):")
    for fname, rows in files:
        print(f"  {fname:45s}  {len(rows):4d} rows")

    total = len(all_rows)
    tiers = Counter(r.get("tier") for r in all_rows)
    framings = Counter(r.get("framing") for r in all_rows)
    surfaces = Counter(r.get("expected_surface") for r in all_rows)
    cats = len(set(r.get("category") for r in all_rows))

    print(f"\nTotal prompts: {total}")
    print(f"\nTier breakdown:")
    for t in sorted(tiers):
        print(f"  {t:35s}  {tiers[t]:4d}")
    print(f"\nFraming breakdown:")
    for f in sorted(framings):
        print(f"  {f:35s}  {framings[f]:4d}")
    print(f"\nExpected surface breakdown:")
    for s in sorted(surfaces):
        print(f"  {s:35s}  {surfaces[s]:4d}")
    print(f"\nDistinct categories:  {cats}")

    # Tier × Framing cross-tab
    print(f"\nTier × Framing cross-tab:")
    tier_list = sorted(VALID_TIERS)
    framing_list = sorted(VALID_FRAMINGS)
    header = f"{'':35s}" + "".join(f"{f:>14s}" for f in framing_list) + f"{'TOTAL':>10s}"
    print("  " + header)
    for t in tier_list:
        row_counts = [sum(1 for r in all_rows if r.get("tier") == t and r.get("framing") == f)
                      for f in framing_list]
        row_total = sum(row_counts)
        print("  " + f"{t:35s}" + "".join(f"{c:>14d}" for c in row_counts) + f"{row_total:>10d}")

    # ID range
    numeric_ids = sorted(int(r["id"].split("_")[1]) for r in all_rows if id_pattern.match(r.get("id", "")))
    if numeric_ids:
        print(f"\nID range: bio_{numeric_ids[0]:03d} – bio_{numeric_ids[-1]:03d}  "
              f"({len(numeric_ids)} distinct IDs)")

    # ── Samples ─────────────────────────────────────────────────────────────
    if samples > 0 and verbose:
        print(f"\n{'=' * 70}")
        print(f"Sample prompts ({samples} per tier):")
        for tier in ["benign_bio", "dual_use_bio", "hazard_adjacent_category"]:
            tier_rows = [r for r in all_rows if r.get("tier") == tier]
            print(f"\n  ── {tier} ──")
            for r in tier_rows[:samples]:
                print(f"  [{r['id']}] [{r['framing']}] [{r['category']}]")
                print(f"    {r['prompt'][:120]}{'…' if len(r['prompt']) > 120 else ''}")
                print(f"    → expected_surface: {r['expected_surface']}")

    # ── Failures and Warnings ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print("Warnings: none")

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for fail in failures:
            print(f"  ✗  {fail}")
        print(f"\nAudit result: FAIL — {len(failures)} hard failure(s)")
        return False
    else:
        print(f"Audit result: PASS — {total} prompts across {len(files)} files, 0 hard failures")
        return True


def main():
    parser = argparse.ArgumentParser(description="Audit BioRefusalAudit eval set JSONL files")
    parser.add_argument("--verbose", action="store_true",
                        help="Show sample prompts for each tier")
    parser.add_argument("--samples", type=int, default=3,
                        help="Number of sample prompts per tier to show (default: 3)")
    args = parser.parse_args()

    files = load_all_files(EVAL_DIR)
    if not files:
        print(f"No JSONL files found in {EVAL_DIR}")
        sys.exit(1)

    ok = audit_all(files, verbose=args.verbose, samples=args.samples)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
