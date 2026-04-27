"""
Analyze legality confound results: extract per-compound per-sub-category refuse rates.

Usage:
    python scripts/analyze_legality_confound.py [--runs-glob "runs/legality-*/report.json"]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


# Fallback map for psilocybin_control_v1 records without category field
CATEGORY_MAP = {
    "psi_001": "pharmacology", "psi_002": "pharmacology",
    "psi_003": "pharmacology", "psi_004": "pharmacology",
    "psi_005": "cultivation",  "psi_006": "cultivation",
    "psi_007": "cultivation",  "psi_008": "cultivation",
    "psi_009": "clinical",     "psi_010": "clinical",
    "psi_011": "clinical",     "psi_012": "clinical",
    "psi_013": "legal",        "psi_014": "legal",
    "psi_015": "legal",
}

# Loaded at startup from JSONL: id → category string
_JSONL_CATEGORY: dict[str, str] = {}


def load_jsonl_categories(*paths: Path) -> None:
    """Populate _JSONL_CATEGORY from one or more eval-set JSONL files."""
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pid = obj.get("id", "")
                    cat = obj.get("category", "")
                    if pid and cat:
                        _JSONL_CATEGORY[pid] = cat
                except json.JSONDecodeError:
                    pass


def get_subcategory(record: dict) -> str:
    """Extract sub-category: JSONL lookup → report field → ID fallback."""
    pid = record.get("id", "")
    # 1. JSONL join (most reliable)
    if pid in _JSONL_CATEGORY:
        cat = _JSONL_CATEGORY[pid]
        parts = cat.split("_")
        return parts[-1] if len(parts) >= 2 else cat
    # 2. category field in report record
    cat = record.get("category") or ""
    if cat:
        parts = cat.split("_")
        return parts[-1] if len(parts) >= 2 else cat
    # 3. static ID map for psilocybin_control_v1
    return CATEGORY_MAP.get(pid, "unknown")


def get_compound(record: dict) -> str:
    pid = record.get("id", "")
    # JSONL join first
    if pid in _JSONL_CATEGORY:
        cat = _JSONL_CATEGORY[pid]
        for compound in ("mescaline", "lsd", "cannabis", "psilocybin"):
            if cat.startswith(compound):
                return compound
    # Fallback: field + ID heuristics
    cat = record.get("category") or ""
    if "mescaline" in cat or "mescaline" in pid:
        return "mescaline"
    if "lsd" in cat or pid.startswith("lsd_"):
        return "lsd"
    if "can_" in pid or "cannabis" in cat:
        return "cannabis"
    if "psi_" in pid or "psilocybin" in cat:
        return "psilocybin"
    tier = record.get("tier", "")
    if tier == "psilocybin_control":
        return "psilocybin"
    return "unknown"


def analyze(report_path: Path):
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    model = data.get("model_name", report_path.parent.name)

    # compound -> subcategory -> [labels]
    results = defaultdict(lambda: defaultdict(list))
    for r in records:
        compound = get_compound(r)
        subcat = get_subcategory(r)
        label = r.get("surface_label", "unknown").lower()
        results[compound][subcat].append(label)

    print(f"\n=== {model} | {report_path.parent.name} ===")
    print(f"  Total records: {len(records)}")

    header = f"{'Compound':12} {'Sub-cat':14} {'n':>4} {'refuse%':>8} {'hedge%':>8} {'comply%':>8}"
    print("  " + header)
    print("  " + "-" * len(header))

    for compound in sorted(results):
        for subcat in sorted(results[compound]):
            labels = results[compound][subcat]
            n = len(labels)
            refuse_pct = 100 * labels.count("refuse") / n
            hedge_pct = 100 * labels.count("hedge") / n
            comply_pct = 100 * (labels.count("comply") + labels.count("partial")) / n
            print(f"  {compound:12} {subcat:14} {n:>4} {refuse_pct:>7.0f}% {hedge_pct:>7.0f}% {comply_pct:>7.0f}%")

    return results, model


def main():
    parser = argparse.ArgumentParser(description="Analyze legality confound results")
    parser.add_argument("--runs-glob", default="runs/legality-*/report.json",
                        help="Glob for report.json files")
    parser.add_argument("--also-psilocybin", action="store_true",
                        help="Include psilocybin control runs from runs/psilocybin-*/report.json")
    parser.add_argument("--eval-sets", nargs="*",
                        default=[
                            "data/eval_set_public/eval_set_legality_confound_v1.jsonl",
                            "data/eval_set_public/eval_set_public_v1.jsonl",
                        ],
                        help="JSONL eval sets to join for category lookup")
    args = parser.parse_args()

    # Pre-load category lookup from eval set JSONLs
    load_jsonl_categories(*[Path(p) for p in args.eval_sets])
    print(f"[info] JSONL category lookup: {len(_JSONL_CATEGORY)} prompt IDs loaded")

    reports = sorted(Path(".").glob(args.runs_glob))
    if args.also_psilocybin:
        reports += sorted(Path(".").glob("runs/psilocybin-*/report.json"))

    if not reports:
        print(f"No reports found matching {args.runs_glob}")
        return

    all_results = {}
    for rp in reports:
        results, model = analyze(rp)
        all_results[rp.parent.name] = (results, model)

    # Cross-run summary: psilocybin cultivation refuse% per model
    print("\n\n=== CROSS-RUN SUMMARY: psilocybin cultivation vs cannabis cultivation ===")
    print("(Confound check: both Schedule I; only psilocybin should show elevated refusals)")
    print()
    header = f"{'Run':40} {'Model':30} {'psi_cult%':>10} {'can_cult%':>10} {'n_psi':>6} {'n_can':>6}"
    print(header)
    print("-" * len(header))
    for run_name, (results, model) in sorted(all_results.items()):
        psi_labels = results.get("psilocybin", {}).get("cultivation", [])
        can_labels = results.get("cannabis", {}).get("cultivation", [])
        psi_n = len(psi_labels)
        can_n = len(can_labels)
        psi_pct = f"{100 * psi_labels.count('refuse') / psi_n:.0f}%" if psi_n > 0 else "—"
        can_pct = f"{100 * can_labels.count('refuse') / can_n:.0f}%" if can_n > 0 else "—"
        print(f"  {run_name:38} {model:30} {psi_pct:>10} {can_pct:>10} {psi_n:>6} {can_n:>6}")


if __name__ == "__main__":
    main()
