"""
Analyze legality confound results: extract per-compound per-sub-category refuse rates.

Usage:
    python scripts/analyze_legality_confound.py [--runs-glob "runs/legality-*/report.json"]
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


COMPOUND_PREFIXES = {
    "psilocybin": ["psi_"],
    "mescaline": ["mescaline_"],
    "lsd": ["lsd_"],
    "cannabis": ["can_"],
}

CATEGORY_MAP = {
    # psilocybin_control_v1 IDs (legacy mapping from index)
    "psi_001": "pharmacology", "psi_002": "pharmacology",
    "psi_003": "pharmacology", "psi_004": "pharmacology",
    "psi_005": "cultivation",  "psi_006": "cultivation",
    "psi_007": "cultivation",  "psi_008": "cultivation",
    "psi_009": "clinical",     "psi_010": "clinical",
    "psi_011": "clinical",     "psi_012": "clinical",
    "psi_013": "legal",        "psi_014": "legal",
    "psi_015": "legal",
}


def get_subcategory(record: dict) -> str:
    """Extract sub-category from category field or ID-based fallback."""
    cat = record.get("category") or ""
    if cat:
        # Strip compound prefix: "psilocybin_cultivation" → "cultivation"
        parts = cat.split("_")
        if len(parts) >= 2:
            return parts[-1]
        return cat
    # Fallback for psilocybin_control_v1 records without category field
    pid = record.get("id", "")
    return CATEGORY_MAP.get(pid, "unknown")


def get_compound(record: dict) -> str:
    cat = record.get("category") or ""
    pid = record.get("id", "")
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
    args = parser.parse_args()

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

    # Cross-run summary: compound × subcat × model
    print("\n\n=== CROSS-RUN SUMMARY: psilocybin cultivation vs hazard-adjacent ===")
    print("(Showing only cultivation sub-category for the confound check)")
    print()


if __name__ == "__main__":
    main()
