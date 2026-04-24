"""Print a summary table of all intervention records in runs/interventions/.

Usage:
    python scripts/summarize_interventions.py [--dir runs/interventions]
"""
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/interventions")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print("No intervention JSON files found.")
        return

    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"SKIP {f.name}: {e}")
            continue
        s = d.get("intervention_summary", {})
        rows.append({
            "id": d.get("prompt_id", f.stem),
            "tier": d.get("prompt_tier", "?"),
            "framing": d.get("prompt_framing", "?"),
            "category": d.get("category", "?"),
            "baseline_label": d.get("baseline", {}).get("label", "?"),
            "ablated_label": d.get("ablated", {}).get("label", "?"),
            "boosted_label": d.get("boosted", {}).get("label", "?"),
            "baseline_D": d.get("baseline", {}).get("divergence", float("nan")),
            "dD_ablate": s.get("divergence_delta_ablate", float("nan")),
            "dD_boost": s.get("divergence_delta_boost", float("nan")),
            "label_changed": s.get("label_changed_on_ablate", False),
            "named_circuit": s.get("qualifies_as_named_circuit", False),
        })

    # Print table
    hdr = f"{'ID':10s} {'Tier':30s} {'Framing':12s} {'Base':8s} {'Ablated':8s} {'D_base':7s} {'dD_abl':8s} {'dD_bst':8s} {'LC':4s} {'NC':4s}"
    print(hdr)
    print("-" * len(hdr))
    nc_count = 0
    for r in rows:
        lc = "YES" if r["label_changed"] else "no"
        nc = "YES" if r["named_circuit"] else "no"
        if r["named_circuit"]: nc_count += 1
        print(f"{r['id']:10s} {r['tier']:30s} {r['framing']:12s} {r['baseline_label']:8s} {r['ablated_label']:8s} "
              f"{r['baseline_D']:7.3f} {r['dD_ablate']:8.3f} {r['dD_boost']:8.3f} {lc:4s} {nc:4s}")

    print(f"\nTotal: {len(rows)} interventions, {nc_count} qualify as named circuits")

if __name__ == "__main__":
    main()
