"""Run after 6 new interventions complete. Prints updated summary and patches
STATUS.md with the final intervention count and NC rate.

Usage:
    python scripts/report_new_interventions.py
"""
from __future__ import annotations
import json, re
from pathlib import Path


def main():
    records = []
    for f in sorted(Path("runs/interventions").glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        s = d.get("intervention_summary", {})
        records.append({
            "id": d.get("prompt_id", f.stem),
            "tier": d.get("prompt_tier", "?"),
            "framing": d.get("prompt_framing", "?"),
            "baseline_label": d.get("baseline", {}).get("label", "?"),
            "ablated_label": d.get("ablated", {}).get("label", "?"),
            "boosted_label": d.get("boosted", {}).get("label", "?"),
            "baseline_D": d.get("baseline", {}).get("divergence", 0),
            "dD_ablate": s.get("divergence_delta_ablate", 0),
            "dD_boost": s.get("divergence_delta_boost", 0),
            "label_changed": s.get("label_changed_on_ablate", False),
            "named_circuit": s.get("qualifies_as_named_circuit", False),
        })

    nc = sum(r["named_circuit"] for r in records)
    print(f"\n{'='*80}")
    print(f"INTERVENTION SUMMARY — {len(records)} records, {nc} named circuits ({nc}/{len(records)} = {nc/max(1,len(records)):.0%})")
    print(f"{'='*80}")
    print(f"{'ID':10s} {'Tier':30s} {'Base':8s} {'Ablated':8s} {'D_base':7s} {'dD_abl':8s} {'NC':4s}")
    print("-"*80)
    for r in records:
        nc_mark = "YES" if r["named_circuit"] else "no"
        print(f"{r['id']:10s} {r['tier']:30s} {r['baseline_label']:8s} {r['ablated_label']:8s} "
              f"{r['baseline_D']:7.3f} {r['dD_ablate']:8.3f} {nc_mark:4s}")

    # Tier breakdown
    print()
    tiers_nc = {}
    for r in records:
        t = r["tier"]
        tiers_nc.setdefault(t, [0, 0])
        tiers_nc[t][1] += 1
        if r["named_circuit"]:
            tiers_nc[t][0] += 1
    for tier, (nc_t, total_t) in sorted(tiers_nc.items()):
        print(f"  {tier}: {nc_t}/{total_t} named circuits")

    # Framing breakdown
    print()
    framings_nc = {}
    for r in records:
        fr = r["framing"]
        framings_nc.setdefault(fr, [0, 0])
        framings_nc[fr][1] += 1
        if r["named_circuit"]:
            framings_nc[fr][0] += 1
    for framing, (nc_f, total_f) in sorted(framings_nc.items()):
        print(f"  {framing}: {nc_f}/{total_f} named circuits")

    # Update STATUS.md
    status = Path("STATUS.md")
    text = status.read_text(encoding="utf-8")
    old_marker = "🔄 6 additional interventions RUNNING"
    # Compute comply→refuse count dynamically
    comply_refuse = [r for r in records if r["baseline_label"] == "comply" and r["ablated_label"] == "refuse"]
    cr_ids = "/".join(r["id"] for r in comply_refuse)
    new_block = (
        f"- ✅ interventions COMPLETE: {nc} named circuits out of {len(records)} total "
        f"({nc}/{len(records)} = {nc/max(1,len(records)):.0%}). "
        f"Counterintuitive finding: {len(comply_refuse)}/{len(records)} cases ({cr_ids}) showed comply→refuse on ablate, "
        f"suggesting refusal_circuitry features serve compliance-enabling roles in some contexts. "
        f"See `scripts/summarize_interventions.py` for full table."
    )
    if old_marker in text:
        import re
        # Replace the entire RUNNING block (main line + indented note line if present)
        text = re.sub(
            r"- 🔄 6 additional interventions RUNNING.*?(?=\n- |\n\n|\Z)",
            new_block,
            text,
            flags=re.DOTALL
        )
        status.write_text(text, encoding="utf-8")
        print(f"\nSTATUS.md updated: marked interventions COMPLETE ({len(records)} total).")
    else:
        print("\nSTATUS.md marker not found — update manually.")


if __name__ == "__main__":
    main()
