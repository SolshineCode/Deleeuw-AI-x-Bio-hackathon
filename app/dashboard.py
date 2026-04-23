"""Simplified Streamlit dashboard.

Per specialist review (2026-04-23): "Make the dashboard show one prompt, one
completion, one divergence value, one feature panel, and one intervention
result; cut everything else."

Launch:
    streamlit run app/dashboard.py

The dashboard renders the latest `runs/*/report.json` it finds (prefers
`runs/gemma-2-2b-it-L12-tuned/` when present) and shows for one selected
prompt:

  - prompt text + tier/framing
  - model completion + surface label + regex-match evidence
  - divergence D (with prior-vs-fitted T comparison if calibration_delta.json exists)
  - top-10 firing features with category labels
  - intervention result panel (if runs/interventions/<prompt_id>_*.json exists)

No baseline classification on fresh text, no mocked paths. Reads the
authoritative artifact JSONs; if nothing is there, tells the user to run
the flagship pipeline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from biorefusalaudit.features.feature_profiler import FeatureCatalog  # noqa: E402
from biorefusalaudit.scoring.divergence import FEATURE_CATEGORIES  # noqa: E402


st.set_page_config(page_title="BioRefusalAudit", layout="wide")
st.title("BioRefusalAudit")
st.caption(
    "One flagship view: one prompt, one completion, one divergence, one feature panel, "
    "one intervention. Reads the authoritative run artifacts — no mocked data."
)


def _find_latest_report() -> Path | None:
    """Prefer the tuned run; fall back to whatever has the freshest report.json."""
    for candidate in [
        REPO / "runs" / "gemma-2-2b-it-L12-tuned" / "report.json",
        REPO / "runs" / "gemma-2-2b-it-L12-activations" / "report.json",
        REPO / "runs" / "gemma-2-2b-it-L12-fast" / "report.json",
    ]:
        if candidate.exists():
            return candidate
    reports = sorted(REPO.glob("runs/*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _find_intervention(prompt_id: str) -> list[Path]:
    """List any intervention JSONs matching this prompt id."""
    paths = list((REPO / "runs" / "interventions").glob(f"{prompt_id}_*.json")) \
        if (REPO / "runs" / "interventions").exists() else []
    return paths


def _load_catalog() -> FeatureCatalog | None:
    path = REPO / "data/feature_catalog/gemma-2-2b-it.json"
    if not path.exists():
        return None
    try:
        return FeatureCatalog.load(path)
    except Exception:
        return None


report_path = _find_latest_report()
if report_path is None:
    st.error("No report.json found under runs/. Run the flagship pipeline first: "
             "`bash scripts/flagship_pipeline.sh`.")
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))
records = report.get("records", [])
st.sidebar.success(f"Loaded report: `{report_path.relative_to(REPO)}`")
st.sidebar.caption(f"Model: `{report.get('model_name', '?')}` · SAE: `{report.get('sae_source', '?')}`")
st.sidebar.caption(f"{len(records)} prompts")

if not records:
    st.warning("Report has no records — run produced zero completions.")
    st.stop()

# Aggregate view
agg = report.get("aggregate", {})
cols = st.columns(3)
tiers = ["benign_bio", "dual_use_bio", "hazard_adjacent_category"]
for col, tier in zip(cols, tiers):
    a = agg.get(tier, {})
    with col:
        st.metric(
            f"{tier} mean D",
            f"{a.get('mean_divergence', 0.0):.3f}" if a else "—",
            help=f"std={a.get('std_divergence', 0.0):.3f}; n={a.get('n', 0)}" if a else None,
        )

st.divider()

# Per-prompt selector
st.subheader("Per-prompt audit")
ids = [r["id"] for r in records]
chosen = st.selectbox("Prompt id", ids, index=0)
rec = next(r for r in records if r["id"] == chosen)

col_l, col_r = st.columns([3, 2])
with col_l:
    st.markdown(f"**Tier:** `{rec['tier']}` · **Framing:** `{rec['framing']}`")
    st.markdown("**Prompt**")
    st.code(rec.get("prompt", "")[:600] + ("…" if len(rec.get("prompt", "")) > 600 else ""))
    st.markdown("**Completion**")
    st.code(rec.get("completion", "")[:800] + ("…" if len(rec.get("completion", "")) > 800 else ""))

with col_r:
    st.metric("Divergence D", f"{rec['divergence']:.3f}")
    st.metric("Surface label", rec["surface_label"])
    if "judge_disagreement" in rec:
        st.metric("Judge disagreement", f"{rec['judge_disagreement']:.2f}")
    st.markdown("**Flags**")
    for flag, active in rec.get("flags", {}).items():
        st.markdown(f"- {flag}: {'🚩 active' if active else '· clear'}")

st.divider()

st.subheader("Feature panel")
f_vec = np.asarray(rec.get("feature_vec", []))
if f_vec.size >= 5:
    cat_cols = st.columns(len(FEATURE_CATEGORIES))
    for c, cat in zip(cat_cols, FEATURE_CATEGORIES):
        with c:
            val = float(f_vec[list(FEATURE_CATEGORIES).index(cat)]) if len(f_vec) > list(FEATURE_CATEGORIES).index(cat) else 0.0
            st.metric(cat, f"{val:.3f}")
st.caption("5-category projection of the SAE feature vector for this prompt. Source catalog: `data/feature_catalog/gemma-2-2b-it.json`.")

catalog = _load_catalog()
if catalog is not None:
    cat_note = getattr(catalog, "model_name", "")
    st.caption(f"Catalog source: {cat_note}")

st.divider()

st.subheader("Intervention")
interventions = _find_intervention(chosen)
if interventions:
    for ipath in interventions:
        data = json.loads(ipath.read_text(encoding="utf-8"))
        s = data.get("intervention_summary", {})
        st.markdown(
            f"**`{ipath.name}`** — category `{data.get('category', '?')}` · top-"
            f"{len(data.get('feature_ids_intervened', []))} features"
        )
        icols = st.columns(3)
        with icols[0]:
            st.markdown("**Baseline**")
            st.caption(f"label: `{data['baseline']['label']}` · D={data['baseline']['divergence']:.3f}")
            st.code(data['baseline']['completion'][:300] + "…")
        with icols[1]:
            st.markdown("**Ablated**")
            st.caption(f"label: `{data['ablated']['label']}` · D={data['ablated']['divergence']:.3f}")
            st.code(data['ablated']['completion'][:300] + "…")
        with icols[2]:
            st.markdown(f"**Boosted ({data['boosted'].get('boost_factor', '?')}×)**")
            st.caption(f"label: `{data['boosted']['label']}` · D={data['boosted']['divergence']:.3f}")
            st.code(data['boosted']['completion'][:300] + "…")
        st.info(
            f"**Named circuit?** {'✓ YES' if s.get('qualifies_as_named_circuit') else '✗ no'} · "
            f"label changed on ablate: {s.get('label_changed_on_ablate')} · "
            f"|ΔD_ablate|={s.get('divergence_delta_ablate', 0):.3f} · "
            f"|ΔD_boost|={s.get('divergence_delta_boost', 0):.3f}"
        )
else:
    st.info("No intervention record for this prompt. Run: "
            f"`python scripts/run_intervention.py --prompt-id {chosen} --category refusal_circuitry`")

st.divider()
st.caption(f"BioRefusalAudit — AIxBio Hackathon 2026. Report: `{report_path.relative_to(REPO)}`.")
