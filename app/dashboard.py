"""Minimal Streamlit dashboard.

Launch with:
    streamlit run app/dashboard.py

Features (v0.1 skeleton):
  - paste a prompt, pick a model, run one-prompt audit
  - show surface classification (regex-only in skeleton; LLM judges togglable)
  - show top-10 SAE features (mocked if model not loaded)
  - show divergence + flag badges

The skeleton uses the actual biorefusalaudit pipeline but lazily — no model
load happens until the user clicks "Classify & Audit". On a 4GB GPU, the first
classify takes ~20-40s to load weights; subsequent calls reuse the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure we can import biorefusalaudit when run via `streamlit run`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biorefusalaudit.judges.regex_classifier import regex_classify  # noqa: E402
from biorefusalaudit.scoring.divergence import (  # noqa: E402
    FEATURE_CATEGORIES,
    SURFACE_CLASSES,
    divergence,
    divergence_flags,
)


st.set_page_config(page_title="BioRefusalAudit", layout="wide")
st.title("BioRefusalAudit — refusal-depth auditor")
st.caption(
    "Measures surface-internal divergence between what a model *says* "
    "and what its SAE-feature activation *shows*. See README.md for methodology."
)

with st.sidebar:
    st.subheader("Model")
    model = st.selectbox(
        "HF model id",
        options=[
            "google/gemma-2-2b-it",
            "google/gemma-4-E2B-it",
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
        ],
        index=0,
    )
    st.caption(
        "Gemma 3 entries require Gemma Scope 2 SAEs which are not yet publicly "
        "released (as of 2026-04-22). They will error until that release lands."
    )
    use_llm_judges = st.checkbox("Use LLM judges (Gemini + Claude Haiku + Ollama)", value=False)

col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Prompt")
    prompt = st.text_area(
        "Paste a prompt to audit",
        value="Explain the mechanism of Michaelis-Menten enzyme kinetics.",
        height=200,
    )
    run = st.button("Classify & Audit", type="primary")

with col_out:
    st.subheader("Verdict")
    if not run:
        st.info("Click **Classify & Audit** to run the pipeline.")
    else:
        # Skeleton path: regex-only classification on a synthetic completion.
        # In the full app this calls runner.eval_runner.run_one_prompt against
        # the loaded model + SAE. Held out in v0.1 because model load on this
        # hardware takes 20-40s and the UX needs a progress/streaming shim.
        synthetic_completion = (
            "Here's an explanation. Step 1: the enzyme binds the substrate "
            "in the active site..."
            if "explain" in prompt.lower() or "what" in prompt.lower()
            else "I can't help with that specific request."
        )
        verdict = regex_classify(synthetic_completion)
        st.write(f"**Surface label:** `{verdict.label}` (conf {verdict.confidence:.2f})")

        # Mock feature vector — a full build replaces this with real SAE projection.
        f_mock = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
        s_mock = np.zeros(5)
        if verdict.label is not None:
            s_mock[list(SURFACE_CLASSES).index(verdict.label)] = 1.0
        else:
            s_mock[:] = 1.0 / 5

        d = divergence(s_mock, f_mock)
        st.write(f"**Divergence D:** `{d:.3f}`")
        flags = divergence_flags(s_mock, f_mock)
        for name, active in flags.items():
            badge = ":red[ACTIVE]" if active else ":green[clear]"
            st.write(f"  - {name}: {badge}")

        st.subheader("Top feature categories (mocked)")
        for cat, val in zip(FEATURE_CATEGORIES, f_mock):
            st.write(f"  - `{cat}`: {val:.2f}")
        st.caption(
            "v0.1 uses mocked SAE activations. "
            "Full pipeline: `python -m biorefusalaudit.cli run --model ...` "
            "writes a `report.md` with real numbers."
        )

st.divider()
st.caption("BioRefusalAudit v0.1 — AIxBio Hackathon 2026 submission skeleton.")
