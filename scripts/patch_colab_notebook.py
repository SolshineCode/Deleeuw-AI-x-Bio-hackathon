"""
Patches notebooks/colab_biorefusalaudit.ipynb with:
  1. Robust clone cell (force-clean dir before git clone -> no more exit 128)
  2. --no-llm-judges on G2 9B full run + Llama full run (prevent uniform-prior judge failure)
  3. Rejudge cells after each full run (regex re-classification matches local pipeline)
  4. Updated intro with corrected key findings + link to interactive demo
Run: python scripts/patch_colab_notebook.py
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/colab_biorefusalaudit.ipynb")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }

# ── 1. Cell 0: update intro markdown ────────────────────────────────────────
cells[0]["source"] = [
    "# BioRefusalAudit on Colab T4 — bigger models via free GPU\n",
    "\n",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/notebooks/colab_biorefusalaudit.ipynb)\n",
    "\n",
    "Runs the full BioRefusalAudit eval pipeline on a Colab T4 (16 GB VRAM), covering models the local 4 GB GTX 1650 cannot fit:\n",
    "\n",
    "- **Gemma 2 9B-IT** + **Gemma Scope 1 9B residual SAEs** (layer 20, width 16k)\n",
    "- **Llama 3.1 8B-Instruct** + **Llama Scope residual SAEs**\n",
    "\n",
    "Both models run at bnb 4-bit on T4. Estimated wall clock: ~60-90 minutes per model on a free T4.\n",
    "\n",
    "## Key findings from local 2B-IT runs (confirmed before this notebook)\n",
    "\n",
    "- **0 genuine refusals** across 75 prompts at both 80 and 200 token budgets (corrected after judge-consortium failure; see `TROUBLESHOOTING.md`)\n",
    "- **Hazard-adjacent tier: 100% hedge** at both token budgets — behavioral posture is token-budget-stable\n",
    "- **Named-circuit qualification: 60/75 (80%)** with inverted tier ordering (benign 87% > dual-use 80% > hazard-adjacent 73%)\n",
    "- Mean divergence by tier (200 tok): benign 0.467, dual-use 0.655, hazard-adjacent 0.669\n",
    "\n",
    "This notebook extends those findings to 9B-scale models as a cross-architecture check.\n",
    "\n",
    "**Interactive demo** (explore results in browser, no server needed): `demo/interactive_explorer.html`\n",
    "\n",
    "## Structure\n",
    "\n",
    "1. Setup: check GPU, install deps, clone repo\n",
    "2. Auth: HF\\_TOKEN + GITHUB\\_TOKEN from Colab secrets\n",
    "3. Run: Gemma 2 9B-IT smoke test -> full eval -> regex rejudge\n",
    "4. Run: Llama 3.1 8B-Instruct full eval -> regex rejudge\n",
    "5. Cross-model scaling figure\n",
    "6. Upload results to HuggingFace (or download as zip)\n",
]

# ── 2. Cell 8: robust clone (force-clean to prevent exit 128) ───────────────
# Find it by detecting the REPO = 'SolshineCode/...' line
clone_idx = next(
    i for i, c in enumerate(cells)
    if c["cell_type"] == "code" and any("REPO = 'SolshineCode" in ln for ln in c["source"])
)
cells[clone_idx]["source"] = [
    "import subprocess\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "REPO = 'SolshineCode/Deleeuw-AI-x-Bio-hackathon'\n",
    "BRANCH = 'main'\n",
    "WORK = Path('/content/Deleeuw-AI-x-Bio-hackathon')\n",
    "\n",
    "# Force-clean any partial/stale clone from a prior Colab session.\n",
    "# This prevents exit 128 ('destination path already exists and is not empty').\n",
    "if WORK.exists():\n",
    "    print(f'Removing existing {WORK} ...')\n",
    "    shutil.rmtree(WORK)\n",
    "\n",
    "subprocess.run(\n",
    "    ['git', 'clone', '--depth', '1', '--branch', BRANCH,\n",
    "     f'https://github.com/{REPO}.git', str(WORK)],\n",
    "    check=True\n",
    ")\n",
    "\n",
    "%cd /content/Deleeuw-AI-x-Bio-hackathon\n",
    "!git log --oneline -3\n",
]

# ── 3. Cell 12: G2 9B-IT full run — add --no-llm-judges ─────────────────────
g2_full_idx = next(
    i for i, c in enumerate(cells)
    if c["cell_type"] == "code"
    and any("colab_gemma-2-9b-it" in ln for ln in c["source"])
    and not any("smoke" in ln for ln in c["source"])
    and not any("no-llm-judges" in ln for ln in c["source"])
)
# Rebuild source with --no-llm-judges before the 2>&1 redirect
new_src = []
for ln in cells[g2_full_idx]["source"]:
    if "quantize 4bit" in ln and "2>&1" in ln:
        # Split: keep quantize on one line, add flag, then redirect
        new_src.append("    --quantize 4bit \\\n")
        new_src.append("    --no-llm-judges 2>&1 | tee runs/colab_gemma-2-9b-it/stderr.log | tail -60\n")
    elif "quantize 4bit" in ln:
        new_src.append("    --quantize 4bit \\\n")
        new_src.append("    --no-llm-judges 2>&1 | tee runs/colab_gemma-2-9b-it/stderr.log | tail -60\n")
    else:
        new_src.append(ln)
cells[g2_full_idx]["source"] = new_src

# ── 4. Insert rejudge cell after G2 9B full run ──────────────────────────────
g2_rejudge_cell = code_cell([
    "# Regex re-judge G2 9B results (avoids uniform-prior judge-failure artifact).\n",
    "# Mirrors the local rejudge pipeline; corrected report written to a -rejudged/ subdir.\n",
    "!python scripts/rejudge_stored_completions.py \\\n",
    "    --report runs/colab_gemma-2-9b-it/report.json \\\n",
    "    --out    runs/colab_gemma-2-9b-it-rejudged/report.json\n",
    "\n",
    "import json\n",
    "from collections import Counter\n",
    "data = json.load(open('runs/colab_gemma-2-9b-it-rejudged/report.json'))\n",
    "recs = data.get('records', [])\n",
    "labels = Counter(r['surface_label'] for r in recs)\n",
    "print('G2 9B-IT corrected surface labels:', dict(labels))\n",
    "print('Tier breakdown:')\n",
    "for tier in ('benign_bio', 'dual_use_bio', 'hazard_adjacent_category'):\n",
    "    tier_recs = [r for r in recs if r.get('tier') == tier]\n",
    "    t_labels = Counter(r['surface_label'] for r in tier_recs)\n",
    "    print(f'  {tier}: {dict(t_labels)}')\n",
])
cells.insert(g2_full_idx + 1, g2_rejudge_cell)

# ── 5. Find Llama cell (index shifted by 1 after insert) ────────────────────
llama_idx = next(
    i for i, c in enumerate(cells)
    if c["cell_type"] == "code"
    and any("llama" in ln.lower() for ln in c["source"])
    and any("colab_llama" in ln for ln in c["source"])
)

# Add --no-llm-judges to Llama cell if missing
if not any("no-llm-judges" in ln for ln in cells[llama_idx]["source"]):
    new_src = []
    for ln in cells[llama_idx]["source"]:
        if "quantize 4bit" in ln:
            new_src.append("    --quantize 4bit \\\n")
            new_src.append("    --no-llm-judges 2>&1 | tee runs/colab_llama-3.1-8b-it/stderr.log | tail -60\n")
        else:
            new_src.append(ln)
    cells[llama_idx]["source"] = new_src

# ── 6. Insert rejudge cell after Llama full run ──────────────────────────────
llama_rejudge_cell = code_cell([
    "# Regex re-judge Llama results.\n",
    "!python scripts/rejudge_stored_completions.py \\\n",
    "    --report runs/colab_llama-3.1-8b-it/report.json \\\n",
    "    --out    runs/colab_llama-3.1-8b-it-rejudged/report.json\n",
    "\n",
    "import json\n",
    "from collections import Counter\n",
    "data = json.load(open('runs/colab_llama-3.1-8b-it-rejudged/report.json'))\n",
    "recs = data.get('records', [])\n",
    "labels = Counter(r['surface_label'] for r in recs)\n",
    "print('Llama 3.1 8B corrected surface labels:', dict(labels))\n",
    "print('Tier breakdown:')\n",
    "for tier in ('benign_bio', 'dual_use_bio', 'hazard_adjacent_category'):\n",
    "    tier_recs = [r for r in recs if r.get('tier') == tier]\n",
    "    t_labels = Counter(r['surface_label'] for r in tier_recs)\n",
    "    print(f'  {tier}: {dict(t_labels)}')\n",
])
cells.insert(llama_idx + 1, llama_rejudge_cell)

# ── Write back ───────────────────────────────────────────────────────────────
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched {NB_PATH} ({len(cells)} cells)")
print(f"Clone cell index: {clone_idx}")
print(f"G2 9B full cell index: {g2_full_idx}")
print(f"Llama cell index: {llama_idx}")
