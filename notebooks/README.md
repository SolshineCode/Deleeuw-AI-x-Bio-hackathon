# Notebooks

## `colab_biorefusalaudit.ipynb` — run on Colab T4

For the models that don't fit a 4 GB local GPU (Gemma 2 9B-IT, Llama 3.1 8B-Instruct), the Colab notebook runs the full eval pipeline on a free T4 (16 GB VRAM).

**Open in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/notebooks/colab_biorefusalaudit.ipynb)

**Prerequisites (set these before running):**
1. Colab runtime → `T4 GPU` (free tier).
2. Colab userdata secrets (left sidebar → key icon):
   - `HF_TOKEN` — HuggingFace token with Gemma + Llama licenses accepted.
   - `GITHUB_TOKEN` — optional; only if pushing commits back from the notebook.

**Wall clock estimates (free T4):**
- Setup + auth + clone + `pip install`: ~3 min
- Gemma 2 9B-IT full eval (75 prompts): ~45 min
- Llama 3.1 8B-Instruct full eval (75 prompts): ~45 min
- Cross-model scaling figure + download: ~1 min
- **Total: ~95 min** — comfortably under the free T4 session limit.

**Output artifacts:**
- `runs/colab_gemma-2-9b-it/report.md` + `report.json`
- `runs/colab_llama-3.1-8b-it/report.md` + `report.json`
- `runs/cross_model/scaling_plot.png`
- `colab_runs.zip` — downloaded to your laptop

**Safety note:**
The notebook commits `runs/` *in the Colab workspace* but **does not push to GitHub automatically**, honoring the repo's no-push-without-explicit-approval policy. Pull the branch to your laptop, inspect, and push manually.

## Planned follow-on notebooks

- `01_feature_discovery.ipynb` — Neuronpedia automated feature candidate pull per category
- `02_eval_walkthrough.ipynb` — step-by-step pipeline demo with visualizations
- `03_cross_model_scaling.ipynb` — fuller scaling analysis across all model runs
- `04_case_studies.ipynb` — single-prompt deep-dives on the highest-divergence cases
