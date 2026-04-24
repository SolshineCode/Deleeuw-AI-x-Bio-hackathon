# Reviewer Quickstart — One-Command Reproduction

Per specialist review: "Freeze one known-good environment and export a one-command reproduction path with expected runtime and artifact names."

## Environment

Known-good local box:
- Windows 11, Git Bash / WSL for scripts, Python 3.13.5 via Anaconda.
- NVIDIA GTX 1650 Ti Max-Q, 4 GB VRAM, driver 581.57, CUDA 13.0 runtime.
- `torch 2.6.0+cu124`, `transformers 5.6.0`, `sae_lens 6.39.0`, `accelerate 1.13.0`, `bitsandbytes 0.49.2`.
- Gemma 2 2B-IT: `device_map="auto"` (partial GPU + CPU offload; fp16 ≈ 5 GB > 4 GB VRAM).
- Gemma 4 E2B-IT: 4-bit quantized via bitsandbytes, `device_map={"": 0}` (forces all weights to cuda:0; `device_map="auto"` silently picks CPU for multimodal model classes — see `TROUBLESHOOTING.md`).

HF weights & SAE caches (download on first run; ~5 GB + 280 MB):
- `google/gemma-2-2b-it`
- `google/gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82`

Set once: `huggingface-cli login` with an account that has accepted the Gemma license.

## Reproduce the full flagship result

```bash
cd Deleeuw-AI-x-Bio-hackathon

# One-time setup:
python -m venv .venv
source .venv/Scripts/activate              # Git Bash on Windows
pip install -e ".[dev,dashboard]"
# CUDA torch (Python 3.13 wheels not on default pypi):
pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124

# One-time verify (should print 51 passed):
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m pytest tests/ -q -m "not integration"

# Flagship pipeline (~3h on the GTX 1650 box; ~90 min on a Colab T4):
bash scripts/flagship_pipeline.sh google/gemma-2-2b-it 12 "layer_12/width_16k/average_l0_82"
```

## Expected artifacts

| Path | Content | Expected size |
|---|---|---|
| `runs/flagship/pass1_activations/report.json` | Pass-1 eval (stub catalog; baseline divergence numbers) | ~75 records, 200-400 KB |
| `runs/flagship/pass1_activations/activations.npz` | Per-prompt d_sae=16 384 feature vectors | ~5 MB (compressed) |
| `data/feature_catalog/gemma-2-2b-it.json` | v0.2 auto-tuned catalog (overwrites v0.1 stub) | ~2 KB |
| `configs/calibration_gemma2_2b.yaml` | Fitted T (appended; prior T preserved) | ~1 KB |
| `runs/flagship/pass2_tuned/report.json` | Pass-2 eval with tuned catalog + fitted T | ~75 records, 200-400 KB |
| `runs/flagship/pass2_tuned/report.md` | Human-readable pass-2 aggregate table | ~1 KB |
| `runs/flagship/interventions/*.json` | Per-intervention baseline / ablated / boosted | ~9 files, 5-10 KB each |
| `demo/scaling_plot.png` | Cross-config divergence bar chart | ~70 KB |

## Colab T4 alternative

If local GPU is not available or is too small for larger models:

1. Open the notebook in Colab: [`notebooks/colab_biorefusalaudit.ipynb`](notebooks/colab_biorefusalaudit.ipynb) (Open-in-Colab badge in the notebook header).
2. Colab Runtime → Change runtime type → T4 GPU.
3. Colab userdata secrets → add `HF_TOKEN` (Gemma-accepted).
4. Runtime → Run all. Expected wall clock ~90 min for Gemma 2 9B-IT + Llama 3.1 8B-Instruct.

## Smoke test only

If you just want to verify the pipeline loads and runs one prompt end-to-end:

```bash
mkdir -p runs/smoke
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m biorefusalaudit.cli run \
    --model google/gemma-2-2b-it \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out runs/smoke \
    --sae-id "layer_12/width_16k/average_l0_82" \
    --layer 12 \
    --max-new-tokens 30 \
    --limit 3
```

Expected: 3 records in `runs/smoke/report.json`, each with `divergence`, `surface_label`, `feature_vec`, `flags`. Wall clock ~3 min on the local GPU.

## Troubleshooting

- **Hang after "Running judge consortium: regex"** — pre-2026-04-23 code had a silent device-mismatch in `project_activations`. Pull `main` (commit `446074d` or later).
- **`import sae_lens` segfaults** — known Py 3.13 + torch 2.6 ABI break. BioRefusalAudit bypasses sae_lens for Gemma Scope 1 via `_load_gemma_scope_direct`. No action required.
- **`OutOfMemoryError` on model load** — model + SAE together need ~3.5 GB VRAM. If you see OOM on a 4 GB GPU, close other GPU apps (browsers, ollama) and retry.
- **Daemon doesn't pick up eval completion** — the chain daemon polls for `^\[biorefusalaudit\] Processed` in stderr.log. If your shell doesn't flush, try `unbuffer` or force the eval with `python -u`.

## Full commit log

All work is on `main`. The merge history tells the story:

| PR | Commit | What |
|---|---|---|
| #1 | `1b7d0c4` | MVP: package + 75-prompt eval set + docs + paper skeleton |
| #2 | `474ca12` | Layer-12 provenance + specialist review response + activation dumping |
| #3 | `446074d` | Calibration + intervention scripts + attribution modules |
| #4 | `4f97544` | Flagship pipeline + custom SAE trainer |
| #5 | `897b49a` | Post-eval chain daemon + simplified dashboard |
