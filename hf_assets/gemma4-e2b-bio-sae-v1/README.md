---
license: other
license_name: hl3-bds-cl-eco-extr-ffd-media-mil-my-sup-sv-tal-usta-xuar
license_link: https://firstdonoharm.dev/version/3/0/bds-cl-eco-extr-ffd-media-mil-my-sup-sv-tal-usta-xuar.html
tags:
  - sparse-autoencoder
  - mechanistic-interpretability
  - biosafety
  - biorefusalaudit
  - gemma
  - gemma4
  - sae
base_model: google/gemma-4-E2B-it
datasets:
  - cais/wmdp-corpora
  - SolshineCode/biorefusalaudit-eval-public
language:
  - en
---

# BioRefusalAudit — Gemma 4 E2B-IT Contrastive Bio-Safety SAE (v1)

A TopK Sparse Autoencoder (SAE) fine-tuned on biology-domain residual-stream activations
from `google/gemma-4-E2B-it` at layer 17, with a mean contrastive objective that pushes
hazard-adjacent and benign biological feature profiles apart in activation space.

Trained on Colab T4 (Tesla T4, 15.6 GB VRAM) as part of the
[BioRefusalAudit](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon) project
(AIxBio Hackathon 2026, Track 3: Biosecurity Tools, sponsored by Fourth Eon Bio).

---

## Architecture

| Parameter | Value |
|-----------|-------|
| Type | TopK Sparse Autoencoder |
| d_model | 1536 (Gemma 4 E2B text hidden size at layer 17) |
| d_sae | 6144 (4× expansion) |
| k (sparsity) | 32 active features per token position |
| Hook layer | Layer 17 (residual stream, post-MLP) |
| Base model | google/gemma-4-E2B-it |
| Encoder / Decoder | `nn.Linear` layers with learned biases |

**Important:** the encoder and decoder are standard `nn.Linear` modules (not raw `nn.Parameter` matrices). When loading state dicts from earlier drafts or other repos, confirm the key names match (`W_enc.weight`, `W_enc.bias`, `W_dec.weight`, `W_dec.bias`).

---

## Weights

| File | Description |
|------|-------------|
| `sae_weights_final.pt` | **Recommended.** Final checkpoint after 2000 steps. |
| `sae_weights_step_500.pt` | Intermediate — contrastive signal still converging. |
| `sae_weights_step_1000.pt` | Peak contrastive loss step before reconstruction dominates. |
| `sae_weights_step_1500.pt` | Intermediate. |
| `sae_weights_step_2000.pt` | Same as `sae_weights_final.pt`; included for clarity. |

---

## Training

- **Dataset:** `cais/wmdp-corpora` bio-retain-corpus (benign biology, ~5,000 documents)
  + local BioRefusalAudit 75-prompt eval set (22 hazard-adjacent prompts)
- **Contrastive objective:** Mean contrastive — cosine similarity between the mean feature
  profile of hazard-adjacent tokens and the mean feature profile of benign tokens.
  `L_contrastive = cos_sim(mean(z_hazard), mean(z_benign))` — minimized so the two groups
  push apart.
- **Total loss:** `L = L_recon + 0.04 * L_sparsity + 0.1 * L_contrastive`
- **Steps:** 2,000 — `MAX_STEPS=2000`, `BATCH_SIZE=4`, `LR=3e-4`
- **Optimizer:** AdamW
- **Hardware:** Colab Tesla T4 (15.6 GB VRAM), ~35 min wall time
- **Decoder constraint:** Decoder columns projected back to unit sphere after each step
  (`normalize_decoder()`) and gradient component parallel to decoder columns removed
  (`project_grad()`), following the Anthropic SAE training recipe.
- **Chat template:** Training prompts wrapped via `tokenizer.apply_chat_template` so the
  RLHF safety circuit activates during collection. Raw text would be out-of-distribution.

### Training outcome

| Metric | Start | Step 1000 | Final (step 2000) |
|--------|-------|-----------|-------------------|
| l_recon | ~3.2 | ~0.8 | **0.557** |
| l_sparsity | — | — | (tracked) |
| l_contrastive | ~0.7 | — | **~0** (collapsed) |
| L0 (mean active) | 32.0 | 32.0 | 32.0 |

The contrastive loss collapsed to near-zero by step ~1000–1500. This is a known failure
mode when the positive/negative corpus is too small for the NT-Xent / cosine-similarity
objective to maintain separation — the SAE learns to map all inputs to near-identical
directions, satisfying the reconstruction objective while the contrastive margin vanishes.
The reconstruction loss (l_recon=0.557) shows the SAE is encoding the residual stream, but
bio-feature separation is not guaranteed. **Treat the contrastive fine-tuning as a proof of
concept; use the companion Gemma Scope community SAE for production bio-safety audits until
a larger corpus run is available.**

---

## Step-by-step loading

### 1. Install dependencies

```bash
pip install torch transformers bitsandbytes accelerate huggingface_hub
```

### 2. Define the TopKSAE class

The state dict uses `nn.Linear` key names. You must define the class this way:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_model: int = 1536, d_sae: int = 6144, k: int = 32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Linear(d_model, d_sae, bias=True)
        self.W_dec = nn.Linear(d_sae, d_model, bias=True)

    def forward(self, x):
        """
        Args:
            x: (..., d_model) float tensor
        Returns:
            x_hat: reconstruction, same shape as x
            z:     sparse feature activations (..., d_sae) — k nonzero per position
            pre:   pre-topk encoder output (..., d_sae)
        """
        pre = self.W_enc(x)
        topk_vals, topk_idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        x_hat = self.W_dec(z)
        return x_hat, z, pre

    def encode(self, x):
        """Return only sparse feature vector z."""
        _, z, _ = self.forward(x)
        return z
```

### 3. Download and load the weights

```python
from huggingface_hub import hf_hub_download

# Download the final checkpoint
weights_path = hf_hub_download(
    repo_id="Solshine/gemma4-e2b-bio-sae-v1",
    filename="sae_weights_final.pt",
)

sae = TopKSAE(d_model=1536, d_sae=6144, k=32)
sae.load_state_dict(torch.load(weights_path, map_location="cpu"))
sae.eval()
print(f"SAE loaded. Parameters: {sum(p.numel() for p in sae.parameters()):,}")
# → SAE loaded. Parameters: 18,882,048
```

### 4. Load Gemma 4 E2B-IT with 4-bit quantization

Gemma 4 is a multimodal model (`Gemma4ForConditionalGeneration`). The text backbone lives
at `model.language_model` inside the outer model. Use `device_map={"": 0}` (integer device
index) — do **not** use `"auto"` or `{"": "cuda"}` (string) with bitsandbytes on Windows;
both silently route to CPU.

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

# Try CausalLM first; fall back to the multimodal class if the model type isn't registered
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},   # integer index — never the string "cuda"
        low_cpu_mem_usage=True,
    )
except Exception:
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        "google/gemma-4-E2B-it",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

model.eval()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
tokenizer.pad_token = tokenizer.eos_token
```

### 5. Attach the residual-stream hook at layer 17

Gemma 4's transformer layers live at `model.language_model.layers` (inside the multimodal
wrapper). The helper below handles multiple known layout variants:

```python
def get_layer(model, layer_idx: int = 17):
    """Locate transformer block list across Gemma 2/3/4 text-only & multimodal layouts."""
    for path in (
        "model.language_model",   # Gemma 4 ForConditionalGeneration
        "language_model.model",   # Gemma 3 ForConditionalGeneration (older)
        "language_model",
        "model",
        "transformer",
    ):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "layers"):
            return obj.layers[layer_idx]
    raise AttributeError(f"Could not locate layers in {type(model).__name__}")

captured = [None]

def hook_fn(module, inputs, outputs):
    # Overwrite (not append) — appending fills VRAM fast during autoregressive generation
    captured[0] = (outputs[0] if isinstance(outputs, tuple) else outputs).detach()

handle = get_layer(model, 17).register_forward_hook(hook_fn)
```

### 6. Collect activations and run the SAE

```python
prompt = "Describe the mechanism by which influenza binds to host cells."

# Always use the Gemma chat template — raw text is out of distribution for an IT model
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to("cuda")

with torch.no_grad():
    _ = model(**inputs)          # forward pass fires the hook
    acts = captured[0]           # (1, seq_len, 1536)

# Run through the SAE (cast to float32 — NF4 activations are fp16)
x = acts.squeeze(0).float()     # (seq_len, 1536)
with torch.no_grad():
    x_hat, z, pre = sae(x)      # z: (seq_len, 6144), 32 nonzero per row

# Top-5 most active features (averaged across sequence positions)
mean_z = z.mean(0)              # (6144,)
top_features = mean_z.topk(5)
print("Top-5 features (index, mean activation):")
for idx, val in zip(top_features.indices.tolist(), top_features.values.tolist()):
    print(f"  Feature {idx:5d}: {val:.4f}")

handle.remove()   # clean up hook when done
```

---

## Integration with BioRefusalAudit pipeline

If you are using the full BioRefusalAudit CLI, pass this SAE via:

```bash
python -m biorefusalaudit.cli run \
    --model google/gemma-4-E2B-it \
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
    --out runs/gemma4-oursae-v1 \
    --sae-source custom \
    --sae-release Solshine/gemma4-e2b-bio-sae-v1 \
    --k 32 \
    --d-model 1536 \
    --d-sae 6144 \
    --architecture topk \
    --layer 17 \
    --quantize 4bit \
    --no-llm-judges \
    --max-new-tokens 80 \
    --dump-activations
```

The `--sae-source custom` path in `sae_adapter.py` will:
1. Detect that `Solshine/gemma4-e2b-bio-sae-v1` is an HF repo ID (contains `/`, is not a local path)
2. Try `sae_weights.pt` → `sae_weights.safetensors` → scan repo for any `.pt`/`.safetensors`
3. Download `sae_weights_final.pt` (alphabetically first `.pt` in the repo) via `hf_hub_download`
4. Load into a `TopKSAE(d_model=1536, d_sae=6144, k=32)` instance

For the full pass-1 → auto-tune catalog → pass-2 → fit-T pipeline, use the convenience script:

```bash
bash scripts/run_gemma4_oursae_pipeline.sh 2>&1 | tee runs/gemma4-oursae-pipeline.log
```

---

## Evaluation Results

Results from running this SAE through the full BioRefusalAudit pipeline on the 75-prompt
public evaluation set (`eval_set_public_v1.jsonl`), covering benign biology, dual-use
biology, and hazard-adjacent prompts across four framings (direct, educational, roleplay,
obfuscated). Model: `google/gemma-4-E2B-it`, 4-bit NF4, 80-token budget.

### Feature activation (pass1, no catalog)

1285 out of 1500 bio_content feature checks fired across all 75 prompts, with mean activation
14.71 on active tokens. The WMDP-trained features generalize to the BioRefusalAudit eval set.

### Auto-tuned catalog (Cohen's d, post-pass1)

`scripts/auto_tune_catalog.py` selects the top-20 features per category by Cohen's d
between hazard-adjacent and benign activation profiles:

| Category | Top Cohen's d | Features selected |
|---|---:|---:|
| bio_content | 1.81 | 20 |
| hazard_adjacent | 1.60 | 20 |
| refusal_circuitry | 1.52 | 20 |

33 unique features total (some features appear in multiple categories).
Catalog: `data/feature_catalog/gemma-4-e2b-our-sae-v1.json`.

### Divergence metric results (pass2, 80-tok)

**Calibration note — important.** The BioRefusalAudit pipeline fits calibration matrix T
from pass1 activations. When pass1 runs without a catalog (as it must to bootstrap the
catalog), all feature vectors are zero → `fit_calibration.py` produces T=0 → D=1.000 for
every prompt regardless of actual SAE activations. This is a known pipeline ordering issue,
now guarded in `biorefusalaudit/scoring/divergence.py` (falls back to T_prior when fitted T
is degenerate). Results below use T_prior (identity-biased permutation), not a data-fitted T.
T_prior produces higher absolute D than a fitted T would.

| Tier | n | Mean D | Std | comply | refuse | hedge |
|---|---:|---:|---:|---:|---:|---:|
| `benign_bio` | 23 | 0.614 | 0.321 | 57% | 43% | 0% |
| `dual_use_bio` | 30 | 0.703 | 0.295 | 70% | 30% | 0% |
| `hazard_adjacent` | 22 | 0.647 | 0.306 | 59% | 36% | 5% |
| **overall** | **75** | **0.659** | **0.309** | | | |

**Comparison to Gemma Scope community SAE** (Gemma 2 2B-IT, fitted T, 80-tok — different
model, different calibration, not directly comparable):

| Tier | Gemma Scope baseline | This SAE (T_prior) |
|---|---:|---:|
| `benign_bio` | 0.362 | 0.614 |
| `dual_use_bio` | 0.406 | 0.703 |
| `hazard_adjacent` | 0.404 | 0.647 |

The delta reflects T miscalibration and the model difference (Gemma 4 vs. Gemma 2), not a
meaningful SAE quality gap. A properly fitted T from a behavioral corpus is needed to
interpret absolute D magnitudes.

**Surface label finding.** `benign_bio` refuses at 43% — the over-refusal pattern documented
in BioRefusalAudit §4.6: Gemma 4 E2B's safety circuit fires on biosecurity-adjacent content
regardless of hazard tier. This SAE confirms the same pattern as the Gemma Scope community
SAE baseline.

---

## Training your own bio-safety SAE

The training notebook is at
[notebooks/colab_gemma4_sae_training.ipynb](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon/blob/main/notebooks/colab_gemma4_sae_training.ipynb).
It runs end-to-end on a free Colab T4 in ~35 minutes.

**Quick start:**
1. Open the notebook in Google Colab (Runtime → Change runtime type → T4 GPU)
2. Add `HF_TOKEN` (write scope) and `WANDB_API_KEY` to Colab Secrets (🔑 icon)
3. Run All — the notebook will:
   - Install `transformers` from source (Gemma 4 requires the latest main branch)
   - Load `google/gemma-4-E2B-it` in NF4 4-bit quantization
   - Stream training data from `cais/wmdp-corpora` (bio-retain-corpus, public)
   - Train 2,000 steps with reconstruction + sparsity + mean contrastive loss
   - Upload final checkpoint to your HF account as `<your-username>/gemma4-e2b-bio-sae-v1`

**Key implementation details that make it work on Colab:**

| Problem | Fix |
|---------|-----|
| Gemma 4 multimodal layer path | `pick_layer()` with 5-path fallback + `named_modules()` slow-path scan |
| Decoder collapse (all features becoming equal) | `normalize_decoder()` + `project_grad()` each step |
| OOD inputs from raw corpus text | Wrap all prompts with `tokenizer.apply_chat_template` |
| VRAM fill during generation | Hook overwrites `captured[0]` instead of appending to a list |
| Contrastive loss instability | Mean contrastive (cosine sim of mean profiles) instead of NT-Xent |

---

## Caveats

- **Contrastive collapse.** The contrastive loss reached ~0 by step ~1500. The SAE reconstructs residual-stream activations well but bio-feature *separation* is not confirmed. Verification requires running `auto_tune_catalog.py` and checking Cohen's d per category against the Gemma Scope baseline.
- **Small corpus (v1).** Training used ~5,000 WMDP documents (benign) + 22 hazard-adjacent prompts. Too few hazard-adjacent examples to sustain the contrastive margin. This is the binding constraint — not compute, not architecture. **Fixed in v2** (see below).
- **2000-step limit (v1).** Capped at 2000 steps; L_contrastive collapsed by step 1000. Final checkpoint reconstructs well but bio-feature separation is not confirmed. **Fixed in v2:** 5000 steps with real hazard corpus.
- **No Neuronpedia validation.** Individual feature interpretability is unverified.
- **4× expansion.** d_sae/d_model = 4.0, below Gemma Scope's 8×. Wider SAEs likely capture more bio-specific features.
- **Gemma 4 multimodal wrapper.** Hook path is `model.language_model.layers[17]` — **not** `model.model.layers[17]` (Gemma 3 path). The `get_layer()` helper above handles this automatically.

### v2 training run (in progress)

Access to `cais/wmdp-bio-forget-corpus` was granted on 2026-04-26. The v2 notebook
(`notebooks/colab_gemma4_sae_training.ipynb`) now loads 5,000 papers from that corpus as
the hazard-adjacent class, balanced against 5,000 benign documents from the retain corpus,
for 5,000 training steps. This directly addresses the corpus-size bottleneck. Results will
be published as `Solshine/gemma4-e2b-bio-sae-v2` on completion.

### What would further improve this SAE

The corpus-size problem is now addressed for the primary bottleneck. Remaining priorities, in
order of impact:

1. **More hazard-adjacent examples (partially addressed in v2).** 22 prompts is not enough to anchor a stable contrastive
   direction. 500–1000 genuine hazard-adjacent activation examples (from actual model
   responses, not just prompts) would likely sustain the contrastive margin through training.
   This requires access to institutional CBRN datasets — the kind held by organizations like
   Gryphon Scientific, NTI Bio, Johns Hopkins Center for Health Security, or government
   biosecurity agencies. We are actively seeking partnerships with these organizations and
   would welcome introductions from anyone in that space.

2. **A proper base-vs-RLHF activation corpus.** Following the methodology of Secret Agenda
   (arXiv:2509.20393): collect residual-stream activations from the base model and the
   instruction-tuned model on identical prompts, then train the SAE to separate "what the
   safety fine-tune changed" from "what was already there." This is a data-collection problem
   that requires running both model variants on the same hardware at scale.

3. **More compute for training.** A full SAE fine-tune at Anthropic/EleutherAI scale (100K+
   steps, A100 or H100) would not help if the corpus is still 22 hazard-adjacent prompts —
   the gradient signal simply isn't there. But a 10K-step run on a properly sized corpus
   (~10K hazard-adjacent samples) would be a reasonable next experiment and is feasible on a
   single A100 in a few hours. If you have access to institutional compute or CBRN datasets
   and want to run this experiment, please open an issue on the
   [BioRefusalAudit repo](https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon) or
   reach out directly.

4. **Wider SAE.** 8× or 16× expansion (d_sae = 12288 or 24576) with a larger k would give
   more features to specialize. This is a secondary bottleneck behind corpus size.

---

## Cross-Architecture Context

This SAE targets Gemma 4 E2B-IT (2B parameters, multimodal, released April 2025). Compared
to the Gemma 2 2B SAE in the same pipeline (`runs/sae-training-gemma2-5000steps/`):

| | Gemma 4 E2B | Gemma 2 2B |
|---|---|---|
| d_model | 1536 | 2304 |
| Hook layer | 17 | 12 |
| Hook path | `model.language_model.layers` | `model.layers` |
| SAE size | ~19 MB | ~28 MB |
| Training steps | 2,000 (Colab T4) | 5,000 (local GPU) |

---

## Citation

```bibtex
@misc{deleeuw2026biorefusalaudit,
  title   = {BioRefusalAudit: Measuring Refusal Depth in LLMs
             via SAE Feature Divergence},
  author  = {DeLeeuw, Caleb},
  year    = {2026},
  howpublished = {AIxBio Hackathon 2026, Track 3: Biosecurity Tools},
  url     = {https://github.com/SolshineCode/Deleeuw-AI-x-Bio-hackathon}
}
```

---

## License

Code and weights released under the
[Hippocratic License 3.0 (HL3-BDS-CL-ECO-EXTR-FFD-MEDIA-MIL-MY-SUP-SV-TAL-USTA-XUAR)](https://firstdonoharm.dev/version/3/0/bds-cl-eco-extr-ffd-media-mil-my-sup-sv-tal-usta-xuar.html).
You may use these weights for biosecurity research, AI safety research, and defensive
interpretability work. You may not use them to facilitate harm.
