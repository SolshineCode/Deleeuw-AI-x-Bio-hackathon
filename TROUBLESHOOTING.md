# Troubleshooting

Common issues encountered on the tested environment (Windows 11, Python 3.13, GTX 1650 Ti Max-Q 4GB) and their fixes. The table from [README §Troubleshooting](./README.md#troubleshooting) is expanded below, with additional rows for Windows + Python 3.13 + low-VRAM specifics.

---

## HuggingFace symlink error on Windows

**Symptom.** On first `from_pretrained()` call, you see something like:

```
OSError: [WinError 1314] A required privilege is not held by the client
```

or a warning that the HF cache is falling back to copies instead of symlinks, followed by slow model loads and doubled disk usage.

Windows does not allow unprivileged processes to create NTFS symlinks by default. The HuggingFace Hub client tries to use them for cache deduplication.

**Fix.** Either set the environment variable to disable symlinks entirely, or run your shell as administrator once so the symlinks succeed:

```powershell
# Option A — disable symlinks (recommended; safe, minor disk cost)
$env:HF_HUB_DISABLE_SYMLINKS = "1"

# Option B — run PowerShell as admin (right-click → Run as administrator)
# Then re-run the download.
```

Put the env var in your venv activation script or `.env` if you want it permanent.

---

## CUDA OOM on 4GB GPU

**Symptom.**

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

when loading Gemma 3 1B or larger in fp16.

**Fix.** Use the 270M model for local dev, or enable 4-bit quantization via bitsandbytes for 1B:

```powershell
python -m biorefusalaudit.cli run `
    --model google/gemma-3-270m-it `
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl `
    --out runs/local

# For 1B with quantization:
python -m biorefusalaudit.cli run `
    --model google/gemma-3-1b-it --quantize 4bit `
    --eval-set data/eval_set_public/eval_set_public_v1.jsonl `
    --out runs/local_1b
```

Anything beyond 1B on a 4GB card should be rented on A100 — see [docs/COMPUTE.md](docs/COMPUTE.md).

---

## `sae_lens` ImportError

**Symptom.**

```
ImportError: cannot import name 'JumpReLUSAE' from 'sae_lens'
```

or a general missing-attribute error when `biorefusalaudit.models.sae_adapter` tries to import Gemma Scope 2 helpers.

**Fix.** Gemma Scope 2 (released Dec 2025) requires `sae_lens >= 4.0`. Pin it:

```powershell
pip install "sae_lens>=4.0"
```

If you installed from a stale `requirements.lock`, regenerate it. If you see a conflict against `transformers` or `torch`, match `sae_lens>=4.0`'s ranges; it pulled forward both for Gemma 3 support.

---

## PowerShell execution policy blocks scripts

**Symptom.** `.\scripts\run_eval.ps1` or `.\scripts\setup.ps1` fails with:

```
File ... cannot be loaded because running scripts is disabled on this system.
```

**Fix.** Set an execution policy for your user scope only (not machine-wide):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

`RemoteSigned` allows local scripts to run without a signature while still requiring a signature for downloaded scripts. If your organization policy blocks this, use `-Scope Process` for a single session.

---

## Neuronpedia API rate limit

**Symptom.** Feature-discovery notebook or `feature_discovery.py` returns 429 errors partway through a run, or feature catalogs come back incomplete.

**Fix.** The tool caches Neuronpedia responses at `data/.neuronpedia_cache/`. If a run was partial, re-running will resume from cache. If you need to force a refresh (e.g. after an upstream Neuronpedia update), delete the cache directory:

```powershell
Remove-Item -Recurse -Force data\.neuronpedia_cache
```

If you are hitting rate limits repeatedly on a fresh run, pause and resume — do not parallelize Neuronpedia queries across processes.

---

## Gemini CLI auth fails

**Symptom.** Judge consortium logs `gemini: authentication failed` or the CLI prompts for interactive login in a non-interactive script.

**Fix.** Log in once, interactively, to cache credentials:

```powershell
gemini auth login
```

Credentials cache at `~/.gemini/`. If the judge is running under a different user (e.g. a scheduled task, a service account), that user needs its own cached token. If you are using a raw API key instead of OAuth, set `GEMINI_API_KEY` in `.env` and confirm the judge loads it — see `.env.example`.

---

## bitsandbytes wheel missing (Windows)

**Symptom.** `pip install bitsandbytes` succeeds but `import bitsandbytes` fails with:

```
RuntimeError: ... CUDA_HOME is not set ...
```

or loading `--quantize 4bit` crashes with a missing `libbitsandbytes_cuda*.dll`.

The mainline bitsandbytes wheels targeted Linux CUDA layouts for a long time, and Windows support has historically required community pre-built wheels.

**Fix.** Install a Windows-targeted pre-built wheel from the bitsandbytes GitHub releases page that matches your CUDA version. For example:

```powershell
# Replace with the actual latest release URL for your CUDA version
pip install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/<tag>/bitsandbytes-<ver>-py3-none-win_amd64.whl
```

After install, verify:

```powershell
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

If you do not have MSVC Build Tools and the wheel refuses to resolve, skip 4-bit and use the 270M fp16 path instead.

---

## `sae_lens` release id for Gemma Scope 2 unknown

**Symptom.** `SAE.from_pretrained(release="gemma-scope-2-...")` raises `KeyError` or the release dropdown on the `sae_lens` side does not list a Gemma Scope 2 id matching your target model.

Gemma Scope 2 release naming inside `sae_lens` has shifted at least once since the Dec 2025 announcement, and the library's static release registry sometimes lags the HF hub.

**Fix.** Fall back to direct HF hub download and load the JumpReLU SAE manually:

```powershell
huggingface-cli download google/gemma-scope-2-2b-pt-res `
    --include "*layer_14*" --local-dir models/gemma_scope_2_layer_14
```

Then load the SAE state dict into a local JumpReLU module. The JumpReLU forward and decoder-norm math are written out in [docs/METHOD.md](docs/METHOD.md) under the calibration section; if you need more than that, the Gemma Scope 2 technical report has the exact formulation. The adapter at `biorefusalaudit/models/sae_adapter.py` already has a `load_from_hub_manual()` path for this fallback — pass `use_sae_lens_registry=False`.

---

## `claude -p` subprocess timing out

**Symptom.** Judge consortium logs

```
ClaudeHaikuJudge: subprocess timed out after 120s
```

and the affected prompts fall back to the next judge in the consortium (or to "unknown" if none remain).

Claude Code's `claude -p` subprocess default timeout in the judge adapter is 120 seconds, which is plenty for small single-prompt judgments but can be exceeded when the worker is cold, the machine is under load, or the prompt is long.

**Fix.** Raise the timeout via env var:

```powershell
$env:CLAUDE_TIMEOUT = "240"
```

or parallelize judge calls across a process pool to hide cold-start latency. `judges/consortium.py` exposes a `max_workers` parameter; set it to 4–8 on a machine with spare cores. Do not raise it so high that you trip Claude Code's own rate limits.

If timeouts persist, fall back to `ANTHROPIC_API_KEY` (direct SDK) instead of the subprocess — the judge prefers the SDK when the env var is set.

---

## Gemma 4 (multimodal) loads on CPU despite CUDA being available

**Symptom.** You launch a 4-bit eval run on Gemma 4 E2B, bitsandbytes imports cleanly, `torch.cuda.is_available()` returns `True`, and the log says `quantize=4bit` — but `nvidia-smi` shows 0 MiB of GPU memory used and 0% GPU utilization for the entire run. Everything executes on CPU at ~1–5 tok/s instead of ~30–60 tok/s.

You may also see a misleading log line during model loading:

```
Current model requires 6178 bytes of buffer for offloaded layers,
which seems does not fit any GPU's remaining memory. Falling back to CPU.
```

6178 bytes is trivially small (not MiB — bytes). The number is not a VRAM estimate; it is a metadata artifact from `accelerate`'s device-map planner when it encounters a model class it cannot size correctly.

**Root cause.** `device_map="auto"` delegates device placement to `accelerate`'s automatic memory planner. That planner calls `infer_auto_device_map()`, which walks the model's module tree and estimates per-layer VRAM usage using HF's internal size estimator. For `AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it")` the actual loaded class is `Gemma4ForConditionalGeneration` — a multimodal architecture containing a vision encoder, a vision-language projector, and a language model. The size estimator does not know how to account for the interleaved cross-attention and vision-tower weights, reports a near-zero or nonsensical VRAM estimate, and falls back to CPU for all layers as the "safe" choice.

The LM-only weights at 4-bit (approximately 1 GB for a 2B-parameter model) fit easily within a 4 GB card. The planner failure is purely a metadata/estimation bug, not a real capacity problem.

**Fix.** Replace `device_map="auto"` with `device_map={"": 0}` for quantized models when CUDA is available. The `{"": 0}` syntax means "put every module in the default namespace on device 0 (cuda:0)". This bypasses the planner entirely and lets bitsandbytes handle the quantized-layer placement directly.

The fix is implemented in `biorefusalaudit/models/model_adapter.py` `load_model()`:

```python
# Before (broken for Gemma4ForConditionalGeneration):
kwargs["device_map"] = "auto"

# After (forces all quantized layers to the resolved target device):
kwargs["device_map"] = {"": device} if "cuda" in device else "auto"
```

This applies to both `quantize="4bit"` and `quantize="8bit"` paths.

**Verification.** After the model finishes loading (~5–10 min for Gemma 4 E2B weight streaming), run:

```powershell
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

You should see your Python PID with ~1000–1300 MiB used. During token generation the GPU utilization (`nvidia-smi dmon`) should spike to 30–80% per token batch. If you still see 0 MiB, the model weights have not moved to GPU — confirm bitsandbytes CUDA kernels are working:

```python
import bitsandbytes as bnb, torch
print(bnb.__version__)                # should be >=0.43.0
l = bnb.nn.Linear4bit(16, 16).cuda() # should not raise
```

**Affected models.** Any HF model whose `AutoModelForCausalLM` resolves to a multimodal class (`ForConditionalGeneration`, `ForImageTextToText`, etc.) will have this problem with `device_map="auto"`. Confirmed on `google/gemma-4-E2B-it` (`Gemma4ForConditionalGeneration`). Standard causal LMs (`Gemma2ForCausalLM`, `LlamaForCausalLM`) are not affected — their size estimators work correctly.

**Do not** use `device_map="auto"` for quantized Gemma 4 on this machine. The chain script (`scripts/gemma4_post_eval_chain.sh`) already passes `--quantize 4bit` explicitly and the adapter enforces `{"": 0}` at load time.

---

## Slow generation on long outputs (VRAM accumulation in residual hook)

**Symptom.** On a 4 GB GPU with a quantized model, most prompts generate quickly (1–3 s) but every few prompts takes 150–200 s. The fast prompts are ones where the model outputs a short refusal (few tokens before EOS); the slow ones are where the model reaches `--max-new-tokens`. `nvidia-smi` shows GPU memory nearly full (≈3920 MiB / 3936 MiB) and utilization oscillating between 10–30% during slow prompts — suggesting VRAM pressure is forcing per-step CPU spill.

**Root cause.** The `residual_stream_hook` in `model_adapter.py` fires on every transformer forward pass. During autoregressive generation, `generate()` calls the model once per new token — so for a 200-token generation, the hook fires 200 times. The original implementation appended each capture to a list: `captured.append(resid.detach())`. Each capture has shape `(1, seq_len, d_model)` where `seq_len` grows by one per step. The accumulated list for a 200-token generation with a 512-token input holds 200 tensors totaling:

```
sum_{t=1}^{200} (512 + t) × 1536 × 4 bytes ≈ 240 MB
```

On a 4 GB GPU already holding ~1 GB of 4-bit model weights, plus CUDA kernel overhead and KV cache, this 240 MB accumulation pushed past the VRAM limit. `accelerate` / PyTorch then silently offloads KV cache or intermediate buffers to RAM, causing the generation step timing to drop from GPU-speed (~30 tok/s) to RAM-bandwidth-limited speed (~1–3 tok/s).

**Fix.** Overwrite instead of append in the hook. Since `getter()` only reads the last element (`captured[-1]`), all intermediate captures are wasted. The fix keeps exactly one tensor in memory regardless of generation length:

```python
# Before (buggy — accumulates one tensor per autoregressive step):
captured.append(resid.detach())

# After (fixed — overwrites in-place, constant VRAM footprint):
if captured:
    captured[0] = resid.detach()
else:
    captured.append(resid.detach())
```

This is implemented in `biorefusalaudit/models/model_adapter.py` in the `residual_stream_hook` function. After the fix, generation time for 200-token outputs drops from ~165 s back to the expected ~6–8 s on the GTX 1650 Ti at 4-bit.

**Why it didn't affect Gemma 2 2B-IT.** The Gemma 2 path runs in fp16 with `device_map="auto"` (partial GPU + CPU offload), so the residual tensors are split across devices and the GPU pressure is lower. The fp16 model also has more headroom per parameter than the bitsandbytes 4-bit layout. On Gemma 4 E2B at 4-bit the available headroom is ~15 MiB, so the accumulation manifests immediately.

---

## `HF_TOKEN` gated access for Gemma 3

**Symptom.** `from_pretrained("google/gemma-3-270m-it")` fails with:

```
OSError: You are trying to access a gated repo. ... 401 Client Error
```

The Gemma 3 family is gated on Hugging Face and requires license acceptance before a token can pull weights.

**Fix.** This is a two-step process, and both steps are required:

1. **Accept the license on the HF web UI.** Go to `https://huggingface.co/google/gemma-3-270m-it` (and each other Gemma variant you plan to use), click through the Gemma Terms of Use, and wait for the "Access granted" confirmation. This is per-model — you must accept for each size (270M, 1B, 4B, etc.) separately.
2. **Log in the CLI with a token that has `read` scope.**

   ```powershell
   huggingface-cli login
   # paste your token with read scope when prompted
   ```

   Or set `HF_TOKEN` in `.env` and restart your shell / reactivate venv.

If you see the 401 after completing both steps, confirm the HF account that accepted the license is the same account whose token you're using. Mismatched accounts are the common cause.

---

## Chat template missing → empty/looping completions on IT models (CONFIRMED 2026-04-23)

**Symptom:** 30–40% of prompts produce empty completions; others loop or echo the prompt. Regex judge classifies all as "refuse," inflating refusal counts to 100%. Gemma 4 E2B pass-4: 30 empty, 12 loops, 33 nominally coherent (most echoes).

**Root cause:** `generate_completion()` passed raw `prompt` string to the tokenizer with no chat template. Instruction-tuned models trained on `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n` see raw text as out-of-distribution and either emit EOS immediately (empty) or echo/loop the prompt text. Gemma 2 2B-IT is more tolerant; Gemma 4 is not.

**Fix (2026-04-23):** `_apply_chat_template(lm, prompt)` in `model_adapter.py` calls `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` when `tokenizer.chat_template` is set. Falls back to raw prompt for base models. Prefix-stripping at decode time updated to try both formatted and raw prefix.

**Impact on Gemma 4 E2B pass-4 results:** Surface labels (all-refuse) are suspect for empty/loop completions. Raw feature vectors and `hazard_features_active_despite_refusal` flags are real. Re-run with fix = pass-5.

**Detection:** `report.json` records with `completion == ""` or completions where `len(set(completion.split()[:30])) / 30 < 0.3`.

---

## Ghost Python processes cause 5-7× slowdown (VRAM contention)

**Symptom:** Long generations jump from ~85s to ~400-570s; short refusals remain fast (1-7s). No OOM error — it looks like VRAM accumulation, but the hook fix is in place.

**Root cause:** On Windows, killing a Git Bash shell (`kill <PID>` or Ctrl-C) does not kill Python subprocesses launched from it. The subprocess becomes a background orphan and continues holding VRAM. Launching a second eval run while the first is still running puts two Gemma 4 models on a 4 GB card: each gets ~2 GB, both spill to CPU constantly, and all long-generation prompts become 5-7× slower.

**Diagnosis:**
```powershell
Get-Process python* | Select-Object Id, CPU, WorkingSet, StartTime | Format-Table
```
If you see two Python processes with similar start times and similar CPU accumulation, they're competing.

**Fix:** Kill all old Python processes before launching a new run:
```powershell
# Identify old PIDs (e.g., started before 18:20):
Get-Process python* | Format-Table Id, StartTime

# Kill specific PIDs:
Stop-Process -Id <OLD_PID_1>, <OLD_PID_2> -Force

# Verify only the new run remains:
Get-Process python* | Format-Table Id, CPU, StartTime
```
On Git Bash, use `powershell.exe -Command "Stop-Process -Id <PID> -Force"`.

**Occurred:** 2026-04-23, pass-4 Gemma 4 E2B run. Killing 2 ghost processes (PIDs 33260, 40240 from 18:20:33) immediately restored expected throughput. Subsequent prompts completed in normal time.

---

## `AttributeError: 'Gemma4Model' object has no attribute 'layers'` in train_sae_local.py

**Symptom:** `scripts/train_sae_local.py` crashes immediately after model loading with `AttributeError: 'Gemma4Model' object has no attribute 'layers'` (line 91).

**Root cause:** `Gemma4ForConditionalGeneration` has a nested architecture: `model.model` returns a `Gemma4Model` (the multimodal wrapper), which does NOT directly expose `.layers`. The text transformer layers live at `model.model.language_model.layers`. The original code only checked `hasattr(model, "language_model")`, which is False for `Gemma4ForConditionalGeneration` (whose top-level attr is `.model`, not `.language_model`).

**Fix (in train_sae_local.py lines 87–95):** Walk multiple candidate paths:
```python
def _find_layer(m, idx):
    paths = [
        lambda m: m.model.language_model.layers[idx],  # Gemma4ForConditionalGeneration
        lambda m: m.language_model.model.layers[idx],  # alt multimodal
        lambda m: m.model.layers[idx],                 # standard CausalLM
        lambda m: m.model.model.layers[idx],           # double-wrapped
    ]
    for fn in paths:
        try:
            return fn(m)
        except (AttributeError, IndexError, TypeError):
            continue
    raise AttributeError(f"Cannot find layer {idx} in {type(m).__name__}")
target_layer = _find_layer(model, layer)
```

**Occurred:** 2026-04-24, first SAE training attempt.
