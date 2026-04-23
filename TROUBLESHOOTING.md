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
