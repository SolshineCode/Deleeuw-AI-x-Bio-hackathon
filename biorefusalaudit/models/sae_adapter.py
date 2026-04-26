"""SAE loader: supports sae_lens-indexed SAEs (Gemma Scope 1, Llama Scope)
and custom state_dict SAEs (Solshine Gemma 4 E2B, Secret Agenda releases).

Gemma Scope 2 (for Gemma 3) is not yet publicly released as of 2026-04-22;
this adapter will dispatch to the sae_lens path when it becomes available.
In the interim the methodology is demonstrated on:
  - Gemma 2 2B + Gemma Scope 1 residual SAEs (google/gemma-scope-2b-pt-res)
  - Gemma 4 E2B + Solshine custom TopK SAEs (Solshine/gemma4_e2b_*)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LoadedSAE:
    source: str
    name: str
    d_model: int
    d_sae: int
    architecture: str  # "jumprelu" | "topk" | "relu"
    hook_layer: int
    sae_module: nn.Module


class TopKSAE(nn.Module):
    """Minimal TopK SAE: encode → ReLU → top-k keep → decode."""

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        pre_relu = torch.relu(pre)
        if self.k >= self.d_sae:
            return pre_relu
        topk_vals, topk_idx = torch.topk(pre_relu, self.k, dim=-1)
        out = torch.zeros_like(pre_relu)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE (Gemma Scope family): encode → pre-activation → step above threshold."""

    def __init__(self, d_model: int, d_sae: int, threshold: torch.Tensor | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        if threshold is None:
            threshold = torch.ones(d_sae) * 0.05
        self.threshold = nn.Parameter(threshold, requires_grad=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        mask = (pre > self.threshold).float()
        return pre * mask

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, self.decode(z)


class _NullSAE(nn.Module):
    """Placeholder for cross-architecture runs with no community SAE.

    Returns a (batch, 1) zero feature vector so D=1.0 throughout.
    Judge-based compliance labels (comply/refuse/hedge) are still computed.
    Use when no SAE is available for the target model architecture.
    """

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1, dtype=torch.float32, device=x.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return z, x


def load_sae_from_state_dict(
    path: str | Path, architecture: str, d_model: int, d_sae: int, k: int | None = None
) -> nn.Module:
    """Load a custom SAE from a .pt or .safetensors file.

    Handles three weight layouts:
    - nn.Parameter format: keys W_enc (d_model, d_sae), b_enc, W_dec (d_sae, d_model), b_dec
    - nn.Linear format:   keys W_enc.weight (d_sae, d_model), W_enc.bias,
                               W_dec.weight (d_model, d_sae), W_dec.bias
      nn.Linear stores weight as (out, in), i.e. transposed relative to the matmul
      convention used by nn.Parameter SAEs. Colab-trained checkpoints use this layout.
    - encoder/decoder format: keys encoder.weight (d_sae, d_model), encoder.bias,
                               decoder.weight (d_model, d_sae), decoder.bias
      Used by EleutherAI sparsify SAEs and qresearch community SAEs.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if architecture == "topk":
        if k is None:
            raise ValueError("topk architecture requires k")
        sae = TopKSAE(d_model, d_sae, k)
    elif architecture in ("jumprelu", "relu"):
        # "relu" = L1-trained ReLU SAE (no learned threshold); load as JumpReLU threshold=0
        threshold = state.get("threshold", None)
        if threshold is None and architecture == "relu":
            threshold = torch.zeros(d_sae)
        sae = JumpReLUSAE(d_model, d_sae, threshold=threshold)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    if "W_enc.weight" in state:
        # nn.Linear layout — transpose weights to match nn.Parameter convention.
        # W_enc.weight: (d_sae, d_model) → adapter W_enc: (d_model, d_sae)
        # W_dec.weight: (d_model, d_sae) → adapter W_dec: (d_sae, d_model)
        sae.W_enc.data.copy_(state["W_enc.weight"].T.float())
        sae.b_enc.data.copy_(state["W_enc.bias"].float())
        sae.W_dec.data.copy_(state["W_dec.weight"].T.float())
        sae.b_dec.data.copy_(state["W_dec.bias"].float())
    elif "encoder.weight" in state:
        # EleutherAI/qresearch encoder/decoder format.
        # encoder.weight: (d_sae, d_model) → W_enc: (d_model, d_sae)
        # decoder.weight: (d_model, d_sae) → W_dec: (d_sae, d_model)
        sae.W_enc.data.copy_(state["encoder.weight"].T.float())
        sae.b_enc.data.copy_(state["encoder.bias"].float())
        sae.W_dec.data.copy_(state["decoder.weight"].T.float())
        sae.b_dec.data.copy_(state["decoder.bias"].float())
    else:
        missing_keys = []
        for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
            if key in state:
                getattr(sae, key).data.copy_(state[key])
            else:
                missing_keys.append(key)
        if missing_keys:
            raise ValueError(f"state dict missing keys: {missing_keys}")

    sae.eval()
    return sae


def _load_llama_scope_direct(repo_id: str, sae_path: str, layer: int) -> LoadedSAE:
    """Direct loader for Llama Scope JumpReLU SAEs from fnlp/Llama3_1-8B-Base-LXR-Nx.

    Bypasses sae_lens (segfaults on torch 2.6 + Py 3.13 on Windows).
    Weight file: <repo_id> / <sae_path>/checkpoints/final.safetensors
    Keys: encoder.weight (d_sae, d_model), decoder.weight (d_model, d_sae),
          encoder.bias (d_sae), decoder.bias (d_model).
    Architecture: JumpReLU with threshold=ones (≈ pre * step(pre > 1.0)).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sf_path = hf_hub_download(
        repo_id=repo_id,
        filename="final.safetensors",
        subfolder=f"{sae_path}/checkpoints",
    )
    sd = load_file(sf_path, device="cpu")
    # encoder.weight shape: (d_sae, d_model) — transpose to (d_model, d_sae) for W_enc
    W_enc = sd["encoder.weight"].T.float()   # (d_model, d_sae)
    W_dec = sd["decoder.weight"].T.float()   # (d_sae, d_model)
    b_enc = sd["encoder.bias"].float()       # (d_sae,)
    b_dec = sd["decoder.bias"].float()       # (d_model,)
    d_model, d_sae = W_enc.shape
    threshold = torch.ones(d_sae)

    sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae, threshold=threshold)
    sae.W_enc.data.copy_(W_enc)
    sae.W_dec.data.copy_(W_dec)
    sae.b_enc.data.copy_(b_enc)
    sae.b_dec.data.copy_(b_dec)
    sae.eval()
    return LoadedSAE(
        source="llama_scope_direct",
        name=f"{repo_id}/{sae_path}",
        d_model=d_model,
        d_sae=d_sae,
        architecture="jumprelu",
        hook_layer=layer,
        sae_module=sae,
    )


def _load_gemma_scope_direct(release: str, sae_id: str, layer: int) -> LoadedSAE:
    """Bypass sae_lens and load a Gemma Scope JumpReLU SAE directly from HF.

    Motivated by segfaults on `import sae_lens` under torch 2.6 + Python 3.13.
    Gemma Scope SAEs publish as .npz files keyed {W_enc, W_dec, b_enc, b_dec, threshold}.
    """
    from huggingface_hub import hf_hub_download

    npz_path = hf_hub_download(repo_id=f"google/{release}", filename=f"{sae_id}/params.npz")
    data = np.load(npz_path)
    d_model, d_sae = data["W_enc"].shape  # (2304, 16384) for 2B at width_16k
    sae = JumpReLUSAE(
        d_model=int(d_model),
        d_sae=int(d_sae),
        threshold=torch.from_numpy(data["threshold"]),
    )
    sae.W_enc.data.copy_(torch.from_numpy(data["W_enc"]))
    sae.W_dec.data.copy_(torch.from_numpy(data["W_dec"]))
    sae.b_enc.data.copy_(torch.from_numpy(data["b_enc"]))
    sae.b_dec.data.copy_(torch.from_numpy(data["b_dec"]))
    sae.eval()
    return LoadedSAE(
        source="gemma_scope_direct",
        name=f"{release}/{sae_id}",
        d_model=int(d_model),
        d_sae=int(d_sae),
        architecture="jumprelu",
        hook_layer=layer,
        sae_module=sae,
    )


def _load_dict_learning_direct(repo_id: str, folder_name: str, layer: int) -> LoadedSAE:
    """Direct loader for dictionary-learning SAEs (e.g. andyrdt/saes-qwen2.5-7b-instruct).

    File: <repo_id>/<folder_name>/ae.pt
    Keys: W_enc (d_model, d_sae) or encoder.weight.T, W_dec (d_sae, d_model),
          b_dec (d_model), optionally b_enc (d_sae).
    Architecture: ReLU (standard dictionary learning).
    """
    from huggingface_hub import hf_hub_download

    pt_path = hf_hub_download(repo_id=repo_id, filename=f"{folder_name}/ae.pt")
    sd = torch.load(pt_path, map_location="cpu")

    W_enc = sd.get("W_enc", sd.get("encoder.weight", None))
    if W_enc is None:
        raise KeyError(f"Could not find W_enc in {repo_id}/{folder_name}/ae.pt keys: {list(sd.keys())}")
    if "encoder.weight" in sd and "W_enc" not in sd:
        W_enc = sd["encoder.weight"].T  # (d_sae, d_model) → (d_model, d_sae)
    W_enc = W_enc.float()

    W_dec = sd.get("W_dec", sd.get("decoder.weight", None))
    if W_dec is None:
        raise KeyError(f"Could not find W_dec in ae.pt keys: {list(sd.keys())}")
    if "decoder.weight" in sd and "W_dec" not in sd:
        W_dec = sd["decoder.weight"].T
    W_dec = W_dec.float()

    d_model, d_sae = W_enc.shape
    b_enc_raw = sd.get("b_enc", sd.get("encoder.bias", torch.zeros(d_sae)))
    b_dec_raw = sd.get("b_dec", sd.get("bias", sd.get("decoder_bias", torch.zeros(d_model))))

    # Use standard JumpReLU with threshold=0 (≡ ReLU) for dict-learning SAEs
    sae = JumpReLUSAE(d_model=d_model, d_sae=d_sae, threshold=torch.zeros(d_sae))
    sae.W_enc.data.copy_(W_enc)
    sae.W_dec.data.copy_(W_dec)
    sae.b_enc.data.copy_(b_enc_raw.float())
    sae.b_dec.data.copy_(b_dec_raw.float())
    sae.eval()
    return LoadedSAE(
        source="dict_learning_direct",
        name=f"{repo_id}/{folder_name}",
        d_model=d_model,
        d_sae=d_sae,
        architecture="jumprelu",
        hook_layer=layer,
        sae_module=sae,
    )


def load_sae(source: str, repo_or_path: str, layer: int, **kwargs) -> LoadedSAE:
    """Dispatch loader based on source tag.

    source ∈ {"gemma_scope_1", "gemma_scope_2", "llama_scope", "custom"}.
    For sae_lens-indexed sources we try sae_lens first and fall back to a
    direct HF .npz loader when sae_lens is unusable (segfaults on import
    under torch 2.6 + Py 3.13 were observed on Windows 2026-04-22).
    """
    if source == "none":
        return LoadedSAE(
            source="none",
            name="null-sae",
            d_model=1,
            d_sae=1,
            architecture="none",
            hook_layer=layer,
            sae_module=_NullSAE(),
        )
    if source in {"gemma_scope_1", "gemma_scope_2"}:
        # Direct HF .npz loader — bypasses sae_lens entirely.
        # sae_lens 6.39.0 segfaults on import under torch 2.6 + Py 3.13 on
        # Windows (observed 2026-04-22). A segfault is a signal, not an
        # exception, so try/except around `import sae_lens` doesn't help —
        # the process dies before any fallback runs. Use the direct loader
        # for Gemma Scope unconditionally until the ABI break is resolved.
        sae_id = kwargs.get("sae_id", "")
        return _load_gemma_scope_direct(repo_or_path, sae_id, layer)
    if source == "llama_scope":
        # Use direct HF loader — sae_lens segfaults on torch 2.6 + Py 3.13 on Windows.
        # repo_or_path = HF repo id (e.g. "fnlp/Llama3_1-8B-Base-LXR-8x")
        # kwargs["sae_id"] = the subfolder path (e.g. "Llama3_1-8B-Base-L16R-8x")
        sae_path = kwargs.get("sae_id", "")
        return _load_llama_scope_direct(repo_or_path, sae_path, layer)
    if source == "dict_learning":
        # andyrdt/saes-qwen2.5-7b-instruct style; folder_name = SAE subfolder
        folder = kwargs.get("sae_id", "")
        return _load_dict_learning_direct(repo_or_path, folder, layer)
    if source == "custom":
        state_path = Path(repo_or_path)
        if not state_path.exists() and "/" in repo_or_path:
            from huggingface_hub import HfApi, hf_hub_download
            # Try common filenames or search the repo for a .pt/.safetensors file
            try:
                state_path = Path(hf_hub_download(repo_id=repo_or_path, filename="sae_weights.pt"))
            except Exception:
                try:
                    state_path = Path(hf_hub_download(repo_id=repo_or_path, filename="sae_weights.safetensors"))
                except Exception:
                    # Search the repo for any .pt or .safetensors file
                    api = HfApi()
                    files = api.list_repo_files(repo_id=repo_or_path)
                    weight_files = [f for f in files if f.endswith(".pt") or f.endswith(".safetensors")]
                    if weight_files:
                        state_path = Path(hf_hub_download(repo_id=repo_or_path, filename=weight_files[0]))
                    else:
                        raise RuntimeError(f"No .pt or .safetensors file found in repo {repo_or_path}")
        
        sae = load_sae_from_state_dict(
            state_path,
            architecture=kwargs["architecture"],
            d_model=kwargs["d_model"],
            d_sae=kwargs["d_sae"],
            k=kwargs.get("k"),
        )
        return LoadedSAE(
            source=source,
            name=str(state_path),
            d_model=kwargs["d_model"],
            d_sae=kwargs["d_sae"],
            architecture=kwargs["architecture"],
            hook_layer=layer,
            sae_module=sae,
        )
    raise ValueError(f"unknown SAE source: {source}")


def project_activations(sae: LoadedSAE, activations: torch.Tensor) -> np.ndarray:
    """Encode activations through the SAE; return per-example mean feature magnitudes.

    `activations`: (batch, seq_len, d_model). Returns np.ndarray of shape (batch, d_sae).
    Moves the SAE module to the activations' device on first call to avoid
    device-mismatch errors when the model is on GPU and the SAE loaded on CPU.
    """
    sae.sae_module.eval()
    # Match device + dtype to the incoming activations (one-time move, idempotent).
    target_device = activations.device
    target_dtype = activations.dtype
    try:
        sae_device = next(sae.sae_module.parameters()).device
    except StopIteration:
        sae_device = torch.device("cpu")
    if sae_device != target_device:
        sae.sae_module.to(target_device)
    with torch.no_grad():
        # SAE weights are fp32; cast activations to fp32 for the matmul.
        z = sae.sae_module.encode(activations.to(torch.float32))
        # Mean over sequence positions; "typical activation per feature for this example".
        mean_z = z.mean(dim=1)
    return mean_z.detach().cpu().float().numpy()
