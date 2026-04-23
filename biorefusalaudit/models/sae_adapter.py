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


def load_sae_from_state_dict(
    path: str | Path, architecture: str, d_model: int, d_sae: int, k: int | None = None
) -> nn.Module:
    """Load a custom SAE from a .pt or .safetensors file."""
    state = torch.load(str(path), map_location="cpu")
    if architecture == "topk":
        if k is None:
            raise ValueError("topk architecture requires k")
        sae = TopKSAE(d_model, d_sae, k)
    elif architecture == "jumprelu":
        threshold = state.get("threshold", None)
        sae = JumpReLUSAE(d_model, d_sae, threshold=threshold)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

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


def load_sae(source: str, repo_or_path: str, layer: int, **kwargs) -> LoadedSAE:
    """Dispatch loader based on source tag.

    source ∈ {"gemma_scope_1", "gemma_scope_2", "llama_scope", "custom"}.
    For sae_lens-indexed sources we try sae_lens first and fall back to a
    direct HF .npz loader when sae_lens is unusable (segfaults on import
    under torch 2.6 + Py 3.13 were observed on Windows 2026-04-22).
    """
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
        from sae_lens import SAE  # type: ignore

        sae, cfg, _ = SAE.from_pretrained(release=repo_or_path, sae_id=kwargs.get("sae_id", ""))
        return LoadedSAE(
            source=source,
            name=repo_or_path,
            d_model=cfg["d_in"],
            d_sae=cfg["d_sae"],
            architecture=cfg.get("architecture", "jumprelu"),
            hook_layer=layer,
            sae_module=sae,
        )
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
