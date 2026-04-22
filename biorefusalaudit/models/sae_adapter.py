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


def load_sae(source: str, repo_or_path: str, layer: int, **kwargs) -> LoadedSAE:
    """Dispatch loader based on source tag.

    source ∈ {"gemma_scope_1", "gemma_scope_2", "llama_scope", "custom"}.
    For sae_lens-indexed sources we import sae_lens lazily.
    """
    if source in {"gemma_scope_1", "gemma_scope_2", "llama_scope"}:
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
    """
    sae.sae_module.eval()
    with torch.no_grad():
        z = sae.sae_module.encode(activations)
        # Mean over sequence positions; this is the "typical activation per feature for this example".
        mean_z = z.mean(dim=1)
    return mean_z.detach().cpu().float().numpy()
