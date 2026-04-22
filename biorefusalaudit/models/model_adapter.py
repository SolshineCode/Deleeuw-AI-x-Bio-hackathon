"""HuggingFace Transformers model loader with residual-stream activation hook.

Supports:
  - fp16 loading for small models (Gemma 3 270M, Gemma 2 2B on <8GB)
  - bitsandbytes 4-bit for larger models (Gemma 4 E2B, 4B+ on 4GB GPU)

Returns (model, tokenizer, hook_installer) where hook_installer is a callable
that registers a forward hook on the target transformer layer and returns
a context manager exposing the captured residual-stream tensor.

This is the inference-time hook path. For activation bulk collection use
`collect_residual_activations` which wraps it in a simple loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterator

import torch


@dataclass
class LoadedModel:
    name: str
    model: "torch.nn.Module"
    tokenizer: object
    device: str
    quantize: str | None  # None | "4bit" | "8bit"


def load_model(
    name: str,
    quantize: str | None = None,
    device: str | None = None,
    dtype: str = "float16",
) -> LoadedModel:
    """Load an HF model + tokenizer. Quantize via bitsandbytes if requested."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs: dict = {}
    torch_dtype = getattr(torch, dtype) if dtype != "auto" else None
    if quantize == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    elif quantize == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch_dtype
        kwargs["device_map"] = device

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, device=device, quantize=quantize)


def _resolve_block(model: "torch.nn.Module", layer_idx: int) -> "torch.nn.Module":
    """Best-effort resolver for the target transformer block across HF model families."""
    # Most HF causal LMs expose `.model.layers[i]` (Llama, Gemma, Mistral, Qwen)
    # or `.transformer.h[i]` (GPT-2 style, some Gemma variants).
    for attrs in (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("base_model", "layers"),
    ):
        obj = model
        try:
            for a in attrs:
                obj = getattr(obj, a)
            return obj[layer_idx]
        except (AttributeError, IndexError):
            continue
    raise ValueError(
        f"Could not resolve layer {layer_idx} on {type(model).__name__}; "
        "unknown transformer block attribute path."
    )


@contextlib.contextmanager
def residual_stream_hook(
    model: "torch.nn.Module", layer_idx: int
) -> Iterator[Callable[[], torch.Tensor | None]]:
    """Context manager capturing the residual-stream output of block `layer_idx`.

    Usage:
        with residual_stream_hook(lm.model, 14) as get_resid:
            lm.model(**inputs)
            resid = get_resid()  # shape (batch, seq_len, d_model)
    """
    block = _resolve_block(model, layer_idx)
    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        # Transformer block outputs vary: some return tensor directly, others (tensor,) or (tensor, kv).
        if isinstance(output, tuple):
            resid = output[0]
        else:
            resid = output
        captured.append(resid.detach())

    handle = block.register_forward_hook(_hook)

    def getter() -> torch.Tensor | None:
        return captured[-1] if captured else None

    try:
        yield getter
    finally:
        handle.remove()


def generate_completion(
    lm: LoadedModel, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7
) -> str:
    """Generate a model completion for a single prompt. Returns the decoded text."""
    enc = lm.tokenizer(prompt, return_tensors="pt").to(next(lm.model.parameters()).device)
    with torch.no_grad():
        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=lm.tokenizer.eos_token_id,
        )
    full = lm.tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix if the tokenizer didn't do it for us.
    if full.startswith(prompt):
        full = full[len(prompt):]
    return full.strip()
