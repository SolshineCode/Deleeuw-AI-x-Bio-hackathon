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
        # Use integer device index (0) rather than string "cuda" — bitsandbytes on
        # Windows WDDM silently falls back to CPU when the string form is used and
        # VRAM is tight. Integer index forces the allocation correctly.
        cuda_idx = torch.cuda.current_device() if torch.cuda.is_available() else None
        kwargs["device_map"] = {"": cuda_idx} if cuda_idx is not None else "auto"
    elif quantize == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        cuda_idx = torch.cuda.current_device() if torch.cuda.is_available() else None
        kwargs["device_map"] = {"": cuda_idx} if cuda_idx is not None else "auto"
    else:
        kwargs["torch_dtype"] = torch_dtype
        # Prefer device_map="auto" on CUDA so models larger than VRAM
        # (e.g., Gemma 2 2B fp16 = 5.2GB on a 4GB GTX 1650) auto-offload
        # to CPU for the overflow instead of OOM-ing. Keeps GPU busy at
        # ~90% utilization on the on-device layers.
        kwargs["device_map"] = "auto" if device == "cuda" else device

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    # Verify actual device placement — catches silent CPU fallback from bitsandbytes on WDDM
    try:
        actual = next(model.parameters()).device
        print(f"[model_adapter] Loaded {name} on {actual} (requested: {device}, quantize: {quantize})", flush=True)
        if device == "cuda" and str(actual) == "cpu":
            raise RuntimeError(
                f"Model loaded on CPU despite CUDA request. "
                f"Free VRAM may be insufficient. Requested quantize={quantize}."
            )
    except StopIteration:
        pass  # quantized models may have no plain parameters — skip check
    return LoadedModel(name=name, model=model, tokenizer=tokenizer, device=device, quantize=quantize)


def _resolve_block(model: "torch.nn.Module", layer_idx: int) -> "torch.nn.Module":
    """Best-effort resolver for the target transformer block across HF model families."""
    # Most HF causal LMs expose `.model.layers[i]` (Llama, Gemma, Mistral, Qwen)
    # or `.transformer.h[i]` (GPT-2 style, some Gemma variants).
    # Gemma 4 uses .language_model.model.layers or .model.language_model.layers.
    for attrs in (
        ("model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
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
    model: "torch.nn.Module", layer_idx: int, capture_all: bool = False
) -> Iterator[Callable[[], torch.Tensor | list[torch.Tensor] | None]]:
    """Context manager capturing the residual-stream output of block `layer_idx`.

    If capture_all is True, returns a list of all tensors (one per step)
    offloaded to CPU to save VRAM. Otherwise returns only the final step tensor.

    Usage:
        with residual_stream_hook(lm.model, 14, capture_all=True) as get_resid:
            lm.model(**inputs)
            resids = get_resid()  # list[torch.Tensor] on CPU
    """
    block = _resolve_block(model, layer_idx)
    captured: list[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        # Transformer block outputs vary: some return tensor directly, others (tensor,) or (tensor, kv).
        if isinstance(output, tuple):
            resid = output[0]
        else:
            resid = output
        
        if capture_all:
            # Move to CPU immediately to avoid VRAM overflow on long generations
            captured.append(resid.detach().to("cpu"))
        else:
            # Overwrite rather than append: during autoregressive generation the hook fires
            # once per new token. Accumulating all steps fills VRAM (each tensor grows with
            # sequence length) and causes CPU spill on 4GB cards. We only need the final step.
            if captured:
                captured[0] = resid.detach()
            else:
                captured.append(resid.detach())

    handle = block.register_forward_hook(_hook)

    def getter() -> torch.Tensor | list[torch.Tensor] | None:
        if not captured:
            return None
        return captured if capture_all else captured[0]

    try:
        yield getter
    finally:
        handle.remove()


def _apply_chat_template(lm: LoadedModel, prompt: str) -> str:
    """Wrap prompt in the model's chat template if supported.

    Instruction-tuned models (Gemma 2 IT, Gemma 4 IT, Llama IT) expect
    their special BOS/turn tokens via apply_chat_template. Without this,
    the model sees raw text and either generates nothing (immediate EOS) or
    echoes the prompt in a repetition loop — both of which the regex judge
    classifies as 'refuse', producing false-positive refusal counts.

    Falls back to returning the raw prompt if the tokenizer has no template
    (e.g., base models, GPT-2 style).
    """
    tok = lm.tokenizer
    if getattr(tok, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def generate_completion(
    lm: LoadedModel, prompt: str, max_new_tokens: int = 200, temperature: float = 0.7
) -> str:
    """Generate a model completion for a single prompt. Returns the decoded text."""
    formatted = _apply_chat_template(lm, prompt)
    device = next(lm.model.parameters()).device
    enc = lm.tokenizer(formatted, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = lm.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=lm.tokenizer.eos_token_id,
        )
    # Decode only the *generated* tokens — slice off the input prefix ids.
    # This avoids string-stripping heuristics that break when the tokenizer's
    # skip_special_tokens=True drops template tokens but leaves plain-text
    # turn markers ("user\n", "model\n") in the decoded output.
    completion_ids = out[0][input_len:]
    return lm.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
