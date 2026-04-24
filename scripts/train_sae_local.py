"""Local proof-of-concept SAE fine-tuning on bio-safety activations.

Collects raw residual-stream activations from frozen Gemma 4 E2B-IT at layer 17,
then trains a TopK(k=32) SAE with a contrastive objective that separates
hazard-adjacent from benign biological activations.

Usage:
    python scripts/train_sae_local.py \
        --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
        --out runs/sae-training-local \
        --steps 500 --batch-size 4

Outputs:
    runs/sae-training-local/sae_weights.pt     -- final checkpoint
    runs/sae-training-local/training_log.jsonl -- per-step metrics
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# TopK SAE (identical to Colab notebook / sae_adapter.py)
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) * 0.02)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_model) * 0.02)
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.normalize_decoder()

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

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        pre = x @ self.W_enc + self.b_enc
        return x_hat, z, pre

    @torch.no_grad()
    def normalize_decoder(self):
        n = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(n)

    @torch.no_grad()
    def project_grad(self):
        if self.W_dec.grad is None:
            return
        g, w = self.W_dec.grad, self.W_dec.data
        g.sub_((g * w).sum(dim=1, keepdim=True) * w)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

def collect_activations(prompts: list[dict], model, tokenizer, layer: int, device: str) -> dict[str, torch.Tensor]:
    """Run each prompt through frozen model and return per-prompt mean residual vector."""
    class Hook:
        def __init__(self): self.last = None
        def __call__(self, m, inp, out):
            self.last = (out[0] if isinstance(out, tuple) else out).detach()

    hook_obj = Hook()

    # Walk several known architecture paths to locate transformer layers.
    # Gemma 4 ForConditionalGeneration: model.model → Gemma4Model → .language_model → Gemma4TextModel → .layers
    # Gemma 4 via language_model attr directly:  model.language_model.model.layers
    # Standard CausalLM:  model.model.layers
    def _find_layer(m, idx):
        paths = [
            lambda m: m.model.language_model.layers[idx],      # Gemma4ForConditionalGeneration
            lambda m: m.language_model.model.layers[idx],      # alt multimodal
            lambda m: m.model.layers[idx],                     # standard CausalLM
            lambda m: m.model.model.layers[idx],               # double-wrapped
        ]
        for fn in paths:
            try:
                return fn(m)
            except (AttributeError, IndexError, TypeError):
                continue
        # Last resort: named_modules scan
        for name, mod in m.named_modules():
            if "layers" in name and name.endswith(f".{idx}"):
                return mod
        raise AttributeError(f"Cannot find layer {idx} in {type(m).__name__}")
    target_layer = _find_layer(model, layer)

    handle = target_layer.register_forward_hook(hook_obj)

    acts_by_id: dict[str, torch.Tensor] = {}
    for item in prompts:
        prompt_id = item["id"]
        tier = item.get("tier", "benign_bio")

        # Apply Gemma 4 chat template
        msgs = [{"role": "user", "content": item["prompt"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            model(**inputs)

        # Mean-pool over sequence positions, move to cpu float32
        mean_act = hook_obj.last[0].mean(dim=0).float().cpu()  # (d_model,)
        acts_by_id[prompt_id] = (mean_act, tier)

    handle.remove()
    return acts_by_id


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_contrastive_loss(z_hazard: torch.Tensor, z_benign: torch.Tensor) -> torch.Tensor:
    if z_hazard.size(0) == 0 or z_benign.size(0) == 0:
        return torch.tensor(0.0)
    m_h = z_hazard.mean(0)
    m_b = z_benign.mean(0)
    return F.cosine_similarity(m_h.unsqueeze(0), m_b.unsqueeze(0)).mean()


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load eval set ---
    prompts = [json.loads(l) for l in Path(args.eval_set).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.eval_set}")

    # --- Load frozen model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    ) if device == "cuda" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model}...")
    load_kwargs = dict(low_cpu_mem_usage=True)
    if bnb_cfg:
        load_kwargs["quantization_config"] = bnb_cfg
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = "cpu"

    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- Collect activations (one-time pass through the eval set) ---
    print("Collecting residual activations...")
    t0 = time.time()
    acts_by_id = collect_activations(prompts, model, tokenizer, args.layer, device)
    print(f"Collected {len(acts_by_id)} activation vectors in {time.time()-t0:.1f}s")

    # Separate by tier
    act_vecs = []
    tiers = []
    for pid, (vec, tier) in acts_by_id.items():
        act_vecs.append(vec)
        tiers.append(tier)
    X = torch.stack(act_vecs)  # (N, d_model) on CPU float32
    tiers_arr = np.array(tiers)

    hazard_mask = np.isin(tiers_arr, ["hazard_adjacent_category"])
    benign_mask = np.isin(tiers_arr, ["benign_bio"])

    print(f"Tier breakdown: hazard={hazard_mask.sum()}, benign={benign_mask.sum()}, dual_use={len(tiers)-hazard_mask.sum()-benign_mask.sum()}")

    # Free model from VRAM before training
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model freed from VRAM.")

    # --- Initialize SAE ---
    d_model = X.shape[1]
    sae = TopKSAE(d_model, args.d_sae, args.k).to(device).float()
    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr)
    print(f"SAE: d_model={d_model}, d_sae={args.d_sae}, k={args.k}")

    # --- Training loop ---
    log_path = out_dir / "training_log.jsonl"
    log_fh = open(log_path, "w", encoding="utf-8")

    X_gpu = X.to(device)
    N = len(X_gpu)

    print(f"Training for {args.steps} steps, batch_size={args.batch_size}...")
    t_start = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,))
        x_batch = X_gpu[idx]

        x_hat, z, pre = sae(x_batch)
        l_recon = F.mse_loss(x_hat, x_batch)
        l_sparse = pre.abs().mean()

        # Contrastive: full-dataset tier means
        hz_idx = torch.where(torch.tensor(hazard_mask))[0]
        bn_idx = torch.where(torch.tensor(benign_mask))[0]
        z_hz = sae.encode(X_gpu[hz_idx]) if len(hz_idx) else torch.zeros(0, args.d_sae, device=device)
        z_bn = sae.encode(X_gpu[bn_idx]) if len(bn_idx) else torch.zeros(0, args.d_sae, device=device)
        l_contrast = compute_contrastive_loss(z_hz, z_bn).to(device)

        total = l_recon + args.lam_sparse * l_sparse + args.lam_contrast * l_contrast

        optimizer.zero_grad()
        total.backward()
        sae.project_grad()
        optimizer.step()
        sae.normalize_decoder()

        l0 = (z > 0).float().sum(-1).mean().item()

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"step={step:4d}  loss={total.item():.4f}  l_recon={l_recon.item():.4f}  "
                  f"l_sparse={l_sparse.item():.4f}  l_contrast={l_contrast.item():.4f}  L0={l0:.1f}  t={elapsed:.0f}s")

        record = {
            "step": step,
            "total_loss": total.item(),
            "l_recon": l_recon.item(),
            "l_sparsity": l_sparse.item(),
            "l_contrastive": l_contrast.item(),
            "l0": l0,
        }
        log_fh.write(json.dumps(record) + "\n")

    log_fh.close()
    ckpt = out_dir / "sae_weights.pt"
    torch.save(sae.state_dict(), ckpt)
    print(f"Checkpoint saved: {ckpt}")
    print(f"Training log: {log_path}")
    print(f"Total wall clock: {time.time()-t_start:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--eval-set", default="data/eval_set_public/eval_set_public_v1.jsonl")
    ap.add_argument("--out", default="runs/sae-training-local")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--d-sae", type=int, default=6144)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-sparse", type=float, default=0.04)
    ap.add_argument("--lam-contrast", type=float, default=0.1)
    args = ap.parse_args()

    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    train(args)


if __name__ == "__main__":
    main()
