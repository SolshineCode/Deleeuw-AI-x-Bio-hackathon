"""Local proof-of-concept SAE fine-tuning on bio-safety activations.

Collects raw residual-stream activations from frozen Gemma 2/4 at a specified layer,
then trains a TopK(k=32) SAE with a contrastive objective that separates
hazard-adjacent from benign biological activations.

Usage:
    # Fresh run:
    python scripts/train_sae_local.py \
        --model google/gemma-2-2b-it \
        --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
        --out runs/sae-training-local \
        --layer 12 --steps 5000 --batch-size 8

    # Warm start from existing checkpoint with stronger contrastive signal:
    python scripts/train_sae_local.py \
        --model google/gemma-2-2b-it \
        --eval-set data/eval_set_public/eval_set_public_v1.jsonl \
        --out runs/sae-training-local-continued \
        --layer 12 --steps 5000 --lam-contrast 0.5 \
        --init-from runs/sae-training-gemma2-5000steps/sae_weights.pt \
        --contrastive-mode pairwise

Outputs:
    <out>/sae_weights.pt              -- final checkpoint
    <out>/training_log.jsonl          -- per-step metrics
    <out>/checkpoint_<step>.pt        -- intermediate checkpoints (if --checkpoint-every set)

Changelog:
    2026-04-25: Added --init-from (warm start), --checkpoint-every, --contrastive-mode
                (mean|pairwise|triplet), --temperature. Backward compatible with all
                existing args. Pairwise mode uses NT-Xent-style per-sample cross-tier
                cosine similarity (better gradient landscape than mean-vs-mean for
                small corpora). Triplet mode adds within-tier cohesion term.
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
# TopKSAE
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

def collect_activations(prompts, model, tokenizer, layer, device):
    class Hook:
        def __init__(self): self.last = None
        def __call__(self, m, inp, out):
            self.last = (out[0] if isinstance(out, tuple) else out).detach()

    hook_obj = Hook()

    def _find_layer(m, idx):
        paths = [
            lambda m: m.model.language_model.layers[idx],
            lambda m: m.language_model.model.layers[idx],
            lambda m: m.model.layers[idx],
            lambda m: m.model.model.layers[idx],
        ]
        for fn in paths:
            try:
                return fn(m)
            except (AttributeError, IndexError, TypeError):
                continue
        for name, mod in m.named_modules():
            if "layers" in name and name.endswith(f".{idx}"):
                return mod
        raise AttributeError(f"Cannot find layer {idx} in {type(m).__name__}")

    target_layer = _find_layer(model, layer)
    handle = target_layer.register_forward_hook(hook_obj)

    acts_by_id = {}
    for item in prompts:
        msgs = [{"role": "user", "content": item["prompt"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)
        mean_act = hook_obj.last[0].mean(dim=0).float().cpu()
        acts_by_id[item["id"]] = (mean_act, item.get("tier", "benign_bio"))

    handle.remove()
    return acts_by_id


# ---------------------------------------------------------------------------
# Contrastive loss variants
# ---------------------------------------------------------------------------

def contrastive_mean(z_hazard, z_benign):
    """Original: minimize cosine similarity of tier centroid vectors."""
    if z_hazard.size(0) == 0 or z_benign.size(0) == 0:
        return torch.tensor(0.0)
    m_h = z_hazard.mean(0)
    m_b = z_benign.mean(0)
    return F.cosine_similarity(m_h.unsqueeze(0), m_b.unsqueeze(0)).mean()


def contrastive_pairwise(z_hazard, z_benign, temperature=0.1):
    """NT-Xent style: average cosine similarity across all cross-tier pairs.

    Operates per-sample rather than on centroids, giving better gradients when
    the centroid similarity is near 1 (shared vocabulary case). Minimizing this
    forces each hazard sample to be dissimilar to each benign sample in feature
    space, not just dissimilar on average.
    """
    if z_hazard.size(0) == 0 or z_benign.size(0) == 0:
        return torch.tensor(0.0)
    zh = F.normalize(z_hazard.float(), dim=-1)  # (n_h, d_sae)
    zb = F.normalize(z_benign.float(), dim=-1)  # (n_b, d_sae)
    cross_sim = zh @ zb.T  # (n_h, n_b) — all pairwise cosine sims
    return cross_sim.mean()  # minimize: push all hazard samples away from all benign


def contrastive_triplet(z_hazard, z_benign):
    """Triplet-style: cross-tier push + within-tier pull.

    Minimizes avg(cos_sim(hazard, benign)) while minimizing negative of
    avg(cos_sim(hazard_i, hazard_j)) to maintain hazard cluster cohesion.
    Net effect: push classes apart while keeping each class together.
    """
    cross = contrastive_pairwise(z_hazard, z_benign)
    # Within-tier cohesion: we want hazard samples to be similar to each other.
    # So minimize -cos_sim(hazard_i, hazard_j) = maximize cohesion.
    within_loss = torch.tensor(0.0)
    for z in [z_hazard, z_benign]:
        if z.size(0) > 1:
            zn = F.normalize(z.float(), dim=-1)
            within_sim = (zn @ zn.T)
            # zero diagonal before averaging
            mask = ~torch.eye(zn.size(0), dtype=torch.bool, device=z.device)
            within_sim = within_sim[mask].mean()
            within_loss = within_loss + (1.0 - within_sim)  # want high within-sim
    return cross + 0.3 * within_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(l) for l in Path(args.eval_set).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.eval_set}")

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
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
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

    print("Collecting residual activations...")
    t0 = time.time()
    acts_by_id = collect_activations(prompts, model, tokenizer, args.layer, device)
    print(f"Collected {len(acts_by_id)} activation vectors in {time.time()-t0:.1f}s")

    act_vecs, tiers = [], []
    for pid, (vec, tier) in acts_by_id.items():
        act_vecs.append(vec)
        tiers.append(tier)
    X = torch.stack(act_vecs)
    tiers_arr = np.array(tiers)

    hazard_mask = np.isin(tiers_arr, ["hazard_adjacent_category"])
    benign_mask  = np.isin(tiers_arr, ["benign_bio"])
    print(f"Tier breakdown: hazard={hazard_mask.sum()}, benign={benign_mask.sum()}, dual_use={len(tiers)-hazard_mask.sum()-benign_mask.sum()}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model freed from VRAM.")

    d_model = X.shape[1]
    sae = TopKSAE(d_model, args.d_sae, args.k).to(device).float()

    if args.init_from:
        ckpt_path = Path(args.init_from)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location=device)
            sae.load_state_dict(state)
            print(f"Warm-started from {ckpt_path}")
        else:
            print(f"WARNING: --init-from path not found: {ckpt_path}. Starting fresh.")

    optimizer = torch.optim.AdamW(sae.parameters(), lr=args.lr)
    print(f"SAE: d_model={d_model}, d_sae={args.d_sae}, k={args.k}")
    print(f"Contrastive mode: {args.contrastive_mode}, lam_contrast={args.lam_contrast}, temperature={args.temperature}")

    log_path = out_dir / "training_log.jsonl"
    txt_path = out_dir / "training_log.txt"
    log_fh = open(log_path, "w", encoding="utf-8")
    txt_fh = open(txt_path, "w", encoding="utf-8")

    X_gpu = X.to(device)
    N = len(X_gpu)
    hz_idx = torch.where(torch.tensor(hazard_mask))[0]
    bn_idx = torch.where(torch.tensor(benign_mask))[0]

    contrast_fn = {
        "mean":     lambda zh, zb: contrastive_mean(zh, zb),
        "pairwise": lambda zh, zb: contrastive_pairwise(zh, zb, args.temperature),
        "triplet":  lambda zh, zb: contrastive_triplet(zh, zb),
    }[args.contrastive_mode]

    print(f"Training for {args.steps} steps, batch_size={args.batch_size}...")
    t_start = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch_size,))
        x_batch = X_gpu[idx]

        x_hat, z, pre = sae(x_batch)
        l_recon = F.mse_loss(x_hat, x_batch)
        l_sparse = pre.abs().mean()

        z_hz = sae.encode(X_gpu[hz_idx]) if len(hz_idx) else torch.zeros(0, args.d_sae, device=device)
        z_bn = sae.encode(X_gpu[bn_idx]) if len(bn_idx) else torch.zeros(0, args.d_sae, device=device)
        l_contrast = contrast_fn(z_hz, z_bn).to(device)

        total = l_recon + args.lam_sparse * l_sparse + args.lam_contrast * l_contrast

        optimizer.zero_grad()
        total.backward()
        sae.project_grad()
        optimizer.step()
        sae.normalize_decoder()

        l0 = (z > 0).float().sum(-1).mean().item()

        if step % 50 == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            line = (f"step={step:5d}  loss={total.item():.4f}  l_recon={l_recon.item():.4f}  "
                    f"l_sparse={l_sparse.item():.4f}  l_contrast={l_contrast.item():.4f}  "
                    f"L0={l0:.1f}  t={elapsed:.0f}s")
            print(line)
            txt_fh.write(line + "\n")
            txt_fh.flush()

        log_fh.write(json.dumps({
            "step": step, "total_loss": total.item(), "l_recon": l_recon.item(),
            "l_sparsity": l_sparse.item(), "l_contrastive": l_contrast.item(), "l0": l0,
        }) + "\n")

        if args.checkpoint_every and step > 0 and step % args.checkpoint_every == 0:
            ckpt = out_dir / f"checkpoint_{step:05d}.pt"
            torch.save(sae.state_dict(), ckpt)
            print(f"  [checkpoint] saved {ckpt}")

    log_fh.close()
    txt_fh.close()
    ckpt = out_dir / "sae_weights.pt"
    torch.save(sae.state_dict(), ckpt)
    print(f"Checkpoint saved: {ckpt}")
    print(f"Training log: {log_path}")
    print(f"Total wall clock: {time.time()-t_start:.0f}s")

    # Run analysis script if available
    analysis_script = Path("scripts/analyze_sae_training.py")
    if analysis_script.exists():
        import subprocess
        subprocess.run(["python", str(analysis_script), str(log_path)], check=False)


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
    # New in 2026-04-25:
    ap.add_argument("--init-from", default=None,
                    help="Path to .pt checkpoint to warm-start from (resumes training from these weights)")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="Save intermediate checkpoint every N steps (0 = disabled)")
    ap.add_argument("--contrastive-mode", default="mean",
                    choices=["mean", "pairwise", "triplet"],
                    help="mean: centroid cosine sim (original); pairwise: NT-Xent all pairs; triplet: pairwise + within-tier cohesion")
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Temperature for pairwise contrastive mode")
    args = ap.parse_args()

    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    train(args)


if __name__ == "__main__":
    main()
