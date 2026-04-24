"""Collect multi-token SAE activations for Track B adapter training.

Captures the full residual stream sequence during generation, projects
through the SAE, and saves as a compressed .npz. Uses CPU offloading
to avoid VRAM overflow on long sequences.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from biorefusalaudit.models.model_adapter import (
    load_model,
    residual_stream_hook,
    _apply_chat_template,
)
from biorefusalaudit.models.sae_adapter import load_sae, project_activations
from biorefusalaudit.prompts.prompt_loader import load_jsonl

logging.basicConfig(level=logging.INFO, format="[collect] %(message)s")
log = logging.getLogger("collect")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--eval-set", default="data/eval_set_public/eval_set_public_v1.jsonl")
    ap.add_argument("--sae-release", default="gemma-scope-2b-pt-res")
    ap.add_argument("--sae-id", default="layer_12/width_16k/average_l0_82")
    ap.add_argument("--layer", default=12, type=int)
    ap.add_argument("--max-new-tokens", default=100, type=int)
    ap.add_argument("--limit", default=0, type=int)
    ap.add_argument("--out", default="runs/multitoken_activations.npz")
    args = ap.parse_args()

    log.info("Loading model: %s", args.model)
    lm = load_model(args.model, quantize="4bit")
    
    log.info("Loading SAE: %s / %s", args.sae_release, args.sae_id)
    sae = load_sae("gemma_scope_1", args.sae_release, layer=args.layer, sae_id=args.sae_id)

    prompts = load_jsonl(args.eval_set)
    if args.limit > 0:
        prompts = prompts[:args.limit]
    log.info("Loaded %d prompts", len(prompts))

    all_feats = []
    all_metadata = []

    for i, p in enumerate(prompts):
        log.info("[%d/%d] Processing %s", i+1, len(prompts), p.id)
        
        formatted = _apply_chat_template(lm, p.prompt)
        device = next(lm.model.parameters()).device
        enc = lm.tokenizer(formatted, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]

        with residual_stream_hook(lm.model, args.layer, capture_all=True) as get_resids:
            with torch.no_grad():
                out = lm.model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=lm.tokenizer.eos_token_id,
                )
            resids = get_resids() # list of tensors on CPU
        
        if not resids:
            log.warning("No residuals captured for %s", p.id)
            continue

        # Each resid in resids has shape (1, seq_len_at_step, d_model)
        # We only want the *last* token's residual from each step of generation
        # except for the first step which contains the prompt.
        # But wait, model.generate handles the loop. Each call to hook happens once per step.
        # In the first step (prompt processing), resid has shape (1, prompt_len, d_model).
        # In subsequent steps (token 1, 2, ...), resid has shape (1, 1, d_model).
        
        sequence_feats = []
        for j, r in enumerate(resids):
            # r is on CPU
            # project_activations expects numpy or torch tensor
            # it will internally handle moving to device if needed
            feats = project_activations(sae, r) # (1, seq_len_at_step, d_sae)
            
            if j == 0:
                # Prompt tokens: take all
                sequence_feats.append(feats[0]) # (prompt_len, d_sae)
            else:
                # Generated tokens: take only the last one
                sequence_feats.append(feats[0][-1:]) # (1, d_sae)
        
        # Stack into one (total_tokens, d_sae) array
        full_seq = np.concatenate(sequence_feats, axis=0)
        all_feats.append(full_seq)
        all_metadata.append({
            "id": p.id,
            "tier": p.tier,
            "prompt_len": input_len,
            "gen_len": len(resids) - 1
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        feature_acts=np.array(all_feats, dtype=object),
        metadata=np.array(all_metadata)
    )
    log.info("Saved multi-token activations to %s", args.out)

if __name__ == "__main__":
    main()
