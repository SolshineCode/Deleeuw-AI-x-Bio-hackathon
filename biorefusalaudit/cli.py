"""Click CLI entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import yaml

from biorefusalaudit.prompts.prompt_loader import load_jsonl
from biorefusalaudit.prompts.safety_review import check_no_hazard_bodies

logging.basicConfig(level=logging.INFO, format="[biorefusalaudit] %(message)s")
log = logging.getLogger("biorefusalaudit")


@click.group()
def cli() -> None:
    """BioRefusalAudit — refusal-depth auditing for bio-safety prompts."""


@cli.command()
@click.option("--model", "model_name", required=True, help="HF model id, e.g., google/gemma-2-2b-it")
@click.option("--eval-set", "eval_set", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--sae-source", default="gemma_scope_1", show_default=True)
@click.option("--sae-release", default="gemma-scope-2b-pt-res", show_default=True)
@click.option("--sae-id", default="layer_12/width_16k/average_l0_82", show_default=True)
@click.option("--layer", default=12, type=int, show_default=True)
@click.option("--quantize", type=click.Choice(["none", "4bit", "8bit"]), default="none")
@click.option("--catalog", type=click.Path(), help="Feature catalog JSON (per-model)")
@click.option("--calibration", type=click.Path(), help="Calibration YAML with T matrix")
@click.option("--use-llm-judges/--no-llm-judges", default=False)
@click.option("--max-new-tokens", default=50, show_default=True, type=int,
              help="Completion length cap. 50 keeps wall-clock ~1 min/prompt on partial-GPU; 200 is canonical but 5x slower.")
@click.option("--temperature", default=0.7, show_default=True, type=float)
@click.option("--limit", default=0, type=int, help="If > 0, only run the first N prompts (useful for smoke tests).")
@click.option("--dump-activations/--no-dump-activations", default=False,
              help="Persist per-prompt d_sae feature vectors to <out>/activations.npz for catalog tuning + interventions.")
def run(
    model_name,
    eval_set,
    out_dir,
    sae_source,
    sae_release,
    sae_id,
    layer,
    quantize,
    catalog,
    calibration,
    use_llm_judges,
    max_new_tokens,
    temperature,
    limit,
    dump_activations,
) -> None:
    """Run a full eval against one model and write report.md + report.json."""
    from biorefusalaudit.features.feature_profiler import FeatureCatalog
    from biorefusalaudit.models.model_adapter import load_model
    from biorefusalaudit.models.sae_adapter import load_sae
    from biorefusalaudit.reporting.report import write_report
    from biorefusalaudit.runner.eval_runner import run_eval

    log.info("Loading model: %s (quantize=%s)", model_name, quantize)
    q = None if quantize == "none" else quantize
    lm = load_model(model_name, quantize=q)

    log.info("Loading SAE: %s / %s (layer %d)", sae_source, sae_release, layer)
    sae = load_sae(sae_source, sae_release, layer=layer, sae_id=sae_id)

    log.info("Loading eval set: %s", eval_set)
    prompts = load_jsonl(eval_set)
    if limit and limit > 0:
        prompts = prompts[:limit]
        log.info("%d prompts loaded (--limit %d)", len(prompts), limit)
    else:
        log.info("%d prompts loaded", len(prompts))

    problems = check_no_hazard_bodies(eval_set)
    if problems:
        raise click.ClickException("Safety review failed:\n" + "\n".join(f"- {p}" for p in problems))

    if catalog:
        cat = FeatureCatalog.load(catalog)
    else:
        log.warning("No feature catalog supplied — empty categorization (all zero).")
        cat = FeatureCatalog(model_name=model_name, sae_source=sae_release, categories={})

    T = None
    if calibration:
        cfg = yaml.safe_load(Path(calibration).read_text(encoding="utf-8"))
        T = np.array(cfg["T"], dtype=np.float64)
        log.info("Loaded calibration T from %s", calibration)

    llm_judges = ()
    if use_llm_judges:
        from biorefusalaudit.judges.llm_judges import (
            claude_haiku_judge,
            gemini_judge,
            ollama_judge,
        )

        llm_judges = (gemini_judge, claude_haiku_judge, ollama_judge)

    log.info("Running judge consortium: regex%s (max_new_tokens=%d, T=%.2f)",
             " + gemini + haiku + ollama" if use_llm_judges else "", max_new_tokens, temperature)
    activation_sink: list = [] if dump_activations else None
    report = run_eval(
        prompts=prompts,
        lm=lm,
        sae=sae,
        catalog=cat,
        hook_layer=layer,
        T=T,
        llm_judges=llm_judges,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        activation_sink=activation_sink,
        eval_set_path=str(eval_set),
    )
    md, js = write_report(report, out_dir)
    log.info("Processed %d/%d prompts.", len(report.records), len(prompts))
    log.info("Report: %s", md)
    for tier, a in sorted(report.aggregate.items()):
        log.info("Mean divergence (%s): %.3f", tier, a["mean_divergence"])

    if activation_sink:
        from pathlib import Path as _Path
        out_p = _Path(out_dir) / "activations.npz"
        ids = np.array([r["id"] for r in activation_sink])
        tiers = np.array([r["tier"] for r in activation_sink])
        framings = np.array([r["framing"] for r in activation_sink])
        feats = np.stack([r["feature_acts"] for r in activation_sink], axis=0)
        np.savez_compressed(out_p, ids=ids, tiers=tiers, framings=framings, feature_acts=feats)
        log.info("Dumped per-prompt SAE activations to %s (%d x %d)", out_p, feats.shape[0], feats.shape[1])


@cli.command()
@click.option("--eval-set", required=True, type=click.Path(exists=True))
def check_safety(eval_set) -> None:
    """Run the tier-3 hygiene check on a public eval set."""
    problems = check_no_hazard_bodies(eval_set)
    if problems:
        for p in problems:
            click.echo(f"- {p}")
        raise click.ClickException(f"{len(problems)} safety problem(s) found.")
    click.echo("OK — no safety problems detected.")


@cli.command("trace-cases")
@click.option("--report", required=True, type=click.Path(exists=True),
              help="Path to a run's report.json")
@click.option("--top-k", default=3, type=int)
@click.option("--out", type=click.Path(),
              help="Optional JSON output path; defaults to <report-dir>/selected_cases.json")
def trace_cases(report: str, top_k: int, out: str | None) -> None:
    """Select top-K prompts per criterion for downstream attribution / intervention.

    Criteria: high_divergence, false_refusal, false_compliance, high_judge_disagreement.
    """
    from biorefusalaudit.runner.trace_selected_cases import select_cases, write_selection

    cases = select_cases(report, top_k_per_criterion=top_k)
    out_path = Path(out) if out else Path(report).parent / "selected_cases.json"
    write_selection(cases, out_path)
    click.echo(f"Wrote {out_path} — {len(cases)} cases")
    for c in cases:
        click.echo(f"  [{c.criterion:>24}] {c.prompt_id} ({c.tier}/{c.framing}) rank={c.rank_value:.3f}  {c.notes}")


@cli.command("minimal-pairs")
@click.option("--eval-set", required=True, type=click.Path(exists=True))
@click.option("--out", default="data/eval_set_public/minimal_pairs.json", type=click.Path())
def minimal_pairs(eval_set: str, out: str) -> None:
    """Identify minimal-pair prompts across (benign_bio, dual_use_bio) sharing framing + near-topic.

    Used for before/after intervention comparison: the benign member is the baseline,
    the dual-use member is the probe. A model with a robust refusal-circuit should
    show distinct feature-activation patterns; a shallow refuser will show similar patterns.
    """
    import json
    from biorefusalaudit.prompts.prompt_loader import load_jsonl

    prompts = load_jsonl(eval_set)
    benign = [p for p in prompts if p.tier == "benign_bio"]
    dual_use = [p for p in prompts if p.tier == "dual_use_bio"]

    # Pair within framing; same-category heuristic based on shared keywords in rationale/category
    pairs: list[dict] = []
    for b in benign:
        same_framing = [d for d in dual_use if d.framing == b.framing]
        if not same_framing:
            continue
        # Prefer paired categories by token overlap on `category`
        def overlap(p):
            a = set(b.category.lower().replace("_", " ").split())
            c = set(p.category.lower().replace("_", " ").split())
            return len(a & c)
        best = max(same_framing, key=overlap)
        pairs.append({
            "benign": {"id": b.id, "category": b.category, "prompt": b.prompt[:120]},
            "dual_use": {"id": best.id, "category": best.category, "prompt": best.prompt[:120]},
            "framing": b.framing,
            "category_overlap": overlap(best),
        })

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    click.echo(f"Wrote {out} — {len(pairs)} minimal pairs (by framing)")


if __name__ == "__main__":
    cli()
