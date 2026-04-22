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
@click.option("--sae-id", default="layer_12/width_16k/average_l0_71", show_default=True)
@click.option("--layer", default=12, type=int, show_default=True)
@click.option("--quantize", type=click.Choice(["none", "4bit", "8bit"]), default="none")
@click.option("--catalog", type=click.Path(), help="Feature catalog JSON (per-model)")
@click.option("--calibration", type=click.Path(), help="Calibration YAML with T matrix")
@click.option("--use-llm-judges/--no-llm-judges", default=False)
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

    log.info("Running judge consortium: regex%s", " + gemini + haiku + ollama" if use_llm_judges else "")
    report = run_eval(
        prompts=prompts,
        lm=lm,
        sae=sae,
        catalog=cat,
        hook_layer=layer,
        T=T,
        llm_judges=llm_judges,
        eval_set_path=str(eval_set),
    )
    md, js = write_report(report, out_dir)
    log.info("Processed %d/%d prompts.", len(report.records), len(prompts))
    log.info("Report: %s", md)
    for tier, a in sorted(report.aggregate.items()):
        log.info("Mean divergence (%s): %.3f", tier, a["mean_divergence"])


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


if __name__ == "__main__":
    cli()
