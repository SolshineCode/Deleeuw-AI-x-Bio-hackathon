"""Orchestrate (model + SAE + eval_set) -> RunReport.

Pipeline per prompt:
  1. Generate model completion
  2. Capture residual stream at target layer during generation
  3. Project through SAE -> feature activation vector
  4. Categorize -> 5-category vector f
  5. Run judge consortium on (prompt, completion) -> soft surface distribution s
  6. Compute divergence D(s, f, T) and flags
  7. Append PromptRecord

T (alignment matrix) is loaded from a per-model calibration config; if not
present, identity is used as a fallback (flagged in the report).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from biorefusalaudit.features.feature_profiler import FeatureCatalog, categorize
from biorefusalaudit.judges.consortium import (
    JudgeVote,
    aggregate_votes,
    soft_distribution_vector,
)
from biorefusalaudit.judges.regex_classifier import regex_classify
from biorefusalaudit.models.model_adapter import (
    LoadedModel,
    generate_completion,
    residual_stream_hook,
)
from biorefusalaudit.models.sae_adapter import LoadedSAE, project_activations
from biorefusalaudit.prompts.prompt_loader import DualUsePrompt
from biorefusalaudit.reporting.report import PromptRecord, RunReport
from biorefusalaudit.scoring.divergence import (
    FEATURE_CATEGORIES,
    divergence,
    divergence_flags,
)

log = logging.getLogger("biorefusalaudit.runner")

JudgeFn = Callable[[str, str], JudgeVote]


def _regex_as_vote(text: str) -> JudgeVote:
    v = regex_classify(text)
    label = v.label if v.label is not None else "refuse"
    return JudgeVote(
        judge_name="regex",
        label=label,
        confidence=v.confidence,
        raw_response=", ".join(v.matched_patterns)[:200],
    )


def run_one_prompt(
    prompt: DualUsePrompt,
    lm: LoadedModel,
    sae: LoadedSAE,
    catalog: FeatureCatalog,
    hook_layer: int,
    T: np.ndarray | None,
    llm_judges: Sequence[JudgeFn] = (),
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    activation_sink: list | None = None,
) -> PromptRecord:
    with residual_stream_hook(lm.model, hook_layer) as get_resid:
        completion = generate_completion(
            lm, prompt.prompt, max_new_tokens=max_new_tokens, temperature=temperature
        )
        resid = get_resid()

    if resid is None:
        raise RuntimeError(f"no activations captured for prompt {prompt.id}")

    feature_acts = project_activations(sae, resid)  # (batch=1, d_sae)
    f_vec = categorize(feature_acts[0], catalog)

    # Per specialist review — persist the full d_sae feature vector so
    # the catalog-tuning + intervention pipelines have real activation
    # data, not just the 5-category projection.
    if activation_sink is not None:
        activation_sink.append(
            {"id": prompt.id, "tier": prompt.tier, "framing": prompt.framing,
             "feature_acts": feature_acts[0].astype(np.float32)}
        )

    # Judge consortium
    votes: list[JudgeVote] = [_regex_as_vote(completion)]
    for judge in llm_judges:
        try:
            votes.append(judge(prompt.prompt, completion))
        except Exception as e:  # noqa: BLE001
            log.warning("judge %s failed: %s", getattr(judge, "__name__", judge), e)
    verdict = aggregate_votes(votes)
    s_vec = soft_distribution_vector(verdict)

    d = divergence(s_vec, f_vec, T)
    flags = divergence_flags(s_vec, f_vec)

    return PromptRecord(
        id=prompt.id,
        tier=prompt.tier,
        framing=prompt.framing,
        prompt=prompt.prompt,
        completion=completion,
        surface_label=verdict.label,
        surface_soft=dict(verdict.soft_distribution),
        feature_vec=list(f_vec),
        feature_categories=list(FEATURE_CATEGORIES),
        divergence=d,
        flags=flags,
        judge_disagreement=verdict.disagreement,
    )


def run_eval(
    prompts: Sequence[DualUsePrompt],
    lm: LoadedModel,
    sae: LoadedSAE,
    catalog: FeatureCatalog,
    hook_layer: int,
    T: np.ndarray | None = None,
    llm_judges: Sequence[JudgeFn] = (),
    eval_set_path: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    activation_sink: list | None = None,
) -> RunReport:
    import time

    report = RunReport(
        model_name=lm.name,
        sae_source=f"{sae.source}:{sae.name}",
        eval_set_path=eval_set_path,
    )
    n = len(prompts)
    t_start = time.time()
    for i, p in enumerate(prompts):
        t_prompt = time.time()
        try:
            rec = run_one_prompt(
                p, lm, sae, catalog, hook_layer, T, llm_judges,
                max_new_tokens=max_new_tokens, temperature=temperature,
                activation_sink=activation_sink,
            )
            report.records.append(rec)
            elapsed = time.time() - t_prompt
            total_elapsed = time.time() - t_start
            log.info(
                "[%d/%d] %s (%s/%s) D=%.3f label=%s (%.1fs; eta=%.0fs)",
                i + 1, n, p.id, p.tier, p.framing, rec.divergence, rec.surface_label,
                elapsed, (total_elapsed / (i + 1)) * (n - i - 1),
            )
        except Exception as e:  # noqa: BLE001
            import traceback
            log.error("[%d/%d] %s failed: %s\n%s", i + 1, n, p.id, e, traceback.format_exc())
    report.compute_aggregate()
    log.info("Processed %d/%d prompts in %.1fs", len(report.records), n, time.time() - t_start)
    return report
