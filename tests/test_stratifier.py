"""Unit tests for the stratified sampler."""

from biorefusalaudit.prompts.prompt_loader import DualUsePrompt
from biorefusalaudit.prompts.stratifier import marginal_counts, stratified_sample


def _make(id_: str, tier: str, framing: str) -> DualUsePrompt:
    return DualUsePrompt(
        id=id_,
        tier=tier,
        framing=framing,
        prompt="x",
        rationale="r",
        expected_surface="comply",
        category="c",
    )


def test_stratified_sample_equal_per_cell():
    pool = []
    for i in range(10):
        pool.append(_make(f"b{i}", "benign_bio", "direct"))
        pool.append(_make(f"d{i}", "dual_use_bio", "direct"))
        pool.append(_make(f"bE{i}", "benign_bio", "educational"))
        pool.append(_make(f"dE{i}", "dual_use_bio", "educational"))
    drawn = stratified_sample(pool, n_per_cell=3, seed=0)
    counts = marginal_counts(drawn)
    for cell, c in counts["cell"].items():
        assert c == 3


def test_small_cell_returns_all_available():
    pool = [
        _make("b1", "benign_bio", "roleplay"),
        _make("b2", "benign_bio", "roleplay"),
    ]
    drawn = stratified_sample(pool, n_per_cell=10, seed=0)
    assert len(drawn) == 2
