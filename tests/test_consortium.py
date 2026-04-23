"""Unit tests for the judge consortium aggregator."""

import pytest

from biorefusalaudit.judges.consortium import (
    ConsortiumVerdict,
    JudgeVote,
    aggregate_votes,
    soft_distribution_vector,
)


def test_unanimous_refuse_high_confidence():
    votes = [
        JudgeVote("regex", "refuse", 0.95),
        JudgeVote("gemini", "refuse", 0.9),
        JudgeVote("claude_haiku", "refuse", 0.9),
        JudgeVote("ollama", "refuse", 0.8),
    ]
    v = aggregate_votes(votes)
    assert v.label == "refuse"
    assert v.soft_distribution["refuse"] > 0.95
    assert v.disagreement < 0.05
    assert v.flagged_for_review is False


def test_split_vote_flagged_for_review():
    votes = [
        JudgeVote("regex", "refuse", 0.9),
        JudgeVote("gemini", "comply", 0.9),
        JudgeVote("claude_haiku", "refuse", 0.9),
        JudgeVote("ollama", "comply", 0.9),
    ]
    v = aggregate_votes(votes)
    assert v.disagreement > 0.3
    assert v.flagged_for_review is True


def test_empty_votes_raises():
    with pytest.raises(ValueError):
        aggregate_votes([])


def test_soft_distribution_vector_length_5_sums_to_one():
    votes = [JudgeVote("gemini", "comply", 1.0)]
    v = aggregate_votes(votes)
    arr = soft_distribution_vector(v)
    assert arr.shape == (5,)
    assert abs(arr.sum() - 1.0) < 1e-6


def test_zero_confidence_all_judges_falls_back_to_review():
    votes = [JudgeVote("regex", "refuse", 0.0)]
    v = aggregate_votes(votes)
    assert v.flagged_for_review is True
