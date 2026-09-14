"""Retrieval-quality metrics: precision@k / recall@k against a golden
set's labeled relevant ids (precision@5 and recall@10). Pure functions, no I/O, no LLM;
the numbers app/evals/run.py aggregates come from here, fully unit-tested
with zero mocking needed, same posture as app/profile/weighting.py.
"""

from __future__ import annotations

from dataclasses import dataclass


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Of the top-k retrieved ids, what fraction are actually relevant.
    An empty top-k (nothing retrieved) scores 0.0, not undefined; a
    system that returns nothing gets no credit."""
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for i in top if i in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Of everything actually relevant, what fraction shows up in the
    top-k retrieved ids. An empty relevant set (a mislabeled or
    accidentally-empty golden pair) scores 0.0 rather than dividing by
    zero; callers should filter those out before scoring (see
    app/evals/run.py, which skips and notes them instead of silently
    including a meaningless 0.0)."""
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for i in top if i in relevant)
    return hits / len(relevant)


@dataclass
class SystemScore:
    """One system's (dense or BM25) mean precision@5 / recall@10 across
    every golden pair it was scored against, plus how many pairs actually
    contributed. A pair whose collection has zero indexed rows for this
    account contributes nothing, tracked separately so a thin corpus
    can't silently masquerade as a perfect (or perfectly zero) score."""

    precision_at_5: float
    recall_at_10: float
    pairs_scored: int


def mean_system_score(scores: list[tuple[float, float]]) -> SystemScore:
    """scores is a list of (precision_at_5, recall_at_10) tuples, one per
    scored golden pair, for one system. Empty input is a real, valid
    "nothing scored yet" state (a fresh golden set, or an account with no
    labels), not an error: returns all-zero with pairs_scored=0 so the
    caller can tell "zero" apart from "never measured" by
    checking pairs_scored, not by the score values alone.
    """
    if not scores:
        return SystemScore(precision_at_5=0.0, recall_at_10=0.0, pairs_scored=0)
    precision = sum(s[0] for s in scores) / len(scores)
    recall = sum(s[1] for s in scores) / len(scores)
    return SystemScore(precision_at_5=precision, recall_at_10=recall, pairs_scored=len(scores))
