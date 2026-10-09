"""Retrieval-quality metrics against a golden set's labeled relevant ids:
precision@5, recall@10, nDCG@10 and reciprocal rank per pair, their means
per system, and bootstrap confidence intervals on those means. Pure
functions, no I/O, no LLM; the numbers app/evals/run.py aggregates come
from here, fully unit-tested with zero mocking needed, same posture as
app/profile/weighting.py.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

PRECISION_K = 5
RECALL_K = 10
NDCG_K = 10

# Resamples for the bootstrap intervals, and a fixed seed so the same
# scores always print the same interval.
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 0


# Ids are ints for search hits, skill names for app/evals/candidates.py.
def precision_at_k[Id: Hashable](
    retrieved: Sequence[Id], relevant: AbstractSet[Id], k: int
) -> float:
    """Of k result slots, what fraction hold a relevant id.

    The denominator is k, not the number of ids returned. Dividing by what
    came back pays a system for returning less: two hits and nothing else
    would score 1.000, while the same two hits inside a longer list score
    0.400. Under a fixed k the length of the list is neutral, which is what
    makes two systems that return different numbers of results comparable,
    and it is the standard definition besides. An empty retrieval scores
    0.0: a system that returns nothing gets no credit.
    """
    hits = sum(1 for i in retrieved[:k] if i in relevant)
    return hits / k


def recall_at_k[Id: Hashable](
    retrieved: Sequence[Id], relevant: AbstractSet[Id], k: int
) -> float:
    """Of everything actually relevant, what fraction shows up in the
    top-k retrieved ids. An empty relevant set (a mislabeled or
    accidentally-empty golden pair) scores 0.0 rather than dividing by
    zero; callers should filter those out before scoring (see
    app/evals/run.py, which skips and notes them instead of silently
    including a meaningless 0.0).

    Note the ceiling: a pair with more than k relevant ids cannot reach
    1.0. nDCG@k normalises by the best ranking actually possible, so read
    the two together when pairs have many relevant documents."""
    if not relevant:
        return 0.0
    hits = sum(1 for i in retrieved[:k] if i in relevant)
    return hits / len(relevant)


def ndcg_at_k[Id: Hashable](
    retrieved: Sequence[Id], relevant: AbstractSet[Id], k: int
) -> float:
    """Binary-relevance nDCG: hits near the top count more than hits near
    the bottom, divided by the score of a perfect ranking, so 1.0 is
    reachable whatever the number of relevant ids."""
    dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(retrieved[:k]) if i in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def reciprocal_rank[Id: Hashable](retrieved: Sequence[Id], relevant: AbstractSet[Id]) -> float:
    """1 / the rank of the first relevant id, 0.0 when none is retrieved:
    how far down a reader has to go before the first useful result."""
    for rank, i in enumerate(retrieved):
        if i in relevant:
            return 1 / (rank + 1)
    return 0.0


@dataclass
class PairScore:
    """Every metric for one golden pair under one system."""

    precision_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    reciprocal_rank: float
    retrieved: int


def score_pair[Id: Hashable](retrieved: Sequence[Id], relevant: AbstractSet[Id]) -> PairScore:
    return PairScore(
        precision_at_5=precision_at_k(retrieved, relevant, PRECISION_K),
        recall_at_10=recall_at_k(retrieved, relevant, RECALL_K),
        ndcg_at_10=ndcg_at_k(retrieved, relevant, NDCG_K),
        reciprocal_rank=reciprocal_rank(retrieved, relevant),
        retrieved=len(retrieved),
    )


def wilson_interval(correct: int, total: int) -> tuple[float, float] | None:
    """95 percent Wilson score interval for a proportion. Unlike a
    bootstrap, it stays honest at the edges: 30 correct out of 30 gives
    roughly (0.89, 1.0), not a point at 1.0, which is what a perfect score
    on a small set actually supports."""
    if total <= 0:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values: Sequence[float]) -> tuple[float, float] | None:
    """95 percent percentile bootstrap interval on the mean of `values`:
    resample the pairs with replacement, take each resample's mean, and
    report the 2.5th and 97.5th percentiles. None below two values, where
    there is nothing to resample. With tens of pairs the interval is wide,
    which is the point: a gap between two systems narrower than it is not
    evidence of a difference."""
    if len(values) < 2:
        return None
    import numpy as np

    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = rng.choice(data, size=(_BOOTSTRAP_RESAMPLES, len(data)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_difference(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, tuple[float, float] | None]:
    """Mean of first minus second over the same pairs, with a bootstrap
    interval on that difference. Paired, so per-pair difficulty cancels
    out: an interval that excludes zero is a difference these pairs
    support; one that spans zero is not."""
    if len(first) != len(second):
        raise ValueError("paired_difference needs scores for the same pairs")
    diffs = [a - b for a, b in zip(first, second, strict=True)]
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    return mean, bootstrap_ci(diffs)


@dataclass
class SystemScore:
    """One system's mean metrics across every golden pair it was scored
    against, plus how many pairs actually contributed. A pair whose
    collection has zero indexed rows for this account contributes nothing,
    tracked separately so a thin corpus can't silently masquerade as a
    perfect (or perfectly zero) score.

    `retrieved_avg` is how many ids the system returned per pair on
    average. Since precision@5 divides by 5 whatever comes back, two
    systems can post the same score while one fills its window and the
    other returns three results; keyword search does exactly that on a
    corpus whose documents carry few tokens.

    `ci95` holds the bootstrap interval for each mean, keyed by field name.
    """

    precision_at_5: float
    recall_at_10: float
    pairs_scored: int
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    retrieved_avg: float = 0.0
    ci95: dict[str, list[float] | None] = field(default_factory=dict)


def mean_system_score(scores: Sequence[PairScore]) -> SystemScore:
    """Means over one system's per-pair scores. Empty input is a real,
    valid "nothing scored yet" state (a fresh golden set, or an account
    with no labels), not an error: returns all-zero with pairs_scored=0 so
    the caller can tell "zero" apart from "never measured" by checking
    pairs_scored, not by the score values alone.
    """
    if not scores:
        return SystemScore(precision_at_5=0.0, recall_at_10=0.0, pairs_scored=0)
    n = len(scores)
    columns = {
        "precision_at_5": [s.precision_at_5 for s in scores],
        "recall_at_10": [s.recall_at_10 for s in scores],
        "ndcg_at_10": [s.ndcg_at_10 for s in scores],
        "mrr": [s.reciprocal_rank for s in scores],
    }
    ci95: dict[str, list[float] | None] = {}
    for name, values in columns.items():
        interval = bootstrap_ci(values)
        ci95[name] = list(interval) if interval is not None else None
    return SystemScore(
        precision_at_5=sum(columns["precision_at_5"]) / n,
        recall_at_10=sum(columns["recall_at_10"]) / n,
        pairs_scored=n,
        ndcg_at_10=sum(columns["ndcg_at_10"]) / n,
        mrr=sum(columns["mrr"]) / n,
        retrieved_avg=sum(s.retrieved for s in scores) / n,
        ci95=ci95,
    )
