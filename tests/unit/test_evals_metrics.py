import pytest

from app.evals.metrics import (
    PairScore,
    bootstrap_ci,
    mean_system_score,
    ndcg_at_k,
    paired_difference,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_pair,
)


def test_precision_at_k_counts_hits_in_top_k_only():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 99}
    assert precision_at_k(retrieved, relevant, k=5) == 2 / 5


def test_precision_at_k_ignores_beyond_k():
    retrieved = [1, 2, 3]
    relevant = {3}
    # 3 is outside the top-2 window
    assert precision_at_k(retrieved, relevant, k=2) == 0.0


def test_precision_at_k_empty_retrieved_is_zero():
    assert precision_at_k([], {1, 2}, k=5) == 0.0


def test_recall_at_k_fraction_of_relevant_found():
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 7, 9}
    assert recall_at_k(retrieved, relevant, k=10) == 2 / 4


def test_recall_at_k_empty_relevant_set_is_zero_not_error():
    assert recall_at_k([1, 2], set(), k=10) == 0.0


def test_recall_at_k_perfect_recall():
    retrieved = [1, 2, 3]
    relevant = {1, 2, 3}
    assert recall_at_k(retrieved, relevant, k=10) == 1.0


def _pair(p: float, r: float, ndcg: float = 0.0, rr: float = 0.0, n: int = 10) -> PairScore:
    return PairScore(
        precision_at_5=p, recall_at_10=r, ndcg_at_10=ndcg, reciprocal_rank=rr, retrieved=n
    )


def test_mean_system_score_averages_across_pairs():
    result = mean_system_score([_pair(1.0, 0.5, 1.0, 1.0, 10), _pair(0.0, 0.5, 0.0, 0.0, 4)])
    assert result.precision_at_5 == 0.5
    assert result.recall_at_10 == 0.5
    assert result.ndcg_at_10 == 0.5
    assert result.mrr == 0.5
    assert result.retrieved_avg == 7.0
    assert result.pairs_scored == 2
    assert result.ci95["precision_at_5"] is not None


def test_mean_system_score_empty_input_is_zero_with_zero_pairs_scored():
    result = mean_system_score([])
    assert result.precision_at_5 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.pairs_scored == 0


def test_precision_at_k_divides_by_k_not_by_results_returned():
    # Two hits and nothing else would read 1.0 under a returned-count denominator.
    assert precision_at_k([1, 2], {1, 2}, k=5) == 2 / 5


def test_ndcg_rewards_hits_near_the_top():
    relevant = {1}
    assert ndcg_at_k([1, 2, 3], relevant, k=10) == 1.0
    assert ndcg_at_k([2, 3, 1], relevant, k=10) == pytest.approx(0.5)
    assert ndcg_at_k([2, 3], relevant, k=10) == 0.0


def test_ndcg_reaches_one_with_more_relevant_than_k():
    relevant = set(range(20))
    assert ndcg_at_k(list(range(10)), relevant, k=10) == pytest.approx(1.0)


def test_ndcg_empty_relevant_is_zero():
    assert ndcg_at_k([1, 2], set(), k=10) == 0.0


def test_reciprocal_rank_is_one_over_first_hit():
    assert reciprocal_rank([5, 6, 1], {1}) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 6], {1}) == 0.0


def test_score_pair_fills_every_metric():
    score = score_pair([1, 2, 3], {1, 3})
    assert score.precision_at_5 == 2 / 5
    assert score.recall_at_10 == 1.0
    assert score.reciprocal_rank == 1.0
    assert score.retrieved == 3


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    low, high = bootstrap_ci(values)
    assert low < 0.5 < high
    assert bootstrap_ci(values) == (low, high)


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([0.5]) is None
    assert bootstrap_ci([]) is None


def test_paired_difference_spanning_zero_and_not():
    mean, interval = paired_difference([0.5, 0.6, 0.4, 0.5], [0.5, 0.4, 0.6, 0.5])
    assert mean == pytest.approx(0.0)
    assert interval is not None and interval[0] <= 0 <= interval[1]
    mean, interval = paired_difference([0.9, 0.8, 0.9, 0.85], [0.1, 0.2, 0.15, 0.1])
    assert interval is not None and interval[0] > 0


def test_paired_difference_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_difference([0.1], [0.1, 0.2])
