from app.evals.metrics import mean_system_score, precision_at_k, recall_at_k


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


def test_mean_system_score_averages_across_pairs():
    scores = [(1.0, 0.5), (0.0, 0.5)]
    result = mean_system_score(scores)
    assert result.precision_at_5 == 0.5
    assert result.recall_at_10 == 0.5
    assert result.pairs_scored == 2


def test_mean_system_score_empty_input_is_zero_with_zero_pairs_scored():
    result = mean_system_score([])
    assert result.precision_at_5 == 0.0
    assert result.recall_at_10 == 0.0
    assert result.pairs_scored == 0
