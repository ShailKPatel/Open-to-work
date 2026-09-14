"""app/evals/bm25.py: real rank_bm25, no mocking. The point of this
class is to be a real keyword baseline, so its own tests run the real
ranking algorithm against fixed text, not a stand-in.
"""

from app.evals.bm25 import Bm25Corpus


def test_empty_corpus_returns_nothing():
    corpus = Bm25Corpus([])
    assert corpus.top_k("python backend", k=5) == []


def test_ranks_the_keyword_matching_document_first():
    corpus = Bm25Corpus(
        [
            (1, "Skill: Python. Evidence: declared_dependency."),
            (2, "Skill: Woodworking. Evidence: manual."),
        ]
    )
    hits = corpus.top_k("python developer", k=5)
    assert hits[0] == 1


def test_top_k_limits_results():
    corpus = Bm25Corpus([(i, "python backend service") for i in range(1, 11)])
    hits = corpus.top_k("python", k=3)
    assert len(hits) == 3


def test_query_with_no_overlap_still_returns_something_not_error():
    corpus = Bm25Corpus([(1, "python backend"), (2, "go microservices")])
    hits = corpus.top_k("completely unrelated woodworking terms", k=5)
    assert len(hits) == 2  # both scored (likely near-zero), none crash
