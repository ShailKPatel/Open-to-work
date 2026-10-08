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


def test_query_with_no_overlap_returns_nothing_not_padding():
    corpus = Bm25Corpus([(1, "python backend"), (2, "go microservices")])
    assert corpus.top_k("completely unrelated woodworking terms", k=5) == []


def test_only_documents_sharing_a_token_are_returned():
    corpus = Bm25Corpus([(1, "python backend"), (2, "go microservices"), (3, "python data")])
    assert sorted(corpus.top_k("python", k=5)) == [1, 3]


def test_term_in_every_document_still_matches():
    """BM25Okapi gives such a term a negative idf; membership is overlap, so
    the documents are still returned."""
    corpus = Bm25Corpus([(1, "python api"), (2, "python data")])
    assert sorted(corpus.top_k("python", k=5)) == [1, 2]


def test_ties_break_by_id_not_corpus_order():
    corpus = Bm25Corpus([(9, "skill python"), (4, "skill python"), (7, "skill python")])
    assert corpus.top_k("python", k=5) == [4, 7, 9]
