"""Keyword retrieval: BM25 over a fixed (id, text) corpus.

Used two ways. app/retrieval/search.py fuses it with dense search, because
evidence documents are short skill labels and an exact skill name is the
strongest signal they carry. app/evals/ keeps it on its own as the
baseline every retrieval change is compared against.

`rank_bm25`'s `BM25Okapi` (pure Python, no C extension, no extra system
dependency) rather than a hand-rolled TF-IDF: BM25 is the standard keyword
baseline, and anything weaker would make a comparison against it
meaningless.
"""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-word tokens only. Kept simple: this
    exists to be the plain baseline, so a smarter tokenizer (stemming,
    stopword removal, n-grams) would blur the exact comparison it exists
    to make: "does the smarter system beat plain keyword
    matching," not "does plain keyword matching plus some NLP beat it."
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class Bm25Corpus:
    """Wraps BM25Okapi over a fixed (id, text) corpus, exposing the same
    "query text in, ranked ids out" shape as app/retrieval/search.py's
    dense search functions, so app/evals/run.py can score both systems
    through one interface. An empty corpus (an account with nothing
    indexed yet in this collection) returns no results rather than
    raising; a thin corpus is a real, scoreable (as all-zero) state, not
    an error.
    """

    def __init__(self, items: list[tuple[int, str]]):
        self._ids = [item_id for item_id, _ in items]
        self._corpus_tokens = [tokenize(text) for _, text in items]
        # Token sets alongside the lists: BM25 needs the lists (term
        # frequency counts repeats), while whether a document matched at all
        # is a set question.
        self._corpus_token_sets = [set(tokens) for tokens in self._corpus_tokens]
        self._bm25 = None
        if self._corpus_tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens)

    def top_k(self, query_text: str, k: int) -> list[int]:
        """Ranked ids for the documents sharing at least one token with the
        query, best first, at most `k` of them.

        Documents sharing no token are left out rather than used to pad the
        list to `k`: one of them was not found by keyword matching, so
        returning it overstates this baseline, and sorting an all-equal score
        array would hand back corpus order dressed up as a ranking.

        Membership is token overlap, not a positive score: BM25Okapi gives a
        term found in every document a negative idf, so filtering on score
        would drop real matches on the corpus's most common skills. Ties
        break by id, so a reindex that reorders rows cannot reorder results.
        """
        if self._bm25 is None:
            return []
        tokens = tokenize(query_text)
        query_tokens = set(tokens)
        scores = self._bm25.get_scores(tokens)
        matched = [i for i, doc in enumerate(self._corpus_token_sets) if query_tokens & doc]
        matched.sort(key=lambda i: (-scores[i], self._ids[i]))
        return [self._ids[i] for i in matched[:k]]
