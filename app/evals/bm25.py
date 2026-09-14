"""Keyword-only baseline retrieval, kept permanently for comparison.
`rank_bm25`'s `BM25Okapi` (pure Python, no C extension, no extra system
dependency) rather than a hand-rolled TF-IDF: BM25 is the standard keyword
baseline retrieval work is compared against, and anything weaker would make
a "dense beats baseline" result meaningless.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
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
        self._corpus_tokens = [_tokenize(text) for _, text in items]
        self._bm25 = None
        if self._corpus_tokens:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus_tokens)

    def top_k(self, query_text: str, k: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked_indices[:k]]
