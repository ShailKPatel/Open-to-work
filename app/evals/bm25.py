"""The keyword-only baseline the eval harness compares the app's retrieval
against. The implementation lives in app/retrieval/keyword.py, since the
app's own hybrid search uses it too; it is re-exported here so the baseline
has one obvious home in the eval package.
"""

from app.retrieval.keyword import Bm25Corpus, tokenize

__all__ = ["Bm25Corpus", "tokenize"]
