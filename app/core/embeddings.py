"""Local sentence-transformers embeddings. No external API, no key.

Batched, cached by content hash in the `embedding_cache` table. Identical
text is never re-encoded, even across runs.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from sqlalchemy import select

from app.core.db import EmbeddingCache, get_db

# Changing this invalidates every stored vector (and the Qdrant collection
# sizes follow it), so it is a code constant rather than a user setting.
# Six models were compared under the app's hybrid search
# (evals/results/embedding-retrieval-20261008.md): all landed within about
# 0.05 precision@5 of each other, and this one tied best on real text, so
# no switch was worth re-embedding every stored vector.
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache
def _get_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(
    texts: list[str], _encode_fn: Any = None, model_name: str | None = None
) -> list[list[float]]:
    """_encode_fn is an injection point for tests; production callers never
    pass it; it defaults to the sentence-transformers model's .encode().

    model_name overrides EMBEDDING_MODEL for callers whose vectors never
    meet the retrieval ones. The skill map is the only one: it compares
    skills to each other inside a single layout and never touches Qdrant,
    so it can use a model chosen for that job (see
    app/profile/skill_map.py) without the collections having to be
    rebuilt at a new width. The cache is keyed by model already, so two
    models coexist row for row.
    """
    if not texts:
        return []

    model_name = model_name or EMBEDDING_MODEL
    hashes = [_content_hash(t) for t in texts]

    db = get_db()
    try:
        cached_rows = db.execute(
            select(EmbeddingCache).where(
                EmbeddingCache.model == model_name, EmbeddingCache.content_hash.in_(hashes)
            )
        ).scalars().all()
        cache_by_hash = {row.content_hash: row.vector_json for row in cached_rows}

        # First occurrence of each uncached text only: a batch can repeat a
        # text (two roles both claiming "Python"), and a second cache row
        # for the same hash would violate the unique index.
        first_index = {h: i for i, h in reversed(list(enumerate(hashes)))}
        miss_indices = sorted(i for h, i in first_index.items() if h not in cache_by_hash)
        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            if _encode_fn is not None:
                fresh_vectors = _encode_fn(miss_texts)
            else:
                model = _get_model(model_name)
                fresh_vectors = model.encode(
                    miss_texts, batch_size=32, normalize_embeddings=True
                ).tolist()

            for idx, vector in zip(miss_indices, fresh_vectors, strict=True):
                h = hashes[idx]
                vector_list = list(vector)
                db.add(EmbeddingCache(content_hash=h, model=model_name, vector_json=vector_list))
                cache_by_hash[h] = vector_list
            db.commit()

        return [cache_by_hash[h] for h in hashes]
    finally:
        db.close()
