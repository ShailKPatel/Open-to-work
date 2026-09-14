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
from app.core.settings import get_settings


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache
def _get_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts: list[str], _encode_fn: Any = None) -> list[list[float]]:
    """_encode_fn is an injection point for tests; production callers never
    pass it; it defaults to the sentence-transformers model's .encode().
    """
    if not texts:
        return []

    model_name = get_settings().embedding_model
    hashes = [_content_hash(t) for t in texts]

    db = get_db()
    try:
        cached_rows = db.execute(
            select(EmbeddingCache).where(
                EmbeddingCache.model == model_name, EmbeddingCache.content_hash.in_(hashes)
            )
        ).scalars().all()
        cache_by_hash = {row.content_hash: row.vector_json for row in cached_rows}

        miss_indices = [i for i, h in enumerate(hashes) if h not in cache_by_hash]
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
