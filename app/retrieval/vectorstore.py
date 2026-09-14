"""Qdrant connection + collection bootstrap.

Search itself lives in app/retrieval/search.py.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.settings import get_settings


@lru_cache
def get_client() -> Any:
    from qdrant_client import QdrantClient

    url = get_settings().qdrant_url
    if url == ":memory:":
        # test-only escape hatch: no real Qdrant server needed
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url)


def ensure_collection(name: str, vector_size: int, distance: str = "Cosine") -> None:
    from qdrant_client.models import Distance, VectorParams

    client = get_client()
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance(distance)),
    )
