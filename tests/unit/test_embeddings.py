from pathlib import Path

import app.core.db as db_module
from app.core.db import init_db
from app.core.embeddings import embed
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def test_embed_empty_list_returns_empty(tmp_path):
    _reset_db(tmp_path)
    assert embed([]) == []


def test_embed_calls_encoder_once_per_unique_text(tmp_path):
    _reset_db(tmp_path)
    calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(t)), 0.0] for t in texts]

    result = embed(["hello", "world"], _encode_fn=fake_encode)

    assert result == [[5.0, 0.0], [5.0, 0.0]]
    assert len(calls) == 1
    assert calls[0] == ["hello", "world"]


def test_embed_second_call_hits_cache(tmp_path):
    _reset_db(tmp_path)
    calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[1.0, 2.0] for _ in texts]

    embed(["repeat me"], _encode_fn=fake_encode)
    embed(["repeat me"], _encode_fn=fake_encode)

    assert len(calls) == 1


def test_embed_partial_cache_hit_only_encodes_misses(tmp_path):
    _reset_db(tmp_path)
    calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(i), 0.0] for i, _ in enumerate(texts)]

    embed(["cached one"], _encode_fn=fake_encode)
    result = embed(["cached one", "new one"], _encode_fn=fake_encode)

    assert len(calls) == 2
    assert calls[1] == ["new one"]
    assert result[0] == [0.0, 0.0]  # served from cache, not re-encoded


def test_embed_repeated_text_in_one_batch_is_encoded_and_cached_once(tmp_path):
    _reset_db(tmp_path)
    calls: list[list[str]] = []

    def fake_encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        return [[float(len(t)), 1.0] for t in texts]

    result = embed(["python", "go", "python"], _encode_fn=fake_encode)

    assert result == [[6.0, 1.0], [2.0, 1.0], [6.0, 1.0]]
    assert calls == [["python", "go"]]
