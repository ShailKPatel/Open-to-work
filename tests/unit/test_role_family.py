"""app/profile/role_family.py: canonical job-title clustering. Real Qdrant
(:memory: mode) for the similarity search, injected complete() for the
canonicalization LLM call, same conventions as test_search.py /
test_extract.py respectively.
"""

from pathlib import Path
from unittest.mock import MagicMock

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import RoleFamily, get_db, init_db
from app.core.settings import get_settings
from app.profile.role_family import resolve_role_family


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _fake_embed(monkeypatch):
    """"ml"-flavored titles cluster together, everything else is far away;
    deterministic, so a near-duplicate title can be asserted to actually
    reuse the existing family rather than just returning something."""

    def _encode(texts):
        return [
            [1.0, 0.0] if "ml" in t.lower() or "machine learning" in t.lower() else [0.0, 1.0]
            for t in texts
        ]

    import app.retrieval.index as index_module
    import app.retrieval.search as search_module

    monkeypatch.setattr(index_module, "embed", _encode)
    monkeypatch.setattr(search_module, "embed", _encode)


def test_blank_title_returns_none(tmp_path):
    _reset(tmp_path)
    assert resolve_role_family("   ") is None


def test_new_title_creates_a_family_via_llm(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    fake_response = MagicMock()
    fake_response.parsed = {"canonical_name": "Machine Learning Engineer"}
    monkeypatch.setattr(
        "app.profile.role_family.complete", MagicMock(return_value=fake_response)
    )

    family = resolve_role_family("ML Engineer")

    assert family is not None
    assert family.canonical_name == "Machine Learning Engineer"


def test_near_duplicate_title_reuses_existing_family(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    from app.retrieval.index import index_role_family

    db = get_db()
    existing = RoleFamily(canonical_name="Machine Learning Engineer")
    db.add(existing)
    db.commit()
    db.refresh(existing)
    existing_id = existing.id
    db.close()
    index_role_family(existing)

    fake_complete = MagicMock()
    monkeypatch.setattr("app.profile.role_family.complete", fake_complete)

    family = resolve_role_family("Machine Learning Engineer II")

    assert family is not None
    assert family.id == existing_id
    fake_complete.assert_not_called()  # reused via search, no LLM call spent


def test_llm_failure_degrades_to_none_not_raise(tmp_path, monkeypatch):
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.profile.role_family.complete", MagicMock(return_value=fake_response)
    )

    assert resolve_role_family("Some Unique Title") is None


def test_concurrent_insert_race_reuses_winners_row(tmp_path, monkeypatch):
    """Two resolutions for the same brand-new canonical_name: the loser's
    IntegrityError is caught and it reuses the winner's row instead of
    raising."""
    _reset(tmp_path)
    _fake_embed(monkeypatch)

    fake_response = MagicMock()
    fake_response.parsed = {"canonical_name": "Data Engineer"}
    monkeypatch.setattr(
        "app.profile.role_family.complete", MagicMock(return_value=fake_response)
    )

    # Pre-create the row to simulate "someone else already won the race"
    # right before this call's own commit.
    db = get_db()
    db.add(RoleFamily(canonical_name="Data Engineer"))
    db.commit()
    db.close()

    family = resolve_role_family("Data Eng")

    assert family is not None
    assert family.canonical_name == "Data Engineer"
