import math
from pathlib import Path

from sqlalchemy import select

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import (
    Account,
    Experience,
    ExperienceSkillEvidence,
    Repository,
    SkillEvidence,
    SkillMapCache,
    get_db,
    init_db,
)
from app.core.settings import get_settings
from app.profile import skill_map


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    account_id = account.id
    db.close()
    return account_id


def _repo(account_id: int, name: str, language: str, skills: list[str]) -> None:
    db = get_db()
    repo = Repository(
        account_id=account_id,
        github_id=abs(hash(name)) % 10_000_000,
        name=name,
        full_name=f"octocat/{name}",
        url=f"https://github.com/octocat/{name}",
        primary_language=language,
    )
    db.add(repo)
    db.commit()
    for skill in skills:
        db.add(
            SkillEvidence(
                skill=skill,
                repo_id=repo.id,
                evidence_type="manifest",
                weight=1.0,
                confidence=1.0,
                source_files_json=[],
            )
        )
    db.commit()
    db.close()


class _Group:
    """Stands in for app/api/skills.py's SkillGroup, which build_layout
    only reads five attributes off."""

    def __init__(self, name: str, starred: bool = False):
        self.name = name
        self.starred = starred
        self.manual_skill_id = None
        self.sources = []


def _fake_embed(texts, **kwargs):
    """Distinct directions per text, so the projection has real
    neighbourhoods to preserve instead of one degenerate point."""
    return [[math.cos(i * 0.7), math.sin(i * 0.7), (i % 4) * 0.05] for i, _ in enumerate(texts)]


def test_skill_text_carries_evidence_context(tmp_path):
    _reset_db(tmp_path)
    account_id = _account()
    _repo(account_id, "api", "Python", ["FastAPI", "SQLAlchemy", "PostgreSQL"])

    contexts = skill_map.skill_contexts(account_id)
    text = skill_map.skill_text("FastAPI", contexts["fastapi"])

    assert "FastAPI" in text
    assert "Python" in text
    # The co-occurring skills are the part that separates a name from
    # whatever else happens to be spelled like it.
    assert "SQLAlchemy" in text
    assert "PostgreSQL" in text


def test_skill_text_without_context_still_types_the_name(tmp_path):
    _reset_db(tmp_path)
    text = skill_map.skill_text("Rust", None)
    assert text.startswith("Rust,")
    assert "software engineering" in text


def test_contexts_do_not_leak_between_containers(tmp_path):
    _reset_db(tmp_path)
    account_id = _account()
    _repo(account_id, "web", "TypeScript", ["React", "Vite"])
    _repo(account_id, "model", "Python", ["PyTorch", "NumPy"])

    contexts = skill_map.skill_contexts(account_id)

    assert "Vite" in contexts["react"].co_skills
    assert "PyTorch" not in contexts["react"].co_skills


def test_experience_evidence_contributes_context(tmp_path):
    _reset_db(tmp_path)
    account_id = _account()
    db = get_db()
    experience = Experience(account_id=account_id, title="Engineer", company="Acme")
    db.add(experience)
    db.commit()
    for skill in ("Kubernetes", "Terraform"):
        db.add(
            ExperienceSkillEvidence(
                skill=skill,
                experience_id=experience.id,
                evidence_type="manual",
                weight=1.0,
                confidence=1.0,
                source_files_json=[],
            )
        )
    db.commit()
    db.close()

    contexts = skill_map.skill_contexts(account_id)
    assert "Terraform" in contexts["kubernetes"].co_skills


def test_fingerprint_changes_with_text_and_is_stable_otherwise():
    first = skill_map.layout_fingerprint(["Python, a technology", "React, a technology"])
    same = skill_map.layout_fingerprint(["Python, a technology", "React, a technology"])
    different = skill_map.layout_fingerprint(["Python, a technology", "Vue, a technology"])

    assert first == same
    assert first != different


def test_build_layout_places_every_skill_and_labels_clusters(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _account()
    monkeypatch.setattr("app.profile.skill_map.embed", _fake_embed)

    groups = [_Group(name) for name in ("Python", "React", "Docker", "PyTorch", "Redis", "Vite")]
    payload = skill_map.build_layout(account_id, groups)

    assert len(payload["nodes"]) == len(groups)
    assert {n["name"] for n in payload["nodes"]} == {g.name for g in groups}
    assert payload["clusters"]

    cluster_ids = {c["id"] for c in payload["clusters"]}
    for node in payload["nodes"]:
        assert node["cluster_id"] in cluster_ids
        assert isinstance(node["x"], float)
        assert isinstance(node["y"], float)

    # A cluster is named after one of its own members, not an index.
    names = {g.name for g in groups}
    for cluster in payload["clusters"]:
        assert cluster["label"] in names
        assert cluster["count"] == sum(
            1 for n in payload["nodes"] if n["cluster_id"] == cluster["id"]
        )


def test_projection_handles_tiny_profiles(monkeypatch):
    import numpy as np

    for n in (1, 2, 3, 4, 5):
        vectors = np.array([[math.cos(i), math.sin(i), 0.1 * i] for i in range(n)])
        coords = skill_map.project_2d(vectors)
        assert coords.shape == (n, 2)
        assert np.isfinite(coords).all()


def test_cached_layout_is_returned_and_replaced(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _account()
    monkeypatch.setattr("app.profile.skill_map.embed", _fake_embed)

    groups = [_Group(name) for name in ("Python", "React", "Docker", "PyTorch")]
    texts = [skill_map.skill_text(g.name, None) for g in groups]
    fingerprint = skill_map.layout_fingerprint(texts)

    assert skill_map.load_cached(account_id, fingerprint) is None

    payload = skill_map.build_layout(account_id, groups)
    skill_map.store_cached(account_id, fingerprint, payload)
    assert skill_map.load_cached(account_id, fingerprint) == payload

    # A different skill set must not be served the old layout, and must
    # not accumulate a second row for the same account.
    assert skill_map.load_cached(account_id, "some-other-fingerprint") is None
    skill_map.store_cached(account_id, "some-other-fingerprint", {"clusters": [], "nodes": []})
    assert skill_map.load_cached(account_id, fingerprint) is None

    db = get_db()
    rows = db.execute(
        select(SkillMapCache).where(SkillMapCache.account_id == account_id)
    ).all()
    db.close()
    assert len(rows) == 1


def test_map_endpoint_serves_the_cache_without_embedding(tmp_path, monkeypatch):
    """The second request must not embed anything: that is the whole
    point of storing the layout."""
    _reset_db(tmp_path)
    account_id = _account()
    _repo(account_id, "api", "Python", ["FastAPI", "SQLAlchemy", "PostgreSQL", "Redis"])

    calls = {"n": 0}

    def counting_embed(texts, **kwargs):
        calls["n"] += 1
        return _fake_embed(texts)

    monkeypatch.setattr("app.profile.skill_map.embed", counting_embed)

    from fastapi.testclient import TestClient

    from app.api.main import app

    client = TestClient(app)
    first = client.get(f"/api/skills/map?account_id={account_id}")
    second = client.get(f"/api/skills/map?account_id={account_id}")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert calls["n"] == 1


def test_tiny_profiles_get_one_cluster():
    import numpy as np

    coords = np.array([[0.0, 0.0], [10.0, 10.0]])
    labels, n_clusters = skill_map.cluster(coords, coords)

    assert n_clusters == 1
    assert set(labels.tolist()) == {0}


def test_cluster_label_falls_back_when_a_group_is_empty():
    import numpy as np

    assert skill_map.cluster_label([], np.zeros((0, 2)), []) == "Group"


def test_layout_of_a_single_skill_is_placed_at_the_origin(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _account()
    monkeypatch.setattr("app.profile.skill_map.embed", _fake_embed)

    payload = skill_map.build_layout(account_id, [_Group("Python")])

    assert payload["nodes"][0]["x"] == 0.0
    assert payload["nodes"][0]["y"] == 0.0
    assert payload["clusters"] == [{"id": 0, "label": "Python", "count": 1}]
