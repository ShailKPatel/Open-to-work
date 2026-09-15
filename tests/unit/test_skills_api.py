from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import Account, Repository, get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _mock_embed(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )


def _make_account(**overrides) -> int:
    db = get_db()
    defaults = dict(first_name="Ada", last_name="Lovelace", github_username="octocat")
    defaults.update(overrides)
    account = Account(**defaults)
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _make_repo(account_id: int, **overrides) -> int:
    db = get_db()
    defaults = dict(
        account_id=account_id, github_id=1, name="proj", full_name="octocat/proj", url=""
    )
    defaults.update(overrides)
    repo = Repository(**defaults)
    db.add(repo)
    db.commit()
    db.refresh(repo)
    repo_id = repo.id
    db.close()
    return repo_id


def test_list_skills_unions_project_experience_and_manual(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()

    repo_id = _make_repo(account_id)
    client.post(f"/api/projects/{repo_id}/skills", json={"skill": "Python"})

    exp = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    client.post(f"/api/experience/{exp['id']}/skills", json={"skill": "Vector databases"})

    client.post("/api/skills", json={"account_id": account_id, "name": "Leadership"})

    resp = client.get(f"/api/skills?account_id={account_id}")

    assert resp.status_code == 200
    names = {g["name"] for g in resp.json()}
    assert names == {"Python", "Vector databases", "Leadership"}

    by_name = {g["name"]: g for g in resp.json()}
    assert by_name["Python"]["sources"][0]["type"] == "project"
    assert by_name["Vector databases"]["sources"][0]["type"] == "experience"
    assert by_name["Leadership"]["sources"] == []
    assert by_name["Leadership"]["manual_skill_id"] is not None


def test_list_skills_groups_by_casefold(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    repo_id = _make_repo(account_id)
    client.post(f"/api/projects/{repo_id}/skills", json={"skill": "python"})
    client.post("/api/skills", json={"account_id": account_id, "name": "Python"})

    resp = client.get(f"/api/skills?account_id={account_id}")

    body = resp.json()
    assert len(body) == 1
    assert body[0]["manual_skill_id"] is not None
    assert len(body[0]["sources"]) == 1


def test_list_skills_scoped_to_account(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account(github_username="a")
    account_b = _make_account(github_username="b")
    client = _client()
    client.post("/api/skills", json={"account_id": account_a, "name": "A-skill"})
    client.post("/api/skills", json={"account_id": account_b, "name": "B-skill"})

    resp = client.get(f"/api/skills?account_id={account_a}")

    assert [g["name"] for g in resp.json()] == ["A-skill"]


def test_create_skill_rejects_blank_name(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resp = _client().post("/api/skills", json={"account_id": account_id, "name": "  "})
    assert resp.status_code == 422


def test_create_duplicate_skill_conflicts(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    client.post("/api/skills", json={"account_id": account_id, "name": "Leadership"})

    resp = client.post("/api/skills", json={"account_id": account_id, "name": "Leadership"})

    assert resp.status_code == 409


def test_delete_manual_skill(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/skills", json={"account_id": account_id, "name": "Leadership"}
    ).json()

    resp = client.delete(f"/api/skills/{created['id']}")

    assert resp.status_code == 200
    assert client.get(f"/api/skills?account_id={account_id}").json() == []


def test_star_skill_marks_group_and_sorts_first(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    repo_id = _make_repo(account_id)
    client.post(f"/api/projects/{repo_id}/skills", json={"skill": "python"})
    client.post("/api/skills", json={"account_id": account_id, "name": "Airflow"})

    # different casing than either source: stars key on casefolded name
    resp = client.post(
        "/api/skills/star", json={"account_id": account_id, "name": " PYTHON ", "starred": True}
    )
    assert resp.status_code == 200

    body = client.get(f"/api/skills?account_id={account_id}").json()
    assert [(g["name"], g["starred"]) for g in body] == [("python", True), ("Airflow", False)]


def test_star_skill_is_idempotent_and_unstars(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    client.post("/api/skills", json={"account_id": account_id, "name": "Leadership"})
    star = {"account_id": account_id, "name": "Leadership", "starred": True}

    assert client.post("/api/skills/star", json=star).status_code == 200
    assert client.post("/api/skills/star", json=star).status_code == 200
    assert client.get(f"/api/skills?account_id={account_id}").json()[0]["starred"] is True

    client.post("/api/skills/star", json={**star, "starred": False})
    assert client.get(f"/api/skills?account_id={account_id}").json()[0]["starred"] is False


def test_star_without_skill_creates_no_group(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    client.post(
        "/api/skills/star", json={"account_id": account_id, "name": "Ghost", "starred": True}
    )

    assert client.get(f"/api/skills?account_id={account_id}").json() == []


def test_star_skill_rejects_blank_name(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resp = _client().post(
        "/api/skills/star", json={"account_id": account_id, "name": " ", "starred": True}
    )
    assert resp.status_code == 422


def test_delete_unknown_skill_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().delete("/api/skills/999")
    assert resp.status_code == 404


def test_get_skills_map_empty(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resp = _client().get(f"/api/skills/map?account_id={account_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["clusters"] == []
    assert data["nodes"] == []


def test_get_skills_map_with_skills(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()

    client.post("/api/skills", json={"account_id": account_id, "name": "Python"})
    client.post("/api/skills", json={"account_id": account_id, "name": "FastAPI"})
    client.post("/api/skills", json={"account_id": account_id, "name": "Docker"})
    client.post("/api/skills", json={"account_id": account_id, "name": "React"})

    resp = client.get(f"/api/skills/map?account_id={account_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 4
    assert len(data["clusters"]) > 0

    names = {n["name"] for n in data["nodes"]}
    assert names == {"Python", "FastAPI", "Docker", "React"}

    for node in data["nodes"]:
        assert "x" in node and isinstance(node["x"], float)
        assert "y" in node and isinstance(node["y"], float)
        assert "cluster_id" in node
        assert "cluster_color" in node

