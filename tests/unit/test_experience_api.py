from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import Account, get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


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


def _mock_embed(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )


def test_create_and_get_experience(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().post(
        "/api/experience",
        json={"account_id": account_id, "title": "Engineer", "company": "Acme"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "Engineer"
    assert body["company"] == "Acme"
    assert body["skills"] == []


def test_create_experience_requires_title_and_company(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    assert client.post(
        "/api/experience", json={"account_id": account_id, "title": "  ", "company": "Acme"}
    ).status_code == 422
    assert client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "  "}
    ).status_code == 422


def test_list_experience_scoped_to_account(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account(github_username="a")
    account_b = _make_account(github_username="b")
    client = _client()
    client.post("/api/experience", json={"account_id": account_a, "title": "A", "company": "X"})
    client.post("/api/experience", json={"account_id": account_b, "title": "B", "company": "Y"})

    resp = client.get(f"/api/experience?account_id={account_a}")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["title"] == "A"


def test_update_experience_partial(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()

    resp = client.patch(f"/api/experience/{created['id']}", json={"location": "Remote"})

    assert resp.status_code == 200
    assert resp.json()["location"] == "Remote"
    assert resp.json()["title"] == "Eng"  # untouched


def test_update_experience_unknown_id_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().patch("/api/experience/999", json={"location": "Remote"})
    assert resp.status_code == 404


def test_add_skill_to_experience_and_index(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()

    resp = client.post(
        f"/api/experience/{created['id']}/skills", json={"skill": "Vector databases"}
    )

    assert resp.status_code == 200
    assert len(resp.json()["skills"]) == 1
    assert resp.json()["skills"][0]["skill"] == "Vector databases"

    from app.retrieval.index import COLLECTION

    client_qdrant = vectorstore_module.get_client()
    assert client_qdrant.count(COLLECTION).count == 1


def test_add_skill_rejects_blank(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()

    resp = client.post(f"/api/experience/{created['id']}/skills", json={"skill": "   "})
    assert resp.status_code == 422


def test_update_and_delete_skill(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    added = client.post(
        f"/api/experience/{created['id']}/skills", json={"skill": "Python"}
    ).json()
    skill_id = added["skills"][0]["id"]

    updated = client.patch(
        f"/api/experience/{created['id']}/skills/{skill_id}", json={"skill": "Rust"}
    )
    assert updated.status_code == 200
    assert updated.json()["skills"][0]["skill"] == "Rust"

    deleted = client.delete(f"/api/experience/{created['id']}/skills/{skill_id}")
    assert deleted.status_code == 200
    assert deleted.json()["skills"] == []


def test_skill_scoped_to_owning_experience(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    exp_a = client.post(
        "/api/experience", json={"account_id": account_id, "title": "A", "company": "X"}
    ).json()
    exp_b = client.post(
        "/api/experience", json={"account_id": account_id, "title": "B", "company": "Y"}
    ).json()
    added = client.post(f"/api/experience/{exp_a['id']}/skills", json={"skill": "Go"}).json()
    skill_id = added["skills"][0]["id"]

    resp = client.delete(f"/api/experience/{exp_b['id']}/skills/{skill_id}")
    assert resp.status_code == 404


def test_delete_experience_cascades_skills(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    client.post(f"/api/experience/{created['id']}/skills", json={"skill": "Python"})

    resp = client.delete(f"/api/experience/{created['id']}")

    assert resp.status_code == 200
    assert client.get(f"/api/experience/{created['id']}").status_code == 404


def test_delete_experience_unknown_id_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().delete("/api/experience/999")
    assert resp.status_code == 404


def test_add_point_to_experience_and_index(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    assert created["points"] == []

    resp = client.post(
        f"/api/experience/{created['id']}/points", json={"text": "Led team of 5"}
    )

    assert resp.status_code == 200
    assert len(resp.json()["points"]) == 1
    assert resp.json()["points"][0]["text"] == "Led team of 5"

    from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION

    client_qdrant = vectorstore_module.get_client()
    assert client_qdrant.count(EXPERIENCE_POINTS_COLLECTION).count == 1


def test_add_point_rejects_blank(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()

    resp = client.post(f"/api/experience/{created['id']}/points", json={"text": "   "})
    assert resp.status_code == 422


def test_points_keep_insertion_order(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    client.post(f"/api/experience/{created['id']}/points", json={"text": "First"})
    resp = client.post(f"/api/experience/{created['id']}/points", json={"text": "Second"})

    texts = [p["text"] for p in resp.json()["points"]]
    assert texts == ["First", "Second"]


def test_update_and_delete_point(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    added = client.post(
        f"/api/experience/{created['id']}/points", json={"text": "Original"}
    ).json()
    point_id = added["points"][0]["id"]

    updated = client.patch(
        f"/api/experience/{created['id']}/points/{point_id}", json={"text": "Revised"}
    )
    assert updated.status_code == 200
    assert updated.json()["points"][0]["text"] == "Revised"

    deleted = client.delete(f"/api/experience/{created['id']}/points/{point_id}")
    assert deleted.status_code == 200
    assert deleted.json()["points"] == []

    from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION

    client_qdrant = vectorstore_module.get_client()
    assert client_qdrant.count(EXPERIENCE_POINTS_COLLECTION).count == 0


def test_point_scoped_to_owning_experience(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    exp_a = client.post(
        "/api/experience", json={"account_id": account_id, "title": "A", "company": "X"}
    ).json()
    exp_b = client.post(
        "/api/experience", json={"account_id": account_id, "title": "B", "company": "Y"}
    ).json()
    added = client.post(f"/api/experience/{exp_a['id']}/points", json={"text": "Point"}).json()
    point_id = added["points"][0]["id"]

    resp = client.delete(f"/api/experience/{exp_b['id']}/points/{point_id}")
    assert resp.status_code == 404


def test_delete_experience_cascades_points(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _mock_embed(monkeypatch)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    client.post(f"/api/experience/{created['id']}/points", json={"text": "Point"})

    resp = client.delete(f"/api/experience/{created['id']}")

    assert resp.status_code == 200
    assert client.get(f"/api/experience/{created['id']}").status_code == 404

    from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION

    client_qdrant = vectorstore_module.get_client()
    assert client_qdrant.count(EXPERIENCE_POINTS_COLLECTION).count == 0


def test_exclude_from_resume_round_trips_and_survives_other_edits(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/experience", json={"account_id": account_id, "title": "Eng", "company": "Acme"}
    ).json()
    assert created["exclude_from_resume"] is False

    resp = client.patch(f"/api/experience/{created['id']}", json={"exclude_from_resume": True})
    assert resp.status_code == 200
    assert resp.json()["exclude_from_resume"] is True

    client.patch(f"/api/experience/{created['id']}", json={"title": "Engineer"})
    client.patch(f"/api/experience/{created['id']}", json={"exclude_from_resume": None})
    (row,) = client.get(f"/api/experience?account_id={account_id}").json()
    assert (row["title"], row["exclude_from_resume"]) == ("Engineer", True)


def test_dates_stored_as_month_and_year_on_create_and_edit(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    created = client.post(
        "/api/experience",
        json={
            "account_id": account_id, "title": "Engineer", "company": "Acme",
            "start_date": "2026-01-01", "end_date": "03/2028",
        },
    ).json()
    assert (created["start_date"], created["end_date"]) == ("Jan 2026", "Mar 2028")

    edited = client.patch(f"/api/experience/{created['id']}", json={"end_date": None}).json()
    assert edited["end_date"] is None

    bad = client.post(
        "/api/experience",
        json={"account_id": account_id, "title": "E", "company": "A", "start_date": "Foo 2020"},
    )
    assert bad.status_code == 422
