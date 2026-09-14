from pathlib import Path

import app.core.db as db_module
from app.core.db import get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _make_account() -> int:
    from app.core.db import Account

    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def test_create_and_list_education(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.post(
        "/api/education",
        json={
            "account_id": account_id,
            "institution": "State University",
            "degree": "B.Eng in Computer Science",
            "location": "Remote",
            "start_date": "2020-08-01",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["institution"] == "State University"
    assert body["end_date"] is None

    listed = client.get(f"/api/education?account_id={account_id}").json()
    assert len(listed) == 1
    assert listed[0]["degree"] == "B.Eng in Computer Science"


def test_create_rejects_blank_institution_or_degree(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp1 = client.post(
        "/api/education",
        json={"account_id": account_id, "institution": "  ", "degree": "BA"},
    )
    assert resp1.status_code == 422

    resp2 = client.post(
        "/api/education",
        json={"account_id": account_id, "institution": "State University", "degree": "  "},
    )
    assert resp2.status_code == 422


def test_update_education(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/education",
        json={"account_id": account_id, "institution": "State University", "degree": "BA"},
    ).json()

    resp = client.patch(f"/api/education/{created['id']}", json={"degree": "BSc"})
    assert resp.status_code == 200
    assert resp.json()["degree"] == "BSc"


def test_update_unknown_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().patch("/api/education/999999", json={"degree": "BSc"})
    assert resp.status_code == 404


def test_delete_education(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/education",
        json={"account_id": account_id, "institution": "State University", "degree": "BA"},
    ).json()

    resp = client.delete(f"/api/education/{created['id']}")
    assert resp.status_code == 200
    assert client.get(f"/api/education?account_id={account_id}").json() == []


def test_delete_unknown_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().delete("/api/education/999999")
    assert resp.status_code == 404


def test_education_scoped_to_account(tmp_path):
    _reset_db(tmp_path)
    from app.core.db import Account

    account_a = _make_account()
    db = get_db()
    b = Account(first_name="Grace", last_name="Hopper", github_username="ghopper")
    db.add(b)
    db.commit()
    db.refresh(b)
    account_b = b.id
    db.close()

    client = _client()
    client.post(
        "/api/education",
        json={"account_id": account_a, "institution": "A Uni", "degree": "BA"},
    )
    client.post(
        "/api/education",
        json={"account_id": account_b, "institution": "B Uni", "degree": "BSc"},
    )

    listed_a = client.get(f"/api/education?account_id={account_a}").json()
    assert len(listed_a) == 1
    assert listed_a[0]["institution"] == "A Uni"


def test_delete_account_removes_education(tmp_path):
    _reset_db(tmp_path)
    from app.core.db import Education

    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()
    client.post(
        "/api/education",
        json={"account_id": account["id"], "institution": "State University", "degree": "BA"},
    )

    resp = client.delete(f"/accounts/{account['id']}")
    assert resp.status_code == 200

    db = get_db()
    assert db.query(Education).filter_by(account_id=account["id"]).count() == 0
    db.close()
