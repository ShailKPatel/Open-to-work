"""app/core/auth_sources_store.py + app/api/auth_sources.py: CRUD for
authenticated job-source login profiles. Real crypto (app/core/crypto.py),
real DB, only Playwright itself (app/ingest/jobs/auth_fetch.py) is
exercised separately in test_auth_fetch.py; test-login here is monkeypatched
at the router boundary, same posture as other endpoint tests in this suite.
"""

from pathlib import Path

import app.core.db as db_module
from app.core.auth_sources_store import (
    add_source,
    delete_source,
    list_sources,
    resolve_credentials,
    set_enabled,
)
from app.core.db import Account, get_db, init_db
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
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


_KWARGS = dict(
    label="Wellfound (alt)",
    site_domain="wellfound.com",
    login_url="https://wellfound.com/login",
    username_selector="#user",
    password_selector="#pass",
    submit_selector="#submit",
    post_login_wait_selector=None,
    username="me@example.com",
    password="hunter2",
    acknowledged_risk=True,
)


def test_add_source_requires_acknowledged_risk(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    kwargs = {**_KWARGS, "acknowledged_risk": False}
    try:
        add_source(account_id=account_id, **kwargs)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_add_source_masks_username_never_stores_plaintext(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)
    assert row["masked_username"] != "me@example.com"
    assert "me@example.com" not in str(row)


def test_resolve_credentials_decrypts_real_username_and_password(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)
    resolved = resolve_credentials(row["id"])
    assert resolved is not None
    _, credentials = resolved
    assert credentials == {"username": "me@example.com", "password": "hunter2"}


def test_resolve_credentials_none_when_disabled(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)
    set_enabled(row["id"], False)
    assert resolve_credentials(row["id"]) is None


def test_list_sources_scoped_by_account(tmp_path):
    _reset_db(tmp_path)
    a1 = _make_account()
    a2 = _make_account()
    add_source(account_id=a1, **_KWARGS)
    add_source(account_id=a2, **{**_KWARGS, "label": "Other account's source"})

    assert len(list_sources(a1)) == 1
    assert len(list_sources(a2)) == 1


def test_delete_source_removes_it(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)
    delete_source(row["id"])
    assert list_sources(account_id) == []


def test_create_via_api_rejects_missing_acknowledgment(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    resp = client.post(
        "/api/auth-sources",
        json={"account_id": account_id, **{**_KWARGS, "acknowledged_risk": False}},
    )
    assert resp.status_code == 422


def test_create_via_api_succeeds_with_acknowledgment(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    resp = client.post("/api/auth-sources", json={"account_id": account_id, **_KWARGS})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "Wellfound (alt)"
    assert "password" not in body


def test_api_test_login_maps_login_failure_to_502(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)

    from app.ingest.jobs.auth_fetch import AuthLoginFailedError

    def _raise(*args, **kwargs):
        raise AuthLoginFailedError("bad selector")

    monkeypatch.setattr("app.api.auth_sources.test_login", _raise)

    client = _client()
    resp = client.post(f"/api/auth-sources/{row['id']}/test-login")
    assert resp.status_code == 502


def test_api_delete_and_enabled_toggle(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    row = add_source(account_id=account_id, **_KWARGS)
    client = _client()

    resp = client.patch(f"/api/auth-sources/{row['id']}/enabled", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    resp = client.delete(f"/api/auth-sources/{row['id']}")
    assert resp.status_code == 200
    assert client.get(f"/api/auth-sources?account_id={account_id}").json() == []
