from pathlib import Path

import app.core.db as db_module
from app.core.db import Account, get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
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


def test_get_contact_defaults_to_null(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().get(f"/api/accounts/{account_id}/contact")

    assert resp.status_code == 200
    assert resp.json() == {
        "account_id": account_id,
        "first_name": "Ada",
        "last_name": "Lovelace",
        "contact_email": None,
        "contact_phone": None,
        "contact_location": None,
        "emails": [],
        "phones": [],
        "github_username": "octocat",
    }


def test_get_contact_github_username_null_when_account_has_none(tmp_path):
    """github_username is surfaced read-only from Account (see
    app/api/contact.py's ContactInfo docstring). An empty string on the
    account (no GitHub at signup) reports as null, not "", so the contact
    page's GitHub quick-link only renders when there's really a username.
    """
    _reset_db(tmp_path)
    account_id = _make_account(github_username="")

    resp = _client().get(f"/api/accounts/{account_id}/contact")

    assert resp.status_code == 200
    assert resp.json()["github_username"] is None


def test_get_contact_unknown_account_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().get("/api/accounts/999/contact")
    assert resp.status_code == 404


def test_patch_contact_partial_update(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.patch(f"/api/accounts/{account_id}/contact", json={"contact_email": "a@b.com"})
    assert resp.status_code == 200
    assert resp.json()["contact_email"] == "a@b.com"
    assert resp.json()["contact_phone"] is None

    resp2 = client.patch(f"/api/accounts/{account_id}/contact", json={"contact_phone": "555"})
    assert resp2.status_code == 200
    # earlier field untouched by this second, unrelated patch
    assert resp2.json()["contact_email"] == "a@b.com"
    assert resp2.json()["contact_phone"] == "555"


def test_add_list_update_delete_social_link(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    add = client.post(
        f"/api/accounts/{account_id}/social-links",
        json={"platform": "linkedin", "url": "https://linkedin.com/in/x"},
    )
    assert add.status_code == 200
    link_id = add.json()["id"]

    listed = client.get(f"/api/accounts/{account_id}/social-links")
    assert len(listed.json()) == 1

    updated = client.patch(
        f"/api/accounts/{account_id}/social-links/{link_id}",
        json={"url": "https://linkedin.com/in/y"},
    )
    assert updated.status_code == 200
    assert updated.json()["url"] == "https://linkedin.com/in/y"
    assert updated.json()["platform"] == "linkedin"  # untouched

    deleted = client.delete(f"/api/accounts/{account_id}/social-links/{link_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/accounts/{account_id}/social-links").json() == []


def test_add_social_link_rejects_blank_url(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resp = _client().post(
        f"/api/accounts/{account_id}/social-links",
        json={"platform": "linkedin", "url": "   "},
    )
    assert resp.status_code == 422


def test_social_link_scoped_to_owning_account(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account(github_username="a")
    account_b = _make_account(github_username="b")
    client = _client()

    add = client.post(
        f"/api/accounts/{account_a}/social-links",
        json={"platform": "github", "url": "https://github.com/a"},
    )
    link_id = add.json()["id"]

    resp = client.delete(f"/api/accounts/{account_b}/social-links/{link_id}")
    assert resp.status_code == 404


def test_multiple_emails_and_phones_with_primary(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    # Add email 1 (becomes primary by default)
    e1 = client.post(
        f"/api/accounts/{account_id}/emails", json={"email": "primary@example.com"}
    ).json()
    assert e1["is_primary"] is True

    # Add email 2 (not primary)
    e2 = client.post(
        f"/api/accounts/{account_id}/emails", json={"email": "secondary@example.com"}
    ).json()
    assert e2["is_primary"] is False

    # Check contact endpoint return
    c = client.get(f"/api/accounts/{account_id}/contact").json()
    assert c["contact_email"] == "primary@example.com"
    assert len(c["emails"]) == 2

    # Switch primary to e2
    up = client.patch(
        f"/api/accounts/{account_id}/emails/{e2['id']}", json={"is_primary": True}
    ).json()
    assert up["is_primary"] is True

    c2 = client.get(f"/api/accounts/{account_id}/contact").json()
    assert c2["contact_email"] == "secondary@example.com"

    # Add phone 1 (primary by default), then phone 2 asking for primary,
    # which has to demote phone 1.
    p1 = client.post(f"/api/accounts/{account_id}/phones", json={"phone": "+1-555-0100"}).json()
    assert p1["is_primary"] is True
    p2 = client.post(
        f"/api/accounts/{account_id}/phones",
        json={"phone": "+1-555-0200", "is_primary": True},
    ).json()
    assert p2["is_primary"] is True

    c3 = client.get(f"/api/accounts/{account_id}/contact").json()
    assert c3["contact_phone"] == "+1-555-0200"
    assert len(c3["phones"]) == 2

