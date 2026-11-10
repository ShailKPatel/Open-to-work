"""Keys belong to one account: the migration that gave old shared keys an
owner, the first-run signup that stores its key with the new profile, the
monthly budget counted per account, and repo batches kept to one account.
The key routes and store are covered in test_api_keys.py, the public demo's
gate in test_demo.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import app.core.db as db_module
from app.core.db import Account, ApiKey, LLMCall, Repository, get_db, init_db
from app.core.db.engine import get_engine
from app.core.db.migrations import _migrate_api_keys_owner
from app.core.settings import get_settings


@pytest.fixture(autouse=True)
def _db(tmp_path: Path, monkeypatch):
    db_module.reset_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    init_db()
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("valid", "This key is working."),
    )
    yield
    db_module.reset_engine()
    get_settings.cache_clear()


def _accounts(*names: str) -> list[int]:
    db = get_db()
    try:
        rows = [Account(first_name=n, last_name="x", github_username="") for n in names]
        db.add_all(rows)
        db.commit()
        return [r.id for r in rows]
    finally:
        db.close()


def _legacy_key(label: str, allowed: list[int], active: bool = True) -> None:
    """A row as it was before keys had owners."""
    with get_engine().connect() as conn:
        conn.execute(
            text(
                "INSERT INTO api_keys (provider, label, encrypted_credentials, masked_preview, "
                "is_active, enabled, allowed_account_ids, status, created_at) VALUES "
                "('gemini', :label, 'enc', '{}', :active, 1, :allowed, 'valid', "
                "CURRENT_TIMESTAMP)"
            ),
            {"label": label, "active": active, "allowed": json.dumps(allowed)},
        )
        conn.commit()


def _rows() -> list[tuple[str, int | None, bool]]:
    db = get_db()
    try:
        return sorted(
            (k.label, k.account_id, k.is_active) for k in db.query(ApiKey).all()
        )
    finally:
        db.close()


def test_migration_gives_each_account_its_own_copy_of_a_shared_key():
    a, b = _accounts("A", "B")
    _legacy_key("Shared", [])
    _legacy_key("Only B", [b], active=False)

    _migrate_api_keys_owner(get_engine())

    assert _rows() == [("Only B", b, False), ("Shared", a, True), ("Shared", b, True)]
    # Running again changes nothing.
    _migrate_api_keys_owner(get_engine())
    assert len(_rows()) == 3


def test_migration_leaves_one_active_key_per_account_and_provider():
    (a,) = _accounts("A")
    _legacy_key("First", [], active=True)
    _legacy_key("Second", [], active=True)

    _migrate_api_keys_owner(get_engine())

    assert _rows() == [("First", a, True), ("Second", a, False)]


def test_migration_with_no_accounts_leaves_keys_unowned():
    _legacy_key("Orphan", [])
    _migrate_api_keys_owner(get_engine())
    assert _rows() == [("Orphan", None, True)]


def test_migration_drops_owners_that_no_longer_exist():
    (a,) = _accounts("A")
    _legacy_key("Gone and kept", [a, 9999])
    _migrate_api_keys_owner(get_engine())
    assert _rows() == [("Gone and kept", a, True)]


def _client() -> TestClient:
    from app.api.main import app

    return TestClient(app)


def test_signup_stores_the_wizard_key_as_the_new_profiles_own():
    from app.core import api_keys_store

    response = _client().post(
        "/accounts",
        data={
            "first_name": "Ada", "last_name": "L", "github_username": "",
            "api_key_provider": "gemini", "api_key": "AIza-signup",
        },
    )
    assert response.status_code == 200
    account_id = response.json()["id"]
    [key] = api_keys_store.list_keys(account_id)
    assert key["provider"] == "gemini" and key["is_active"] is True
    assert "AIza-signup" not in str(key)
    assert api_keys_store.resolve_dispatch_keys("gemini", account_id)[0].credentials == {
        "api_key": "AIza-signup"
    }


def test_signup_with_a_refused_key_still_makes_the_profile(monkeypatch):
    from app.core import api_keys_store

    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("invalid", "Rejected."),
    )
    response = _client().post(
        "/accounts",
        data={
            "first_name": "Ada", "last_name": "L", "github_username": "",
            "api_key_provider": "gemini", "api_key": "bad",
        },
    )
    assert response.status_code == 200
    assert api_keys_store.list_keys(response.json()["id"]) == []


def test_signup_ignores_an_unknown_provider():
    from app.core import api_keys_store

    response = _client().post(
        "/accounts",
        data={
            "first_name": "Ada", "last_name": "L", "github_username": "",
            "api_key_provider": "nope", "api_key": "x",
        },
    )
    assert response.status_code == 200
    assert api_keys_store.list_keys() == []


def test_deleting_a_profile_deletes_its_keys():
    from app.core import api_keys_store

    a, b = _accounts("A", "B")
    api_keys_store.add_key("gemini", "A's", {"api_key": "k"}, None, account_id=a)
    api_keys_store.add_key("gemini", "B's", {"api_key": "k"}, None, account_id=b)

    assert _client().delete(f"/accounts/{a}").status_code == 200

    assert [k["label"] for k in api_keys_store.list_keys()] == ["B's"]


def test_monthly_budget_counts_each_accounts_own_spend():
    from app.core.app_settings import update_llm_settings
    from app.core.llm import BudgetExceededError, complete

    a, b = _accounts("A", "B")
    update_llm_settings(monthly_budget_usd=1.0)
    db = get_db()
    try:
        db.add(
            LLMCall(
                tier="bulk", model="gemini/x", prompt_hash="h", tokens_in=1, tokens_out=1,
                cost_usd=5.0, latency_ms=1, cached=False, account_id=a,
            )
        )
        db.commit()
    finally:
        db.close()

    def answer(**kwargs):
        message = MagicMock(content="ok")
        return MagicMock(choices=[MagicMock(message=message)], usage=None)

    with pytest.raises(BudgetExceededError):
        complete("bulk", [{"role": "user", "content": "a"}], account_id=a, _completion_fn=answer)
    response = complete(
        "bulk", [{"role": "user", "content": "b"}], account_id=b, _completion_fn=answer
    )
    assert response.content == "ok"


def test_repo_batches_never_mix_two_accounts(monkeypatch):
    from app.profile.extract import prefetch_repo_facts

    a, b = _accounts("A", "B")
    repos = []
    for i, owner in enumerate([a, b, a, b], start=1):
        repo = Repository(
            github_id=i, name=f"p{i}", full_name=f"o/p{i}", url=f"https://x/{i}",
            readme=f"Project {i} uses Rust.", manifests_json={}, account_id=owner,
        )
        repo.id = i
        repos.append(repo)
    fake = MagicMock(return_value=MagicMock(parsed={"repos": []}))
    monkeypatch.setattr("app.profile.extract.complete", fake)

    prefetch_repo_facts(repos)

    assert sorted(call.kwargs["account_id"] for call in fake.call_args_list) == [a, b]
