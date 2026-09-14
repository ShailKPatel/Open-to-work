
from github import UnknownObjectException

import app.core.db as db_module
from app.core.db import init_db
from app.core.settings import get_settings


def _reset_db(tmp_path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["RESUME_STORAGE_DIR"] = str(tmp_path / "resumes")
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _make_account(client, github_username="octocat"):
    return client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": github_username},
    ).json()


def test_account_creation_seeds_one_source(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()

    assert len(sources) == 1
    assert sources[0]["kind"] == "user"
    assert sources[0]["github_username"] == "octocat"
    assert sources[0]["raw_input"] == "octocat"
    assert sources[0]["last_synced_at"] is None


def test_add_source_bare_username(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    resp = client.post("/api/sources", json={"account_id": account["id"], "raw_input": "torvalds"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "user"
    assert body["github_username"] == "torvalds"
    assert body["repo_full_name"] is None

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    assert len(sources) == 2  # seeded one + this one


def test_add_source_profile_url(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    resp = client.post(
        "/api/sources",
        json={"account_id": account["id"], "raw_input": "https://github.com/torvalds"},
    )
    body = resp.json()
    assert body["kind"] == "user"
    assert body["github_username"] == "torvalds"


def test_add_source_repo_url(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    resp = client.post(
        "/api/sources",
        json={
            "account_id": account["id"],
            "raw_input": "https://github.com/octocat/Hello-World",
        },
    )
    body = resp.json()
    assert body["kind"] == "repo"
    assert body["github_username"] == "octocat"
    assert body["repo_full_name"] == "octocat/Hello-World"


def test_add_source_invalid_input_rejected(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    resp = client.post(
        "/api/sources",
        json={"account_id": account["id"], "raw_input": "not a valid thing at all!!"},
    )
    assert resp.status_code == 422

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    assert len(sources) == 1  # only the auto-seeded one; bad input wasn't added


def test_add_duplicate_source_rejected(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)  # seeds "octocat" already

    resp = client.post(
        "/api/sources", json={"account_id": account["id"], "raw_input": "octocat"}
    )
    assert resp.status_code == 409

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    assert len(sources) == 1  # still just the one


def test_add_duplicate_source_detected_across_input_forms(tmp_path):
    """"octocat" and "https://github.com/octocat" are the same target:
    dedup compares the parsed target, not the raw string."""
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)  # seeds "octocat" already

    resp = client.post(
        "/api/sources",
        json={"account_id": account["id"], "raw_input": "https://github.com/octocat"},
    )
    assert resp.status_code == 409


def test_duplicate_check_does_not_confuse_user_and_repo_kind(tmp_path):
    """"octocat" (user) and "octocat/Hello-World" (repo) share a username
    but are different targets; adding both must succeed."""
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)  # seeds "octocat" (user) already

    resp = client.post(
        "/api/sources",
        json={
            "account_id": account["id"],
            "raw_input": "https://github.com/octocat/Hello-World",
        },
    )
    assert resp.status_code == 200

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    assert len(sources) == 2


def test_duplicate_check_scoped_per_account(tmp_path):
    """Two different accounts can each add the same username; dedup is
    per-account, not global."""
    _reset_db(tmp_path)
    client = _client()
    a = _make_account(client, github_username="octocat")
    b = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
    ).json()

    resp = client.post("/api/sources", json={"account_id": b["id"], "raw_input": "octocat"})
    assert resp.status_code == 200

    sources_a = client.get(f"/api/sources?account_id={a['id']}").json()
    sources_b = client.get(f"/api/sources?account_id={b['id']}").json()
    assert len(sources_a) == 1
    assert len(sources_b) == 2


def test_add_source_unknown_account_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()

    resp = client.post("/api/sources", json={"account_id": 999999, "raw_input": "octocat"})
    assert resp.status_code == 404


def test_delete_source(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    added = client.post(
        "/api/sources", json={"account_id": account["id"], "raw_input": "torvalds"}
    ).json()

    resp = client.delete(f"/api/sources/{added['id']}")
    assert resp.status_code == 200

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    assert len(sources) == 1  # back to just the seeded one


def test_delete_unknown_source_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.delete("/api/sources/999999")
    assert resp.status_code == 404


def test_deleting_account_removes_its_sources(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    client.delete(f"/accounts/{account['id']}")

    resp = client.get(f"/api/sources?account_id={account['id']}")
    assert resp.json() == []


def test_sync_source_stream_user_kind(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    seen = {}

    def fake_progress(username, include_forks=True, client=None, account_id=None, run_id=None):
        seen["username"] = username
        seen["account_id"] = account_id
        yield {"stage": "checking_profile", "username": username}
        yield {"stage": "done", "total_repos": 0, "fetched": 0, "cache_hits": 0}

    monkeypatch.setattr("app.api.sources.sync_account_progress", fake_progress)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    with client.stream("GET", f"/api/sources/{source_id}/sync/stream") as resp:
        body = "".join(resp.iter_text())

    assert seen["username"] == "octocat"
    assert seen["account_id"] == account["id"]
    assert '"stage": "done"' in body

    refreshed = client.get(f"/api/sources?account_id={account['id']}").json()
    assert refreshed[0]["last_synced_at"] is not None  # marked synced on clean finish


def test_sync_source_stream_repo_kind(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    added = client.post(
        "/api/sources",
        json={
            "account_id": account["id"],
            "raw_input": "https://github.com/octocat/Hello-World",
        },
    ).json()
    seen = {}

    def fake_single_progress(
        repo_full_name, attribution_username, client=None, account_id=None, run_id=None
    ):
        seen["repo_full_name"] = repo_full_name
        seen["attribution_username"] = attribution_username
        yield {"stage": "checking_profile", "username": attribution_username}
        yield {"stage": "listing_repos", "total_hint": 1}
        yield {
            "stage": "repo_progress",
            "index": 1,
            "total_hint": 1,
            "name": repo_full_name,
            "cache_hit": False,
        }
        yield {"stage": "done", "total_repos": 1, "fetched": 1, "cache_hits": 0}

    monkeypatch.setattr("app.api.sources.sync_single_repo_progress", fake_single_progress)

    with client.stream("GET", f"/api/sources/{added['id']}/sync/stream") as resp:
        body = "".join(resp.iter_text())

    assert seen["repo_full_name"] == "octocat/Hello-World"
    assert seen["attribution_username"] == "octocat"  # the account's own username, not "torvalds"
    assert '"stage": "done"' in body


def test_sync_source_stream_not_found_sends_error_and_does_not_mark_synced(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    def fake_progress(username, include_forks=True, client=None, account_id=None, run_id=None):
        raise UnknownObjectException(404, "Not Found", {})
        yield  # pragma: no cover - unreachable, makes this a generator

    monkeypatch.setattr("app.api.sources.sync_account_progress", fake_progress)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    with client.stream("GET", f"/api/sources/{source_id}/sync/stream") as resp:
        body = "".join(resp.iter_text())

    assert '"stage": "error"' in body

    refreshed = client.get(f"/api/sources?account_id={account['id']}").json()
    assert refreshed[0]["last_synced_at"] is None  # not marked synced, it failed


def test_sync_source_stream_rate_limited_does_not_mark_synced(tmp_path, monkeypatch):
    """A "rate_limited" partial batch is a graceful generator stop (no
    exception) and must not be treated as a clean "done" finish. See
    app/api/sources.py's events(): only a literal "done" stage marks
    last_synced_at."""
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)

    def fake_progress(username, include_forks=True, client=None, account_id=None, run_id=None):
        yield {"stage": "checking_profile", "username": username}
        yield {"stage": "listing_repos", "total_hint": 8}
        yield {
            "stage": "rate_limited",
            "completed": 3,
            "total_hint": 8,
            "detail": "GitHub may be rate-limiting us: 3 of 8 repos saved.",
        }

    monkeypatch.setattr("app.api.sources.sync_account_progress", fake_progress)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    with client.stream("GET", f"/api/sources/{source_id}/sync/stream") as resp:
        body = "".join(resp.iter_text())

    assert '"stage": "rate_limited"' in body
    assert "3 of 8" in body

    refreshed = client.get(f"/api/sources?account_id={account['id']}").json()
    assert refreshed[0]["last_synced_at"] is None  # partial, stays retryable


def test_sync_source_stream_forwards_client_run_id(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    seen = {}

    def fake_progress(username, include_forks=True, client=None, account_id=None, run_id=None):
        seen["run_id"] = run_id
        yield {"stage": "done", "total_repos": 0, "fetched": 0, "cache_hits": 0}

    monkeypatch.setattr("app.api.sources.sync_account_progress", fake_progress)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    with client.stream(
        "GET", f"/api/sources/{source_id}/sync/stream?run_id=attempt-xyz"
    ) as resp:
        list(resp.iter_text())

    assert seen["run_id"] == "attempt-xyz"  # client-supplied, not the source's own id


def test_sync_source_stream_defaults_run_id_to_source_id(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    seen = {}

    def fake_progress(username, include_forks=True, client=None, account_id=None, run_id=None):
        seen["run_id"] = run_id
        yield {"stage": "done", "total_repos": 0, "fetched": 0, "cache_hits": 0}

    monkeypatch.setattr("app.api.sources.sync_account_progress", fake_progress)

    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    with client.stream("GET", f"/api/sources/{source_id}/sync/stream") as resp:
        list(resp.iter_text())

    assert seen["run_id"] == str(source_id)  # no client run_id given -> falls back


def test_cancel_source_sync_flags_client_run_id(tmp_path):
    from app.ingest.github.cancellation import clear, is_cancelled

    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    resp = client.post(f"/api/sources/{source_id}/sync/cancel?run_id=attempt-xyz")

    assert resp.status_code == 200
    assert is_cancelled("attempt-xyz") is True
    assert is_cancelled(str(source_id)) is False  # not the fallback key
    clear("attempt-xyz")


def test_cancel_source_sync_defaults_to_source_id(tmp_path):
    from app.ingest.github.cancellation import clear, is_cancelled

    _reset_db(tmp_path)
    client = _client()
    account = _make_account(client)
    sources = client.get(f"/api/sources?account_id={account['id']}").json()
    source_id = sources[0]["id"]

    resp = client.post(f"/api/sources/{source_id}/sync/cancel")

    assert resp.status_code == 200
    assert is_cancelled(str(source_id)) is True
    clear(str(source_id))


def test_sync_source_stream_unknown_source_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/api/sources/999999/sync/stream")
    assert resp.status_code == 404
