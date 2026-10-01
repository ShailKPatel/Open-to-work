import datetime as dt
import os
from pathlib import Path

from fastapi.testclient import TestClient

import app.core.db as db_module
from app.api.main import app
from app.core.db import Account, GitHubSyncRun, init_db
from app.core.settings import get_settings
from app.ingest.github import background


def _setup(tmp_path: Path) -> tuple[TestClient, int]:
    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()
    session = db_module.get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="ada")
    session.add(account)
    session.commit()
    account_id = account.id
    session.add_all([
        GitHubSyncRun(
            key="github-sync:user:ada-api", account_id=account_id, kind="user",
            github_username="ada-api", attribution_username="ada", state="rate_limited",
            completed=10, total_hint=23,
            reset_at=dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC),
            resume_at=dt.datetime(2026, 1, 1, 12, 0, 15, tzinfo=dt.UTC),
        ),
        GitHubSyncRun(
            key="github-sync:user:ada-old", account_id=account_id, kind="user",
            github_username="ada-old", attribution_username="ada", state="done",
            completed=5, total_hint=5,
        ),
    ])
    session.commit()
    session.close()
    return TestClient(app), account_id


def test_lists_only_unfinished_syncs_with_utc_times(tmp_path):
    client, account_id = _setup(tmp_path)

    [run] = client.get(f"/api/github-sync?account_id={account_id}").json()

    assert run["label"] == "ada-api"
    assert (run["state"], run["completed"], run["total_hint"]) == ("rate_limited", 10, 23)
    assert run["running"] is False
    assert run["reset_at"] == "2026-01-01T12:00:00Z"


def test_resume_starts_the_same_target_again(tmp_path, monkeypatch):
    client, account_id = _setup(tmp_path)
    started = []
    monkeypatch.setattr(
        background, "start_sync", lambda target, run_id: started.append(target) or run_id
    )
    run_id = client.get(f"/api/github-sync?account_id={account_id}").json()[0]["id"]

    resp = client.post(f"/api/github-sync/{run_id}/resume")

    assert resp.status_code == 200
    assert [(t.kind, t.github_username, t.account_id) for t in started] == [
        ("user", "ada-api", account_id)
    ]


def test_dismiss_then_the_list_is_empty(tmp_path):
    client, account_id = _setup(tmp_path)
    run_id = client.get(f"/api/github-sync?account_id={account_id}").json()[0]["id"]

    assert client.post(f"/api/github-sync/{run_id}/dismiss").status_code == 200
    assert client.get(f"/api/github-sync?account_id={account_id}").json() == []


def test_unknown_run_is_404(tmp_path):
    client, _ = _setup(tmp_path)

    assert client.post("/api/github-sync/99999/resume").status_code == 404
    assert client.post("/api/github-sync/99999/dismiss").status_code == 404
