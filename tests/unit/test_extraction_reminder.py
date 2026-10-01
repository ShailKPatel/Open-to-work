import datetime as dt
import os

import pytest
from fastapi.testclient import TestClient

import app.core.db as db_module
from app.core.db import Account, Repository, init_db
from app.core.settings import get_settings
from app.profile import jobs


@pytest.fixture
def account_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()
    session = db_module.get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="ada")
    session.add(account)
    session.commit()
    account_id = account.id
    session.close()
    # no background extraction thread in these tests
    monkeypatch.setattr(jobs, "extraction_snapshot", lambda account_id: None)
    monkeypatch.setattr(jobs, "_earliest_key_retry", lambda: None)
    return account_id


def _repos(account_id, statuses, error=None, owner="ada"):
    session = db_module.get_db()
    for i, status in enumerate(statuses, start=1):
        session.add(Repository(
            github_id=account_id * 1000 + i, name=f"r{i}", full_name=f"{owner}/r{i}",
            url=f"https://github.com/{owner}/r{i}", account_id=account_id,
            skill_extraction_status=status,
            skill_extraction_error=error if status in ("rate_limited", "failed") else None,
        ))
    session.commit()
    session.close()


def test_run_cut_off_by_a_limit_says_where_it_stopped_and_why(account_id, monkeypatch):
    retry = dt.datetime(2026, 9, 30, 6, 0, tzinfo=dt.UTC)
    monkeypatch.setattr(jobs, "_earliest_key_retry", lambda: retry)
    _repos(
        account_id,
        ["extracted", "extracted", "no_signal", "extracted", "rate_limited", "pending", "pending"],
        error=(
            "Monthly budget cap reached. Processing stopped here so the remaining "
            "projects don't fail the same way; run it again later to continue."
        ),
    )

    status = jobs.extraction_reminder(account_id)

    assert status["running"] is False
    assert (status["done"], status["total_repos"]) == (4, 7)
    assert (status["waiting"], status["rate_limited"], status["failed"]) == (3, 1, 0)
    assert status["next_repo"] == "ada/r5"  # the fifth one, where it stopped
    assert status["reason"] == "Monthly budget cap reached."
    assert status["retry_at"] == retry


def test_run_the_app_died_in_resumes_from_the_first_pending(account_id):
    _repos(account_id, ["extracted", "extracted", "pending", "pending"])

    status = jobs.extraction_reminder(account_id)

    assert (status["done"], status["waiting"], status["rate_limited"]) == (2, 2, 0)
    assert status["next_repo"] == "ada/r3"
    assert status["reason"] is None
    assert status["retry_at"] is None


def test_only_failures_left_reports_the_failure(account_id):
    _repos(account_id, ["extracted", "failed"], error="README could not be parsed")

    status = jobs.extraction_reminder(account_id)

    assert (status["waiting"], status["failed"]) == (0, 1)
    assert status["reason"] == "README could not be parsed"


def test_live_progress_comes_from_the_running_job(account_id, monkeypatch):
    _repos(account_id, ["extracted", "pending"])
    monkeypatch.setattr(
        jobs,
        "extraction_snapshot",
        lambda a: {"running": True, "index": 2, "total": 2, "name": "ada/r2"},
    )

    status = jobs.extraction_reminder(account_id)

    assert (status["running"], status["index"], status["current"]) == (True, 2, "ada/r2")


def test_key_back_in_rotation_continues_only_limit_stopped_accounts(account_id, monkeypatch):
    session = db_module.get_db()
    other = Account(first_name="Bo", last_name="B", github_username="bo")
    session.add(other)
    session.commit()
    other_id = other.id
    session.close()
    _repos(account_id, ["extracted", "rate_limited"], error="Rate limited.")
    _repos(other_id, ["pending"], owner="bo")  # closed mid-run, not limit-stopped
    started = []
    monkeypatch.setattr(jobs, "start_extraction", lambda a: started.append(a) or True)

    assert jobs.resume_rate_limited_extractions() == [account_id]
    assert started == [account_id]


def test_refresh_pass_continues_extraction_when_a_key_recovers(monkeypatch):
    from app.core import api_keys_store, key_refresh

    calls = []
    monkeypatch.setattr(api_keys_store, "recheck_keys", lambda scope: [{"status": "valid"}])
    monkeypatch.setattr(jobs, "resume_rate_limited_extractions", lambda: calls.append(1) or [])

    key_refresh.refresh_due_keys()
    assert calls == [1]

    monkeypatch.setattr(api_keys_store, "recheck_keys", lambda scope: [{"status": "rate_limited"}])
    key_refresh.refresh_due_keys()
    assert calls == [1]  # nothing recovered, nothing continued


def test_status_endpoint(account_id):
    from app.api.main import app

    _repos(account_id, ["extracted", "pending"])

    resp = TestClient(app).get(f"/api/projects/process-pending/status?account_id={account_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert (body["done"], body["waiting"], body["next_repo"]) == (1, 1, "ada/r2")
