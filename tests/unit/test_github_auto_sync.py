import datetime as dt
import os

import pytest

import app.core.db as db_module
from app.core.db import Account, GitHubSyncRun, SyncSource, init_db
from app.core.settings import get_settings
from app.ingest.github import auto_sync, background

NOW = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)


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
    return account_id


def _source(account_id, username, last_synced_at, repo=None):
    session = db_module.get_db()
    session.add(
        SyncSource(
            account_id=account_id,
            raw_input=repo or username,
            kind="repo" if repo else "user",
            github_username=username,
            repo_full_name=repo,
            last_synced_at=last_synced_at,
        )
    )
    session.commit()
    session.close()


def _labels(due):
    return {account: [t.label for t in targets] for account, targets in due.items()}


def test_only_sources_synced_before_and_now_stale_are_due(account_id):
    _source(account_id, "stale", NOW - dt.timedelta(days=8))
    _source(account_id, "fresh", NOW - dt.timedelta(days=1))
    _source(account_id, "never", None)
    _source(account_id, "grace", NOW - dt.timedelta(days=30), repo="grace/compiler")

    due = auto_sync.due_targets(NOW)

    assert _labels(due) == {account_id: ["stale", "grace/compiler"]}
    repo_target = due[account_id][1]
    # commits on someone else's repo are credited to the profile's login
    assert repo_target.attribution_username == "ada"


def test_rate_limited_or_recently_attempted_syncs_are_left_alone(account_id):
    for name in ("limited", "failed_lately", "failed_long_ago", "plain"):
        _source(account_id, name, NOW - dt.timedelta(days=10))
    session = db_module.get_db()
    for name, state, days_ago in (
        ("limited", "rate_limited", 20),
        ("failed_lately", "error", 2),
        ("failed_long_ago", "error", 9),
    ):
        session.add(
            GitHubSyncRun(
                key=background.account_key(name),
                kind="user",
                github_username=name,
                attribution_username=name,
                account_id=account_id,
                state=state,
                updated_at=NOW - dt.timedelta(days=days_ago),
            )
        )
    session.commit()
    session.close()

    due = auto_sync.due_targets(NOW)

    assert _labels(due) == {account_id: ["failed_long_ago", "plain"]}


def test_check_syncs_due_sources_then_starts_extraction(account_id, monkeypatch):
    _source(account_id, "stale", NOW - dt.timedelta(days=8))
    calls = []
    monkeypatch.setattr(
        background,
        "start_all",
        lambda acct, targets: calls.append(("sync", acct, [t.label for t in targets])),
    )
    monkeypatch.setattr(background, "follow", lambda key: iter(()))
    monkeypatch.setattr(
        "app.profile.jobs.start_extraction", lambda acct: calls.append(("extract", acct))
    )

    synced = auto_sync.sync_due_sources(NOW)

    assert synced == [account_id]
    assert calls == [("sync", account_id, ["stale"]), ("extract", account_id)]


def test_check_with_nothing_due_does_nothing(account_id, monkeypatch):
    _source(account_id, "fresh", NOW - dt.timedelta(days=1))
    monkeypatch.setattr(
        background, "start_all", lambda *a: pytest.fail("nothing is due, nothing to sync")
    )

    assert auto_sync.sync_due_sources(NOW) == []
