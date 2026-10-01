import datetime as dt
import os
import threading

import pytest
from github import RateLimitExceededException, UnknownObjectException

import app.core.db as db_module
from app.core.db import Account, GitHubSyncRun, SyncSource, init_db
from app.core.settings import get_settings
from app.ingest.github import background
from app.ingest.github.client import RateLimitWindowExhausted


def _wait_until_finished(key: str) -> None:
    for _ in background.follow(key, poll_interval=0.01):
        pass


def test_sync_runs_without_anyone_following_it_and_late_followers_get_every_event():
    key = background.account_key("NoOneWatching")
    release = threading.Event()

    def make_events(run_id):
        yield {"stage": "checking_profile"}
        release.wait(5)
        yield {"stage": "done", "total_repos": 0, "fetched": 0, "cache_hits": 0}

    background.start(key, "first", make_events, not_found_detail="missing")

    assert background.is_running(key)
    release.set()
    events = list(background.follow(key, poll_interval=0.01))

    assert [e["stage"] for e in events] == ["checking_profile", "done"]
    assert {e["run_id"] for e in events} == {"first"}
    assert not background.is_running(key)


def test_second_start_joins_the_running_sync_and_returns_its_run_id():
    key = background.repo_key("octocat/Joined")
    release = threading.Event()
    starts = []

    def make_events(run_id):
        starts.append(run_id)
        release.wait(5)
        yield {"stage": "done", "total_repos": 1, "fetched": 1, "cache_hits": 0}

    assert background.start(key, "first", make_events, not_found_detail="x") == "first"
    for _ in range(100):  # the worker records its run_id as its first step
        if background.start(key, "second", make_events, not_found_detail="x") == "first":
            break
    else:
        raise AssertionError("second start did not join the first run")
    release.set()
    _wait_until_finished(key)

    assert starts == ["first"]


def _raising(exc):
    def make_events(run_id):
        raise exc
        yield  # pragma: no cover - makes this a generator

    return make_events


def test_errors_become_events_with_readable_details():
    import datetime as dt

    reset = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
    cases = {
        "missing-user": (UnknownObjectException(404, "Not Found", {}), "'missing-user' not found"),
        "limited": (RateLimitExceededException(403, "limit", {}), "try again shortly"),
        "hourly": (
            RateLimitWindowExhausted(RateLimitExceededException(403, "limit", {}), reset),
            "hourly request limit",
        ),
        "crashed": (ValueError("boom"), "see server logs"),
    }
    for name, (exc, expected) in cases.items():
        key = background.account_key(f"errors-{name}")
        background.start(key, name, _raising(exc), not_found_detail=f"'{name}' not found")
        events = list(background.follow(key, poll_interval=0.01))

        assert events[-1]["stage"] == "error"
        assert expected in events[-1]["detail"]
        if name == "hourly":
            assert events[-1]["reset_at"] == reset.isoformat()


def test_follow_on_a_key_that_never_ran_yields_nothing():
    assert list(background.follow(background.account_key("never-ran"))) == []


@pytest.fixture
def db(tmp_path, monkeypatch):
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


@pytest.fixture
def timers(monkeypatch):
    """Records scheduled automatic resumes instead of starting real timers."""
    scheduled = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            self.delay, self.fn, self.args = delay, fn, args
            self.cancelled = False
            self.daemon = False

        def start(self):
            scheduled.append(self)

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(background.threading, "Timer", FakeTimer)
    return scheduled


def _run(target, events):
    background.start_sync(target, "run")
    _wait_until_finished(target.key)


def _row(key):
    session = db_module.get_db()
    try:
        return session.query(GitHubSyncRun).filter_by(key=key).one()
    finally:
        session.close()


def _progress(n, total):
    return [
        {"stage": "repo_progress", "index": i, "total_hint": total, "name": f"ada/r{i}"}
        for i in range(1, n + 1)
    ]


def test_cut_off_sync_is_remembered_and_scheduled_to_resume(db, timers, monkeypatch):
    reset = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=30)
    events = [
        {"stage": "listing_repos", "total_hint": 23},
        *_progress(10, 23),
        {
            "stage": "rate_limited",
            "completed": 10,
            "total_hint": 23,
            "detail": "limit",
            "reset_at": reset.isoformat(),
        },
    ]
    monkeypatch.setattr(
        background, "sync_account_progress", lambda *a, **kw: iter(events)
    )
    target = background.SyncTarget("user", "ada-cut", "ada", account_id=db)

    _run(target, events)

    row = _row(target.key)
    assert (row.state, row.completed, row.total_hint) == ("rate_limited", 10, 23)
    assert len(timers) == 1 and timers[0].args == (target.key,)
    assert 29 * 60 < timers[0].delay < 31 * 60
    [pending] = background.pending_runs(db)
    assert pending["state"] == "rate_limited"
    assert pending["completed"] == 10
    assert pending["resume_at"] > reset


def test_resume_counts_up_from_where_the_last_run_stopped(db, timers, monkeypatch):
    target = background.SyncTarget("user", "ada-resume", "ada", account_id=db)
    reset = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=30)
    first = [*_progress(10, 23), {
        "stage": "rate_limited", "completed": 10, "total_hint": 23,
        "detail": "limit", "reset_at": reset.isoformat(),
    }]
    monkeypatch.setattr(background, "sync_account_progress", lambda *a, **kw: iter(first))
    _run(target, first)

    seen = []
    release = threading.Event()

    def second_run(*args, **kwargs):
        # races through the 10 cached repos first; the count must not drop
        yield from _progress(3, 23)
        seen.append(_row(target.key).completed)
        release.wait(5)
        yield from _progress(20, 23)[3:]
        yield {"stage": "rate_limited", "completed": 20, "total_hint": 23, "detail": "limit"}

    monkeypatch.setattr(background, "sync_account_progress", second_run)
    row_id = _row(target.key).id
    background.resume(row_id)
    assert timers[0].cancelled  # resuming by hand calls the automatic one off
    release.set()
    _wait_until_finished(target.key)

    assert seen == [10]
    row = _row(target.key)
    assert (row.state, row.completed) == ("rate_limited", 20)
    assert row.resume_at is None  # no reset time known, so nothing scheduled


def test_finished_sync_leaves_the_reminder_and_marks_sources_synced(db, monkeypatch):
    session = db_module.get_db()
    session.add(SyncSource(
        account_id=db, raw_input="ada-done", kind="user", github_username="ada-done"
    ))
    session.commit()
    session.close()
    events = [*_progress(2, 2), {"stage": "done", "total_repos": 2, "fetched": 2, "cache_hits": 0}]
    monkeypatch.setattr(background, "sync_account_progress", lambda *a, **kw: iter(events))
    target = background.SyncTarget("user", "ada-done", "ada", account_id=db)

    _run(target, events)

    assert _row(target.key).state == "done"
    assert background.pending_runs(db) == []
    session = db_module.get_db()
    assert session.query(SyncSource).one().last_synced_at is not None
    session.close()


def test_dismiss_hides_the_reminder_and_cancels_the_automatic_resume(db, timers, monkeypatch):
    reset = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5)
    events = [{"stage": "error", "detail": "limit", "reset_at": reset.isoformat()}]
    monkeypatch.setattr(background, "sync_account_progress", lambda *a, **kw: iter(events))
    target = background.SyncTarget("user", "ada-dismiss", "ada", account_id=db)
    _run(target, events)
    assert _row(target.key).state == "rate_limited"  # limit hit before any repo

    assert background.dismiss(_row(target.key).id)

    assert timers[0].cancelled
    assert background.pending_runs(db) == []


def test_auto_resume_skips_a_row_that_was_dismissed_meanwhile(db, timers, monkeypatch):
    starts = []
    monkeypatch.setattr(background, "start_sync", lambda target, run_id: starts.append(target))
    session = db_module.get_db()
    session.add(GitHubSyncRun(
        key="github-sync:user:gone", account_id=db, kind="user", github_username="gone",
        attribution_username="ada", state="dismissed",
    ))
    session.commit()
    session.close()

    background._auto_resume("github-sync:user:gone")

    assert starts == []


def test_restart_makes_interrupted_syncs_resumable_and_reschedules_timers(db, timers):
    resume_at = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=10)
    session = db_module.get_db()
    session.add_all([
        GitHubSyncRun(
            key="github-sync:user:was-running", account_id=db, kind="user",
            github_username="was-running", attribution_username="ada",
            state="running", completed=4,
        ),
        GitHubSyncRun(
            key="github-sync:user:was-waiting", account_id=db, kind="user",
            github_username="was-waiting", attribution_username="ada",
            state="rate_limited", completed=10, resume_at=resume_at,
        ),
    ])
    session.commit()
    session.close()

    background.restore_after_restart()

    interrupted = _row("github-sync:user:was-running")
    assert (interrupted.state, interrupted.completed) == ("error", 4)
    assert [t.args for t in timers] == [("github-sync:user:was-waiting",)]
    assert 9 * 60 < timers[0].delay < 11 * 60


def test_sync_all_walks_every_target_in_turn_without_a_page(db, monkeypatch):
    order = []
    release = threading.Event()

    def fake(username, account_id=None, run_id=None, **kwargs):
        order.append(("start", username))
        if username == "ada-first":
            release.wait(5)
        yield {"stage": "done", "total_repos": 0, "fetched": 0, "cache_hits": 0}
        order.append(("end", username))

    monkeypatch.setattr(background, "sync_account_progress", fake)
    targets = [
        background.SyncTarget("user", "ada-first", "ada", account_id=db),
        background.SyncTarget("user", "ada-second", "ada", account_id=db),
    ]

    assert background.start_all(db, targets) is True
    assert background.start_all(db, targets) is False  # already going
    for _ in range(200):
        if order:
            break
        threading.Event().wait(0.01)
    assert order == [("start", "ada-first")]  # the second waits its turn
    release.set()
    for _ in background.follow(background.all_sources_key(db), poll_interval=0.01):
        pass

    assert order == [
        ("start", "ada-first"), ("end", "ada-first"),
        ("start", "ada-second"), ("end", "ada-second"),
    ]
    assert not background.is_running(background.all_sources_key(db))
