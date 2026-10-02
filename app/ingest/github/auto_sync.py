"""Keeps synced GitHub sources fresh without anyone clicking Sync.

Once a source has synced, this module comes back for it every
SYNC_INTERVAL: the account's sources go through background.start_all one
after another, as "Sync all" does, and skill extraction is kicked off
afterwards for whatever came back pending. A resync is cheap by design
(see sync.py): one listing call per hundred repos finds new and pushed
repos, an untouched repo costs nothing more, and a pushed repo only goes
back to the LLM when its README changed.

Same shape as app/core/key_refresh.py: one daemon thread, a check at
startup and then every CHECK_INTERVAL. The clock that decides what is due
is each source's last_synced_at in the database, so a machine restarted
daily and one left running for weeks both resync on schedule.

Only sources that have finished a sync before are picked up. One that
never has is either new (its first sync is the page's job) or failing,
and one cut off by GitHub's hourly limit already has its own resume timer
(background.py), so it is left to that.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading

from sqlalchemy import or_, select

from app.core.db import Account, GitHubSyncRun, SyncSource, get_db
from app.ingest.github import background

logger = logging.getLogger(__name__)

# How stale a source gets before it is synced again.
SYNC_INTERVAL = dt.timedelta(days=7)

# How often to look for due sources. Only the resolution of the clock
# above; a check that finds nothing due makes no GitHub call.
CHECK_INTERVAL_SECONDS = 60 * 60

# Let the app finish starting first; nothing here is urgent.
FIRST_CHECK_DELAY_SECONDS = 60


def _utc(value: dt.datetime) -> dt.datetime:
    # SQLite hands datetimes back naive; they were stored as UTC
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)


def due_targets(now: dt.datetime | None = None) -> dict[int, list[background.SyncTarget]]:
    """Sources last synced more than SYNC_INTERVAL ago, grouped by
    account, skipping any whose sync is running or waiting on GitHub's
    limit to reset."""
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    try:
        # A run that failed or was stopped is retried after the same
        # interval rather than every check, so a source that keeps failing
        # (a renamed account, a deleted repo) costs one call a week.
        waiting = set(
            db.execute(
                select(GitHubSyncRun.key).where(
                    or_(
                        GitHubSyncRun.state.in_(("running", "rate_limited")),
                        GitHubSyncRun.updated_at > (now - SYNC_INTERVAL).replace(tzinfo=None),
                    )
                )
            ).scalars()
        )
        rows = db.execute(
            select(SyncSource)
            .where(SyncSource.last_synced_at.is_not(None))
            .order_by(SyncSource.created_at)
        ).scalars()
        due: dict[int, list[background.SyncTarget]] = {}
        seen: set[str] = set()
        for row in rows:
            assert row.last_synced_at is not None
            if now - _utc(row.last_synced_at) < SYNC_INTERVAL:
                continue
            account = db.get(Account, row.account_id)
            target = background.SyncTarget(
                kind=row.kind,
                github_username=row.github_username,
                # commits are credited to the profile's own login
                attribution_username=account.github_username if account else row.github_username,
                account_id=row.account_id,
                repo_full_name=row.repo_full_name if row.kind == "repo" else None,
            )
            if target.key in waiting or target.key in seen or background.is_running(target.key):
                continue
            seen.add(target.key)
            due.setdefault(row.account_id, []).append(target)
        return due
    finally:
        db.close()


def sync_due_sources(now: dt.datetime | None = None) -> list[int]:
    """One check: syncs every due source, then starts skill extraction for
    each account that had any. Blocks until those syncs finish, so the
    next check never overlaps this one. Returns the account ids synced."""
    due = due_targets(now)
    for account_id, targets in due.items():
        logger.info(
            "auto-syncing %d GitHub source(s) of account %d not synced in %d days",
            len(targets),
            account_id,
            SYNC_INTERVAL.days,
        )
        background.start_all(account_id, targets)
        for _ in background.follow(background.all_sources_key(account_id)):
            pass  # wait for the whole list before extracting
        # imported here to keep the LLM stack out of this module's import
        from app.profile.jobs import start_extraction

        start_extraction(account_id)
    return list(due)


def start_auto_sync(stop: threading.Event | None = None) -> threading.Thread:
    """Starts the background check and returns its thread. Daemon, so it
    never holds up shutdown; a failed check is logged and tried again on
    the next interval."""
    stop = stop or threading.Event()

    def loop() -> None:
        first = True
        while not stop.wait(FIRST_CHECK_DELAY_SECONDS if first else CHECK_INTERVAL_SECONDS):
            first = False
            try:
                sync_due_sources()
            except Exception:
                logger.exception("GitHub auto-sync check failed; trying again next interval")

    thread = threading.Thread(target=loop, name="github-auto-sync", daemon=True)
    thread.start()
    return thread
