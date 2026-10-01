"""Runs a GitHub sync in a background thread instead of inside the SSE
response that asked for it.

Before this, the sync generator ran in the streaming response itself, so
closing the tab or opening another page ended the request and the sync
with it. Now the worker thread owns the sync (app/core/jobs.py) and an SSE
stream only follows its event log: leaving the page stops the following,
not the fetching, and coming back replays the log and keeps following.

One job per target (a GitHub account or a single repo), shared by /sync
and the fetch-data page, so both pages show the same run and a second
click joins it instead of starting a parallel one against the same quota.

Each target's latest outcome is kept in the github_sync_runs table. When
GitHub's hourly limit cuts a sync off part way, that row remembers how far
it got, the header widget reminds the person on every page, and a timer
resumes the sync on its own once the limit resets. Already saved repos
are cache hits on the resumed run, so it carries on from where it stopped.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from github import BadCredentialsException, RateLimitExceededException, UnknownObjectException
from sqlalchemy import select

from app.core import jobs
from app.core.db import GitHubSyncRun, SyncSource, get_db
from app.ingest.github.sync import sync_account_progress, sync_single_repo_progress

logger = logging.getLogger(__name__)

# Past the reset moment before resuming, so GitHub has rolled the window
# over by the time the first call goes out.
_RESUME_GRACE_SECONDS = 15.0

_timers: dict[str, threading.Timer] = {}
_timers_lock = threading.Lock()


def account_key(username: str) -> str:
    return f"github-sync:user:{username.lower()}"


def repo_key(full_name: str) -> str:
    return f"github-sync:repo:{full_name.lower()}"


def _error_event(e: Exception, not_found_detail: str) -> dict[str, Any]:
    if isinstance(e, UnknownObjectException):
        return {"stage": "error", "detail": not_found_detail}
    if isinstance(e, BadCredentialsException):
        return {"stage": "error", "detail": "GitHub rejected the configured token/credentials"}
    if isinstance(e, RateLimitExceededException):
        reset_at = getattr(e, "reset_at", None)
        if reset_at is not None:
            return {
                "stage": "error",
                "detail": "GitHub's hourly request limit is used up; sync again after it resets.",
                "reset_at": reset_at.isoformat(),
            }
        return {"stage": "error", "detail": "GitHub rate limit exhausted; try again shortly"}
    logger.exception("github sync crashed")
    return {"stage": "error", "detail": "Sync failed, see server logs."}


def start(
    key: str,
    run_id: str,
    make_events: Callable[[str], Iterator[dict[str, Any]]],
    not_found_detail: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Starts make_events(run_id) in the background unless a sync for key
    is already running. Returns the run_id of the sync that is actually
    running, which is the earlier one when this call joined it; the Stop
    button needs that id, not the one it proposed. on_event sees every
    event, including the error event a crash turns into."""

    def worker(job: jobs.Job) -> None:
        job.set_state(run_id=run_id)
        log: list[dict[str, Any]] = []

        def emit(event: dict[str, Any]) -> None:
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    logger.exception("recording github sync event failed")
            log.append({**event, "run_id": run_id})
            job.set_state(events=tuple(log))

        try:
            for event in make_events(run_id):
                emit(event)
        except Exception as e:
            emit(_error_event(e, not_found_detail))

    if jobs.start(key, worker):
        return run_id
    return (jobs.snapshot(key) or {}).get("run_id") or run_id


def is_running(key: str) -> bool:
    state = jobs.snapshot(key)
    return bool(state and state["running"])


def follow(key: str, poll_interval: float = 0.3) -> Iterator[dict[str, Any]]:
    """Every event of the current (or last) sync for key, from the first
    one, then new ones as they land, until the sync finishes. Yields
    nothing if key never ran."""
    sent = 0
    while True:
        state = jobs.snapshot(key)
        if state is None:
            return
        events = state.get("events") or ()
        yield from events[sent:]
        sent = len(events)
        if not state["running"]:
            # the thread may have logged its last event between the copy
            # above and the liveness check
            final = (jobs.snapshot(key) or {}).get("events") or ()
            yield from final[sent:]
            return
        time.sleep(poll_interval)


@dataclass(frozen=True)
class SyncTarget:
    """What to sync: a whole account (kind "user") or one repo (kind
    "repo", commits credited to attribution_username)."""

    kind: str
    github_username: str
    attribution_username: str
    account_id: int | None = None
    repo_full_name: str | None = None

    @property
    def key(self) -> str:
        if self.kind == "repo" and self.repo_full_name:
            return repo_key(self.repo_full_name)
        return account_key(self.github_username)

    @property
    def label(self) -> str:
        if self.kind == "repo" and self.repo_full_name:
            return self.repo_full_name
        return self.github_username

    def events(self, run_id: str) -> Iterator[dict[str, Any]]:
        if self.kind == "repo" and self.repo_full_name:
            return sync_single_repo_progress(
                self.repo_full_name,
                self.attribution_username,
                account_id=self.account_id,
                run_id=run_id,
            )
        return sync_account_progress(
            self.github_username, account_id=self.account_id, run_id=run_id
        )


def _target_from_row(row: GitHubSyncRun) -> SyncTarget:
    return SyncTarget(
        kind=row.kind,
        github_username=row.github_username,
        attribution_username=row.attribution_username,
        account_id=row.account_id,
        repo_full_name=row.repo_full_name,
    )


def _utc(value: dt.datetime | None) -> dt.datetime | None:
    # SQLite hands datetimes back naive; they were stored as UTC
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=dt.UTC)
    return value


def _parse_reset(event: dict[str, Any]) -> dt.datetime | None:
    raw = event.get("reset_at")
    return dt.datetime.fromisoformat(raw) if raw else None


def _update_row(key: str, **fields: Any) -> None:
    db = get_db()
    try:
        row = db.execute(select(GitHubSyncRun).where(GitHubSyncRun.key == key)).scalar_one_or_none()
        if row is not None:
            for name, value in fields.items():
                setattr(row, name, value)
            db.commit()
    finally:
        db.close()


def _mark_sources_synced(target: SyncTarget) -> None:
    """Every fetch-data entry pointing at this target counts as synced,
    whichever page started the sync."""
    db = get_db()
    try:
        query = select(SyncSource).where(
            SyncSource.kind == target.kind,
            SyncSource.github_username == target.github_username,
            SyncSource.repo_full_name == target.repo_full_name,
        )
        if target.account_id is not None:
            query = query.where(SyncSource.account_id == target.account_id)
        now = dt.datetime.now(dt.UTC)
        for source in db.execute(query).scalars():
            source.last_synced_at = now
        db.commit()
    finally:
        db.close()


def _cancel_timer(key: str) -> None:
    with _timers_lock:
        timer = _timers.pop(key, None)
    if timer is not None:
        timer.cancel()


def _schedule_resume(key: str, reset_at: dt.datetime) -> dt.datetime:
    resume_at = reset_at + dt.timedelta(seconds=_RESUME_GRACE_SECONDS)
    delay = max(1.0, (resume_at - dt.datetime.now(dt.UTC)).total_seconds())
    timer = threading.Timer(delay, _auto_resume, args=(key,))
    timer.daemon = True
    _cancel_timer(key)
    with _timers_lock:
        _timers[key] = timer
    timer.start()
    return resume_at


def _auto_resume(key: str) -> None:
    with _timers_lock:
        _timers.pop(key, None)
    db = get_db()
    try:
        row = db.execute(select(GitHubSyncRun).where(GitHubSyncRun.key == key)).scalar_one_or_none()
        if row is None or row.state != "rate_limited" or row.resume_at is None:
            return  # dismissed, stopped, or already resumed by hand
        target = _target_from_row(row)
    finally:
        db.close()
    logger.info("GitHub limit reset, resuming sync of %s", target.label)
    start_sync(target, str(uuid.uuid4()))
    if target.account_id is not None:
        # no page is around to kick skill extraction off, as the sync
        # pages do on a click; imported here to keep the LLM stack out of
        # this module's import
        from app.profile.jobs import start_extraction

        start_extraction(target.account_id)


def _recorder(target: SyncTarget, baseline: int) -> Callable[[dict[str, Any]], None]:
    """Keeps the target's github_sync_runs row in step with its events.
    `baseline` is how many repos earlier runs already saved, so a resumed
    run's count only moves up (10 -> 20), never back to 1 while it races
    through the repos it already has."""
    key = target.key

    def record(event: dict[str, Any]) -> None:
        stage = event.get("stage")
        if stage == "listing_repos":
            _update_row(key, total_hint=event.get("total_hint"))
        elif stage == "repo_progress":
            _update_row(
                key,
                completed=max(baseline, event.get("index") or 0),
                total_hint=event.get("total_hint"),
            )
        elif stage == "done":
            _update_row(
                key,
                state="done",
                completed=event.get("total_repos") or 0,
                total_hint=event.get("total_repos"),
                detail=None,
                reset_at=None,
                resume_at=None,
            )
            _mark_sources_synced(target)
        elif stage in ("rate_limited", "cancelled", "error"):
            reset_at = _parse_reset(event)
            if stage == "cancelled":
                state = "cancelled"
            elif stage == "rate_limited" or reset_at is not None:
                state = "rate_limited"
            else:
                state = "error"
            resume_at = _schedule_resume(key, reset_at) if reset_at is not None else None
            fields: dict[str, Any] = {
                "state": state,
                "detail": event.get("detail"),
                "reset_at": reset_at,
                "resume_at": resume_at,
            }
            if "completed" in event:
                fields["completed"] = max(baseline, event["completed"] or 0)
            if event.get("total_hint"):
                fields["total_hint"] = event["total_hint"]
            _update_row(key, **fields)

    return record


def start_sync(target: SyncTarget, run_id: str) -> str:
    """Starts (or joins) the background sync of target and records its
    progress for the reminder widget. Returns the run_id actually running,
    as start() does."""
    key = target.key
    if is_running(key):
        return (jobs.snapshot(key) or {}).get("run_id") or run_id
    _cancel_timer(key)

    db = get_db()
    try:
        row = db.execute(select(GitHubSyncRun).where(GitHubSyncRun.key == key)).scalar_one_or_none()
        if row is None:
            row = GitHubSyncRun(key=key)
            db.add(row)
            baseline = 0
        else:
            resuming = row.state in ("rate_limited", "cancelled", "error", "dismissed")
            baseline = row.completed if resuming else 0
        row.kind = target.kind
        row.github_username = target.github_username
        row.repo_full_name = target.repo_full_name
        row.attribution_username = target.attribution_username
        if target.account_id is not None:
            row.account_id = target.account_id
        row.state = "running"
        row.completed = baseline
        row.detail = None
        row.resume_at = None
        db.commit()
    finally:
        db.close()

    return start(
        key,
        run_id,
        target.events,
        not_found_detail=f"'{target.label}' not found on GitHub",
        on_event=_recorder(target, baseline),
    )


def all_sources_key(account_id: int) -> str:
    return f"github-sync:all:{account_id}"


def start_all(account_id: int, targets: list[SyncTarget]) -> bool:
    """Syncs every target, one after another, in a background thread of
    its own, so "Sync all" (and the first sync after signup) keeps going
    through the list when the page that asked is left, instead of the page
    driving the loop and stopping at whichever source it was on. One at a
    time, because they share one GitHub quota. A target cut off by the
    limit gets its automatic resume like any other, and the rest still
    run (each fails fast while the limit lasts, and is scheduled too).
    Returns False if a sync-all for this account is already going."""

    def worker(job: jobs.Job) -> None:
        job.set_state(total=len(targets), index=0)
        for index, target in enumerate(targets, start=1):
            job.set_state(index=index, current=target.label)
            start_sync(target, str(uuid.uuid4()))
            for _ in follow(target.key):
                pass  # wait for this one to finish before the next

    return jobs.start(all_sources_key(account_id), worker)


def resume(run_row_id: int) -> str | None:
    """Resume button: runs the row's target again now. None if no such row."""
    db = get_db()
    try:
        row = db.get(GitHubSyncRun, run_row_id)
        target = _target_from_row(row) if row is not None else None
    finally:
        db.close()
    if target is None:
        return None
    return start_sync(target, str(uuid.uuid4()))


def dismiss(run_row_id: int) -> bool:
    """Hides the reminder and calls off any automatic resume. The next
    sync of the same target brings the row back."""
    db = get_db()
    try:
        row = db.get(GitHubSyncRun, run_row_id)
        if row is None:
            return False
        key = row.key
        if not is_running(key):
            row.state = "dismissed"
        row.resume_at = None
        db.commit()
    finally:
        db.close()
    _cancel_timer(key)
    return True


def pending_runs(account_id: int) -> list[dict[str, Any]]:
    """Syncs this account should hear about: in progress, or stopped
    before finishing and not dismissed."""
    db = get_db()
    try:
        rows = list(
            db.execute(
                select(GitHubSyncRun)
                .where(
                    GitHubSyncRun.account_id == account_id,
                    GitHubSyncRun.state.not_in(("done", "dismissed")),
                )
                .order_by(GitHubSyncRun.updated_at.desc())
            ).scalars()
        )
        return [
            {
                "id": row.id,
                "kind": row.kind,
                "label": _target_from_row(row).label,
                "state": row.state,
                "running": is_running(row.key),
                "completed": row.completed,
                "total_hint": row.total_hint,
                "detail": row.detail,
                "reset_at": _utc(row.reset_at),
                "resume_at": _utc(row.resume_at),
            }
            for row in rows
        ]
    finally:
        db.close()


def restore_after_restart() -> None:
    """At startup: a row still "running" belongs to a sync the last
    process took down with it, so it becomes resumable; a scheduled
    automatic resume gets its timer back (firing right away if the reset
    already passed while the app was off)."""
    db = get_db()
    try:
        rows = list(
            db.execute(
                select(GitHubSyncRun).where(GitHubSyncRun.state.in_(("running", "rate_limited")))
            ).scalars()
        )
        to_schedule = []
        for row in rows:
            if row.state == "running":
                row.state = "error"
                row.detail = "The app stopped while this sync was running."
            elif row.resume_at is not None:
                to_schedule.append((row.key, _utc(row.resume_at)))
        db.commit()
    finally:
        db.close()
    for key, resume_at in to_schedule:
        _schedule_resume(key, resume_at - dt.timedelta(seconds=_RESUME_GRACE_SECONDS))
