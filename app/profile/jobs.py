"""Background skill-extraction worker, one per account, run as its own
daemon thread via app/core/jobs.py. Kept separate from GitHub
sync's own thread: fetching a repo list is cheap and fast, extracting skills
from each repo's README is a slow LLM call per repo, so the two run as two
independent pipelines rather than one blocking on the other.

Kicking off extraction is safe to call any time repos might have gone
"pending": right when a GitHub sync starts, again when it finishes, on
every /projects page load, whatever. start_extraction() is a no-op if one's
already running for that account, so callers never need to check first.

The worker doesn't take one fixed snapshot of "pending repos right now" and
quit; it re-checks after each pass. That's what lets it run in
parallel with an in-flight GitHub sync: sync commits repos one at a time,
marking each "pending" as it lands, and this worker picks them up as they
appear rather than only seeing whatever was already there when it started.
It gives up after a few consecutive empty passes (nothing pending, several
checks in a row) rather than polling forever.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Iterator
from typing import Any

from sqlalchemy import select

from app.core.db import Repository, get_db
from app.core.jobs import Job
from app.core.jobs import snapshot as job_snapshot
from app.core.jobs import start as start_job
from app.core.jobs import stream as job_stream
from app.profile.build import _SKIP_STATUSES, build_profile_progress

logger = logging.getLogger(__name__)

# How many consecutive "nothing pending" checks before the worker decides
# the queue is actually empty rather than just between two sync commits.
_EMPTY_PASSES_BEFORE_STOP = 3
_EMPTY_PASS_DELAY_SECONDS = 1.5


def _job_key(account_id: int) -> str:
    return f"extraction:{account_id}"


def _eligible_repos(account_id: int) -> list[Repository]:
    """Same eligibility as build_profile()'s own per-repo skip check:
    everything except already-`extracted`/`no_signal`, so this also
    reprocesses `failed`/`rate_limited` repos, matching the old
    process-pending endpoint's scope. Gating *whether* to auto-start on a
    plain page visit (vs. requiring an explicit "Continue processing"
    click for a rate-limited account) is the caller's job, not this
    query's (see projects.html's hasPending check).
    """
    db = get_db()
    try:
        return list(
            db.execute(
                select(Repository).where(
                    Repository.account_id == account_id,
                    Repository.skill_extraction_status.not_in(_SKIP_STATUSES),
                )
            ).scalars()
        )
    finally:
        db.close()


def _worker(account_id: int, job: Job) -> None:
    total_done = 0
    empty_passes = 0
    job.set_state(stage="running", index=0, total=0, name="")
    while True:
        repos = _eligible_repos(account_id)
        if not repos:
            empty_passes += 1
            if empty_passes >= _EMPTY_PASSES_BEFORE_STOP:
                break
            time.sleep(_EMPTY_PASS_DELAY_SECONDS)
            continue
        empty_passes = 0

        for event in build_profile_progress(repos):
            if event["stage"] == "repo_progress":
                job.set_state(
                    stage="repo_progress",
                    index=total_done + event["index"],
                    total=total_done + event["total"],
                    name=event["name"],
                    status=event["status"],
                )
            elif event["stage"] == "rate_limited":
                job.set_state(
                    stage="rate_limited",
                    index=total_done + event["index"],
                    total=total_done + event["total"],
                    detail=(
                        "LLM rate limit or budget cap reached. Stopped early; "
                        "already-extracted repos are unaffected. Retry later."
                    ),
                )
                return
            elif event["stage"] == "done":
                total_done += event["processed"]

    job.set_state(stage="done", index=total_done, total=total_done)


def start_extraction(account_id: int) -> bool:
    """Starts (or confirms already-running) the background extraction job
    for this account. Returns True if this call actually started a new
    thread, False if one was already in flight; either way, callers can
    immediately start tailing extraction_stream(account_id)."""
    return start_job(_job_key(account_id), lambda job: _worker(account_id, job))


def extraction_snapshot(account_id: int) -> dict | None:
    return job_snapshot(_job_key(account_id))


def extraction_stream(account_id: int) -> Iterator[dict[str, Any]]:
    return job_stream(_job_key(account_id))


_STOP_NOTE = " Processing stopped here"


def _earliest_key_retry() -> dt.datetime | None:
    """When the soonest out-of-quota API key is due back, if any."""
    from app.core import api_keys_store

    times = [
        dt.datetime.fromisoformat(key["retry_at"])
        for key in api_keys_store.list_keys()
        if key["enabled"] and key["status"] == "rate_limited" and key["retry_at"]
    ]
    return min(times) if times else None


def extraction_reminder(account_id: int) -> dict[str, Any]:
    """Where this account's skill extraction stands, for the reminder
    widget and the sync page. Rebuilt from each repo's saved
    skill_extraction_status, so it still knows where a run stopped after
    the app was closed part way: repos already done stay done, and the
    first one still waiting is where Continue picks up.

    `reason` is the saved error of the repo a limit stopped on (budget cap,
    rate limit, or out of keys), without the generic "stopped here" tail.
    `retry_at` is when the soonest exhausted API key is due back; the key
    refresh pass continues the run on its own then
    (resume_rate_limited_extractions below)."""
    db = get_db()
    try:
        repos = list(
            db.execute(
                select(Repository)
                .where(Repository.account_id == account_id)
                .order_by(Repository.id)
            ).scalars()
        )
    finally:
        db.close()

    by_status: dict[str, list[Repository]] = {}
    for repo in repos:
        by_status.setdefault(repo.skill_extraction_status, []).append(repo)
    limited = by_status.get("rate_limited", [])
    waiting = limited + by_status.get("pending", [])
    failed = by_status.get("failed", [])

    reason = None
    if limited:
        error = limited[0].skill_extraction_error or ""
        reason = error.split(_STOP_NOTE)[0].strip() or None
    elif failed and not waiting:
        reason = failed[0].skill_extraction_error

    next_up = (waiting or failed)[:1]
    snapshot = extraction_snapshot(account_id) or {}
    running = bool(snapshot.get("running"))
    return {
        "running": running,
        "index": snapshot.get("index") if running else None,
        "total": snapshot.get("total") if running else None,
        "current": snapshot.get("name") if running else None,
        "total_repos": len(repos),
        "done": len(by_status.get("extracted", [])) + len(by_status.get("no_signal", [])),
        "waiting": len(waiting),
        "failed": len(failed),
        "rate_limited": len(limited),
        "next_repo": next_up[0].full_name if next_up else None,
        "reason": reason,
        "retry_at": _earliest_key_retry() if limited else None,
    }


def resume_rate_limited_extractions() -> list[int]:
    """Continues every account's extraction that a limit stopped, once an
    API key is back in rotation (called from app/core/key_refresh.py).
    Only runs that were cut off by a limit: an app closed mid-run waits
    for the person to press Continue, since each call costs money."""
    db = get_db()
    try:
        account_ids: list[int] = sorted(
            {
                account_id
                for account_id in db.execute(
                    select(Repository.account_id).where(
                        Repository.skill_extraction_status == "rate_limited",
                        Repository.account_id.is_not(None),
                    )
                ).scalars()
                if account_id is not None
            }
        )
    finally:
        db.close()
    return [account_id for account_id in account_ids if start_extraction(account_id)]
