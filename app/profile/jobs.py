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

Each repo is attempted at most once per run. Re-checking is for repos that
appeared since the last pass, not for repos this run already tried: a
"failed" repo stays eligible for the *next* run (that is what makes a
transient failure retryable) but is not offered to this one again, so a
repo that fails for a reason no retry fixes cannot turn the re-check loop
into an endless one. The run ends when a pass finds nothing new, after a
few consecutive empty checks so it isn't fooled by the gap between two
sync commits.
"""

from __future__ import annotations

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
    # Repos this run has already handed to build_profile_progress. A repo
    # left "failed" stays eligible on purpose (see _eligible_repos, and
    # Repository.skill_extraction_status), which is right across runs and
    # wrong inside one: without this set, a repo that fails for a reason no
    # retry fixes (a README the provider refuses, a model name that does
    # not exist, a prompt too long) is re-fetched by the next pass, fails
    # the same way, and the loop never ends, spending one provider request
    # per pass for the life of the process. Tried once here, retried by the
    # next run. A repo the concurrent sync commits while this run is going
    # is not in the set, so the re-check loop still does its real job.
    attempted: set[int] = set()
    job.set_state(stage="running", index=0, total=0, name="")
    while True:
        repos = [r for r in _eligible_repos(account_id) if r.id not in attempted]
        if not repos:
            empty_passes += 1
            if empty_passes >= _EMPTY_PASSES_BEFORE_STOP:
                break
            time.sleep(_EMPTY_PASS_DELAY_SECONDS)
            continue
        empty_passes = 0
        attempted.update(r.id for r in repos)

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
