"""Resume builds that run in a worker thread instead of inside the request
that asked for one.

A build takes a minute or more (one quality-tier call, then Tectonic and
the page-fit loop). Run inside the request, closing the tab or opening
another page left the result with nobody to hand it to. Here the request
only starts the build (app/core/jobs.py owns the thread) and returns its
id; the build page polls that id, and the header reminder
(app/web/templates/_header.html) lists every build for the account, so
the end of one is announced on whatever page is open.

This record is in memory, like app/core/jobs.py itself: a restart loses
it. The work is not lost: graph.py checkpoints every step, and a stopped or
waiting build is an incomplete library row a retry carries on from.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core import jobs

logger = logging.getLogger(__name__)

# Finished builds kept per account, newest first, so the list (and the
# PDF bytes each one holds) stays bounded. Running ones are never pruned.
_KEEP_FINISHED = 10
_FINISHED = ("done", "error")

_lock = threading.Lock()
_builds: dict[str, Build] = {}


@dataclass
class Build:
    id: str
    account_id: int
    job_posting_id: int
    label: str
    # The incomplete library resume this build is retrying, if any.
    retry_of: int | None = None
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))
    stage: str = "Starting"
    # "running", "awaiting_approval", "done" or "error".
    state: str = "running"
    detail: str | None = None
    finished_at: dt.datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    pdf_bytes: bytes | None = None
    dismissed: bool = False
    # What carries the build on once a decision arrives, while it waits.
    continuation: Callable[[dict[str, Any]], None] | None = None

    def set_stage(self, stage: str) -> None:
        with _lock:
            self.stage = stage

    def finish(self, pdf_bytes: bytes, result: dict[str, Any]) -> None:
        with _lock:
            self.pdf_bytes = pdf_bytes
            self.result = result
            self.state = "done"
            self.stage = "Done"
            self.finished_at = dt.datetime.now(dt.UTC)

    def await_approval(
        self,
        result: dict[str, Any],
        continuation: Callable[[dict[str, Any]], None],
        resume_id: int | None,
    ) -> None:
        """Parks the build on its draft. resume_id is the incomplete library
        row holding its place, which a retry then finds this build by."""
        with _lock:
            self.result = result
            self.continuation = continuation
            self.retry_of = resume_id if resume_id is not None else self.retry_of
            self.state = "awaiting_approval"
            self.stage = "Waiting for your approval"

    def take_continuation(self) -> Callable[[dict[str, Any]], None] | None:
        """Moves a waiting build back to running and hands over what carries
        it on. None when it is not waiting, so two clicks start one run."""
        with _lock:
            continuation = self.continuation
            if self.state != "awaiting_approval" or continuation is None:
                return None
            self.continuation = None
            self.result = {}
            self.state = "running"
            self.stage = "Starting"
            return continuation

    def fail(self, detail: str, result: dict[str, Any] | None = None) -> None:
        with _lock:
            self.state = "error"
            self.detail = detail
            self.result = result or {}
            self.finished_at = dt.datetime.now(dt.UTC)

    def public(self) -> dict[str, Any]:
        with _lock:
            return {
                "id": self.id,
                "account_id": self.account_id,
                "job_posting_id": self.job_posting_id,
                "label": self.label,
                "retry_of": self.retry_of,
                "stage": self.stage,
                "state": self.state,
                "running": self.state == "running",
                "detail": self.detail,
                "started_at": self.started_at.isoformat(),
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "has_pdf": self.pdf_bytes is not None,
                **self.result,
            }


def start(
    account_id: int,
    job_posting_id: int,
    label: str,
    run: Callable[[Build], None],
    retry_of: int | None = None,
) -> Build:
    """Registers a build and starts `run(build)` in its own thread. `run`
    reports through build.set_stage(), then build.finish() or
    build.fail(); one that raises or returns without either is marked
    failed here, so a build can never sit at "running" forever."""
    build = Build(
        id=uuid.uuid4().hex,
        account_id=account_id,
        job_posting_id=job_posting_id,
        label=label,
        retry_of=retry_of,
    )
    with _lock:
        _builds[build.id] = build
        _prune(account_id)
    run_again(build, run)
    return build


def run_again(build: Build, run: Callable[[Build], None]) -> None:
    """Runs `run(build)` in a new thread, with start()'s guarantee that it
    cannot be left at "running" forever. The key is new each time because
    the thread that parked this build may not have exited yet."""

    def worker(job: jobs.Job) -> None:
        try:
            run(build)
        except Exception:
            logger.exception("resume build %s crashed", build.id)
            build.fail("Internal error, see server logs.")
            return
        if build.state == "running":
            build.fail("The build stopped without a result.")

    jobs.start(f"resume_build:{build.id}:{uuid.uuid4().hex}", worker)


def _prune(account_id: int) -> None:
    """Caller holds _lock."""
    finished = sorted(
        (b for b in _builds.values() if b.account_id == account_id and b.state in _FINISHED),
        key=lambda b: b.started_at,
        reverse=True,
    )
    for old in finished[_KEEP_FINISHED:]:
        del _builds[old.id]


def get(build_id: str) -> Build | None:
    with _lock:
        return _builds.get(build_id)


def list_for_account(account_id: int, job_posting_id: int | None = None) -> list[Build]:
    """Newest first, dismissed ones left out."""
    with _lock:
        rows = [
            b for b in _builds.values()
            if b.account_id == account_id
            and not b.dismissed
            and (job_posting_id is None or b.job_posting_id == job_posting_id)
        ]
    return sorted(rows, key=lambda b: b.started_at, reverse=True)


def running_retry_of(resume_id: int) -> Build | None:
    """The build currently retrying this incomplete resume, or waiting for
    approval on it, if any."""
    with _lock:
        for build in _builds.values():
            if build.retry_of == resume_id and build.state not in _FINISHED:
                return build
    return None


def running_count() -> int:
    with _lock:
        return sum(1 for build in _builds.values() if build.state == "running")


def forget_account(account_id: int) -> None:
    """Drops an erased account's builds, finished PDFs included."""
    with _lock:
        for build_id in [b.id for b in _builds.values() if b.account_id == account_id]:
            del _builds[build_id]


def dismiss(build_id: str) -> bool:
    """Hides a finished build from the list. A running or waiting one stays
    listed: hiding its card is the page's business, the build carries on."""
    with _lock:
        build = _builds.get(build_id)
        if build is None or build.state not in _FINISHED:
            return False
        build.dismissed = True
        return True
