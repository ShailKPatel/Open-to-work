"""Resume builds that run in a worker thread instead of inside the request
that asked for one.

A build takes a minute or more (one quality-tier call, then Tectonic and
the page-fit loop). Run inside the request, closing the tab or opening
another page left the result with nobody to hand it to. Here the request
only starts the build (app/core/jobs.py owns the thread) and returns its
id; the build page polls that id, and the header reminder
(app/web/templates/_header.html) lists every build for the account, so
the end of one is announced on whatever page is open.

Not persisted, like app/core/jobs.py itself: a restart loses the record
of a build in flight. The resume a finished build saved to the library
is a normal row and survives; the page tells a watched build that went
missing apart from one that finished.
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
    # "running", "done" or "error".
    state: str = "running"
    detail: str | None = None
    finished_at: dt.datetime | None = None
    result: dict[str, Any] = field(default_factory=dict)
    pdf_bytes: bytes | None = None
    dismissed: bool = False

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

    def worker(job: jobs.Job) -> None:
        try:
            run(build)
        except Exception:
            logger.exception("resume build %s crashed", build.id)
            build.fail("Internal error, see server logs.")
            return
        if build.state == "running":
            build.fail("The build stopped without a result.")

    jobs.start(f"resume_build:{build.id}", worker)
    return build


def _prune(account_id: int) -> None:
    """Caller holds _lock."""
    finished = sorted(
        (b for b in _builds.values() if b.account_id == account_id and b.state != "running"),
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
    """The build currently retrying this incomplete resume, if any."""
    with _lock:
        for build in _builds.values():
            if build.retry_of == resume_id and build.state == "running":
                return build
    return None


def dismiss(build_id: str) -> bool:
    """Hides a finished build from the list. A running one stays listed:
    hiding its card is the page's business, the build carries on."""
    with _lock:
        build = _builds.get(build_id)
        if build is None or build.state == "running":
            return False
        build.dismissed = True
        return True
