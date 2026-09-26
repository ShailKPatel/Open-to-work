"""In-memory background-job registry.

Decouples long-running work (skill extraction, in particular) from the HTTP
request that kicked it off. A worker runs in its own daemon thread; SSE/poll
endpoints just read a snapshot of its progress. Closing the browser tab or
navigating to another page does not stop the worker; only the thread
finishing (or the process restarting) does. That is what lets someone start
processing and then go look at the dashboard: the dashboard and the worker
are not the same request.

Not persisted: job state lives only for this process's lifetime. A restart
loses in-flight progress display (the underlying DB writes already
committed by the worker are not lost, just the "X of Y" progress line),
which is fine for a single-process local app.

Keyed by an arbitrary string (callers namespace their own keys, e.g.
f"extraction:{account_id}") so this stays generic across job types rather
than baking in "one job per account" here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()
_jobs: dict[str, Job] = {}


@dataclass
class Job:
    key: str
    thread: threading.Thread | None = None
    _state: dict = field(default_factory=lambda: {"stage": "starting"})
    _state_lock: threading.Lock = field(default_factory=threading.Lock)
    # True from the moment start() registers this job until its thread is
    # either running or known to have failed to start. `thread.is_alive()`
    # alone cannot cover that gap: the job is in the registry before the
    # thread object exists, so a second start() for the same key would read
    # "not running" and launch a duplicate worker. Counted as running, so
    # that window answers the same way the rest of the job's life does.
    _starting: bool = True

    def set_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            self._state = {**self._state, **kwargs}

    def _thread_handed_off(self) -> None:
        """Called by start() once the thread is running, or once starting it
        has definitively failed. Either way this job no longer occupies the
        pre-thread window, and `thread.is_alive()` is the whole answer from
        here on."""
        with self._state_lock:
            self._starting = False

    def snapshot(self) -> dict:
        with self._state_lock:
            state = dict(self._state)
            starting = self._starting
        state["running"] = starting or (self.thread is not None and self.thread.is_alive())
        return state


def start(key: str, worker: Callable[[Job], None]) -> bool:
    """Starts `worker(job)` in a new daemon thread under `key`, unless a job
    with that key is already running, in which case this is a no-op. Returns
    True if a new thread was started, False if one was already in flight.
    The expected pattern is to call start() to make sure a job is running,
    then stream or poll its state separately; False is not an error.

    Two concurrent calls for one key start one worker, not two. The job is
    marked running (Job._starting) inside the registry lock, before the
    thread it will run on exists, so the second caller sees a live job
    rather than a half-registered one. The thread is then built and started
    outside the lock, which keeps a caller-supplied worker from ever
    running while this module holds the registry lock.
    """
    with _registry_lock:
        existing = _jobs.get(key)
        if existing is not None and existing.snapshot()["running"]:
            return False
        job = Job(key=key)
        _jobs[key] = job

    def run() -> None:
        try:
            worker(job)
        except Exception:
            logger.exception("background job %r crashed", key)
            job.set_state(stage="error", detail="Internal error, see server logs.")

    # try/finally, not a plain call afterwards: if building or starting the
    # thread raises, the job must stop counting as running, or this key is
    # permanently occupied by a job with nothing behind it and no later
    # start() can replace it until the process restarts.
    try:
        job.thread = threading.Thread(target=run, daemon=True, name=f"job:{key}")
        job.thread.start()
    finally:
        job._thread_handed_off()
    return True


def snapshot(key: str) -> dict | None:
    """Current state, or None if this key has never been started."""
    with _registry_lock:
        job = _jobs.get(key)
    return job.snapshot() if job is not None else None


def stream(key: str, poll_interval: float = 0.3) -> Iterator[dict]:
    """Yields state dicts as they change, until the job's thread finishes
    (one final yield with running=False) or the key was never started (no
    yields at all). Polling, not push: the simplest thing that works for a
    single-process app where state changes at most a few times a second
    (once per repo processed).
    """
    last: dict | None = None
    while True:
        state = snapshot(key)
        if state is None:
            return
        if state != last:
            yield state
            last = state
        if not state["running"]:
            return
        time.sleep(poll_interval)
