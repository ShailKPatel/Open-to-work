"""Mid-sync cancellation: "stop the sync there" from the UI.

A plain in-process set, not a DB table or a queue: this app is
self-hosted, single instance, single account per device (see
app/core/db.py's Account docstring); there's no second process or worker
that needs to see a cancellation request, so nothing heavier is needed.

Keyed by an arbitrary string "run id": a SyncSource's own id (stringified)
for the sync-sources page (one sync per source can be in flight at a
time, so the source's id is already a unique-enough key), or a
client-generated token for the ad hoc /sync/github/stream path, which has
no persistent id of its own.

The generator being cancelled is responsible for checking is_cancelled()
between units of work (per-repo here) and clearing its own flag; nothing
here does that automatically. This does NOT abort a request already in
flight to GitHub (the current repo's fetch still completes); it just
stops the loop from starting the next one, same granularity as the
rate-limit partial-progress path in sync.py.
"""

from __future__ import annotations

import threading

_cancelled: set[str] = set()
_lock = threading.Lock()


def request_cancel(run_id: str) -> None:
    with _lock:
        _cancelled.add(run_id)


def is_cancelled(run_id: str) -> bool:
    with _lock:
        return run_id in _cancelled


def clear(run_id: str) -> None:
    with _lock:
        _cancelled.discard(run_id)
