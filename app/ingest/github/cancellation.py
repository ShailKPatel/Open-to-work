"""Stops a running GitHub sync between repos.

An in-process set of run ids: the app is a single process, so no other
worker needs to see the request. A run id is a SyncSource id, or a token
the client generates per sync.

The sync loop checks is_cancelled() before each repo and clears its own
flag. A fetch already in flight still completes; only the next repo is
not started.
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
