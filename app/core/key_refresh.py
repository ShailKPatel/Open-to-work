"""Comes back for the keys that were out of quota.

An exhausted key is the one failure in this app that repairs itself: the
quota window rolls over and the key works again. Without something to
notice that, the /apis page keeps showing a dead key, and dispatch keeps
sorting a perfectly good key last (app/core/api_keys_store.py's
_dispatch_order) until somebody clicks a button. This module is that
something: one daemon thread that runs the "due" recheck pass at startup
and then every REFRESH_INTERVAL, which covers both cases the user cares
about, a machine that gets restarted daily and one left running for a
week.

Deliberately a thread and an interval rather than a cron entry or a task
queue: this is a single-process local app, the pass is a handful of cheap
HTTP GETs, and nothing outside the process needs to know it happened. The
same pass is exposed as a button on /apis (POST /api/api-keys/recheck) for
anyone who does not want to wait for the next tick.

Blocked and rejected keys are never in this pass. That is the whole point
of keeping them in their own status: a revoked key cannot come back on its
own, so asking the provider about it every half hour is noise, and the
answer would not change until someone fixes it in the provider's console.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# How often to come back for keys whose cooldown has elapsed. A key's own
# retry_at is what decides whether it is checked, so this is only the
# resolution of that clock, not a per-key interval.
REFRESH_INTERVAL_SECONDS = 30 * 60

# Give the app a moment to finish starting before the first pass: it is
# not urgent, and a fresh start has nothing to dispatch yet.
FIRST_PASS_DELAY_SECONDS = 5


def refresh_due_keys() -> list[dict]:
    """One pass. Returns the keys it looked at, already re-serialized, so
    a caller can report what changed."""
    from app.core import api_keys_store

    checked = api_keys_store.recheck_keys("due")
    if not checked:
        return []
    recovered = [k for k in checked if k["status"] == "valid"]
    logger.info(
        "rechecked %d key(s) whose cooldown had elapsed; %d back in rotation",
        len(checked),
        len(recovered),
    )
    return checked


def start_key_refresh(stop: threading.Event | None = None) -> threading.Thread:
    """Starts the background pass and returns its thread. Daemon, so it
    never holds up shutdown, and every error is swallowed: a key refresh
    failing must not take the app down with it."""
    stop = stop or threading.Event()

    def loop() -> None:
        first = True
        while not stop.wait(FIRST_PASS_DELAY_SECONDS if first else REFRESH_INTERVAL_SECONDS):
            first = False
            try:
                refresh_due_keys()
            except Exception:
                logger.exception("key refresh pass failed; trying again next interval")

    thread = threading.Thread(target=loop, name="api-key-refresh", daemon=True)
    thread.start()
    return thread
