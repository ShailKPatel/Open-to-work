"""Rechecks keys that ran out of quota.

A quota window rolls over by itself, so an exhausted key usually works
again later. One daemon thread runs the "due" recheck pass at startup and
then every REFRESH_INTERVAL_SECONDS, so both a machine restarted daily and one
left running for a week catch it. /apis offers the same pass as a button
(POST /api/api-keys/recheck).

Rejected and blocked keys are never in this pass: they cannot recover on
their own, so they wait until someone rechecks them.
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
    if recovered:
        # skill extraction a spent key cut off can carry on now
        from app.profile.jobs import resume_rate_limited_extractions

        resumed = resume_rate_limited_extractions()
        if resumed:
            logger.info("continued skill extraction for account(s) %s", resumed)
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
