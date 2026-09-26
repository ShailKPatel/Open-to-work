"""When an exhausted key is worth looking at again.

A key that hit a quota is not a broken key: it is a working key inside a
window it has already filled, and that window rolls over on its own. The
only question is when. This module answers it and nothing else: given the
provider and whatever the provider said when it refused the call, it
returns the kind of limit that was hit and the earliest moment a recheck
is worth making (app/core/api_keys_store.py stores both on the key, and
app/core/key_refresh.py is what comes back at that moment).

LiteLLM normalizes a 429 into one exception type and does not carry the
provider's own reset information, so the per-provider reading of that
detail happens here. Gemini is the one provider read properly today: its
429 body names the quota that was hit (`quotaId`, whose name says whether
it is a per-minute or a per-day limit) and often a `retryDelay`, so a
per-minute burst comes back in a minute while a used-up daily allowance
waits for the day to roll over. Every other provider gets
DEFAULT_COOLDOWN, which is not a guess about its limits, only a
reasonable interval after which asking again is cheap. Adding a provider
means adding a branch to plan_cooldown() and nothing else.

A cooldown elapsing is not proof the quota reset. No provider exposes
"how much quota do I have left" on a free endpoint, and the cheap check in
app/core/llm_providers.py lists models rather than generating, so it
answers 200 for a key whose generation quota is still spent. The cooldown
is therefore the moment to stop treating the key as dead and let a real
call decide, which is exactly what record_dispatch_outcome() records.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# How long to wait before rechecking a key whose provider said nothing
# useful about when its limit resets.
DEFAULT_COOLDOWN = dt.timedelta(hours=1)

# A provider asking for a shorter wait than this is rounded up to it: a
# recheck a few seconds after a 429 only spends a request to be told the
# same thing.
MIN_COOLDOWN = dt.timedelta(seconds=60)

# Used when a recheck could not reach the provider at all (a network
# error, not a refusal). Shorter than DEFAULT_COOLDOWN because nothing
# was learned about the key, only about the connection.
UNREACHABLE_COOLDOWN = dt.timedelta(minutes=15)

# Gemini's free-tier per-day quotas roll over at midnight Pacific, not at
# midnight local time and not 24h after the request that filled them.
_GEMINI_DAILY_RESET_ZONE = "America/Los_Angeles"

# "per_minute"  a short-window burst limit; back in a minute or so
# "per_day"     a daily allowance; back when the provider's day rolls over
# "quota"       the provider said quota, without saying which window
# "unknown"     no usable detail, or a provider whose 429s aren't read yet
Kind = str

_RETRY_DELAY_RE = re.compile(r"retry[ _-]?delay\"?\s*[:=]\s*\"?(\d+(?:\.\d+)?)s", re.IGNORECASE)
_QUOTA_ID_RE = re.compile(r"quota(?:Id|Metric)\"?\s*:\s*\"?([\w./-]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Cooldown:
    """What was hit, and when to look again. `retry_at` is always in the
    future relative to the `now` it was planned against."""

    kind: Kind
    retry_at: dt.datetime


def plan_cooldown(provider: str, detail: str | None, now: dt.datetime | None = None) -> Cooldown:
    """The cooldown to store for a key the provider just refused on quota
    grounds. `detail` is the provider's own message (LiteLLM's exception
    text, or the body of a failed check), which is where the reset
    information hides when there is any.
    """
    now = now or dt.datetime.now(dt.UTC)
    text = detail or ""
    delay = _retry_delay(text)

    if provider == "gemini":
        kind = _gemini_kind(text)
        if kind == "per_day":
            # The daily window is what matters, but a provider asking for
            # longer than that is still honored.
            return Cooldown(kind, max(_next_daily_reset(now), now + (delay or MIN_COOLDOWN)))
        if kind == "per_minute":
            return Cooldown(kind, now + max(delay or MIN_COOLDOWN, MIN_COOLDOWN))
        return Cooldown(kind, now + max(delay or DEFAULT_COOLDOWN, MIN_COOLDOWN))

    return Cooldown("unknown", now + max(delay or DEFAULT_COOLDOWN, MIN_COOLDOWN))


def postpone(cooldown_from: dt.datetime | None = None) -> dt.datetime:
    """The next time to try after a recheck that learned nothing (the
    provider was unreachable). Keeps the key in the waiting-on-quota group
    instead of promoting it on a failed lookup."""
    return (cooldown_from or dt.datetime.now(dt.UTC)) + UNREACHABLE_COOLDOWN


def _retry_delay(text: str) -> dt.timedelta | None:
    match = _RETRY_DELAY_RE.search(text)
    if not match:
        return None
    try:
        return dt.timedelta(seconds=float(match.group(1)))
    except ValueError:
        return None


def _gemini_kind(text: str) -> Kind:
    """Reads the quota name out of a Gemini 429 body. The names are of the
    form "GenerateRequestsPerDayPerProjectPerModel-FreeTier", so the
    window is in the name itself; the prose fallbacks below cover a body
    that carries the message without the structured violation."""
    # A 429 body names both the metric and the quota id, and only the
    # latter carries the window, so every name in the body is scanned
    # rather than just the first one that matches.
    names = [name.lower() for name in _QUOTA_ID_RE.findall(text)]
    if any("perday" in name or "per_day" in name for name in names):
        return "per_day"
    if any("perminute" in name or "per_minute" in name for name in names):
        return "per_minute"
    low = text.lower()
    if "per day" in low or "daily limit" in low:
        return "per_day"
    if "per minute" in low:
        return "per_minute"
    if "resource_exhausted" in low or "quota" in low:
        return "quota"
    return "unknown"


def _next_daily_reset(now: dt.datetime) -> dt.datetime:
    try:
        zone = ZoneInfo(_GEMINI_DAILY_RESET_ZONE)
    except (ZoneInfoNotFoundError, KeyError):
        # No tz database on this machine; a day from now is late enough to
        # be past any midnight reset.
        logger.warning("no %s timezone data; using a 24h daily cooldown", _GEMINI_DAILY_RESET_ZONE)
        return now + dt.timedelta(days=1)
    local = now.astimezone(zone)
    midnight = (local + dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return midnight.astimezone(dt.UTC)
