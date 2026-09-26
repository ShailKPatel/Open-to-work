"""Tests for app/core/key_cooldown.py and app/core/key_refresh.py: how
long an exhausted key waits before it is worth asking about again, and the
pass that comes back for it. No network here: the cooldown planner reads
text, and the refresh pass is pointed at a stubbed check.
"""

import datetime as dt
from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.db import init_db
from app.core.key_cooldown import DEFAULT_COOLDOWN, MIN_COOLDOWN, plan_cooldown
from app.core.settings import get_settings

NOW = dt.datetime(2026, 9, 27, 18, 30, tzinfo=dt.UTC)

# What Gemini actually sends back on a 429: the quota that was hit, whose
# id names the window, plus how long it wants us to wait.
GEMINI_DAILY = (
    'litellm.RateLimitError: geminiException - {"error":{"code":429,"message":"You exceeded '
    'your current quota.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.'
    'com/google.rpc.QuotaFailure","violations":[{"quotaMetric":"generativelanguage.googleapis.'
    'com/generate_content_free_tier_requests","quotaId":"GenerateRequestsPerDayPerProjectPer'
    'Model-FreeTier"}]},{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"53s"'
    "}]}}"
)
GEMINI_PER_MINUTE = GEMINI_DAILY.replace("PerDay", "PerMinute")


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


@pytest.fixture(autouse=True)
def _stub_validate_credentials(monkeypatch):
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("valid", "This key is working."),
    )


# ---------------------------------------------------------------------------
# app/core/key_cooldown.py
# ---------------------------------------------------------------------------


def test_gemini_daily_quota_waits_for_the_provider_day_to_roll_over():
    """A used-up daily allowance is not back in a minute, whatever
    retryDelay says: it comes back when Gemini's day (midnight Pacific)
    rolls over, so the wait is planned from that and not from the request
    that filled it."""
    cooldown = plan_cooldown("gemini", GEMINI_DAILY, NOW)

    assert cooldown.kind == "per_day"
    assert cooldown.retry_at > NOW + dt.timedelta(hours=4)  # 18:30 UTC is 11:30 Pacific
    assert cooldown.retry_at < NOW + dt.timedelta(days=1)


def test_gemini_per_minute_quota_comes_back_in_about_a_minute():
    cooldown = plan_cooldown("gemini", GEMINI_PER_MINUTE, NOW)

    assert cooldown.kind == "per_minute"
    # The body asked for 53s, rounded up to the minimum: a recheck a few
    # seconds early only spends a request to hear the same refusal.
    assert cooldown.retry_at == NOW + MIN_COOLDOWN


def test_a_provider_asking_for_a_long_wait_is_honoured():
    detail = 'quotaId: "GenerateRequestsPerMinute", retryDelay: "600s"'

    cooldown = plan_cooldown("gemini", detail, NOW)

    assert cooldown.retry_at == NOW + dt.timedelta(seconds=600)


def test_a_quota_with_no_named_window_still_gets_a_wait():
    cooldown = plan_cooldown("gemini", "RESOURCE_EXHAUSTED: out of quota", NOW)

    assert cooldown.kind == "quota"
    assert cooldown.retry_at == NOW + DEFAULT_COOLDOWN


@pytest.mark.parametrize("provider", ["openai", "anthropic", "mistral", "bedrock"])
def test_providers_whose_429s_are_not_read_yet_get_the_default_wait(provider):
    """Only Gemini's refusals are read in detail today. Every other
    provider gets the same interval rather than a made-up guess at its
    limits, and the kind says so instead of claiming a window."""
    cooldown = plan_cooldown(provider, "Rate limit reached", NOW)

    assert cooldown.kind == "unknown"
    assert cooldown.retry_at == NOW + DEFAULT_COOLDOWN


def test_a_wait_is_never_shorter_than_the_minimum():
    cooldown = plan_cooldown("openai", 'retryDelay: "1s"', NOW)

    assert cooldown.retry_at == NOW + MIN_COOLDOWN


# ---------------------------------------------------------------------------
# app/core/key_refresh.py
# ---------------------------------------------------------------------------


def test_refresh_pass_returns_the_keys_it_rechecked(tmp_path, monkeypatch):
    from app.core import api_keys_store
    from app.core.key_refresh import refresh_due_keys

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("gemini", "Free tier", {"api_key": "g-1"}, None)
    api_keys_store.record_dispatch_outcome(
        added["id"], ok=False, rate_limited=True, provider_detail=GEMINI_PER_MINUTE
    )
    # Its wait has not passed yet, so the pass leaves it alone.
    assert refresh_due_keys() == []

    _expire_cooldown(added["id"])
    checked = refresh_due_keys()

    assert [k["id"] for k in checked] == [added["id"]]
    assert checked[0]["status"] == "valid"


def test_refresh_thread_runs_a_pass_and_stops_when_told(monkeypatch):
    """The background loop is a thread with an interval and nothing else,
    so the test only has to see it run once and exit on its event."""
    import threading

    from app.core import key_refresh

    passes: list[int] = []

    def one_pass():
        passes.append(1)
        return []

    monkeypatch.setattr(key_refresh, "FIRST_PASS_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(key_refresh, "REFRESH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(key_refresh, "refresh_due_keys", one_pass)

    stop = threading.Event()
    thread = key_refresh.start_key_refresh(stop)
    for _ in range(200):
        if passes:
            break
        thread.join(0.01)
    stop.set()
    thread.join(1)

    assert passes
    assert not thread.is_alive()


def test_a_failing_pass_does_not_kill_the_thread(monkeypatch):
    import threading

    from app.core import key_refresh

    calls: list[int] = []

    def boom():
        calls.append(1)
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(key_refresh, "FIRST_PASS_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(key_refresh, "REFRESH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(key_refresh, "refresh_due_keys", boom)

    stop = threading.Event()
    thread = key_refresh.start_key_refresh(stop)
    for _ in range(200):
        if len(calls) > 1:
            break
        thread.join(0.01)
    stop.set()
    thread.join(1)

    assert len(calls) > 1  # it came back for the next interval


def _expire_cooldown(key_id: int) -> None:
    """Moves a key's wait into the past, which is what the passage of time
    would do."""
    from app.core.db import ApiKey, get_db

    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        row.retry_at = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)
        db.commit()
    finally:
        db.close()
