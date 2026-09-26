"""Tests for app/core/api_keys_store.py and app/api/api_keys.py, the LLM
key manager behind /apis. validate_credentials' HTTP calls are stubbed so
this suite never leaves the machine, except the network-free path (a
missing required field) that validate_credentials resolves before it
reaches for httpx.
"""

from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.db import init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def _stub_validate_credentials(monkeypatch):
    """Every test gets a network-free "valid" check by default; individual
    tests override this (or rely on the real function's network-free
    missing-field path, see module docstring) when they need a different
    outcome."""
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("valid", "This key is working."),
    )


# ---------------------------------------------------------------------------
# app/core/api_keys_store.py: direct store tests
# ---------------------------------------------------------------------------


def test_add_key_first_for_provider_is_active_second_is_not(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    first, _ = api_keys_store.add_key("openai", "First", {"api_key": "sk-1"}, None)
    second, _ = api_keys_store.add_key("openai", "Second", {"api_key": "sk-2"}, None)

    assert first["is_active"] is True
    assert second["is_active"] is False


def test_add_key_rejects_invalid_credentials_without_persisting(tmp_path, monkeypatch):
    """Missing a required field is caught inside the real
    validate_credentials() before any network call (see module
    docstring), so un-stub it for this test to run the actual rejection
    logic instead of the autouse "always valid" fake."""
    from app.core import api_keys_store
    from app.core.llm_providers import validate_credentials

    monkeypatch.setattr("app.core.api_keys_store.validate_credentials", validate_credentials)
    _reset_db(tmp_path)
    row, detail = api_keys_store.add_key("openai", "Bad", {"api_key": ""}, None)

    assert row is None
    assert "Missing" in detail
    assert api_keys_store.list_keys() == []


def test_masked_preview_never_contains_the_raw_secret(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    row, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-super-secret-value"}, None)

    assert "sk-super-secret-value" not in str(row["masked"])
    assert row["masked"]["api_key"].startswith("sk-")
    assert row["masked"]["api_key"].endswith("lue")


def test_update_key_label_and_budget_cap(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Original", {"api_key": "sk-1"}, None)

    updated = api_keys_store.update_key(added["id"], label="Renamed", budget_cap_usd=5.0)
    assert updated["label"] == "Renamed"
    assert updated["budget_cap_usd"] == 5.0


def test_update_key_leaves_budget_cap_unchanged_when_not_passed(tmp_path):
    """budget_cap_usd/allowed_account_ids use `...` as "leave unchanged"
    since None/[] are both meaningful values (no cap, no restriction);
    see the docstring in app/core/api_keys_store.py."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, 10.0)

    updated = api_keys_store.update_key(added["id"], label="New label")
    assert updated["budget_cap_usd"] == 10.0


def test_update_key_unknown_id_returns_none(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    assert api_keys_store.update_key(999999, label="x") is None


def test_set_enabled_toggles_and_disabled_key_skipped_by_dispatch(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)

    assert api_keys_store.resolve_dispatch_key("openai", None) is not None
    api_keys_store.set_enabled(added["id"], False)
    assert api_keys_store.resolve_dispatch_key("openai", None) is None

    api_keys_store.set_enabled(added["id"], True)
    assert api_keys_store.resolve_dispatch_key("openai", None) is not None


def test_delete_key_promotes_next_oldest_when_active_one_is_removed(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    first, _ = api_keys_store.add_key("openai", "First", {"api_key": "sk-1"}, None)
    second, _ = api_keys_store.add_key("openai", "Second", {"api_key": "sk-2"}, None)
    assert first["is_active"] is True

    api_keys_store.delete_key(first["id"])

    keys = api_keys_store.list_keys()
    assert len(keys) == 1
    assert keys[0]["id"] == second["id"]
    assert keys[0]["is_active"] is True


def test_delete_key_of_inactive_row_does_not_touch_the_active_one(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    first, _ = api_keys_store.add_key("openai", "First", {"api_key": "sk-1"}, None)
    second, _ = api_keys_store.add_key("openai", "Second", {"api_key": "sk-2"}, None)

    api_keys_store.delete_key(second["id"])

    keys = api_keys_store.list_keys()
    assert len(keys) == 1
    assert keys[0]["id"] == first["id"]
    assert keys[0]["is_active"] is True


def test_delete_key_unknown_id_is_a_safe_no_op(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    api_keys_store.delete_key(999999)  # must not raise


def test_activate_key_switches_active_flag_within_provider_only(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    first, _ = api_keys_store.add_key("openai", "First", {"api_key": "sk-1"}, None)
    second, _ = api_keys_store.add_key("openai", "Second", {"api_key": "sk-2"}, None)
    other_provider, _ = api_keys_store.add_key(
        "mistral", "Other", {"api_key": "ms-1"}, None
    )

    api_keys_store.activate_key(second["id"])

    keys = {k["id"]: k for k in api_keys_store.list_keys()}
    assert keys[first["id"]]["is_active"] is False
    assert keys[second["id"]]["is_active"] is True
    # a different provider's active key is untouched by this call
    assert keys[other_provider["id"]]["is_active"] is True


def test_check_key_revalidates_in_place(tmp_path, monkeypatch):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)

    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("invalid", "no longer works"),
    )
    checked = api_keys_store.check_key(added["id"])

    assert checked["status"] == "invalid"
    assert checked["last_check_detail"] == "no longer works"


def test_check_key_unknown_id_returns_none(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    assert api_keys_store.check_key(999999) is None


def test_resolve_dispatch_key_respects_account_allow_list(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key(
        "openai", "Restricted", {"api_key": "sk-1"}, None, allowed_account_ids=[5]
    )

    assert api_keys_store.resolve_dispatch_key("openai", 5) is not None
    assert api_keys_store.resolve_dispatch_key("openai", 6) is None
    assert api_keys_store.resolve_dispatch_key("openai", None) is None


def test_resolve_dispatch_key_falls_back_past_a_disabled_active_key(tmp_path):
    """An enabled-but-not-active key still serves the provider if the
    active one is disabled; resolve_dispatch_key must not just check
    is_active and stop there."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    active, _ = api_keys_store.add_key("openai", "Active", {"api_key": "sk-1"}, None)
    backup, _ = api_keys_store.add_key("openai", "Backup", {"api_key": "sk-2"}, None)
    api_keys_store.set_enabled(active["id"], False)

    resolved = api_keys_store.resolve_dispatch_key("openai", None)
    assert resolved is not None
    key_id, credentials, _ = resolved
    assert key_id == backup["id"]
    assert credentials == {"api_key": "sk-2"}


def test_resolve_dispatch_key_none_when_nothing_configured(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    assert api_keys_store.resolve_dispatch_key("openai", None) is None


def test_get_active_status_reports_unconfigured_and_configured(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    assert api_keys_store.get_active_status("openai")["configured"] is False

    api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    status = api_keys_store.get_active_status("openai")
    assert status["configured"] is True
    assert status["status"] == "valid"


def test_resolve_dispatch_keys_returns_every_usable_key_in_try_order(tmp_path):
    """app/core/llm.py walks this list top to bottom on failover, so the
    order is the policy: healthy before marked-dead, active before the
    rest, oldest before newest."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    active, _ = api_keys_store.add_key("openai", "Active", {"api_key": "sk-1"}, None)
    second, _ = api_keys_store.add_key("openai", "Second", {"api_key": "sk-2"}, None)
    third, _ = api_keys_store.add_key("openai", "Third", {"api_key": "sk-3"}, None)

    assert [k.id for k in api_keys_store.resolve_dispatch_keys("openai", None)] == [
        active["id"], second["id"], third["id"]
    ]

    # A key the provider complained about last time goes to the back of
    # the queue rather than out of it: quotas reset, and dropping it would
    # leave a device with one key unable to dispatch at all.
    api_keys_store.record_dispatch_outcome(active["id"], ok=False, rate_limited=True)
    assert [k.id for k in api_keys_store.resolve_dispatch_keys("openai", None)] == [
        second["id"], third["id"], active["id"]
    ]


def test_resolve_dispatch_keys_leaves_out_what_cannot_serve_the_call(tmp_path):
    """Disabled keys and keys fenced to other accounts are not "try them
    last", they are not candidates at all."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    usable, _ = api_keys_store.add_key("openai", "Usable", {"api_key": "sk-1"}, None)
    off, _ = api_keys_store.add_key("openai", "Switched off", {"api_key": "sk-2"}, None)
    api_keys_store.add_key(
        "openai", "Someone else's", {"api_key": "sk-3"}, None, allowed_account_ids=[99]
    )
    api_keys_store.set_enabled(off["id"], False)

    assert [k.id for k in api_keys_store.resolve_dispatch_keys("openai", 7)] == [usable["id"]]
    assert api_keys_store.resolve_dispatch_keys("anthropic", 7) == []


def test_resolve_dispatch_keys_carries_the_label_and_cap_for_each(tmp_path):
    """The label is what a failure names ("Personal: out of quota"), and
    the cap is checked before a call is spent on that key, so both travel
    with the credentials rather than being read back later."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    api_keys_store.add_key("openai", "Personal", {"api_key": "sk-1"}, 12.5)

    key = api_keys_store.resolve_dispatch_keys("openai", None)[0]
    assert key.label == "Personal"
    assert key.credentials == {"api_key": "sk-1"}
    assert key.budget_cap_usd == 12.5


def test_a_key_that_answers_again_stops_being_marked_dead(tmp_path):
    """Set when a dispatch succeeds on a key an earlier run had written
    off, so the /apis page reflects the quota having reset instead of
    showing a working key as rate limited until someone clicks check."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)

    api_keys_store.record_dispatch_outcome(added["id"], ok=True)

    row = next(k for k in api_keys_store.list_keys() if k["id"] == added["id"])
    assert row["status"] == "valid"
    assert "Working again" in row["last_check_detail"]


def test_record_dispatch_outcome_success_is_a_no_op(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=True)

    row = next(k for k in api_keys_store.list_keys() if k["id"] == added["id"])
    assert row["status"] == "valid"  # unchanged from add_key's own check


def test_record_dispatch_outcome_marks_invalid_and_rate_limited(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)

    api_keys_store.record_dispatch_outcome(added["id"], ok=False, detail="bad key")
    row = next(k for k in api_keys_store.list_keys() if k["id"] == added["id"])
    assert row["status"] == "invalid"
    assert row["last_check_detail"] == "bad key"

    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)
    row = next(k for k in api_keys_store.list_keys() if k["id"] == added["id"])
    assert row["status"] == "rate_limited"


def test_an_exhausted_key_records_when_it_ran_out_and_when_to_look_again(tmp_path):
    """The quota clock, which is what makes an exhausted key recoverable
    without anyone watching it: when it ran out, which window it hit, and
    the first moment asking again is worth a request."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("gemini", "Free tier", {"api_key": "g-1"}, None)

    api_keys_store.record_dispatch_outcome(
        added["id"],
        ok=False,
        rate_limited=True,
        detail="Out of quota.",
        provider_detail='"quotaId":"GenerateRequestsPerMinute-FreeTier","retryDelay":"90s"',
    )

    row = _key(added["id"])
    assert row["status"] == "rate_limited"
    assert row["exhaustion_kind"] == "per_minute"
    assert row["exhausted_at"] is not None
    assert row["retry_at"] is not None
    assert row["auto_rechecked"] is True
    assert row["recheck_due"] is False  # its wait has not passed yet


def test_a_blocked_key_is_kept_apart_from_a_quota_and_carries_no_clock(tmp_path):
    """A provider shutting a key off is the one failure that does not fix
    itself, so it gets its own status and no cooldown: nothing should come
    back and ask again on a timer."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("gemini", "Suspended", {"api_key": "g-1"}, None)

    api_keys_store.record_dispatch_outcome(
        added["id"], ok=False, blocked=True, detail="This key is suspended."
    )

    row = _key(added["id"])
    assert row["status"] == "blocked"
    assert row["retry_at"] is None
    assert row["exhausted_at"] is None
    assert row["auto_rechecked"] is False
    assert row["recheck_due"] is False


def test_a_key_that_answers_again_loses_its_quota_clock(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)

    api_keys_store.record_dispatch_outcome(added["id"], ok=True)

    row = _key(added["id"])
    assert row["status"] == "valid"
    assert (row["retry_at"], row["exhausted_at"], row["exhaustion_kind"]) == (None, None, None)


def test_recheck_due_only_touches_exhausted_keys_whose_wait_is_over(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    waiting, _ = api_keys_store.add_key("openai", "Still waiting", {"api_key": "sk-1"}, None)
    due, _ = api_keys_store.add_key("openai", "Wait is over", {"api_key": "sk-2"}, None)
    blocked, _ = api_keys_store.add_key("openai", "Blocked", {"api_key": "sk-3"}, None)
    healthy, _ = api_keys_store.add_key("openai", "Fine", {"api_key": "sk-4"}, None)
    api_keys_store.record_dispatch_outcome(waiting["id"], ok=False, rate_limited=True)
    api_keys_store.record_dispatch_outcome(due["id"], ok=False, rate_limited=True)
    api_keys_store.record_dispatch_outcome(blocked["id"], ok=False, blocked=True)
    _expire_cooldown(due["id"])

    checked = api_keys_store.recheck_keys("due")

    assert [k["id"] for k in checked] == [due["id"]]
    assert _key(due["id"])["status"] == "valid"
    assert _key(waiting["id"])["status"] == "rate_limited"
    assert _key(blocked["id"])["status"] == "blocked"
    assert _key(healthy["id"])["status"] == "valid"


def test_recheck_exhausted_skips_the_wait_and_recheck_blocked_is_the_only_way_back(tmp_path):
    """The two manual scopes. "exhausted" is someone not wanting to wait
    out the timer; "blocked" is the only thing that ever looks at a key the
    provider shut off, which is why it is never automatic."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    waiting, _ = api_keys_store.add_key("openai", "Waiting", {"api_key": "sk-1"}, None)
    blocked, _ = api_keys_store.add_key("openai", "Blocked", {"api_key": "sk-2"}, None)
    api_keys_store.record_dispatch_outcome(waiting["id"], ok=False, rate_limited=True)
    api_keys_store.record_dispatch_outcome(blocked["id"], ok=False, blocked=True)

    assert [k["id"] for k in api_keys_store.recheck_keys("exhausted")] == [waiting["id"]]
    assert _key(waiting["id"])["status"] == "valid"
    assert _key(blocked["id"])["status"] == "blocked"

    assert [k["id"] for k in api_keys_store.recheck_keys("blocked")] == [blocked["id"]]
    assert _key(blocked["id"])["status"] == "valid"


def test_recheck_leaves_a_key_waiting_when_the_provider_cannot_be_reached(tmp_path, monkeypatch):
    """A failed lookup teaches nothing about the key, so it stays in the
    waiting group with its wait pushed out rather than being promoted or
    written off."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)
    _expire_cooldown(added["id"])
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: ("unknown", "Couldn't reach the provider to check this."),
    )

    api_keys_store.recheck_keys("due")

    row = _key(added["id"])
    assert row["status"] == "rate_limited"
    assert row["recheck_due"] is False  # pushed out, so the next pass isn't a hot loop


def test_recheck_replans_the_wait_when_the_provider_says_no_again(tmp_path, monkeypatch):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("gemini", "Key", {"api_key": "g-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)
    _expire_cooldown(added["id"])
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials",
        lambda provider, credentials: (
            "rate_limited",
            'Out of quota. Provider said: "quotaId":"GenerateRequestsPerDay-FreeTier"',
        ),
    )

    api_keys_store.recheck_keys("due")

    row = _key(added["id"])
    assert row["status"] == "rate_limited"
    assert row["exhaustion_kind"] == "per_day"
    assert row["recheck_due"] is False


def test_switched_off_keys_are_never_rechecked(tmp_path):
    """A key that is off is not dispatched with, so spending a request to
    learn its status is waste."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Off", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)
    _expire_cooldown(added["id"])
    api_keys_store.set_enabled(added["id"], False)

    assert api_keys_store.recheck_keys("due") == []
    assert _key(added["id"])["status"] == "rate_limited"


def test_recheck_rejects_a_scope_it_does_not_know(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    with pytest.raises(ValueError, match="unknown recheck scope"):
        api_keys_store.recheck_keys("everything")


def test_dispatch_prefers_a_key_whose_wait_is_over_to_one_still_waiting(tmp_path):
    """Both keys are exhausted, so neither is dropped, but the one whose
    quota window has rolled over is the one more likely to answer and goes
    first."""
    from app.core import api_keys_store

    _reset_db(tmp_path)
    still_waiting, _ = api_keys_store.add_key("openai", "Waiting", {"api_key": "sk-1"}, None)
    wait_over, _ = api_keys_store.add_key("openai", "Over", {"api_key": "sk-2"}, None)
    blocked, _ = api_keys_store.add_key("openai", "Blocked", {"api_key": "sk-3"}, None)
    api_keys_store.record_dispatch_outcome(still_waiting["id"], ok=False, rate_limited=True)
    api_keys_store.record_dispatch_outcome(wait_over["id"], ok=False, rate_limited=True)
    api_keys_store.record_dispatch_outcome(blocked["id"], ok=False, blocked=True)
    _expire_cooldown(wait_over["id"])

    assert [k.id for k in api_keys_store.resolve_dispatch_keys("openai", None)] == [
        wait_over["id"], still_waiting["id"], blocked["id"]
    ]


def _key(key_id: int) -> dict:
    from app.core import api_keys_store

    return next(k for k in api_keys_store.list_keys() if k["id"] == key_id)


def _expire_cooldown(key_id: int) -> None:
    """Moves a key's wait into the past, which is what the passage of time
    would do."""
    import datetime as dt

    from app.core.db import ApiKey, get_db

    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        row.retry_at = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# app/api/api_keys.py: HTTP surface
# ---------------------------------------------------------------------------


def test_list_providers_covers_every_registered_provider(tmp_path):
    _reset_db(tmp_path)
    body = _client().get("/api/api-keys/providers").json()
    providers = {p["provider"] for p in body}
    assert providers == {
        "openai", "anthropic", "gemini", "mistral", "azure_openai", "bedrock", "ollama",
    }


def test_default_provider_reflects_settings(tmp_path):
    from app.core.app_settings import update_llm_settings

    _reset_db(tmp_path)
    body = _client().get("/api/api-keys/default-provider").json()
    assert body["bulk"] == "gemini"
    assert body["bulk_label"] == "Gemini"

    update_llm_settings(bulk_model="anthropic/claude-haiku-4-5")
    body = _client().get("/api/api-keys/default-provider").json()
    assert body["bulk"] == "anthropic"
    assert body["quality"] == "gemini"


def test_add_key_endpoint_rejects_unknown_provider(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post(
        "/api/api-keys", json={"provider": "not-a-provider", "credentials": {}}
    )
    assert resp.status_code == 422


def test_add_key_endpoint_rejects_invalid_credentials(tmp_path, monkeypatch):
    """Same un-stubbing as test_add_key_rejects_invalid_credentials_
    without_persisting above; this exercises the real missing-field
    check through the HTTP layer."""
    from app.core.llm_providers import validate_credentials

    monkeypatch.setattr("app.core.api_keys_store.validate_credentials", validate_credentials)
    _reset_db(tmp_path)
    resp = _client().post(
        "/api/api-keys", json={"provider": "openai", "credentials": {"api_key": ""}}
    )
    assert resp.status_code == 422
    assert _client().get("/api/api-keys").json() == []


def test_add_key_endpoint_never_returns_the_raw_credential(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post(
        "/api/api-keys",
        json={"provider": "openai", "label": "Mine", "credentials": {"api_key": "sk-real-value"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "sk-real-value" not in str(body)
    assert body["is_active"] is True


def test_update_key_endpoint_clears_budget_cap(tmp_path):
    client = _client()
    _reset_db(tmp_path)
    added = client.post(
        "/api/api-keys",
        json={
            "provider": "openai", "credentials": {"api_key": "sk-1"}, "budget_cap_usd": 5.0,
        },
    ).json()

    resp = client.patch(
        f"/api/api-keys/{added['id']}",
        json={"budget_cap_usd": None, "has_budget_cap": False},
    )
    assert resp.status_code == 200
    assert resp.json()["budget_cap_usd"] is None


def test_update_key_endpoint_unknown_id_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().patch("/api/api-keys/999999", json={"label": "x"})
    assert resp.status_code == 404


def test_check_key_endpoint_unknown_id_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post("/api/api-keys/999999/check")
    assert resp.status_code == 404


def test_activate_enable_disable_endpoints(tmp_path):
    client = _client()
    _reset_db(tmp_path)
    first = client.post(
        "/api/api-keys", json={"provider": "openai", "credentials": {"api_key": "sk-1"}}
    ).json()
    second = client.post(
        "/api/api-keys", json={"provider": "openai", "credentials": {"api_key": "sk-2"}}
    ).json()

    resp = client.post(f"/api/api-keys/{second['id']}/activate")
    assert resp.json()["is_active"] is True
    assert client.get("/api/api-keys").json()[0]["is_active"] is False  # first, now inactive

    resp = client.post(f"/api/api-keys/{first['id']}/disable")
    assert resp.json()["enabled"] is False

    resp = client.post(f"/api/api-keys/{first['id']}/enable")
    assert resp.json()["enabled"] is True


def test_activate_endpoint_unknown_id_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post("/api/api-keys/999999/activate")
    assert resp.status_code == 404


def test_delete_key_endpoint(tmp_path):
    client = _client()
    _reset_db(tmp_path)
    added = client.post(
        "/api/api-keys", json={"provider": "openai", "credentials": {"api_key": "sk-1"}}
    ).json()

    resp = client.delete(f"/api/api-keys/{added['id']}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
    assert client.get("/api/api-keys").json() == []


def test_delete_key_endpoint_unknown_id_is_still_reported_deleted(tmp_path):
    """Documents existing behavior: delete_key() no-ops on a missing row
    rather than 404ing (see app/core/api_keys_store.py), and the endpoint
    mirrors that: idempotent rather than an error."""
    _reset_db(tmp_path)
    resp = _client().delete("/api/api-keys/999999")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}


def test_recheck_endpoint_reports_what_it_checked_and_what_came_back(tmp_path):
    from app.core import api_keys_store

    _reset_db(tmp_path)
    added, _ = api_keys_store.add_key("openai", "Key", {"api_key": "sk-1"}, None)
    api_keys_store.record_dispatch_outcome(added["id"], ok=False, rate_limited=True)

    client = _client()
    # Its wait has not passed, so the automatic scope has nothing to do.
    due = client.post("/api/api-keys/recheck", json={"scope": "due"}).json()
    assert (due["checked"], due["recovered"]) == (0, 0)

    skipped_wait = client.post("/api/api-keys/recheck", json={"scope": "exhausted"}).json()
    assert (skipped_wait["checked"], skipped_wait["recovered"]) == (1, 1)
    assert skipped_wait["keys"][0]["status"] == "valid"
    assert skipped_wait["keys"][0]["retry_at"] is None


def test_recheck_endpoint_defaults_to_the_automatic_scope(tmp_path):
    _reset_db(tmp_path)

    body = _client().post("/api/api-keys/recheck").json()

    assert body["scope"] == "due"
    assert body["checked"] == 0


def test_recheck_endpoint_rejects_an_unknown_scope(tmp_path):
    _reset_db(tmp_path)

    res = _client().post("/api/api-keys/recheck", json={"scope": "everything"})

    assert res.status_code == 422
