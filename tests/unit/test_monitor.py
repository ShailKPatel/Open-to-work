"""Tests for app/api/monitor.py, the /monitor page's backend: the per-key,
per-account, per-provider, and per-tier LLM usage breakdown
(LLMCall.account_id/key_id, see app/core/db/models.py), plus the event log and
live-status endpoints.
"""

from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.db import Account, ApiKey, LLMCall, get_db, init_db
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
def _stub_github_status(monkeypatch):
    """MonitorStatus's github field hits a real GitHub API call
    (app.ingest.github.client.GitHubClient), stubbed the same way
    conftest.py's account-creation fixture stubs it, so these tests never
    touch the network."""

    class _FakeGH:
        def get_rate_limit(self):
            import datetime as _dt

            core = type("Core", (), {
                "limit": 60,
                "remaining": 42,
                "reset": _dt.datetime.now(_dt.UTC),
            })()
            resources = type("Resources", (), {"core": core})()
            return type("RateLimit", (), {"resources": resources})()

    class _FakeGitHubClient:
        def __init__(self, *args, **kwargs):
            self._gh = _FakeGH()

    monkeypatch.setattr("app.ingest.github.client.GitHubClient", _FakeGitHubClient)


def _make_account(**overrides) -> int:
    db = get_db()
    defaults = dict(first_name="Ada", last_name="Lovelace", github_username="octocat")
    defaults.update(overrides)
    account = Account(**defaults)
    db.add(account)
    db.commit()
    account_id = account.id
    db.close()
    return account_id


def _make_key(provider: str = "openai", label: str = "Test key") -> int:
    """Inserted directly (not via api_keys_store.add_key()) so these tests
    never make the real validate_credentials() network call; monitor.py
    only ever reads masked/status fields off the row, never decrypts."""
    db = get_db()
    key = ApiKey(
        provider=provider,
        label=label,
        encrypted_credentials="unused-in-these-tests",
        masked_preview={"api_key": "sk-•••••••"},
        is_active=True,
        enabled=True,
        allowed_account_ids=[],
        status="valid",
        budget_cap_usd=None,
    )
    db.add(key)
    db.commit()
    key_id = key.id
    db.close()
    return key_id


def _make_call(**overrides) -> int:
    db = get_db()
    defaults = dict(
        tier="bulk",
        model="openai/gpt-4o-mini",
        prompt_hash="hash",
        response_json={"content": "x"},
        tokens_in=100,
        tokens_out=50,
        cost_usd=1.0,
        latency_ms=10,
        cached=False,
        account_id=None,
        key_id=None,
    )
    defaults.update(overrides)
    row = LLMCall(**defaults)
    db.add(row)
    db.commit()
    row_id = row.id
    db.close()
    return row_id


def test_status_lists_configured_keys_with_month_usage(tmp_path):
    _reset_db(tmp_path)
    key_id = _make_key(label="My OpenAI key")
    _make_call(key_id=key_id, cost_usd=1.23)
    _make_call(key_id=key_id, cost_usd=2.00)
    _make_call(key_id=None, cost_usd=99.0)  # unattributed, must not count here

    resp = _client().get("/api/monitor/status")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["keys"]) == 1
    key_out = body["keys"][0]
    assert key_out["id"] == key_id
    assert key_out["label"] == "My OpenAI key"
    assert key_out["provider_label"] == "OpenAI"
    assert key_out["calls_month"] == 2
    assert key_out["spent_usd_month"] == pytest.approx(3.23)
    # LlmStatus's own spend total is still the WHOLE month, unattributed
    # calls included. The per-key breakdown is additive detail, not a
    # replacement.
    assert body["llm"]["spent_usd"] == pytest.approx(102.23)


def test_status_key_with_no_calls_shows_zero_not_missing(tmp_path):
    _reset_db(tmp_path)
    _make_key()

    body = _client().get("/api/monitor/status").json()
    assert body["keys"][0]["calls_month"] == 0
    assert body["keys"][0]["spent_usd_month"] == 0.0


def test_llm_usage_breaks_down_by_account_provider_key_and_tier(tmp_path):
    _reset_db(tmp_path)
    acc1 = _make_account(first_name="Ada", last_name="Lovelace")
    acc2 = _make_account(first_name="Grace", last_name="Hopper", github_username="ghopper")
    key1 = _make_key(provider="openai", label="OpenAI key")

    _make_call(
        account_id=acc1, key_id=key1, model="openai/gpt-4o-mini", tier="bulk",
        cost_usd=1.0, tokens_in=100, tokens_out=50, cached=False,
    )
    _make_call(
        account_id=acc2, key_id=None, model="mistral/mistral-small-latest", tier="quality",
        cost_usd=2.0, tokens_in=10, tokens_out=10, cached=True,
    )
    _make_call(
        account_id=None, key_id=None, model="openai/gpt-4o-mini", tier="bulk", cost_usd=0.5,
    )

    body = _client().get("/api/monitor/llm/usage").json()
    assert body["totals"]["calls"] == 3
    assert body["totals"]["cost_usd"] == pytest.approx(3.5)
    assert body["totals"]["cached_calls"] == 1

    by_provider = {b["key"]: b for b in body["by_provider"]}
    assert by_provider["openai"]["calls"] == 2
    assert by_provider["openai"]["cost_usd"] == pytest.approx(1.5)
    assert by_provider["mistral"]["calls"] == 1

    by_key = {b["key"]: b for b in body["by_key"]}
    assert by_key[str(key1)]["label"] == "OpenAI key"
    assert by_key[str(key1)]["cost_usd"] == pytest.approx(1.0)
    assert by_key["none"]["label"] == "(unattributed)"
    assert by_key["none"]["cost_usd"] == pytest.approx(2.5)

    by_account = {b["key"]: b for b in body["by_account"]}
    assert by_account[str(acc1)]["label"] == "Ada Lovelace"
    assert by_account[str(acc2)]["label"] == "Grace Hopper"
    assert by_account["none"]["label"] == "(unattributed)"
    assert by_account["none"]["cost_usd"] == pytest.approx(0.5)

    by_tier = {b["key"]: b for b in body["by_tier"]}
    assert by_tier["bulk"]["calls"] == 2
    assert by_tier["quality"]["calls"] == 1


def test_llm_usage_account_filter_narrows_every_breakdown(tmp_path):
    _reset_db(tmp_path)
    acc1 = _make_account()
    acc2 = _make_account(github_username="ghopper")
    _make_call(account_id=acc1, model="openai/gpt-4o-mini", cost_usd=1.0)
    _make_call(account_id=acc2, model="mistral/mistral-small-latest", cost_usd=5.0)

    body = _client().get(f"/api/monitor/llm/usage?account_id={acc1}").json()
    assert body["totals"]["calls"] == 1
    assert body["totals"]["cost_usd"] == pytest.approx(1.0)
    assert [b["key"] for b in body["by_provider"]] == ["openai"]


def test_llm_usage_provider_filter(tmp_path):
    _reset_db(tmp_path)
    _make_call(model="openai/gpt-4o-mini", cost_usd=1.0)
    _make_call(model="mistral/mistral-small-latest", cost_usd=5.0)

    body = _client().get("/api/monitor/llm/usage?provider=mistral").json()
    assert body["totals"]["calls"] == 1
    assert body["totals"]["cost_usd"] == pytest.approx(5.0)


def test_llm_usage_key_filter(tmp_path):
    _reset_db(tmp_path)
    key1 = _make_key(label="Key one")
    key2 = _make_key(label="Key two")
    _make_call(key_id=key1, cost_usd=1.0)
    _make_call(key_id=key2, cost_usd=9.0)

    body = _client().get(f"/api/monitor/llm/usage?key_id={key1}").json()
    assert body["totals"]["calls"] == 1
    assert body["totals"]["cost_usd"] == pytest.approx(1.0)


def test_llm_usage_days_window_excludes_old_calls(tmp_path):
    import datetime as dt

    _reset_db(tmp_path)
    db = get_db()
    old = LLMCall(
        tier="bulk", model="openai/gpt-4o-mini", prompt_hash="old",
        response_json={"content": "x"}, cost_usd=7.0,
        created_at=dt.datetime.now(dt.UTC) - dt.timedelta(days=400),
    )
    db.add(old)
    db.commit()
    db.close()
    _make_call(cost_usd=1.0)

    body = _client().get("/api/monitor/llm/usage?days=30").json()
    assert body["totals"]["calls"] == 1
    assert body["totals"]["cost_usd"] == pytest.approx(1.0)


def test_llm_calls_returns_resolved_labels_and_supports_filters(tmp_path):
    _reset_db(tmp_path)
    acc = _make_account()
    key_id = _make_key(label="My key")
    call_id = _make_call(account_id=acc, key_id=key_id, tier="quality", model="openai/gpt-4o")
    _make_call(account_id=None, key_id=None, tier="bulk", model="mistral/mistral-small-latest")

    resp = _client().get("/api/monitor/llm/calls")
    rows = resp.json()
    assert len(rows) == 2
    ours = next(r for r in rows if r["id"] == call_id)
    assert ours["account_name"] == "Ada Lovelace"
    assert ours["key_label"] == "My key"
    assert ours["provider"] == "openai"
    assert ours["provider_label"] == "OpenAI"

    by_tier = _client().get("/api/monitor/llm/calls?tier=quality").json()
    assert [r["id"] for r in by_tier] == [call_id]

    by_account = _client().get(f"/api/monitor/llm/calls?account_id={acc}").json()
    assert [r["id"] for r in by_account] == [call_id]

    by_provider = _client().get("/api/monitor/llm/calls?provider=mistral").json()
    assert len(by_provider) == 1
    assert by_provider[0]["id"] != call_id


def test_llm_calls_limit_is_capped(tmp_path):
    _reset_db(tmp_path)
    for _ in range(5):
        _make_call()

    rows = _client().get("/api/monitor/llm/calls?limit=2").json()
    assert len(rows) == 2


def test_events_endpoint_filters_by_account_id(tmp_path):
    _reset_db(tmp_path)
    from app.core.rate_limits import record_event

    acc1 = _make_account()
    acc2 = _make_account(github_username="ghopper")
    record_event("llm", "rate_limited", "for acc1", account_id=acc1)
    record_event("llm", "rate_limited", "for acc2", account_id=acc2)
    record_event("github", "rate_limited", "no account context")

    all_events = _client().get("/api/monitor/events").json()
    assert len(all_events) == 3

    acc1_events = _client().get(f"/api/monitor/events?account_id={acc1}").json()
    assert len(acc1_events) == 1
    assert acc1_events[0]["detail"] == "for acc1"
    assert acc1_events[0]["account_id"] == acc1


def test_llm_usage_breaks_down_by_purpose(tmp_path):
    """Which feature the money went to, which a model name and a tier do not
    say on their own."""
    _reset_db(tmp_path)
    _make_call(purpose="repo_facts", cost_usd=1.0)
    _make_call(purpose="repo_facts", cost_usd=2.0, cached=True)
    _make_call(purpose="pagefit_trim", cost_usd=0.5)
    _make_call(purpose=None, cost_usd=0.25)  # a row from before the column existed

    body = _client().get("/api/monitor/llm/usage").json()

    by_purpose = {b["key"]: b for b in body["by_purpose"]}
    assert by_purpose["repo_facts"]["label"] == "Project extraction (skills + links)"
    assert by_purpose["repo_facts"]["calls"] == 2
    assert by_purpose["repo_facts"]["cached_calls"] == 1
    assert by_purpose["repo_facts"]["cost_usd"] == pytest.approx(3.0)
    assert by_purpose["pagefit_trim"]["label"] == "Page-fit trimming"
    assert by_purpose["none"]["label"] == "(unattributed)"
    assert by_purpose["none"]["cost_usd"] == pytest.approx(0.25)


def test_an_unlabelled_purpose_shows_its_own_name(tmp_path):
    """A call site that ships before anyone adds a readable label still
    appears in the breakdown, under its raw purpose."""
    _reset_db(tmp_path)
    _make_call(purpose="something_new", cost_usd=1.0)

    body = _client().get("/api/monitor/llm/usage").json()

    by_purpose = {b["key"]: b for b in body["by_purpose"]}
    assert by_purpose["something_new"]["label"] == "something_new"


def test_llm_usage_purpose_filter_narrows_every_breakdown(tmp_path):
    _reset_db(tmp_path)
    _make_call(purpose="repo_facts", tier="bulk", cost_usd=1.0)
    _make_call(purpose="resume_build", tier="quality", cost_usd=4.0)

    body = _client().get("/api/monitor/llm/usage?purpose=repo_facts").json()

    assert body["totals"]["cost_usd"] == pytest.approx(1.0)
    assert [b["key"] for b in body["by_tier"]] == ["bulk"]


def test_llm_calls_carry_and_filter_by_purpose(tmp_path):
    _reset_db(tmp_path)
    facts_call = _make_call(purpose="repo_facts")
    _make_call(purpose="resume_build")

    rows = _client().get("/api/monitor/llm/calls?purpose=repo_facts").json()

    assert [r["id"] for r in rows] == [facts_call]
    assert rows[0]["purpose"] == "repo_facts"
