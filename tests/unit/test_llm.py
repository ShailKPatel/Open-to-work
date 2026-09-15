from dataclasses import dataclass, field
from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.app_settings import update_llm_settings
from app.core.db import init_db
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMUnavailableError,
    complete,
    file_part,
    image_part,
    system_message,
    user_message,
)
from app.core.settings import get_settings


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class FakeResponse:
    choices: list = field(default_factory=list)
    usage: FakeUsage = field(default_factory=FakeUsage)


def _reset_db(tmp_path: Path, monthly_budget_usd: float = 20.0):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()
    update_llm_settings(monthly_budget_usd=monthly_budget_usd)


def _fake_completion_fn(response_text: str, cost: float = 0.01, calls: list | None = None):
    def _fn(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=response_text))])

    return _fn


def test_complete_records_call_and_cost(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.0123, raising=False
    )
    calls: list = []
    result = complete(
        "bulk",
        [{"role": "user", "content": "hello"}],
        _completion_fn=_fake_completion_fn("hi there", calls=calls),
    )

    assert result.content == "hi there"
    assert result.cached is False
    assert result.cost_usd == 0.0123
    assert len(calls) == 1


def test_complete_second_identical_call_hits_cache(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.05, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    first = complete("bulk", messages, _completion_fn=_fake_completion_fn("hi", calls=calls))
    second = complete("bulk", messages, _completion_fn=_fake_completion_fn("hi", calls=calls))

    assert first.cost_usd == 0.05
    assert second.cost_usd == 0.0
    assert second.cached is True
    assert second.content == "hi"
    # underlying completion function only invoked once; the cache path
    # must not call it at all
    assert len(calls) == 1


def test_complete_different_tier_is_not_a_cache_hit(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, _completion_fn=_fake_completion_fn("a", calls=calls))
    complete("quality", messages, _completion_fn=_fake_completion_fn("b", calls=calls))

    assert len(calls) == 2


def test_complete_refuses_dispatch_over_budget(tmp_path, monkeypatch):
    _reset_db(tmp_path, monthly_budget_usd=0.01)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.05, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "expensive"}]

    complete("bulk", messages, _completion_fn=_fake_completion_fn("first", calls=calls))
    assert len(calls) == 1

    try:
        complete(
            "bulk",
            [{"role": "user", "content": "different prompt"}],
            _completion_fn=_fake_completion_fn("second", calls=calls),
        )
        raised = False
    except BudgetExceededError:
        raised = True

    assert raised is True
    # budget check must happen before dispatch, so no second call is made
    assert len(calls) == 1


def test_complete_wraps_litellm_rate_limit_error(tmp_path, monkeypatch):
    """Real litellm.RateLimitError (the exception class a provider 429
    raises through litellm), not a stand-in, so the wrap is proven against
    the real type rather than something string-matched."""
    import litellm

    _reset_db(tmp_path)

    def raises_rate_limit(**kwargs):
        raise litellm.RateLimitError(
            message="quota exceeded", llm_provider="openai", model="gpt-4o-mini"
        )

    try:
        complete("bulk", [{"role": "user", "content": "x"}], _completion_fn=raises_rate_limit)
        raised = None
    except LLMRateLimitedError as e:
        raised = e

    assert raised is not None
    assert "limiting requests" in str(raised)
    assert "litellm" not in str(raised)


def test_complete_does_not_wrap_other_errors(tmp_path, monkeypatch):
    """Only rate-limit errors get translated. Anything else (network
    error, auth error, etc.) propagates as whatever litellm actually
    raised, unchanged."""
    _reset_db(tmp_path)

    def raises_something_else(**kwargs):
        raise ConnectionError("network unreachable")

    try:
        complete("bulk", [{"role": "user", "content": "x"}], _completion_fn=raises_something_else)
        raised = None
    except Exception as e:
        raised = e

    assert isinstance(raised, ConnectionError)
    assert not isinstance(raised, LLMRateLimitedError)


_GEMINI_503 = (
    'GeminiException - {\n  "error": {\n    "code": 503,\n    "message": "This model is '
    'currently experiencing high demand.",\n    "status": "UNAVAILABLE"\n  }\n}'
)


def _overloaded():
    import litellm

    return litellm.ServiceUnavailableError(
        message=_GEMINI_503, llm_provider="gemini", model="gemini-flash-latest"
    )


def _ok_response(text: str = "ok") -> FakeResponse:
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=text))])


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("app.core.llm._sleep", sleeps.append)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0, raising=False)
    return sleeps


def test_overloaded_provider_is_retried_until_it_answers(tmp_path, no_sleep):
    _reset_db(tmp_path)
    calls: list[str] = []

    def flaky(**kwargs):
        calls.append(kwargs["model"])
        if len(calls) < 3:
            raise _overloaded()
        return _ok_response()

    result = complete("quality", [user_message("hi")], _completion_fn=flaky)

    assert result.content == "ok"
    assert calls == ["gemini/gemini-flash-latest"] * 3
    assert len(no_sleep) == 2


def test_model_still_overloaded_falls_back_to_the_other_tier_model(tmp_path, no_sleep):
    from sqlalchemy import select

    from app.core.db import LLMCall, get_db

    _reset_db(tmp_path)
    calls: list[str] = []

    def primary_down(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "gemini/gemini-flash-latest":
            raise _overloaded()
        return _ok_response("from fallback")

    result = complete("quality", [user_message("hi")], _completion_fn=primary_down)

    assert result.content == "from fallback"
    assert result.model == "gemini/gemini-flash-lite-latest"
    assert calls == ["gemini/gemini-flash-latest"] * 3 + ["gemini/gemini-flash-lite-latest"]
    db = get_db()
    try:
        row = db.execute(select(LLMCall).order_by(LLMCall.id.desc())).scalars().first()
    finally:
        db.close()
    assert row.model == "gemini/gemini-flash-lite-latest"


def test_overloaded_everywhere_raises_a_readable_error(tmp_path, no_sleep):
    _reset_db(tmp_path)
    calls: list[str] = []

    def always_down(**kwargs):
        calls.append(kwargs["model"])
        raise _overloaded()

    with pytest.raises(LLMUnavailableError) as info:
        complete("quality", [user_message("hi")], _completion_fn=always_down)

    message = str(info.value)
    assert message.startswith("Gemini is too busy to answer right now")
    assert "not a problem with your key" in message
    assert "litellm" not in message and "{" not in message
    # callers that stop a batch on a rate limit stop on this too
    assert isinstance(info.value, LLMRateLimitedError)
    assert len(calls) == 5


def test_no_fallback_to_a_model_on_a_different_provider(tmp_path, no_sleep):
    _reset_db(tmp_path)
    update_llm_settings(bulk_model="openai/gpt-4o-mini")
    calls: list[str] = []

    def always_down(**kwargs):
        calls.append(kwargs["model"])
        raise _overloaded()

    with pytest.raises(LLMUnavailableError):
        complete("quality", [user_message("hi")], _completion_fn=always_down)

    assert set(calls) == {"gemini/gemini-flash-latest"}


def test_connection_failure_is_retried_and_explained(tmp_path, no_sleep):
    import litellm

    _reset_db(tmp_path)

    def offline(**kwargs):
        raise litellm.APIConnectionError(
            message="Connection refused", llm_provider="gemini", model="gemini-flash-latest"
        )

    with pytest.raises(LLMUnavailableError, match="Couldn't reach Gemini"):
        complete("quality", [user_message("hi")], _completion_fn=offline)

    assert len(no_sleep) == 3


@pytest.mark.parametrize(
    ("exception_name", "expected"),
    [
        ("NotFoundError", 'doesn\'t recognize the model "gemini-flash-lite-latest"'),
        ("BadRequestError", 'couldn\'t process this request with "gemini-flash-lite-latest": '
                            'Invalid "model" field.'),
        ("ContextWindowExceededError", "too long"),
        ("ContentPolicyViolationError", "safety rules"),
    ],
)
def test_provider_errors_become_readable_messages(tmp_path, no_sleep, exception_name, expected):
    import litellm

    _reset_db(tmp_path)

    def fails(**kwargs):
        raise getattr(litellm, exception_name)(
            message='GeminiException - {"error": {"code": 400, "message": "Invalid \\"model\\" '
            'field."}}',
            llm_provider="gemini",
            model="gemini-flash-lite-latest",
        )

    with pytest.raises(LLMProviderError) as info:
        complete("bulk", [user_message("hi")], _completion_fn=fails)

    assert expected in str(info.value)
    assert "litellm" not in str(info.value)
    assert no_sleep == []


def test_user_message_text_only_is_plain_string_content():
    msg = user_message("describe this repo")
    assert msg == {"role": "user", "content": "describe this repo"}


def test_user_message_with_images_builds_multimodal_content():
    msg = user_message("what's in this screenshot?", images=[b"\x89PNG\r\n"])
    assert msg["role"] == "user"
    assert msg["content"][0] == {"type": "text", "text": "what's in this screenshot?"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_image_part_passes_through_url_unchanged():
    part = image_part("https://example.com/screenshot.png")
    assert part == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/screenshot.png"},
    }


def test_system_message():
    assert system_message("be terse") == {"role": "system", "content": "be terse"}


def test_file_part_bytes_becomes_base64_file_data():
    part = file_part(b"%PDF-1.4 fake", mime_type="application/pdf")
    assert part["type"] == "file"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_file_part_string_passes_through_as_file_id():
    part = file_part("https://example.com/doc.pdf")
    assert part == {"type": "file", "file": {"file_id": "https://example.com/doc.pdf"}}


def test_user_message_with_files_builds_multimodal_content():
    msg = user_message("summarize this doc", files=[b"%PDF-1.4 fake"])
    assert msg["content"][0] == {"type": "text", "text": "summarize this doc"}
    assert msg["content"][1]["type"] == "file"


def test_user_message_with_both_images_and_files():
    msg = user_message("compare these", images=[b"img"], files=[b"doc"])
    types = [part["type"] for part in msg["content"]]
    assert types == ["text", "image_url", "file"]


def test_complete_response_carries_token_counts(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    result = complete(
        "bulk",
        [{"role": "user", "content": "hello"}],
        _completion_fn=_fake_completion_fn("hi"),
    )
    assert result.tokens_in == 10  # FakeUsage defaults
    assert result.tokens_out == 5


def test_complete_records_account_id_even_via_injected_completion_fn(tmp_path, monkeypatch):
    """account_id is a plain parameter to complete(), independent of the
    key-resolution branch a test's _completion_fn bypasses (see LLMCall's
    docstring in app/core/db.py); it should land on the row either way."""
    from sqlalchemy import select

    from app.core.db import LLMCall, get_db

    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    complete(
        "bulk",
        [{"role": "user", "content": "hello"}],
        account_id=7,
        _completion_fn=_fake_completion_fn("hi"),
    )

    db = get_db()
    row = db.execute(select(LLMCall).order_by(LLMCall.id.desc())).scalars().first()
    db.close()
    assert row.account_id == 7
    assert row.key_id is None  # never resolved a key on the injected-fn path


def test_complete_resolves_and_records_key_id_on_real_dispatch(tmp_path, monkeypatch):
    """The one path that DOES resolve a real ApiKey: no _completion_fn
    passed, so complete() calls app.core.api_keys_store.resolve_dispatch_key()
    itself and should stamp the resolved key's id onto the LLMCall row."""
    from sqlalchemy import select

    from app.core import api_keys_store
    from app.core.db import LLMCall, get_db

    _reset_db(tmp_path)
    update_llm_settings(bulk_model="openai/gpt-4o-mini")
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials", lambda provider, creds: ("valid", "ok")
    )
    added, _ = api_keys_store.add_key("openai", "Test key", {"api_key": "sk-test"}, None)
    assert added is not None
    key_id = added["id"]

    monkeypatch.setattr(
        "litellm.completion",
        _fake_completion_fn("hi from openai"),
        raising=False,
    )
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.02, raising=False
    )

    complete("bulk", [{"role": "user", "content": "hello"}], account_id=3)

    db = get_db()
    row = db.execute(select(LLMCall).order_by(LLMCall.id.desc())).scalars().first()
    db.close()
    assert row.key_id == key_id
    assert row.account_id == 3
    assert row.cost_usd == 0.02


def test_complete_accepts_multimodal_user_message(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.02, raising=False
    )
    calls: list = []
    messages = [user_message("what does this diagram show?", images=[b"fake-bytes"])]

    result = complete(
        "quality", messages, _completion_fn=_fake_completion_fn("a flowchart", calls=calls)
    )

    assert result.content == "a flowchart"
    # the multimodal message shape reached the completion call untouched
    assert calls[0]["messages"][0]["content"][1]["type"] == "image_url"


def _real_dispatch_env(tmp_path: Path, monkeypatch) -> None:
    """No _completion_fn: complete() resolves a stored key and calls
    litellm.completion itself, which each test replaces."""
    _reset_db(tmp_path)
    update_llm_settings(bulk_model="openai/gpt-4o-mini")
    monkeypatch.setattr(
        "app.core.api_keys_store.validate_credentials", lambda provider, creds: ("valid", "ok")
    )


def _add_openai_key(budget_cap_usd: float | None = None) -> int:
    from app.core import api_keys_store

    added, detail = api_keys_store.add_key(
        "openai", "Test key", {"api_key": "sk-test"}, budget_cap_usd
    )
    assert added is not None, detail
    return added["id"]


def _key_status(key_id: int) -> str:
    from app.core.db import ApiKey, get_db

    db = get_db()
    try:
        return db.get(ApiKey, key_id).status
    finally:
        db.close()


def test_real_dispatch_without_any_key_raises_before_calling_the_provider(tmp_path, monkeypatch):
    _real_dispatch_env(tmp_path, monkeypatch)
    calls: list = []
    monkeypatch.setattr("litellm.completion", _fake_completion_fn("x", calls=calls), raising=False)

    with pytest.raises(ApiKeyMissingError, match="for this account"):
        complete("bulk", [user_message("hi")], account_id=5)

    assert calls == []


def test_per_key_budget_cap_refuses_dispatch_once_reached(tmp_path, monkeypatch):
    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(budget_cap_usd=0.05)
    calls: list = []
    monkeypatch.setattr("litellm.completion", _fake_completion_fn("ok", calls=calls), raising=False)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.10, raising=False)

    complete("bulk", [user_message("first")])
    with pytest.raises(BudgetExceededError, match="key budget"):
        complete("bulk", [user_message("second")])

    assert len(calls) == 1


def test_per_key_budget_cap_ignores_other_providers_spend(tmp_path, monkeypatch):
    from app.core.llm import _record

    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(budget_cap_usd=0.05)
    _record(
        tier="bulk",
        model="anthropic/some-model",
        prompt_hash="other-provider-call",
        response_json={"content": "x"},
        tokens_in=0,
        tokens_out=0,
        cost_usd=1.0,
        latency_ms=0,
        cached=False,
    )
    monkeypatch.setattr("litellm.completion", _fake_completion_fn("ok"), raising=False)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0, raising=False)

    assert complete("bulk", [user_message("hi")]).content == "ok"


def test_provider_auth_error_marks_the_key_invalid_and_propagates(tmp_path, monkeypatch):
    import litellm

    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def bad_key(**kwargs):
        raise litellm.AuthenticationError(
            message="invalid api key", llm_provider="openai", model="gpt-4o-mini"
        )

    monkeypatch.setattr("litellm.completion", bad_key, raising=False)

    with pytest.raises(LLMProviderError, match="rejected the API key"):
        complete("bulk", [user_message("hi")])

    assert _key_status(key_id) == "invalid"


def test_provider_rate_limit_marks_the_key_rate_limited_and_wraps(tmp_path, monkeypatch):
    import litellm

    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def out_of_quota(**kwargs):
        raise litellm.RateLimitError(
            message="quota exceeded", llm_provider="openai", model="gpt-4o-mini"
        )

    monkeypatch.setattr("litellm.completion", out_of_quota, raising=False)

    with pytest.raises(LLMRateLimitedError):
        complete("bulk", [user_message("hi")])

    assert _key_status(key_id) == "rate_limited"


def test_unrelated_dispatch_error_leaves_the_key_status_alone(tmp_path, monkeypatch):
    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def network_blip(**kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr("litellm.completion", network_blip, raising=False)

    with pytest.raises(ConnectionError):
        complete("bulk", [user_message("hi")])

    assert _key_status(key_id) == "valid"


def test_unpriced_model_is_recorded_at_zero_cost(tmp_path, monkeypatch):
    _reset_db(tmp_path)

    def no_price(completion_response):
        raise ValueError("model not in price map")

    monkeypatch.setattr("litellm.completion_cost", no_price, raising=False)

    result = complete("bulk", [user_message("hi")], _completion_fn=_fake_completion_fn("hi"))

    assert result.cost_usd == 0.0


def test_schema_requests_structured_output_and_parses_json(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0, raising=False)
    schema = {"type": "object", "properties": {"skills": {"type": "array"}}}
    calls: list = []

    good = complete(
        "bulk",
        [user_message("a")],
        schema=schema,
        _completion_fn=_fake_completion_fn('{"skills": ["Go"]}', calls=calls),
    )
    bad = complete(
        "bulk", [user_message("b")], schema=schema, _completion_fn=_fake_completion_fn("not json")
    )

    assert calls[0]["response_format"]["json_schema"]["schema"] == schema
    assert good.parsed == {"skills": ["Go"]}
    assert bad.parsed is None
    assert bad.content == "not json"


def test_embed_delegates_to_the_local_embedding_model(monkeypatch):
    from app.core.llm import embed

    monkeypatch.setattr(
        "app.core.embeddings.embed", lambda texts: [[float(len(t))] for t in texts]
    )

    assert embed(["ab", "c"]) == [[2.0], [1.0]]
