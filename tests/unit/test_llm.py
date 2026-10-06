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
    _with_prompt_caching,
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

    db_module.reset_engine()
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


def test_complete_different_model_per_tier_is_not_a_cache_hit(tmp_path, monkeypatch):
    """The tiers default to different models, and a different model can
    answer differently, so its answer is not this one's."""
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, _completion_fn=_fake_completion_fn("a", calls=calls))
    complete("quality", messages, _completion_fn=_fake_completion_fn("b", calls=calls))

    assert len(calls) == 2


def test_one_model_on_both_tiers_is_one_prompt(tmp_path, monkeypatch):
    """Keyed on the model, not the tier: with the same model set for both,
    the second call is the first call's prompt and is served from it rather
    than billed again."""
    _reset_db(tmp_path)
    update_llm_settings(bulk_model="gemini/x", quality_model="gemini/x")
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, _completion_fn=_fake_completion_fn("a", calls=calls))
    second = complete("quality", messages, _completion_fn=_fake_completion_fn("b", calls=calls))

    assert len(calls) == 1
    assert second.cached is True
    assert second.cost_usd == 0.0


def test_whitespace_only_difference_is_the_same_prompt(tmp_path, monkeypatch):
    """A README refetched or a job posting repasted routinely differs by
    trailing spaces, CRLF line endings and blank lines, and by nothing
    else. Same answer, so same prompt."""
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.02, raising=False
    )
    calls: list = []

    complete(
        "bulk",
        [{"role": "user", "content": "line one\nline two"}],
        _completion_fn=_fake_completion_fn("hi", calls=calls),
    )
    second = complete(
        "bulk",
        [{"role": "user", "content": "  line one   \r\n\n\n\nline two  \n"}],
        _completion_fn=_fake_completion_fn("hi", calls=calls),
    )

    assert len(calls) == 1
    assert second.cached is True


def test_different_wording_is_still_a_different_prompt(tmp_path, monkeypatch):
    """The whitespace normalization above must not collapse prompts that
    actually differ."""
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.02, raising=False
    )
    calls: list = []

    complete(
        "bulk",
        [{"role": "user", "content": "line one"}],
        _completion_fn=_fake_completion_fn("a", calls=calls),
    )
    complete(
        "bulk",
        [{"role": "user", "content": "line  one"}],
        _completion_fn=_fake_completion_fn("b", calls=calls),
    )

    assert len(calls) == 2


def test_purpose_is_recorded_on_the_call_row(tmp_path, monkeypatch):
    from sqlalchemy import select

    from app.core.db import LLMCall, get_db

    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, purpose="repo_facts", _completion_fn=_fake_completion_fn("hi"))
    # Same prompt from another feature: still one dispatch, and the cached
    # row carries its own caller's purpose rather than the first one's.
    complete("bulk", messages, purpose="resume_build", _completion_fn=_fake_completion_fn("hi"))

    db = get_db()
    try:
        rows = list(db.execute(select(LLMCall).order_by(LLMCall.id)).scalars())
    finally:
        db.close()
    assert [(r.purpose, r.cached) for r in rows] == [
        ("repo_facts", False),
        ("resume_build", True),
    ]


def test_purpose_is_not_part_of_the_cache_key(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, purpose="a", _completion_fn=_fake_completion_fn("x", calls=calls))
    complete("bulk", messages, purpose="b", _completion_fn=_fake_completion_fn("x", calls=calls))

    assert len(calls) == 1


_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _ask_json(messages: list, text: str, calls: list, **kwargs):
    return complete(
        "bulk",
        messages,
        schema=_SCHEMA,
        _completion_fn=_fake_completion_fn(text, calls=calls),
        **kwargs,
    )


def test_unparseable_response_is_not_served_from_cache(tmp_path, monkeypatch):
    """A truncated or empty answer would otherwise be replayed for free on
    every retry and fail the same way each time."""
    from app.core.db import LLMCall, get_db

    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.02, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    first = _ask_json(messages, '{"ok": tr', calls)
    second = _ask_json(messages, '{"ok": true}', calls)

    assert first.parsed is None
    assert second.cached is False
    assert second.parsed == {"ok": True}
    assert len(calls) == 2
    db = get_db()
    try:
        rows = db.query(LLMCall).order_by(LLMCall.id).all()
    finally:
        db.close()
    # The failed call still counts toward spend, just without content.
    assert [r.cost_usd for r in rows] == [0.02, 0.02]
    assert rows[0].response_json is None


def test_unparseable_row_already_in_the_cache_is_not_served(tmp_path, monkeypatch):
    from app.core.app_settings import get_llm_settings
    from app.core.llm import _prompt_hash, _record

    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.0, raising=False
    )
    messages = [{"role": "user", "content": "hello"}]
    model = get_llm_settings().model_for("bulk")
    _record(
        tier="bulk", model=model, prompt_hash=_prompt_hash(model, messages, _SCHEMA, None),
        response_json={"content": ""}, tokens_in=0, tokens_out=0,
        cost_usd=0.0, latency_ms=0, cached=False,
    )
    calls: list = []

    result = _ask_json(messages, '{"ok": true}', calls)

    assert result.cached is False
    assert result.parsed == {"ok": True}
    assert len(calls) == 1


def test_bypass_cache_makes_a_real_call_and_records_it(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    _ask_json(messages, '{"ok": false}', calls)
    retried = _ask_json(messages, '{"ok": true}', calls, bypass_cache=True)
    later = _ask_json(messages, "unused", calls)

    assert retried.cached is False
    assert retried.parsed == {"ok": True}
    assert len(calls) == 2
    # The fresh answer is what a later normal call reuses.
    assert later.cached is True
    assert later.parsed == {"ok": True}


def test_without_a_schema_any_response_is_cached(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    complete("bulk", messages, _completion_fn=_fake_completion_fn("not json", calls=calls))
    second = complete("bulk", messages, _completion_fn=_fake_completion_fn("other", calls=calls))

    assert second.cached is True
    assert second.content == "not json"
    assert len(calls) == 1


def test_cache_version_is_part_of_the_key(monkeypatch):
    from app.core import llm

    messages = [{"role": "user", "content": "hello"}]
    before = llm._prompt_hash("openai/gpt-4o-mini", messages, None, None)
    monkeypatch.setattr(llm, "_CACHE_VERSION", llm._CACHE_VERSION + 1)

    assert llm._prompt_hash("openai/gpt-4o-mini", messages, None, None) != before


def test_cache_is_not_shared_between_accounts(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "litellm.completion_cost", lambda completion_response: 0.01, raising=False
    )
    calls: list = []
    messages = [{"role": "user", "content": "hello"}]

    first = complete(
        "bulk", messages, account_id=1, _completion_fn=_fake_completion_fn("a", calls=calls)
    )
    other = complete(
        "bulk", messages, account_id=2, _completion_fn=_fake_completion_fn("b", calls=calls)
    )
    again = complete(
        "bulk", messages, account_id=1, _completion_fn=_fake_completion_fn("c", calls=calls)
    )

    assert (first.cached, other.cached, again.cached) == (False, False, True)
    assert (other.content, again.content) == ("b", "a")
    assert len(calls) == 2


def test_anthropic_system_prompt_is_marked_for_prompt_caching():
    messages = [system_message("fixed instructions"), user_message("per-item text")]

    marked = _with_prompt_caching("anthropic", messages)

    assert marked[0]["content"] == [
        {
            "type": "text",
            "text": "fixed instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert marked[1] == messages[1]
    # the caller's own list is untouched
    assert messages[0]["content"] == "fixed instructions"


def test_providers_that_cache_by_themselves_get_the_messages_unchanged():
    messages = [system_message("fixed instructions"), user_message("per-item text")]

    for provider in ("gemini", "openai", "mistral", "ollama"):
        assert _with_prompt_caching(provider, messages) == messages


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
    docstring in app/core/db/models.py); it should land on the row either way."""
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
    passed, so complete() calls app.core.api_keys_store.resolve_dispatch_keys()
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


def _add_openai_key(
    budget_cap_usd: float | None = None,
    label: str = "Test key",
    api_key: str = "sk-test",
) -> int:
    from app.core import api_keys_store

    added, detail = api_keys_store.add_key(
        "openai", label, {"api_key": api_key}, budget_cap_usd
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


def _permission_denied(message: str) -> Exception:
    """litellm's 403 exception, which unlike its siblings insists on a real
    response object."""
    import httpx
    import litellm

    return litellm.PermissionDeniedError(
        message=message,
        llm_provider="openai",
        model="gpt-4o-mini",
        response=httpx.Response(
            403, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        ),
    )


def test_a_suspended_key_is_marked_blocked_not_just_invalid(tmp_path, monkeypatch):
    """A provider shutting the key off is kept apart from a rejected key
    and from a quota, because it is the one failure no waiting fixes: it
    lands in its own group on /apis and nothing rechecks it on a timer."""

    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def suspended(**kwargs):
        raise _permission_denied("Your account has been suspended")

    monkeypatch.setattr("litellm.completion", suspended, raising=False)

    with pytest.raises(LLMProviderError, match="has blocked this key"):
        complete("bulk", [user_message("hi")])

    assert _key_status(key_id) == "blocked"


def test_a_model_this_key_may_not_use_leaves_the_key_alone(tmp_path, monkeypatch):
    """The other side of the same error class: the key is healthy, it just
    isn't allowed near that model, so its status must not be touched."""
    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def not_allowed(**kwargs):
        raise _permission_denied("Project does not have access to model gpt-4o-mini")

    monkeypatch.setattr("litellm.completion", not_allowed, raising=False)

    with pytest.raises(LLMProviderError, match="isn't allowed to use"):
        complete("bulk", [user_message("hi")])

    assert _key_status(key_id) == "valid"


def test_a_rate_limited_key_gets_a_wait_planned_from_what_the_provider_said(tmp_path, monkeypatch):
    """The 429 path hands the provider's own text to
    app/core/key_cooldown.py, which is the only place the reset window is
    written down."""
    import litellm

    from app.core import api_keys_store

    _real_dispatch_env(tmp_path, monkeypatch)
    key_id = _add_openai_key()

    def out_of_quota(**kwargs):
        raise litellm.RateLimitError(
            message='quota exceeded, "retryDelay":"300s"',
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    monkeypatch.setattr("litellm.completion", out_of_quota, raising=False)

    with pytest.raises(LLMRateLimitedError):
        complete("bulk", [user_message("hi")])

    row = next(k for k in api_keys_store.list_keys() if k["id"] == key_id)
    assert row["status"] == "rate_limited"
    assert row["retry_at"] is not None
    assert row["recheck_due"] is False


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


# ---------------------------------------------------------------------------
# Key failover: app/core/llm.py's _dispatch_over_keys()
#
# What these pin down is the point of the whole mechanism: a key dying is
# not the end of the request, and so not the end of whatever multi-step
# job the request is one call of. The caller never learns a swap happened
# (it gets a normal LLMResponse), the key that gave out is marked, and the
# key that answered is the one billed.
# ---------------------------------------------------------------------------


def _keyed_completion_fn(behaviour: dict, calls: list | None = None):
    """A fake litellm.completion that answers differently per credential.
    `behaviour` maps api_key -> either an exception to raise or the text
    to answer with, so a test says "sk-1 is out of quota, sk-2 works"
    without caring how complete() got there."""

    def _fn(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        outcome = behaviour[kwargs["api_key"]]
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(choices=[FakeChoice(FakeMessage(outcome))])

    return _fn


def _rate_limited() -> Exception:
    import litellm

    return litellm.RateLimitError(
        message="quota exceeded", llm_provider="openai", model="gpt-4o-mini"
    )


def _rejected() -> Exception:
    import litellm

    return litellm.AuthenticationError(
        message="invalid api key", llm_provider="openai", model="gpt-4o-mini"
    )


def test_a_dead_key_hands_the_same_request_to_the_next_one(tmp_path, monkeypatch):
    """The heart of it: key one is out of quota, key two answers, and the
    caller sees an ordinary successful response. A job halfway through its
    steps keeps going instead of stopping with two steps done."""
    from sqlalchemy import select

    from app.core.db import LLMCall, get_db

    _real_dispatch_env(tmp_path, monkeypatch)
    first = _add_openai_key(label="First", api_key="sk-1")
    second = _add_openai_key(label="Second", api_key="sk-2")
    calls: list = []
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": _rate_limited(), "sk-2": "answered"}, calls=calls),
        raising=False,
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.02)

    result = complete("bulk", [user_message("hi")])

    assert result.content == "answered"
    assert [c["api_key"] for c in calls] == ["sk-1", "sk-2"]
    assert _key_status(first) == "rate_limited"
    assert _key_status(second) == "valid"

    db = get_db()
    row = db.execute(select(LLMCall).order_by(LLMCall.id.desc())).scalars().first()
    db.close()
    assert row.key_id == second  # the key that actually paid, not the first one tried


def test_a_rejected_key_also_hands_over(tmp_path, monkeypatch):
    """Not just quotas: a credential the provider refuses outright is the
    other everyday way a key dies mid-run."""
    _real_dispatch_env(tmp_path, monkeypatch)
    first = _add_openai_key(label="First", api_key="sk-1")
    _add_openai_key(label="Second", api_key="sk-2")
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": _rejected(), "sk-2": "answered"}),
        raising=False,
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    assert complete("bulk", [user_message("hi")]).content == "answered"
    assert _key_status(first) == "invalid"


def test_a_key_past_its_own_cap_is_skipped_without_spending_a_call(tmp_path, monkeypatch):
    """A cap is checked before dispatch, so the capped key costs nothing
    to skip, and the next key takes the request."""
    from app.core.llm import _record

    _real_dispatch_env(tmp_path, monkeypatch)
    capped = _add_openai_key(budget_cap_usd=0.05, label="Capped", api_key="sk-1")
    _add_openai_key(label="Spare", api_key="sk-2")
    _record(
        tier="bulk", model="openai/gpt-4o-mini", prompt_hash="earlier-call",
        response_json={"content": "x"}, tokens_in=0, tokens_out=0,
        cost_usd=0.50, latency_ms=0, cached=False, key_id=capped,
    )
    calls: list = []
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-2": "answered"}, calls=calls),
        raising=False,
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    assert complete("bulk", [user_message("hi")]).content == "answered"
    assert [c["api_key"] for c in calls] == ["sk-2"]


def test_one_key_s_spending_does_not_count_against_another_s_cap(tmp_path, monkeypatch):
    """Caps are per key, read off LLMCall.key_id. A provider-wide total
    would retire every key on that provider as soon as the smallest cap
    was reached, which is exactly the failure this whole feature exists
    to avoid."""
    from app.core.llm import _key_month_spend_usd, _record

    _real_dispatch_env(tmp_path, monkeypatch)
    other = _add_openai_key(label="Other", api_key="sk-1")
    mine = _add_openai_key(budget_cap_usd=1.00, label="Mine", api_key="sk-2")
    _record(
        tier="bulk", model="openai/gpt-4o-mini", prompt_hash="other-key-call",
        response_json={"content": "x"}, tokens_in=0, tokens_out=0,
        cost_usd=5.00, latency_ms=0, cached=False, key_id=other,
    )
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": _rate_limited(), "sk-2": "answered"}),
        raising=False,
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    assert complete("bulk", [user_message("hi")]).content == "answered"
    assert _key_month_spend_usd(mine) == 0.0


def test_an_overloaded_provider_does_not_burn_the_other_keys(tmp_path, monkeypatch):
    """A 503 is the provider's own health, identical for every key. Trying
    them all would spend the spare keys' quota on a request that cannot
    succeed and delay the real error by however long the retries take."""
    import litellm

    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="First", api_key="sk-1")
    _add_openai_key(label="Second", api_key="sk-2")
    monkeypatch.setattr("app.core.llm._sleep", lambda s: None)
    calls: list = []
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn(
            {
                "sk-1": litellm.ServiceUnavailableError(
                    message="overloaded", llm_provider="openai", model="gpt-4o-mini"
                ),
                "sk-2": "never reached",
            },
            calls=calls,
        ),
        raising=False,
    )

    with pytest.raises(LLMUnavailableError):
        complete("bulk", [user_message("hi")])

    assert {c["api_key"] for c in calls} == {"sk-1"}


def test_a_prompt_the_model_cannot_take_fails_once_not_once_per_key(tmp_path, monkeypatch):
    """Same reasoning for a request-shaped error: no key makes an
    over-long prompt fit."""
    import litellm

    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="First", api_key="sk-1")
    _add_openai_key(label="Second", api_key="sk-2")
    calls: list = []
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn(
            {
                "sk-1": litellm.ContextWindowExceededError(
                    message="too long", llm_provider="openai", model="gpt-4o-mini"
                ),
                "sk-2": "never reached",
            },
            calls=calls,
        ),
        raising=False,
    )

    with pytest.raises(LLMProviderError, match="too long"):
        complete("bulk", [user_message("hi")])

    assert len(calls) == 1


def test_only_once_every_key_is_spent_does_the_request_fail(tmp_path, monkeypatch):
    """And the error then names each key and its own reason, because "why
    did this stop?" is answered by the whole list, not by whichever key
    happened to be tried last."""
    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="Personal", api_key="sk-1")
    _add_openai_key(label="Work", api_key="sk-2")
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": _rate_limited(), "sk-2": _rejected()}),
        raising=False,
    )

    with pytest.raises(LLMProviderError) as excinfo:
        complete("bulk", [user_message("hi")])

    message = str(excinfo.value)
    assert "All 2 OpenAI keys failed" in message
    assert "Personal: " in message and "limiting requests" in message
    assert "Work: " in message and "rejected the API key" in message


def test_a_single_key_keeps_its_own_message_verbatim(tmp_path, monkeypatch):
    """Nothing to fall back to means there is no failover to explain, so
    the message stays the one that says what to do about that key."""
    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="Only", api_key="sk-1")
    monkeypatch.setattr(
        "litellm.completion", _keyed_completion_fn({"sk-1": _rejected()}), raising=False
    )

    with pytest.raises(LLMProviderError, match="^OpenAI rejected the API key"):
        complete("bulk", [user_message("hi")])


def test_a_key_marked_dead_earlier_is_tried_last_and_revived_if_it_works(tmp_path, monkeypatch):
    """Quotas reset. A key marked rate_limited an hour ago is worth
    keeping in rotation, just not first, and a run where it answers
    clears the mark so /apis stops showing a dead key that works."""
    from app.core import api_keys_store

    _real_dispatch_env(tmp_path, monkeypatch)
    stale = _add_openai_key(label="Yesterday's", api_key="sk-1")
    healthy = _add_openai_key(label="Healthy", api_key="sk-2")
    api_keys_store.record_dispatch_outcome(stale, ok=False, rate_limited=True, detail="429")
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    # Healthy key first, although the stale one is the active one.
    calls: list = []
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": "stale", "sk-2": "healthy"}, calls=calls),
        raising=False,
    )
    assert complete("bulk", [user_message("hi")]).content == "healthy"
    assert [c["api_key"] for c in calls] == ["sk-2"]

    # And when the healthy one dies, the stale one answers and is cleared.
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": "stale", "sk-2": _rate_limited()}),
        raising=False,
    )
    assert complete("bulk", [user_message("second prompt")]).content == "stale"
    assert _key_status(stale) == "valid"
    assert _key_status(healthy) == "rate_limited"


def test_a_key_swap_is_written_to_the_event_log(tmp_path, monkeypatch):
    """The request succeeded, so nothing else in the app would record
    that a key gave out. /monitor's event log is the trail."""
    from app.core.rate_limits import list_events

    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="Personal", api_key="sk-1")
    _add_openai_key(label="Work", api_key="sk-2")
    monkeypatch.setattr(
        "litellm.completion",
        _keyed_completion_fn({"sk-1": _rate_limited(), "sk-2": "answered"}),
        raising=False,
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    complete("bulk", [user_message("hi")], account_id=4)

    failovers = [e for e in list_events(source="llm") if e.kind == "key_failover"]
    assert len(failovers) == 1
    assert failovers[0].detail.startswith("Personal: ")
    assert failovers[0].account_id == 4


def test_is_out_of_keys_tells_batches_when_to_stop(tmp_path, monkeypatch):
    """The signal a batch caller (app/profile/build.py) uses to stop
    instead of spending one doomed call per remaining item."""
    from app.core.llm import LLMDispatchError, is_out_of_keys

    _real_dispatch_env(tmp_path, monkeypatch)
    _add_openai_key(label="Only", api_key="sk-1")
    monkeypatch.setattr(
        "litellm.completion", _keyed_completion_fn({"sk-1": _rate_limited()}), raising=False
    )

    with pytest.raises(LLMRateLimitedError) as excinfo:
        complete("bulk", [user_message("hi")])

    assert is_out_of_keys(excinfo.value)
    assert is_out_of_keys(ApiKeyMissingError("none stored"))
    assert is_out_of_keys(BudgetExceededError("cap reached"))
    # This one item's own problem, and the next item might be fine.
    assert not is_out_of_keys(LLMProviderError("prompt too long"))
    assert not is_out_of_keys(LLMUnavailableError("provider down"))
    assert not is_out_of_keys(LLMDispatchError("something else"))
    assert not is_out_of_keys(ValueError("not an LLM problem at all"))
