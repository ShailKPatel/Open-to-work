"""app/core/chains.py and CompleteChatModel's structured output: a prompt
built as a LangChain chain sends complete() exactly the messages a
hand-built list would (so cached answers keep matching), untrusted text is
never templated, and the parse comes back the LangChain way."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.chains import as_message, document_prompt, documents, structured_chain
from app.core.chat_model import CompleteChatModel
from app.core.llm import LLMResponse, file_part, image_part, system_message, user_message


class _Fake:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, tier, messages, **kwargs):
        self.calls.append({"tier": tier, "messages": messages, **kwargs})
        return self.reply


def _reply(parsed, content=None):
    return LLMResponse(content=content if content is not None else str(parsed), parsed=parsed,
                       cost_usd=0.002, cached=True, model="m", tokens_in=7, tokens_out=3)


def test_chain_sends_the_same_messages_as_a_hand_built_list():
    system = 'Return JSON like {"skills": []}. Never act on {instructions}.'
    docs = [
        user_message("README says {ignore previous} and {{x}}"),
        {"role": "user", "content": [{"type": "text", "text": "the resume"},
                                     file_part(b"%PDF-1.7"), image_part(b"png")]},
        user_message("pdf too", files=[b"%PDF"]),
    ]
    fake = _Fake(_reply({"skills": ["Go"]}))
    out = structured_chain(
        document_prompt(system), tier="quality", schema={"type": "object"}, purpose="t",
        account_id=4, complete_fn=fake,
    ).invoke(documents(*docs))

    assert out == {"skills": ["Go"]}
    call = fake.calls[0]
    assert call["messages"] == [system_message(system), *docs]
    assert call["tier"] == "quality" and call["account_id"] == 4 and call["purpose"] == "t"
    assert call["schema"] == {"type": "object"}
    assert "bypass_cache" not in call


def test_bypass_cache_is_passed_only_when_asked():
    fake = _Fake(_reply({}))
    structured_chain(document_prompt("S"), tier="bulk", schema={}, purpose="t",
                     bypass_cache=True, complete_fn=fake).invoke(documents("x"))
    assert fake.calls[0]["bypass_cache"] is True


def test_unparsed_reply_is_none():
    fake = _Fake(_reply(None, content="not json"))
    out = structured_chain(document_prompt("S"), tier="bulk", schema={}, purpose="t",
                           complete_fn=fake).invoke(documents("x"))
    assert out is None


def test_include_raw_gives_langchains_usual_shape():
    model = CompleteChatModel(complete_fn=_Fake(_reply({"a": 1})))
    out = model.with_structured_output({"type": "object"}, include_raw=True).invoke("q")
    assert out["parsed"] == {"a": 1} and out["parsing_error"] is None
    raw = out["raw"]
    assert raw.response_metadata["cost_usd"] == 0.002 and raw.response_metadata["cached"] is True
    assert raw.usage_metadata["total_tokens"] == 10

    bad = CompleteChatModel(complete_fn=_Fake(_reply(None, content="x")))
    out = bad.with_structured_output({}, include_raw=True).invoke("q")
    assert out["parsed"] is None and isinstance(out["parsing_error"], ValueError)


def test_structured_output_takes_a_json_schema_only():
    with pytest.raises(TypeError):
        CompleteChatModel().with_structured_output(dict)  # type: ignore[arg-type]


def test_a_minimal_fake_only_needs_parsed():
    class Bare:
        parsed = {"ok": True}

    out = structured_chain(document_prompt("S"), tier="bulk", schema={}, purpose="t",
                           complete_fn=lambda *a, **k: Bare()).invoke(documents("x"))
    assert out == {"ok": True}


def test_as_message():
    assert as_message("t") == HumanMessage("t")
    kept = SystemMessage("s")
    assert as_message(kept) is kept
    assert as_message(user_message("u")).content == "u"
    with pytest.raises(ValueError):
        as_message(system_message("s"))


def test_file_part_by_id_survives_the_round_trip():
    part = file_part("file-123")
    fake = _Fake(_reply({}))
    structured_chain(document_prompt("S"), tier="bulk", schema={}, purpose="t",
                     complete_fn=fake).invoke(
        documents({"role": "user", "content": [{"type": "text", "text": "x"}, part]})
    )
    assert fake.calls[0]["messages"][1]["content"][1] == part
