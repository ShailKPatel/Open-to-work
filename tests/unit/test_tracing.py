"""app/core/tracing.py: off by default, and what a trace would contain."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langsmith import Client
from langsmith.run_helpers import traceable, tracing_context

from app.core.llm import file_part
from app.core.settings import get_settings
from app.core.tracing import configure_tracing, trace_inputs, trace_outputs


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(name, "unset")
    yield
    get_settings.cache_clear()


def test_stray_env_var_cannot_turn_tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("TRACING_ENABLED", "false")
    get_settings.cache_clear()
    assert configure_tracing() is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_setting_without_a_key_stays_off(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("TRACING_ENABLED", "true")
    get_settings.cache_clear()
    assert configure_tracing() is False


def test_setting_and_key_turn_it_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    monkeypatch.setenv("TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "p")
    get_settings.cache_clear()
    assert configure_tracing() is True
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_attachments_never_reach_a_trace():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "t"}, file_part(b"%PDF-secret")]},
    ]
    traced = trace_inputs({"tier": "bulk", "messages": messages, "_completion_fn": object()})
    assert "file_data" not in str(traced)
    assert traced["messages"][1]["content"][1] == {"type": "file", "omitted": True}
    assert "_completion_fn" not in traced


def test_a_traced_call_reports_tokens_and_passes_the_response_through():
    client = MagicMock(spec=Client)
    client.tracing_sample_rate = None
    response = SimpleNamespace(
        content="c", model="m", cached=False, cost_usd=0.01, tokens_in=3, tokens_out=4
    )

    @traceable(run_type="llm", process_inputs=trace_inputs, process_outputs=trace_outputs)
    def call(tier, messages):
        return response

    with tracing_context(enabled=True, client=client):
        assert call("bulk", [{"role": "user", "content": "x"}]) is response

    assert client.create_run.called or client.update_run.called
    sent = str(client.create_run.call_args) + str(client.update_run.call_args)
    assert "total_tokens" in sent and "'cost_usd': 0.01" in sent


def test_a_failed_call_has_no_outputs_to_record():
    assert trace_outputs(None) == {}
