"""LangSmith tracing, off unless TRACING_ENABLED is set.

When it is on, resume content leaves this machine for LangSmith, so one
explicit setting decides, and a LANGSMITH_TRACING variable left in the
environment cannot turn it on alone. LangGraph reports its own nodes;
complete() is traced as an LLM run so model, tokens and cost show up too.
Attachments are replaced by a marker: a compiled PDF is large, and a trace
needs to say that one was sent, not carry it.
"""

from __future__ import annotations

import os
from typing import Any

from langsmith import utils as langsmith_utils

from app.core.settings import get_settings


def configure_tracing() -> bool:
    settings = get_settings()
    enabled = settings.tracing_enabled and bool(os.environ.get("LANGSMITH_API_KEY"))
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ[name] = "true" if enabled else "false"
    if enabled:
        os.environ.setdefault("LANGSMITH_PROJECT", "open-to-work")
    langsmith_utils.get_env_var.cache_clear()  # type: ignore[attr-defined]
    return enabled


def _without_attachments(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {
        **message,
        "content": [
            part if part.get("type") == "text" else {"type": part.get("type"), "omitted": True}
            for part in content
        ],
    }


def trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "tier": inputs.get("tier"),
        "purpose": inputs.get("purpose"),
        "account_id": inputs.get("account_id"),
        "messages": [_without_attachments(m) for m in inputs.get("messages") or []],
    }


def trace_outputs(response: Any) -> dict[str, Any]:
    """LangSmith reads token counts from usage_metadata. A call that raised
    has no response; the error is recorded on the run instead."""
    if response is None:
        return {}
    return {
        "content": response.content,
        "model": response.model,
        "cached": response.cached,
        "cost_usd": response.cost_usd,
        "usage_metadata": {
            "input_tokens": response.tokens_in,
            "output_tokens": response.tokens_out,
            "total_tokens": response.tokens_in + response.tokens_out,
        },
    }
