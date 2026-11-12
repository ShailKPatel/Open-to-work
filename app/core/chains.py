"""Every prompt in the app as a LangChain runnable: a ChatPromptTemplate
piped into CompleteChatModel (app/core/chat_model.py), so the call itself
still goes through complete() with its cache, budget, key failover and
cost ledger.

    chain = structured_chain(PROMPT, tier="bulk", schema=SCHEMA, purpose="repo_facts")
    parsed = chain.invoke({"documents": [user_message(text)]})

The prompt shape is fixed on purpose. Fixed instructions are the system
message. Everything that changes per call, most of it untrusted (job
postings, READMEs, resumes, model output being revised), goes in as
ready-made user messages through a MessagesPlaceholder. No such text is
ever a template string, so a brace or a "{var}" inside a README or a
posting is never interpolated, and none of it reaches the system prompt.

Tracing: with LangSmith on (app/core/tracing.py), each invoke is one run
named after its purpose, with the prompt, the model turn and the parse as
children.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from app.core.app_settings import Tier
from app.core.chat_model import CompleteChatModel
from app.core.llm import LLMResponse


def document_prompt(system: str) -> ChatPromptTemplate:
    """Fixed instructions, then the call's documents as user messages.
    The system text is a SystemMessage object, not a template, so braces
    in it (JSON examples) are sent exactly as written."""
    return ChatPromptTemplate.from_messages(
        [SystemMessage(system), MessagesPlaceholder("documents")]
    )


def as_message(message: dict | str | BaseMessage) -> BaseMessage:
    """A user message as complete() would take it (app.core.llm's
    user_message() dict, multimodal parts included), or plain text, as a
    HumanMessage with the same content."""
    if isinstance(message, BaseMessage):
        return message
    if isinstance(message, str):
        return HumanMessage(message)
    if message.get("role") != "user":
        raise ValueError("documents are user messages")
    return HumanMessage(content=message["content"])


def documents(*messages: dict | str | BaseMessage) -> dict[str, Any]:
    """The input a document_prompt() takes."""
    return {"documents": [as_message(m) for m in messages]}


def structured_chain(
    prompt: ChatPromptTemplate,
    *,
    tier: Tier,
    schema: dict[str, Any],
    purpose: str,
    account_id: int | None = None,
    bypass_cache: bool = False,
    complete_fn: Callable[..., LLMResponse] | None = None,
) -> Runnable[dict[str, Any], dict[str, Any] | None]:
    """prompt | model.with_structured_output(schema): its output is the
    reply parsed against `schema`, or None when it did not parse. LLM
    errors from complete() (budget, missing key, provider down) propagate
    unchanged, so each caller keeps its own failure handling."""
    model = CompleteChatModel(
        tier=tier,
        purpose=purpose,
        account_id=account_id,
        bypass_cache=bypass_cache,
        complete_fn=complete_fn,
    )
    chain: Runnable[dict[str, Any], dict[str, Any] | None] = (
        prompt | model.with_structured_output(schema)
    ).with_config(run_name=purpose)
    return chain
