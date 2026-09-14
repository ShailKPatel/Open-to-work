"""Every LLM call in the app goes through here. No direct provider SDK calls
anywhere else.

One entrypoint, not one method per modality: `complete(tier, messages, schema)`.
`messages` is the standard chat-message format used by LiteLLM: a list of
{"role", "content"} dicts where `content` is either a plain string (text-only)
or a list of typed parts (text + image_url, for multimodal). Callers never
hand-build that JSON: `user_message(text, images=...)` constructs one, with or
without attachments, and the same `complete()` call handles both. No separate
multimodal client, no per-modality wrapper to keep in sync.

Content-hash cache: identical (tier, model, messages, schema) never bills
twice. Budget cap: enforced before dispatch, not after. Every call, cached or
not, gets a row in LLMCall so the dashboard shows real call volume alongside
real spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import func, select

from app.core.db import LLMCall, get_db
from app.core.llm_providers import (
    PROVIDER_LABELS,
    litellm_kwargs,
    model_prefix_for_provider,
    provider_of_model,
)
from app.core.rate_limits import record_event
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

Tier = Literal["bulk", "quality"]


class BudgetExceededError(RuntimeError):
    """Raised before dispatch when the monthly cap would be breached."""


class ApiKeyMissingError(RuntimeError):
    """Raised right before a real (non-injected) LLM dispatch when no
    active key is stored for the tier's provider yet (see
    app/core/api_keys_store.py and the /apis page). Not raised any earlier
    in complete(): a cache hit or a test's injected _completion_fn never
    needs a key."""


class LLMRateLimitedError(RuntimeError):
    """The provider itself rate-limited us (HTTP 429 / quota exceeded).
    Wraps litellm's own exception so callers never need to import litellm
    to tell "try again later" apart from "this input failed." Distinct
    from BudgetExceededError: that one is our own cap, checked before
    dispatch; this is the provider's, hit after dispatch. Both mean the
    same thing to a caller running a batch (stop, don't keep spending calls
    that will fail the same way), which is why app/profile/build.py catches
    both together.
    """


@dataclass
class LLMResponse:
    content: str
    parsed: dict | None
    cost_usd: float
    cached: bool
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


def _model_for_tier(tier: Tier) -> str:
    settings = get_settings()
    return settings.llm_bulk_model if tier == "bulk" else settings.llm_quality_model


def _prompt_hash(tier: str, model: str, messages: list[dict], schema: dict | None) -> str:
    payload = json.dumps(
        {"tier": tier, "model": model, "messages": messages, "schema": schema},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _month_spend_usd() -> float:
    import datetime as dt

    db = get_db()
    try:
        month_start = dt.datetime.now(dt.UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        total = db.execute(
            select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                LLMCall.created_at >= month_start
            )
        ).scalar_one()
        return float(total)
    finally:
        db.close()


def month_spend_usd() -> float:
    """Public twin of _month_spend_usd(), for callers outside this module
    that just want to display the number (app/api/monitor.py) rather than
    check it against the budget cap before dispatch."""
    return _month_spend_usd()


def _provider_month_spend_usd(provider: str) -> float:
    """Spend attributed to one provider this month, for that provider's
    optional per-key budget_cap_usd (app/core/db.py's ApiKey). Derived
    from LLMCall.model's own "<provider>/..." prefix rather than a
    separate column. Accurate as long as at most one key per provider is
    active at a time, which app/core/api_keys_store.py enforces. If the
    active key for a provider is swapped mid-month, this reads as that
    provider's total spend across whichever keys were active. It is an
    approximation, not a per-key ledger.
    """
    import datetime as dt

    db = get_db()
    try:
        month_start = dt.datetime.now(dt.UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        total = db.execute(
            select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                LLMCall.created_at >= month_start,
                LLMCall.model.like(f"{model_prefix_for_provider(provider)}/%"),
            )
        ).scalar_one()
        return float(total)
    finally:
        db.close()


def _lookup_cache(prompt_hash: str) -> LLMCall | None:
    db = get_db()
    try:
        return db.execute(
            select(LLMCall)
            .where(LLMCall.prompt_hash == prompt_hash, LLMCall.response_json.is_not(None))
            .order_by(LLMCall.id.desc())
        ).scalars().first()
    finally:
        db.close()


def _record(
    *,
    tier: str,
    model: str,
    prompt_hash: str,
    response_json: dict,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: int,
    cached: bool,
    account_id: int | None = None,
    key_id: int | None = None,
) -> None:
    db = get_db()
    try:
        db.add(
            LLMCall(
                tier=tier,
                model=model,
                prompt_hash=prompt_hash,
                response_json=response_json,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                cached=cached,
                account_id=account_id,
                key_id=key_id,
            )
        )
        db.commit()
    finally:
        db.close()


def complete(
    tier: Tier,
    messages: list[dict],
    schema: dict | None = None,
    account_id: int | None = None,
    _completion_fn: Any = None,
) -> LLMResponse:
    """_completion_fn is an injection point for tests; production callers
    never pass it; it defaults to litellm.completion.

    account_id is optional context for app/core/api_keys_store.py's
    per-key allow-list (a key can be restricted to specific profiles on
    a shared device): pass it whenever the caller already knows which
    account this call is for (most do, e.g. app/profile/extract.py and
    app/resume_build/orchestrator.py). None means "no account context",
    which only unrestricted keys can serve.
    """
    settings = get_settings()
    model = _model_for_tier(tier)
    prompt_hash = _prompt_hash(tier, model, messages, schema)

    cached_row = _lookup_cache(prompt_hash)
    if cached_row is not None:
        # _lookup_cache only returns rows where response_json IS NOT NULL,
        # but the column itself is nullable; narrow it for mypy.
        cached_response = cached_row.response_json
        assert cached_response is not None
        _record(
            tier=tier,
            model=model,
            prompt_hash=prompt_hash,
            response_json=cached_response,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_ms=0,
            cached=True,
            account_id=account_id,
        )
        content = cached_response.get("content", "")
        parsed = _try_parse(content) if schema else None
        return LLMResponse(content=content, parsed=parsed, cost_usd=0.0, cached=True, model=model)

    spent = _month_spend_usd()
    if spent >= settings.monthly_budget_usd:
        detail = (
            f"monthly budget ${settings.monthly_budget_usd:.2f} reached "
            f"(spent ${spent:.2f}); call refused before dispatch"
        )
        record_event("llm", "budget_exceeded", detail, context=model, account_id=account_id)
        raise BudgetExceededError(detail)

    provider = provider_of_model(model)
    key_id: int | None = None
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if _completion_fn is None:
        from app.core.api_keys_store import resolve_dispatch_key

        resolved = resolve_dispatch_key(provider, account_id)
        if resolved is None:
            raise ApiKeyMissingError(
                f"no {PROVIDER_LABELS.get(provider, provider)} API key available "
                f"{'for this account ' if account_id is not None else ''}"
                "(add or enable one from Manage APIs, /apis)"
            )
        key_id, credentials, key_budget = resolved

        if key_budget is not None:
            provider_spent = _provider_month_spend_usd(provider)
            if provider_spent >= key_budget:
                detail = (
                    f"{PROVIDER_LABELS.get(provider, provider)} key budget ${key_budget:.2f} "
                    f"reached (spent ${provider_spent:.2f}); call refused before dispatch"
                )
                record_event("llm", "budget_exceeded", detail, context=model, account_id=account_id)
                raise BudgetExceededError(detail)

        kwargs.update(litellm_kwargs(provider, credentials))

        import litellm

        _completion_fn = litellm.completion

    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }

    start = time.monotonic()
    try:
        response = _completion_fn(**kwargs)
    except Exception as e:
        import litellm

        # Only these two exception types say anything about the KEY
        # itself; anything else (network blip, malformed request, ...)
        # leaves its stored status untouched rather than guessing.
        if key_id is not None:
            from app.core.api_keys_store import record_dispatch_outcome

            if isinstance(e, litellm.AuthenticationError):
                record_dispatch_outcome(key_id, ok=False, detail=str(e))
            elif isinstance(e, litellm.RateLimitError):
                record_dispatch_outcome(key_id, ok=False, rate_limited=True, detail=str(e))
        if isinstance(e, litellm.RateLimitError):
            record_event("llm", "rate_limited", str(e), context=model, account_id=account_id)
            raise LLMRateLimitedError(str(e)) from e
        raise
    latency_ms = int((time.monotonic() - start) * 1000)

    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(usage, "completion_tokens", 0) or 0

    cost_usd = _safe_completion_cost(response)

    _record(
        tier=tier,
        model=model,
        prompt_hash=prompt_hash,
        response_json={"content": content},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=False,
        account_id=account_id,
        key_id=key_id,
    )

    parsed = _try_parse(content) if schema else None
    return LLMResponse(
        content=content,
        parsed=parsed,
        cost_usd=cost_usd,
        cached=False,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _safe_completion_cost(response: Any) -> float:
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        logger.warning("could not compute cost for response; recording $0.00")
        return 0.0


def _try_parse(content: str) -> dict | None:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def embed(texts: list[str]) -> list[list[float]]:
    from app.core.embeddings import embed as _embed

    return _embed(texts)


ImageInput = str | bytes
"""Either a URL/data-URI string (passed through as-is) or raw image bytes
(base64-encoded into a data URI here)."""


def image_part(image: ImageInput, mime_type: str = "image/png") -> dict:
    """One multimodal content part. Bytes get base64-encoded into a data
    URI; a string is assumed to already be a URL or data URI and passed
    through untouched.
    """
    if isinstance(image, bytes):
        import base64

        url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
    else:
        url = image
    return {"type": "image_url", "image_url": {"url": url}}


def file_part(file: ImageInput, mime_type: str = "application/pdf") -> dict:
    """One document content part (PDF, etc) in LiteLLM's `type: "file"` shape
    (https://docs.litellm.ai/docs/completion/document_understanding). Bytes
    get base64-encoded into `file_data`; a string is assumed to be a URL and
    passed as `file_id`.
    """
    if isinstance(file, bytes):
        import base64

        encoded = base64.b64encode(file).decode("ascii")
        return {"type": "file", "file": {"file_data": f"data:{mime_type};base64,{encoded}"}}
    return {"type": "file", "file": {"file_id": file}}


def user_message(
    text: str,
    images: list[ImageInput] | None = None,
    files: list[ImageInput] | None = None,
) -> dict:
    """Build one user-role message for `complete()`. No attachments: plain-
    string content (works with every model). With images and/or files: a
    multimodal content list (text part + one part per attachment). Same
    `complete()` call either way.
    """
    if not images and not files:
        return {"role": "user", "content": text}
    parts: list[dict] = [{"type": "text", "text": text}]
    parts.extend(image_part(img) for img in images or [])
    parts.extend(file_part(f) for f in files or [])
    return {"role": "user", "content": parts}


def system_message(text: str) -> dict:
    return {"role": "system", "content": text}
