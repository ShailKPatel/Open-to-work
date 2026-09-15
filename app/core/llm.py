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
import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from app.core.app_settings import Tier, get_llm_settings
from app.core.db import LLMCall, get_db
from app.core.llm_providers import (
    PROVIDER_LABELS,
    litellm_kwargs,
    model_prefix_for_provider,
    provider_of_model,
)
from app.core.rate_limits import record_event

logger = logging.getLogger(__name__)


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


class LLMUnavailableError(LLMRateLimitedError):
    """The provider is overloaded or unreachable (HTTP 500/502/503/504, a
    timeout, a dropped connection) even after complete() retried and tried
    the other tier's model. A subclass of LLMRateLimitedError because it
    means the same thing to every caller: stop, and try again later."""


class LLMProviderError(RuntimeError):
    """The provider refused the request for a reason retrying won't fix: a
    rejected key, an unknown model name, or input it can't handle."""


# Every exception message raised here is shown to the person using the app
# as-is (saved as an extraction error, or returned as an HTTP detail), so it
# says what happened and what to do rather than the provider's raw payload.
# The raw error still goes to the log.

# Waits before each retry when a provider is overloaded, first for the
# tier's own model, then for the fallback model. Bounded so a request never
# hangs for long.
_RETRY_DELAYS_S = (3.0, 8.0)
_FALLBACK_RETRY_DELAYS_S = (5.0,)
_TRANSIENT_STATUS = {500, 502, 503, 504}
_sleep = time.sleep


@dataclass
class LLMResponse:
    content: str
    parsed: dict | None
    cost_usd: float
    cached: bool
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


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
    settings = get_llm_settings()
    model = settings.model_for(tier)
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
            f"The monthly budget of ${settings.monthly_budget_usd:.2f} for AI calls is used up "
            f"(${spent:.2f} spent this month), so this request was not sent. "
            "Raise the budget on Manage APIs, or wait until next month."
        )
        record_event("llm", "budget_exceeded", detail, context=model, account_id=account_id)
        raise BudgetExceededError(detail)

    provider = provider_of_model(model)
    label = PROVIDER_LABELS.get(provider, provider)
    key_id: int | None = None
    kwargs: dict[str, Any] = {"model": model, "messages": messages}

    if _completion_fn is None:
        from app.core.api_keys_store import resolve_dispatch_key

        resolved = resolve_dispatch_key(provider, account_id)
        if resolved is None:
            raise ApiKeyMissingError(
                f"No {label} API key is available"
                f"{' for this account' if account_id is not None else ''}. "
                "Add one, or turn an existing one on, from Manage APIs."
            )
        key_id, credentials, key_budget = resolved

        if key_budget is not None:
            provider_spent = _provider_month_spend_usd(provider)
            if provider_spent >= key_budget:
                detail = (
                    f"The {label} key budget of ${key_budget:.2f} is used up "
                    f"(${provider_spent:.2f} spent this month), so this request was not sent. "
                    "Raise or remove the key's cap on Manage APIs."
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

    # A model that stays overloaded through its retries hands over to the
    # other tier's model when that one runs on the same provider, and so
    # can use the same key. The call is then recorded under the model that
    # actually answered.
    candidates = [model]
    other_model = settings.model_for("quality" if tier == "bulk" else "bulk")
    if other_model != model and provider_of_model(other_model) == provider:
        candidates.append(other_model)

    for i, candidate in enumerate(candidates):
        kwargs["model"] = candidate
        delays = _RETRY_DELAYS_S if i == 0 else _FALLBACK_RETRY_DELAYS_S
        try:
            response, latency_ms = _call_with_retries(_completion_fn, kwargs, delays)
        except Exception as e:
            if _is_transient(e) and i + 1 < len(candidates):
                logger.warning(
                    "%s still unavailable after retries (%s); falling back to %s",
                    candidate, type(e).__name__, candidates[i + 1],
                )
                continue
            readable = _readable_provider_error(e, candidate, label, key_id, account_id)
            if readable is None:
                raise
            raise readable from e
        if candidate != model:
            model = candidate
            prompt_hash = _prompt_hash(tier, model, messages, schema)
        break

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


def _call_with_retries(
    completion_fn: Any, kwargs: dict[str, Any], delays: tuple[float, ...]
) -> tuple[Any, int]:
    """One model: the call, plus a retry after each of `delays` while the
    provider is overloaded or unreachable. Returns (response, latency_ms).
    Any other error, or the last overloaded one, is raised unchanged."""
    for attempt in range(len(delays) + 1):
        start = time.monotonic()
        try:
            response = completion_fn(**kwargs)
        except Exception as e:
            if attempt == len(delays) or not _is_transient(e):
                raise
            logger.warning(
                "%s unavailable (%s); retrying in %.0fs",
                kwargs["model"], type(e).__name__, delays[attempt],
            )
            _sleep(delays[attempt])
            continue
        return response, int((time.monotonic() - start) * 1000)
    raise AssertionError("unreachable")


def _is_transient(e: Exception) -> bool:
    import litellm

    if isinstance(
        e,
        (
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.ServiceUnavailableError,
            litellm.InternalServerError,
            litellm.BadGatewayError,
        ),
    ):
        return True
    return isinstance(e, litellm.APIError) and getattr(e, "status_code", None) in _TRANSIENT_STATUS


def _provider_detail(e: Exception) -> str:
    """The provider's own one-line explanation, pulled out of litellm's
    'litellm.X: ProviderException - {json}' message when there is one."""
    text = str(e)
    match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match:
        text = match.group(1).replace('\\"', '"').replace("\\n", " ")
    else:
        text = re.sub(r"^(litellm\.\w+:\s*)+", "", text)
        text = re.sub(r"^\w+Exception - ", "", text)
    text = " ".join(text.split())
    return text if len(text) <= 240 else text[:237] + "..."


def _readable_provider_error(
    e: Exception, model: str, label: str, key_id: int | None, account_id: int | None
) -> Exception | None:
    """A litellm exception as one of this module's exceptions, with a
    message for the person using the app, recording what it says about the
    key on the way. Only a rejected key or a rate limit says anything about
    the key itself; every other error leaves its stored status alone. None
    for anything that isn't a provider error, which the caller re-raises
    unchanged."""
    import litellm

    from app.core.api_keys_store import record_dispatch_outcome

    if not type(e).__module__.startswith("litellm"):
        return None
    logger.warning("LLM call to %s failed: %s", model, e)
    name = model.split("/", 1)[1] if "/" in model else model
    detail = _provider_detail(e)

    if isinstance(e, litellm.RateLimitError):
        message = (
            f"{label} is limiting requests from this key right now: too many requests in a "
            "short time, or its free quota is used up. Wait a minute and try again, or check "
            f"the key's limits in your {label} account."
        )
        if key_id is not None:
            record_dispatch_outcome(key_id, ok=False, rate_limited=True, detail=message)
        record_event("llm", "rate_limited", detail, context=model, account_id=account_id)
        return LLMRateLimitedError(message)
    if isinstance(e, (litellm.Timeout, litellm.APIConnectionError)):
        return LLMUnavailableError(
            f"Couldn't reach {label}: the connection failed or timed out, even after several "
            "tries. Check your internet connection and try again."
        )
    if _is_transient(e):
        return LLMUnavailableError(
            f"{label} is too busy to answer right now (\"{name}\" is getting more requests "
            "than it can handle), and it still failed after several tries. This is on "
            f"{label}'s side, not a problem with your key. Try again in a few minutes, or "
            "pick a different model on Manage APIs."
        )
    if isinstance(e, litellm.AuthenticationError):
        message = f"{label} rejected the API key. Check it, or add a new one, on Manage APIs."
        if key_id is not None:
            record_dispatch_outcome(key_id, ok=False, detail=message)
        return LLMProviderError(message)
    if isinstance(e, litellm.PermissionDeniedError):
        return LLMProviderError(
            f"{label} says this key isn't allowed to use \"{name}\". Pick a different model "
            f"on Manage APIs, or check the key's permissions. ({detail})"
        )
    if isinstance(e, litellm.NotFoundError):
        return LLMProviderError(
            f"{label} doesn't recognize the model \"{name}\". Pick a different model on "
            "Manage APIs."
        )
    if isinstance(e, litellm.ContextWindowExceededError):
        return LLMProviderError(
            f"This is too long for \"{name}\" to read in one go. Try a shorter file, or pick "
            "a model with a larger context window on Manage APIs."
        )
    if isinstance(e, litellm.ContentPolicyViolationError):
        return LLMProviderError(
            f"{label} refused to process this content under its safety rules. ({detail})"
        )
    if isinstance(e, (litellm.BadRequestError, litellm.UnprocessableEntityError)):
        return LLMProviderError(f"{label} couldn't process this request with \"{name}\": {detail}")
    return LLMProviderError(f"{label} returned an unexpected error: {detail}")


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
