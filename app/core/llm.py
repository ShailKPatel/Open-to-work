"""Every LLM call in the app goes through complete(). Nothing else
imports a provider SDK.

Messages use LiteLLM's chat format; user_message() builds one with or
without image and file attachments, so text and multimodal calls share
one path.

- Cache: an identical (model, messages, schema) is answered from the
  llm_calls table instead of billing again. Whitespace in the text is
  normalized for the key, so a README fetched twice is one prompt. A
  response that does not parse when a schema was asked for is never
  served from the cache, and bypass_cache skips the lookup for a retry
  the person asked for.
- Budget: the monthly cap is checked before dispatch.
- Recording: every call, cached or not, gets an LLMCall row with cost,
  tokens and the `purpose` the caller passed.
- Key failover: when the provider blames the key (quota, rejected
  credential, model not allowed), the next stored key takes the same
  request. Errors every key would hit alike (provider overloaded, prompt
  too long, refused content) fail at once. See _dispatch_over_keys().
- Prompt caching: for providers that need it said explicitly, the system
  message is marked so a shared prefix is billed at the cached rate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select

from app.core.app_settings import Tier, get_llm_settings
from app.core.db import LLMCall, get_db
from app.core.llm_providers import (
    PROVIDER_LABELS,
    is_blocked_detail,
    litellm_kwargs,
    provider_of_model,
)
from app.core.rate_limits import record_event

if TYPE_CHECKING:  # imported for types only; the real import stays deferred
    from app.core.api_keys_store import DispatchKey

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """Raised before dispatch when the monthly cap would be breached."""


class ApiKeyMissingError(RuntimeError):
    """No usable key is stored for the tier's provider. Raised only at
    real dispatch, so cache hits and test fakes never need a key."""


class LLMDispatchError(RuntimeError):
    """Base for errors from one dispatch attempt. blames_key is True when
    the provider blamed the credential, so another stored key is worth
    trying; False when every key would fail the same way."""

    blames_key = False


class LLMRateLimitedError(LLMDispatchError):
    """The provider rate-limited us (429, quota used up). Unlike
    BudgetExceededError, which is our own cap checked before dispatch.
    Batch callers treat both the same: stop rather than spend calls that
    will fail alike."""


class LLMUnavailableError(LLMRateLimitedError):
    """The provider is overloaded or unreachable (HTTP 500/502/503/504, a
    timeout, a dropped connection) even after complete() retried and tried
    the other tier's model. A subclass of LLMRateLimitedError because it
    means the same thing to every caller: stop, and try again later."""


class LLMProviderError(LLMDispatchError):
    """The provider refused the request for a reason retrying won't fix: a
    rejected key, an unknown model name, or input it can't handle."""


def is_out_of_keys(error: Exception) -> bool:
    """True when nothing is left to dispatch with: no key, a budget used
    up, or every key spent. A batch uses this to stop after the first such
    failure instead of failing every remaining item the same way; items
    never attempted stay pending, so the next run picks up from there.
    """
    if isinstance(error, (ApiKeyMissingError, BudgetExceededError)):
        return True
    return isinstance(error, LLMDispatchError) and error.blames_key


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


def _canonical_text(text: str) -> str:
    """Text as the cache key sees it: line endings normalized, trailing
    spaces and blank lines dropped. The request itself is never altered."""
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _canonical_messages(messages: list[dict]) -> list[dict]:
    """Messages with their text run through _canonical_text, so prompts
    that differ only in whitespace share one cache row."""
    canonical: list[dict] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            canonical.append({"role": message.get("role"), "content": _canonical_text(content)})
        elif isinstance(content, list):
            parts: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append({"type": "text", "text": _canonical_text(part.get("text", ""))})
                else:
                    parts.append(part)
            canonical.append({"role": message.get("role"), "content": parts})
        else:
            canonical.append(message)
    return canonical


# Part of every cache key. Bump it when response parsing or post-processing
# changes, so answers cached under the old handling are asked for again.
_CACHE_VERSION = 1


def _prompt_hash(
    model: str, messages: list[dict], schema: dict | None, account_id: int | None
) -> str:
    """Keyed on the model rather than the tier: with the same model on
    both tiers, one prompt is one cache entry. Keyed on the account too,
    so an answer paid for by a key restricted to one profile is never
    served to another."""
    payload = json.dumps(
        {
            "version": _CACHE_VERSION,
            "model": model,
            "account_id": account_id,
            "messages": _canonical_messages(messages),
            "schema": schema,
        },
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
    """This month's total LLM spend, for display (app/api/monitor.py)."""
    return _month_spend_usd()


def _key_month_spend_usd(key_id: int) -> float:
    """This month's spend charged to one key, for its own budget cap.
    Per key rather than per provider, since keys rotate within a request
    and one key's cap should not retire the others."""
    import datetime as dt

    db = get_db()
    try:
        month_start = dt.datetime.now(dt.UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        total = db.execute(
            select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                LLMCall.created_at >= month_start,
                LLMCall.key_id == key_id,
            )
        ).scalar_one()
        return float(total)
    finally:
        db.close()


# Providers that need an explicit marker on the part of the prompt worth
# caching on their side. Gemini and OpenAI match a repeated prefix by
# themselves with nothing extra in the request, and would reject a marker
# they do not define, so they are deliberately absent here.
_EXPLICIT_CACHE_PROVIDERS = {"anthropic", "bedrock"}


def _with_prompt_caching(provider: str, messages: list[dict]) -> list[dict]:
    """A copy of the messages with the system prompt marked for
    provider-side caching, on providers that need the marker. Every call
    site puts fixed instructions in the system message, so it is a shared
    prefix across a batch. Applied after the cache key is computed.
    Providers ignore the marker below their minimum cacheable length."""
    if provider not in _EXPLICIT_CACHE_PROVIDERS:
        return messages
    marked: list[dict] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "system" and isinstance(content, str):
            marked.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            marked.append(message)
    return marked


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
    response_json: dict | None,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    latency_ms: int,
    cached: bool,
    account_id: int | None = None,
    key_id: int | None = None,
    purpose: str | None = None,
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
                purpose=purpose,
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
    purpose: str | None = None,
    _completion_fn: Any = None,
    *,
    bypass_cache: bool = False,
) -> LLMResponse:
    """Sends one request for `tier` and returns the response.

    purpose labels the feature spending the call ("repo_facts",
    "resume_build", ...) for /monitor; it is not part of the cache key.
    account_id lets keys restricted to certain profiles serve the call;
    None can only use unrestricted keys. It is part of the cache key.
    Callers never pick a key: every usable one is tried in order, and the
    LLMCall row records which paid.
    bypass_cache skips the cache lookup, for a retry the person asked for;
    the fresh answer is still recorded, so later calls can reuse it.
    _completion_fn replaces litellm.completion in tests.
    """
    settings = get_llm_settings()
    model = settings.model_for(tier)
    prompt_hash = _prompt_hash(model, messages, schema, account_id)

    cached_row = None if bypass_cache else _lookup_cache(prompt_hash)
    cached_response = cached_row.response_json if cached_row is not None else None
    cached_content = (cached_response or {}).get("content", "")
    if cached_response is not None and (not schema or _try_parse(cached_content) is not None):
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
            purpose=purpose,
        )
        parsed = _try_parse(cached_content) if schema else None
        return LLMResponse(
            content=cached_content, parsed=parsed, cost_usd=0.0, cached=True, model=model
        )

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
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _with_prompt_caching(provider, messages),
    }

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

    if _completion_fn is not None:
        response, latency_ms, answered_model = _dispatch(
            _completion_fn, kwargs, candidates, label, None, account_id
        )
    else:
        from app.core.api_keys_store import resolve_dispatch_keys

        keys = resolve_dispatch_keys(provider, account_id)
        if not keys:
            raise ApiKeyMissingError(
                f"No {label} API key is available"
                f"{' for this account' if account_id is not None else ''}. "
                "Add one, or turn an existing one on, from Manage APIs."
            )

        import litellm

        response, latency_ms, answered_model, key_id = _dispatch_over_keys(
            litellm.completion, kwargs, candidates, provider, label, keys, account_id
        )

    if answered_model != model:
        model = answered_model
        prompt_hash = _prompt_hash(model, messages, schema, account_id)

    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens_in = getattr(usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(usage, "completion_tokens", 0) or 0

    cost_usd = _safe_completion_cost(response)
    parsed = _try_parse(content) if schema else None

    # A response that does not parse is still recorded for its spend and
    # tokens, but without its content, so the cache never replays it.
    _record(
        tier=tier,
        model=model,
        prompt_hash=prompt_hash,
        response_json=None if schema and parsed is None else {"content": content},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        cached=False,
        account_id=account_id,
        key_id=key_id,
        purpose=purpose,
    )

    return LLMResponse(
        content=content,
        parsed=parsed,
        cost_usd=cost_usd,
        cached=False,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _dispatch(
    completion_fn: Any,
    kwargs: dict[str, Any],
    candidates: list[str],
    label: str,
    key_id: int | None,
    account_id: int | None,
) -> tuple[Any, int, str]:
    """One credential's worth of trying: each model in `candidates` with
    its retries, the second one only if the first stays overloaded.
    Returns (response, latency_ms, the model that answered).

    Raises this module's own exception for a provider error (the caller
    reads `blames_key` on it to decide whether another key is worth
    trying) and anything else unchanged.
    """
    for i, candidate in enumerate(candidates):
        attempt_kwargs = {**kwargs, "model": candidate}
        delays = _RETRY_DELAYS_S if i == 0 else _FALLBACK_RETRY_DELAYS_S
        try:
            response, latency_ms = _call_with_retries(completion_fn, attempt_kwargs, delays)
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
        return response, latency_ms, candidate
    raise AssertionError("unreachable")


@dataclass
class _KeyAttempt:
    """One stored key that did not serve this request, and why. Kept so
    the error raised once they are all gone can name every key and its
    own reason rather than only the last one's."""

    label: str
    error: Exception


def _key_budget_error(
    key: DispatchKey, label: str, model: str, account_id: int | None
) -> BudgetExceededError | None:
    """This key's own monthly cap, checked before spending a call on it.
    None means it has room (or has no cap)."""
    if key.budget_cap_usd is None:
        return None
    spent = _key_month_spend_usd(key.id)
    if spent < key.budget_cap_usd:
        return None
    detail = (
        f"The {label} key budget of ${key.budget_cap_usd:.2f} is used up "
        f"(${spent:.2f} spent this month), so this request was not sent. "
        "Raise or remove the key's cap on Manage APIs."
    )
    record_event("llm", "budget_exceeded", detail, context=model, account_id=account_id)
    return BudgetExceededError(detail)


def _dispatch_over_keys(
    completion_fn: Any,
    kwargs: dict[str, Any],
    candidates: list[str],
    provider: str,
    label: str,
    keys: list[DispatchKey],
    account_id: int | None,
) -> tuple[Any, int, str, int]:
    """Tries the request on each stored key in turn and returns
    (response, latency_ms, answering model, key id) from the first that
    answers. A long job is many separate calls, so a key dying midway
    would otherwise strand it at that step.

    A key is skipped only when the provider blamed it or its own cap is
    used up; any other error is raised at once, since every key would hit
    it. Raises once all keys are spent, with one line per key.
    """
    from app.core.api_keys_store import record_dispatch_outcome

    attempts: list[_KeyAttempt] = []
    for position, key in enumerate(keys):
        remaining = len(keys) - position - 1
        budget_error = _key_budget_error(key, label, kwargs["model"], account_id)
        if budget_error is not None:
            attempts.append(_KeyAttempt(key.label, budget_error))
            _log_key_gave_up(label, key, budget_error, remaining, account_id)
            continue

        try:
            response, latency_ms, answered = _dispatch(
                completion_fn,
                {**kwargs, **litellm_kwargs(provider, key.credentials)},
                candidates,
                label,
                key.id,
                account_id,
            )
        except LLMDispatchError as e:
            if not e.blames_key:
                raise
            attempts.append(_KeyAttempt(key.label, e))
            _log_key_gave_up(label, key, e, remaining, account_id)
            continue

        if attempts:
            logger.info(
                "%s key %r answered after %d key(s) failed; the request went through",
                label, key.label, len(attempts),
            )
        # Clears a "rate_limited"/"invalid" mark left by an earlier run
        # on a key that plainly works again; a healthy key is not written.
        record_dispatch_outcome(key.id, ok=True)
        return response, latency_ms, answered, key.id

    raise _keys_exhausted_error(attempts, label, kwargs["model"], account_id)


def _log_key_gave_up(
    label: str, key: DispatchKey, error: Exception, remaining: int, account_id: int | None
) -> None:
    what_next = (
        f"switching to the next {label} key ({remaining} left)"
        if remaining
        else "no other key left to try"
    )
    logger.warning(
        "%s key %r (id=%s) is out: %s; %s", label, key.label, key.id, error, what_next
    )
    record_event(
        "llm",
        "key_failover" if remaining else "keys_exhausted",
        f"{key.label}: {error}",
        context=f"{label} key id={key.id}",
        account_id=account_id,
    )


def _keys_exhausted_error(
    attempts: list[_KeyAttempt], label: str, model: str, account_id: int | None
) -> Exception:
    """The error raised once every key is spent. With one key, that key's
    error unchanged; with several, one line per key. The type comes from
    the last attempt, so API routes map it to the same HTTP status."""
    last = attempts[-1].error
    if len(attempts) == 1:
        return last
    lines = "\n".join(f"- {a.label}: {a.error}" for a in attempts)
    detail = (
        f"All {len(attempts)} {label} keys failed on this request, so it could not be "
        f"finished:\n{lines}\n"
        "Add another key, or fix one of these, on Manage APIs, then run it again: "
        "the steps that already finished are kept and will not be redone."
    )
    record_event("llm", "keys_exhausted", detail, context=model, account_id=account_id)
    return type(last)(detail)


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


def _blames_key(error: LLMDispatchError) -> LLMDispatchError:
    """Marks an error as the key's fault, so _dispatch_over_keys() tries
    the next stored key instead of giving up here."""
    error.blames_key = True
    return error


def _key_blocked(label: str, detail: str, key_id: int | None) -> LLMDispatchError:
    """The provider has shut this key off rather than refused one request.
    Worth a failover (another stored key may be healthy), never worth an
    automatic recheck (see app/core/api_keys_store.py's recheck_keys)."""
    from app.core.api_keys_store import record_dispatch_outcome

    message = (
        f"{label} has blocked this key: it looks suspended, revoked, or not enabled for its "
        f"project. Waiting will not fix it, so fix or replace it in your {label} account and "
        f"then recheck it on Manage APIs. ({detail})"
    )
    if key_id is not None:
        record_dispatch_outcome(key_id, ok=False, blocked=True, detail=message)
    return _blames_key(LLMProviderError(message))


def _readable_provider_error(
    e: Exception, model: str, label: str, key_id: int | None, account_id: int | None
) -> LLMDispatchError | None:
    """Turns a litellm exception into this module's error with a message
    for the person using the app, and records what it says about the key.
    blames_key on the result decides failover. None for a non-provider
    error, which the caller re-raises."""
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
            # The untruncated exception text goes along as provider_detail:
            # that is where a provider says which quota window was hit and
            # how long it wants us to wait (app/core/key_cooldown.py).
            record_dispatch_outcome(
                key_id,
                ok=False,
                rate_limited=True,
                detail=message,
                provider_detail=str(e),
            )
        record_event("llm", "rate_limited", detail, context=model, account_id=account_id)
        return _blames_key(LLMRateLimitedError(message))
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
        if is_blocked_detail(str(e)):
            return _key_blocked(label, detail, key_id)
        message = f"{label} rejected the API key. Check it, or add a new one, on Manage APIs."
        if key_id is not None:
            record_dispatch_outcome(key_id, ok=False, detail=message)
        return _blames_key(LLMProviderError(message))
    if isinstance(e, litellm.PermissionDeniedError):
        if is_blocked_detail(str(e)):
            # A forbidden key, not a forbidden model: suspended, revoked,
            # or its API never enabled. Marked `blocked` so it lands in the
            # needs-attention group on /apis and no automatic recheck
            # keeps asking a question only the provider's console can answer.
            return _key_blocked(label, detail, key_id)
        # Not marked at all: the key is fine, it just may not use this
        # model. Another stored key for the same provider may be allowed
        # to, so this is still worth a failover.
        return _blames_key(LLMProviderError(
            f"{label} says this key isn't allowed to use \"{name}\". Pick a different model "
            f"on Manage APIs, or check the key's permissions. ({detail})"
        ))
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
