"""Every LLM call in the app goes through here. No direct provider SDK calls
anywhere else.

One entrypoint, not one method per modality: `complete(tier, messages, schema)`.
`messages` is the standard chat-message format used by LiteLLM: a list of
{"role", "content"} dicts where `content` is either a plain string (text-only)
or a list of typed parts (text + image_url, for multimodal). Callers never
hand-build that JSON: `user_message(text, images=...)` constructs one, with or
without attachments, and the same `complete()` call handles both. No separate
multimodal client, no per-modality wrapper to keep in sync.

Content-hash cache: identical (model, messages, schema) never bills twice,
with the messages' text normalized for whitespace first so a prompt that
differs only in blank lines or line endings is still one prompt. Keyed on
the model rather than the tier, since that is what decides the answer.
Budget cap: enforced before dispatch, not after. Every call, cached or not,
gets a row in LLMCall, tagged with the `purpose` its caller passed, so the
dashboard shows real call volume and which feature spent it alongside real
spend.

Key failover: a provider blaming the key it was handed (quota used up,
credential rejected, this key not allowed near that model) is not the end
of the request. Every key stored for that provider is tried in turn, and
the request only fails once they are all spent, with one line per key
saying what happened to each. That matters most in the middle of a long
job: a multi-step job is many separate complete() calls, so a key dying at
step three leaves steps one and two already done and committed, and
switching keys lets step three finish rather than stranding the job there.
Errors that every key would hit identically (an overloaded provider, a
prompt that is too long, refused content) skip failover entirely, so a
doomed request fails once instead of once per key. See
_dispatch_over_keys().

Prompt caching: the dispatch copy of a prompt gets the system message
marked for provider-side caching where that needs saying explicitly
(_with_prompt_caching below). The local cache above only helps an identical
repeat; this is what makes the shared instructions in front of every
per-item prompt cheap the second time.
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
    """Raised right before a real (non-injected) LLM dispatch when no
    usable key is stored for the tier's provider yet (see
    app/core/api_keys_store.py and the /apis page). Not raised any earlier
    in complete(): a cache hit or a test's injected _completion_fn never
    needs a key."""


class LLMDispatchError(RuntimeError):
    """Base for the errors raised about one dispatch attempt.

    `blames_key` is what drives key failover: True means the provider
    blamed the credential we dispatched with (quota used up, key
    rejected, this key not allowed near that model), so the next stored
    key is worth trying. False means the next key would fail identically
    (the provider is down, the prompt is too long, the content was
    refused), so trying one would only waste a call and delay the real
    error. Set per instance in _readable_provider_error(), which is the
    one place that knows which of the two a provider exception is.
    """

    blames_key = False


class LLMRateLimitedError(LLMDispatchError):
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


class LLMProviderError(LLMDispatchError):
    """The provider refused the request for a reason retrying won't fix: a
    rejected key, an unknown model name, or input it can't handle."""


def is_out_of_keys(error: Exception) -> bool:
    """True when this error means there is nothing left to dispatch with
    for that provider: no key stored, a budget used up, or every stored
    key tried and spent (quota gone, credential rejected, model not
    permitted).

    For a caller working through a batch, that is the difference between
    "this item failed" and "every remaining item is about to fail the
    same way". Batches use it to stop after the first one rather than
    spending a doomed call per item and filling each with the same
    message (app/profile/build.py, app/profile/extract.py,
    app/profile/skill_review.py). Stopping is also what makes the run
    resumable: items never attempted keep their pending status, so the
    next run carries on from there instead of redoing what worked.

    False for a failure that is about this one request (a prompt too
    long, content refused) or about the provider's own health, which a
    later item might not hit.
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
    """Whitespace-insensitive form of one text block: line endings
    normalized, trailing spaces gone, blank lines dropped. Used for cache
    keying only, never for the request itself, which always carries the
    caller's original text unchanged.

    Blank lines go rather than getting collapsed to one, because the
    differences this exists to absorb are not tidy: the same README read
    twice, once before and once after a cleaning pass that left a
    different number of gaps behind, is one prompt to a model and should
    be one prompt here.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def _canonical_messages(messages: list[dict]) -> list[dict]:
    """The messages as the cache key sees them: same roles, same order,
    same attachments, text normalized by _canonical_text.

    Why normalize at all: two prompts that differ only in trailing
    spaces, line endings, or a run of blank lines get the same response
    from the provider, so the second one should be served from the first
    one's row instead of being billed again. That difference is not
    hypothetical: a job posting pasted twice, or a README fetched twice
    from GitHub, routinely differs by exactly that much and nothing else.
    """
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


def _prompt_hash(model: str, messages: list[dict], schema: dict | None) -> str:
    """Keyed on the model, not the tier. What comes back depends on the
    model and the messages; the tier is only how this app routed there. A
    device with the same model set for both tiers would otherwise pay
    twice for one prompt, once under "bulk" and once under "quality".
    """
    payload = json.dumps(
        {"model": model, "messages": _canonical_messages(messages), "schema": schema},
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


def _key_month_spend_usd(key_id: int) -> float:
    """This month's spend charged to one stored key, for that key's
    optional budget_cap_usd (app/core/db/models.py's ApiKey). Read off
    LLMCall.key_id, the same column /monitor's per-key breakdown uses, so
    a cap and the number shown next to it always agree.

    Keyed on the key rather than on the provider because keys now
    genuinely rotate within a single request (see _dispatch_over_keys
    below): a provider-wide total would charge every key for every other
    key's calls and retire the whole provider as soon as the cheapest cap
    was reached. Calls recorded before key attribution existed carry no
    key_id and count against no cap.
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
    """The messages to dispatch, with the system prompt marked for
    provider-side prompt caching where that needs saying explicitly.

    Every call site here puts its fixed instructions in a system message
    and the per-item text after it, so the system message is a real shared
    prefix across an entire batch: one repo's extraction and the next
    repo's send the identical bytes. Marked, a provider bills that prefix
    at its cached rate on every call after the first.

    Returns a copy; the caller's own list is never mutated. Applied after
    the cache key is computed, so a marker can never change which local
    rows a prompt matches. Below a provider's minimum cacheable prefix
    (1024 tokens on Anthropic's larger models, 2048 on the small ones) the
    marker is ignored rather than rejected, so short prompts are not a
    special case here.
    """
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
    response_json: dict,
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
) -> LLMResponse:
    """_completion_fn is an injection point for tests; production callers
    never pass it; it defaults to litellm.completion.

    purpose is a short stable label for which feature spent this call
    ("repo_facts", "resume_build", "pagefit_trim", ...). Recorded on the
    LLMCall row and grouped by /monitor's usage breakdown, so spend can be
    attributed to a feature rather than only to a model or a tier. Never
    part of the cache key: the same prompt reached from two features is
    still one prompt.

    account_id is optional context for app/core/api_keys_store.py's
    per-key allow-list (a key can be restricted to specific profiles on
    a shared device): pass it whenever the caller already knows which
    account this call is for (most do, e.g. app/profile/extract.py and
    app/resume_build/orchestrator.py). None means "no account context",
    which only unrestricted keys can serve.

    A caller never sees or picks a key. Every key stored for the tier's
    provider that may serve this account is tried in order until one
    answers (see _dispatch_over_keys), and the LLMCall row records which
    one actually paid, so a swap mid-job shows up on /monitor without the
    caller doing anything.
    """
    settings = get_llm_settings()
    model = settings.model_for(tier)
    prompt_hash = _prompt_hash(model, messages, schema)

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
            purpose=purpose,
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
        prompt_hash = _prompt_hash(model, messages, schema)

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
        purpose=purpose,
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
    """The same request against each stored key for this provider in
    turn, stopping at the first that answers. Returns (response,
    latency_ms, the model that answered, the key id that paid for it).

    This is what keeps a long job alive. A multi-step job (extract, then
    select, then trim) is many separate complete() calls, and a key that
    dies partway leaves every step before it already done and committed.
    Failing here would strand the job at that step; switching keys and
    answering means the step finishes and the job simply carries on to
    the next one, with the swap visible in the log and on /monitor rather
    than silent.

    A key is only abandoned when the provider blamed the key itself
    (`blames_key`, see LLMDispatchError) or its own budget cap is used
    up. An overloaded provider, a prompt that is too long, or content the
    provider refused would fail identically on every key, so those are
    raised straight away instead of burning the rest of the keys on a
    request that cannot succeed.

    Raises only once every key is spent, with one line per key saying
    what happened to it.
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
    """The one exception raised once every key is spent.

    A single stored key keeps that key's own message and type verbatim:
    with nothing to fall back to there is no failover to explain, and the
    message already says what to do. Several keys get one line each, so
    the answer to "why did this stop?" is the whole list (this one is out
    of quota, that one was rejected, this one hit its cap) and not just
    whichever happened to be tried last. The type comes from the last
    attempt, which keeps the HTTP status each API route already maps it
    to (app/api/resume_build.py's _map_llm_error).
    """
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
    """A litellm exception as one of this module's exceptions, with a
    message for the person using the app, recording what it says about the
    key on the way. Only a rejected key or a rate limit says anything about
    the key itself; every other error leaves its stored status alone. None
    for anything that isn't a provider error, which the caller re-raises
    unchanged.

    Whether the returned error carries `blames_key` is what decides
    failover: a quota, a rejected credential, or a model this key may not
    use are all worth retrying on the next key; an overloaded provider or
    a prompt the model cannot accept are not."""
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
