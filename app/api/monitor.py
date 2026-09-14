"""Read-only view over app/core/rate_limits.py's event log, a live
snapshot of both providers' current headroom, and a per-key/per-account/
per-provider/per-tier breakdown of every LLM call ever made (LLMCall,
attributed via account_id/key_id, see app/core/db.py's docstrings on
those columns). Backs the /monitor page. Nothing here writes; recording
happens at the call sites themselves (app/ingest/github/client.py,
app/core/llm.py) so this module can stay a pure reporting layer.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select

from app.core import api_keys_store, rate_limits
from app.core.db import Account, LLMCall, get_db
from app.core.llm import month_spend_usd
from app.core.llm_providers import PROVIDER_LABELS, provider_of_model
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitor")

# Label used for the "no account context" / "no attributable key" buckets
# in the breakdowns below, rather than a bare `null` the UI has to guess
# at. Applies to real gaps (a call made with account_id=None) and to rows
# that predate the account_id/key_id columns (an older LLMCall row,
# see the migration's docstring in app/core/db.py) alike; there's no way
# to tell those two apart after the fact, and both mean the same thing to
# someone reading the breakdown: "this spend isn't attributed to one."
_UNATTRIBUTED = "(unattributed)"


class RateLimitEventOut(BaseModel):
    id: int
    source: str
    kind: str
    detail: str
    context: str | None
    account_id: int | None
    created_at: str

    model_config = {"from_attributes": True}


class GithubStatus(BaseModel):
    ok: bool
    limit: int | None = None
    remaining: int | None = None
    reset_at: str | None = None
    error: str | None = None


class LlmStatus(BaseModel):
    monthly_budget_usd: float
    spent_usd: float
    bulk_model: str
    quality_model: str


class KeyStatusOut(BaseModel):
    """One configured ApiKey (app/core/api_keys_store.py) plus this
    month's usage attributed to it, for the /monitor page's "usage by API
    key" view. `spent_usd`/`calls` only count LLMCall rows with this
    key's id in key_id; a key that's never actually dispatched (or every
    call went through the cache) legitimately shows 0."""

    id: int
    provider: str
    provider_label: str
    label: str
    enabled: bool
    is_active: bool
    status: str
    budget_cap_usd: float | None
    spent_usd_month: float
    calls_month: int


class MonitorStatus(BaseModel):
    github: GithubStatus
    llm: LlmStatus
    counts_24h: dict[str, int]
    counts_7d: dict[str, int]
    keys: list[KeyStatusOut]


class UsageBucket(BaseModel):
    key: str
    label: str
    calls: int
    cached_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class UsageTotals(BaseModel):
    calls: int
    cached_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class UsageBreakdown(BaseModel):
    """LLM usage over a trailing window, optionally narrowed by
    account_id/provider/key_id (the query params on GET /llm/usage).
    Those filters apply to the underlying row set BEFORE every grouping
    below, so e.g. account_id=5 shows just that profile's own
    provider/key/tier/model mix, not the whole device's."""

    days: int
    totals: UsageTotals
    by_provider: list[UsageBucket]
    by_key: list[UsageBucket]
    by_account: list[UsageBucket]
    by_tier: list[UsageBucket]
    by_model: list[UsageBucket]


class LLMCallOut(BaseModel):
    id: int
    created_at: str
    tier: str
    model: str
    provider: str
    provider_label: str
    account_id: int | None
    account_name: str | None
    key_id: int | None
    key_label: str | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cached: bool
    latency_ms: int


@router.get("/events", response_model=list[RateLimitEventOut])
def list_events(
    source: str | None = None, limit: int = 50, account_id: int | None = None
) -> list[RateLimitEventOut]:
    limit = max(1, min(limit, 200))
    rows = rate_limits.list_events(source=source, limit=limit, account_id=account_id)
    return [
        RateLimitEventOut(
            id=row.id,
            source=row.source,
            kind=row.kind,
            detail=row.detail,
            context=row.context,
            account_id=row.account_id,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


def _github_status() -> GithubStatus:
    """Live headroom straight from GitHub's API, separate from the event
    log, since a quiet log doesn't mean there's currently headroom (an
    unauthenticated 60/hr budget can be sitting near zero without a single
    logged event, if nothing here has actually been rejected yet)."""
    try:
        from app.ingest.github.client import GitHubClient

        core = GitHubClient()._gh.get_rate_limit().resources.core
        return GithubStatus(
            ok=True,
            limit=core.limit,
            remaining=core.remaining,
            reset_at=core.reset.astimezone(dt.UTC).isoformat(),
        )
    except Exception as e:
        logger.warning("could not fetch live GitHub rate-limit status: %s", e)
        return GithubStatus(ok=False, error=str(e))


def _month_start() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _key_usage_this_month() -> dict[int, tuple[int, float]]:
    """{key_id: (calls, cost_usd)} for the current calendar month, real
    dispatches only (see LLMCall.key_id's docstring for why a mocked/
    cached call never has one)."""
    db = get_db()
    try:
        rows = db.execute(
            select(
                LLMCall.key_id,
                func.count(LLMCall.id),
                func.coalesce(func.sum(LLMCall.cost_usd), 0.0),
            )
            .where(LLMCall.created_at >= _month_start(), LLMCall.key_id.is_not(None))
            .group_by(LLMCall.key_id)
        ).all()
        return {key_id: (calls, float(cost)) for key_id, calls, cost in rows}
    finally:
        db.close()


def _key_statuses() -> list[KeyStatusOut]:
    """Every configured ApiKey (app/core/api_keys_store.py), across every
    provider, not just whichever tier's model happens to be active, with
    this month's real usage merged in. The single source of truth for
    "which key did this app actually spend money through", one row per
    key rather than the one-model-per-tier summary LlmStatus gives."""
    usage = _key_usage_this_month()
    out = []
    for key in api_keys_store.list_keys():
        calls, cost = usage.get(key["id"], (0, 0.0))
        out.append(
            KeyStatusOut(
                id=key["id"],
                provider=key["provider"],
                provider_label=PROVIDER_LABELS.get(key["provider"], key["provider"]),
                label=key["label"],
                enabled=key["enabled"],
                is_active=key["is_active"],
                status=key["status"],
                budget_cap_usd=key["budget_cap_usd"],
                spent_usd_month=cost,
                calls_month=calls,
            )
        )
    return out


@router.get("/status", response_model=MonitorStatus)
def status() -> MonitorStatus:
    settings = get_settings()
    return MonitorStatus(
        github=_github_status(),
        llm=LlmStatus(
            monthly_budget_usd=settings.monthly_budget_usd,
            spent_usd=month_spend_usd(),
            bulk_model=settings.llm_bulk_model,
            quality_model=settings.llm_quality_model,
        ),
        counts_24h=rate_limits.count_events_since(24),
        counts_7d=rate_limits.count_events_since(24 * 7),
        keys=_key_statuses(),
    )


def _filtered_calls_query(*, days: int, account_id: int | None, key_id: int | None):
    """Every filter except `provider` applies here: `provider` isn't its
    own column, it's derived from the litellm model-string prefix (see
    app/core/llm_providers.py), so callers filter rows by it in Python
    after fetching. Call volume on a local single-user device is small
    enough this never needs to be a query."""
    since = dt.datetime.now(dt.UTC) - dt.timedelta(days=days)
    stmt = select(LLMCall).where(LLMCall.created_at >= since)
    if account_id is not None:
        stmt = stmt.where(LLMCall.account_id == account_id)
    if key_id is not None:
        stmt = stmt.where(LLMCall.key_id == key_id)
    return stmt


def _provider_label(model: str) -> str:
    provider = provider_of_model(model)
    return PROVIDER_LABELS.get(provider, provider)


def _account_names(db) -> dict[int, str]:
    return {
        a.id: f"{a.first_name} {a.last_name}".strip()
        for a in db.execute(select(Account)).scalars()
    }


def _key_labels() -> dict[int, str]:
    return {k["id"]: k["label"] for k in api_keys_store.list_keys()}


def _bucketize(rows: list[LLMCall], key_fn, label_fn) -> list[UsageBucket]:
    buckets: dict[str, list] = {}
    for row in rows:
        k = key_fn(row)
        label = label_fn(row)
        if k not in buckets:
            buckets[k] = [label, 0, 0, 0, 0, 0.0]
        b = buckets[k]
        b[1] += 1
        b[2] += 1 if row.cached else 0
        b[3] += row.tokens_in
        b[4] += row.tokens_out
        b[5] += row.cost_usd
    return [
        UsageBucket(
            key=k,
            label=v[0],
            calls=v[1],
            cached_calls=v[2],
            tokens_in=v[3],
            tokens_out=v[4],
            cost_usd=v[5],
        )
        for k, v in sorted(buckets.items(), key=lambda kv: kv[1][5], reverse=True)
    ]


@router.get("/llm/usage", response_model=UsageBreakdown)
def llm_usage(
    days: int = 30,
    account_id: int | None = None,
    provider: str | None = None,
    key_id: int | None = None,
) -> UsageBreakdown:
    days = max(1, min(days, 365))
    stmt = _filtered_calls_query(days=days, account_id=account_id, key_id=key_id)
    db = get_db()
    try:
        rows = list(db.execute(stmt).scalars().all())
        if provider is not None:
            rows = [r for r in rows if provider_of_model(r.model) == provider]

        accounts = _account_names(db)
        keys = _key_labels()
    finally:
        db.close()

    totals = UsageTotals(
        calls=len(rows),
        cached_calls=sum(1 for r in rows if r.cached),
        tokens_in=sum(r.tokens_in for r in rows),
        tokens_out=sum(r.tokens_out for r in rows),
        cost_usd=sum(r.cost_usd for r in rows),
    )

    by_provider = _bucketize(
        rows, lambda r: provider_of_model(r.model), lambda r: _provider_label(r.model)
    )
    by_key = _bucketize(
        rows,
        lambda r: str(r.key_id) if r.key_id is not None else "none",
        lambda r: keys.get(r.key_id, f"key #{r.key_id}") if r.key_id is not None else _UNATTRIBUTED,
    )
    by_account = _bucketize(
        rows,
        lambda r: str(r.account_id) if r.account_id is not None else "none",
        lambda r: accounts.get(r.account_id, f"account #{r.account_id}")
        if r.account_id is not None
        else _UNATTRIBUTED,
    )
    by_tier = _bucketize(rows, lambda r: r.tier, lambda r: r.tier)
    by_model = _bucketize(rows, lambda r: r.model, lambda r: r.model)

    return UsageBreakdown(
        days=days,
        totals=totals,
        by_provider=by_provider,
        by_key=by_key,
        by_account=by_account,
        by_tier=by_tier,
        by_model=by_model,
    )


@router.get("/llm/calls", response_model=list[LLMCallOut])
def llm_calls(
    days: int = 30,
    account_id: int | None = None,
    provider: str | None = None,
    key_id: int | None = None,
    tier: str | None = None,
    limit: int = 50,
) -> list[LLMCallOut]:
    """Individual LLMCall rows, newest first: the drill-down under the
    aggregate /llm/usage breakdown, same filter set plus `tier`."""
    limit = max(1, min(limit, 200))
    stmt = _filtered_calls_query(days=days, account_id=account_id, key_id=key_id)
    if tier is not None:
        stmt = stmt.where(LLMCall.tier == tier)
    stmt = stmt.order_by(LLMCall.id.desc())
    # provider isn't a column (see _filtered_calls_query) so it's filtered
    # in Python after fetch below; over-fetch a safety-capped window to
    # still find `limit` matches without scanning the whole table.
    stmt = stmt.limit(limit if provider is None else max(limit * 20, 500))

    db = get_db()
    try:
        rows = list(db.execute(stmt).scalars().all())
        if provider is not None:
            rows = [r for r in rows if provider_of_model(r.model) == provider]
        rows = rows[:limit]

        accounts = _account_names(db)
        keys = _key_labels()
    finally:
        db.close()

    return [
        LLMCallOut(
            id=r.id,
            created_at=r.created_at.isoformat(),
            tier=r.tier,
            model=r.model,
            provider=provider_of_model(r.model),
            provider_label=_provider_label(r.model),
            account_id=r.account_id,
            account_name=accounts.get(r.account_id) if r.account_id is not None else None,
            key_id=r.key_id,
            key_label=keys.get(r.key_id) if r.key_id is not None else None,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            cost_usd=r.cost_usd,
            cached=r.cached,
            latency_ms=r.latency_ms,
        )
        for r in rows
    ]
