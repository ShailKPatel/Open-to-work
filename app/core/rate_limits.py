"""Append-only log of rate-limit / budget-cap hits, written from
app/ingest/github/client.py (GitHub) and app/core/llm.py (provider rate
limits and the monthly budget cap). Backs the /monitor page (app/api/monitor.py).

record_event() is best-effort: it must never be the reason a
real request fails, so any DB error here is swallowed and logged, not
raised. Callers that already caught a rate-limit exception and are about
to re-raise or return an error to their own caller shouldn't have that
error handling short-circuited by a logging problem.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.core.db import RateLimitEvent, get_db

logger = logging.getLogger(__name__)

Source = str  # "github" | "llm"
Kind = str  # "rate_limited" | "budget_exceeded"


def record_event(
    source: Source,
    kind: Kind,
    detail: str,
    context: str | None = None,
    account_id: int | None = None,
) -> None:
    db = get_db()
    try:
        db.add(
            RateLimitEvent(
                source=source, kind=kind, detail=detail, context=context, account_id=account_id
            )
        )
        db.commit()
    except Exception:
        logger.exception(
            "could not record rate-limit event (%s/%s); continuing anyway", source, kind
        )
    finally:
        db.close()


def list_events(
    source: Source | None = None, limit: int = 50, account_id: int | None = None
) -> list[RateLimitEvent]:
    db = get_db()
    try:
        stmt = select(RateLimitEvent).order_by(RateLimitEvent.id.desc()).limit(limit)
        if source:
            stmt = stmt.where(RateLimitEvent.source == source)
        if account_id is not None:
            stmt = stmt.where(RateLimitEvent.account_id == account_id)
        return list(db.execute(stmt).scalars().all())
    finally:
        db.close()


def count_events_since(hours: int, account_id: int | None = None) -> dict[str, int]:
    """{"github": N, "llm": N} counts over the trailing `hours` window, for
    the monitor page's summary tiles. account_id narrows to one profile's
    own events; GitHub events never carry one (see RateLimitEvent's
    docstring), so filtering by account_id only ever counts LLM events."""
    import datetime as dt

    from sqlalchemy import func

    db = get_db()
    try:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
        stmt = (
            select(RateLimitEvent.source, func.count(RateLimitEvent.id))
            .where(RateLimitEvent.created_at >= since)
            .group_by(RateLimitEvent.source)
        )
        if account_id is not None:
            stmt = stmt.where(RateLimitEvent.account_id == account_id)
        rows = db.execute(stmt).all()
        return {source: count for source, count in rows}
    finally:
        db.close()
