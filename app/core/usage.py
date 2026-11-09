"""Counts of what happens on a running instance: builds started, resumes
finished and what the judge did to them. Read back by the demo's per-day
limit (app/api/demo.py) and by scripts/usage_report.py. Holds ids and
numbers only, never resume text.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from sqlalchemy import func, select

from app.core.db import UsageEvent, get_db

logger = logging.getLogger(__name__)

BUILD_STARTED = "build_started"
RESUME_BUILT = "resume_built"
DEMO_WRITE = "demo_write"
DEMO_INTAKE = "demo_intake"


def start_of_utc_day() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def record(account_id: int, kind: str, data: dict[str, Any] | None = None) -> None:
    """Best-effort: a metric must never fail the thing it is measuring."""
    db = get_db()
    try:
        db.add(UsageEvent(account_id=account_id, kind=kind, data_json=data or {}))
        db.commit()
    except Exception:
        logger.exception("could not record usage event %s", kind)
    finally:
        db.close()


def count_today(account_id: int, kind: str) -> int:
    db = get_db()
    try:
        return db.execute(
            select(func.count(UsageEvent.id)).where(
                UsageEvent.account_id == account_id,
                UsageEvent.kind == kind,
                UsageEvent.created_at >= start_of_utc_day(),
            )
        ).scalar_one()
    finally:
        db.close()
