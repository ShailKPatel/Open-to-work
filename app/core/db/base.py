"""Declarative base and the shared default-timestamp helper.

Its own module so app/core/db/models.py and app/core/db/migrations.py can
both import it without importing each other.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
