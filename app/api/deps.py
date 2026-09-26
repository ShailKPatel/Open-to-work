"""Shared FastAPI dependencies.

Kept here rather than in app/core/db/ so that package stays free of any
web-framework import: it is also used by the ingest, profile and eval
code, none of which runs under FastAPI.

`DbSession` replaces the hand-rolled `db = get_db()` / `try` /
`finally: db.close()` block that every route used to open with. FastAPI
closes the session when the response is finished, including when the
route raised, so the guarantee is the same one the `finally` gave.

Routes that outlive their own return value still manage a session by
hand. The SSE endpoints (app/api/sources.py's sync_source_stream,
app/api/projects.py's process_pending) deliberately read what they need
and close *before* handing back a generator, because a dependency-held
session would stay open for the whole life of the stream, which can be
minutes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.db import get_db


def db_session() -> Iterator[Session]:
    db = get_db()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(db_session)]
