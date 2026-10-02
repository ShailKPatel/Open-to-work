"""Shared FastAPI dependencies, kept out of app/core/db so that package
has no web-framework import.

DbSession gives a route a session that FastAPI closes when the response
finishes, including after an exception.

Streaming routes (sources.py's sync_source_stream, projects.py's
process_pending) open and close their own session before returning the
generator, since a dependency-held session would stay open for the whole
stream.
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
