"""Engine creation, SQLite tuning, pre-migration backups, and the session
factory. No model imports: this module is what models and migrations are
wired onto, not the other way round.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = get_settings().database_url
        is_sqlite = url.startswith("sqlite")
        if is_sqlite:
            path = url.split("///")[-1]
            if path and path != ":memory:":
                import os

                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        _engine = create_engine(url)
        if is_sqlite:
            _configure_sqlite(_engine)
    return _engine


def _configure_sqlite(engine: Engine) -> None:
    """WAL mode + a busy timeout. Without this, two connections writing in
    quick succession, e.g. core.llm.complete()'s own get_db() call for
    LLMCall bookkeeping, invoked from inside app.profile.build's
    already-open per-repo transaction, hit
    'sqlite3.OperationalError: database is locked' under SQLite's default
    rollback-journal locking when extracting skills for several repos in a
    row. WAL allows concurrent readers alongside a single writer instead of
    locking the whole file; the busy timeout makes a genuine write/write
    collision wait and retry instead of failing immediately.
    """
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


_BACKUP_KEEP = 10


def _backup_sqlite_file(url: str) -> None:
    """Snapshot the live SQLite file into data/backups/ before init_db()
    touches anything, so a startup that ends up looking at an empty or
    missing DB (a bad restart, a deploy that lands in the wrong directory,
    an accidental delete) always has a recent restore point next to it:
    `cp data/backups/<newest>.db data/open_to_work.db` restores it. This
    only ever adds files.

    Uses sqlite3's own `.backup()` API rather than a raw file copy: WAL
    mode (see _configure_sqlite) means the freshest writes can still be
    sitting in a `-wal` file, invisible to a plain `cp` of just the main
    `.db` file. `.backup()` reads through a live connection and always
    produces a fully consistent snapshot regardless of WAL state.

    Best-effort and silent on failure: a backup that can't be taken
    should never be the reason the app fails to start.
    """
    if not url.startswith("sqlite:///") or url.endswith(":memory:"):
        return
    db_path = Path(url.removeprefix("sqlite:///"))
    if not db_path.exists() or db_path.stat().st_size == 0:
        return  # nothing real to back up yet
    try:
        backups_dir = db_path.parent / "backups"
        backups_dir.mkdir(exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        dest = backups_dir / f"{db_path.stem}-{stamp}{db_path.suffix}"

        import sqlite3

        src_conn = sqlite3.connect(str(db_path))
        try:
            dest_conn = sqlite3.connect(str(dest))
            try:
                src_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        finally:
            src_conn.close()

        # prune to the most recent _BACKUP_KEEP: a safety net, not an archive
        backups = sorted(backups_dir.glob(f"{db_path.stem}-*{db_path.suffix}"))
        for stale in backups[:-_BACKUP_KEEP]:
            stale.unlink(missing_ok=True)
    except Exception:
        logger.exception("could not back up %s before init; continuing anyway", db_path)


def get_db() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def reset_engine() -> None:
    """Drop the cached engine and session factory so the next get_engine()
    / get_db() builds a new one from the current settings.

    A test seam: the suite points DATABASE_URL at a fresh temp file per
    test and needs the next call to pick that up instead of reusing the
    engine bound to the previous file. Disposes the old engine first so
    its pooled connections close now rather than at garbage collection.
    Nothing in the running app calls this.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
