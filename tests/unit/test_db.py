"""Coverage for the SQLite snapshot taken at startup: without it, a deleted or
replaced database file would leave the app starting over on an empty database
with nothing to restore from. See app/core/db/engine.py::_backup_sqlite_file.
"""

import sqlite3
from pathlib import Path

from app.core.db import (
    _backup_sqlite_file,
    _migrate_accounts_contact_columns,
    _migrate_job_postings_account_id,
    _migrate_job_postings_tracking_columns,
    init_db,
)


def test_no_backup_when_file_does_not_exist(tmp_path):
    db_path = tmp_path / "app.db"
    _backup_sqlite_file(f"sqlite:///{db_path}")
    assert not (tmp_path / "backups").exists()


def test_no_backup_when_file_is_empty(tmp_path):
    db_path = tmp_path / "app.db"
    db_path.touch()  # 0 bytes, nothing real to preserve
    _backup_sqlite_file(f"sqlite:///{db_path}")
    assert not (tmp_path / "backups").exists()


def test_memory_db_never_backed_up(tmp_path):
    _backup_sqlite_file("sqlite:///:memory:")
    assert not (tmp_path / "backups").exists()


def _make_real_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES ('real data')")
    conn.commit()
    conn.close()


def test_backup_created_and_contains_real_data(tmp_path):
    db_path = tmp_path / "app.db"
    _make_real_db(db_path)

    _backup_sqlite_file(f"sqlite:///{db_path}")

    backups = list((tmp_path / "backups").glob("app-*.db"))
    assert len(backups) == 1
    conn = sqlite3.connect(str(backups[0]))
    assert conn.execute("SELECT val FROM t").fetchone() == ("real data",)
    conn.close()


def test_old_backups_pruned_beyond_keep_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.db.engine._BACKUP_KEEP", 3)
    db_path = tmp_path / "app.db"
    _make_real_db(db_path)
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    # 5 pre-existing backups, oldest to newest by filename (matches the
    # real naming scheme's sort order; ISO timestamps sort lexically)
    for i in range(5):
        _make_real_db(backups_dir / f"app-2026010{i}T000000Z.db")

    _backup_sqlite_file(f"sqlite:///{db_path}")  # adds a 6th, newest

    backups = sorted(backups_dir.glob("app-*.db"))
    assert len(backups) == 3  # _BACKUP_KEEP, oldest ones pruned
    # the ones that survived are the newest, including the just-added one
    assert backups[-1].name.startswith("app-2026")  # the fresh one sorts last


def test_backup_failure_does_not_raise(tmp_path):
    """A backup that can't be taken must never be the reason startup
    fails: a realistic failure (can't write to the backups directory),
    not a contrived one, and _backup_sqlite_file itself must swallow it."""
    db_path = tmp_path / "app.db"
    _make_real_db(db_path)
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    backups_dir.chmod(0o400)  # read-only; the write inside should fail

    try:
        _backup_sqlite_file(f"sqlite:///{db_path}")  # must not raise
    finally:
        backups_dir.chmod(0o700)  # restore so pytest can clean up tmp_path


def test_init_db_takes_a_backup_first(tmp_path, monkeypatch):
    """Confirms the wiring (init_db() calls the backup function
    before touching the schema) via a spy, not by re-testing
    _backup_sqlite_file's own behavior again."""
    import os

    from app.core.settings import get_settings

    db_path = tmp_path / "app.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    get_settings.cache_clear()

    calls = []
    monkeypatch.setattr(
        "app.core.db.migrations._backup_sqlite_file", lambda url: calls.append(url)
    )
    try:
        init_db()
    finally:
        get_settings.cache_clear()

    assert calls == [f"sqlite:///{db_path}"]


def test_migrate_accounts_contact_columns_adds_missing_columns(tmp_path):
    """Simulates an already-existing accounts table from before
    contact_email/contact_phone existed. create_all() only creates
    missing TABLES, never alters an existing one's columns, so without
    this migration these two columns would never appear on an upgraded
    instance."""
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE accounts (id INTEGER PRIMARY KEY, first_name TEXT, "
                "last_name TEXT, github_username TEXT)"
            )
        )
        conn.commit()

    _migrate_accounts_contact_columns(engine)

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(accounts)"))}
    assert {"contact_email", "contact_phone"} <= cols


def test_migrate_accounts_contact_columns_idempotent(tmp_path):
    """Running the migration twice (two app restarts, both against the
    same already-migrated file) must not raise; SQLite errors on
    ALTER TABLE ADD COLUMN for a column that already exists."""
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE accounts (id INTEGER PRIMARY KEY, first_name TEXT)")
        )
        conn.commit()

    _migrate_accounts_contact_columns(engine)
    _migrate_accounts_contact_columns(engine)  # must not raise

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(accounts)"))}
    assert {"contact_email", "contact_phone"} <= cols


def test_migrate_job_postings_account_id_adds_missing_column(tmp_path):
    """Simulates a job_postings table from before account_id existed, same
    reasoning as the
    accounts-contact-columns migration above.
    """
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE job_postings (id INTEGER PRIMARY KEY, source TEXT, "
                "external_id TEXT, company TEXT, title TEXT, content_hash TEXT)"
            )
        )
        conn.commit()

    _migrate_job_postings_account_id(engine)

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
    assert "account_id" in cols


def test_migrate_job_postings_account_id_idempotent(tmp_path):
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE job_postings (id INTEGER PRIMARY KEY)"))
        conn.commit()

    _migrate_job_postings_account_id(engine)
    _migrate_job_postings_account_id(engine)  # must not raise

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
    assert "account_id" in cols


def test_migrate_job_postings_tracking_columns_adds_missing_columns(tmp_path):
    """Simulates a job_postings table from before the applied-tracker/
    role-family/screenshot columns existed (session before this one)."""
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE job_postings (id INTEGER PRIMARY KEY, source TEXT, "
                "external_id TEXT, company TEXT, title TEXT, content_hash TEXT)"
            )
        )
        conn.commit()

    _migrate_job_postings_tracking_columns(engine)

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
    assert {"applied", "applied_at", "applied_notes", "role_family_id", "screenshot_path"} <= cols


def test_migrate_job_postings_tracking_columns_idempotent(tmp_path):
    from sqlalchemy import create_engine, text

    db_path = tmp_path / "app.db"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE job_postings (id INTEGER PRIMARY KEY)"))
        conn.commit()

    _migrate_job_postings_tracking_columns(engine)
    _migrate_job_postings_tracking_columns(engine)  # must not raise

    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
    assert "applied" in cols
