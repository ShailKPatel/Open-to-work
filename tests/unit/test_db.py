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
    _migrate_job_postings_salary_columns,
    _migrate_job_postings_tracking_columns,
    _migrate_repositories_profile_readme_column,
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


def test_migrate_job_postings_salary_columns_backfills_from_extracted_text(tmp_path):
    """Rows extracted before the annual salary columns existed get them
    filled from the salary text already stored, no reprocess needed."""
    import json

    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    with engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE job_postings (id INTEGER PRIMARY KEY, extracted_json JSON)")
        )
        conn.execute(
            text(
                "INSERT INTO job_postings (id, extracted_json) "
                "VALUES (1, :a), (2, :b), (3, NULL)"
            ),
            {
                "a": json.dumps({"salary_range": "1.5 to 1.6 lakh per month"}),
                "b": json.dumps({"salary_range": ""}),
            },
        )
        conn.commit()

    _migrate_job_postings_salary_columns(engine)
    _migrate_job_postings_salary_columns(engine)  # idempotent

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, salary_min_annual, salary_max_annual, salary_currency "
                "FROM job_postings ORDER BY id"
            )
        ).all()
    assert rows == [(1, 1800000, 1920000, "INR"), (2, None, None, None), (3, None, None, None)]


def test_migrate_repositories_profile_readme_column_flags_existing_profile_repos(tmp_path):
    """Repos synced before the column existed: the "owner/owner" one is
    flagged at once, so it leaves the projects list without waiting for a
    change on GitHub. Matching ignores case, as GitHub logins do."""
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    with engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE repositories (id INTEGER PRIMARY KEY, name TEXT, full_name TEXT)")
        )
        conn.execute(
            text(
                "INSERT INTO repositories (id, name, full_name) VALUES "
                "(1, 'Octocat', 'octocat/Octocat'), (2, 'proj', 'octocat/proj'), "
                "(3, 'octocat', 'someone/octocat')"
            )
        )
        conn.commit()

    _migrate_repositories_profile_readme_column(engine)
    _migrate_repositories_profile_readme_column(engine)  # idempotent

    with engine.connect() as conn:
        flags = dict(conn.execute(text("SELECT id, is_profile_readme FROM repositories")).all())
    assert flags == {1: 1, 2: 0, 3: 0}


def test_migrate_repositories_profile_readme_column_recategorizes_every_startup(tmp_path):
    """The column already exists but a profile repo was saved unflagged
    (a project, as before this feature): the next startup moves it."""
    from sqlalchemy import create_engine, text

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE repositories (id INTEGER PRIMARY KEY, name TEXT, full_name TEXT, "
                "is_profile_readme BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO repositories (id, name, full_name) VALUES "
                "(1, 'octocat', 'octocat/octocat'), (2, 'proj', 'octocat/proj')"
            )
        )
        conn.commit()

    _migrate_repositories_profile_readme_column(engine)

    with engine.connect() as conn:
        flags = dict(conn.execute(text("SELECT id, is_profile_readme FROM repositories")).all())
    assert flags == {1: 1, 2: 0}


def test_migrate_exclude_from_resume_columns_adds_flag_to_every_archivable_table(tmp_path):
    """Tables from before archiving existed get the flag, defaulting to
    included, and a second run is a no-op."""
    from sqlalchemy import create_engine, text

    from app.core.db.migrations import _migrate_exclude_from_resume_columns

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    tables = ("repositories", "experiences", "education")
    with engine.connect() as conn:
        for table in tables:
            conn.execute(text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)"))
            conn.execute(text(f"INSERT INTO {table} (id) VALUES (1)"))
        conn.commit()

    _migrate_exclude_from_resume_columns(engine)
    _migrate_exclude_from_resume_columns(engine)

    with engine.connect() as conn:
        for table in tables:
            assert conn.execute(text(f"SELECT exclude_from_resume FROM {table}")).scalar() == 0


def test_drop_retired_tables_removes_them_and_is_idempotent(tmp_path):
    from sqlalchemy import create_engine, inspect, text

    from app.core.db.migrations import _drop_retired_tables

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    with engine.connect() as conn:
        for table in ("auth_sources", "detections", "match_results", "accounts"):
            conn.execute(text(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY)"))
        conn.commit()

    _drop_retired_tables(engine)
    _drop_retired_tables(engine)

    assert inspect(engine).get_table_names() == ["accounts"]


def _index_shape(engine, table: str) -> dict[str, tuple[bool, tuple[str, ...]]]:
    from sqlalchemy import text

    with engine.connect() as conn:
        shape = {}
        for row in conn.execute(text(f"PRAGMA index_list({table})")):
            columns = tuple(r[2] for r in conn.execute(text(f"PRAGMA index_info({row[1]})")))
            shape[row[1]] = (bool(row[2]), columns)
        return shape


def _old_schema_db(path: Path):
    """Today's tables with the global unique indexes they carried before
    repos and postings became unique per account, holding one row per
    account and rows in every table that points at them."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from app.core.db import (
        Account,
        Base,
        JobPosting,
        ProjectLink,
        Repository,
        Resume,
        SkillEvidence,
    )
    from app.core.db.migrations import _PER_ACCOUNT_UNIQUE

    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for table, column, old_index, new_index in _PER_ACCOUNT_UNIQUE:
            conn.execute(text(f"DROP INDEX {new_index}"))
            conn.execute(text(f"DROP INDEX {old_index}"))
            conn.execute(text(f"CREATE UNIQUE INDEX {old_index} ON {table} ({column})"))
    with Session(engine) as db:
        db.add_all(
            [
                Account(id=1, first_name="A", last_name="A", github_username="a"),
                Account(id=2, first_name="B", last_name="B", github_username="b"),
            ]
        )
        db.flush()
        db.add_all(
            [
                Repository(id=7, account_id=1, github_id=100, name="x", full_name="o/x", url=""),
                Repository(id=9, account_id=2, github_id=200, name="y", full_name="o/y", url=""),
                JobPosting(
                    id=3, account_id=1, source="pasted", external_id="e", company="c",
                    title="t", raw_text_quarantined="text", content_hash="h1",
                ),
            ]
        )
        db.flush()
        db.add_all(
            [
                ProjectLink(repo_id=7, label="Demo", url="https://example.com"),
                SkillEvidence(
                    skill="Python", repo_id=9, evidence_type="readme", weight=1.0, confidence=1.0
                ),
                Resume(account_id=1, filename="r.pdf", mime_type="application/pdf",
                       job_posting_id=3),
            ]
        )
        db.commit()
    engine.dispose()


def _init_db_at(path: Path, monkeypatch) -> None:
    from app.core.db import reset_engine
    from app.core.settings import get_settings

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    get_settings.cache_clear()
    reset_engine()
    try:
        init_db()
    finally:
        get_settings.cache_clear()


def test_per_account_unique_migration_keeps_rows_ids_and_references(tmp_path, monkeypatch):
    import pytest
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError

    db_path = tmp_path / "app.db"
    _old_schema_db(db_path)
    assert _index_shape(create_engine(f"sqlite:///{db_path}"), "repositories")[
        "ix_repositories_github_id"
    ] == (True, ("github_id",))

    _init_db_at(db_path, monkeypatch)

    engine = create_engine(f"sqlite:///{db_path}")
    repos = _index_shape(engine, "repositories")
    postings = _index_shape(engine, "job_postings")
    assert repos["ix_repositories_github_id"] == (False, ("github_id",))
    assert repos["ix_repositories_full_name"] == (False, ("full_name",))
    assert repos["uq_repositories_account_github_id"] == (True, ("account_id", "github_id"))
    assert repos["uq_repositories_account_full_name"] == (True, ("account_id", "full_name"))
    assert postings["ix_job_postings_content_hash"] == (False, ("content_hash",))
    assert postings["uq_job_postings_account_content_hash"] == (
        True,
        ("account_id", "content_hash"),
    )
    assert list((tmp_path / "backups").iterdir())

    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT id, account_id, github_id, full_name FROM repositories ORDER BY id")
        ).all() == [(7, 1, 100, "o/x"), (9, 2, 200, "o/y")]
        postings = conn.execute(text("SELECT id, account_id, content_hash FROM job_postings"))
        assert postings.all() == [(3, 1, "h1")]
        assert conn.execute(text("SELECT repo_id FROM project_links")).scalar() == 7
        assert conn.execute(text("SELECT repo_id FROM skill_evidence")).scalar() == 9
        assert conn.execute(text("SELECT job_posting_id FROM resumes")).scalar() == 3
        assert conn.execute(text("PRAGMA foreign_key_check")).all() == []

    insert_repo = text(
        "INSERT INTO repositories (account_id, github_id, name, full_name, url, is_fork, "
        "stars, manifests_json, commits_authored, fetched_at, skill_extraction_status, "
        "starred, is_profile_readme, exclude_from_resume) VALUES (:acc, :gh, 'n', :fn, '', "
        "0, 0, '{}', 0, '2026-01-01', 'pending', 0, 0, 0)"
    )
    insert_posting = text(
        "INSERT INTO job_postings (account_id, source, external_id, company, title, "
        "raw_text_quarantined, content_hash, fetched_at, extraction_status, applied) "
        "VALUES (:acc, 'pasted', 'e', 'c', 't', 'text', 'h1', '2026-01-01', 'pending', 0)"
    )
    with engine.connect() as conn:
        conn.execute(insert_repo, {"acc": 2, "gh": 100, "fn": "o/x"})
        conn.execute(insert_posting, {"acc": 2})
        conn.commit()
        for statement, params in (
            (insert_repo, {"acc": 1, "gh": 100, "fn": "o/other"}),
            (insert_repo, {"acc": 1, "gh": 101, "fn": "o/x"}),
            (insert_posting, {"acc": 1}),
        ):
            with pytest.raises(IntegrityError):
                conn.execute(statement, params)
            conn.rollback()


def test_per_account_unique_migration_twice_is_a_no_op(tmp_path, monkeypatch):
    from sqlalchemy import create_engine, text

    from app.core.db import Base

    db_path = tmp_path / "app.db"
    _old_schema_db(db_path)
    _init_db_at(db_path, monkeypatch)
    engine = create_engine(f"sqlite:///{db_path}")

    def snapshot():
        with engine.connect() as conn:
            rows = {
                table: conn.execute(text(f"SELECT * FROM {table} ORDER BY id")).all()
                for table in ("repositories", "job_postings")
            }
        return rows, {t: _index_shape(engine, t) for t in ("repositories", "job_postings")}

    before = snapshot()
    _init_db_at(db_path, monkeypatch)
    assert snapshot() == before

    fresh = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(fresh)
    assert {t: _index_shape(fresh, t) for t in ("repositories", "job_postings")} == before[1]


def test_per_account_unique_migration_rolls_back_on_failure(tmp_path, monkeypatch):
    """A failure partway leaves every old index in place, not half of them."""
    import pytest
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    from app.core.db import migrations

    db_path = tmp_path / "app.db"
    _old_schema_db(db_path)
    monkeypatch.setattr(
        migrations,
        "_PER_ACCOUNT_UNIQUE",
        (*migrations._PER_ACCOUNT_UNIQUE, ("job_postings", "missing", "ix_none", "uq_none")),
    )
    engine = create_engine(f"sqlite:///{db_path}")

    with pytest.raises(OperationalError):
        migrations._migrate_per_account_unique_indexes(engine)

    repos = _index_shape(engine, "repositories")
    assert repos["ix_repositories_github_id"] == (True, ("github_id",))
    assert "uq_repositories_account_github_id" not in repos
    assert _index_shape(engine, "job_postings")["ix_job_postings_content_hash"][0] is True
