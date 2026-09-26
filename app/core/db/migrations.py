"""Lightweight in-place column migrations, plus init_db().

Deliberately not Alembic: this is a local single-file SQLite app, and each
function here adds a column (or backfills a table) only if it is missing,
so startup is idempotent and an older database upgrades itself in place.
Each function reads the current shape with PRAGMA table_info rather than
trusting a recorded version, which is why skipping releases is safe and why
running them twice costs nothing.

What that buys in idempotence it gives up in coverage: a column added to a
model with no matching function here is never created, because create_all()
only ever adds whole tables. _verify_schema() at the end of init_db() is
the backstop, and turns that omission into a startup failure naming the
column instead of a 500 on whichever page queries it first.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.db.base import Base
from app.core.db.engine import _backup_sqlite_file, get_engine
from app.core.settings import get_settings

logger = logging.getLogger(__name__)


def _migrate_accounts_contact_columns(engine: Engine) -> None:
    """Add accounts.contact_email / contact_phone / contact_location to an
    already-existing accounts table. create_all() only creates missing
    tables and never alters an existing one's columns, so without this an
    upgraded instance would never get them. Idempotent: checks PRAGMA
    table_info first, so running this on a fresh table (fresh install, or
    a test's :memory: db) that already has all columns via create_all is a
    safe no-op.
    """
    if engine.dialect.name != "sqlite":
        return  # PRAGMA table_info is SQLite-specific; only dialect this project runs


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(accounts)"))}
        for column in ("contact_email", "contact_phone", "contact_location"):
            if column not in existing:
                conn.execute(text(f"ALTER TABLE accounts ADD COLUMN {column} TEXT"))
        conn.commit()


def _migrate_repositories_curation_columns(engine: Engine) -> None:
    """Add repositories.starred to an already-existing repositories table,
    same reasoning and same idempotent PRAGMA-check pattern as
    _migrate_accounts_contact_columns above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(repositories)"))}
        if "starred" not in existing:
            conn.execute(
                text("ALTER TABLE repositories ADD COLUMN starred BOOLEAN NOT NULL DEFAULT 0")
            )
        conn.commit()


def _migrate_resumes_name_column(engine: Engine) -> None:
    """Add resumes.name to an already-existing resumes table, same
    reasoning and pattern as _migrate_repositories_curation_columns above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        if "name" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN name TEXT"))
        conn.commit()


def _migrate_job_postings_account_id(engine: Engine) -> None:
    """Add job_postings.account_id to an already-existing job_postings
    table, same reasoning and pattern as _migrate_resumes_name_column
    above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN account_id INTEGER"))
        conn.commit()


def _migrate_resumes_build_columns(engine: Engine) -> None:
    """Add resumes.job_posting_id/template/content_json/compiled_path/
    compiled_at to an already-existing resumes table, same reasoning and
    pattern as _migrate_resumes_name_column above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        if "job_posting_id" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN job_posting_id INTEGER"))
        if "template" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN template TEXT"))
        if "content_json" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN content_json JSON"))
        if "compiled_path" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN compiled_path TEXT"))
        if "compiled_at" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN compiled_at DATETIME"))
        conn.commit()


def _migrate_job_postings_extraction_columns(engine: Engine) -> None:
    """Add job_postings.extraction_status/extraction_error/extracted_at to
    an already-existing job_postings table, same reasoning and pattern as
    _migrate_job_postings_account_id above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "extraction_status" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE job_postings ADD COLUMN extraction_status "
                    "TEXT NOT NULL DEFAULT 'pending'"
                )
            )
        if "extraction_error" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN extraction_error TEXT"))
        if "extracted_at" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN extracted_at DATETIME"))
        conn.commit()


def _migrate_llm_calls_attribution_columns(engine: Engine) -> None:
    """Add llm_calls.account_id/key_id/purpose to an already-existing
    llm_calls table, same reasoning and pattern as
    _migrate_resumes_name_column above. Backs the /monitor
    per-key/per-account/per-purpose usage breakdown (app/api/monitor.py).
    Rows written before this migration have them NULL, which the breakdown
    shows as an "unattributed" bucket rather than dropping that spend.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(llm_calls)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE llm_calls ADD COLUMN account_id INTEGER"))
        if "key_id" not in existing:
            conn.execute(text("ALTER TABLE llm_calls ADD COLUMN key_id INTEGER"))
        if "purpose" not in existing:
            conn.execute(text("ALTER TABLE llm_calls ADD COLUMN purpose TEXT"))
        conn.commit()


def _migrate_rate_limit_events_account_column(engine: Engine) -> None:
    """Add rate_limit_events.account_id to an already-existing table, same
    pattern as _migrate_llm_calls_attribution_columns above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(rate_limit_events)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE rate_limit_events ADD COLUMN account_id INTEGER"))
        conn.commit()


def _migrate_job_postings_tracking_columns(engine: Engine) -> None:
    """Add job_postings.applied/applied_at/applied_notes/role_family_id/
    screenshot_path to an already-existing job_postings table, same
    reasoning and pattern as _migrate_job_postings_account_id above.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "applied" not in existing:
            conn.execute(
                text("ALTER TABLE job_postings ADD COLUMN applied BOOLEAN NOT NULL DEFAULT 0")
            )
        if "applied_at" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN applied_at DATE"))
        if "applied_notes" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN applied_notes TEXT"))
        if "role_family_id" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN role_family_id INTEGER"))
        if "screenshot_path" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN screenshot_path TEXT"))
        conn.commit()


def _migrate_contact_items(engine: Engine) -> None:
    """Ensures contact_emails and contact_phones tables exist and populates
    them from existing accounts.contact_email / contact_phone if present.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        try:
            accounts = conn.execute(
                text("SELECT id, contact_email, contact_phone FROM accounts")
            ).fetchall()
            for acc_id, email, phone in accounts:
                if email:
                    existing_email = conn.execute(
                        text("SELECT id FROM contact_emails WHERE account_id = :acc_id"),
                        {"acc_id": acc_id},
                    ).fetchone()
                    if not existing_email:
                        conn.execute(
                            text(
                                "INSERT INTO contact_emails (account_id, email, is_primary)"
                                " VALUES (:acc_id, :email, 1)"
                            ),
                            {"acc_id": acc_id, "email": email},
                        )
                if phone:
                    existing_phone = conn.execute(
                        text("SELECT id FROM contact_phones WHERE account_id = :acc_id"),
                        {"acc_id": acc_id},
                    ).fetchone()
                    if not existing_phone:
                        conn.execute(
                            text(
                                "INSERT INTO contact_phones (account_id, phone, is_primary)"
                                " VALUES (:acc_id, :phone, 1)"
                            ),
                            {"acc_id": acc_id, "phone": phone},
                        )
            conn.commit()
        except Exception:
            logger.exception("Error during contact items migration; continuing")


def _migrate_api_keys_exhaustion_columns(engine: Engine) -> None:
    """Add api_keys.exhausted_at / retry_at / exhaustion_kind to an
    already-existing api_keys table, same idempotent PRAGMA-check pattern
    as the migrations above. A key stored before this existed simply has
    them empty: a NULL retry_at reads as "due now", so the first refresh
    pass (app/core/key_refresh.py) picks up an already-exhausted key
    instead of leaving it waiting forever.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(api_keys)"))}
        for column, column_type in (
            ("exhausted_at", "DATETIME"),
            ("retry_at", "DATETIME"),
            ("exhaustion_kind", "TEXT"),
        ):
            if column not in existing:
                conn.execute(text(f"ALTER TABLE api_keys ADD COLUMN {column} {column_type}"))
        conn.commit()


class SchemaOutOfDateError(RuntimeError):
    """A table exists but is missing a column the models declare, so some
    migration above does not cover it. Raised at startup rather than left
    for the first query that touches the column."""


def _verify_schema(engine: Engine) -> None:
    """Every column the models declare exists on every table that exists.

    create_all() creates missing tables and never alters an existing one, so
    a column added to a model without a matching migration above is simply
    absent, and nothing notices until a query selects it. That failure is
    invisible at startup: /health answers 200, the container healthcheck
    passes, and then one page 500s. Worse, a missing column on api_keys
    takes every LLM call in the app with it, because resolve_dispatch_keys
    selects them.

    So the check is the whole point rather than a version number: a stored
    version says where a database is believed to be, this says whether it
    actually matches what the code will ask for. Runs after the migrations,
    against the tables as they now are.

    Only missing columns are an error. A column the database has and the
    models no longer declare is fine and expected: SQLite cannot drop a
    column without rebuilding the table, so retired ones are left in place
    and unread (see Repository.starred's note about the old `rating`).
    """
    if engine.dialect.name != "sqlite":
        return  # PRAGMA table_info is SQLite-specific, the only dialect here

    missing: dict[str, list[str]] = {}
    with engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            if not rows:
                continue  # table does not exist at all; create_all owns that
            present = {row[1] for row in rows}
            absent = [c.name for c in table.columns if c.name not in present]
            if absent:
                missing[table_name] = absent

    if missing:
        detail = "; ".join(f"{table}: {', '.join(cols)}" for table, cols in sorted(missing.items()))
        raise SchemaOutOfDateError(
            "This database is missing columns the code expects, so some queries would "
            f"fail at the moment they run: {detail}. A column added to a model needs a "
            "matching migration in app/core/db/migrations.py; add one (or restore a "
            "backup from data/backups/) and start again."
        )


def init_db() -> None:
    _backup_sqlite_file(get_settings().database_url)
    engine = get_engine()
    Base.metadata.create_all(engine)
    _migrate_accounts_contact_columns(engine)
    _migrate_repositories_curation_columns(engine)
    _migrate_resumes_name_column(engine)
    _migrate_job_postings_account_id(engine)
    _migrate_resumes_build_columns(engine)
    _migrate_job_postings_extraction_columns(engine)
    _migrate_llm_calls_attribution_columns(engine)
    _migrate_rate_limit_events_account_column(engine)
    _migrate_job_postings_tracking_columns(engine)
    _migrate_contact_items(engine)
    _migrate_api_keys_exhaustion_columns(engine)
    _verify_schema(engine)
