"""Lightweight in-place column migrations, plus init_db().

Deliberately not Alembic: this is a local single-file SQLite app, and each
function here adds a column (or backfills a table) only if it is missing,
so startup is idempotent and an older database upgrades itself in place.
"""

from __future__ import annotations

import json
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


def _migrate_resumes_history_columns(engine: Engine) -> None:
    """Add resumes.experiences_json/education_json to an already-existing
    resumes table, same reasoning and pattern as
    _migrate_resumes_name_column above. Rows extracted before this column
    existed read as empty until reprocessed.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        for column in ("experiences_json", "education_json"):
            if column not in existing:
                conn.execute(
                    text(f"ALTER TABLE resumes ADD COLUMN {column} JSON NOT NULL DEFAULT '[]'")
                )
        conn.commit()


def _migrate_resumes_contact_column(engine: Engine) -> None:
    """Add resumes.contact_json, same pattern as
    _migrate_resumes_history_columns above. Rows extracted before it
    existed read as an empty contact block until reprocessed.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        if "contact_json" not in existing:
            conn.execute(
                text("ALTER TABLE resumes ADD COLUMN contact_json JSON NOT NULL DEFAULT '{}'")
            )
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
    screenshot_path/screenshot_paths to an already-existing job_postings table, same
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
        if "screenshot_paths" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN screenshot_paths JSON"))
        conn.commit()


def _migrate_job_postings_source_columns(engine: Engine) -> None:
    """Add job_postings.source_text/source_links, same pattern as
    _migrate_job_postings_tracking_columns above.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "source_text" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN source_text TEXT"))
        if "source_links" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN source_links JSON"))
        conn.commit()


def _migrate_job_postings_salary_columns(engine: Engine) -> None:
    """Add job_postings.salary_min_annual/salary_max_annual/salary_currency
    and fill them for rows extracted before they existed, from the
    salary_range text already in extracted_json (app/profile/salary.py),
    so older postings filter by pay without being reprocessed.
    """
    if engine.dialect.name != "sqlite":
        return

    from app.profile.salary import parse_salary

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "salary_min_annual" in existing:
            return
        conn.execute(text("ALTER TABLE job_postings ADD COLUMN salary_min_annual INTEGER"))
        conn.execute(text("ALTER TABLE job_postings ADD COLUMN salary_max_annual INTEGER"))
        conn.execute(text("ALTER TABLE job_postings ADD COLUMN salary_currency TEXT"))
        rows = conn.execute(
            text("SELECT id, extracted_json FROM job_postings WHERE extracted_json IS NOT NULL")
        ).all()
        for posting_id, raw in rows:
            try:
                extracted = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except ValueError:
                continue
            salary = parse_salary(extracted.get("salary_range", ""))
            conn.execute(
                text(
                    "UPDATE job_postings SET salary_min_annual = :lo, "
                    "salary_max_annual = :hi, salary_currency = :cur WHERE id = :id"
                ),
                {"lo": salary.min, "hi": salary.max, "cur": salary.currency, "id": posting_id},
            )
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


def _migrate_skills_source_resume_column(engine: Engine) -> None:
    """Add skills.source_resume_id to an already-existing skills table,
    same idempotent PRAGMA-check pattern as the migrations above. When the
    column is first added, rows that came from a resume before it existed
    are attributed to the earliest resume of the same account whose tags
    contain that name; everything else stays null, i.e. added by hand.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(skills)"))}
        if "source_resume_id" in existing:
            return
        conn.execute(text("ALTER TABLE skills ADD COLUMN source_resume_id INTEGER"))

        first_resume: dict[tuple[int, str], int] = {}
        resumes = conn.execute(
            text("SELECT id, account_id, tags_json FROM resumes ORDER BY uploaded_at, id")
        )
        for resume_id, account_id, tags_json in resumes:
            try:
                tags = json.loads(tags_json) if isinstance(tags_json, str) else tags_json
            except ValueError:
                continue
            for tag in tags or []:
                if isinstance(tag, str) and tag.strip():
                    first_resume.setdefault((account_id, tag.strip().casefold()), resume_id)

        for skill_id, account_id, name in conn.execute(
            text("SELECT id, account_id, name FROM skills")
        ).all():
            resume_id = first_resume.get((account_id, name.strip().casefold()))
            if resume_id is not None:
                conn.execute(
                    text("UPDATE skills SET source_resume_id = :rid WHERE id = :sid"),
                    {"rid": resume_id, "sid": skill_id},
                )
        conn.commit()


def _migrate_repositories_profile_readme_column(engine: Engine) -> None:
    """Add repositories.is_profile_readme, same idempotent PRAGMA-check
    pattern as the migrations above, then recategorize every repo by name
    on each startup, not only the one that adds the column: a profile
    README repo saved as a project (before the flag existed, or by a copy
    of the app that didn't know it) leaves the projects list at the next
    start instead of waiting for a sync or a reprocess. Same rule as
    is_profile_repo() in models.py: owner equals name, ignoring case.
    """
    if engine.dialect.name != "sqlite":
        return


    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(repositories)"))}
        if "is_profile_readme" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE repositories ADD COLUMN "
                    "is_profile_readme BOOLEAN NOT NULL DEFAULT 0"
                )
            )
        conn.execute(
            text(
                "UPDATE repositories SET is_profile_readme = "
                "(lower(full_name) = lower(name) || '/' || lower(name)) "
                "WHERE is_profile_readme != "
                "(lower(full_name) = lower(name) || '/' || lower(name))"
            )
        )
        conn.commit()


def _migrate_repositories_readme_sha_column(engine: Engine) -> None:
    """Add repositories.readme_sha, same idempotent PRAGMA-check pattern
    as the migrations above. Rows synced before it existed stay NULL; the
    sync then hashes the stored README text instead (sync.py's
    _known_readme_sha), so an upgrade doesn't reprocess every repo.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(repositories)"))}
        if "readme_sha" not in existing:
            conn.execute(text("ALTER TABLE repositories ADD COLUMN readme_sha TEXT"))
        conn.commit()


def _migrate_exclude_from_resume_columns(engine: Engine) -> None:
    """Add exclude_from_resume to repositories, experiences and education, same
    idempotent PRAGMA-check pattern as the migrations above. Existing rows
    default to included, which is how they behaved before.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        for table in ("repositories", "experiences", "education"):
            existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})"))}
            if "exclude_from_resume" not in existing:
                conn.execute(
                    text(
                        f"ALTER TABLE {table} ADD COLUMN "
                        "exclude_from_resume BOOLEAN NOT NULL DEFAULT 0"
                    )
                )
        conn.commit()


def _migrate_resumes_build_state_column(engine: Engine) -> None:
    """Add resumes.build_state_json, same idempotent PRAGMA-check pattern
    as the migrations above. Existing rows read as finished, which they
    are."""
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        if "build_state_json" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN build_state_json JSON"))
        conn.commit()


def _migrate_education_extras_columns(engine: Engine) -> None:
    """Add education.grade/details, same idempotent PRAGMA-check pattern
    as the migrations above. Existing rows read as no grade and no
    details, which renders exactly as before."""
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(education)"))}
        if "grade" not in existing:
            conn.execute(text("ALTER TABLE education ADD COLUMN grade TEXT"))
        if "details" not in existing:
            conn.execute(
                text("ALTER TABLE education ADD COLUMN details JSON NOT NULL DEFAULT '[]'")
            )
        conn.commit()


def _migrate_month_year_date_column_types(engine: Engine) -> None:
    """Retype experiences/education start_date and end_date from DATE to
    VARCHAR on databases created before the month-and-year change. SQLite
    gives a DATE column numeric affinity, so a year-only "2023" was stored
    and read back as the integer 2023. SQLite cannot change a column's type
    in place, so each one is copied into a new text column that takes its
    name.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        for table in ("experiences", "education"):
            types = {
                row[1]: str(row[2]).upper()
                for row in conn.execute(text(f"PRAGMA table_info({table})"))
            }
            for column in ("start_date", "end_date"):
                if types.get(column) != "DATE":
                    continue
                tmp = f"{column}_text"
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {tmp} VARCHAR"))
                conn.execute(text(f"UPDATE {table} SET {tmp} = CAST({column} AS TEXT)"))
                conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {column}"))
                conn.execute(text(f"ALTER TABLE {table} RENAME COLUMN {tmp} TO {column}"))
        conn.commit()


def _migrate_month_year_dates(engine: Engine) -> None:
    """Rewrite experiences/education start_date and end_date, and the
    dates inside each resume's experiences_json/education_json snapshot,
    from the ISO dates they were once stored as ("2026-03-01") to the
    month-and-year form they are stored as now ("mar 2026", see
    app/profile/month_year.py). Runs on every startup; a value already in
    that form normalizes to itself, so only old rows get written. An
    unreadable value is left alone rather than dropped.
    """
    if engine.dialect.name != "sqlite":
        return

    from app.profile.month_year import normalize_or_none

    def fixed(value: object) -> object:
        return normalize_or_none(value) or value

    with engine.connect() as conn:
        for table in ("experiences", "education"):
            rows = conn.execute(text(f"SELECT id, start_date, end_date FROM {table}")).all()
            for row_id, start, end in rows:
                new_start, new_end = fixed(start), fixed(end)
                if (new_start, new_end) != (start, end):
                    conn.execute(
                        text(f"UPDATE {table} SET start_date = :s, end_date = :e WHERE id = :id"),
                        {"s": new_start, "e": new_end, "id": row_id},
                    )

        rows = conn.execute(
            text("SELECT id, experiences_json, education_json FROM resumes")
        ).all()
        for resume_id, *columns in rows:
            updates = {}
            for column, raw in zip(("experiences_json", "education_json"), columns, strict=True):
                try:
                    items = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    continue
                if not isinstance(items, list):
                    continue
                changed = False
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    for key in ("start_date", "end_date"):
                        if item.get(key) and fixed(item[key]) != item[key]:
                            item[key] = fixed(item[key])
                            changed = True
                if changed:
                    updates[column] = json.dumps(items)
            for column, value in updates.items():
                conn.execute(
                    text(f"UPDATE resumes SET {column} = :v WHERE id = :id"),
                    {"v": value, "id": resume_id},
                )
        conn.commit()


# (table, column, old global unique index, new per-account unique index)
_PER_ACCOUNT_UNIQUE = (
    ("repositories", "github_id", "ix_repositories_github_id", "uq_repositories_account_github_id"),
    ("repositories", "full_name", "ix_repositories_full_name", "uq_repositories_account_full_name"),
    (
        "job_postings",
        "content_hash",
        "ix_job_postings_content_hash",
        "uq_job_postings_account_content_hash",
    ),
)


def _migrate_per_account_unique_indexes(engine: Engine) -> None:
    """Turn the global unique indexes on repositories.github_id/full_name
    and job_postings.content_hash into per-account ones, so two accounts
    can hold the same repo or posting as separate rows. Older databases
    carry these as unique indexes (unique=True, index=True), not inline
    UNIQUE constraints, so no table rebuild is needed: each one is dropped
    and recreated as a plain index, then the (account_id, column) unique
    index is added. Row ids and every foreign key are untouched. Runs in
    one explicit transaction, since the sqlite3 driver would otherwise
    commit each DDL statement on its own. Idempotent: an index that is
    already non-unique is left alone and the new ones use IF NOT EXISTS.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.connect() as conn:
        conn.exec_driver_sql("BEGIN")
        for table, column, old_index, new_index in _PER_ACCOUNT_UNIQUE:
            is_unique = {
                row[1]: bool(row[2]) for row in conn.execute(text(f"PRAGMA index_list({table})"))
            }
            if is_unique.get(old_index):
                conn.execute(text(f"DROP INDEX {old_index}"))
                conn.execute(text(f"CREATE INDEX {old_index} ON {table} ({column})"))
            conn.execute(
                text(
                    f"CREATE UNIQUE INDEX IF NOT EXISTS {new_index} "
                    f"ON {table} (account_id, {column})"
                )
            )
        conn.commit()


# Tables an older database may still carry that no model declares any more.
# auth_sources held encrypted job-site logins, so it is dropped rather than
# left on disk; the other two were never written to.
_RETIRED_TABLES = ("auth_sources", "detections", "match_results")


def _drop_retired_tables(engine: Engine) -> None:
    with engine.connect() as conn:
        for table in _RETIRED_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        conn.commit()


def init_db() -> None:
    _backup_sqlite_file(get_settings().database_url)
    engine = get_engine()
    Base.metadata.create_all(engine)
    _migrate_accounts_contact_columns(engine)
    _migrate_repositories_curation_columns(engine)
    _migrate_resumes_name_column(engine)
    _migrate_job_postings_account_id(engine)
    _migrate_resumes_build_columns(engine)
    _migrate_resumes_history_columns(engine)
    _migrate_resumes_contact_column(engine)
    _migrate_job_postings_extraction_columns(engine)
    _migrate_llm_calls_attribution_columns(engine)
    _migrate_rate_limit_events_account_column(engine)
    _migrate_job_postings_tracking_columns(engine)
    _migrate_job_postings_salary_columns(engine)
    _migrate_job_postings_source_columns(engine)
    _migrate_contact_items(engine)
    _migrate_api_keys_exhaustion_columns(engine)
    _migrate_skills_source_resume_column(engine)
    _migrate_repositories_profile_readme_column(engine)
    _migrate_repositories_readme_sha_column(engine)
    _migrate_exclude_from_resume_columns(engine)
    _migrate_resumes_build_state_column(engine)
    _migrate_education_extras_columns(engine)
    _migrate_month_year_date_column_types(engine)
    _migrate_month_year_dates(engine)
    _migrate_per_account_unique_indexes(engine)
    _drop_retired_tables(engine)
