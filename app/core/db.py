"""SQLite engine, session, ORM models, and lightweight column migrations.

A few tables (Detection, MatchResult) are declared ahead of the features
that will write to them.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Account(Base):
    """A local profile on this device, not a login. No password, no
    server session: the client remembers which account it's using
    (localStorage) and passes account_id with requests that need it.
    Multiple people can share one instance/device by picking their own
    account on the first screen; each account's data is scoped by
    `account_id` on the tables below.
    """

    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String)
    last_name: Mapped[str] = mapped_column(String)
    github_username: Mapped[str] = mapped_column(String, index=True)
    # Legacy single-file mirror: the most recently ingested Resume row's
    # filename/path (app/profile/resume_ingest.py keeps these in sync on
    # every upload, signup or from /resume). Real resume data now lives in
    # the `resumes` table below, this pair only exists so AccountSummary.
    # has_resume stays a cheap column read instead of a join, and so an
    # older deployment upgrading in place doesn't lose the file it already
    # collected at signup.
    resume_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    resume_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Contact info, added via _migrate_accounts_contact_columns() below
    # rather than a fresh table, since this is data ABOUT the account
    # itself (resume header material), same tier as first/last name.
    # Nullable: collected progressively, not required at signup.
    contact_email: Mapped[str | None] = mapped_column(String, nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String, nullable=True)
    contact_location: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class SyncSource(Base):
    """One thing to fetch GitHub data from, for a given account, kept
    separate from Account.github_username. That field is identity (who this
    profile is, used for commit attribution); a SyncSource is purely
    "where do we pull evidence from," and an account can have any number of
    them: their own username, a second GitHub account, or a single repo they
    contributed to but don't own (kind="repo": commit attribution then
    credits Account.github_username, not the repo's owner, since it's a
    project someone else's account holds but this person worked on).

    One is auto-created at account signup (kind="user", the signup
    username) so the fetch-data page isn't empty on first visit; from there
    it's just one row among any others the person adds.

    Parsed by app/ingest/github/source_parser.py from whatever raw text
    they paste: bare username, profile URL, or repo URL.
    """

    __tablename__ = "sync_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    raw_input: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)  # "user" | "repo"
    github_username: Mapped[str] = mapped_column(String)
    # "owner/repo", set only when kind == "repo"
    repo_full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_synced_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"), nullable=True)
    github_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    full_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    url: Mapped[str] = mapped_column(String)
    is_fork: Mapped[bool] = mapped_column(default=False)
    primary_language: Mapped[str | None] = mapped_column(String, nullable=True)
    stars: Mapped[int] = mapped_column(Integer, default=0)
    readme: Mapped[str | None] = mapped_column(Text, nullable=True)
    # GitHub's short "About" one-liner: repo metadata, not repo content;
    # comes free with the repo-list call, no extra API request. Fallback
    # extraction source when there's no README, per app/profile/extract.py.
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    manifests_json: Mapped[dict] = mapped_column(JSON, default=dict)
    commits_authored: Mapped[int] = mapped_column(Integer, default=0)
    last_commit_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # caching: skip refetch of readme/manifests/stats when unchanged
    pushed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    # Per-repo skill-extraction status: lets a mid-batch LLM failure (auth
    # revoked, malformed input) fail one repo without losing progress on
    # every other repo, and gives the UI something to show a retry button
    # against. "pending": never attempted (or pushed_at changed since last
    # attempt, on sync.py's cache-miss path). "extracted": succeeded.
    # "failed": LLM call raised for reasons specific to this repo, see
    # skill_extraction_error. "no_signal": no README and no description,
    # skipped by design, not a failure (see extract.py's NoSourceTextError).
    # "rate_limited": the provider rate limit or the budget cap was hit.
    # Distinct from "failed": this repo didn't do anything wrong, and every
    # repo after it in the same batch would fail the same way right now, so
    # build.py stops the batch here instead of continuing (see build.py's
    # module docstring). Retried on the next process-pending call, same as
    # "pending"/"failed".
    #
    # Both "failed" and "rate_limited" are retried on the next non-forced
    # build_profile()/process-pending call, neither is in _SKIP_STATUSES.
    skill_extraction_status: Mapped[str] = mapped_column(String, default="pending")
    skill_extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills_extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Manual curation, optional and independent of sync/extraction: "which
    # of my projects should a resume lead with." Capped at 3 per account,
    # enforced in app/api/projects.py (not here: the ORM layer doesn't see
    # "this account's other repos" without a query). Feeds project
    # ordering, not skill extraction. (A 1-10 `rating` column used to sit
    # beside this; it was dropped from the model as too much manual work.
    # Older databases still carry the unused column, nothing reads it.)
    starred: Mapped[bool] = mapped_column(default=False)


class ProjectLink(Base):
    """One outbound link on a project: GitHub, a YouTube demo video, a live
    deployed version, docs, etc. `platform` is free-form on purpose (same
    reasoning as SocialLink.platform above), not an enum: a project can have
    any number of these, with whatever label fits.

    `source` distinguishes a link a person typed in by hand from one the
    first-pass README extraction found (app/profile/extract.py's
    extract_links_from_repo), mirrors SkillEvidence.evidence_type's
    manual-vs-derived split, for the same reason: Reprocess deletes and
    re-derives "readme_extracted" rows without touching "manual" ones (see
    build.py's _process_repo), and editing an extracted link's label/url
    flips it to "manual" so a later Reprocess can't silently overwrite
    someone's correction.
    """

    __tablename__ = "project_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), index=True)
    # e.g. "GitHub", "YouTube Video", "Live Demo", "Documentation", free text,
    # not an enum, same reasoning as SocialLink.platform.
    label: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    # "manual" | "readme_extracted"
    source: Mapped[str] = mapped_column(String, default="manual")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SkillEvidence(Base):
    __tablename__ = "skill_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    skill: Mapped[str] = mapped_column(String, index=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"))
    evidence_type: Mapped[str] = mapped_column(String)
    weight: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    source_files_json: Mapped[list] = mapped_column(JSON, default=list)
    manual_override: Mapped[str | None] = mapped_column(String, nullable=True)


class SocialLink(Base):
    """One contact-adjacent link for an account: LinkedIn, a second GitHub
    profile (separate from Account.github_username, which is sync
    identity, not a display link), Instagram, a personal site, etc.
    `platform` is free-form on purpose (not an enum column) so a new kind
    of link never needs a migration; `label` is only meaningful when
    platform == "other", for a link that doesn't fit the common set.
    """

    __tablename__ = "social_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    # "linkedin" | "github" | "instagram" | "website" | "other"
    platform: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experience(Base):
    """A job/role, entered manually, there's no GitHub-shaped source to
    sync this from, so unlike Repository there's no is_manual flag or
    synthetic-id trick: every row here is manual. account_id is NOT NULL
    (unlike Repository/Profile's nullable account_id, which is nullable
    only for pre-account-era compatibility this table has no need for).
    end_date left null means "current role", no separate is_current flag,
    one source of truth for the same fact.

    No free-text `description` field on purpose (dropped in the same pass
    that added ExperiencePoint's Qdrant indexing, see
    scripts/migrate_experience_description_to_points.py for the one-time
    backfill). A single paragraph is one blob that resume-building semantic
    search can only take or leave whole. ExperiencePoint rows are the only
    place detail goes, each one its own retrievable unit.
    """

    __tablename__ = "experiences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    title: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    start_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ExperienceSkillEvidence(Base):
    """SkillEvidence's exact shape, FK'd to Experience instead of
    Repository. Kept as a separate table rather than widening
    SkillEvidence itself: SkillEvidence.repo_id is NOT NULL and SQLite
    can't relax a NOT NULL constraint without a full table rebuild, which
    would put existing synced data at risk. evidence_type is "manual" only
    for now (no LLM extraction from experience text yet), so there is no
    analog to "failed"/"rate_limited"/"no_signal".
    """

    __tablename__ = "experience_skill_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    skill: Mapped[str] = mapped_column(String, index=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"))
    evidence_type: Mapped[str] = mapped_column(String, default="manual")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    # unused today, kept so this row shares SkillEvidence's exact shape,
    # see app/profile/evidence.py, the helper both tables' routers share.
    source_files_json: Mapped[list] = mapped_column(JSON, default=list)
    manual_override: Mapped[str | None] = mapped_column(String, nullable=True)


class ExperiencePoint(Base):
    """One bullet point under an Experience: "increased X", "led team of
    Y." This is the only place experience detail lives (Experience has
    no description field, see its docstring). Every row here is its own
    unit, embedded and upserted into Qdrant on create/update
    (app/retrieval/index.py's index_experience_points, wired from
    app/api/experience.py) so a resume-building LLM pass can semantically
    search across a person's whole point history and pull back whichever
    ones actually match a given job, instead of being handed one
    take-it-or-leave-it paragraph. More points, covering more angles
    (a shipped feature, a leadership moment, a metric-backed win), means
    more get found later. Encouraged in the UI, not enforced here.
    order_index is append-only (set to current max+1 on create); there's
    no reorder endpoint yet, so it only ever reflects insertion order.
    """

    __tablename__ = "experience_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Education(Base):
    """A school/degree entry, entered manually, same posture as Experience:
    no GitHub-shaped source to sync this from, every row is manual,
    account_id is NOT NULL. No points sub-table like Experience has: a
    degree line doesn't split into independently retrievable units the way
    job-history detail does. end_date left null means "in progress" (an
    expected graduation date is start_date's counterpart, not this),
    same one-source-of-truth reasoning as Experience.end_date.

    Included in a generated resume the same way Experience is: full,
    unconditional, every row, sequential, never picked or trimmed by the
    LLM (app/resume_build/context.py's build_education_context). A
    person's own degree history isn't something semantic search gets to
    curate any more than a job history is.
    """

    __tablename__ = "education"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    institution: Mapped[str] = mapped_column(String)
    degree: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    start_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Skill(Base):
    """A skill with NO evidence link at all, added by hand, not backed by
    any project or experience. A skill that DOES have evidence is never
    rowed here; it's derived at read time by GET /api/skills from
    SkillEvidence + ExperienceSkillEvidence. This table exists purely so
    "I have this skill but nothing in here demonstrates it yet" has
    somewhere to live.
    """

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("account_id", "name", name="uq_skills_account_name"),)


class SkillStar(Base):
    """A starred skill. Skills are a display-time union (see
    app/api/skills.py), so there's no single skill row to put a `starred`
    flag on; instead a star is keyed the same way GET /api/skills groups:
    account + name.strip().casefold(). A star whose skill later disappears
    (evidence deleted) is simply never matched, and comes back if the
    skill does.
    """

    __tablename__ = "skill_stars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name_key: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("account_id", "name_key", name="uq_skill_stars_account_name_key"),
    )


class Resume(Base):
    """One uploaded resume file for an account. Any number per account:
    a later job-application flow picks one (or, further out, an LLM
    builds a custom one) from this list rather than the account having
    exactly one "the" resume. Every field below stays manually editable
    after upload, whether or not extraction ever ran, same "extraction
    produces a starting point, not a lock" pattern as SkillEvidence/
    ProjectLink elsewhere in this file: editing tags/target_roles/summary
    by hand does not get silently overwritten except by an explicit
    reprocess.

    Distinct from Account.resume_filename/resume_path, kept only as a
    mirror of the most recent upload for backward compatibility (see that
    field's docstring); nothing reads from those two columns to decide
    what resumes exist, only this table does.

    Storage: one file on disk per row, under
    `{resume_storage_dir}/{account_id}/{id}_{original filename}` (the id
    prefix keeps two uploads of, say, "resume.pdf" for the same account
    from colliding), written and cleaned up by
    app/profile/resume_ingest.py.
    """

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    # Person-chosen label, e.g. "Backend, 2026 batch", "for Acme". Separate
    # from filename: the file's own name is often something generic like
    # "resume.pdf" or "resume (3).pdf", not something anyone can tell
    # versions apart by at a glance. Nullable, stays unset until someone
    # types one in; the UI falls back to filename when it's empty, this
    # column is never auto-filled from it.
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    stored_path: Mapped[str] = mapped_column(String, default="")
    mime_type: Mapped[str] = mapped_column(String)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Manual, always editable, never touched by (re)extraction.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LLM-extracted (multimodal, see app/profile/resume_extract.py) on
    # upload and on demand via POST /api/resume/{id}/reprocess; each field
    # independently editable afterward through PATCH /api/resume/{id}.
    tags_json: Mapped[list] = mapped_column(JSON, default=list)
    target_roles_json: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # "pending": not yet run. "extracted": succeeded, the three fields
    # above reflect it. "failed": the LLM call itself errored (rate limit,
    # bad response), retryable. "unsupported_type": the file's mime type
    # isn't one app/core/llm.py's multimodal helpers can read today (only
    # PDF and common image types), not retryable without a different
    # upload, kept distinct from "failed" so the UI can say why instead of
    # offering a Retry that will just fail the same way again.
    extraction_status: Mapped[str] = mapped_column(String, default="pending")
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Structured content + compiled artifact, added for the AI-editable
    # resume library (app/resume_build/orchestrator.py's
    # edit_resume_content). Distinct from the extraction fields above,
    # which describe an *uploaded* file; these describe a resume this app
    # itself renders. job_posting_id is set for a resume that originated
    # from POST /api/resume-build/generate (see app/api/resume_build.py),
    # left null for one that started as a plain upload and, if ever, was
    # later "adopted" (see content_json below). template/content_json are
    # both null until a compiled version exists at all. content_json is
    # the same header/summary/experience/projects/education/technologies/
    # skills dict build_resume_data() returns: the one and only piece of
    # this row an LLM edit is allowed to touch is content_json's
    # summary/projects/skills, never experience/education/header, see
    # that module's docstring. compiled_path/compiled_at are the PDF
    # rendered from content_json, kept separate from stored_path (the
    # original upload, if this row started as one, never overwritten by
    # an edit) so "layout unchanged, only facts" holds literally: the
    # original file a person uploaded is never touched.
    job_posting_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_postings.id"), nullable=True
    )
    template: Mapped[str | None] = mapped_column(String, nullable=True)
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    compiled_path: Mapped[str | None] = mapped_column(String, nullable=True)
    compiled_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    skills_json: Mapped[dict] = mapped_column(JSON, default=dict)

    @property
    def skills(self) -> list[dict]:
        """Aggregated per-skill dicts computed into `skills_json` by
        `app.profile.build`. SkillEvidence rows key off `repo_id`, not
        `profile_id` (no such FK in this schema), so this is not a list of
        raw ORM rows.
        """
        return [{"skill": name, **data} for name, data in (self.skills_json or {}).items()]


class RoleFamily(Base):
    """A canonical job-title cluster: "ML Engineer", "Machine Learning
    Engineer", "Applied ML Engineer" all resolve to one row here, so
    analytics (app/api/job_analytics.py) roll up by what a role actually
    is, not by exact title string. Global, not per-account, same posture
    as JobPosting.content_hash's shared-cache reasoning: a title's
    canonical family doesn't depend on which local profile pasted it.

    Resolution (app/profile/role_family.py) is retrieval-first, not an LLM
    guess every time: a new title is embedded and searched against the
    `role_families` Qdrant collection (index_role_family in
    app/retrieval/index.py); a close-enough existing row is reused, and
    only a new cluster costs one cheap bulk-tier LLM call to
    produce a clean canonical name. canonical_name is unique so two
    concurrent "no match found" resolutions for near-identical titles
    can't both insert; the loser's IntegrityError is caught and re-reads
    the winner's row instead (see role_family.py).
    """

    __tablename__ = "role_families"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class JobPosting(Base):
    """content_hash is globally unique on purpose, not per-account: a real
    job posting's text is the same regardless of which local account
    pastes or fetches it, so this table is a shared content cache, not a
    per-account list. account_id (added via _migrate_job_postings_account_id,
    see below) records whichever account first created a given row; a
    second account pasting byte-identical text gets that same row back
    (app/api/job_postings.py's create_posting) rather than a duplicate, but
    then won't see it in their own GET /api/job-postings?account_id= list
    since ownership didn't transfer. This edge case (two accounts pasting
    identical text) is accepted rather than solved with a composite unique
    constraint, which would need a table-recreating migration.
    """

    __tablename__ = "job_postings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"), nullable=True)
    # "pasted" | "url" | "screenshot" | "authenticated"
    source: Mapped[str] = mapped_column(String, index=True)
    external_id: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    # untrusted: never string-formatted into a prompt template
    raw_text_quarantined: Mapped[str] = mapped_column(Text)
    extracted_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, unique=True, index=True)
    apply_url: Mapped[str | None] = mapped_column(String, nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Structured extraction (app/profile/job_extract.py) of
    # raw_text_quarantined into extracted_json: salary_range,
    # employment_type, seniority, experience_required, skills_required
    # (now a list of {"skill", "level"} dicts, level in ""/junior/mid/
    # senior/expert, see job_extract.py; an older row's list[str]
    # shape is still read correctly, see JobPostingSummary.from_posting),
    # other_requirements, role_summary. Same three-column status pattern
    # as Resume's own extraction_status/extraction_error/extracted_at
    # above it in this file, run once on create (best-effort, a failed
    # extraction never blocks saving the posting itself) and again on
    # demand via POST /api/job-postings/{id}/reprocess.
    extraction_status: Mapped[str] = mapped_column(String, default="pending")
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Application tracker: a plain manual record, never touched by
    # extraction/reprocess. applied_at defaults to the moment it's marked,
    # but stays editable (PATCH) for someone logging an application after
    # the fact. Unmarking clears both, one source of truth, same
    # null-means-unset pattern as Experience.end_date elsewhere in this file.
    applied: Mapped[bool] = mapped_column(default=False)
    applied_at: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    applied_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Canonical role cluster (app/profile/role_family.py), set best-effort
    # right after a successful title extraction, same posture as every
    # other post-extraction enrichment in this file (never blocks saving
    # the posting itself). Null until resolved, or if resolution failed.
    role_family_id: Mapped[int | None] = mapped_column(
        ForeignKey("role_families.id"), nullable=True
    )

    # Set only for source == "screenshot": the uploaded image, kept on disk
    # (same per-account storage convention as Resume.stored_path) so the
    # original can be viewed later, distinct from raw_text_quarantined
    # (the LLM's own transcription of it, used everywhere text is needed).
    screenshot_path: Mapped[str | None] = mapped_column(String, nullable=True)


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    posting_id: Mapped[int] = mapped_column(ForeignKey("job_postings.id"))
    kind: Mapped[str] = mapped_column(String)
    span: Mapped[str] = mapped_column(String)
    snippet: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    posting_id: Mapped[int] = mapped_column(ForeignKey("job_postings.id"))
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"))
    score: Mapped[float] = mapped_column(Float)
    gaps_json: Mapped[list] = mapped_column(JSON, default=list)
    bullets_json: Mapped[list] = mapped_column(JSON, default=list)
    citations_json: Mapped[list] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tier: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    prompt_hash: Mapped[str] = mapped_column(String, index=True)
    # The response itself, so a repeated call can be served from here without
    # re-dispatching. `cached` on a row means "this row's response_json was
    # reused", not "this row is a cache".
    response_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cached: Mapped[bool] = mapped_column(default=False)
    # Attribution, added for the /monitor per-key/per-account usage
    # breakdown (app/api/monitor.py). Both nullable: a cache hit or a
    # call with no account context (account_id=None passed to complete())
    # legitimately has no account; key_id is only set on a real dispatch
    # that went through app/core/api_keys_store.resolve_dispatch_key(); a
    # test's injected _completion_fn never resolves a key, so key_id stays
    # NULL for mocked calls and for rows written before this column existed.
    # No FK constraint to api_keys/accounts (informational ids, not
    # enforced), so a since-deleted account or key doesn't break old rows.
    account_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RateLimitEvent(Base):
    """One row per time an outbound call got rate-limited or budget-capped:
    GitHub (app/ingest/github/client.py) or the LLM provider / our own
    monthly cap (app/core/llm.py). Distinct from Repository.
    skill_extraction_status == "rate_limited": that field is per-repo,
    transient, and overwritten on the next retry, this table is an
    append-only log, so the monitor page (app/api/monitor.py) can show
    "how often has this actually happened" instead of only "is it
    happening right now". Written via app/core/rate_limits.py's
    record_event(), which is best-effort (a logging failure here must
    never break the caller's real request).
    """

    __tablename__ = "rate_limit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # "github" | "llm"
    source: Mapped[str] = mapped_column(String, index=True)
    # "rate_limited" | "budget_exceeded"
    kind: Mapped[str] = mapped_column(String, index=True)
    detail: Mapped[str] = mapped_column(Text)
    # free-form: model name, endpoint, username, whatever identifies what
    # was being called when this happened. Nullable: not every call site
    # has something meaningful to put here.
    context: Mapped[str | None] = mapped_column(String, nullable=True)
    # Which profile this happened for, when known. Only the LLM side
    # (app/core/llm.py) ever has one to give; app/ingest/github/client.py
    # has no per-account credential concept (one shared GITHUB_TOKEN), so
    # its events keep this NULL. Same informational-id, no-FK pattern as
    # LLMCall.account_id above.
    account_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, index=True
    )


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String, index=True)
    model: Mapped[str] = mapped_column(String)
    vector_json: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("content_hash", "model", name="uq_embedding_cache_hash_model"),
    )


class ApiKey(Base):
    """One stored credential set for one LLM provider, managed from the
    /apis page (app/api/api_keys.py). More than one row per `provider` is
    allowed (rotation, labeled keys for different budgets, etc):
    `is_active` marks the single row app/core/llm.py dispatches with for
    that provider at any given time. At most one active per provider is
    enforced in app/core/api_keys_store.py rather than with a partial
    unique index, which isn't worth it for a single-process local app.

    `encrypted_credentials` is the WHOLE provider-shaped field dict (see
    PROVIDERS in app/core/llm_providers.py) JSON-encoded then encrypted as
    one blob, never split field-by-field. Some providers need more than
    one secret (AWS Bedrock: access key + secret key; Azure OpenAI: key +
    endpoint + api version), and blob-encoding means a provider gaining a
    field later never needs a schema migration here.

    `masked_preview` is a JSON dict of that same shape with every secret
    field pre-masked (first 3 / last 3 chars, dots between) and non-secret
    config fields (api_base, deployment, region, ...) shown in full,
    computed once at save time so listing keys never decrypts anything.

    `budget_cap_usd` is optional and per-key, separate from (and checked
    in addition to) Settings.monthly_budget_usd's existing global cap.

    `enabled` is a manual on/off switch, distinct from `is_active`:
    disabling a key takes it out of dispatch consideration entirely (even
    if it's the active one for its provider) without losing its place;
    re-enabling it needs no re-activation. `allowed_account_ids` is an
    optional allow-list of Account ids (empty list, the default = every
    account on this device may use it, not "no one may"); a non-empty
    list restricts the key to only those accounts, checked in
    app/core/api_keys_store.py's resolve_dispatch_key() against whichever
    account_id the calling code passes into app/core/llm.py's complete().
    `status` is one of unknown|valid|invalid|rate_limited. The last one is
    set only from a real dispatch hitting a 429 (see
    record_dispatch_outcome()), never from the cheap check_key() call,
    which can't observe quota exhaustion. `last_check_detail` carries the
    human-readable outcome message (from app/core/llm_providers.py's
    validate_credentials(), or a dispatch failure) alongside
    `last_checked_at`, so the /apis page can show not just a status but
    what actually happened last time.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String, index=True)
    label: Mapped[str] = mapped_column(String)
    encrypted_credentials: Mapped[str] = mapped_column(Text)
    masked_preview: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(default=False)
    enabled: Mapped[bool] = mapped_column(default=True)
    allowed_account_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="unknown")
    last_checked_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_check_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuthSource(Base):
    """One stored login profile for fetching a job posting from a
    login-walled site (Wellfound, LinkedIn, etc) on the account holder's
    own behalf, via a real automated browser login
    (app/ingest/jobs/auth_fetch.py, Playwright). Opt-in per source:
    acknowledged_risk must be True to create a row, enforced in
    app/api/auth_sources.py, not just a UI checkbox. Meant for a person's
    own credentials on their own job search, not bulk scraping. The UI
    carries a standing warning: automated login is fragile (breaks on any
    site UI change, CAPTCHA, or 2FA), is against most sites' terms of
    service, and can get the logged-in account flagged or banned. Use an
    alternate account, never a primary one.

    `encrypted_credentials` is a JSON-encoded {"username", "password"}
    blob, encrypted the same way app/core/db.py's ApiKey stores provider
    credentials (app/core/crypto.py): never plaintext, never logged.
    The three CSS selectors let a generic Playwright driver log into an
    arbitrary site without hardcoding per-site scraping logic: the account
    holder supplies them once, by inspecting the site's login form.
    """

    __tablename__ = "auth_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    label: Mapped[str] = mapped_column(String)
    # e.g. "wellfound.com". Advisory only (shown in the UI so a target URL
    # can be checked against it before use), not enforced against the URL
    # passed to fetch_job_url_authenticated at fetch time.
    site_domain: Mapped[str] = mapped_column(String)
    login_url: Mapped[str] = mapped_column(String)
    username_selector: Mapped[str] = mapped_column(String)
    password_selector: Mapped[str] = mapped_column(String)
    submit_selector: Mapped[str] = mapped_column(String)
    # Optional: a selector Playwright waits for after submit to know login
    # actually succeeded (e.g. a nav element only shown when signed in).
    # Empty means "just wait for navigation," a weaker signal.
    post_login_wait_selector: Mapped[str | None] = mapped_column(String, nullable=True)
    encrypted_credentials: Mapped[str] = mapped_column(Text)
    masked_username: Mapped[str] = mapped_column(String)
    acknowledged_risk: Mapped[bool] = mapped_column(default=False)
    enabled: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String, default="unknown")  # unknown|valid|invalid
    last_checked_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_check_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
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


def _configure_sqlite(engine) -> None:
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
    def _set_sqlite_pragma(dbapi_connection, _):
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


def _migrate_accounts_contact_columns(engine) -> None:
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

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(accounts)"))}
        for column in ("contact_email", "contact_phone", "contact_location"):
            if column not in existing:
                conn.execute(text(f"ALTER TABLE accounts ADD COLUMN {column} TEXT"))
        conn.commit()


def _migrate_repositories_curation_columns(engine) -> None:
    """Add repositories.starred to an already-existing repositories table,
    same reasoning and same idempotent PRAGMA-check pattern as
    _migrate_accounts_contact_columns above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(repositories)"))}
        if "starred" not in existing:
            conn.execute(
                text("ALTER TABLE repositories ADD COLUMN starred BOOLEAN NOT NULL DEFAULT 0")
            )
        conn.commit()


def _migrate_resumes_name_column(engine) -> None:
    """Add resumes.name to an already-existing resumes table, same
    reasoning and pattern as _migrate_repositories_curation_columns above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(resumes)"))}
        if "name" not in existing:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN name TEXT"))
        conn.commit()


def _migrate_job_postings_account_id(engine) -> None:
    """Add job_postings.account_id to an already-existing job_postings
    table, same reasoning and pattern as _migrate_resumes_name_column
    above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(job_postings)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE job_postings ADD COLUMN account_id INTEGER"))
        conn.commit()


def _migrate_resumes_build_columns(engine) -> None:
    """Add resumes.job_posting_id/template/content_json/compiled_path/
    compiled_at to an already-existing resumes table, same reasoning and
    pattern as _migrate_resumes_name_column above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

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


def _migrate_job_postings_extraction_columns(engine) -> None:
    """Add job_postings.extraction_status/extraction_error/extracted_at to
    an already-existing job_postings table, same reasoning and pattern as
    _migrate_job_postings_account_id above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

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


def _migrate_llm_calls_attribution_columns(engine) -> None:
    """Add llm_calls.account_id/key_id to an already-existing llm_calls
    table, same reasoning and pattern as _migrate_resumes_name_column
    above. Backs the /monitor per-key/per-account usage breakdown
    (app/api/monitor.py). Rows written before this migration have both
    NULL, which the breakdown shows as an "unattributed" bucket rather than
    dropping that spend.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(llm_calls)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE llm_calls ADD COLUMN account_id INTEGER"))
        if "key_id" not in existing:
            conn.execute(text("ALTER TABLE llm_calls ADD COLUMN key_id INTEGER"))
        conn.commit()


def _migrate_rate_limit_events_account_column(engine) -> None:
    """Add rate_limit_events.account_id to an already-existing table, same
    pattern as _migrate_llm_calls_attribution_columns above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(rate_limit_events)"))}
        if "account_id" not in existing:
            conn.execute(text("ALTER TABLE rate_limit_events ADD COLUMN account_id INTEGER"))
        conn.commit()


def _migrate_job_postings_tracking_columns(engine) -> None:
    """Add job_postings.applied/applied_at/applied_notes/role_family_id/
    screenshot_path to an already-existing job_postings table, same
    reasoning and pattern as _migrate_job_postings_account_id above.
    """
    if engine.dialect.name != "sqlite":
        return

    from sqlalchemy import text

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


def get_db() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()
