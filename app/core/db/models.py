"""Every ORM model in the app, one table per class."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, _now


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
    # The latest uploaded resume's file, mirrored from the resumes table so
    # "has a resume" is a column read. Resume rows are the real record.
    resume_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    resume_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Resume header details, filled in over time rather than at signup.
    contact_email: Mapped[str | None] = mapped_column(String, nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String, nullable=True)
    contact_location: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # Sent by the signup form, one per filled-in form, so the same form
    # sent twice (a double click, a retry after a dropped connection)
    # answers with the profile it already made.
    request_id: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)


class SyncSource(Base):
    """A GitHub user or single repo an account pulls evidence from. Separate
    from Account.github_username, which is identity: commits are always
    credited to that login, so a repo the person contributed to but does
    not own (kind="repo") still counts as their work. One "user" source is
    created at signup. Parsed by app/ingest/github/source_parser.py.
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


class GitHubSyncRun(Base):
    """Latest outcome of syncing one GitHub target (an account or a single
    repo), one row per target. Lets a sync that GitHub cut off part way
    ("10 of 23 saved") be remembered across page loads and restarts, shown
    as a reminder on every page, and resumed, by the person or on its own
    once GitHub's hourly limit resets (app/ingest/github/background.py).

    `key` is the background job key (github-sync:user:<name> or
    github-sync:repo:<owner/name>). `state` is running | done |
    rate_limited | cancelled | error | dismissed. `completed` counts repos
    saved so far, carried over across resumes so a resumed run reads
    10 -> 20 instead of starting from 0. `resume_at` is set only while an
    automatic resume is scheduled.
    """

    __tablename__ = "github_sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String, unique=True, index=True)
    account_id: Mapped[int | None] = mapped_column(
        ForeignKey("accounts.id"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String)  # "user" | "repo"
    github_username: Mapped[str] = mapped_column(String)
    repo_full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    # who commits are credited to: the account's own login
    attribution_username: Mapped[str] = mapped_column(String)
    state: Mapped[str] = mapped_column(String, default="running")
    completed: Mapped[int] = mapped_column(Integer, default=0)
    total_hint: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    reset_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resume_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


def is_profile_repo(full_name: str) -> bool:
    """True for "owner/owner": the repo whose README GitHub shows on the
    owner's profile page. Case-insensitive, as GitHub logins are. Sync,
    extraction and the startup migration all decide is_profile_readme with
    this, so a repo saved before the flag existed is recategorized by
    whichever of them reaches it first."""
    owner, _, name = full_name.partition("/")
    return bool(name) and owner.lower() == name.lower()


class Repository(Base):
    """One GitHub repo (or hand-added project) as seen by one account.
    github_id and full_name are unique per account, not globally: two
    profiles that sync the same repo each get their own row and their own
    evidence.
    """

    __tablename__ = "repositories"
    # Unique indexes, not UniqueConstraint: an index can be added to an
    # existing table, so an upgraded database ends up with the same schema
    # as a fresh one (migrations.py's _migrate_per_account_unique_indexes).
    __table_args__ = (
        Index("uq_repositories_account_github_id", "account_id", "github_id", unique=True),
        Index("uq_repositories_account_full_name", "account_id", "full_name", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"), nullable=True)
    github_id: Mapped[int] = mapped_column(Integer, index=True)
    name: Mapped[str] = mapped_column(String)
    full_name: Mapped[str] = mapped_column(String, index=True)
    url: Mapped[str] = mapped_column(String)
    is_fork: Mapped[bool] = mapped_column(default=False)
    primary_language: Mapped[str | None] = mapped_column(String, nullable=True)
    stars: Mapped[int] = mapped_column(Integer, default=0)
    readme: Mapped[str | None] = mapped_column(Text, nullable=True)
    # GitHub's "About" line; the extraction source when there is no README.
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    manifests_json: Mapped[dict] = mapped_column(JSON, default=dict)
    commits_authored: Mapped[int] = mapped_column(Integer, default=0)
    last_commit_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Unchanged since the last sync means README, manifests and stats are not refetched.
    pushed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Git blob SHA of the stored README. An unchanged SHA in the root listing
    # means the README did not change, so it is neither downloaded nor sent
    # back to the LLM.
    readme_sha: Mapped[str | None] = mapped_column(String, nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    # Skill extraction, tracked per repo so one failure does not lose the
    # rest of a batch:
    #   pending       not attempted, or the README changed since
    #   extracted     done
    #   failed        this repo's call errored (skill_extraction_error)
    #   no_signal     no README and no description, skipped on purpose
    #   rate_limited  provider limit or budget hit; build.py stops the batch,
    #                 since every later repo would fail the same way
    # failed and rate_limited are retried on the next run.
    skill_extraction_status: Mapped[str] = mapped_column(String, default="pending")
    skill_extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    skills_extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # "Lead with this project" on a resume. At most 3 per account, enforced
    # in app/api/projects.py. Older databases may still carry an unused
    # `rating` column.
    starred: Mapped[bool] = mapped_column(default=False)

    # The owner's GitHub profile README: the repo named after its owner
    # ("hi/hi"), whose README GitHub shows on the profile page itself. It
    # is not a project, so every project list, count and resume candidate
    # filters it out, but it goes through skill extraction like any repo
    # (people list their stack there) and is shown on the portfolio
    # overview instead. Set by app/ingest/github/sync.py on every upsert.
    is_profile_readme: Mapped[bool] = mapped_column(default=False)

    # Kept on the Projects page but left out of resume building entirely:
    # no candidate list, no picker, no skill suggestion drawn only from it.
    # For a small utility repo that is real work but not resume material.
    # Sync never touches it, so it survives every re-fetch.
    exclude_from_resume: Mapped[bool] = mapped_column(default=False)


class ProjectLink(Base):
    """A link on a project (demo video, live site, docs). source is
    "readme_extracted" for one the README extraction found and "manual"
    otherwise. Reprocess replaces only extracted links, and editing an
    extracted link makes it manual, so a correction is never overwritten.
    """

    __tablename__ = "project_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    repo_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), index=True)
    # Free text: "YouTube Video", "Live Demo", "Documentation", ...
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
    """A contact link (LinkedIn, GitHub, a personal site). platform is free
    text so a new kind of link needs no migration; label is only used when
    platform is "other".
    """

    __tablename__ = "social_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    # "linkedin" | "github" | "instagram" | "website" | "other"
    platform: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ContactEmail(Base):
    """A contact email. The primary one goes on generated resumes."""

    __tablename__ = "contact_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    email: Mapped[str] = mapped_column(String)
    is_primary: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ContactPhone(Base):
    """A contact phone number. The primary one goes on generated resumes."""

    __tablename__ = "contact_phones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    phone: Mapped[str] = mapped_column(String)
    is_primary: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Experience(Base):
    """A role the person held, entered by hand or read from an uploaded
    resume. end_date left null means "current role".

    There is no free-text description: a paragraph is one blob that
    resume-building search can only take or leave whole, so detail lives
    in ExperiencePoint rows, each one its own retrievable unit.
    """

    __tablename__ = "experiences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    title: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    # "mar 2026", or "2026" when only the year is known (app/profile/month_year.py).
    start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    end_date: Mapped[str | None] = mapped_column(String, nullable=True)
    # Archived: kept in the profile, never used in resume building.
    exclude_from_resume: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ExperienceSkillEvidence(Base):
    """SkillEvidence for a role instead of a repo. A separate table because
    SkillEvidence.repo_id is NOT NULL, and SQLite cannot relax that without
    rebuilding the table. evidence_type is "manual" or "resume".
    """

    __tablename__ = "experience_skill_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    skill: Mapped[str] = mapped_column(String, index=True)
    experience_id: Mapped[int] = mapped_column(ForeignKey("experiences.id"))
    evidence_type: Mapped[str] = mapped_column(String, default="manual")
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    # Unused; keeps the shape identical for app/profile/evidence.py.
    source_files_json: Mapped[list] = mapped_column(JSON, default=list)
    manual_override: Mapped[str | None] = mapped_column(String, nullable=True)


class ExperiencePoint(Base):
    """One bullet under a role. Each point is embedded on its own
    (app/retrieval/index.py), so resume building can pick the ones that fit
    a job instead of a whole paragraph. order_index is insertion order.
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
    """A degree. end_date left null means "in progress". Every entry that
    is not archived goes on a generated resume unchanged; the LLM never
    picks or trims education. grade ("CGPA 8.9/10") and details
    (coursework, honors) are optional and take no space when empty.
    """

    __tablename__ = "education"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    institution: Mapped[str] = mapped_column(String)
    degree: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    # "mar 2026", or "2026" when only the year is known (app/profile/month_year.py).
    start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    end_date: Mapped[str | None] = mapped_column(String, nullable=True)
    grade: Mapped[str | None] = mapped_column(String, nullable=True)
    details: Mapped[list] = mapped_column(JSON, default=list)
    # Archived: kept on the page, never on a resume or in the counts.
    exclude_from_resume: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Skill(Base):
    """A skill nothing in the profile demonstrates yet, added by hand or
    read from a resume. Skills with evidence are derived from the evidence
    tables at read time (GET /api/skills) and never stored here.
    """

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    # The resume that added this skill, null when typed in by hand. Cleared,
    # not cascaded, when that resume is deleted.
    source_resume_id: Mapped[int | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("account_id", "name", name="uq_skills_account_name"),)


class ResumeProfileLink(Base):
    """A profile row that merging a resume created or matched
    (app/profile/resume_profile_merge.py), so a resume can answer "what in
    my profile came from this file?" without re-reading it.

    kind names the table ref_id points into: experience, experience_point,
    experience_skill, education or skill. created is false when the row
    already existed. A plain id rather than a foreign key per kind, so
    unlink_profile_rows() cleans these up wherever a target is deleted.
    """

    __tablename__ = "resume_profile_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    resume_id: Mapped[int] = mapped_column(ForeignKey("resumes.id"), index=True)
    kind: Mapped[str] = mapped_column(String)
    ref_id: Mapped[int] = mapped_column(Integer)
    created: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("resume_id", "kind", "ref_id", name="uq_resume_profile_links_ref"),
    )


class SkillStar(Base):
    """A starred skill. Skills are a union computed at read time, so a star
    is keyed by account and casefolded name rather than a row id; it comes
    back if the skill disappears and returns.
    """

    __tablename__ = "skill_stars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name_key: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("account_id", "name_key", name="uq_skill_stars_account_name_key"),
    )


class SkillArchive(Base):
    """An archived skill: kept on the Skills page under Archived, but left
    out of resume building, the skill map, the portfolio counts and job
    analytics' "have it" check. Keyed by account + casefolded name for the
    same reason as SkillStar, which this mirrors.
    """

    __tablename__ = "skill_archives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name_key: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("account_id", "name_key", name="uq_skill_archives_account_name_key"),
    )


class SkillVerdict(Base):
    """Whether a skill name belongs on this account's skills list, decided
    once and remembered (see app/profile/skill_review.py). verdict is
    "approved" or "rejected"; decided_by is "llm" (batch review of names
    found automatically) or "user" (adding the skill by hand, which always
    approves). Keyed like SkillStar: account + casefolded name.
    """

    __tablename__ = "skill_verdicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name_key: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    verdict: Mapped[str] = mapped_column(String)
    decided_by: Mapped[str] = mapped_column(String)
    decided_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint("account_id", "name_key", name="uq_skill_verdicts_account_name_key"),
    )


class SkillMapCache(Base):
    """The finished 2D skill map for an account (app/profile/skill_map.py),
    stored whole because building it means loading the embedding model and
    fitting t-SNE. fingerprint covers the layout version, the model and
    each skill's embedded text; a mismatch rebuilds the row.
    """

    __tablename__ = "skill_map_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String)
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("account_id", name="uq_skill_map_cache_account"),)


class Resume(Base):
    """A resume in the account's library: an uploaded file, one this app
    generated, or both. Every extracted field stays editable, and only an
    explicit reprocess overwrites it. Uploaded files are stored under
    {resume_storage_dir}/{account_id}/{id}_{filename}
    (app/profile/resume_ingest.py).
    """

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    # Display name such as "Backend, for Acme"; the UI falls back to filename.
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    stored_path: Mapped[str] = mapped_column(String, default="")
    mime_type: Mapped[str] = mapped_column(String)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Manual, always editable, never touched by (re)extraction.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Read from the file by app/profile/resume_extract.py; editable via PATCH.
    tags_json: Mapped[list] = mapped_column(JSON, default=list)
    target_roles_json: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # What this file says about work history and education, as extracted
    # ({company, title, dates, points} / {institution, degree, dates}).
    # A snapshot of this version only: the editable copies live in the
    # Experience and Education tables, and only a reprocess rewrites these.
    experiences_json: Mapped[list] = mapped_column(JSON, default=list)
    education_json: Mapped[list] = mapped_column(JSON, default=list)
    # The same snapshot for the header: {name, location, emails, phones,
    # links}. Merged into the account's contact rows.
    contact_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # pending | extracted | failed (retryable) | unsupported_type (the model
    # cannot read this file type, so Retry would fail the same way).
    extraction_status: Mapped[str] = mapped_column(String, default="pending")
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # For a failed read whose cause was the AI provider: app/core/llm.py's
    # error_kind() ("provider_unavailable", "no_key", ...). None otherwise.
    extraction_error_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # A resume this app renders. content_json is what build_resume_data()
    # returns; an LLM edit may change only its summary, projects and skills,
    # never experience, education or the header. compiled_path is the PDF
    # built from it, kept apart from stored_path so an uploaded original is
    # never overwritten. job_posting_id is set when it was built for a job.
    job_posting_id: Mapped[int | None] = mapped_column(
        ForeignKey("job_postings.id"), nullable=True
    )
    template: Mapped[str | None] = mapped_column(String, nullable=True)
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    compiled_path: Mapped[str | None] = mapped_column(String, nullable=True)
    compiled_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Set while a build is unfinished (model busy, Tectonic timeout) and
    # cleared when a retry completes it: the request, any tailored content,
    # page-fit progress and the error. content_json stays null until then,
    # so nothing reading finished resumes sees half-built content.
    build_state_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )


class RoleFamily(Base):
    """A canonical job-title cluster: "ML Engineer" and "Machine Learning
    Engineer" resolve to one row, so analytics group by role rather than
    exact title. Shared across accounts.

    app/profile/role_family.py embeds a new title and reuses the nearest
    existing family when it is close enough; only a new cluster costs an
    LLM call to name it. canonical_name is unique, so two concurrent
    resolutions of near-identical titles cannot both insert.
    """

    __tablename__ = "role_families"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class JobPosting(Base):
    """A saved job posting. content_hash is unique per account: saving
    the same text twice returns the account's existing row, and another
    account saving identical text gets a row of its own.
    """

    __tablename__ = "job_postings"
    __table_args__ = (
        Index("uq_job_postings_account_content_hash", "account_id", "content_hash", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id"), nullable=True)
    # "pasted" | "screenshot" | "mixed"; older rows may also say "url" or
    # "authenticated", from when links were fetched
    source: Mapped[str] = mapped_column(String, index=True)
    external_id: Mapped[str] = mapped_column(String)
    company: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    # untrusted: never string-formatted into a prompt template
    raw_text_quarantined: Mapped[str] = mapped_column(Text)
    extracted_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, index=True)
    apply_url: Mapped[str | None] = mapped_column(String, nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Fields read from the text by app/profile/job_extract.py into
    # extracted_json (salary, seniority, skills_required as {skill, level}
    # dicts, ...). Runs on save without ever blocking it, and again on
    # POST /api/job-postings/{id}/reprocess.
    extraction_status: Mapped[str] = mapped_column(String, default="pending")
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Same as Resume.extraction_error_kind.
    extraction_error_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    extracted_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Pay as whole-number annual amounts in salary_currency, parsed from
    # extracted_json["salary_range"] (app/profile/salary.py) whenever
    # extraction runs, so postings can be filtered and sorted by pay. Either
    # bound may be null ("up to 20 LPA" has no minimum); both null means no
    # salary was stated or it didn't read as a number. Editable via PATCH.
    salary_min_annual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_max_annual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String, nullable=True)

    # Application tracking, entered by hand. Marking applied stamps today;
    # unmarking clears the date and notes.
    applied: Mapped[bool] = mapped_column(default=False)
    applied_at: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    applied_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Set after extraction when a role family resolves; null otherwise.
    role_family_id: Mapped[int | None] = mapped_column(
        ForeignKey("role_families.id"), nullable=True
    )

    # Screenshots attached to the posting, kept on disk in the order given.
    # screenshot_path is the first one, and the only record on older rows.
    screenshot_path: Mapped[str | None] = mapped_column(String, nullable=True)
    screenshot_paths: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # What was handed in when saving, kept as given: the pasted text before
    # any screenshot transcription was joined onto it, and every link found.
    # Null on rows saved before these existed.
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_links: Mapped[list | None] = mapped_column(JSON, nullable=True)


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
    # Who the call was for and which key answered it (after a failover, not
    # the first one tried). Both back /monitor's breakdowns and per-key
    # budget caps. Null for cache hits, calls with no account, and test
    # fakes. Plain ids, no foreign keys, so deleting an account or key
    # leaves the history readable.
    account_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # Which feature spent the call ("repo_facts", "resume_build", ...), so
    # /monitor can show where spend went. Null lands in "unattributed".
    purpose: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RateLimitEvent(Base):
    """Append-only log of rate limits, budget caps, keys dropped from
    rotation and stopped runs, for GitHub and the LLM. /monitor reads it to
    show how often these happen, not just whether one is happening now.
    Written best-effort through app/core/rate_limits.py.
    """

    __tablename__ = "rate_limit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # "github" | "llm"
    source: Mapped[str] = mapped_column(String, index=True)
    # "rate_limited" | "budget_exceeded"
    kind: Mapped[str] = mapped_column(String, index=True)
    detail: Mapped[str] = mapped_column(Text)
    # What was being called: a model name, endpoint or username.
    context: Mapped[str | None] = mapped_column(String, nullable=True)
    # Only LLM events have one; GitHub uses a single shared token.
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
    """A stored credential for one LLM provider, managed on /apis. Several
    keys per provider back each other up: is_active marks the one tried
    first, and when the provider rejects it mid-request the next takes over
    (app/core/llm.py).

    encrypted_credentials is the whole provider field dict, encrypted as
    one blob, since some providers need several fields (Bedrock, Azure).
    masked_preview is computed at save time so listing keys never decrypts.
    budget_cap_usd is an optional per-key cap on top of the global budget.
    enabled switches a key off without losing its place, and a non-empty
    allowed_account_ids restricts it to those accounts.

    status is unknown, valid, invalid, rate_limited or blocked.
    rate_limited is temporary: exhausted_at, exhaustion_kind and retry_at
    (from app/core/key_cooldown.py) say when a recheck is worth making, and
    app/core/key_refresh.py makes it. invalid and blocked are only
    rechecked on request. No status removes a key from rotation; it only
    moves it down the dispatch order, so a device with one key can still
    try it.
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
    exhausted_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exhaustion_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    budget_cap_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AppSetting(Base):
    """Device-wide user choices, one row per setting: the model for each
    LLM tier and the global monthly budget. Read and written only through
    app/core/app_settings.py, which owns the keys and their defaults. A
    missing row means "use the default", so a fresh database needs no
    seeding."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )
