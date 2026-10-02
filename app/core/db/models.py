"""Every ORM model in the app, one table per class.

A few tables (Detection, MatchResult) are declared ahead of the features
that will write to them.
"""

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
    # Git blob SHA of the stored README. A push that leaves the README
    # alone keeps the same SHA in the root listing, so the sync knows the
    # extraction source is unchanged without downloading or re-reading it,
    # and skips the LLM reprocess.
    readme_sha: Mapped[str | None] = mapped_column(String, nullable=True)
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


class ContactEmail(Base):
    """Direct contact email address for an account. An account can have
    multiple contact emails; `is_primary` indicates the primary/starred
    email used by default in resume building.
    """

    __tablename__ = "contact_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    email: Mapped[str] = mapped_column(String)
    is_primary: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ContactPhone(Base):
    """Direct contact phone number for an account. An account can have
    multiple contact phone numbers; `is_primary` indicates the primary/starred
    phone number used by default in resume building.
    """

    __tablename__ = "contact_phones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    phone: Mapped[str] = mapped_column(String)
    is_primary: Mapped[bool] = mapped_column(default=False)
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
    # "mar 2026", or "2026" when only the year is known (app/profile/month_year.py).
    start_date: Mapped[str | None] = mapped_column(String, nullable=True)
    end_date: Mapped[str | None] = mapped_column(String, nullable=True)
    # Same meaning as Repository.exclude_from_resume: the role stays in the
    # profile (and a resume naming it again still matches it, rather than
    # adding a copy), but resume building never sees it or its points.
    exclude_from_resume: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class ExperienceSkillEvidence(Base):
    """SkillEvidence's exact shape, FK'd to Experience instead of
    Repository. Kept as a separate table rather than widening
    SkillEvidence itself: SkillEvidence.repo_id is NOT NULL and SQLite
    can't relax a NOT NULL constraint without a full table rebuild, which
    would put existing synced data at risk. evidence_type is "manual" for
    one added by hand or "resume" for one a resume tied to that role
    (app/profile/resume_profile_merge.py); there is no analog to
    "failed"/"rate_limited"/"no_signal".
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
    unconditional, every row not marked exclude_from_resume, sequential,
    never picked or trimmed by the LLM (app/resume_build/context.py's
    build_education_context). A person's own degree history isn't
    something semantic search gets to curate any more than a job history is.

    grade and details are optional extras some people want on a resume
    and others leave off: grade is one line such as "CGPA 8.9/10" shown
    beside the degree, details is a list of lines (coursework, rank,
    honors, thesis) shown as bullets under it. Left empty, neither takes
    any space in the rendered resume.
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
    # Same meaning as Repository.exclude_from_resume: archived, kept on the
    # Education page but never on a resume or in the portfolio counts.
    exclude_from_resume: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Skill(Base):
    """A skill with NO evidence link at all, not backed by any project or
    experience: either added by hand or pulled from a resume's tags. A
    skill that DOES have evidence is never rowed here; it's derived at
    read time by GET /api/skills from SkillEvidence +
    ExperienceSkillEvidence. This table exists purely so "I have this
    skill but nothing in here demonstrates it yet" has somewhere to live.
    """

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    # The resume whose extraction added this row, null for one typed in
    # by hand. An id, not a copied name, so the Skills page always shows
    # the resume's current name after a rename. Cleared when that resume
    # is deleted (app/api/resume.py), leaving the skill in place.
    source_resume_id: Mapped[int | None] = mapped_column(
        ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("account_id", "name", name="uq_skills_account_name"),)


class ResumeProfileLink(Base):
    """One profile row a resume's extraction landed on: the experience,
    bullet point, experience skill, education entry or freestanding skill
    that merging that resume created or matched
    (app/profile/resume_profile_merge.py). Lets a resume answer "what in
    my profile came from this file?" from the database alone, without
    re-reading the file or guessing by name.

    kind is "experience", "experience_point", "experience_skill",
    "education" or "skill"; ref_id is the id in that kind's table
    (experiences, experience_points, experience_skill_evidence,
    education, skills). created is true when this resume's merge added
    the row and false when the row already existed and the resume only
    matched it. A plain id rather than a foreign key per kind, so rows are
    removed by hand wherever their target is deleted (unlink_profile_rows
    in app/profile/resume_profile_merge.py), same as the other manual
    cleanups SQLite needs here without foreign key enforcement.
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
    """The finished 2D skill map for an account (see
    app/profile/skill_map.py). Stored whole, as the JSON the map endpoint
    returns, because building it costs an embedding-model load and a
    t-SNE fit, and the answer only changes when the skills do.

    fingerprint hashes everything that went into the layout: the layout
    version, the embedding model, and the exact text each skill was
    embedded as (which carries its evidence context). A request whose
    fingerprint does not match this row rebuilds and replaces it, so
    there is one row per account and never a stale map served by
    mistake.
    """

    __tablename__ = "skill_map_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String)
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("account_id", name="uq_skill_map_cache_account"),)


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
    # What this particular file says about work history and schooling, as
    # extracted: list of {company, title, start_date, end_date, points} and
    # {institution, degree, location, start_date, end_date} dicts, dates as
    # "mar 2026" / "2026" strings (app/profile/month_year.py) or
    # null. A read-only snapshot of this one version; the
    # editable, deduplicated copies live in the Experience/Education tables
    # (app/profile/resume_profile_merge.py), so these are only rewritten by
    # a reprocess, never by PATCH /api/resume/{id}.
    experiences_json: Mapped[list] = mapped_column(JSON, default=list)
    education_json: Mapped[list] = mapped_column(JSON, default=list)
    # Same read-only snapshot for the header contact block: {name,
    # location, emails, phones, links: [{platform, url, label}]}. The
    # editable copies are Account.contact_location and the ContactEmail/
    # ContactPhone/SocialLink rows it gets merged into.
    contact_json: Mapped[dict] = mapped_column(JSON, default=dict)

    # "pending": not yet run. "extracted": succeeded, the fields
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

    # Set only on a generated resume whose build stopped partway (a model
    # too busy to answer, a Tectonic timeout), cleared once a retry
    # finishes it. Holds what app/api/resume_build.py needs to pick the
    # build up where it stopped: the original request, the tailored
    # content if that step had finished (reserve included), how far the
    # page fit got, and the error. content_json stays null until the
    # build finishes, so nothing that reads finished resumes sees this
    # one's half-built content.
    build_state_json: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
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
    # "pasted" | "url" | "screenshot" | "authenticated" | "mixed"
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
    # employment_type, work_mode, seniority, experience_required, skills_required
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

    # Pay as whole-number annual amounts in salary_currency, parsed from
    # extracted_json["salary_range"] (app/profile/salary.py) whenever
    # extraction runs, so postings can be filtered and sorted by pay. Either
    # bound may be null ("up to 20 LPA" has no minimum); both null means no
    # salary was stated or it didn't read as a number. Editable via PATCH.
    salary_min_annual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_max_annual: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(String, nullable=True)

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

    # Images attached when the posting was added, kept on disk (same
    # per-account storage convention as Resume.stored_path) so the
    # originals can be viewed later, distinct from raw_text_quarantined
    # (the LLM's own transcription of them, used everywhere text is
    # needed). screenshot_paths holds every image in the order given;
    # screenshot_path is the first of them, and the only record on rows
    # saved before screenshot_paths existed.
    screenshot_path: Mapped[str | None] = mapped_column(String, nullable=True)
    screenshot_paths: Mapped[list | None] = mapped_column(JSON, nullable=True)


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
    # legitimately has no account; key_id is only set on a real dispatch,
    # and is the key that actually answered, which after a failover is not
    # the first one tried (app/core/llm.py's _dispatch_over_keys). It also
    # backs each key's own budget cap, so a cap and the number /monitor
    # shows beside it are the same figure. A test's injected
    # _completion_fn never resolves a key, so key_id stays NULL for mocked
    # calls and for rows written before this column existed.
    # No FK constraint to api_keys/accounts (informational ids, not
    # enforced), so a since-deleted account or key doesn't break old rows.
    account_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    key_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    # Which feature spent this call: "repo_facts", "resume_build",
    # "pagefit_trim", "skill_review", and so on, passed by every call site
    # through app/core/llm.py's complete(). A model name and a tier say how
    # a call was routed, not what it was for, so without this the usage
    # page can show that spend went up without showing where. Grouped as
    # /monitor's by_purpose breakdown (app/api/monitor.py). Nullable: rows
    # written before this column existed, and any call site that passes
    # nothing, land in an "unattributed" bucket rather than being dropped.
    purpose: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class RateLimitEvent(Base):
    """One row per time an outbound call got rate-limited or budget-capped,
    a key dropped out of rotation, or a multi-step run stopped: GitHub
    (app/ingest/github/client.py) or the LLM provider / our own monthly cap
    / key failover (app/core/llm.py, app/core/pipeline.py). Distinct from
    Repository.
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
    allowed, and they back each other up: `is_active` marks the row
    app/core/llm.py dispatches with first, and if the provider blames that
    key mid-request (quota gone, credential rejected) the next usable row
    for the provider takes the same request over. At most one active per
    provider is enforced in app/core/api_keys_store.py rather than with a
    partial unique index, which isn't worth it for a single-process local
    app.

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
    in addition to) the global monthly budget in AppSetting.

    `enabled` is a manual on/off switch, distinct from `is_active`:
    disabling a key takes it out of dispatch consideration entirely (even
    if it's the active one for its provider) without losing its place;
    re-enabling it needs no re-activation. `allowed_account_ids` is an
    optional allow-list of Account ids (empty list, the default = every
    account on this device may use it, not "no one may"); a non-empty
    list restricts the key to only those accounts, checked in
    app/core/api_keys_store.py's resolve_dispatch_keys() against whichever
    account_id the calling code passes into app/core/llm.py's complete().
    `status` is one of unknown|valid|invalid|rate_limited|blocked, and
    splits into two groups that are treated very differently.
    `rate_limited` is temporary: the key filled a quota window that rolls
    over by itself, so `exhausted_at` records when that happened,
    `exhaustion_kind` which window was hit (per_minute|per_day|quota|
    unknown, read from the provider's own refusal by
    app/core/key_cooldown.py) and `retry_at` the earliest moment a recheck
    is worth making. app/core/key_refresh.py comes back at that moment,
    at startup and on an interval, and clears the three fields once the
    key answers again. `invalid` (the credential was rejected) and
    `blocked` (the provider forbade this key: suspended, revoked, or its
    API not enabled) are not temporary and never rechecked automatically,
    only when someone asks for it on /apis; they carry no cooldown fields.
    None of the four statuses takes a key out of rotation, they only
    change where it sits in the dispatch order (see
    app/core/api_keys_store.py's _dispatch_order), because a device with
    one key must still be able to try it. `last_check_detail` carries the
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
    blob, encrypted the same way the ApiKey model above stores provider
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
