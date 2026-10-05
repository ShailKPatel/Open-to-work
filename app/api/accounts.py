"""Local device profiles, not user accounts in the auth sense. No
password, no server session. The client (browser) remembers which one it's
using; the server just lists/creates them and, if given one, ingests an
optional signup resume file through app.profile.resume_ingest, the same
path POST /api/resume uses (see app/api/resume.py).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from github import UnknownObjectException
from pydantic import BaseModel
from sqlalchemy import delete, select

from app.api.deps import DbSession
from app.api.input_limits import file_notes, require_confirmation
from app.core.db import (
    Account,
    Education,
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    GitHubSyncRun,
    JobPosting,
    Profile,
    Repository,
    Resume,
    ResumeProfileLink,
    Skill,
    SkillEvidence,
    SkillStar,
    SkillVerdict,
    SocialLink,
    SyncSource,
)
from app.core.settings import get_settings
from app.ingest.github.client import GitHubClient
from app.profile.resume_ingest import GeneratedResumeError, ingest_resume, is_generated_pdf

router = APIRouter()
logger = logging.getLogger(__name__)


class AccountSummary(BaseModel):
    id: int
    first_name: str
    last_name: str
    github_username: str
    has_resume: bool

    @classmethod
    def from_account(cls, account: Account) -> AccountSummary:
        return cls(
            id=account.id,
            first_name=account.first_name,
            last_name=account.last_name,
            github_username=account.github_username,
            has_resume=account.resume_path is not None,
        )


@router.get("/accounts", response_model=list[AccountSummary])
def list_accounts(db: DbSession) -> list[AccountSummary]:
    accounts = db.execute(select(Account).order_by(Account.created_at)).scalars().all()
    return [AccountSummary.from_account(a) for a in accounts]


def _github_user_exists(username: str) -> bool | None:
    """True/False if we could actually check GitHub, None if the check
    itself failed (bad/expired token, rate limit, network), a caller
    should only block on a confirmed False. Blocking someone from making a
    local profile because our own GitHub token died is worse than letting
    a typo through to be caught at sync time instead."""
    try:
        GitHubClient().repo_count_hint(username)
        return True
    except UnknownObjectException:
        return False
    except Exception:
        logger.warning(
            "could not verify GitHub username %r before account creation",
            username,
            exc_info=True,
        )
        return None


@router.post("/accounts", response_model=AccountSummary)
def create_account(
    first_name: str = Form(...),
    last_name: str = Form(...),
    github_username: str = Form(""),
    resume: UploadFile | None = File(None),
    confirm_large: bool = Form(False),
    *,
    db: DbSession,
) -> AccountSummary:
    # GitHub is optional at signup, non-technical users, or anyone without
    # a GitHub account, leave it blank and build their profile by hand
    # instead. Only check/seed a sync source when one was actually given.
    username = github_username.strip()
    if username and _github_user_exists(username) is False:
        raise HTTPException(status_code=422, detail=f"GitHub user '{username}' not found")
    if resume is not None and resume.filename:
        # Checked before the account exists, so a refused file leaves nothing behind.
        data = resume.file.read()
        if is_generated_pdf(data):
            raise HTTPException(status_code=422, detail=str(GeneratedResumeError()))
        require_confirmation("file", file_notes(data), confirm_large)
        resume.file.seek(0)

    account = Account(
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        github_username=username,
    )
    db.add(account)
    db.commit()
    db.refresh(account)

    if username:
        # Seed one sync source with the signup username so the
        # fetch-data page (see app/api/sources.py) isn't empty on
        # first visit, it's just the first row from there on,
        # editable/removable like any other, and separate from
        # account.github_username itself.
        db.add(
            SyncSource(
                account_id=account.id,
                raw_input=username,
                kind="user",
                github_username=username,
            )
        )
        db.commit()

    if resume is not None and resume.filename:
        # Same ingestion path POST /api/resume uses (app/api/resume.py):
        # saves the file, creates its Resume row, mirrors it onto
        # account.resume_filename/resume_path, and runs extraction.
        ingest_resume(db, account.id, resume)
        db.refresh(account)

    return AccountSummary.from_account(account)


def _delete_qdrant_points(skill_evidence_ids: list[int]) -> None:
    """Best-effort, SQLite is the source of truth, Qdrant is a derived
    index. A Qdrant hiccup shouldn't block deleting someone's actual data.
    """
    if not skill_evidence_ids:
        return
    try:
        from app.retrieval.index import COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(COLLECTION):
            client.delete(collection_name=COLLECTION, points_selector=skill_evidence_ids)
    except Exception:
        logger.exception("could not clean up Qdrant points for deleted account; continuing")


def _delete_resume_qdrant_points(resume_ids: list[int]) -> None:
    """Twin of _delete_qdrant_points above, against the resumes collection
    (app/retrieval/index.py). Same reasoning, same best-effort silence."""
    if not resume_ids:
        return
    try:
        from app.retrieval.index import RESUME_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(RESUME_COLLECTION):
            client.delete(collection_name=RESUME_COLLECTION, points_selector=resume_ids)
    except Exception:
        logger.exception("could not clean up resume Qdrant points for deleted account; continuing")


def _delete_point_qdrant_vectors(point_ids: list[int]) -> None:
    """Twin of _delete_qdrant_points above, against the experience_points
    collection (app/retrieval/index.py). Same reasoning, same best-effort
    silence."""
    if not point_ids:
        return
    try:
        from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(EXPERIENCE_POINTS_COLLECTION):
            client.delete(collection_name=EXPERIENCE_POINTS_COLLECTION, points_selector=point_ids)
    except Exception:
        logger.exception("could not clean up point Qdrant vectors for deleted account; continuing")


def _delete_job_posting_qdrant_points(posting_ids: list[int]) -> None:
    """Twin of _delete_qdrant_points above, against the job_postings
    collection (app/retrieval/index.py). Same reasoning, same best-effort
    silence. Note: job posting rows are a shared content cache (see
    JobPosting's own docstring), only the ones actually owned by this
    account (account_id == this account) are deleted here or in SQLite,
    same scoping the SQLite delete below already uses."""
    if not posting_ids:
        return
    try:
        from app.retrieval.index import JOB_POSTINGS_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(JOB_POSTINGS_COLLECTION):
            client.delete(collection_name=JOB_POSTINGS_COLLECTION, points_selector=posting_ids)
    except Exception:
        logger.exception(
            "could not clean up job posting Qdrant points for deleted account; continuing"
        )


@router.delete("/accounts/{account_id}")
def delete_account(account_id: int, *, db: DbSession) -> dict:
    """Erases everything tied to this account: synced repos, skill
    evidence (SQLite and Qdrant), experience, its skill evidence and its
    points (SQLite and Qdrant), education, contact-adjacent social links,
    freestanding manual skills, any profile snapshot, every resume
    (SQLite, Qdrant, and its file), every job posting this account
    created (SQLite, Qdrant, and any screenshot file), and the account
    itself. No soft-delete, this is what "delete" means here, per the
    confirmation prompt the client shows before calling this.
    """
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")

    repo_ids = list(
        db.execute(
            select(Repository.id).where(Repository.account_id == account_id)
        ).scalars()
    )
    evidence_ids = (
        list(
            db.execute(
                select(SkillEvidence.id).where(SkillEvidence.repo_id.in_(repo_ids))
            ).scalars()
        )
        if repo_ids
        else []
    )

    experience_ids = list(
        db.execute(
            select(Experience.id).where(Experience.account_id == account_id)
        ).scalars()
    )
    experience_evidence_ids = (
        list(
            db.execute(
                select(ExperienceSkillEvidence.id).where(
                    ExperienceSkillEvidence.experience_id.in_(experience_ids)
                )
            ).scalars()
        )
        if experience_ids
        else []
    )

    # Both evidence tables share the skill_evidence Qdrant collection
    # (see app/retrieval/index.py). Experience-linked points are stored
    # at evidence.id + an offset (fixes a real numeric collision with
    # repo-linked ids sharing the same collection, see index.py's
    # _EXPERIENCE_EVIDENCE_ID_OFFSET), passing the raw id here would
    # either delete nothing (no point exists at that id) or, worse,
    # delete an unrelated repo-evidence point that happens to share
    # that raw number. Must go through experience_evidence_point_id().
    from app.retrieval.index import experience_evidence_point_id

    _delete_qdrant_points(
        evidence_ids + [experience_evidence_point_id(i) for i in experience_evidence_ids]
    )

    point_ids = (
        list(
            db.execute(
                select(ExperiencePoint.id).where(
                    ExperiencePoint.experience_id.in_(experience_ids)
                )
            ).scalars()
        )
        if experience_ids
        else []
    )
    _delete_point_qdrant_vectors(point_ids)

    resume_ids = list(
        db.execute(select(Resume.id).where(Resume.account_id == account_id)).scalars()
    )
    _delete_resume_qdrant_points(resume_ids)

    posting_ids = list(
        db.execute(select(JobPosting.id).where(JobPosting.account_id == account_id)).scalars()
    )
    _delete_job_posting_qdrant_points(posting_ids)

    if repo_ids:
        db.execute(delete(SkillEvidence).where(SkillEvidence.repo_id.in_(repo_ids)))
        db.execute(delete(Repository).where(Repository.account_id == account_id))
    if experience_ids:
        db.execute(
            delete(ExperienceSkillEvidence).where(
                ExperienceSkillEvidence.experience_id.in_(experience_ids)
            )
        )
        db.execute(
            delete(ExperiencePoint).where(
                ExperiencePoint.experience_id.in_(experience_ids)
            )
        )
        db.execute(delete(Experience).where(Experience.account_id == account_id))
    db.execute(delete(Education).where(Education.account_id == account_id))
    db.execute(delete(SocialLink).where(SocialLink.account_id == account_id))
    db.execute(delete(Skill).where(Skill.account_id == account_id))
    db.execute(delete(SkillStar).where(SkillStar.account_id == account_id))
    db.execute(delete(SkillVerdict).where(SkillVerdict.account_id == account_id))
    db.execute(delete(Profile).where(Profile.account_id == account_id))
    db.execute(delete(SyncSource).where(SyncSource.account_id == account_id))
    db.execute(delete(GitHubSyncRun).where(GitHubSyncRun.account_id == account_id))
    db.execute(
        delete(ResumeProfileLink).where(
            ResumeProfileLink.resume_id.in_(
                select(Resume.id).where(Resume.account_id == account_id)
            )
        )
    )
    db.execute(delete(Resume).where(Resume.account_id == account_id))
    db.execute(delete(JobPosting).where(JobPosting.account_id == account_id))

    # Removes every resume file for this account too: they all live
    # under this one per-account directory (app/profile/resume_ingest.py).
    resume_dir = Path(get_settings().resume_storage_dir) / str(account_id)
    if resume_dir.exists():
        shutil.rmtree(resume_dir, ignore_errors=True)

    # Same per-account-directory shape for job-posting screenshots
    # (app/api/job_postings.py's from_screenshot).
    screenshot_dir = Path(get_settings().job_screenshot_storage_dir) / str(account_id)
    if screenshot_dir.exists():
        shutil.rmtree(screenshot_dir, ignore_errors=True)

    db.delete(account)
    db.commit()
    return {"deleted": True, "account_id": account_id}
