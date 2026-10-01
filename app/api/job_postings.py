"""Job posting record: paste text, fetch a public URL, upload a
screenshot, or fetch a login-walled URL via a stored authenticated source
(app/core/db/models.py's AuthSource, app/ingest/jobs/auth_fetch.py). Every path
ends up as one JobPosting row, structured-extracted the same way
(app/profile/job_extract.py), best-effort role-family resolved
(app/profile/role_family.py), and best-effort indexed for semantic search
(app/retrieval/index.py's index_job_posting).

raw_text lands in JobPosting.raw_text_quarantined untouched regardless of
which path produced it (typed by a human, scraped from a public page,
transcribed by an LLM from a screenshot, or fetched from behind a login).
This text must never be string-formatted into a prompt template or enter
a system prompt; every extraction call passes it as its own separate
user-role document instead.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.core.db import JobPosting, RoleFamily, get_db
from app.core.settings import get_settings
from app.ingest.jobs.auth_fetch import (
    AuthFetchError,
    AuthLoginFailedError,
    AuthSourceNotFoundError,
    fetch_job_url_authenticated,
)
from app.ingest.jobs.url_fetch import JobUrlFetchError, fetch_job_url
from app.profile.job_extract import JobExtractionError, extract_job_posting, parse_skills_required
from app.profile.job_screenshot_extract import (
    ScreenshotExtractionError,
    UnsupportedScreenshotType,
    extract_job_posting_from_image,
)
from app.profile.role_family import resolve_role_family
from app.profile.salary import parse_salary

router = APIRouter(prefix="/api/job-postings")
logger = logging.getLogger(__name__)

_UNSPECIFIED = "(unspecified)"


def _content_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def _apply_salary(posting: JobPosting) -> None:
    """Re-derives the annual salary columns from the extracted salary text.
    Called whenever extracted_json is (re)written, so a reprocess or a
    manual salary edit never leaves stale numbers behind."""
    salary = parse_salary((posting.extracted_json or {}).get("salary_range", ""))
    posting.salary_min_annual = salary.min
    posting.salary_max_annual = salary.max
    posting.salary_currency = salary.currency


def _run_extraction(posting: JobPosting, db: Session) -> None:
    """Best-effort structured extraction, same posture as
    app/profile/resume_ingest.py's run_extraction: a failed call never
    loses the posting itself, just leaves extraction_status at "failed"
    for POST /{id}/reprocess to retry. Backfills company/title/location
    only where they were left at their unset default, never overwrites
    a value actually typed in. On success, also (best-effort, never
    blocking the posting save) resolves this posting's canonical role
    family and indexes it for semantic search.
    """
    try:
        extraction = extract_job_posting(
            posting.raw_text_quarantined, account_id=posting.account_id
        )
        posting.extracted_json = extraction.as_extracted_json()
        _apply_salary(posting)
        if posting.company == _UNSPECIFIED and extraction.company:
            posting.company = extraction.company
        if posting.title == _UNSPECIFIED and extraction.title:
            posting.title = extraction.title
        if not posting.location and extraction.location:
            posting.location = extraction.location
        posting.extraction_status = "extracted"
        posting.extraction_error = None
        posting.extracted_at = dt.datetime.now(dt.UTC)
    except JobExtractionError as e:
        posting.extraction_status = "failed"
        posting.extraction_error = str(e)
    except Exception as e:
        logger.warning("job posting extraction failed for id=%s", posting.id, exc_info=True)
        posting.extraction_status = "failed"
        posting.extraction_error = str(e)
    db.commit()
    db.refresh(posting)

    if posting.extraction_status != "extracted":
        return

    try:
        family = resolve_role_family(posting.title, account_id=posting.account_id)
        if family is not None:
            posting.role_family_id = family.id
            db.commit()
            db.refresh(posting)
    except Exception:
        logger.warning("role family resolution failed for posting id=%s", posting.id, exc_info=True)

    try:
        from app.retrieval.index import index_job_posting

        index_job_posting(posting, account_id=posting.account_id)
    except Exception:
        logger.warning("could not index posting id=%s for search", posting.id, exc_info=True)


def _create_posting_from_text(
    *,
    account_id: int,
    raw_text: str,
    source: str,
    external_id: str,
    company: str | None = None,
    title: str | None = None,
    location: str | None = None,
    apply_url: str | None = None,
    screenshot_path: str | None = None,
) -> JobPosting:
    """Shared core of every ingestion path (paste/url/screenshot/
    authenticated): dedup by content hash, create the row, run extraction.
    Raises HTTPException(422) on blank text. The caller-facing wrapper
    for each entry point translates its own failure modes (a bad URL, an
    unsupported image type, a failed login) into their own error before
    ever reaching here, so a 422 from this function specifically always
    means "the text we ended up with was empty."
    """
    raw_text = raw_text.strip()
    if not raw_text:
        raise HTTPException(status_code=422, detail="no readable text to save")

    content_hash = _content_hash(raw_text)
    db = get_db()
    try:
        existing = db.execute(
            select(JobPosting).where(JobPosting.content_hash == content_hash)
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        posting = JobPosting(
            account_id=account_id,
            source=source,
            external_id=external_id,
            company=(company or "").strip() or _UNSPECIFIED,
            title=(title or "").strip() or _UNSPECIFIED,
            location=(location or "").strip() or None,
            apply_url=(apply_url or None),
            raw_text_quarantined=raw_text,
            content_hash=content_hash,
            screenshot_path=screenshot_path,
        )
        db.add(posting)
        db.commit()
        db.refresh(posting)
        _run_extraction(posting, db)
        return posting
    finally:
        db.close()


class JobPostingCreate(BaseModel):
    account_id: int
    raw_text: str
    company: str | None = None
    title: str | None = None
    location: str | None = None


class RequiredSkillOut(BaseModel):
    skill: str
    level: str


class RoleFamilyOut(BaseModel):
    id: int
    canonical_name: str


class JobPostingSummary(BaseModel):
    id: int
    source: str
    company: str
    title: str
    location: str | None
    apply_url: str | None
    fetched_at: dt.datetime
    extraction_status: str
    extraction_error: str | None
    salary_range: str
    salary_min_annual: int | None
    salary_max_annual: int | None
    salary_currency: str | None
    employment_type: str
    work_mode: str
    seniority: str
    experience_required: str
    skills_required: list[RequiredSkillOut]
    other_requirements: list[str]
    role_summary: str
    applied: bool
    applied_at: dt.date | None
    applied_notes: str | None
    role_family: RoleFamilyOut | None
    screenshot_path: str | None

    @classmethod
    def from_posting(
        cls, p: JobPosting, role_family: RoleFamily | None = None
    ) -> JobPostingSummary:
        extracted = p.extracted_json or {}
        return cls(
            id=p.id, source=p.source, company=p.company, title=p.title,
            location=p.location, apply_url=p.apply_url, fetched_at=p.fetched_at,
            extraction_status=p.extraction_status, extraction_error=p.extraction_error,
            salary_range=extracted.get("salary_range", ""),
            salary_min_annual=p.salary_min_annual,
            salary_max_annual=p.salary_max_annual,
            salary_currency=p.salary_currency,
            employment_type=extracted.get("employment_type", ""),
            work_mode=extracted.get("work_mode", ""),
            seniority=extracted.get("seniority", ""),
            experience_required=extracted.get("experience_required", ""),
            skills_required=[
                RequiredSkillOut(**s)
                for s in parse_skills_required(extracted.get("skills_required", []))
            ],
            other_requirements=extracted.get("other_requirements", []),
            role_summary=extracted.get("role_summary", ""),
            applied=p.applied,
            applied_at=p.applied_at,
            applied_notes=p.applied_notes,
            role_family=(
                RoleFamilyOut(id=role_family.id, canonical_name=role_family.canonical_name)
                if role_family is not None
                else None
            ),
            screenshot_path=p.screenshot_path,
        )


class JobPostingDetail(JobPostingSummary):
    raw_text: str

    @classmethod
    def from_posting(cls, p: JobPosting, role_family: RoleFamily | None = None) -> JobPostingDetail:
        return cls(
            **JobPostingSummary.from_posting(p, role_family).model_dump(),
            raw_text=p.raw_text_quarantined,
        )


def _role_family_for(db: Session, posting: JobPosting) -> RoleFamily | None:
    if posting.role_family_id is None:
        return None
    return db.get(RoleFamily, posting.role_family_id)


@router.post("", response_model=JobPostingDetail)
def create_posting(body: JobPostingCreate, *, db: DbSession) -> JobPostingDetail:
    posting = _create_posting_from_text(
        account_id=body.account_id,
        raw_text=body.raw_text,
        source="pasted",
        external_id=_content_hash(body.raw_text.strip()),
        company=body.company,
        title=body.title,
        location=body.location,
    )
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


class JobPostingFromUrl(BaseModel):
    account_id: int
    url: str


@router.post("/from-url", response_model=JobPostingDetail)
def create_posting_from_url(body: JobPostingFromUrl, *, db: DbSession) -> JobPostingDetail:
    """Public-URL ingestion, no login (see app/ingest/jobs/url_fetch.py).
    A page that needs login or renders via client-side JS will fail here
    with a 422 explaining why; use /from-authenticated-url for a
    login-walled source instead.
    """
    try:
        title_guess, text = fetch_job_url(body.url)
    except JobUrlFetchError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    posting = _create_posting_from_text(
        account_id=body.account_id,
        raw_text=text,
        source="url",
        external_id=body.url.strip(),
        title=title_guess or None,
        apply_url=body.url.strip(),
    )
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


class JobPostingFromAuthUrl(BaseModel):
    account_id: int
    url: str
    auth_source_id: int


@router.post("/from-authenticated-url", response_model=JobPostingDetail)
def create_posting_from_authenticated_url(
    body: JobPostingFromAuthUrl,
    *,
    db: DbSession,
) -> JobPostingDetail:
    """Login-walled URL ingestion via a stored AuthSource
    (app/core/db/models.py, app/api/auth_sources.py): a real automated browser
    login, see app/ingest/jobs/auth_fetch.py's module docstring for the
    risks.
    """
    try:
        title_guess, text = fetch_job_url_authenticated(body.auth_source_id, body.url)
    except AuthSourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AuthLoginFailedError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except AuthFetchError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    posting = _create_posting_from_text(
        account_id=body.account_id,
        raw_text=text,
        source="authenticated",
        external_id=body.url.strip(),
        title=title_guess or None,
        apply_url=body.url.strip(),
    )
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


@router.post("/from-screenshot", response_model=JobPostingDetail)
def create_posting_from_screenshot(
    account_id: int = Form(...),
    file: UploadFile = File(...),
    *,
    db: DbSession,
) -> JobPostingDetail:
    """Screenshot ingestion: the LLM transcribes the visible posting text
    (see app/profile/job_screenshot_extract.py) and that transcription
    becomes raw_text_quarantined, exactly as if it had been pasted. The
    original image is also kept on disk (screenshot_path) so it can be
    viewed later, same per-account storage convention as
    app/api/resume.py's uploads.
    """
    file_bytes = file.file.read()
    mime_type = file.content_type or ""
    try:
        result = extract_job_posting_from_image(file_bytes, mime_type, account_id=account_id)
    except UnsupportedScreenshotType as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ScreenshotExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    settings = get_settings()
    account_dir = Path(settings.job_screenshot_storage_dir) / str(account_id)
    account_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "screenshot").name
    content_hash = _content_hash(result.raw_text_transcribed)
    stored_path = account_dir / f"{content_hash[:16]}_{safe_name}"
    stored_path.write_bytes(file_bytes)

    existing = db.execute(
        select(JobPosting).where(JobPosting.content_hash == content_hash)
    ).scalar_one_or_none()
    if existing is not None:
        return JobPostingDetail.from_posting(existing, _role_family_for(db, existing))

    posting = JobPosting(
        account_id=account_id,
        source="screenshot",
        external_id=content_hash,
        company=result.extraction.company or _UNSPECIFIED,
        title=result.extraction.title or _UNSPECIFIED,
        location=result.extraction.location or None,
        raw_text_quarantined=result.raw_text_transcribed,
        content_hash=content_hash,
        screenshot_path=str(stored_path),
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)
    # Extraction already ran (as part of reading the screenshot); store
    # it directly rather than re-running text extraction on the
    # transcription, which would just cost a second LLM call to
    # re-derive the same structured fields the image call already
    # produced.
    posting.extracted_json = result.extraction.as_extracted_json()
    _apply_salary(posting)
    posting.extraction_status = "extracted"
    posting.extracted_at = dt.datetime.now(dt.UTC)
    db.commit()
    db.refresh(posting)

    try:
        family = resolve_role_family(posting.title, account_id=account_id)
        if family is not None:
            posting.role_family_id = family.id
            db.commit()
            db.refresh(posting)
    except Exception:
        logger.warning(
            "role family resolution failed for posting id=%s", posting.id, exc_info=True
        )
    try:
        from app.retrieval.index import index_job_posting

        index_job_posting(posting, account_id=account_id)
    except Exception:
        logger.warning("could not index posting id=%s for search", posting.id, exc_info=True)

    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


@router.get("", response_model=list[JobPostingSummary])
def list_postings(
    account_id: int, min_salary: int | None = None, *, db: DbSession
) -> list[JobPostingSummary]:
    """min_salary (annual, same currency units as stored) keeps postings
    whose upper bound, or lower bound when only that is known, reaches it.
    Postings with no parsed salary are left out when it is set."""
    query = select(JobPosting).where(JobPosting.account_id == account_id)
    if min_salary is not None:
        best = func.coalesce(JobPosting.salary_max_annual, JobPosting.salary_min_annual)
        query = query.where(best >= min_salary)
    rows = list(db.execute(query.order_by(JobPosting.fetched_at.desc())).scalars())
    families: dict[int, RoleFamily] = {
        f.id: f for f in db.execute(select(RoleFamily)).scalars()
    }
    return [
        JobPostingSummary.from_posting(
            r, families.get(r.role_family_id) if r.role_family_id is not None else None
        )
        for r in rows
    ]


@router.get("/{posting_id}", response_model=JobPostingDetail)
def posting_detail(posting_id: int, *, db: DbSession) -> JobPostingDetail:
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


class JobPostingUpdate(BaseModel):
    company: str | None = None
    title: str | None = None
    location: str | None = None
    apply_url: str | None = None
    applied: bool | None = None
    applied_at: dt.date | None = None
    applied_notes: str | None = None
    salary_range: str | None = None
    salary_min_annual: int | None = None
    salary_max_annual: int | None = None
    salary_currency: str | None = None
    employment_type: str | None = None
    work_mode: str | None = None
    seniority: str | None = None
    experience_required: str | None = None


# Fields that live in extracted_json rather than their own column.
_EXTRACTED_TEXT_FIELDS = (
    "salary_range", "employment_type", "work_mode", "seniority", "experience_required",
)


@router.patch("/{posting_id}", response_model=JobPostingDetail)
def update_posting(posting_id: int, body: JobPostingUpdate, *, db: DbSession) -> JobPostingDetail:
    fields = body.model_dump(exclude_unset=True)
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    if "company" in fields:
        company = fields["company"].strip() if fields["company"] else None
        posting.company = company or _UNSPECIFIED
    if "title" in fields:
        title = fields["title"].strip() if fields["title"] else None
        posting.title = title or _UNSPECIFIED
    if "location" in fields:
        posting.location = (fields["location"].strip() if fields["location"] else None) or None
    if "apply_url" in fields:
        apply_url = fields["apply_url"].strip() if fields["apply_url"] else None
        posting.apply_url = apply_url or None
    if "applied" in fields:
        posting.applied = bool(fields["applied"])
        # Marking applied with no explicit date stamps today; unmarking
        # clears both applied_at and applied_notes, one source of
        # truth for "this hasn't been applied to."
        if posting.applied and posting.applied_at is None and "applied_at" not in fields:
            posting.applied_at = dt.date.today()
        if not posting.applied and "applied_at" not in fields:
            posting.applied_at = None
            posting.applied_notes = None
    if "applied_at" in fields:
        posting.applied_at = fields["applied_at"]
    if "applied_notes" in fields:
        posting.applied_notes = (
            fields["applied_notes"].strip() if fields["applied_notes"] else None
        ) or None
    edited = {k: (fields[k] or "").strip() for k in _EXTRACTED_TEXT_FIELDS if k in fields}
    if edited:
        # A fresh dict, so SQLAlchemy sees the JSON column change.
        posting.extracted_json = {**(posting.extracted_json or {}), **edited}
        if "salary_range" in edited:
            _apply_salary(posting)
    # Explicit numbers win over whatever the salary text parsed to.
    for key in ("salary_min_annual", "salary_max_annual"):
        if key in fields:
            setattr(posting, key, fields[key])
    if "salary_currency" in fields:
        currency = (fields["salary_currency"] or "").strip().upper()
        posting.salary_currency = currency or None
    db.commit()
    db.refresh(posting)
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


@router.post("/{posting_id}/reprocess", response_model=JobPostingDetail)
def reprocess_posting(posting_id: int, *, db: DbSession) -> JobPostingDetail:
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    _run_extraction(posting, db)
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


@router.delete("/{posting_id}")
def delete_posting(posting_id: int, *, db: DbSession) -> dict:
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    db.delete(posting)
    db.commit()
    return {"deleted": True}
