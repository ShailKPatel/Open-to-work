"""Job posting record: POST /compose takes any mix of pasted text, links
and screenshots for one posting and works out what each piece is. The
single-input routes remain for pasted text and a single screenshot. Every
path ends up as one JobPosting row, structured-extracted the same way
(app/profile/job_extract.py), best-effort role-family resolved
(app/profile/role_family.py), and best-effort indexed for semantic search
(app/retrieval/index.py's index_job_posting).

raw_text lands in JobPosting.raw_text_quarantined untouched regardless of
which path produced it (typed by a human or transcribed by an LLM from a
screenshot).
This text must never be string-formatted into a prompt template or enter
a system prompt; every extraction call passes it as its own separate
user-role document instead.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.input_limits import file_notes, files_notes, require_confirmation, text_notes
from app.core.db import (
    JobPosting,
    Resume,
    RoleFamily,
    get_db,
)
from app.core.settings import get_settings
from app.profile.job_extract import (
    JobExtraction,
    JobExtractionError,
    extract_job_posting,
    parse_skills_required,
)
from app.profile.job_place import cities, work_mode_kind
from app.profile.job_screenshot_extract import (
    ScreenshotExtractionError,
    UnsupportedScreenshotType,
    extract_job_posting_from_image,
    extract_job_posting_from_images,
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


def _run_extraction(posting: JobPosting, db: Session, bypass_cache: bool = False) -> None:
    """Best-effort structured extraction, same posture as
    app/profile/resume_ingest.py's run_extraction: a failed call never
    loses the posting itself, just leaves extraction_status at "failed"
    for POST /{id}/reprocess to retry. Backfills company/title/location
    only where they were left at their unset default, never overwrites
    a value actually typed in. On success, also (best-effort, never
    blocking the posting save) resolves this posting's canonical role
    family and indexes it for semantic search. bypass_cache is set by
    Reprocess, so the posting is read again rather than replayed.
    """
    try:
        extraction = extract_job_posting(
            posting.raw_text_quarantined,
            account_id=posting.account_id,
            bypass_cache=bypass_cache,
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

    if posting.extraction_status == "extracted":
        _enrich(posting, db)


def _store_extraction(posting: JobPosting, extraction: JobExtraction, db: Session) -> None:
    """Saves an extraction that already ran elsewhere (a screenshot read
    does extraction in the same call), with the same backfill rules as
    _run_extraction, then enriches."""
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
    db.commit()
    db.refresh(posting)
    _enrich(posting, db)


def _enrich(posting: JobPosting, db: Session) -> None:
    """Best-effort role family and search index for an extracted posting;
    neither ever blocks the save."""
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
    screenshot_paths: list[str] | None = None,
    source_text: str | None = None,
    source_links: list[str] | None = None,
    extraction: JobExtraction | None = None,
) -> JobPosting:
    """Shared core of the text-based ingestion paths: dedup by content
    hash, create the row, run extraction. Raises HTTPException(422) on
    blank text. Callers translate their own failure modes (an unsupported
    image type, say) before reaching here, so a 422 from this function
    always means "the text we ended up with was empty."
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
            screenshot_path=screenshot_paths[0] if screenshot_paths else None,
            screenshot_paths=screenshot_paths or None,
            source_text=source_text or None,
            source_links=source_links or None,
        )
        db.add(posting)
        db.commit()
        db.refresh(posting)
        if extraction is not None:
            _store_extraction(posting, extraction, db)
        else:
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
    # Set on the second send, once the person has agreed to process text
    # past the soft size limit (app/api/input_limits.py).
    confirm_large: bool = False


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
    # Normalised by app/profile/job_place.py: remote, hybrid, on_site or
    # unknown, and the cities an on-site or hybrid posting names (remote
    # ones are their own place, so the list page never files them under
    # a city).
    work_mode_kind: str
    cities: list[str]
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
    # Served by GET /api/job-postings/{id}/images/{index}, in the order given.
    image_count: int

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
            work_mode_kind=work_mode_kind(extracted.get("work_mode", ""), p.location),
            cities=cities(p.location),
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
            image_count=len(_image_paths(p)),
        )


class JobPostingDetail(JobPostingSummary):
    raw_text: str
    # What was given when saving: the pasted text as typed and the links
    # in it. Images are served by GET /{id}/images/{index}.
    source_text: str
    source_links: list[str]
    # Everything extracted, as Markdown to paste into an LLM chat.
    context_text: str

    @classmethod
    def from_posting(cls, p: JobPosting, role_family: RoleFamily | None = None) -> JobPostingDetail:
        summary = JobPostingSummary.from_posting(p, role_family)
        return cls(
            **summary.model_dump(),
            raw_text=p.raw_text_quarantined,
            source_text=_source_text(p),
            source_links=list(p.source_links or []),
            context_text=_context_text(summary),
        )


def _source_text(posting: JobPosting) -> str:
    if posting.source_text is not None:
        return posting.source_text
    # Older rows: a pasted-only posting's text is exactly what was pasted,
    # anything else had a transcription joined onto it.
    return posting.raw_text_quarantined if posting.source == "pasted" else ""


def _context_text(job: JobPostingSummary) -> str:
    """The extracted fields as Markdown, laid out for an LLM to read.
    Fields with nothing in them are left out rather than shown blank."""
    def known(value: str | None) -> str:
        value = (value or "").strip()
        return "" if value == _UNSPECIFIED else value

    title, company = known(job.title), known(job.company)
    heading = " at ".join(part for part in (title, company) if part) or "Untitled"
    lines = [f"# Job posting: {heading}", ""]

    salary = ""
    if job.salary_min_annual is not None or job.salary_max_annual is not None:
        bounds = " to ".join(
            f"{n:,}" for n in (job.salary_min_annual, job.salary_max_annual) if n is not None
        )
        if job.salary_min_annual is None:
            bounds = f"up to {bounds}"
        elif job.salary_max_annual is None:
            bounds = f"from {bounds}"
        currency = f" {job.salary_currency}" if job.salary_currency else ""
        salary = f"{bounds}{currency} per year"
    if job.salary_range:
        salary = f"{salary} (as posted: {job.salary_range})" if salary else job.salary_range

    applied = ""
    if job.applied:
        applied = f"Applied on {job.applied_at.isoformat()}" if job.applied_at else "Applied"
        if job.applied_notes:
            applied += f"; notes: {job.applied_notes}"

    facts = [
        ("Title", title),
        ("Company", company),
        ("Role family", job.role_family.canonical_name if job.role_family else ""),
        ("Location", known(job.location)),
        ("Work mode", job.work_mode),
        ("Employment type", job.employment_type),
        ("Seniority", job.seniority),
        ("Experience required", job.experience_required),
        ("Salary", salary),
        ("Apply link", job.apply_url or ""),
        ("Application status", applied),
    ]
    lines += [f"- {label}: {value.strip()}" for label, value in facts if value and value.strip()]

    if job.role_summary:
        lines += ["", "## Role summary", "", job.role_summary.strip()]
    if job.skills_required:
        lines += ["", "## Required skills", ""]
        lines += [
            f"- {s.skill} ({s.level})" if s.level else f"- {s.skill}"
            for s in job.skills_required
        ]
    if job.other_requirements:
        lines += ["", "## Other requirements", ""]
        lines += [f"- {r}" for r in job.other_requirements]
    return "\n".join(lines) + "\n"


def _image_paths(posting: JobPosting) -> list[str]:
    if posting.screenshot_paths:
        return list(posting.screenshot_paths)
    return [posting.screenshot_path] if posting.screenshot_path else []


def _role_family_for(db: Session, posting: JobPosting) -> RoleFamily | None:
    if posting.role_family_id is None:
        return None
    return db.get(RoleFamily, posting.role_family_id)


@router.post("", response_model=JobPostingDetail)
def create_posting(body: JobPostingCreate, *, db: DbSession) -> JobPostingDetail:
    require_confirmation("text", text_notes(body.raw_text), body.confirm_large)
    posting = _create_posting_from_text(
        account_id=body.account_id,
        raw_text=body.raw_text,
        source="pasted",
        external_id=_content_hash(body.raw_text.strip()),
        source_text=body.raw_text.strip(),
        source_links=_find_urls(body.raw_text),
        company=body.company,
        title=body.title,
        location=body.location,
    )
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


@router.post("/from-screenshot", response_model=JobPostingDetail)
def create_posting_from_screenshot(
    account_id: int = Form(...),
    file: UploadFile = File(...),
    confirm_large: bool = Form(False),
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
    require_confirmation("file", file_notes(file_bytes), confirm_large)
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
        screenshot_paths=[str(stored_path)],
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)
    # Extraction already ran (as part of reading the screenshot); store
    # it directly rather than re-running text extraction on the
    # transcription, which would just cost a second LLM call to
    # re-derive the same structured fields the image call already
    # produced.
    _store_extraction(posting, result.extraction, db)

    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


_URL_RE = re.compile(r"https?://[^\s<>\"'\]\)]+", re.IGNORECASE)


def _find_urls(*texts: str) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for match in _URL_RE.findall(text or ""):
            url = match.rstrip(".,;:!?")
            if url not in seen:
                seen.append(url)
    return seen


class JobPostingComposed(JobPostingDetail):
    # Inputs that could not be used (screenshots that could not be read,
    # say) while the rest still made a posting.
    warnings: list[str]


@router.post("/compose", response_model=JobPostingComposed)
def compose_posting(
    account_id: int = Form(...),
    text: str = Form(""),
    links: str = Form(""),
    files: list[UploadFile] | None = File(None),
    confirm_large: bool = Form(False),
    *,
    db: DbSession,
) -> JobPostingComposed:
    """One way in for every kind of input: any mix of pasted text, links
    and screenshots describing a single posting. Nobody has to say which
    kind they are giving; links are also picked out of the pasted text.

    Links are never opened. The first one is kept as the apply link, and
    the posting itself has to come from the text or the screenshots.
    """
    warnings: list[str] = []
    urls = _find_urls(links, text)
    typed = _URL_RE.sub("", text).strip()

    images: list[tuple[bytes, str, str]] = []
    for upload in files or []:
        data = upload.file.read()
        if not data:
            continue
        mime = (upload.content_type or "").lower()
        if not mime.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail=f"{upload.filename or 'that file'} is not an image; attach screenshots only",
            )
        images.append((data, mime, Path(upload.filename or "screenshot").name))
    require_confirmation(
        "job posting",
        text_notes(text) + files_notes([data for data, _, _ in images]),
        confirm_large,
    )

    kinds: set[str] = set()
    sections: list[str] = []
    if typed:
        sections.append(text.strip())
        kinds.add("pasted")

    extraction: JobExtraction | None = None
    if images:
        try:
            result = extract_job_posting_from_images(
                [(data, mime) for data, mime, _ in images],
                context_text="\n\n".join(sections),
                account_id=account_id,
            )
        except UnsupportedScreenshotType as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except ScreenshotExtractionError as e:
            if not sections:
                raise HTTPException(status_code=422, detail=str(e)) from e
            warnings.append(f"Could not read the screenshots: {e}")
        else:
            extraction = result.extraction
            if result.raw_text_transcribed:
                sections.append(result.raw_text_transcribed)
            kinds.add("screenshot")

    if not sections:
        if urls:
            detail = (
                "Only a link was given. Paste the posting text or add a "
                "screenshot as well; the link is kept as the apply link."
            )
        else:
            detail = "Add the posting text or a screenshot."
        raise HTTPException(status_code=422, detail=detail)

    raw_text = "\n\n".join(sections)
    # Every image is kept with the posting, read or not, so it can be
    # looked at again next to the text.
    screenshot_paths: list[str] = []
    if images:
        account_dir = Path(get_settings().job_screenshot_storage_dir) / str(account_id)
        account_dir.mkdir(parents=True, exist_ok=True)
        prefix = _content_hash(raw_text.strip())[:16]
        for i, (data, _, name) in enumerate(images):
            stored = account_dir / f"{prefix}_{i}_{name}"
            stored.write_bytes(data)
            screenshot_paths.append(str(stored))

    apply_url = urls[0] if urls else None
    posting = _create_posting_from_text(
        account_id=account_id,
        raw_text=raw_text,
        source=kinds.pop() if len(kinds) == 1 else "mixed",
        external_id=apply_url or _content_hash(raw_text.strip()),
        apply_url=apply_url,
        screenshot_paths=screenshot_paths,
        source_text=text.strip(),
        source_links=urls,
        extraction=extraction,
    )
    return JobPostingComposed(
        **JobPostingDetail.from_posting(posting, _role_family_for(db, posting)).model_dump(),
        warnings=warnings,
    )


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


@router.get("/{posting_id}/images/{index}")
def posting_image(posting_id: int, index: int, *, db: DbSession) -> FileResponse:
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    paths = _image_paths(posting)
    if not 0 <= index < len(paths) or not Path(paths[index]).is_file():
        raise HTTPException(status_code=404, detail="no such image on this posting")
    return FileResponse(paths[index])


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
    _run_extraction(posting, db, bypass_cache=True)
    return JobPostingDetail.from_posting(posting, _role_family_for(db, posting))


def _delete_qdrant_point(posting_id: int) -> None:
    """Best-effort, same reasoning as accounts.py's _delete_qdrant_points:
    a missing or unreachable Qdrant must not block deleting the row."""
    try:
        from app.retrieval.index import JOB_POSTINGS_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(JOB_POSTINGS_COLLECTION):
            client.delete(collection_name=JOB_POSTINGS_COLLECTION, points_selector=[posting_id])
    except Exception:
        logger.exception("could not clean up Qdrant point for deleted job posting; continuing")


@router.delete("/{posting_id}")
def delete_posting(posting_id: int, *, db: DbSession) -> dict:
    """Removes the posting and everything derived from it: its required
    skills and other extracted fields live on the row itself, its search
    vector and image files are removed too.
    Resumes built for it are the person's own and stay in the library,
    only unlinked from the posting."""
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    images = _image_paths(posting)
    _delete_qdrant_point(posting_id)
    db.execute(
        update(Resume).where(Resume.job_posting_id == posting_id).values(job_posting_id=None)
    )
    db.delete(posting)
    db.commit()
    for path in images:
        Path(path).unlink(missing_ok=True)
    return {"deleted": True}
