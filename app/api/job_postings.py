"""Job posting record: POST /compose takes any mix of pasted text, links
and screenshots for one posting and works out what each piece is. The
single-input routes remain for pasted text and a single screenshot. Every
path ends up as one JobPosting row, structured-extracted the same way
(app/profile/job_extract.py), best-effort role-family resolved
(app/profile/role_family.py), and best-effort indexed for semantic search
(app/retrieval/index.py's index_job_posting). Saving returns at once with
extraction_status "pending"; the LLM read runs in a worker thread
(app/core/jobs.py) and the pages poll until it is done.

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
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.input_limits import file_notes, files_notes, require_confirmation, text_notes
from app.core import jobs
from app.core.db import (
    JobPosting,
    Resume,
    RoleFamily,
    get_db,
)
from app.core.filetypes import sniff_file, sniff_image_type
from app.core.llm import error_kind
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
    extract_job_posting_from_images,
)
from app.profile.role_family import resolve_role_family
from app.profile.salary import parse_salary

router = APIRouter(prefix="/api/job-postings")
logger = logging.getLogger(__name__)

_UNSPECIFIED = "(unspecified)"

# extraction_status while a background read is under way.
_READING = "pending"
_INTERRUPTED = "The app restarted before this posting was read. Reprocess it to try again."


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


def _run_extraction(
    posting: JobPosting, db: Session, bypass_cache: bool = False, note: str | None = None
) -> None:
    """Best-effort structured extraction, same posture as
    app/profile/resume_ingest.py's run_extraction: a failed call never
    loses the posting itself, just leaves extraction_status at "failed"
    for POST /{id}/reprocess to retry. Backfills company/title/location
    only where they were left at their unset default, never overwrites
    a value actually typed in. On success, also (best-effort, never
    blocking the posting save) resolves this posting's canonical role
    family and indexes it for semantic search. bypass_cache is set by
    Reprocess, so the posting is read again rather than replayed. note
    is kept in extraction_error on success, for a read that only partly
    worked (screenshots that could not be read next to usable text).
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
        posting.extraction_error = note
        posting.extraction_error_kind = None
        posting.extracted_at = dt.datetime.now(dt.UTC)
    except JobExtractionError as e:
        posting.extraction_status = "failed"
        posting.extraction_error = str(e)
        posting.extraction_error_kind = None
    except Exception as e:
        logger.warning("job posting extraction failed for id=%s", posting.id, exc_info=True)
        posting.extraction_status = "failed"
        posting.extraction_error = str(e)
        posting.extraction_error_kind = error_kind(e)
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
    posting.extraction_error_kind = None
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


def _job_key(posting_id: int) -> str:
    return f"job_posting_read:{posting_id}"


def _reading(posting_id: int) -> bool:
    state = jobs.snapshot(_job_key(posting_id))
    return state is not None and state["running"]


def _start_extraction(posting_id: int, bypass_cache: bool = False) -> bool:
    """Reads the posting in a worker thread (app/core/jobs.py), so saving
    returns at once and the page polls until extraction_status leaves
    "pending". False when a read of this posting is already running: a
    second click or a resent request never starts another.
    """
    return jobs.start(
        _job_key(posting_id), lambda job: _read_posting(posting_id, bypass_cache=bypass_cache)
    )


def _read_posting(posting_id: int, bypass_cache: bool = False) -> None:
    """The worker. Reads the screenshots when the posting has some whose
    text never landed in raw_text, otherwise extracts from the saved text.
    Whatever goes wrong (the posting deleted mid-read, say), the row never
    stays "pending".
    """
    db = get_db()
    try:
        posting = db.get(JobPosting, posting_id)
        if posting is None:
            return
        try:
            if _needs_image_read(posting):
                _read_images(posting, db)
            else:
                _run_extraction(posting, db, bypass_cache=bypass_cache)
        except Exception:
            logger.warning("reading job posting id=%s failed", posting_id, exc_info=True)
            db.rollback()
            db.execute(
                update(JobPosting)
                .where(JobPosting.id == posting_id, JobPosting.extraction_status == _READING)
                .values(
                    extraction_status="failed",
                    extraction_error="Internal error, see server logs.",
                    extraction_error_kind=None,
                )
            )
            db.commit()
    finally:
        db.close()


def _typed_text(posting: JobPosting) -> str:
    """What the person typed next to the screenshots, or "" when it was
    only links. Sent along with the images, and kept as the posting's
    text when the images cannot be read."""
    text = (posting.source_text or "").strip()
    return text if _URL_RE.sub("", text).strip() else ""


def _needs_image_read(posting: JobPosting) -> bool:
    """Screenshots whose transcription is not in raw_text yet: just saved,
    or an earlier read failed."""
    return bool(_image_paths(posting)) and (
        posting.raw_text_quarantined.strip() == _typed_text(posting)
    )


def _read_images(posting: JobPosting, db: Session) -> None:
    """One call over every saved screenshot, the typed text as context.
    When the images cannot be read the typed text alone still makes a
    posting, with a note saying the screenshots were skipped."""
    typed = _typed_text(posting)
    images: list[tuple[bytes, str]] = []
    for path in _image_paths(posting):
        stored = Path(path)
        data = stored.read_bytes() if stored.is_file() else b""
        mime = sniff_image_type(data)
        if mime is not None:
            images.append((data, mime))
    try:
        if not images:
            raise ScreenshotExtractionError("the screenshots saved with this posting are missing")
        result = extract_job_posting_from_images(
            images, context_text=typed, account_id=posting.account_id
        )
    except Exception as e:
        if not isinstance(e, ScreenshotExtractionError | UnsupportedScreenshotType):
            logger.warning("screenshot read failed for posting id=%s", posting.id, exc_info=True)
        if typed:
            posting.source = "pasted"
            _run_extraction(posting, db, note=f"Could not read the screenshots: {e}")
        else:
            posting.extraction_status = "failed"
            posting.extraction_error = str(e)
            posting.extraction_error_kind = error_kind(e)
            db.commit()
        return
    posting.source = "mixed" if typed else "screenshot"
    if result.raw_text_transcribed:
        posting.raw_text_quarantined = "\n\n".join(
            part for part in (typed, result.raw_text_transcribed) if part
        )
    # The image call already extracted the fields; running text
    # extraction on the transcription would pay a second call for the
    # same answer.
    _store_extraction(posting, result.extraction, db)


def fail_interrupted_extractions() -> int:
    """Called at startup. Reads run in this process (app/core/jobs.py is
    memory only), so a posting still "pending" now lost its worker to the
    restart. Marked failed with a note to reprocess instead of showing as
    in progress forever. Returns how many were marked."""
    db = get_db()
    try:
        result = db.execute(
            update(JobPosting)
            .where(JobPosting.extraction_status == _READING)
            .values(
                extraction_status="failed",
                extraction_error=_INTERRUPTED,
                extraction_error_kind=None,
            )
        )
        db.commit()
        return int(getattr(result, "rowcount", 0) or 0)
    finally:
        db.close()


def _input_hash(text: str, images: list[bytes]) -> str:
    """Dedup key for what was sent. Text alone hashes as before; with
    screenshots it covers the image bytes too, since their text is not
    known until the background read."""
    if not images:
        return _content_hash(text)
    digest = hashlib.sha256(text.encode("utf-8"))
    for data in images:
        digest.update(b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def _find_posting(db: Session, account_id: int, content_hash: str) -> JobPosting | None:
    return db.execute(
        select(JobPosting).where(
            JobPosting.account_id == account_id, JobPosting.content_hash == content_hash
        )
    ).scalar_one_or_none()


def _save_posting(
    db: Session,
    *,
    account_id: int,
    raw_text: str,
    content_hash: str,
    source: str,
    external_id: str,
    company: str | None = None,
    title: str | None = None,
    location: str | None = None,
    apply_url: str | None = None,
    images: list[tuple[bytes, str]] | None = None,
    source_text: str | None = None,
    source_links: list[str] | None = None,
) -> tuple[JobPosting, bool]:
    """Shared core of every way in: dedup by content hash, keep any
    screenshots (bytes, file name) on disk, create the row as "pending"
    and start reading it in the background. Returns (posting, created).
    A posting this account already saved comes back as it is, read or
    still reading, and nothing new starts for it.
    """
    existing = _find_posting(db, account_id, content_hash)
    if existing is not None:
        return existing, False

    screenshot_paths: list[str] = []
    if images:
        account_dir = Path(get_settings().job_screenshot_storage_dir) / str(account_id)
        account_dir.mkdir(parents=True, exist_ok=True)
        for i, (data, name) in enumerate(images):
            stored = account_dir / f"{content_hash[:16]}_{i}_{Path(name).name}"
            stored.write_bytes(data)
            screenshot_paths.append(str(stored))

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
        extraction_status=_READING,
    )
    db.add(posting)
    try:
        db.commit()
    except IntegrityError:
        # The same input sent twice at once; the other request saved it.
        # Its screenshots have the same names and bytes, so nothing to undo.
        db.rollback()
        existing = _find_posting(db, account_id, content_hash)
        if existing is None:
            raise
        return existing, False
    db.refresh(posting)
    _start_extraction(posting.id)
    db.refresh(posting)
    return posting, True


def _check_image(data: bytes, filename: str | None) -> None:
    if sniff_image_type(data) is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{filename or 'that file'} is not a PNG, JPEG, WebP or GIF image; "
                "attach screenshots only"
            ),
        )


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
    # Set when a failed read was the AI provider's doing (app/core/llm.py's
    # error_kind), so the page can point to Manage APIs.
    extraction_error_kind: str | None = None
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
            extraction_error_kind=p.extraction_error_kind,
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
    """Saves the text and returns at once with extraction_status
    "pending"; the fields are read in the background."""
    require_confirmation("text", text_notes(body.raw_text), body.confirm_large)
    raw_text = body.raw_text.strip()
    if not raw_text:
        raise HTTPException(status_code=422, detail="no readable text to save")
    content_hash = _content_hash(raw_text)
    posting, _ = _save_posting(
        db,
        account_id=body.account_id,
        raw_text=raw_text,
        content_hash=content_hash,
        source="pasted",
        external_id=content_hash,
        source_text=raw_text,
        source_links=_find_urls(raw_text),
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
    """Screenshot ingestion: the image is kept on disk (same per-account
    storage convention as app/api/resume.py's uploads) and the posting is
    saved as "pending". In the background the LLM transcribes the visible
    posting text (see app/profile/job_screenshot_extract.py) and that
    transcription becomes raw_text_quarantined, exactly as if it had been
    pasted.
    """
    file_bytes = file.file.read()
    require_confirmation("file", file_notes(file_bytes), confirm_large)
    _check_image(file_bytes, file.filename)
    content_hash = _input_hash("", [file_bytes])
    posting, _ = _save_posting(
        db,
        account_id=account_id,
        raw_text="",
        content_hash=content_hash,
        source="screenshot",
        external_id=content_hash,
        images=[(file_bytes, file.filename or "screenshot")],
    )
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
    # Notes about the save itself (the same posting was already saved,
    # say). Problems reading the inputs show up later in extraction_error.
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
    Saves and returns at once; the posting is read in the background.

    Links are never opened. The first one is kept as the apply link, and
    the posting itself has to come from the text or the screenshots.
    """
    warnings: list[str] = []
    urls = _find_urls(links, text)
    typed = _URL_RE.sub("", text).strip()

    images: list[tuple[bytes, str]] = []
    for upload in files or []:
        data = upload.file.read()
        if not data:
            continue
        _check_image(data, upload.filename)
        images.append((data, upload.filename or "screenshot"))
    require_confirmation(
        "job posting",
        text_notes(text) + files_notes([data for data, _ in images]),
        confirm_large,
    )

    if not typed and not images:
        if urls:
            detail = (
                "Only a link was given. Paste the posting text or add a "
                "screenshot as well; the link is kept as the apply link."
            )
        else:
            detail = "Add the posting text or a screenshot."
        raise HTTPException(status_code=422, detail=detail)

    raw_text = text.strip() if typed else ""
    content_hash = _input_hash(raw_text, [data for data, _ in images])
    apply_url = urls[0] if urls else None
    posting, created = _save_posting(
        db,
        account_id=account_id,
        raw_text=raw_text,
        content_hash=content_hash,
        source=("mixed" if typed else "screenshot") if images else "pasted",
        external_id=apply_url or content_hash,
        apply_url=apply_url,
        images=images,
        source_text=text.strip(),
        source_links=urls,
    )
    if not created:
        warnings.append("This posting was already saved, so the saved one was kept.")
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
    # The type comes from the bytes, never the uploaded file name, so a
    # file named .html can never be served as a page.
    media_type = sniff_file(paths[index]) or "application/octet-stream"
    return FileResponse(paths[index], media_type=media_type)


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
    # A read already running for this posting is left to finish; the
    # posting comes back as it is, still "pending".
    if not _reading(posting_id):
        posting.extraction_status = _READING
        posting.extraction_error = None
        posting.extraction_error_kind = None
        db.commit()
        _start_extraction(posting_id, bypass_cache=True)
        db.refresh(posting)
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
