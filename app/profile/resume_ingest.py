"""Shared resume-file ingestion: saving an uploaded file to disk, creating
its Resume row, and (best-effort, in a worker thread) running multimodal
extraction against it,
including folding whatever skills, work history, education and contact
details it found
into the account's actual profile data (app/profile/resume_profile_merge.py).
Used from both POST /accounts's optional signup resume field
(app/api/accounts.py) and POST /api/resume (app/api/resume.py), so there is
exactly one path that turns "a person just handed us a resume file" into a
stored, tagged Resume row, not two copies of the same steps that could
drift apart.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import mimetypes
from collections.abc import Callable
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core import jobs
from app.core.db import Account, Resume, get_db
from app.core.filetypes import PDF, sniff_type
from app.core.llm import error_kind
from app.core.settings import get_settings
from app.profile.resume_extract import (
    ContactClaim,
    EducationClaim,
    ExperienceClaim,
    ResumeExtraction,
    UnsupportedResumeType,
    extract_resume,
)

logger = logging.getLogger(__name__)


def _experience_dict(claim: ExperienceClaim) -> dict:
    return {
        "company": claim.company,
        "title": claim.title,
        "location": claim.location,
        "start_date": claim.start_date,
        "end_date": claim.end_date,
        "points": list(claim.points),
    }


def _education_dict(claim: EducationClaim) -> dict:
    return {
        "institution": claim.institution,
        "degree": claim.degree,
        "location": claim.location,
        "start_date": claim.start_date,
        "end_date": claim.end_date,
        "grade": claim.grade,
        "details": list(claim.details),
    }


def _contact_dict(claim: ContactClaim) -> dict:
    return {
        "name": claim.name,
        "location": claim.location,
        "emails": list(claim.emails),
        "phones": list(claim.phones),
        "links": [
            {"platform": link.platform, "url": link.url, "label": link.label}
            for link in claim.links
        ],
    }


class DuplicateResumeError(Exception):
    """The uploaded file is byte-for-byte one this account already has."""

    def __init__(self, existing: Resume) -> None:
        super().__init__(f"same file as resume id={existing.id}")
        self.existing = existing


def find_same_file(db: Session, account_id: int, data: bytes) -> Resume | None:
    """This account's stored resume whose file holds exactly `data`, if
    any. Only same-size files are read back, so an upload costs at most a
    read of the one or two files it could possibly equal.
    """
    candidates = db.execute(
        select(Resume).where(Resume.account_id == account_id, Resume.file_size == len(data))
    ).scalars()
    for row in candidates:
        try:
            if row.stored_path and Path(row.stored_path).read_bytes() == data:
                return row
        except OSError:
            continue
    return None


class GeneratedResumeError(Exception):
    """The upload is a resume this app built. Its wording was tailored to
    one job by the LLM, so reading it back would fold that wording into
    the profile as if the person had written it."""

    def __init__(self, existing: Resume | None = None) -> None:
        super().__init__(
            "This PDF was built by Open to Work, so its wording was tailored to one job. "
            "Reading it back would add that wording to your profile as if you wrote it. "
            "Generated resumes are already in your library; upload a resume you wrote instead."
        )
        self.existing = existing


def is_generated_pdf(data: bytes) -> bool:
    """True for a PDF this app compiled, by the Creator field every
    template sets (app/resume_build/latex.py's GENERATED_PDF_CREATOR)."""
    if not data.startswith(b"%PDF"):
        return False
    from pypdf import PdfReader

    from app.resume_build.latex import GENERATED_PDF_CREATOR

    try:
        metadata = PdfReader(io.BytesIO(data)).metadata
    except Exception:
        return False
    return metadata is not None and metadata.creator == GENERATED_PDF_CREATOR


def find_built_copy(db: Session, account_id: int, data: bytes) -> Resume | None:
    """This account's resume whose compiled PDF is exactly `data`. Catches
    a download of a build made before generated PDFs carried a Creator."""
    rows = db.execute(
        select(Resume).where(Resume.account_id == account_id, Resume.compiled_path.is_not(None))
    ).scalars()
    for row in rows:
        try:
            path = Path(row.compiled_path or "")
            if path.stat().st_size == len(data) and path.read_bytes() == data:
                return row
        except OSError:
            continue
    return None


def check_not_generated(db: Session, account_id: int, data: bytes) -> None:
    """Raises GeneratedResumeError when `data` is a resume this app built."""
    built = find_built_copy(db, account_id, data)
    if built is not None or is_generated_pdf(data):
        raise GeneratedResumeError(built)


def _file_type(data: bytes, safe_name: str, declared: str | None) -> str:
    """A PDF or image is typed by its bytes. Anything else keeps the type
    it came with, for download only, unless it claimed to be a PDF or an
    image: then it is plain bytes, so it is neither sent to the resume
    reader nor served back as something it is not."""
    sniffed = sniff_type(data)
    if sniffed is not None:
        return sniffed
    declared = (declared or mimetypes.guess_type(safe_name)[0] or "").lower()
    if not declared or declared == PDF or declared.startswith("image/"):
        return "application/octet-stream"
    return declared


def ingest_resume(
    db: Session,
    account_id: int,
    upload: UploadFile,
    *,
    name: str | None = None,
    notes: str | None = None,
) -> Resume:
    """Saves the uploaded file to disk, creates its Resume row, mirrors it
    onto Account.resume_filename/resume_path (see that field's docstring),
    and starts extraction against it in a worker thread, so the caller
    answers at once with the row still "pending" whatever the AI provider
    is doing. Extraction failing (unsupported file type, LLM error, rate
    limit) never loses the upload itself, it just leaves the row at
    extraction_status "failed" or "unsupported_type" for
    POST /api/resume/{id}/reprocess to retry later.

    `name` is the person-chosen label (see Resume.name's docstring), left
    unset when not given rather than defaulting to the filename, the UI
    falls back to filename for display on its own.

    Raises DuplicateResumeError, before anything is saved, when the file
    is identical to one already in this account's library: a second copy
    would only repeat the same extraction and profile merge. Raises
    GeneratedResumeError, also before saving, for a resume this app built.

    Caller owns opening/closing `db` (same convention as every other
    router in this app).
    """
    safe_name = Path(upload.filename or "resume").name
    data = upload.file.read()
    existing = find_same_file(db, account_id, data)
    if existing is not None:
        raise DuplicateResumeError(existing)
    check_not_generated(db, account_id, data)
    mime_type = _file_type(data, safe_name, upload.content_type)

    row = Resume(
        account_id=account_id,
        filename=safe_name,
        name=(name.strip() if name else None) or None,
        mime_type=mime_type,
        file_size=len(data),
        notes=(notes.strip() if notes else None) or None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    account_dir = Path(get_settings().resume_storage_dir) / str(account_id)
    account_dir.mkdir(parents=True, exist_ok=True)
    dest = account_dir / f"{row.id}_{safe_name}"
    dest.write_bytes(data)
    row.stored_path = str(dest)

    account = db.get(Account, account_id)
    if account is not None:
        account.resume_filename = row.filename
        account.resume_path = row.stored_path
    db.commit()

    start_extraction(row.id)
    db.refresh(row)
    return row


_INTERRUPTED = "The app restarted before this resume was read. Use Retry to read it again."


def _job_key(resume_id: int) -> str:
    return f"resume_read:{resume_id}"


def _start_worker(key: str, work: Callable[[], None]) -> bool:
    """Seam for tests, which run the work inline instead."""
    return jobs.start(key, lambda job: work())


def start_extraction(resume_id: int, bypass_cache: bool = False) -> bool:
    """Reads the saved file in a worker thread (app/core/jobs.py). False
    when a read of this resume is already running: a second click never
    starts another."""
    return _start_worker(
        _job_key(resume_id), lambda: _read_resume(resume_id, bypass_cache=bypass_cache)
    )


def is_reading(resume_id: int) -> bool:
    state = jobs.snapshot(_job_key(resume_id))
    return state is not None and state["running"]


def _read_resume(resume_id: int, bypass_cache: bool = False) -> None:
    """The worker, with its own session. Whatever goes wrong (the file
    gone from disk, the row deleted mid-read), the row never stays
    "pending"."""
    db = get_db()
    try:
        row = db.get(Resume, resume_id)
        if row is None or not row.stored_path:
            return
        try:
            run_extraction(db, row, bypass_cache=bypass_cache)
        except Exception:
            logger.warning("reading resume id=%s failed", resume_id, exc_info=True)
            db.rollback()
            db.execute(
                update(Resume)
                .where(Resume.id == resume_id, Resume.extraction_status == "pending")
                .values(
                    extraction_status="failed",
                    extraction_error="Internal error, see server logs.",
                    extraction_error_kind=None,
                )
            )
            db.commit()
    finally:
        db.close()


def fail_interrupted_reads() -> int:
    """Called at startup. Reads run in this process only, so an uploaded
    resume still "pending" now lost its worker to the restart. Marked
    failed with a note to retry instead of showing as in progress
    forever. Generated resumes (no stored file) use "pending" for an
    unfinished build and are left alone. Returns how many were marked."""
    db = get_db()
    try:
        result = db.execute(
            update(Resume)
            .where(Resume.extraction_status == "pending", Resume.stored_path != "")
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


def run_extraction(
    db: Session,
    row: Resume,
    file_bytes: bytes | None = None,
    *,
    bypass_cache: bool = False,
) -> Resume:
    """(Re)runs extraction against an already-saved Resume row. Reads the
    file back off disk when `file_bytes` isn't already in hand.
    bypass_cache is set by a retry, so the file is read again rather than
    replayed. A failure caused by the AI provider records its
    extraction_error_kind (app/core/llm.py's error_kind) so the page can
    say what to do about it.
    """
    if file_bytes is None:
        file_bytes = Path(row.stored_path).read_bytes()

    extraction: ResumeExtraction | None = None
    try:
        extraction = extract_resume(
            file_bytes, row.mime_type, account_id=row.account_id, bypass_cache=bypass_cache
        )
        row.tags_json = extraction.tags
        row.target_roles_json = extraction.target_roles
        row.summary = extraction.summary
        row.experiences_json = [_experience_dict(c) for c in extraction.experiences]
        row.education_json = [_education_dict(c) for c in extraction.education]
        row.contact_json = _contact_dict(extraction.contact)
        row.extraction_status = "extracted"
        row.extraction_error = None
        row.extraction_error_kind = None
        row.extracted_at = dt.datetime.now(dt.UTC)
    except UnsupportedResumeType as e:
        row.extraction_status = "unsupported_type"
        row.extraction_error = str(e)
        row.extraction_error_kind = None
    except Exception as e:
        logger.warning("resume extraction failed for resume id=%s", row.id, exc_info=True)
        row.extraction_status = "failed"
        row.extraction_error = str(e)
        row.extraction_error_kind = error_kind(e)
    db.commit()
    db.refresh(row)

    if row.extraction_status == "extracted" and extraction is not None:
        try:
            from app.retrieval.index import index_resume

            index_resume(row)
        except Exception:
            logger.exception("could not index resume id=%s into Qdrant; continuing", row.id)

        try:
            from app.profile.resume_profile_merge import merge_resume_into_profile

            merge_resume_into_profile(db, row.account_id, extraction, resume_id=row.id)
        except Exception:
            logger.exception(
                "could not merge resume id=%s into profile; continuing",
                row.id,
            )

    return row
