"""Shared resume-file ingestion: saving an uploaded file to disk, creating
its Resume row, and (best-effort) running multimodal extraction against it,
including folding whatever skills and work history it found into the
account's actual profile data (app/profile/resume_profile_merge.py). Used
from both POST /accounts's optional signup resume field
(app/api/accounts.py) and POST /api/resume (app/api/resume.py), so there is
exactly one path that turns "a person just handed us a resume file" into a
stored, tagged Resume row, not two copies of the same steps that could
drift apart.
"""

from __future__ import annotations

import datetime as dt
import logging
import mimetypes
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.db import Account, Resume
from app.core.settings import get_settings
from app.profile.resume_extract import ResumeExtraction, UnsupportedResumeType, extract_resume

logger = logging.getLogger(__name__)


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
    and runs extraction against it. Extraction failing (unsupported file
    type, LLM error, rate limit) never loses the upload itself, it just
    leaves the row at extraction_status "failed" or "unsupported_type" for
    POST /api/resume/{id}/reprocess to retry later.

    `name` is the person-chosen label (see Resume.name's docstring), left
    unset when not given rather than defaulting to the filename, the UI
    falls back to filename for display on its own.

    Caller owns opening/closing `db` (same convention as every other
    router in this app).
    """
    safe_name = Path(upload.filename or "resume").name
    data = upload.file.read()
    mime_type = (
        upload.content_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    )

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
    db.refresh(row)

    run_extraction(db, row, data)
    return row


def run_extraction(db: Session, row: Resume, file_bytes: bytes | None = None) -> Resume:
    """(Re)runs extraction against an already-saved Resume row. Reads the
    file back off disk when `file_bytes` isn't already in hand (the
    reprocess path, ingest_resume above already has the bytes from the
    original upload, no need to make it re-read its own write).
    """
    if file_bytes is None:
        file_bytes = Path(row.stored_path).read_bytes()

    extraction: ResumeExtraction | None = None
    try:
        extraction = extract_resume(file_bytes, row.mime_type, account_id=row.account_id)
        row.tags_json = extraction.tags
        row.target_roles_json = extraction.target_roles
        row.summary = extraction.summary
        row.extraction_status = "extracted"
        row.extraction_error = None
        row.extracted_at = dt.datetime.now(dt.UTC)
    except UnsupportedResumeType as e:
        row.extraction_status = "unsupported_type"
        row.extraction_error = str(e)
    except Exception as e:
        logger.warning("resume extraction failed for resume id=%s", row.id, exc_info=True)
        row.extraction_status = "failed"
        row.extraction_error = str(e)
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

            merge_resume_into_profile(db, row.account_id, extraction)
        except Exception:
            logger.exception(
                "could not merge resume id=%s into profile skills/experience; continuing",
                row.id,
            )

    return row
