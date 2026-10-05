"""Soft size limits on what one upload or paste sends to the LLM.

Nothing is refused for being large. Past a limit the endpoint answers 409
with detail {"code": LARGE_INPUT_CODE, "message": ...}, and the page asks
the person before sending the same request again with confirm_large set
(_base.html's otwFetchConfirmingLarge). The limits only decide when to
ask, so they are generous: a normal resume or job posting never reaches
them.
"""

from __future__ import annotations

import io
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)

SOFT_MAX_FILE_MB = 10
SOFT_MAX_PDF_PAGES = 10
SOFT_MAX_TEXT_CHARS = 30_000

LARGE_INPUT_CODE = "large_input"


def _pdf_pages(data: bytes) -> int | None:
    """Page count, or None when the bytes are not a PDF pypdf can read;
    a broken file is the extraction step's problem, not this check's."""
    if not data.startswith(b"%PDF"):
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:
        logger.debug("could not count PDF pages for the size check", exc_info=True)
        return None


def file_notes(data: bytes) -> list[str]:
    """What about one file is over a soft limit, as phrases for the
    message ("14 MB", "40 pages")."""
    notes = []
    size_mb = len(data) / (1024 * 1024)
    if size_mb > SOFT_MAX_FILE_MB:
        notes.append(f"{size_mb:.0f} MB")
    pages = _pdf_pages(data)
    if pages is not None and pages > SOFT_MAX_PDF_PAGES:
        notes.append(f"{pages} pages")
    return notes


def files_notes(files: list[bytes]) -> list[str]:
    """file_notes() for several files sent together, sized as one."""
    notes = []
    size_mb = sum(len(data) for data in files) / (1024 * 1024)
    if size_mb > SOFT_MAX_FILE_MB:
        notes.append(f"{size_mb:.0f} MB")
    return notes


def text_notes(text: str) -> list[str]:
    length = len(text.strip())
    return [f"{length:,} characters"] if length > SOFT_MAX_TEXT_CHARS else []


def require_confirmation(what: str, notes: list[str], confirmed: bool) -> None:
    """Raises the 409 the page turns into a confirm dialog, unless nothing
    is over a limit or the person already said yes. `what` names the input
    in the message ("file", "text")."""
    if not notes or confirmed:
        return
    raise HTTPException(
        status_code=409,
        detail={
            "code": LARGE_INPUT_CODE,
            "message": (
                f"This {what} is {' / '.join(notes)} and will use a lot of tokens. "
                "Process it anyway?"
            ),
        },
    )
