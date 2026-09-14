"""Multimodal resume analysis. Given the raw bytes of an uploaded resume,
asks the LLM (via app.core.llm's multimodal message helpers) to read the
document and return skill/keyword tags, the kinds of roles it reads as
suited for, a short summary, and its work-history entries (company, title,
dates). Quality tier, not bulk: this runs once per upload/reprocess, not
across a batch of repos, and misreading a person's own resume is a worse
failure than the extra cost of the better model.

`tags` and `experiences` are also what app/profile/resume_profile_merge.py
folds into the account's actual Skill/Experience tables after a successful
extraction, not just stored on the Resume row, see that module's docstring
for the dedup rules.

Supports the formats app/core/llm.py's multimodal helpers actually cover: a
PDF (sent as a file part) or an image (sent as an image part). Anything
else, .docx, .txt, etc., raises UnsupportedResumeType rather than guessing
at a parse; there is no local document-text extractor in this codebase
yet, and sending arbitrary bytes to the LLM as if they were a PDF would
either fail outright or silently misread.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from app.core.llm import complete, file_part, image_part, system_message

_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
        "target_roles": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["company", "title", "start_date", "end_date"],
            },
        },
    },
    "required": ["tags", "target_roles", "summary", "experiences"],
}

_SYSTEM_PROMPT = (
    "You read an uploaded resume/CV document and describe it structurally, "
    "for someone building a searchable library of resume versions to pick "
    "from later. Extract: `tags`, a list of concrete skills, tools, "
    "technologies, and practices the resume actually demonstrates (not "
    "generic soft-skill filler); `target_roles`, a short list of the kinds "
    "of job titles or role types this specific resume reads as written "
    "for, e.g. \"Backend Engineer\", \"Data Scientist\", \"Engineering "
    "Manager\", inferred from its emphasis and framing, not just its most "
    "recent title; `summary`, two or three sentences describing what this "
    "resume is strong at and who it's a good fit for; and `experiences`, "
    "one entry per work-history role the resume lists, each with the "
    "exact company name, the exact job title, and start_date/end_date as "
    "\"YYYY-MM-DD\" (use \"01\" for a day or month the resume doesn't "
    "give, e.g. a resume saying only \"2021\" becomes \"2021-01-01\"). "
    "Leave end_date as an empty string for a role stated as current/"
    "ongoing, and leave either date as an empty string if it truly can't "
    "be determined at all. Do not include education, only paid or "
    "equivalent work roles. Base every claim on what the document "
    "actually shows, do not invent experience it doesn't contain."
)

_PDF_MIME = "application/pdf"
_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}


class UnsupportedResumeType(Exception):
    """The uploaded file's mime type isn't one app/core/llm.py's multimodal
    helpers can read (only PDF and common image types today), not a
    failure of the LLM call itself, so kept distinct from any other
    extraction error the caller needs to report separately."""


@dataclass
class ExperienceClaim:
    company: str
    title: str
    start_date: dt.date | None
    end_date: dt.date | None


@dataclass
class ResumeExtraction:
    tags: list[str]
    target_roles: list[str]
    summary: str
    experiences: list[ExperienceClaim]


def _parse_date(raw: object) -> dt.date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


def extract_resume(
    file_bytes: bytes, mime_type: str, account_id: int | None = None
) -> ResumeExtraction:
    mime_type = (mime_type or "").lower()
    if mime_type == _PDF_MIME:
        attachment = file_part(file_bytes, mime_type=_PDF_MIME)
    elif mime_type in _IMAGE_MIMES:
        attachment = image_part(file_bytes, mime_type=mime_type)
    else:
        raise UnsupportedResumeType(
            f"can't read {mime_type or 'this file type'} yet, upload a PDF or image instead"
        )

    messages = [
        system_message(_SYSTEM_PROMPT),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the resume to analyze."},
                attachment,
            ],
        },
    ]

    response = complete("quality", messages, schema=_SCHEMA, account_id=account_id)
    if response.parsed is None:
        raise ValueError("LLM response for resume extraction was not valid JSON")

    tags = [str(t).strip() for t in response.parsed.get("tags", []) if str(t).strip()]
    target_roles = [
        str(r).strip() for r in response.parsed.get("target_roles", []) if str(r).strip()
    ]
    summary = str(response.parsed.get("summary", "")).strip()

    experiences = []
    for item in response.parsed.get("experiences", []):
        company = str(item.get("company", "")).strip()
        title = str(item.get("title", "")).strip()
        if not company or not title:
            continue
        experiences.append(
            ExperienceClaim(
                company=company,
                title=title,
                start_date=_parse_date(item.get("start_date")),
                end_date=_parse_date(item.get("end_date")),
            )
        )

    return ResumeExtraction(
        tags=tags, target_roles=target_roles, summary=summary, experiences=experiences
    )
