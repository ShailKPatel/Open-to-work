"""Multimodal resume analysis. Given the raw bytes of an uploaded resume,
asks the LLM (via app.core.llm's multimodal message helpers) to read the
document and return skill/keyword tags, the kinds of roles it reads as
suited for, a short summary, its work-history entries (company, title,
location, dates, bullet points, skills used there), its education entries (institution, degree,
location, dates), and its header contact block (name, location, every
email, phone number, portfolio site and social link). Quality tier, not bulk: this runs once per
upload/reprocess, not across a batch of repos, and misreading a person's
own resume is a worse failure than the extra cost of the better model.

`tags`, `experiences`, `education` and `contact` are also what
app/profile/resume_profile_merge.py folds into the account's actual
Skill/Experience/Education/contact tables after a successful extraction, on top
of the per-resume copy kept on the Resume row itself, see that module's
docstring for the dedup rules.

Supports the formats app/core/llm.py's multimodal helpers actually cover: a
PDF (sent as a file part) or an image (sent as an image part). Anything
else, .docx, .txt, etc., raises UnsupportedResumeType rather than guessing
at a parse; there is no local document-text extractor in this codebase
yet, and sending arbitrary bytes to the LLM as if they were a PDF would
either fail outright or silently misread.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

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
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["company", "title", "start_date", "end_date"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["institution", "degree", "start_date", "end_date"],
            },
        },
        "contact": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "location": {"type": "string"},
                "emails": {"type": "array", "items": {"type": "string"}},
                "phones": {"type": "array", "items": {"type": "string"}},
                "links": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "platform": {"type": "string"},
                            "url": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["platform", "url"],
                    },
                },
            },
            "required": ["name", "location", "emails", "phones", "links"],
        },
    },
    "required": ["tags", "target_roles", "summary", "experiences", "education", "contact"],
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
    "exact company name, the exact job title, `location` if stated "
    "(city/region/country or \"Remote\", empty string otherwise), "
    "start_date/end_date as "
    "\"YYYY-MM-DD\" (use \"01\" for a day or month the resume doesn't "
    "give, e.g. a resume saying only \"2021\" becomes \"2021-01-01\"), "
    "and `points`, a list of key bullet points, accomplishments, or responsibility "
    "statements listed under that role, and `skills`, the concrete skills, "
    "tools, and technologies that role's own bullets and description show "
    "being used there (each also belongs in `tags`). "
    "Leave end_date as an empty string for a role stated as current/"
    "ongoing, and leave either date as an empty string if it truly can't "
    "be determined at all. Only paid or equivalent work roles belong in "
    "`experiences`, never schooling. `education`, one entry per school, "
    "college, or university the resume lists, each with the exact "
    "institution name, the degree or program as written (e.g. \"B.Tech in "
    "Computer Science\", including any major or specialization), "
    "`location` if stated (empty string otherwise), and start_date/"
    "end_date in the same \"YYYY-MM-DD\" format as above; leave end_date "
    "empty for a program still in progress, and use the expected "
    "graduation date as end_date only when the resume gives one. "
    "`contact`, everything the resume says about how to reach its owner, "
    "usually in the header but also anywhere else in the document: "
    "`name`, the person's full name as written; `location`, their city/"
    "region/country as written (empty string if absent); `emails`, every "
    "email address, each exactly as written; `phones`, every phone "
    "number, each exactly as written including any country code; and "
    "`links`, every personal URL (portfolio or personal websites, "
    "LinkedIn, GitHub, GitLab, X/Twitter, Instagram, LeetCode, Kaggle, "
    "Medium, blogs, Behance, Dribbble, and so on), listing every one "
    "separately when there is more than one portfolio or profile. For "
    "each link give `platform`, one of \"linkedin\", \"github\", "
    "\"instagram\", \"website\" (a personal site, portfolio, or blog) or "
    "\"other\"; `url`, the full URL, adding \"https://\" when the resume "
    "prints it without a scheme and expanding a bare handle only when the "
    "platform makes the URL unambiguous; and `label`, the platform's "
    "name for an \"other\" link (e.g. \"LeetCode\", \"X\"), empty "
    "otherwise. Links to a specific project's repo or demo belong to that "
    "project, not here. Read the whole document for these, including "
    "icons with hyperlinks and footers; do not skip any. Base "
    "every claim on what the document actually shows, do not invent "
    "experience or education it doesn't contain."
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
    points: list[str] = field(default_factory=list)
    location: str | None = None
    skills: list[str] = field(default_factory=list)


@dataclass
class EducationClaim:
    institution: str
    degree: str
    location: str | None
    start_date: dt.date | None
    end_date: dt.date | None


@dataclass
class LinkClaim:
    platform: str
    url: str
    label: str | None = None


@dataclass
class ContactClaim:
    name: str | None = None
    location: str | None = None
    emails: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    links: list[LinkClaim] = field(default_factory=list)


@dataclass
class ResumeExtraction:
    tags: list[str]
    target_roles: list[str]
    summary: str
    experiences: list[ExperienceClaim]
    education: list[EducationClaim] = field(default_factory=list)
    contact: ContactClaim = field(default_factory=ContactClaim)


def _parse_date(raw: object) -> dt.date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        return None


_LINK_PLATFORMS = {"linkedin", "github", "instagram", "website", "other"}


def _clean_strings(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(v).strip() for v in raw if str(v or "").strip()]


def _parse_contact(raw: object) -> ContactClaim:
    if not isinstance(raw, dict):
        return ContactClaim()

    links = []
    for item in raw.get("links") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or "").strip()
        if not url:
            continue
        if "://" not in url:
            url = f"https://{url}"
        platform = str(item.get("platform", "") or "").strip().casefold()
        label = str(item.get("label", "") or "").strip() or None
        if platform not in _LINK_PLATFORMS:
            # An unexpected platform name is still a useful label.
            label = label or (platform.title() if platform else None)
            platform = "other"
        links.append(
            LinkClaim(platform=platform, url=url, label=label if platform == "other" else None)
        )

    return ContactClaim(
        name=str(raw.get("name", "") or "").strip() or None,
        location=str(raw.get("location", "") or "").strip() or None,
        emails=_clean_strings(raw.get("emails")),
        phones=_clean_strings(raw.get("phones")),
        links=links,
    )


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

    response = complete(
        "quality", messages, schema=_SCHEMA, account_id=account_id, purpose="resume_extract"
    )
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
        raw_points = item.get("points", [])
        points = [str(p).strip() for p in raw_points if str(p).strip()]
        location = str(item.get("location", "") or "").strip()
        experiences.append(
            ExperienceClaim(
                company=company,
                title=title,
                start_date=_parse_date(item.get("start_date")),
                end_date=_parse_date(item.get("end_date")),
                points=points,
                location=location or None,
                skills=_clean_strings(item.get("skills")),
            )
        )

    education = []
    for item in response.parsed.get("education", []):
        institution = str(item.get("institution", "")).strip()
        degree = str(item.get("degree", "")).strip()
        if not institution or not degree:
            continue
        location = str(item.get("location", "") or "").strip()
        education.append(
            EducationClaim(
                institution=institution,
                degree=degree,
                location=location or None,
                start_date=_parse_date(item.get("start_date")),
                end_date=_parse_date(item.get("end_date")),
            )
        )

    return ResumeExtraction(
        tags=tags,
        target_roles=target_roles,
        summary=summary,
        experiences=experiences,
        education=education,
        contact=_parse_contact(response.parsed.get("contact")),
    )
