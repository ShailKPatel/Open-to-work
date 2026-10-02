"""Multimodal resume analysis. Given the raw bytes of an uploaded resume,
asks the LLM (via app.core.llm's multimodal message helpers) to read the
document and return skill/keyword tags, the kinds of roles it reads as
suited for, a short summary, its work-history entries (company, title,
location, dates, bullet points, skills used there), its education entries (institution, degree,
location, dates), and its header contact block (name, location, every
email, phone number, portfolio site, social link and certificate link,
each link named). Quality tier, not bulk: this runs once per
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

import io
from dataclasses import dataclass, field

from pypdf import PdfReader

from app.core.llm import complete, file_part, image_part, system_message
from app.profile.month_year import normalize_or_none

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
                    "grade": {"type": "string"},
                    "details": {"type": "array", "items": {"type": "string"}},
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
    "company name as written but without any descriptor the resume adds "
    "beside it (\"Acme (Fintech Startup)\" becomes \"Acme\"), the exact "
    "job title, `location` if stated "
    "(city/region/country or \"Remote\", empty string otherwise), "
    "start_date/end_date as a three-letter month and the year, e.g. "
    "\"Mar 2021\", or the year alone, e.g. \"2021\", when the resume "
    "gives no month (never invent one), "
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
    "end_date in the same \"Mar 2021\" format as above; leave end_date "
    "empty for a program still in progress, and use the expected "
    "graduation date as end_date only when the resume gives one. "
    "Also per school, only when the resume states them: `grade`, the "
    "GPA, CGPA, SPI, percentage or class exactly as written (e.g. "
    "\"CGPA: 8.9/10\"), empty string otherwise; and `details`, each "
    "other line listed under that school (relevant coursework, rank, "
    "honors, thesis) as written, empty list otherwise. Never infer "
    "either one. "
    "`contact`, everything the resume says about how to reach its owner, "
    "usually in the header but also anywhere else in the document: "
    "`name`, the person's full name as written; `location`, their city/"
    "region/country as written (empty string if absent); `emails`, every "
    "email address, each exactly as written; `phones`, every phone "
    "number, each exactly as written including any country code; and "
    "`links`, every personal URL (portfolio or personal websites, "
    "LinkedIn, GitHub, GitLab, X/Twitter, Instagram, LeetCode, Kaggle, "
    "Medium, blogs, Behance, Dribbble, and so on), plus every link to the "
    "person's certificates or credentials (a Credly or Accredible badge, a "
    "certificate verification URL, a page or folder of certificates, "
    "including links in a Certifications section), listing every one "
    "separately when there is more than one. For each link give "
    "`platform`, one of \"linkedin\", \"github\", \"instagram\" or "
    "\"other\" (\"website\" only for a site you truly can't name); "
    "`url`, the full URL, adding \"https://\" when the resume prints it "
    "without a scheme and expanding a bare handle only when the platform "
    "makes the URL unambiguous; and `label`, the short name the link "
    "goes by on a resume header, required for \"other\" and empty "
    "otherwise: the word the resume itself shows for it when it shows "
    "one (\"Portfolio\", \"Certificates\"), else what it is, "
    "\"Portfolio\" for a personal or portfolio site, \"Blog\" for a "
    "blog, \"Certificates\" for a page of certificates, the "
    "certification's own short name for a single certificate (e.g. "
    "\"AWS Solutions Architect\"), or the platform's name for a profile "
    "(e.g. \"LeetCode\", \"X\"). Links to a specific project's repo or "
    "demo, or to a paper, belong to that project, not here. Read the "
    "whole document for these, including icons with hyperlinks and "
    "footers; do not skip any. A link can hide behind a word or icon that "
    "doesn't print its URL, so when the message lists the hyperlinks "
    "embedded in the file, check each one against the page. Base "
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
    start_date: str | None
    end_date: str | None
    points: list[str] = field(default_factory=list)
    location: str | None = None
    skills: list[str] = field(default_factory=list)


@dataclass
class EducationClaim:
    institution: str
    degree: str
    location: str | None
    start_date: str | None
    end_date: str | None
    grade: str | None = None
    details: list[str] = field(default_factory=list)


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


def _parse_date(raw: object) -> str | None:
    return normalize_or_none(raw)


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
        elif platform == "website" and label:
            # A named site ("Portfolio") is a custom link, which is the
            # only kind whose name shows on the resume and the Links page.
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


def _embedded_links(pdf_bytes: bytes) -> list[str]:
    """Every web URL the PDF links to, in page order, deduplicated. A link
    behind a word like "Certificates" has no printed URL to read off the
    page, so these go to the LLM as text alongside the file. Best effort:
    a PDF pypdf can't parse just gets no list."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        urls: list[str] = []
        for page in reader.pages:
            for annot in page.get("/Annots") or []:
                obj = annot.get_object()
                action = obj.get("/A")
                if obj.get("/Subtype") != "/Link" or action is None:
                    continue
                uri = str(action.get_object().get("/URI") or "").strip()
                if uri.lower().startswith(("http://", "https://")) and uri not in urls:
                    urls.append(uri)
        return urls
    except Exception:
        return []


def extract_resume(
    file_bytes: bytes, mime_type: str, account_id: int | None = None
) -> ResumeExtraction:
    mime_type = (mime_type or "").lower()
    intro = "Here is the resume to analyze."
    if mime_type == _PDF_MIME:
        attachment = file_part(file_bytes, mime_type=_PDF_MIME)
        embedded = _embedded_links(file_bytes)
        if embedded:
            intro += "\n\nHyperlinks embedded in the file:\n" + "\n".join(embedded)
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
                {"type": "text", "text": intro},
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
                grade=str(item.get("grade", "") or "").strip() or None,
                details=_clean_strings(item.get("details")),
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
