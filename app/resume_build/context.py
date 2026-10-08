"""The parts of a resume's template data that never go through the LLM:
the header (name, location, contact, links), every role and every
education entry, mapped straight from the database.

Roles come with all their points; orchestrator.py narrows those to the
best matches for the job. Projects, skills and the summary are built by
the orchestrator too, and merged with this module's output before
latex.py's render_resume().
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import (
    Account,
    Education,
    Experience,
    ExperiencePoint,
    Repository,
    SkillArchive,
    SocialLink,
)
from app.profile.month_year import newest_first, parse_quiet

_MONTH_ABBR = {
    1: "Jan.", 2: "Feb.", 3: "Mar.", 4: "Apr.", 5: "May", 6: "Jun.",
    7: "Jul.", 8: "Aug.", 9: "Sep.", 10: "Oct.", 11: "Nov.", 12: "Dec.",
}

_PLATFORM_ICONS = {
    "github": r"\faGithub",
    "linkedin": r"\faLinkedin",
    "instagram": r"\faInstagram",
    "website": r"\faGlobe",
}
_DEFAULT_SOCIAL_ICON = r"\faLink"
_HANDLE_PLATFORMS = frozenset({"github", "linkedin", "instagram"})
# Custom ("other") links are named by the user, so match the common names
# by prefix: "Portfolio", "Certificates", "Certifications", "Blog".
_CUSTOM_LABEL_ICONS = (
    ("portfolio", r"\faBriefcase"),
    ("certif", r"\faCertificate"),
    ("blog", r"\faBlog"),
)


def _format_month_year(value: str | None) -> str:
    """Stored "mar 2026" printed as "Mar. 2026", "2026" as itself; an
    unreadable value is printed as stored rather than dropped."""
    parsed = parse_quiet(value)
    if parsed is None:
        return (value or "").strip()
    year, month = parsed
    return str(year) if month is None else f"{_MONTH_ABBR[month]} {year}"


def _date_range(start: str | None, end: str | None) -> str:
    start_text = _format_month_year(start)
    end_text = _format_month_year(end)
    if not start_text:
        return end_text
    return f"{start_text} -- {end_text or 'present'}"


def _absolute_url(url: str) -> str:
    """A link saved as "linkedin.com/in/x" still needs a scheme to be
    clickable in the PDF."""
    url = url.strip()
    if url and "://" not in url and not url.startswith("mailto:"):
        return f"https://{url}"
    return url


def _link_display(platform: str, url: str) -> str:
    """What the header prints for a link, the href stays the full URL.
    Profile platforms (LinkedIn, GitHub, Instagram) print just the
    handle; anything else prints the URL without its scheme, "www." or
    trailing slash, e.g. "jane-doe.dev"."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.strip("/")
    if platform in _HANDLE_PLATFORMS and path:
        segments = path.split("/")
        # linkedin.com/in/<handle>, linkedin.com/company/<handle>
        if platform == "linkedin" and len(segments) > 1 and segments[0] in ("in", "company"):
            return segments[1]
        return segments[0].lstrip("@")
    return f"{host}/{path}" if path else host or url


def build_header_context(
    account: Account,
    social_links: list[SocialLink],
    selected_email: str | None = None,
    selected_phone: str | None = None,
) -> dict[str, Any]:
    """contact_items: location/email/phone, in that fixed order, only the
    ones actually set. social_items: github (from Account.github_username,
    the sync identity, not a SocialLink row, see that model's docstring)
    first if present, then every SocialLink row in whatever order the
    account added them. Allows selecting a specific email/phone for custom
    resume builds.
    """
    contact_items: list[dict[str, Any]] = []
    if account.contact_location:
        contact_items.append(
            {"icon": r"\faMapMarker*", "text": account.contact_location, "href": None}
        )

    email_to_use = selected_email if selected_email is not None else account.contact_email
    if email_to_use:
        contact_items.append(
            {
                "icon": r"\faEnvelope",
                "text": email_to_use,
                "href": f"mailto:{email_to_use}",
            }
        )

    phone_to_use = selected_phone if selected_phone is not None else account.contact_phone
    if phone_to_use:
        contact_items.append(
            {
                "icon": r"\faPhone",
                "text": phone_to_use,
                "href": f"tel:{phone_to_use}",
            }
        )

    social_items: list[dict[str, Any]] = []
    if account.github_username:
        social_items.append(
            {
                "icon": _PLATFORM_ICONS["github"],
                "text": account.github_username,
                "href": f"https://github.com/{account.github_username}",
            }
        )
    for link in social_links:
        platform = (link.platform or "").strip().lower()
        icon = _PLATFORM_ICONS.get(platform, _DEFAULT_SOCIAL_ICON)
        if platform == "other" and link.label:
            name = link.label.strip().lower()
            icon = next(
                (i for prefix, i in _CUSTOM_LABEL_ICONS if name.startswith(prefix)), icon
            )
        href = _absolute_url(link.url)
        if platform == "other" and link.label:
            display = link.label
        else:
            display = _link_display(platform, href)
        social_items.append({"icon": icon, "text": display, "href": href})

    return {
        "full_name": f"{account.first_name} {account.last_name}".strip(),
        "contact_items": contact_items,
        "social_items": social_items,
    }



def excluded_sources(db: Session, account_id: int) -> tuple[set[int], set[int]]:
    """Ids of this account's projects and roles marked exclude_from_resume:
    (repository ids, experience ids). Resume building drops anything that
    comes only from these, so a search hit or skill backed by nothing else
    never reaches the model.
    """
    repo_ids = set(
        db.execute(
            select(Repository.id).where(
                Repository.account_id == account_id, Repository.exclude_from_resume.is_(True)
            )
        ).scalars()
    )
    experience_ids = set(
        db.execute(
            select(Experience.id).where(
                Experience.account_id == account_id, Experience.exclude_from_resume.is_(True)
            )
        ).scalars()
    )
    return repo_ids, experience_ids


def archived_skill_keys(db: Session, account_id: int) -> set[str]:
    """Casefolded names of the skills this account archived by hand
    (SkillArchive). A skill archived only because all of its sources are
    is covered by excluded_sources instead.
    """
    return set(
        db.execute(
            select(SkillArchive.name_key).where(SkillArchive.account_id == account_id)
        ).scalars()
    )


def build_experience_context(db: Session, account_id: int) -> list[dict[str, Any]]:
    """Every Experience row for this account not marked
    exclude_from_resume, newest first (current/
    undated roles sort first, matching the ordering already used
    elsewhere, e.g. app/api/experience.py's list endpoint), each with
    every one of its ExperiencePoint rows in insertion order. All of it,
    unconditionally: the compulsory, non-picked section. "id" is included
    so the orchestrator can scope a per-role semantic search back to this
    exact role (see module docstring); the template itself never reads it.
    """
    roles = newest_first(
        list(
            db.execute(
                select(Experience).where(
                    Experience.account_id == account_id,
                    Experience.exclude_from_resume.is_(False),
                )
            ).scalars()
        )
    )
    if not roles:
        return []

    points_by_experience: dict[int, list[str]] = {}
    points = list(
        db.execute(
            select(ExperiencePoint)
            .where(ExperiencePoint.experience_id.in_([r.id for r in roles]))
            .order_by(ExperiencePoint.order_index.asc())
        ).scalars()
    )
    for p in points:
        points_by_experience.setdefault(p.experience_id, []).append(p.text)

    return [
        {
            "id": role.id,
            "title": role.title,
            "company": role.company,
            "location": role.location,
            "date_range": _date_range(role.start_date, role.end_date),
            "points": points_by_experience.get(role.id, []),
        }
        for role in roles
    ]


def build_education_context(db: Session, account_id: int) -> list[dict[str, Any]]:
    """Every Education row for this account not marked exclude_from_resume,
    newest first: same compulsory, non-picked posture as
    build_experience_context above.
    Shape matches the template's `education` block: institution, degree,
    date_range, grade (None when unset) and details (empty when unset;
    the template skips both then, so they take no space), plus "id" so a
    caller can honor the account holder's own choice to leave an entry
    off one resume (the template never reads it).
    """
    rows = newest_first(
        list(
            db.execute(
                select(Education).where(
                    Education.account_id == account_id, Education.exclude_from_resume.is_(False)
                )
            ).scalars()
        )
    )
    return [
        {
            "id": row.id,
            "institution": row.institution,
            "degree": row.degree,
            "date_range": _date_range(row.start_date, row.end_date),
            "grade": row.grade,
            "details": list(row.details or []),
        }
        for row in rows
    ]
