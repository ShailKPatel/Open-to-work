"""Builds the deterministic parts of a resume's template data: header
(name/location/contact/social links), experience (every role, compulsory
and sequential: job history isn't something an LLM gets to pick and
choose from, for accuracy reasons), and
education (every entry, same posture, a degree history is a fact record
too, not a curated list). This is the "plug data into the right slot"
layer: none of this goes through the LLM,
it's a straight DB-to-template mapping.

Every role's *points* here are still the raw, complete list, every point
this account ever added, not curated. app/resume_build/orchestrator.py is
what narrows each role's points down to the best-matching subset via
semantic search (app/retrieval/search.py's search_experience_points,
scoped per role) before rendering, an account's actual point history
still lives here in full: this module always tells the truth about what
exists, only the orchestrator decides what to show for a given job.

Projects, skills, technologies, and summary are NOT built here: those
need either semantic search over a job posting (projects, skills) or an
LLM pass (summary), which belong to the orchestrator, not this module.
This module's output is one piece of the dict
app/resume_build/latex.py's render_resume() expects; the caller merges
it with whatever the orchestrator produces for the rest.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Account, Education, Experience, ExperiencePoint, SocialLink

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


def _format_month_year(d: dt.date) -> str:
    return f"{_MONTH_ABBR[d.month]} {d.year}"


def _date_range(start: dt.date | None, end: dt.date | None) -> str:
    if start is None and end is None:
        return ""
    if start is None:
        return _format_month_year(end)  # type: ignore[arg-type]
    start_text = _format_month_year(start)
    end_text = "present" if end is None else _format_month_year(end)
    return f"{start_text} -- {end_text}"


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
        display = link.label if platform == "other" and link.label else link.url
        social_items.append({"icon": icon, "text": display, "href": link.url})

    return {
        "full_name": f"{account.first_name} {account.last_name}".strip(),
        "contact_items": contact_items,
        "social_items": social_items,
    }



def build_experience_context(db: Session, account_id: int) -> list[dict[str, Any]]:
    """Every Experience row for this account, newest first (current/
    undated roles sort first, matching the ordering already used
    elsewhere, e.g. app/api/experience.py's list endpoint), each with
    every one of its ExperiencePoint rows in insertion order. All of it,
    unconditionally: the compulsory, non-picked section. "id" is included
    so the orchestrator can scope a per-role semantic search back to this
    exact role (see module docstring); the template itself never reads it.
    """
    roles = list(
        db.execute(
            select(Experience)
            .where(Experience.account_id == account_id)
            .order_by(Experience.start_date.desc().nulls_last())
        ).scalars()
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
    """Every Education row for this account, newest first, unconditionally:
    same compulsory, non-picked posture as build_experience_context above.
    Shape matches the template's `education` block exactly: institution,
    degree, date_range.
    """
    rows = list(
        db.execute(
            select(Education)
            .where(Education.account_id == account_id)
            .order_by(Education.start_date.desc().nulls_last())
        ).scalars()
    )
    return [
        {
            "institution": row.institution,
            "degree": row.degree,
            "date_range": _date_range(row.start_date, row.end_date),
        }
        for row in rows
    ]
