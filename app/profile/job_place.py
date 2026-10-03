"""Where a job is and how it is worked: one rule shared by the job list
filters (app/api/job_postings.py) and the analytics dashboard
(app/api/job_analytics.py), so both always put a posting in the same
bucket.

A remote posting is its own place. Whatever city its location field
names, it can be done from anywhere, so it is never counted under that
city. On-site and hybrid postings are counted under each city they name.
"""

from __future__ import annotations

import re

REMOTE = "remote"
HYBRID = "hybrid"
ON_SITE = "on_site"
UNKNOWN = "unknown"

MODE_LABELS = {
    REMOTE: "Remote",
    HYBRID: "Hybrid",
    ON_SITE: "On-site",
    UNKNOWN: "Not stated",
}

_REMOTE_RE = re.compile(r"\b(remote|wfh|work from home|work-from-home|anywhere)\b", re.I)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.I)
_ON_SITE_RE = re.compile(
    r"\b(on-?site|on site|in-?office|in office|in-?person|in person|office)\b", re.I
)

# Old and new names of the same city, so both land in one bucket.
_CITY_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "gurgaon": "Gurugram",
    "new delhi": "Delhi",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "nyc": "New York",
    "new york city": "New York",
    "sf": "San Francisco",
}

# Words that describe the work mode rather than a place.
_NOT_A_PLACE = re.compile(
    r"\b(remote|wfh|work from home|hybrid|on-?site|on site|in-?office|in office|"
    r"in-?person|anywhere|multiple locations?|various locations?)\b",
    re.I,
)


def work_mode_kind(work_mode: str | None, location: str | None) -> str:
    """remote, hybrid, on_site or unknown. The extracted work mode wins;
    a blank one falls back to what the location field says, since
    postings often write "Remote, India" or "Pune (Hybrid)" there."""
    for text in (work_mode or "", location or ""):
        if _REMOTE_RE.search(text) and not _HYBRID_RE.search(text):
            return REMOTE
        if _HYBRID_RE.search(text):
            return HYBRID
        if _ON_SITE_RE.search(text):
            return ON_SITE
    return UNKNOWN


def cities(location: str | None) -> list[str]:
    """The cities a location names, in order, without duplicates.
    "Pune, Maharashtra, India" is one city (the part before the first
    comma); "Pune / Mumbai" and "Pune or Mumbai" are two."""
    if not location:
        return []
    text = re.sub(r"\([^)]*\)", " ", location)
    out: list[str] = []
    for part in re.split(r"\s*(?:/|\||;|\bor\b|\band\b|&)\s*", text, flags=re.I):
        head = part.split(",")[0]
        head = _NOT_A_PLACE.sub(" ", head)
        head = re.sub(r"^[\s\-:]+|[\s\-:]+$", "", re.sub(r"\s+", " ", head))
        if not head:
            continue
        name = _CITY_ALIASES.get(head.casefold(), head)
        if name.casefold() not in {c.casefold() for c in out}:
            out.append(name)
    return out
