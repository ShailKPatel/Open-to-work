"""Turns a posting's experience text, as extracted (app/profile/job_extract.py's
experience_required), into a span of years so pay can be plotted against
how much experience a job asks for. Deterministic, like app/profile/salary.py.

Handles "3+ years", "2-4 yrs", "2 to 4 years", "at least 5 years",
"up to 2 years", "6 months", "No experience required" and "Freshers".
When the text says nothing usable, the seniority label stands in
("Junior", "Senior", ...) and the result is marked estimated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperienceYears:
    min: float
    # None for an open-ended ask like "3+ years".
    max: float | None
    # True when read off the seniority label rather than stated years.
    estimated: bool = False


_NONE = re.compile(
    r"\bno\s+(prior\s+)?experience\b|\bfreshers?\b|\bentry[\s-]level\b|\bnot\s+required\b", re.I
)
_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"(years?|yrs?|months?|mos?)\b"
_RANGE = re.compile(rf"{_NUM}\s*(?:-|–|—|to)\s*{_NUM}\s*\+?\s*{_UNIT}", re.I)
_UPTO = re.compile(rf"\b(?:up\s*to|upto|max(?:imum)?|less\s+than|under)\s+{_NUM}\s*{_UNIT}", re.I)
_SINGLE = re.compile(rf"{_NUM}\s*\+?\s*{_UNIT}", re.I)

# Typical minimum years behind each seniority label, used only when the
# posting gives no years.
_SENIORITY_YEARS = [
    (re.compile(r"\bintern", re.I), 0.0),
    (re.compile(r"\b(junior|jr|entry|graduate|associate)\b", re.I), 1.0),
    (re.compile(r"\b(mid|intermediate)\b", re.I), 3.0),
    (re.compile(r"\b(staff|principal|architect)\b", re.I), 8.0),
    (re.compile(r"\b(lead|manager)\b", re.I), 7.0),
    (re.compile(r"\b(senior|sr)\b", re.I), 5.0),
]


def _years(value: str, unit: str) -> float:
    n = float(value)
    return round(n / 12, 2) if unit.lower().startswith("mo") else n


def parse_experience(text: str | None, seniority: str | None = None) -> ExperienceYears | None:
    """Years of experience a posting asks for, or None when neither the
    text nor the seniority says."""
    text = (text or "").strip()
    if text:
        if m := _RANGE.search(text):
            low, high = _years(m.group(1), m.group(3)), _years(m.group(2), m.group(3))
            return ExperienceYears(min(low, high), max(low, high))
        if m := _UPTO.search(text):
            return ExperienceYears(0.0, _years(m.group(1), m.group(2)))
        if m := _SINGLE.search(text):
            years = _years(m.group(1), m.group(2))
            # "2 years" reads as a floor the same way "2+ years" does.
            return ExperienceYears(years, None if years else 0.0)
        if _NONE.search(text):
            return ExperienceYears(0.0, 0.0)
    for pattern, years in _SENIORITY_YEARS:
        if seniority and pattern.search(seniority):
            return ExperienceYears(years, None, estimated=True)
    return None
