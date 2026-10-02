"""Month-and-year dates for Experience and Education.

A role or a degree is dated to the month, never the day, so these dates
are stored the way a resume prints them: "Mar 2026", or "2026" alone when
only the year is known. normalize() is the one place every write path
(manual add, edit, resume extraction, the startup migration) goes through,
so the database only ever holds that form.
"""

from __future__ import annotations

import datetime as dt
import re

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_MONTH_NAMES = {name.lower(): i for i, name in enumerate(MONTHS, start=1)}
_MONTH_NAMES.update(
    {
        "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
        "august": 8, "sept": 9, "september": 9, "october": 10, "november": 11,
        "december": 12,
    }
)

_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})(?:[-/.]\d{1,2})?(?:[T ].*)?$")
_MONTH_FIRST = re.compile(r"^(\d{1,2})[-/.](\d{4})$")
_NAMED = re.compile(r"^([a-z]+)\.?,?\s*(\d{4})$")
_YEAR = re.compile(r"^(\d{4})$")


def parse(value: object) -> tuple[int, int | None] | None:
    """(year, month) for anything normalize() accepts, month None for a
    year-only date. None for an empty value. Raises ValueError otherwise.
    """
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value.year, value.month
    text = str(value).strip().lower()
    if not text:
        return None

    year: int
    month: int | None
    if m := _ISO.match(text):
        year, month = int(m.group(1)), int(m.group(2))
    elif m := _MONTH_FIRST.match(text):
        month, year = int(m.group(1)), int(m.group(2))
    elif m := _NAMED.match(text):
        if m.group(1) not in _MONTH_NAMES:
            raise ValueError(f"unknown month in {value!r}")
        month, year = _MONTH_NAMES[m.group(1)], int(m.group(2))
    elif m := _YEAR.match(text):
        year, month = int(m.group(1)), None
    else:
        raise ValueError(f"could not read {value!r} as a month and year")

    if month is not None and not 1 <= month <= 12:
        raise ValueError(f"no month {month} in {value!r}")
    if not 1900 <= year <= 2100:
        raise ValueError(f"year out of range in {value!r}")
    return year, month


def normalize(value: object) -> str | None:
    """ "Mar 2026" (or "2026") from a date, "2026-03-01", "2026-03",
    "03/2026", "march 2026", "Mar. 2026" and the like. None for an empty
    value. Raises ValueError for anything else.
    """
    parsed = parse(value)
    if parsed is None:
        return None
    year, month = parsed
    return str(year) if month is None else f"{MONTHS[month - 1]} {year}"


def normalize_or_none(value: object) -> str | None:
    """normalize(), but an unreadable value reads as unknown instead of
    raising: for LLM output and stored data, where there is nobody to
    show an error to.
    """
    try:
        return normalize(value)
    except ValueError:
        return None


def sort_key(value: str | None) -> tuple[int, int]:
    """Chronological key; an empty or unreadable date sorts before every
    real one, so a newest-first (reverse) sort puts it last.
    """
    parsed = parse_quiet(value)
    if parsed is None:
        return (-1, -1)
    year, month = parsed
    return (year, month or 0)


def parse_quiet(value: object) -> tuple[int, int | None] | None:
    try:
        return parse(value)
    except ValueError:
        return None


def newest_first(rows: list, attr: str = "start_date") -> list:
    return sorted(rows, key=lambda r: sort_key(getattr(r, attr)), reverse=True)
