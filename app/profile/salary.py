"""Turns a posting's salary text, as extracted (app/profile/job_extract.py's
salary_range), into whole-number annual bounds so postings can be sorted and
filtered by pay. Deterministic on purpose: the same string always parses the
same way, and rows extracted before these columns existed can be backfilled
(app/core/db/migrations.py) without another LLM call.

Handles the shapes postings actually use: "₹12-18 LPA", "1.5 to 1.6 lakhs
per month", "$120k-$150k", "$60/hr", "18,00,000 - 24,00,000 per annum",
"up to 20 LPA". A bare number with no period is read as annual, since that
is how almost every posting states pay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Multipliers to annual. Hours and days assume a full-time year.
_PERIOD_TO_ANNUAL = {"year": 1, "month": 12, "week": 52, "day": 260, "hour": 2080}

_PERIOD_PATTERNS = [
    ("hour", r"per\s*hour|/\s*h(ou)?r\b|\bhourly\b|\ban\s+hour\b|\bp\.?h\.?\b"),
    ("day", r"per\s*day|/\s*day\b|\bdaily\b|\ba\s+day\b"),
    ("week", r"per\s*week|/\s*(week|wk)\b|\bweekly\b|\ba\s+week\b"),
    (
        "month",
        r"per\s*month|/\s*(month|mo|mon)\b|\bmonthly\b|\ba\s+month\b|\bp\.?\s?m\.?(?=\W|$)"
        r"|\bpm\b|\bstipend\b",
    ),
    (
        "year",
        r"per\s*(annum|year)|/\s*(year|yr|annum)\b|\bannual(ly)?\b|\byearly\b|\ba\s+year\b"
        r"|\bp\.?\s?a\.?(?=\W|$)|\blpa\b|\bctc\b",
    ),
]

_CURRENCY_PATTERNS = [
    (
        "INR",
        r"₹|\d\s*l\b|\d,\d\d,\d{3}|\brs\.?|\binr\b|\blpa\b|\blakhs?\b|\blacs?\b"
        r"|\bcrores?\b|\bcr\b",
    ),
    ("USD", r"\$|\busd\b"),
    ("EUR", r"€|\beur\b"),
    ("GBP", r"£|\bgbp\b"),
]

_UNIT = {
    "k": 1_000,
    "thousand": 1_000,
    "l": 100_000,
    "lpa": 100_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "cr": 10_000_000,
    "crore": 10_000_000,
    "crores": 10_000_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
}

_AMOUNT = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*"
    r"(thousand|lakhs?|lacs?|lpa|crores?|cr|million|mn|k|l|m)?(?![a-z])",
    re.IGNORECASE,
)


@dataclass
class AnnualSalary:
    min: int | None
    max: int | None
    currency: str | None


def _period(text: str) -> str:
    for period, pattern in _PERIOD_PATTERNS:
        if re.search(pattern, text):
            return period
    return "year"


def _currency(text: str) -> str | None:
    for code, pattern in _CURRENCY_PATTERNS:
        if re.search(pattern, text):
            return code
    return None


def parse_salary(text: str | None) -> AnnualSalary:
    """Best-effort annual bounds from free-text salary. Anything that
    doesn't read as a number comes back as all None rather than a guess."""
    empty = AnnualSalary(min=None, max=None, currency=None)
    if not text or not text.strip():
        return empty
    lowered = text.lower().replace("\u2013", "-").replace("\u2014", "-")

    amounts: list[tuple[float, str | None]] = []
    for match in _AMOUNT.finditer(lowered):
        number = float(match.group(1).replace(",", ""))
        unit = (match.group(2) or "").lower() or None
        amounts.append((number, unit))
    amounts = [(n, u) for n, u in amounts if n > 0]
    if not amounts:
        return empty

    # "12-18 LPA", "1.5 to 1.6 lakhs": a bare leading number shares the
    # unit written after the last one.
    trailing_unit = next((u for _, u in reversed(amounts) if u), None)
    values = [n * _UNIT[u or trailing_unit] if (u or trailing_unit) else n for n, u in amounts]

    multiplier = _PERIOD_TO_ANNUAL[_period(lowered)]
    values = [round(v * multiplier) for v in values[:2]]
    currency = _currency(lowered)

    if len(values) == 1:
        only = values[0]
        if re.search(r"up\s*to|upto|\bmax(imum)?\b|\bunder\b|\bbelow\b", lowered):
            return AnnualSalary(min=None, max=only, currency=currency)
        if re.search(r"\bfrom\b|\bmin(imum)?\b|\+|\bstarting\b|\babove\b", lowered):
            return AnnualSalary(min=only, max=None, currency=currency)
        return AnnualSalary(min=only, max=only, currency=currency)

    low, high = sorted(values)
    return AnnualSalary(min=low, max=high, currency=currency)
