"""Mechanical grounding checks on the text the resume writer composes
from scratch: project bullets and the summary. Experience points have
their own check (orchestrator.py's _is_faithful_reword) because they
start from a real point; these start from nothing but the evidence the
writer was shown, so that evidence is what they are held to.

Two kinds of claim are checked, the two most worth inventing and the two
a string comparison can actually see:

- Numbers. Every number in the text must appear in the evidence.
- Technologies. Any skill name this account has anywhere (repo or role
  evidence) counts as a technology. A project bullet that names one must
  have it in that project's skills, or in that project's own evidence
  text (its description or the account holder's note).

Matching is exact, case-insensitive and on word boundaries, with a small
explicit alias map for spellings that differ in practice. Names of two
characters or fewer, and all-caps acronyms, match case-sensitively, so
the skill "Go" does not fire on "go live" or "REST" on "the rest".
Anything subtler is left to the prompt and the offline groundedness
judge (app/evals/groundedness.py).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Iterable

from app.profile.month_year import parse_quiet

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SEPARATOR_RE = re.compile(r"[\s_-]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Other spellings of the same technology, keyed and valued in _key()
# form. Only pairs seen in practice; nothing here is fuzzy.
_ALIASES = {
    "postgres": "postgresql",
    "sklearn": "scikit learn",
    "k8s": "kubernetes",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
}


def _key(name: str) -> str:
    return _SEPARATOR_RE.sub(" ", name.strip().casefold())


def canonical(name: str) -> str:
    """One key per technology: casefolded, hyphens and spaces equal, and
    an alias mapped onto the name it stands for."""
    key = _key(name)
    return _ALIASES.get(key, key)


def numbers(text: str) -> set[str]:
    """The numbers a text states, thousands separators dropped so "10,000"
    and "10000" are the same claim."""
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text or "")}


def _pattern(spelling: str) -> re.Pattern[str]:
    body = r"[\s_-]+".join(re.escape(part) for part in _SEPARATOR_RE.split(spelling))
    flags = 0 if len(spelling) <= 2 or spelling.isupper() else re.IGNORECASE
    return re.compile(rf"(?<![\w.+#]){body}(?![\w+#])", flags)


class TechVocabulary:
    """Every technology name known for one account, ready to find in text."""

    def __init__(self, names: Iterable[str]):
        spellings: dict[str, str] = {}
        for name in names:
            name = name.strip()
            if name:
                spellings.setdefault(name, canonical(name))
        known = set(spellings.values())
        for alias, target in _ALIASES.items():
            if target in known:
                spellings.setdefault(alias, target)
        # Longest first, so "Docker Compose" is one mention, not also "Docker".
        self._patterns = [
            (_pattern(spelling), key)
            for spelling, key in sorted(spellings.items(), key=lambda s: -len(s[0]))
        ]

    def mentions(self, text: str) -> set[str]:
        """Canonical keys of the technologies `text` names."""
        taken: list[tuple[int, int]] = []
        found: set[str] = set()
        for pattern, key in self._patterns:
            for m in pattern.finditer(text or ""):
                if any(m.start() < end and start < m.end() for start, end in taken):
                    continue
                taken.append(m.span())
                found.add(key)
        return found


def is_grounded_bullet(
    bullet: str, evidence: str, skills: Iterable[str], vocabulary: TechVocabulary
) -> bool:
    """A project bullet states no number its evidence does not, and names
    no technology outside the project's skills and evidence text."""
    if not numbers(bullet) <= numbers(evidence):
        return False
    allowed = {canonical(s) for s in skills} | vocabulary.mentions(evidence)
    return vocabulary.mentions(bullet) <= allowed


def introduces_technology(original: str, rewritten: str, vocabulary: TechVocabulary) -> bool:
    """True when `rewritten` names a technology `original` does not."""
    return not vocabulary.mentions(rewritten) <= vocabulary.mentions(original)


def ground_summary(
    summary: str | None, evidence: Iterable[str], years: float | None = None
) -> tuple[str | None, int]:
    """The summary with every sentence holding an unsupported number
    removed, and how many were removed. A number is supported when the
    evidence states it or it is the account's years of experience,
    rounded either way. None when nothing is left."""
    if not summary or not summary.strip():
        return None, 0
    allowed: set[str] = set()
    for text in evidence:
        allowed |= numbers(text)
    if years is not None:
        allowed |= {str(math.floor(years)), str(math.ceil(years))}
    sentences = [s for s in _SENTENCE_RE.split(summary.strip()) if s]
    kept = [s for s in sentences if numbers(s) <= allowed]
    return (" ".join(kept) or None), len(sentences) - len(kept)


def years_of_experience(
    spans: Iterable[tuple[str | None, str | None]], today: dt.date | None = None
) -> float | None:
    """Years from the earliest role start to the latest role end (today
    for a current role). A year-only start counts from January, a
    year-only end to December. None when no role has a readable start."""
    today = today or dt.date.today()
    starts: list[int] = []
    ends: list[int] = []
    for start, end in spans:
        parsed_start = parse_quiet(start)
        if parsed_start is None:
            continue
        starts.append(parsed_start[0] * 12 + (parsed_start[1] or 1) - 1)
        parsed_end = parse_quiet(end) if end else None
        if parsed_end is None:
            ends.append(today.year * 12 + today.month - 1)
        else:
            ends.append(parsed_end[0] * 12 + (parsed_end[1] or 12) - 1)
    if not starts:
        return None
    return max(0, max(ends) - min(starts) + 1) / 12
