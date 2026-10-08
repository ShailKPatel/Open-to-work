"""Prompt injection detection for pasted job posting text.

A posting is untrusted input: it goes to the model as a quarantined
user-role document (app/profile/job_extract.py), never into a system
prompt, which is the actual defence. This module is the visibility layer
on top: it reports text that looks written to steer a model, so the
account holder can see it. A detection is recorded and logged, never
acted on; the posting is still saved and read as usual.

Rules are deliberately specific. Hiring language is full of words a naive
filter would trip on ("ignore the degree requirement", "write
instructions for operators", "design system prompts"), so each rule
needs a directive aimed at a model, not just a suspicious word. Detection
and false positive rates are measured on evals/synthetic/redteam.yaml and
on every ordinary posting in the eval sets (app/evals/injection.py).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_BIDI = re.compile("[\u202a-\u202e\u2066-\u2069]")

# Each (kind, pattern). Patterns are matched case-insensitively.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_directive",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all|any|the)\b[^.\n]{0,20}"
            r"\b(instructions?|prompts?|rules|directions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_spoofing",
        re.compile(
            r"(\[\s*(system|assistant|developer)\s*\]|^\s*(system|assistant|developer)\s*:"
            r"|<!--\s*(system|assistant|developer)\s*:)",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "new_instruction",
        re.compile(r"\bnew (instruction|task|rule)s?\s*:", re.IGNORECASE),
    ),
    (
        "addressed_to_model",
        re.compile(
            r"\b(note|message|instruction)s? (to|for) (any |the )?"
            r"(ai|llm|model|assistant|language model|resume tool|ai resume tool)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_markup",
        re.compile(
            r"(<!--.*?-->|style\s*=\s*[\"'][^\"']*(display\s*:\s*none|visibility\s*:\s*hidden"
            r"|font-size\s*:\s*[01](px|pt)?\b|color\s*:\s*(white|#fff\b|#ffffff\b)))",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "output_injection",
        re.compile(
            r"[\"']\s*\}\s*,\s*[\"'](company|title|skills_required|salary_range|role_summary)"
            r"[\"']\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        "encoded_payload",
        re.compile(
            r"\b(decode|base64|execute|follow)\b[^\n]{0,30}\b[A-Za-z0-9+/]{24,}={0,2}",
            re.IGNORECASE,
        ),
    ),
    (
        "foreign_override",
        re.compile(
            r"\b(ignora|ignorez|ignoriere|ignorare)\b[^.\n]{0,30}"
            r"\b(instrucciones|instructions|anweisungen|istruzioni)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration",
        re.compile(
            r"\b(append|send|include|post)\b[^.\n]{0,60}\b(api[ _-]?key|token|password|secret)s?\b",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True)
class Detection:
    kind: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _excerpt(text: str, start: int, end: int) -> str:
    """The matched span with a little context, invisible characters made
    visible, so a reader can see what was flagged."""
    snippet = text[max(0, start - 20) : min(len(text), end + 20)]
    snippet = _ZERO_WIDTH.sub("[zero-width]", snippet)
    snippet = _BIDI.sub("[bidi]", snippet)
    return " ".join(snippet.split())[:160]


def detect_injection(text: str) -> list[Detection]:
    """Every rule that fires on `text`, at most one detection per kind,
    in a fixed order. An empty list means nothing looked like an attempt
    to instruct a model."""
    found: list[Detection] = []
    zero_width = _ZERO_WIDTH.search(text)
    if zero_width:
        found.append(
            Detection("zero_width_characters", _excerpt(text, zero_width.start(), zero_width.end()))
        )
    bidi = _BIDI.search(text)
    if bidi:
        found.append(Detection("bidi_control_characters", _excerpt(text, bidi.start(), bidi.end())))
    # Rules run on the text with invisible characters removed, so a
    # directive split by zero-width spaces still reads as one.
    visible = _BIDI.sub("", _ZERO_WIDTH.sub(" ", text))
    for kind, pattern in _RULES:
        match = pattern.search(visible)
        if match:
            found.append(Detection(kind, _excerpt(visible, match.start(), match.end())))
    return found
