"""Page-fit loop: render, compile, check the real page count, and if it's
over the target, ask the LLM (multimodal, given the actual compiled PDF,
not just the text) for the single smallest cut that would help, apply it,
and recompile. Repeats until it fits or nothing safe is left to cut.

Why multimodal and not just "count characters": page overflow is a
layout fact, not a text-length fact, one long word or one extra bullet
can push a whole section onto a second page while a much longer summary
doesn't. One skill or one word can cause an entire new page, and removing
it cuts a whole line. Only the model looking at the actual rendered PDF can tell
which cut, if any, would actually help versus which would trim words for
no layout benefit at all.

One cut per iteration, always the smallest kind available, in this fixed
priority order the LLM is told to follow: a skill first (cheapest, most
flexible), then one project bullet point (dropping the whole project only
if that empties it), then one experience bullet point last and only if a
role would still keep at least one (an experience role itself is never
removable, and neither is the role's company/title/dates, since experience
is compulsory). No LLM call at all happens when the first compile already fits,
this loop costs nothing beyond the orchestrator's own generation call in
the common case.
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from typing import Any

from app.core.llm import complete, system_message, user_message
from app.resume_build.compile import compile_tex
from app.resume_build.latex import render_resume

_DEFAULT_MAX_ITERATIONS = 6

_TRIM_SCHEMA = {
    "type": "object",
    "properties": {
        "cut_type": {
            "type": "string",
            "enum": ["skill", "project_point", "experience_point", "none"],
        },
        "skill": {"type": "string"},
        "project_name": {"type": "string"},
        "point_text": {"type": "string"},
    },
    "required": ["cut_type"],
}

_SYSTEM_PROMPT = (
    "You are shown a compiled resume PDF that is longer than its target "
    "page count. Suggest exactly one small cut that would most likely "
    "bring it back under the limit: prefer dropping one skill from the "
    "skills list, then one bullet point under one project, then, only if "
    "nothing else is available, one bullet point under one experience "
    "role (never suggest removing an entire experience role, its company, "
    "title, or dates, those are compulsory). Only suggest cutting a "
    "project bullet point from a project that has more than one point "
    "left, and an experience bullet point from a role that has more than "
    "one point left. If nothing safe is left to cut, set cut_type to "
    "\"none\". Base your choice on what you actually see causing the "
    "overflow in the PDF (a long line, a near-empty last page, a widow "
    "line), not a guess. Return JSON matching the given schema, nothing "
    "else."
)


class PageFitNotAchievedError(RuntimeError):
    """Ran out of safe cuts (or iterations) before reaching the target
    page count. Carries the best result actually reached, in
    best_tex/best_pdf_bytes/best_page_count, so a caller can still offer
    that rather than nothing.
    """

    def __init__(self, message: str, best_tex: str, best_pdf_bytes: bytes, best_page_count: int):
        super().__init__(message)
        self.best_tex = best_tex
        self.best_pdf_bytes = best_pdf_bytes
        self.best_page_count = best_page_count


@dataclass
class FitResult:
    tex: str
    pdf_bytes: bytes
    page_count: int
    cuts_made: int


def page_count(pdf_bytes: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


def _describe_data(data: dict[str, Any]) -> str:
    lines = ["Skills: " + (", ".join(data.get("skills", [])) or "(none)")]
    for p in data.get("projects", []):
        lines.append(f"Project {p['name']!r} points: " + " | ".join(p.get("points", [])))
    for role in data.get("experience", []):
        lines.append(
            f"Experience {role['company']!r} points: " + " | ".join(role.get("points", []))
        )
    return "\n".join(lines)


def _apply_cut(data: dict[str, Any], suggestion: dict[str, Any]) -> bool:
    """Mutates data in place. Returns True if something was actually
    removed, False if the suggestion didn't match anything current
    (stale reference, hallucinated value) so the caller can treat that
    the same as "none" rather than looping forever on a no-op.
    """
    cut_type = suggestion.get("cut_type")

    if cut_type == "skill":
        skill = str(suggestion.get("skill", "")).strip()
        skills = data.get("skills", [])
        for i, s in enumerate(skills):
            if s.casefold() == skill.casefold():
                del skills[i]
                return True
        return False

    if cut_type == "project_point":
        project_name = str(suggestion.get("project_name", "")).strip()
        point_text = str(suggestion.get("point_text", "")).strip()
        for p in data.get("projects", []):
            if p["name"] != project_name:
                continue
            points = p.get("points", [])
            if point_text in points and len(points) > 1:
                points.remove(point_text)
                return True
        return False

    if cut_type == "experience_point":
        # project_name doubles as the role's company here, keeps the
        # suggestion shape uniform rather than adding a third field name.
        company = str(suggestion.get("project_name", "")).strip()
        point_text = str(suggestion.get("point_text", "")).strip()
        for role in data.get("experience", []):
            if role["company"] != company:
                continue
            points = role.get("points", [])
            if point_text in points and len(points) > 1:
                points.remove(point_text)
                return True
        return False

    return False


def fit_to_page_limit(
    data: dict[str, Any],
    template: str,
    max_pages: int,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    account_id: int | None = None,
) -> FitResult:
    working = copy.deepcopy(data)
    tex = render_resume(f"{template}.tex.j2", working)
    pdf_bytes = compile_tex(tex)
    pages = page_count(pdf_bytes)
    cuts = 0

    while pages > max_pages:
        if cuts >= max_iterations:
            raise PageFitNotAchievedError(
                f"still {pages} pages after {cuts} cuts (limit {max_iterations}), "
                f"target was {max_pages}",
                tex, pdf_bytes, pages,
            )

        response = complete(
            "quality",
            [
                system_message(_SYSTEM_PROMPT),
                user_message(
                    f"Target: {max_pages} page(s). Current: {pages} page(s).\n\n"
                    + _describe_data(working),
                    files=[pdf_bytes],
                ),
            ],
            schema=_TRIM_SCHEMA,
            account_id=account_id,
        )
        suggestion = response.parsed or {"cut_type": "none"}

        if suggestion.get("cut_type") == "none" or not _apply_cut(working, suggestion):
            raise PageFitNotAchievedError(
                f"no more safe cuts available, still {pages} pages, target was {max_pages}",
                tex, pdf_bytes, pages,
            )

        cuts += 1
        tex = render_resume(f"{template}.tex.j2", working)
        pdf_bytes = compile_tex(tex)
        pages = page_count(pdf_bytes)

    return FitResult(tex=tex, pdf_bytes=pdf_bytes, page_count=pages, cuts_made=cuts)
