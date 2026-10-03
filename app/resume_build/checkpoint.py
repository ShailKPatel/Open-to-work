"""What an unfinished resume build had done when it stopped, as the
checklist the resume library shows beside its Retry button.

A build is three steps (app/api/resume_build.py's _run_build): one model
call that tailors every section at once, the page-fit pass, then saving.
The sections all come out of that one call together, so they are either
all written or none are; the checklist still names them one by one, so
"what is left" reads as sections of a resume rather than as pipeline
steps. Built from Resume.build_state_json alone, nothing else.
"""

from __future__ import annotations

from typing import Any


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def progress_steps(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Each step as {"label", "done", "note"}, in build order."""
    if not state:
        return []
    data = state.get("data") or None
    written = data is not None
    data = data or {}

    def section(label: str, note: str | None) -> dict[str, Any]:
        return {"label": label, "done": written, "note": note if written else None}

    experience = data.get("experience") or []
    education = data.get("education") or []
    projects = data.get("projects") or []
    skills = data.get("skills") or []
    steps = [
        section("Header and contact details", None),
        section("Experience", _count(len(experience), "role", "roles")),
        section("Education", _count(len(education), "entry", "entries")),
        section("Summary", None if data.get("summary") else "none written"),
        section("Projects", _count(len(projects), "project", "projects")),
        section("Skills", _count(len(skills), "skill", "skills")),
    ]

    fit_bits = []
    if state.get("reworded"):
        fit_bits.append("wording already tightened")
    cuts = int(state.get("cuts_made") or 0)
    if cuts:
        fit_bits.append(_count(cuts, "cut", "cuts") + " already made")
    steps.append({
        "label": "Final review: fitting to the page count",
        "done": False,
        "note": ", ".join(fit_bits) or None,
    })
    steps.append({"label": "Saving the PDF", "done": False, "note": None})
    return steps
