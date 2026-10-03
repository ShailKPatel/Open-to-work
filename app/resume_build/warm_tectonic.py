"""Fills Tectonic's package cache at image build time, so a resume build
never has to download LaTeX packages or fonts in the middle of a page fit.

Tectonic fetches each package, font and map file the first time a
document asks for it. Over a slow link that first compile can run well
past compile.py's timeout, and the page-fit loop is where it bites: the
tightest density rungs switch to extarticle at 9pt, which pulls in files
no earlier compile needed. Compiling every template once per font size,
with every section and every contact icon present, fetches all of it up
front.

Run as `python -m app.resume_build.warm_tectonic`. It only imports the
LaTeX side of app/resume_build/, so the Dockerfile can run it before the
rest of app/ is copied in.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from app.resume_build.compile import compile_tex
from app.resume_build.latex import render_resume
from app.resume_build.layout import DENSITY_LADDER, layout_for

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_CONTEXT_SOURCE = Path(__file__).parent / "context.py"

# Generous: a cold cache over a slow link is exactly the case this exists for.
_WARM_TIMEOUT_SECONDS = 900


def _icons() -> list[str]:
    """Every icon macro context.py can put in a header. Read from its source
    rather than imported, since context.py pulls in the database layer."""
    return sorted(set(re.findall(r"\\fa[A-Za-z]+\*?", _CONTEXT_SOURCE.read_text())))


def _sample_data() -> dict[str, Any]:
    icons = _icons()
    point = "Built a service handling 50% more load with C# and Python."
    return {
        "full_name": "Sample Name",
        "contact_items": [
            {"icon": icon, "text": "sample", "href": "https://example.com"} for icon in icons
        ],
        "social_items": [{"icon": icons[0], "text": "sample", "href": None}],
        "summary": point,
        "experience": [
            {
                "title": "Engineer",
                "company": "Company",
                "location": "City",
                "date_range": "Jan 2024 -- Present",
                "points": [point, point],
            }
        ],
        "projects": [
            {
                "name": "Project",
                "tagline": "Tagline",
                "date_range": "2024",
                "href": "https://example.com",
                "url_display": "example.com",
                "points": [point],
                "note": point,
            }
        ],
        "education": [
            {
                "institution": "University",
                "degree": "B.Tech",
                "grade": "9.0",
                "date_range": "2020 -- 2024",
                "details": [point],
            }
        ],
        "technologies": ["Python", "Docker"],
        "skills": ["Testing", "Design"],
    }


def _one_density_per_font_size(template: str) -> list[float]:
    seen: dict[int, float] = {}
    for density in DENSITY_LADDER:
        seen.setdefault(layout_for(template, density)["font_pt"], density)
    return list(seen.values())


def main() -> int:
    data = _sample_data()
    for template_file in sorted(_TEMPLATES_DIR.glob("*.tex.j2")):
        template = template_file.name.split(".", 1)[0]
        for density in _one_density_per_font_size(template):
            tex = render_resume(
                template_file.name, {**data, "layout": layout_for(template, density)}
            )
            compile_tex(tex, timeout=_WARM_TIMEOUT_SECONDS)
            print(f"warmed {template} at density {density}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
