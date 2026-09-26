"""Layout knobs the .tex templates read, plus the density ladder
app/resume_build/pagefit.py walks to land a resume on exactly the page
count the user asked for.

Both templates used to hardcode their geometry (margins, section
spacing, bullet spacing, the header name's point size). Hardcoding them
means the only lever a page-fit loop has is deleting real content, which
is the most expensive possible way to lose a fifth of a page, and it
gives no lever at all in the other direction: a two-page resume whose
content only reaches page one has no way to grow into the space it was
asked to fill. Making them parameters turns "fit the page" into a search
over typography first and content second.

A profile is the full knob set at density 1.0, which reproduces byte for
byte what each template rendered before this module existed. layout_for()
scales that profile: below 1.0 tightens (less whitespace, smaller type,
more content per page), above 1.0 opens it up. Whitespace moves fastest,
type size moves in the discrete steps LaTeX actually offers, and every
knob is clamped so no rung of the ladder can produce a resume that reads
as broken rather than merely tight or merely airy.

DENSITY_LADDER is the fixed set of rungs pagefit compiles against. It is
a list, not a continuous range, because every probe costs a real
Tectonic run: thirteen rungs spanning roughly a 35 percent swing in
content per page is enough resolution to hit a target exactly, and few
enough that a binary search over it costs four compiles.
"""

from __future__ import annotations

from typing import Any

# Each profile is that template's original hardcoded geometry. Changing a
# number here changes what the template renders at density 1.0, which is
# what every caller that does not run the page-fit loop gets.
_BASE_PROFILES: dict[str, dict[str, float]] = {
    "onepage": {
        "margin_top_cm": 1.0,
        "margin_bottom_cm": 1.0,
        "margin_side_cm": 2.0,
        "footskip_cm": 1.0,
        "name_pt": 24.0,
        "header_gap_cm": 0.3,
        "section_before_cm": 0.3,
        "section_after_cm": 0.2,
        "bullet_topsep_cm": 0.10,
        "bullet_parsep_cm": 0.10,
        "entry_gap_cm": 0.10,
        "role_gap_cm": 0.10,
        "project_gap_cm": 0.2,
        "education_gap_cm": 0.10,
    },
    "twopage": {
        "margin_top_cm": 1.5,
        "margin_bottom_cm": 1.5,
        "margin_side_cm": 2.0,
        "footskip_cm": 0.8,
        "name_pt": 24.0,
        "header_gap_cm": 0.3,
        "section_before_cm": 0.4,
        "section_after_cm": 0.3,
        "bullet_topsep_cm": 0.15,
        "bullet_parsep_cm": 0.15,
        "entry_gap_cm": 0.15,
        "role_gap_cm": 0.2,
        "project_gap_cm": 0.3,
        "education_gap_cm": 0.10,
    },
}

DEFAULT_TEMPLATE = "onepage"

# Tight to loose. Index 0 is the most content a page can be asked to
# carry before the page-fit loop has to start cutting real content;
# the last rung is the most it can be stretched before it has to start
# adding content back.
DENSITY_LADDER: tuple[float, ...] = (
    0.72, 0.78, 0.84, 0.90, 0.95, 1.00, 1.06, 1.12, 1.19, 1.26, 1.34, 1.42, 1.50,
)

BASE_DENSITY_INDEX = DENSITY_LADDER.index(1.00)

# Vertical whitespace responds to density one for one; horizontal margins
# and the header's type size deliberately do not, since a resume whose
# side margins swing as hard as its section gaps stops looking like the
# same document from one rung to the next.
_SIDE_MARGIN_RESPONSE = 0.5
_NAME_SIZE_RESPONSE = 0.35
_LINE_SPREAD_RESPONSE = 0.45

_MIN_GAP_CM = 0.02
_MARGIN_V_RANGE = (0.7, 2.6)
_MARGIN_SIDE_RANGE = (1.2, 2.8)
_FOOTSKIP_RANGE = (0.5, 1.4)
_NAME_PT_RANGE = (17.0, 30.0)
_LINE_SPREAD_RANGE = (0.93, 1.18)

# article only ships 10, 11 and 12pt. Anything tighter needs extarticle
# (the extsizes package), which Tectonic fetches on demand like every
# other package the preamble asks for.
_EXT_SIZES = {8, 9, 14, 17, 20}


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _font_pt_for(density: float) -> int:
    if density < 0.86:
        return 9
    if density < 1.14:
        return 10
    if density < 1.34:
        return 11
    return 12


def template_key(template: str) -> str:
    """Accepts either the bare template name used by the API and the DB
    ("onepage") or the filename render_resume() takes
    ("onepage.tex.j2"), so callers on both sides of latex.py can ask for
    a layout without first normalising the string themselves.
    """
    key = template.split(".", 1)[0]
    return key if key in _BASE_PROFILES else DEFAULT_TEMPLATE


def layout_for(template: str, density: float = 1.0) -> dict[str, Any]:
    """The knob set a template renders with. density 1.0 reproduces that
    template's original hardcoded geometry exactly; every other value
    scales it. Returns rounded numbers because these land directly in
    LaTeX length arguments, where an unrounded float is just noise in the
    generated source.
    """
    base = _BASE_PROFILES[template_key(template)]
    font_pt = _font_pt_for(density)

    def gap(key: str) -> float:
        return round(max(_MIN_GAP_CM, base[key] * density), 3)

    return {
        "doc_class": "extarticle" if font_pt in _EXT_SIZES else "article",
        "font_pt": font_pt,
        "line_spread": round(
            _clamp(1.0 + (density - 1.0) * _LINE_SPREAD_RESPONSE, _LINE_SPREAD_RANGE), 3
        ),
        "margin_top_cm": round(_clamp(base["margin_top_cm"] * density, _MARGIN_V_RANGE), 2),
        "margin_bottom_cm": round(_clamp(base["margin_bottom_cm"] * density, _MARGIN_V_RANGE), 2),
        "margin_side_cm": round(
            _clamp(
                base["margin_side_cm"] * (1.0 + (density - 1.0) * _SIDE_MARGIN_RESPONSE),
                _MARGIN_SIDE_RANGE,
            ),
            2,
        ),
        "footskip_cm": round(_clamp(base["footskip_cm"] * density, _FOOTSKIP_RANGE), 2),
        "name_pt": round(
            _clamp(
                base["name_pt"] * (1.0 + (density - 1.0) * _NAME_SIZE_RESPONSE), _NAME_PT_RANGE
            ),
            1,
        ),
        "header_gap_cm": gap("header_gap_cm"),
        "section_before_cm": gap("section_before_cm"),
        "section_after_cm": gap("section_after_cm"),
        "bullet_topsep_cm": gap("bullet_topsep_cm"),
        "bullet_parsep_cm": gap("bullet_parsep_cm"),
        "entry_gap_cm": gap("entry_gap_cm"),
        "role_gap_cm": gap("role_gap_cm"),
        "project_gap_cm": gap("project_gap_cm"),
        "education_gap_cm": gap("education_gap_cm"),
    }


def default_layout(template: str) -> dict[str, Any]:
    """What a caller that never runs the page-fit loop renders with: the
    template's own original geometry, unscaled.
    """
    return layout_for(template, 1.0)
