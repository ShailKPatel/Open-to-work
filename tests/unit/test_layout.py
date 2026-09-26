"""app/resume_build/layout.py: the geometry the .tex templates read and
the density ladder app/resume_build/pagefit.py searches.

The load-bearing tests here are test_density_one_matches_the_original_
geometry (density 1.0 has to reproduce what each template hardcoded
before it took parameters, or every resume ever generated silently
changes shape) and test_ladder_is_monotonic (the page-fit search walks
the ladder assuming tighter never means fewer lines per page, and a
knob that moves the wrong way would make that search pick nonsense).
"""

import pytest

from app.resume_build.layout import (
    BASE_DENSITY_INDEX,
    DENSITY_LADDER,
    default_layout,
    layout_for,
    template_key,
)

_ORIGINAL_GEOMETRY = {
    "onepage": {
        "doc_class": "article", "font_pt": 10, "line_spread": 1.0,
        "margin_top_cm": 1.0, "margin_bottom_cm": 1.0, "margin_side_cm": 2.0,
        "footskip_cm": 1.0, "name_pt": 24.0, "header_gap_cm": 0.3,
        "section_before_cm": 0.3, "section_after_cm": 0.2,
        "bullet_topsep_cm": 0.1, "bullet_parsep_cm": 0.1, "entry_gap_cm": 0.1,
        "role_gap_cm": 0.1, "project_gap_cm": 0.2, "education_gap_cm": 0.1,
    },
    "twopage": {
        "doc_class": "article", "font_pt": 10, "line_spread": 1.0,
        "margin_top_cm": 1.5, "margin_bottom_cm": 1.5, "margin_side_cm": 2.0,
        "footskip_cm": 0.8, "name_pt": 24.0, "header_gap_cm": 0.3,
        "section_before_cm": 0.4, "section_after_cm": 0.3,
        "bullet_topsep_cm": 0.15, "bullet_parsep_cm": 0.15, "entry_gap_cm": 0.15,
        "role_gap_cm": 0.2, "project_gap_cm": 0.3, "education_gap_cm": 0.1,
    },
}

# Everything a template interpolates. A knob added to layout_for() but
# never read by a template, or read by a template but never produced
# here, is a rendering failure, so the two sides are pinned together.
_REQUIRED_KNOBS = set(_ORIGINAL_GEOMETRY["onepage"])


@pytest.mark.parametrize("template", ["onepage", "twopage"])
def test_density_one_matches_the_original_geometry(template):
    assert default_layout(template) == _ORIGINAL_GEOMETRY[template]


@pytest.mark.parametrize("template", ["onepage", "twopage"])
def test_every_density_produces_every_knob(template):
    for density in DENSITY_LADDER:
        assert set(layout_for(template, density)) == _REQUIRED_KNOBS


def test_base_index_points_at_density_one():
    assert DENSITY_LADDER[BASE_DENSITY_INDEX] == 1.0


@pytest.mark.parametrize("template", ["onepage", "twopage"])
def test_ladder_is_monotonic(template):
    """Every knob that costs vertical space has to grow with density and
    none may shrink, so that one rung looser can never mean more content
    per page."""
    growing = [
        "font_pt", "line_spread", "margin_top_cm", "margin_bottom_cm",
        "margin_side_cm", "name_pt", "header_gap_cm", "section_before_cm",
        "section_after_cm", "bullet_topsep_cm", "bullet_parsep_cm",
        "entry_gap_cm", "role_gap_cm", "project_gap_cm", "education_gap_cm",
    ]
    rungs = [layout_for(template, d) for d in DENSITY_LADDER]
    for knob in growing:
        values = [r[knob] for r in rungs]
        assert values == sorted(values), f"{knob} moves the wrong way: {values}"
    assert rungs[0]["font_pt"] < rungs[-1]["font_pt"]


@pytest.mark.parametrize("template", ["onepage", "twopage"])
def test_extreme_densities_stay_within_readable_bounds(template):
    """Clamps, not the raw multiplication, decide the edges: no rung may
    produce a resume that reads as broken rather than merely tight or
    merely airy, even well outside the ladder."""
    for density in (0.2, 0.5, 3.0, 10.0):
        layout = layout_for(template, density)
        assert 0.7 <= layout["margin_top_cm"] <= 2.6
        assert 0.7 <= layout["margin_bottom_cm"] <= 2.6
        assert 1.2 <= layout["margin_side_cm"] <= 2.8
        assert 17.0 <= layout["name_pt"] <= 30.0
        assert 0.93 <= layout["line_spread"] <= 1.18
        assert layout["bullet_topsep_cm"] > 0
        assert layout["project_gap_cm"] > 0


def test_sub_ten_point_sizes_switch_document_class():
    """article only ships 10, 11 and 12pt; anything smaller needs
    extarticle, or the document will not compile at all."""
    tightest = layout_for("onepage", DENSITY_LADDER[0])
    assert tightest["font_pt"] == 9
    assert tightest["doc_class"] == "extarticle"
    assert layout_for("onepage", 1.0)["doc_class"] == "article"


def test_template_key_accepts_a_name_or_a_filename():
    assert template_key("onepage") == "onepage"
    assert template_key("twopage.tex.j2") == "twopage"
    assert template_key("something-else") == "onepage"
