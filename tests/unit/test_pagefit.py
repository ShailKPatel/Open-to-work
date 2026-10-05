"""app/resume_build/pagefit.py's fit_to_page_limit(). compile_tex and
page_count are both mocked (tectonic is usually not installed on a dev
host); complete() is mocked the same way every other LLM-calling module
in this codebase tests it.

The mock is a small typesetter simulation rather than a canned list of
page numbers, because what is under test now is a search: the loop walks
the density ladder, and a fixed page-count sequence cannot answer "what
would this rung have produced". _units() counts the lines a rendered .tex
would occupy and _install() turns that into a page count against a
per-page capacity that moves with the rendered layout's own density,
read back out of the .tex the loop actually produced. So a tighter rung
really does hold more, an added project really does cost lines, and a cut
really does buy them back.

What that lets the tests pin down: never cut content the layout could
have absorbed, never trust a stale or hallucinated cut suggestion, never
empty out a project or experience role, never pad a resume that has
nothing real left to add, and never let the page-fit loop's own working
state leak into the resume it returns.
"""

import contextlib
import math
import re
from unittest.mock import MagicMock

import pytest

from app.resume_build.latex import render_resume
from app.resume_build.pagefit import (
    _MAX_PLANNED_CUTS,
    PageFitNotAchievedError,
    _apply_addition,
    _apply_cut,
    _apply_rewrite,
    _is_faithful_shortening,
    fit_to_page_limit,
)

_BASE_DATA = {
    "full_name": "Jane Doe",
    "contact_items": [],
    "social_items": [],
    "summary": "A summary.",
    "experience": [
        {
            "id": 1, "title": "Engineer", "company": "Acme", "location": None,
            "date_range": "Jan. 2020 -- present",
            "points": ["Did the important thing", "Did another thing"],
        }
    ],
    "projects": [
        {
            "name": "Cool Project", "tagline": None, "href": "https://example.com",
            "url_display": "example.com", "date_range": "Jun. 2024",
            "points": ["Built the thing", "Made it fast"], "note": None,
        }
    ],
    "education": [],
    "technologies": [],
    "skills": ["Python", "Rust", "Extra Word Skill"],
}

# _BASE_DATA occupies 12 lines by the count below: 4 bullet points, 2
# two-line entry headers, 3 skills, 1 summary.
_BASE_UNITS = 12

_MARGIN_RE = re.compile(r"top=([\d.]+) cm")
_SKILLS_RE = re.compile(
    r"\\section\{Skills\}.*?\\begin\{onecolentry\}\s*(.*?)\s*\\end\{onecolentry\}", re.S
)


def _units(tex: str) -> int:
    lines = len(re.findall(r"\\item ", tex))
    lines += 2 * len(re.findall(r"\\begin\{twocolentry\}", tex))
    skills = _SKILLS_RE.search(tex)
    if skills:
        lines += len([s for s in skills.group(1).split(",") if s.strip()])
    if r"\section{Summary}" in tex:
        lines += 1
    return lines


def _density_of(tex, base_margin_cm):
    return round(float(_MARGIN_RE.search(tex).group(1)) / base_margin_cm, 2)


def _install(
    monkeypatch,
    capacity_at_base,
    cut_suggestions=None,
    base_margin_cm=1.0,
    broken=(),
    rewrite_suggestions=None,
):
    """capacity_at_base is how many lines fit on one page at density 1.0.
    Tighter layouts hold proportionally more, which is the whole point of
    the ladder; the rendered .tex's own top margin is what identifies
    which rung produced it. `broken` names densities whose .tex will not
    compile, standing in for a rung whose document class needs a package
    this machine cannot fetch.

    `cut_suggestions` is the stream of cuts the model would name, in order,
    not one per call: the real call asks for a plan of several at a time
    (see pagefit's _plan_cuts), so the fake below hands out the next
    _MAX_PLANNED_CUTS of them per call and an empty plan once they run out.
    `rewrite_suggestions` is what the rewording call returns, once.
    """

    def _compile(tex):
        from app.resume_build.compile import CompileError

        if _density_of(tex, base_margin_cm) in broken:
            raise CompileError("simulated missing package for this rung")
        return tex.encode()

    monkeypatch.setattr("app.resume_build.pagefit.compile_tex", _compile)

    def _pages(pdf_bytes):
        tex = pdf_bytes.decode()
        density = _density_of(tex, base_margin_cm)
        capacity = max(1, round(capacity_at_base / density))
        return max(1, math.ceil(_units(tex) / capacity))

    monkeypatch.setattr("app.resume_build.pagefit.page_count", _pages)

    if cut_suggestions is None and rewrite_suggestions is None:
        no_llm = MagicMock()
        monkeypatch.setattr("app.resume_build.pagefit.complete", no_llm)
        return no_llm

    remaining_cuts = list(cut_suggestions or [])
    remaining_rewrites = list(rewrite_suggestions or [])

    def _fake_complete(tier, messages, schema=None, account_id=None, purpose=None):
        resp = MagicMock()
        if purpose == "pagefit_rewrite":
            resp.parsed = {"rewrites": list(remaining_rewrites)}
            remaining_rewrites.clear()
            return resp
        resp.parsed = {"cuts": remaining_cuts[:_MAX_PLANNED_CUTS]}
        del remaining_cuts[:_MAX_PLANNED_CUTS]
        return resp

    monkeypatch.setattr("app.resume_build.pagefit.complete", _fake_complete)
    return _fake_complete


def test_base_units_matches_the_rendered_template():
    """Pins the line count the rest of this module's capacity numbers are
    chosen against, so a template change that moves it fails here rather
    than quietly making every other test in the file meaningless."""
    assert _units(render_resume("onepage.tex.j2", _BASE_DATA)) == _BASE_UNITS


def test_opens_the_layout_up_to_fill_the_page(monkeypatch):
    """Content that already fits must not be left sitting on a half empty
    page: the loop keeps loosening until one more rung would overflow."""
    no_llm = _install(monkeypatch, capacity_at_base=_BASE_UNITS * 1.3)

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.fit_exact
    assert result.cuts_made == 0
    assert result.additions_made == 0
    assert result.density > 1.0
    no_llm.assert_not_called()


def test_tightens_the_layout_instead_of_cutting_content(monkeypatch):
    """A resume that overruns by less than the ladder's range is fixed by
    typography alone. Nothing the account holder wrote is lost and no LLM
    call is made."""
    no_llm = _install(monkeypatch, capacity_at_base=_BASE_UNITS - 2)

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.fit_exact
    assert result.cuts_made == 0
    assert result.density < 1.0
    assert "Extra Word Skill" in result.tex
    assert "Made it fast" in result.tex
    no_llm.assert_not_called()


def test_cuts_a_skill_once_the_layout_has_nothing_left_to_give(monkeypatch):
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": "Extra Word Skill"}],
    )

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.cuts_made == 1
    assert "Extra Word Skill" not in result.tex
    assert "Python" in result.tex  # untouched skills survive
    assert "Extra Word Skill" not in result.data["skills"]


def test_cuts_a_project_point_without_dropping_the_project(monkeypatch):
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[
            {
                "cut_type": "project_point",
                "project_name": "Cool Project",
                "point_text": "Made it fast",
            }
        ],
    )

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert "Made it fast" not in result.tex
    assert "Built the thing" in result.tex
    assert "Cool Project" in result.tex


def test_refuses_to_empty_out_a_project(monkeypatch):
    """Cutting the last remaining point of a project must be refused, not
    silently applied, since that would drop the project's own required
    minimum content invisibly."""
    single_point_data = {
        **_BASE_DATA,
        "projects": [{**_BASE_DATA["projects"][0], "points": ["Only point"]}],
    }
    _install(
        monkeypatch,
        capacity_at_base=7,
        cut_suggestions=[
            {
                "cut_type": "project_point",
                "project_name": "Cool Project",
                "point_text": "Only point",
            }
        ],
    )

    with pytest.raises(PageFitNotAchievedError) as exc_info:
        fit_to_page_limit(single_point_data, "onepage", max_pages=1)

    assert exc_info.value.best_page_count == 2


def test_stale_skill_suggestion_does_not_loop_forever(monkeypatch):
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": "Does Not Exist Anywhere"}],
    )

    with pytest.raises(PageFitNotAchievedError):
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)


def test_none_suggestion_raises_with_best_result_attached(monkeypatch):
    _install(monkeypatch, capacity_at_base=8, cut_suggestions=[{"cut_type": "none"}])

    with pytest.raises(PageFitNotAchievedError) as exc_info:
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert exc_info.value.best_page_count == 2
    assert exc_info.value.best_tex
    assert exc_info.value.best_data is not None


def test_losing_every_key_mid_fit_keeps_the_resume_already_built(monkeypatch):
    """By the time this loop runs, the expensive part is done: the
    content is written and compiled. If the keys run out before the
    trimming pass can ask which entry to drop, the build must hand back
    the resume it has, with the reason, rather than throw away a paid-for
    build over a page count. app/core/llm.py has already tried every key
    by the time this exception arrives."""
    from app.core.llm import LLMRateLimitedError

    no_llm = _install(monkeypatch, capacity_at_base=8)
    no_llm.side_effect = LLMRateLimitedError("All 2 OpenAI keys failed on this request.")

    with pytest.raises(PageFitNotAchievedError) as exc_info:
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert exc_info.value.best_pdf_bytes
    assert exc_info.value.best_page_count == 2
    assert exc_info.value.best_data is not None
    assert "All 2 OpenAI keys failed" in str(exc_info.value)
    assert "trimming step could not run" in str(exc_info.value)


def test_never_suggests_removing_an_experience_role(monkeypatch):
    """The experience role/company/title/dates are never part of any cut
    payload shape this module applies, only its points list, and only
    down to a minimum of one remaining."""
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[
            {
                "cut_type": "experience_point",
                "project_name": "Acme",
                "point_text": "Did another thing",
            }
        ],
    )

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert "Acme" in result.tex
    assert "Did another thing" not in result.tex
    assert "Did the important thing" in result.tex


def test_max_iterations_stops_it(monkeypatch):
    # 10 distinct skills so 10 successive skill-cuts are all individually
    # valid; the loop must still stop at max_iterations rather than
    # continuing forever if the page count never actually drops.
    data = {**_BASE_DATA, "skills": [f"Skill{i}" for i in range(10)]}
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": f"Skill{i}"} for i in range(10)],
    )

    with pytest.raises(PageFitNotAchievedError, match="limit 2"):
        fit_to_page_limit(data, "onepage", max_pages=1, max_iterations=2)


def test_adds_reserve_content_to_reach_a_two_page_target(monkeypatch):
    """The other half of "exactly N pages": content that stops short of
    the target gets the account's own held-back material added back
    until it reaches it."""
    data = {
        **_BASE_DATA,
        "reserve": {
            "projects": [
                {
                    "name": "Held Back Project", "tagline": None, "href": None,
                    "url_display": "Held Back Project", "date_range": "Mar. 2024",
                    "points": ["A real repository description"], "note": None,
                }
            ],
            "experience_points": {},
            "skills": [],
        },
    }
    no_llm = _install(monkeypatch, capacity_at_base=18, base_margin_cm=1.5)

    result = fit_to_page_limit(data, "twopage", max_pages=2)

    assert result.page_count == 2
    assert result.fit_exact
    assert result.additions_made == 1
    assert result.cuts_made == 0
    assert "Held Back Project" in result.tex
    no_llm.assert_not_called()


def test_adds_the_largest_reserve_item_first(monkeypatch):
    """Projects before experience points before skills: the count has to
    converge, and a single extra skill is worth almost no height."""
    reserve = {
        "projects": [
            {
                "name": "Held Back Project", "tagline": None, "href": None,
                "url_display": "Held Back Project", "date_range": "Mar. 2024",
                "points": ["A real repository description"], "note": None,
            }
        ],
        "experience_points": {"1": ["A narrowed-away point"]},
        "skills": ["Held Back Skill"],
    }
    _install(monkeypatch, capacity_at_base=18, base_margin_cm=1.5)

    result = fit_to_page_limit({**_BASE_DATA, "reserve": reserve}, "twopage", max_pages=2)

    assert "Held Back Project" in result.tex
    assert "A narrowed-away point" not in result.tex
    assert "Held Back Skill" not in result.tex


def test_short_content_with_an_empty_reserve_is_not_padded(monkeypatch):
    """An account without two pages of real material gets an honest one
    page back, flagged, rather than a padded second page or an error."""
    no_llm = _install(monkeypatch, capacity_at_base=18, base_margin_cm=1.5)

    result = fit_to_page_limit(_BASE_DATA, "twopage", max_pages=2)

    assert result.page_count == 1
    assert result.fit_exact is False
    assert result.additions_made == 0
    assert result.target_pages == 2
    no_llm.assert_not_called()


def test_reserve_never_reaches_the_rendered_tex_or_the_returned_data(monkeypatch):
    reserve = {"projects": [], "experience_points": {}, "skills": ["Held Back Skill"]}
    _install(monkeypatch, capacity_at_base=_BASE_UNITS * 1.3)

    result = fit_to_page_limit({**_BASE_DATA, "reserve": reserve}, "onepage", max_pages=1)

    assert "reserve" not in result.data
    assert "Held Back Skill" not in result.tex


def test_caller_data_is_never_mutated(monkeypatch):
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": "Extra Word Skill"}],
    )

    fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert _BASE_DATA["skills"] == ["Python", "Rust", "Extra Word Skill"]


def test_a_rung_that_will_not_compile_is_skipped_not_fatal(monkeypatch):
    """The tightest rungs switch document class for a sub-10pt body,
    which needs a package Tectonic has to fetch. Losing one should cost
    the resume a notch of range, not the whole PDF."""
    _install(monkeypatch, capacity_at_base=10, broken=(0.95, 0.9))

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.density == 0.84
    assert result.cuts_made == 0


def test_a_broken_rung_on_the_way_up_is_skipped_too(monkeypatch):
    no_llm = _install(monkeypatch, capacity_at_base=_BASE_UNITS * 1.3, broken=(1.06, 1.12))

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.density == 1.34
    no_llm.assert_not_called()


def test_an_addition_that_overshoots_the_whole_ladder_is_rolled_back(monkeypatch):
    """The addition that finally reaches the target is the only one that
    can overshoot past what even the tightest layout holds. It gets
    undone rather than dragging the resume into the cut loop."""
    huge = {
        "name": "Oversized Project", "tagline": None, "href": None,
        "url_display": "Oversized Project", "date_range": "Mar. 2024",
        "points": [f"Point number {i}" for i in range(45)], "note": None,
    }
    data = {
        **_BASE_DATA,
        "reserve": {"projects": [huge], "experience_points": {}, "skills": []},
    }
    no_llm = _install(monkeypatch, capacity_at_base=18, base_margin_cm=1.5)

    result = fit_to_page_limit(data, "twopage", max_pages=2)

    assert "Oversized Project" not in result.tex
    assert result.page_count == 1
    assert result.fit_exact is False
    assert result.additions_made == 0
    no_llm.assert_not_called()


def test_reserve_items_already_on_the_resume_are_skipped(monkeypatch):
    """A reserve built before an edit can name content the resume now
    shows. Adding it twice would be worse than not filling the page."""
    reserve = {
        "projects": [{**_BASE_DATA["projects"][0]}],
        "experience_points": {"1": ["Did the important thing", "A narrowed-away point"]},
        "skills": ["Python"],
    }
    _install(monkeypatch, capacity_at_base=18, base_margin_cm=1.5)

    result = fit_to_page_limit({**_BASE_DATA, "reserve": reserve}, "twopage", max_pages=2)

    assert result.tex.count("Cool Project") == 1
    assert result.tex.count("Did the important thing") == 1
    assert "A narrowed-away point" in result.tex
    assert result.additions_made == 1


def test_compile_budget_caps_the_search(monkeypatch):
    """A document whose page count refuses to move with density must not
    turn into an unbounded run of Tectonic calls behind a request."""
    _install(monkeypatch, capacity_at_base=_BASE_UNITS * 1.3)

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1, max_compiles=1)

    assert result.page_count == 1
    assert result.density == 1.0


def test_one_plan_covers_several_cuts(monkeypatch):
    """The saving this loop is built around: three cuts, one look at the
    PDF, because the model is asked for an ordered plan rather than for the
    next single cut."""
    data = {**_BASE_DATA, "skills": [f"Skill{i}" for i in range(6)]}
    fake_complete = _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": f"Skill{i}"} for i in range(4)],
    )
    calls: list = []

    def _recording_complete(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_complete(*args, **kwargs)

    monkeypatch.setattr("app.resume_build.pagefit.complete", _recording_complete)

    # Generous compile budget: this test is about how many LLM calls the
    # cuts take, not about the Tectonic ceiling the ladder walks into.
    result = fit_to_page_limit(data, "onepage", max_pages=1, max_compiles=80)

    assert result.cuts_made > 1
    trims = [c for c in calls if c[1]["purpose"] == "pagefit_trim"]
    assert len(trims) == 1
    args, kwargs = trims[0]
    assert kwargs["purpose"] == "pagefit_trim"
    # Bulk tier: ranking items it was handed, not writing anything.
    assert args[0] == "bulk"


def test_a_stale_entry_in_the_plan_is_skipped_not_fatal(monkeypatch):
    """The plan is written against the data as it stood, so a later entry
    can name something an earlier cut already took. That costs the next
    entry, not the build."""
    _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[
            {"cut_type": "skill", "skill": "Already Gone"},
            {"cut_type": "skill", "skill": "Extra Word Skill"},
        ],
    )

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert "Extra Word Skill" not in result.data["skills"]


def test_an_empty_plan_raises_rather_than_looping(monkeypatch):
    _install(monkeypatch, capacity_at_base=8, cut_suggestions=[])

    with pytest.raises(PageFitNotAchievedError):
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)


# --- rewording before cutting ---------------------------------------

_LONG_POINT = "Designed and built the internal billing service handling 40 requests per second"


@pytest.mark.parametrize(
    "shorter, ok",
    [
        ("Built the internal billing service handling 40 requests per second", True),
        ("Building the billing service, 40 requests per second", True),
        # Not shorter.
        (_LONG_POINT + " reliably", False),
        # A number the original never had.
        ("Built the internal billing service handling 400 requests per second", False),
        # A word the original never used: a new claim.
        ("Built the Kubernetes billing service handling 40 requests per second", False),
        # Gutted: a cut wearing a rewording's clothes.
        ("Built billing", False),
        ("Built the billing service \u2014 40 requests per second", False),
        ("", False),
    ],
)
def test_faithful_shortening_check(shorter, ok):
    assert _is_faithful_shortening(_LONG_POINT, shorter) is ok


def test_rewrite_needs_the_original_quoted_exactly():
    data = {
        "summary": "A summary.",
        "experience": [{"company": "Acme", "points": [_LONG_POINT]}],
        "projects": [],
    }
    shorter = "Built the internal billing service handling 40 requests per second"
    misquoted = {
        "target": "experience_point", "owner": "Acme",
        "original": _LONG_POINT.lower(), "shorter": shorter,
    }
    wrong_owner = {**misquoted, "original": _LONG_POINT, "owner": "Initech"}
    assert not _apply_rewrite(data, misquoted)
    assert not _apply_rewrite(data, wrong_owner)

    assert _apply_rewrite(data, {**wrong_owner, "owner": "Acme"})
    assert data["experience"][0]["points"] == [shorter]


def _two_roles_at_acme() -> dict:
    """A promotion: two roles at the same company, each its own row."""
    return {
        "summary": "A summary.",
        "experience": [
            {"id": 1, "company": "Acme", "points": ["Led the team", "Hired four people"]},
            {"id": 2, "company": "Acme", "points": [_LONG_POINT, "Wrote the docs"]},
        ],
        "projects": [],
        "skills": [],
    }


def test_held_back_point_goes_back_to_its_own_role_at_the_same_company():
    data = _two_roles_at_acme()
    reserve = {"projects": [], "experience_points": {"2": ["Fixed the pager"]}, "skills": []}

    assert _apply_addition(data, reserve)

    assert data["experience"][0]["points"] == ["Led the team", "Hired four people"]
    assert data["experience"][1]["points"] == [_LONG_POINT, "Wrote the docs", "Fixed the pager"]


def test_reserve_saved_with_company_keys_still_adds():
    """A build checkpoint saved before the reserve was keyed by role id."""
    data = _two_roles_at_acme()
    reserve = {"projects": [], "experience_points": {"Acme": ["Fixed the pager"]}, "skills": []}

    assert _apply_addition(data, reserve)

    assert data["experience"][0]["points"][-1] == "Fixed the pager"


def test_cut_and_rewrite_reach_the_second_role_at_the_same_company():
    data = _two_roles_at_acme()
    shorter = "Built the internal billing service handling 40 requests per second"

    assert _apply_cut(
        data,
        {"cut_type": "experience_point", "project_name": "Acme", "point_text": "Wrote the docs"},
    )
    assert _apply_rewrite(
        data,
        {
            "target": "experience_point", "owner": "Acme",
            "original": _LONG_POINT, "shorter": shorter,
        },
    )

    assert data["experience"][0]["points"] == ["Led the team", "Hired four people"]
    assert data["experience"][1]["points"] == [shorter]


def test_rewording_runs_once_before_any_cut(monkeypatch):
    """Rewording loses nothing, so it is tried first; it is a model call
    with the PDF attached, so it is tried once. The checked wording lands
    in the resume, and the cut plan is still asked for when the
    rewording alone did not get the page count down."""
    data = {
        **_BASE_DATA,
        "experience": [
            {**_BASE_DATA["experience"][0], "points": [_LONG_POINT, "Did another thing"]}
        ],
    }
    shorter = "Built the internal billing service handling 40 requests per second"
    fake_complete = _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": "Extra Word Skill"}],
        rewrite_suggestions=[
            {"target": "experience_point", "owner": "Acme", "original": _LONG_POINT,
             "shorter": shorter},
            {"target": "summary", "owner": "", "original": "A summary.",
             "shorter": "A brand new claim."},
        ],
    )
    purposes: list = []

    def _recording(*args, **kwargs):
        purposes.append(kwargs["purpose"])
        return fake_complete(*args, **kwargs)

    monkeypatch.setattr("app.resume_build.pagefit.complete", _recording)

    result = fit_to_page_limit(data, "onepage", max_pages=1, max_compiles=80)

    assert purposes[0] == "pagefit_rewrite"
    assert purposes.count("pagefit_rewrite") == 1
    assert "pagefit_trim" in purposes
    assert result.rewrites_made == 1
    assert shorter in result.tex
    assert "A summary." in result.tex


def test_progress_reports_each_model_change_and_a_resumed_fit_skips_rewording(monkeypatch):
    """Every change a model call made is reported with the state needed to
    carry on from it, and a fit started again from that state does not
    ask for the rewording a second time."""
    data = {
        **_BASE_DATA,
        "experience": [
            {**_BASE_DATA["experience"][0], "points": [_LONG_POINT, "Did another thing"]}
        ],
    }
    shorter = "Built the internal billing service handling 40 requests per second"
    fake_complete = _install(
        monkeypatch,
        capacity_at_base=8,
        cut_suggestions=[{"cut_type": "skill", "skill": "Extra Word Skill"}],
        rewrite_suggestions=[
            {"target": "experience_point", "owner": "Acme", "original": _LONG_POINT,
             "shorter": shorter},
        ],
    )
    reports: list = []
    fit_to_page_limit(data, "onepage", max_pages=1, max_compiles=80, on_progress=reports.append)

    assert reports[0]["reworded"] is True
    assert reports[0]["cuts_made"] == 0
    assert shorter in str(reports[0]["data"]["experience"])
    assert "reserve" in reports[0]["data"]

    purposes: list = []

    def _recording(*args, **kwargs):
        purposes.append(kwargs["purpose"])
        return fake_complete(*args, **kwargs)

    monkeypatch.setattr("app.resume_build.pagefit.complete", _recording)
    # The fake's one cut went to the first fit, so this one may run out.
    with contextlib.suppress(PageFitNotAchievedError):
        fit_to_page_limit(
            reports[0]["data"], "onepage", max_pages=1, max_compiles=80,
            reworded=True, cuts_made=reports[0]["cuts_made"],
        )

    assert "pagefit_rewrite" not in purposes


def test_rewrite_that_brings_in_a_new_technology_is_dropped():
    """_is_faithful_shortening ignores words of three letters or fewer, so
    a short tool name like "Go" would slip past it on its own."""
    from app.resume_build.grounding import TechVocabulary

    data = {
        "summary": "",
        "experience": [],
        "projects": [{"name": "api", "points": ["Built the internal billing service in Python"]}],
        "skills": [],
    }
    rewrite = {
        "target": "project_point", "owner": "api",
        "original": "Built the internal billing service in Python",
        "shorter": "Built the billing service in Go",
    }
    vocabulary = TechVocabulary(["Python", "Go"])

    assert not _apply_rewrite(data, rewrite, vocabulary)
    in_python = {**rewrite, "shorter": "Built the billing service in Python"}
    assert _apply_rewrite(data, in_python, vocabulary)
