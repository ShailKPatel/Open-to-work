"""app/resume_build/pagefit.py's fit_to_page_limit(). compile_tex and
page_count are both mocked (tectonic is usually not installed on a dev
host); complete()
is mocked the same way every other LLM-calling module in this codebase
tests it. What's under test is the cut-priority/grounding logic itself:
never trust a stale or hallucinated cut suggestion, never empty out a
project or experience role entirely, stop cleanly when nothing safe is
left.
"""

from unittest.mock import MagicMock

import pytest

from app.resume_build.pagefit import PageFitNotAchievedError, fit_to_page_limit

_BASE_DATA = {
    "full_name": "Jane Doe",
    "contact_items": [],
    "social_items": [],
    "summary": "A summary.",
    "experience": [
        {
            "title": "Engineer", "company": "Acme", "location": None,
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


def _mock_pipeline(monkeypatch, page_counts, cut_suggestions=None):
    monkeypatch.setattr("app.resume_build.pagefit.compile_tex", lambda tex: b"%PDF-fake")
    counts = iter(page_counts)
    monkeypatch.setattr("app.resume_build.pagefit.page_count", lambda pdf: next(counts))

    if cut_suggestions is not None:
        responses = iter(cut_suggestions)

        def _fake_complete(tier, messages, schema=None, account_id=None):
            resp = MagicMock()
            resp.parsed = next(responses)
            return resp

        monkeypatch.setattr("app.resume_build.pagefit.complete", _fake_complete)


def test_already_fits_makes_no_llm_call(monkeypatch):
    _mock_pipeline(monkeypatch, page_counts=[1])
    fake_complete = MagicMock()
    monkeypatch.setattr("app.resume_build.pagefit.complete", fake_complete)

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.cuts_made == 0
    fake_complete.assert_not_called()


def test_cuts_a_skill_and_reaches_target(monkeypatch):
    _mock_pipeline(
        monkeypatch,
        page_counts=[2, 1],
        cut_suggestions=[{"cut_type": "skill", "skill": "Extra Word Skill"}],
    )

    result = fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert result.page_count == 1
    assert result.cuts_made == 1
    assert "Extra Word Skill" not in result.tex
    assert "Python" in result.tex  # untouched skills survive


def test_cuts_a_project_point_without_dropping_the_project(monkeypatch):
    _mock_pipeline(
        monkeypatch,
        page_counts=[2, 1],
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
    _mock_pipeline(
        monkeypatch,
        page_counts=[2],
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
    _mock_pipeline(
        monkeypatch,
        page_counts=[2],
        cut_suggestions=[{"cut_type": "skill", "skill": "Does Not Exist Anywhere"}],
    )

    with pytest.raises(PageFitNotAchievedError):
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)


def test_none_suggestion_raises_with_best_result_attached(monkeypatch):
    _mock_pipeline(monkeypatch, page_counts=[2], cut_suggestions=[{"cut_type": "none"}])

    with pytest.raises(PageFitNotAchievedError) as exc_info:
        fit_to_page_limit(_BASE_DATA, "onepage", max_pages=1)

    assert exc_info.value.best_pdf_bytes == b"%PDF-fake"
    assert exc_info.value.best_page_count == 2


def test_never_suggests_removing_an_experience_role(monkeypatch):
    """The experience role/company/title/dates are never part of any cut
    payload shape this module applies, only its points list, and only
    down to a minimum of one remaining."""
    _mock_pipeline(
        monkeypatch,
        page_counts=[2, 1],
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
    # 6 distinct skills so 6 successive skill-cuts are all individually
    # valid; the loop must still stop at max_iterations rather than
    # continuing forever if the page count never actually drops.
    data = {**_BASE_DATA, "skills": [f"Skill{i}" for i in range(10)]}
    _mock_pipeline(
        monkeypatch,
        page_counts=[2] * 10,
        cut_suggestions=[{"cut_type": "skill", "skill": f"Skill{i}"} for i in range(10)],
    )

    with pytest.raises(PageFitNotAchievedError, match="limit 2"):
        fit_to_page_limit(data, "onepage", max_pages=1, max_iterations=2)
