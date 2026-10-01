from unittest.mock import MagicMock

import pytest

from app.profile.job_extract import JobExtractionError, extract_job_posting


def test_extracts_structured_fields(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {
        "company": "Acme",
        "title": "Backend Engineer",
        "location": "Remote",
        "salary_range": "$120k-$150k",
        "employment_type": "Full-time",
        "seniority": "Senior",
        "experience_required": "5+ years",
        "skills_required": [
            {"skill": "Python", "level": "senior"},
            {"skill": "Postgres", "level": ""},
            {"skill": "", "level": "mid"},
        ],
        "other_requirements": ["Bachelor's degree", "  "],
        "role_summary": "Own the payments backend.",
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.job_extract.complete", fake_complete)

    extraction = extract_job_posting("We are hiring a backend engineer...")

    assert extraction.company == "Acme"
    assert extraction.salary_range == "$120k-$150k"
    # blank-skill entry dropped, valid ones kept with their level
    assert [(s.skill, s.level) for s in extraction.skills_required] == [
        ("Python", "senior"),
        ("Postgres", ""),
    ]
    assert extraction.other_requirements == ["Bachelor's degree"]  # blank entries dropped
    fake_complete.assert_called_once()
    assert fake_complete.call_args.args[0] == "quality"  # quality tier, not bulk


def test_unparseable_llm_response_raises(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.profile.job_extract.complete", MagicMock(return_value=fake_response)
    )

    with pytest.raises(JobExtractionError):
        extract_job_posting("some text")


def test_as_extracted_json_excludes_company_title_location():
    from app.profile.job_extract import JobExtraction, RequiredSkill

    extraction = JobExtraction(
        company="Acme", title="Engineer", location="Remote",
        salary_range="$100k", employment_type="Full-time", seniority="Mid",
        experience_required="3+ years",
        skills_required=[RequiredSkill(skill="Go", level="mid")],
        other_requirements=[], role_summary="Build things.",
    )

    payload = extraction.as_extracted_json()

    assert "company" not in payload  # handled separately by the caller (backfill-only)
    assert payload["salary_range"] == "$100k"
    assert payload["skills_required"] == [{"skill": "Go", "level": "mid"}]


def test_parse_skills_required_reads_both_legacy_and_current_shapes():
    from app.profile.job_extract import parse_skills_required

    # legacy: bare list[str] (rows extracted before levels existed)
    assert parse_skills_required(["Python", "  ", "Go"]) == [
        {"skill": "Python", "level": ""},
        {"skill": "Go", "level": ""},
    ]
    # current: list[{"skill", "level"}], invalid level normalized to ""
    assert parse_skills_required(
        [{"skill": "Python", "level": "senior"}, {"skill": "Go", "level": "expert!"}]
    ) == [
        {"skill": "Python", "level": "senior"},
        {"skill": "Go", "level": ""},
    ]


def test_parse_skills_required_collapses_duplicates():
    from app.profile.job_extract import parse_skills_required

    assert parse_skills_required(
        [
            {"skill": "Python", "level": ""},
            {"skill": "python", "level": "senior"},
            "PYTHON",
            {"skill": "SQL", "level": "mid"},
        ]
    ) == [
        {"skill": "Python", "level": "senior"},
        {"skill": "SQL", "level": "mid"},
    ]
