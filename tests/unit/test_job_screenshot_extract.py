"""app/profile/job_screenshot_extract.py: multimodal screenshot -> job
posting. Same injected-complete() convention as test_resume_extract.py /
test_job_extract.py."""

from unittest.mock import MagicMock

import pytest

from app.profile.job_screenshot_extract import (
    ScreenshotExtractionError,
    UnsupportedScreenshotType,
    extract_job_posting_from_image,
)


def _fake_response(**overrides):
    fake = MagicMock()
    fake.parsed = {
        "raw_text_transcribed": "Backend Engineer at Acme. 5+ years Python.",
        "company": "Acme", "title": "Backend Engineer", "location": "Remote",
        "salary_range": "$120k-$150k", "employment_type": "Full-time",
        "seniority": "Senior", "experience_required": "5+ years",
        "skills_required": [{"skill": "Python", "level": "senior"}],
        "other_requirements": [], "role_summary": "Own the backend.",
        **overrides,
    }
    return fake


def test_unsupported_mime_type_raises_without_calling_llm(monkeypatch):
    fake_complete = MagicMock()
    monkeypatch.setattr("app.profile.job_screenshot_extract.complete", fake_complete)

    with pytest.raises(UnsupportedScreenshotType):
        extract_job_posting_from_image(b"not-an-image", "application/pdf")

    fake_complete.assert_not_called()


def test_extracts_transcription_and_structured_fields(monkeypatch):
    monkeypatch.setattr(
        "app.profile.job_screenshot_extract.complete",
        MagicMock(return_value=_fake_response()),
    )

    result = extract_job_posting_from_image(b"fake-png-bytes", "image/png")

    assert result.raw_text_transcribed == "Backend Engineer at Acme. 5+ years Python."
    assert result.extraction.company == "Acme"
    assert [s.skill for s in result.extraction.skills_required] == ["Python"]


def test_empty_transcription_raises_extraction_error(monkeypatch):
    monkeypatch.setattr(
        "app.profile.job_screenshot_extract.complete",
        MagicMock(return_value=_fake_response(raw_text_transcribed="")),
    )

    with pytest.raises(ScreenshotExtractionError):
        extract_job_posting_from_image(b"fake-png-bytes", "image/png")


def test_unparseable_llm_response_raises(monkeypatch):
    fake = MagicMock()
    fake.parsed = None
    monkeypatch.setattr("app.profile.job_screenshot_extract.complete", MagicMock(return_value=fake))

    with pytest.raises(ScreenshotExtractionError):
        extract_job_posting_from_image(b"fake-png-bytes", "image/png")
