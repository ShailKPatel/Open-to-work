from unittest.mock import MagicMock

import pytest

from app.profile.resume_extract import UnsupportedResumeType, extract_resume


def _pdf_with_links(*uris: str) -> bytes:
    import io

    from pypdf import PdfWriter
    from pypdf.annotations import Link

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    for uri in uris:
        writer.add_annotation(0, Link(rect=(10, 10, 50, 20), url=uri))
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_unsupported_type_raises_without_calling_llm(monkeypatch):
    fake_complete = MagicMock()
    monkeypatch.setattr("app.profile.resume_extract.complete", fake_complete)

    with pytest.raises(UnsupportedResumeType):
        extract_resume(b"hello", "text/plain")

    fake_complete.assert_not_called()


def test_extracts_tags_roles_summary_from_pdf(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {
        "tags": ["Python", "FastAPI", ""],
        "target_roles": ["Backend Engineer", "  "],
        "summary": "Strong backend generalist with API experience.",
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake_complete)

    extraction = extract_resume(b"%PDF-1.4 fake", "application/pdf")

    assert extraction.tags == ["Python", "FastAPI"]  # blank entries dropped
    assert extraction.target_roles == ["Backend Engineer"]  # blank entries dropped
    assert extraction.summary == "Strong backend generalist with API experience."
    fake_complete.assert_called_once()
    assert fake_complete.call_args.args[0] == "quality"  # quality tier, not bulk


def test_extracts_from_image(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "A scanned resume."}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake_complete)

    extraction = extract_resume(b"\x89PNG fake", "image/png")

    assert extraction.summary == "A scanned resume."
    fake_complete.assert_called_once()


def test_extracts_experiences_with_dates(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {
        "tags": [],
        "target_roles": [],
        "summary": "x",
        "experiences": [
            {
                "company": "Acme Corp",
                "title": "Software Engineer",
                "location": "  Berlin, Germany ",
                "start_date": "2020-01-01",
                "end_date": "",
                "points": ["Developed API services", "  ", "Implemented unit tests"],
                "skills": ["FastAPI", " ", "pytest"],
            },
            {"company": "  ", "title": "Intern", "start_date": "", "end_date": ""},  # dropped
            {"company": "Beta LLC", "title": "", "start_date": "", "end_date": ""},  # dropped
        ],
    }
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    extraction = extract_resume(b"%PDF-1.4 fake", "application/pdf")

    assert len(extraction.experiences) == 1  # blank company/title entries dropped
    claim = extraction.experiences[0]
    assert claim.company == "Acme Corp"
    assert claim.title == "Software Engineer"
    assert claim.location == "Berlin, Germany"
    assert claim.start_date == "Jan 2020"
    assert claim.end_date is None  # empty string means "still there" / unknown
    assert claim.points == ["Developed API services", "Implemented unit tests"]
    assert claim.skills == ["FastAPI", "pytest"]


def test_experiences_defaults_to_empty_list_when_field_missing(monkeypatch):
    """Older cached LLMCall rows (or a schema hiccup) might not carry an
    `experiences` key at all, response.parsed.get() should degrade to an
    empty list, not KeyError."""
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "x"}
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    extraction = extract_resume(b"%PDF-1.4 fake", "application/pdf")

    assert extraction.experiences == []


def test_extracts_education_with_dates(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {
        "tags": [],
        "target_roles": [],
        "summary": "x",
        "experiences": [],
        "education": [
            {
                "institution": "Nirma University",
                "degree": "B.Tech in Computer Science",
                "location": "Springfield",
                "start_date": "2022-08-01",
                "end_date": "",
            },
            {"institution": "  ", "degree": "HSC", "start_date": "", "end_date": ""},  # dropped
            {"institution": "Some School", "degree": "", "start_date": "", "end_date": ""},
        ],
    }
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    extraction = extract_resume(b"%PDF-1.4 fake", "application/pdf")

    assert len(extraction.education) == 1  # blank institution/degree entries dropped
    claim = extraction.education[0]
    assert claim.institution == "Nirma University"
    assert claim.degree == "B.Tech in Computer Science"
    assert claim.location == "Springfield"
    assert claim.start_date == "Aug 2022"
    assert claim.end_date is None  # empty string means in progress / unknown


def test_education_defaults_to_empty_list_when_field_missing(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "x"}
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    extraction = extract_resume(b"%PDF-1.4 fake", "application/pdf")

    assert extraction.education == []


def test_unparseable_llm_response_raises(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    with pytest.raises(ValueError):
        extract_resume(b"%PDF-1.4 fake", "application/pdf")


def test_extracts_contact_block(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {
        "tags": [],
        "target_roles": [],
        "summary": "x",
        "contact": {
            "name": "Ada Lovelace",
            "location": "London, UK",
            "emails": ["ada@example.com", " ", "ada@work.example"],
            "phones": ["+44 20 7946 0958"],
            "links": [
                {"platform": "LinkedIn", "url": "linkedin.com/in/ada", "label": ""},
                {"platform": "website", "url": "https://ada.dev", "label": "Portfolio"},
                {"platform": "website", "url": "https://blog.ada.dev"},
                {"platform": "leetcode", "url": "https://leetcode.com/ada"},
                {"platform": "github", "url": ""},
            ],
        },
    }
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    contact = extract_resume(b"%PDF-1.4 fake", "application/pdf").contact

    assert contact.name == "Ada Lovelace"
    assert contact.location == "London, UK"
    assert contact.emails == ["ada@example.com", "ada@work.example"]
    assert contact.phones == ["+44 20 7946 0958"]
    assert [(link.platform, link.url, link.label) for link in contact.links] == [
        ("linkedin", "https://linkedin.com/in/ada", None),
        ("other", "https://ada.dev", "Portfolio"),  # a named site is a custom link
        ("website", "https://blog.ada.dev", None),
        ("other", "https://leetcode.com/ada", "Leetcode"),
    ]


def test_contact_defaults_to_empty_when_field_missing(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "x"}
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )

    contact = extract_resume(b"%PDF-1.4 fake", "application/pdf").contact

    assert contact.name is None
    assert contact.emails == [] and contact.phones == [] and contact.links == []


def test_pdf_hyperlinks_are_listed_for_the_llm(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "x"}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake_complete)

    pdf = _pdf_with_links(
        "https://www.credly.com/badges/abc",
        "mailto:ada@example.com",
        "https://www.credly.com/badges/abc",
    )
    extract_resume(pdf, "application/pdf")

    text = fake_complete.call_args.args[1][1]["content"][0]["text"]
    assert "Hyperlinks embedded in the file:\nhttps://www.credly.com/badges/abc" in text
    assert text.count("credly") == 1  # deduplicated
    assert "mailto:" not in text


def test_unreadable_pdf_still_extracts(monkeypatch):
    fake_response = MagicMock()
    fake_response.parsed = {"tags": [], "target_roles": [], "summary": "x"}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake_complete)

    extract_resume(b"%PDF-1.4 fake", "application/pdf")

    text = fake_complete.call_args.args[1][1]["content"][0]["text"]
    assert text == "Here is the resume to analyze."
