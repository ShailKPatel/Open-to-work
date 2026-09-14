from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.core.db as db_module
from app.core.db import Repository, init_db
from app.core.settings import get_settings
from app.profile.extract import NoSourceTextError, extract_skills_from_repo


def _reset_db(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _repo(readme: str | None = None, description: str | None = None) -> Repository:
    return Repository(
        github_id=1,
        name="proj",
        full_name="octocat/proj",
        url="https://github.com/octocat/proj",
        readme=readme,
        description=description,
        manifests_json={},
    )


def test_no_readme_no_description_raises_without_calling_llm(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_complete = MagicMock()
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    with pytest.raises(NoSourceTextError):
        extract_skills_from_repo(_repo())

    fake_complete.assert_not_called()


def test_extracts_claims_from_readme(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {
        "skills": [
            {"skill": "Kubernetes operators", "confidence": 0.8},
            {"skill": "gRPC", "confidence": 0.6},
        ]
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    claims = extract_skills_from_repo(
        _repo(readme="Built with gRPC and k8s operators.", description="a project")
    )

    assert len(claims) == 2
    assert all(c.evidence_type == "readme_described" for c in claims)
    assert {c.skill for c in claims} == {"Kubernetes operators", "gRPC"}
    # bulk tier: this runs across hundreds of repos
    assert fake_complete.call_args.args[0] == "bulk"


def test_falls_back_to_description_when_no_readme(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 0.7}]}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    claims = extract_skills_from_repo(_repo(readme=None, description="A Rust CLI tool."))

    assert len(claims) == 1
    assert claims[0].evidence_type == "description_described"
    assert claims[0].source_files == ["description"]
    sent_messages = fake_complete.call_args.args[1]
    assert "A Rust CLI tool." in sent_messages[1]["content"]


def test_blank_readme_falls_back_to_description(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    extract_skills_from_repo(_repo(readme="   ", description="fallback text"))

    sent_messages = fake_complete.call_args.args[1]
    assert "fallback text" in sent_messages[1]["content"]


def test_confidence_is_clamped_to_unit_interval(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 5.0}]}
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    claims = extract_skills_from_repo(_repo(readme="uses rust"))
    assert claims[0].confidence == 1.0


def test_null_parsed_response_returns_no_claims(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    assert extract_skills_from_repo(_repo(readme="some readme")) == []


def test_readme_truncated_before_sending_to_llm(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    huge_readme = "x" * 50_000
    extract_skills_from_repo(_repo(readme=huge_readme))

    sent_messages = fake_complete.call_args.args[1]
    user_content = sent_messages[1]["content"]
    assert len(user_content) < 10_000
