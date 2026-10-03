from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.core.db as db_module
from app.core.db import Repository, init_db
from app.core.llm import LLMRateLimitedError
from app.core.settings import get_settings
from app.profile.extract import (
    _BATCH_SIZE,
    NoSourceTextError,
    extract_repo_facts,
    prefetch_repo_facts,
)


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
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
        extract_repo_facts(_repo())

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

    claims = extract_repo_facts(
        _repo(readme="Built with gRPC and k8s operators.", description="a project")
    ).skills

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

    claims = extract_repo_facts(_repo(readme=None, description="A Rust CLI tool.")).skills

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

    extract_repo_facts(_repo(readme="   ", description="fallback text"))

    sent_messages = fake_complete.call_args.args[1]
    assert "fallback text" in sent_messages[1]["content"]


def test_confidence_is_clamped_to_unit_interval(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 5.0}]}
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    claims = extract_repo_facts(_repo(readme="uses rust")).skills
    assert claims[0].confidence == 1.0


def test_null_parsed_response_returns_no_claims(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = None
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    assert extract_repo_facts(_repo(readme="some readme")).skills == []


def test_readme_truncated_before_sending_to_llm(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    huge_readme = "x" * 50_000
    extract_repo_facts(_repo(readme=huge_readme))

    sent_messages = fake_complete.call_args.args[1]
    user_content = sent_messages[1]["content"]
    assert len(user_content) < 10_000


def test_skills_and_links_come_back_from_one_call(tmp_path, monkeypatch):
    """The point of the merged call: both halves of the answer for the cost
    of reading the README once, not twice."""
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {
        "skills": [{"skill": "Rust", "confidence": 0.7}],
        "links": [{"label": "Live Demo", "url": "https://example.com"}],
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    facts = extract_repo_facts(_repo(readme="A Rust CLI tool, demo at example.com"))

    assert [c.skill for c in facts.skills] == ["Rust"]
    assert [(link.label, link.url) for link in facts.links] == [
        ("Live Demo", "https://example.com")
    ]
    assert fake_complete.call_count == 1
    assert fake_complete.call_args.kwargs["purpose"] == "repo_facts"


def test_badges_code_blocks_and_license_are_not_sent(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [], "links": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    readme = (
        "# Proj\n\n"
        "[![build](https://img.shields.io/badge/build-passing.svg)](https://ci.example.com)\n"
        "![logo](logo.png)\n\n"
        "<p align=\"center\">A service that speaks gRPC.</p>\n\n"
        "```bash\npip install proj\nexport TOKEN=secret\n```\n\n"
        "## License\n\nMIT, see LICENSE for the full text.\n"
    )
    extract_repo_facts(_repo(readme=readme))

    sent = fake_complete.call_args.args[1][1]["content"]
    assert "A service that speaks gRPC." in sent
    assert "img.shields.io" not in sent
    assert "pip install proj" not in sent
    assert "MIT" not in sent
    assert "<p align" not in sent


def test_readme_that_is_only_badges_falls_back_to_the_description(tmp_path, monkeypatch):
    """Cleaning can empty a README, which then means the same thing as not
    having one."""
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [], "links": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    facts_repo = _repo(
        readme="![b](https://img.shields.io/x.svg)\n\n## License\n\nMIT\n",
        description="A Rust CLI tool.",
    )
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 0.5}], "links": []}
    facts = extract_repo_facts(facts_repo)

    assert facts.skills[0].evidence_type == "description_described"
    assert "A Rust CLI tool." in fake_complete.call_args.args[1][1]["content"]


def _batch_repos(count: int) -> list[Repository]:
    repos = []
    for i in range(count):
        repo = _repo(readme=f"Project {i} is built with Rust and gRPC.")
        repo.id = i + 1
        repo.name = f"proj{i}"
        repo.full_name = f"octocat/proj{i}"
        repos.append(repo)
    return repos


def test_prefetch_covers_several_repos_in_one_call(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {
        "repos": [
            {
                "repo_id": 1,
                "skills": [{"skill": "Rust", "confidence": 0.8}],
                "links": [{"label": "Docs", "url": "https://docs.example.com"}],
            },
            {"repo_id": 2, "skills": [{"skill": "gRPC", "confidence": 0.6}], "links": []},
            {"repo_id": 3, "skills": [], "links": []},
        ]
    }
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    facts = prefetch_repo_facts(_batch_repos(3))

    assert fake_complete.call_count == 1
    assert fake_complete.call_args.kwargs["purpose"] == "repo_facts_batch"
    assert sorted(facts) == [1, 2, 3]
    assert [c.skill for c in facts[1].skills] == ["Rust"]
    assert facts[1].links[0].label == "Docs"
    assert facts[3].skills == []
    sent = fake_complete.call_args.args[1][1]["content"]
    assert "Repository id=1" in sent and "Repository id=3" in sent


def test_prefetch_splits_into_groups(tmp_path, monkeypatch):
    """_BATCH_SIZE repos per call, so a long list is several calls rather
    than one prompt long enough to blur one repo into another."""
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"repos": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    prefetch_repo_facts(_batch_repos(_BATCH_SIZE * 2 + 1))

    assert fake_complete.call_count == 3


def test_batched_answer_for_a_repo_we_did_not_send_is_dropped(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {
        "repos": [
            {"repo_id": 1, "skills": [{"skill": "Rust", "confidence": 0.8}], "links": []},
            {"repo_id": 999, "skills": [{"skill": "Invented", "confidence": 0.9}], "links": []},
        ]
    }
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    facts = prefetch_repo_facts(_batch_repos(2))

    assert sorted(facts) == [1]


def test_prefetch_stops_on_a_rate_limit_and_returns_what_it_got(tmp_path, monkeypatch):
    """The per-repo path that follows is what records a rate limit against a
    specific repo and stops the batch, so this one only has to stop asking."""
    _reset_db(tmp_path)
    first_response = MagicMock()
    first_response.parsed = {
        "repos": [{"repo_id": 1, "skills": [{"skill": "Rust", "confidence": 0.8}], "links": []}]
    }
    calls: list = []

    def _complete(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return first_response
        raise LLMRateLimitedError("slow down")

    monkeypatch.setattr("app.profile.extract.complete", _complete)

    facts = prefetch_repo_facts(_batch_repos(_BATCH_SIZE * 2))

    assert len(calls) == 2  # the second group's failure stops the third
    assert sorted(facts) == [1]


def test_prefetch_also_stops_when_the_keys_run_out(tmp_path, monkeypatch):
    """Every stored key spent arrives as a provider error, not a 429, so
    the stop is decided by app/core/llm.py's is_out_of_keys(). Without
    that check each remaining group would spend a doomed call."""
    from app.core.llm import LLMProviderError

    _reset_db(tmp_path)
    calls: list = []

    def _complete(*args, **kwargs):
        calls.append(kwargs)
        error = LLMProviderError("All 2 OpenAI keys failed on this request.")
        error.blames_key = True
        raise error

    monkeypatch.setattr("app.profile.extract.complete", _complete)

    assert prefetch_repo_facts(_batch_repos(_BATCH_SIZE * 3)) == {}
    assert len(calls) == 1  # the first group's failure stops the rest


def test_a_group_of_one_uses_the_single_repo_prompt(tmp_path, monkeypatch):
    """Nothing to amortize over one repo, and the single-repo prompt is the
    one whose response the cache can reuse later."""
    _reset_db(tmp_path)
    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 0.8}], "links": []}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.profile.extract.complete", fake_complete)

    repos = _batch_repos(2)
    repos[1].readme = None
    repos[1].description = None

    facts = prefetch_repo_facts(repos)

    assert sorted(facts) == [1]
    assert fake_complete.call_args.kwargs["purpose"] == "repo_facts"
