"""app/evals/groundedness.py: real DB (Repository/SkillEvidence/Resume),
injected complete() for the judge call, same convention as every other
LLM-touching module's tests in this codebase.
"""

from pathlib import Path
from unittest.mock import MagicMock

import app.core.db as db_module
from app.core.db import Account, Repository, Resume, SkillEvidence, get_db, init_db
from app.core.settings import get_settings
from app.evals.groundedness import score_groundedness


def _reset(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _make_account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _make_repo(account_id: int, description="A Python web scraper") -> int:
    db = get_db()
    repo = Repository(
        account_id=account_id, github_id=1, name="scraper", full_name="octocat/scraper",
        url="https://github.com/octocat/scraper", is_fork=False, description=description,
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    repo_id = repo.id
    db.add(
        SkillEvidence(
            skill="Python", repo_id=repo_id, evidence_type="declared_dependency",
            weight=0.7, confidence=1.0,
        )
    )
    db.commit()
    db.close()
    return repo_id


def _make_generated_resume(account_id: int, repo_id: int, points: list[str]):
    db = get_db()
    resume = Resume(
        account_id=account_id, filename="generated.pdf", mime_type="application/pdf",
        content_json={
            "projects": [{"repo_id": repo_id, "name": "scraper", "points": points}]
        },
    )
    db.add(resume)
    db.commit()
    db.close()


def test_no_generated_resumes_returns_none_with_reason(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()

    result = score_groundedness(account_id)

    assert result.score is None
    assert result.checked == 0
    assert "no generated resumes" in result.skipped_reason


def test_grounded_and_ungrounded_bullets_scored(tmp_path, monkeypatch):
    _reset(tmp_path)
    account_id = _make_account()
    repo_id = _make_repo(account_id)
    _make_generated_resume(
        account_id, repo_id,
        points=["Built a Python web scraper", "Deployed a fleet of autonomous drones"],
    )

    def _fake_complete(tier, messages, schema=None, account_id=None):
        content = messages[-1]["content"]
        response = MagicMock()
        response.parsed = {"grounded": "drone" not in content.lower()}
        return response

    monkeypatch.setattr("app.evals.groundedness.complete", _fake_complete)

    result = score_groundedness(account_id)

    assert result.checked == 2
    assert result.score == 0.5  # one grounded, one not


def test_missing_api_key_stops_early_but_keeps_partial_score(tmp_path, monkeypatch):
    _reset(tmp_path)
    account_id = _make_account()
    repo_id = _make_repo(account_id)
    _make_generated_resume(
        account_id, repo_id, points=["Built a Python web scraper", "Second bullet"]
    )

    from app.core.llm import ApiKeyMissingError

    call_count = {"n": 0}

    def _fake_complete(tier, messages, schema=None, account_id=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            response = MagicMock()
            response.parsed = {"grounded": True}
            return response
        raise ApiKeyMissingError("no key configured")

    monkeypatch.setattr("app.evals.groundedness.complete", _fake_complete)

    result = score_groundedness(account_id)

    assert result.checked == 1
    assert result.score == 1.0
    assert "stopped early" in result.skipped_reason


def test_max_checks_bounds_total_llm_calls(tmp_path, monkeypatch):
    _reset(tmp_path)
    account_id = _make_account()
    repo_id = _make_repo(account_id)
    _make_generated_resume(account_id, repo_id, points=[f"Bullet {i}" for i in range(10)])

    fake_response = MagicMock()
    fake_response.parsed = {"grounded": True}
    fake_complete = MagicMock(return_value=fake_response)
    monkeypatch.setattr("app.evals.groundedness.complete", fake_complete)

    result = score_groundedness(account_id, max_checks=3)

    assert result.checked == 3
    assert fake_complete.call_count == 3


def test_project_with_no_repo_match_is_skipped(tmp_path, monkeypatch):
    _reset(tmp_path)
    account_id = _make_account()
    _make_generated_resume(account_id, repo_id=999999, points=["Some bullet"])

    fake_complete = MagicMock()
    monkeypatch.setattr("app.evals.groundedness.complete", fake_complete)

    result = score_groundedness(account_id)

    assert result.checked == 0
    fake_complete.assert_not_called()
