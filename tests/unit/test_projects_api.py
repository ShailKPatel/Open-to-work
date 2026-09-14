from pathlib import Path
from unittest.mock import MagicMock

import pytest

import app.core.db as db_module
from app.core.db import Account, Repository, SkillEvidence, get_db, init_db
from app.core.settings import get_settings


@pytest.fixture(autouse=True)
def _stub_indexing(monkeypatch):
    """Manual skill add/delete (app/api/projects.py) now indexes/cleans up
    Qdrant points: real wiring, not what these tests assert on. Stubbed
    the same way tests/unit/test_build.py stubs it, see that file's
    comment for why.
    """
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts]
    )


def _reset_db(tmp_path: Path):
    import os

    import app.retrieval.vectorstore as vectorstore_module

    db_module._engine = None
    db_module._SessionLocal = None
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _make_account(**overrides) -> int:
    db = get_db()
    defaults = dict(first_name="Ada", last_name="Lovelace", github_username="octocat")
    defaults.update(overrides)
    account = Account(**defaults)
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _make_repo(account_id: int | None, **overrides) -> int:
    db = get_db()
    defaults = dict(
        account_id=account_id,
        github_id=overrides.pop("github_id", 1),
        name="proj",
        full_name="octocat/proj",
        url="https://github.com/octocat/proj",
        is_fork=False,
        readme="uses React",
        description=None,
        manifests_json={},
        skill_extraction_status="pending",
    )
    defaults.update(overrides)
    repo = Repository(**defaults)
    db.add(repo)
    db.commit()
    db.refresh(repo)
    repo_id = repo.id
    db.close()
    return repo_id


def test_list_projects_scoped_to_account(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account(github_username="a")
    account_b = _make_account(github_username="b")
    _make_repo(account_a, github_id=1, full_name="a/proj1")
    _make_repo(account_b, github_id=2, full_name="b/proj1")

    client = _client()
    resp = client.get(f"/api/projects?account_id={account_a}")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["full_name"] == "a/proj1"


def test_list_projects_no_readme_flag(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    _make_repo(account, readme=None, description=None)

    client = _client()
    body = client.get(f"/api/projects?account_id={account}").json()

    assert body[0]["has_readme"] is False


def test_list_and_detail_surface_rate_limited_status_and_clean_message(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(
        account,
        skill_extraction_status="rate_limited",
        skill_extraction_error="LLM rate limit or budget cap reached. Try again later.",
    )

    client = _client()
    list_body = client.get(f"/api/projects?account_id={account}").json()
    detail_body = client.get(f"/api/projects/{repo_id}").json()

    assert list_body[0]["skill_extraction_status"] == "rate_limited"
    assert list_body[0]["skill_extraction_error"] == (
        "LLM rate limit or budget cap reached. Try again later."
    )
    assert detail_body["skill_extraction_status"] == "rate_limited"
    assert detail_body["skill_extraction_error"] == (
        "LLM rate limit or budget cap reached. Try again later."
    )


def test_list_projects_skill_count(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)

    db = get_db()
    db.add(
        SkillEvidence(
            skill="React",
            repo_id=repo_id,
            evidence_type="readme_described",
            weight=0.5,
            confidence=0.9,
        )
    )
    db.add(
        SkillEvidence(
            skill="TypeScript",
            repo_id=repo_id,
            evidence_type="declared_dependency",
            weight=0.5,
            confidence=1.0,
        )
    )
    db.commit()
    db.close()

    client = _client()
    body = client.get(f"/api/projects?account_id={account}").json()

    assert body[0]["skill_count"] == 2


def test_project_detail_includes_full_metadata_and_skills(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(
        account,
        stars=7,
        primary_language="Python",
        commits_authored=12,
        manifests_json={"requirements.txt": {"ecosystem": "pip", "dependencies": ["requests"]}},
    )
    db = get_db()
    db.add(
        SkillEvidence(
            skill="React", repo_id=repo_id, evidence_type="readme_described",
            weight=0.8, confidence=0.9,
        )
    )
    db.add(
        SkillEvidence(
            skill="TypeScript", repo_id=repo_id, evidence_type="declared_dependency",
            weight=0.5, confidence=1.0,
        )
    )
    db.commit()
    db.close()

    client = _client()
    resp = client.get(f"/api/projects/{repo_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["full_name"] == "octocat/proj"
    assert body["stars"] == 7
    assert body["primary_language"] == "Python"
    assert body["commits_authored"] == 12
    assert body["manifests"] == {
        "requirements.txt": {"ecosystem": "pip", "dependencies": ["requests"]}
    }
    assert body["has_readme"] is True
    assert body["readme_preview"] == "uses React"
    assert len(body["skills"]) == 2
    # highest weight first
    assert body["skills"][0]["skill"] == "React"
    assert body["skills"][0]["weight"] == 0.8


def test_project_detail_no_readme(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account, readme=None)

    client = _client()
    body = client.get(f"/api/projects/{repo_id}").json()

    assert body["has_readme"] is False
    assert body["readme_preview"] is None


def test_project_detail_unknown_repo_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/api/projects/999999")
    assert resp.status_code == 404


def test_project_detail_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)

    client = _client()
    resp = client.get(f"/portfolio/projects/{repo_id}")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_reprocess_marks_no_signal_when_nothing_to_extract(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account, readme=None, description=None)

    client = _client()
    resp = client.post(f"/api/projects/{repo_id}/reprocess")

    assert resp.status_code == 200
    body = resp.json()
    assert body["skill_extraction_status"] == "no_signal"


def test_reprocess_unknown_repo_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.post("/api/projects/999999/reprocess")
    assert resp.status_code == 404


def test_reprocess_uses_llm_and_clamps_to_repo(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account, readme="Built with Rust and gRPC.")

    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 0.9}]}
    monkeypatch.setattr(
        "app.profile.extract.complete", MagicMock(return_value=fake_response)
    )

    client = _client()
    resp = client.post(f"/api/projects/{repo_id}/reprocess")

    assert resp.status_code == 200
    body = resp.json()
    assert body["skill_extraction_status"] == "extracted"
    assert body["skill_count"] == 1


def test_process_pending_processes_all_pending_repos(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account = _make_account()
    _make_repo(account, github_id=1, full_name="octocat/a", readme=None, description=None)
    _make_repo(account, github_id=2, full_name="octocat/b", readme=None, description=None)

    client = _client()
    resp = client.post(f"/api/projects/process-pending?account_id={account}")

    assert resp.status_code == 200
    assert resp.json()["considered"] == 2

    body = client.get(f"/api/projects?account_id={account}").json()
    statuses = {row["full_name"]: row["skill_extraction_status"] for row in body}
    assert statuses == {"octocat/a": "no_signal", "octocat/b": "no_signal"}


def test_process_pending_stream_runs_in_background_and_reports_progress(tmp_path, monkeypatch):
    # Short-circuit the "keep polling for stragglers" wait so the test
    # doesn't spend several real seconds idling; see app/profile/jobs.py.
    monkeypatch.setattr("app.profile.jobs._EMPTY_PASSES_BEFORE_STOP", 1)
    monkeypatch.setattr("app.profile.jobs._EMPTY_PASS_DELAY_SECONDS", 0.01)

    _reset_db(tmp_path)
    account = _make_account()
    _make_repo(account, github_id=1, full_name="octocat/a", readme=None, description=None)
    _make_repo(account, github_id=2, full_name="octocat/b", readme=None, description=None)

    client = _client()
    with client.stream(
        "GET", f"/api/projects/process-pending/stream?account_id={account}"
    ) as resp:
        body = "".join(resp.iter_text())

    assert '"stage": "done"' in body
    assert '"total": 2' in body

    statuses = {
        row["full_name"]: row["skill_extraction_status"]
        for row in client.get(f"/api/projects?account_id={account}").json()
    }
    assert statuses == {"octocat/a": "no_signal", "octocat/b": "no_signal"}


def test_create_project_manually(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()

    client = _client()
    resp = client.post(
        "/api/projects",
        json={"account_id": account, "name": "Side Project", "description": "hand-typed"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Side Project"
    assert body["full_name"] == "Side Project"  # defaults to name when omitted
    assert body["description"] == "hand-typed"
    assert body["is_manual"] is True
    assert body["skill_extraction_status"] == "pending"

    listed = client.get(f"/api/projects?account_id={account}").json()
    assert listed[0]["is_manual"] is True


def test_create_project_requires_name(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    client = _client()
    resp = client.post("/api/projects", json={"account_id": account, "name": "  "})
    assert resp.status_code == 422


def test_create_project_duplicate_full_name_conflicts(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    client = _client()
    client.post("/api/projects", json={"account_id": account, "name": "A", "full_name": "x/y"})
    resp = client.post(
        "/api/projects", json={"account_id": account, "name": "B", "full_name": "x/y"}
    )
    assert resp.status_code == 409


def test_update_project_edits_fields(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)

    client = _client()
    resp = client.patch(
        f"/api/projects/{repo_id}",
        json={"name": "Renamed", "description": "new desc", "primary_language": "Rust"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Renamed"
    assert body["description"] == "new desc"
    assert body["primary_language"] == "Rust"
    assert body["full_name"] == "octocat/proj"  # untouched field left alone


def test_update_project_rejects_blank_name(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)
    client = _client()
    resp = client.patch(f"/api/projects/{repo_id}", json={"name": "  "})
    assert resp.status_code == 422


def test_update_project_unknown_repo_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.patch("/api/projects/999999", json={"name": "x"})
    assert resp.status_code == 404


def test_skill_crud_add_edit_delete(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)
    client = _client()

    added = client.post(f"/api/projects/{repo_id}/skills", json={"skill": "Go"})
    assert added.status_code == 200
    skills = added.json()["skills"]
    assert len(skills) == 1
    assert skills[0]["skill"] == "Go"
    assert skills[0]["evidence_type"] == "manual"
    skill_id = skills[0]["id"]

    edited = client.patch(
        f"/api/projects/{repo_id}/skills/{skill_id}", json={"skill": "Golang", "weight": 0.5}
    )
    assert edited.status_code == 200
    edited_skill = edited.json()["skills"][0]
    assert edited_skill["skill"] == "Golang"
    assert edited_skill["weight"] == 0.5

    deleted = client.delete(f"/api/projects/{repo_id}/skills/{skill_id}")
    assert deleted.status_code == 200
    assert deleted.json()["skills"] == []


def test_skill_add_rejects_blank(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)
    client = _client()
    resp = client.post(f"/api/projects/{repo_id}/skills", json={"skill": "  "})
    assert resp.status_code == 422


def test_skill_edit_unknown_skill_404(tmp_path):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account)
    client = _client()
    resp = client.patch(f"/api/projects/{repo_id}/skills/999999", json={"skill": "x"})
    assert resp.status_code == 404


def test_manual_skill_survives_reprocess(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account = _make_account()
    repo_id = _make_repo(account, readme="Built with Rust.")

    fake_response = MagicMock()
    fake_response.parsed = {"skills": [{"skill": "Rust", "confidence": 0.9}]}
    monkeypatch.setattr("app.profile.extract.complete", MagicMock(return_value=fake_response))

    client = _client()
    client.post(f"/api/projects/{repo_id}/skills", json={"skill": "Hand-added"})

    resp = client.post(f"/api/projects/{repo_id}/reprocess")
    assert resp.status_code == 200

    detail = client.get(f"/api/projects/{repo_id}").json()
    skill_names = {s["skill"] for s in detail["skills"]}
    assert "Hand-added" in skill_names
    assert "Rust" in skill_names


def test_process_pending_start_is_idempotent_while_running(tmp_path, monkeypatch):
    monkeypatch.setattr("app.profile.jobs._EMPTY_PASSES_BEFORE_STOP", 1)
    monkeypatch.setattr("app.profile.jobs._EMPTY_PASS_DELAY_SECONDS", 0.05)

    _reset_db(tmp_path)
    account = _make_account()
    _make_repo(account, github_id=1, full_name="octocat/a", readme=None, description=None)

    client = _client()
    first = client.post(f"/api/projects/process-pending/start?account_id={account}")
    assert first.json()["started"] is True

    second = client.post(f"/api/projects/process-pending/start?account_id={account}")
    assert second.json()["started"] is False  # already running: no-op, not an error

    # drain it so the background thread doesn't outlive the test
    with client.stream(
        "GET", f"/api/projects/process-pending/stream?account_id={account}"
    ) as resp:
        "".join(resp.iter_text())
