"""Background resume builds: app/resume_build/background.py and the
/api/resume-build/start and /builds routes in app/api/resume_build.py.
The build itself (build_resume_data, fit_to_page_limit) is mocked at the
router boundary, as in test_resume_build_api.py; what's under test is
that the build runs outside the request, reports how it went, and scores
the finished resume against the posting.
"""

import os
import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.llm import LLMRateLimitedError
from app.resume_build import background
from app.resume_build.pagefit import FitResult


@pytest.fixture(autouse=True)
def _fresh_builds(monkeypatch):
    monkeypatch.setattr(background, "_builds", {})


@pytest.fixture
def posting(tmp_path, monkeypatch):
    import app.core.db as db_module
    import app.retrieval.vectorstore as vectorstore_module
    from app.core.db import Account, JobPosting, get_db, init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    monkeypatch.setitem(os.environ, "DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setitem(os.environ, "RESUME_STORAGE_DIR", str(tmp_path / "resumes"))
    monkeypatch.setitem(os.environ, "QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    init_db()
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    row = JobPosting(
        account_id=account.id, source="pasted", external_id="h", company="Acme",
        title="Backend Engineer", raw_text_quarantined="hiring text", content_hash="h",
        extracted_json={"skills_required": [
            {"skill": "Python", "level": "required"},
            {"skill": "Docker", "level": "required"},
        ]},
    )
    db.add(row)
    db.commit()
    ids = {"account_id": account.id, "job_posting_id": row.id}
    db.close()
    yield ids
    get_settings.cache_clear()


def _client() -> TestClient:
    return TestClient(app)


def _wait_done(client: TestClient, build_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/resume-build/builds/{build_id}").json()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError("build never finished")


def _fake_build(monkeypatch, skills=("Python",)):
    data = {"summary": "s", "skills": list(skills), "projects": []}
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value=data))
    fit = FitResult(tex="x", pdf_bytes=b"%PDF-bg", page_count=1, cuts_made=0)
    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", MagicMock(return_value=fit))


def test_start_answers_before_the_build_finishes(posting, monkeypatch):
    release = threading.Event()
    _fake_build(monkeypatch)

    def _slow(*args, **kwargs):
        release.wait(5)
        return {"summary": "s", "skills": ["Python"], "projects": []}

    monkeypatch.setattr("app.api.resume_build.build_resume_data", _slow)
    client = _client()

    resp = client.post("/api/resume-build/start", json=posting)

    assert resp.status_code == 200
    started = resp.json()
    assert started["running"] is True
    assert started["label"] == "Backend Engineer - Acme"
    release.set()
    assert _wait_done(client, started["id"])["state"] == "done"


def test_finished_build_reports_result_match_and_pdf(posting, monkeypatch):
    _fake_build(monkeypatch, skills=["Python"])
    client = _client()

    build_id = client.post("/api/resume-build/start", json=posting).json()["id"]
    state = _wait_done(client, build_id)

    assert state["state"] == "done"
    assert state["page_count"] == 1
    assert state["page_fit_achieved"] is True
    assert state["resume_id"] is not None
    # One of two required skills listed: half.
    assert state["match_pct"] == 50
    assert state["matched"] == ["Python"]
    assert state["missing"] == ["Docker"]
    pdf = client.get(f"/api/resume-build/builds/{build_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.content == b"%PDF-bg"


def test_failed_build_keeps_the_reason(posting, monkeypatch):
    def _raise(*args, **kwargs):
        raise LLMRateLimitedError("All 2 OpenAI keys failed on this request.")

    monkeypatch.setattr("app.api.resume_build.build_resume_data", _raise)
    client = _client()

    build_id = client.post("/api/resume-build/start", json=posting).json()["id"]
    state = _wait_done(client, build_id)

    assert state["state"] == "error"
    assert "tailoring the content to the job" in state["detail"]
    assert "All 2 OpenAI keys failed" in state["detail"]
    assert client.get(f"/api/resume-build/builds/{build_id}/pdf").status_code == 404


def test_start_refuses_unknown_posting_and_template(posting):
    client = _client()
    unknown = client.post(
        "/api/resume-build/start",
        json={"account_id": posting["account_id"], "job_posting_id": 999999},
    )
    bad_template = client.post(
        "/api/resume-build/start", json={**posting, "template": "threepage"}
    )

    assert unknown.status_code == 404
    assert bad_template.status_code == 422
    assert client.get(
        f"/api/resume-build/builds?account_id={posting['account_id']}"
    ).json() == []


def test_list_filters_by_posting_and_drops_dismissed(posting, monkeypatch):
    _fake_build(monkeypatch)
    client = _client()

    build_id = client.post("/api/resume-build/start", json=posting).json()["id"]
    _wait_done(client, build_id)
    listed = client.get(
        f"/api/resume-build/builds?account_id={posting['account_id']}"
        f"&job_posting_id={posting['job_posting_id']}"
    ).json()
    other = client.get(
        f"/api/resume-build/builds?account_id={posting['account_id']}&job_posting_id=999999"
    ).json()

    assert [b["id"] for b in listed] == [build_id]
    assert other == []
    assert client.post(f"/api/resume-build/builds/{build_id}/dismiss").json() == {
        "dismissed": True
    }
    assert client.get(
        f"/api/resume-build/builds?account_id={posting['account_id']}"
    ).json() == []


def test_unknown_build_is_404():
    assert _client().get("/api/resume-build/builds/nope").status_code == 404


def test_a_crashing_build_is_marked_failed_not_left_running():
    def _crash(build):
        raise RuntimeError("boom")

    build = background.start(1, 2, "label", _crash)
    build_thread_done = time.monotonic() + 5
    while build.state == "running" and time.monotonic() < build_thread_done:
        time.sleep(0.01)

    assert build.state == "error"
    assert build.detail == "Internal error, see server logs."


def test_old_finished_builds_are_pruned_per_account():
    for _ in range(background._KEEP_FINISHED + 3):
        build = background.start(7, 1, "label", lambda b: b.finish(b"%PDF", {}))
        while build.state == "running":
            time.sleep(0.005)
    background.start(7, 1, "last", lambda b: b.finish(b"%PDF", {}))

    assert len(background.list_for_account(7)) <= background._KEEP_FINISHED + 1


def _library(client: TestClient, account_id: int) -> list[dict]:
    return client.get(f"/api/resume?account_id={account_id}").json()


def test_a_build_that_stops_after_tailoring_is_kept_and_retry_skips_that_step(
    posting, monkeypatch
):
    from app.resume_build.compile import CompileError

    data = {
        "summary": "s", "skills": ["Python"], "projects": [{"name": "p"}],
        "experience": [{"id": 1}], "education": [], "reserve": {"skills": ["Go"]},
    }
    tailor = MagicMock(return_value=data)
    monkeypatch.setattr("app.api.resume_build.build_resume_data", tailor)

    def _timeout(*args, **kwargs):
        raise CompileError("tectonic timed out after 60s")

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _timeout)
    client = _client()

    first = _wait_done(client, client.post("/api/resume-build/start", json=posting).json()["id"])

    assert first["state"] == "error"
    assert "tectonic timed out after 60s" in first["detail"]
    resume_id = first["incomplete_resume_id"]
    [row] = _library(client, posting["account_id"])
    assert row["id"] == resume_id
    assert row["build_incomplete"] is True
    assert row["build_stopped_at"] == "Fitting the resume to the page count"
    steps = {s["label"]: s for s in row["build_progress"]}
    assert steps["Summary"]["done"] is True
    assert steps["Projects"]["note"] == "1 project"
    assert steps["Final review: fitting to the page count"]["done"] is False
    # Nothing to match against yet, so the closest-resume search skips it.
    assert client.get(
        "/api/resume/search",
        params={"account_id": posting["account_id"], "job_posting_id": posting["job_posting_id"]},
    ).json() == []

    seen = {}

    def _fit(data, template, max_pages, **kwargs):
        seen["data"] = dict(data)
        return FitResult(tex="x", pdf_bytes=b"%PDF-retry", page_count=1, cuts_made=0)

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _fit)
    retry = client.post(f"/api/resume-build/retry/{resume_id}").json()
    assert retry["retry_of"] == resume_id
    done = _wait_done(client, retry["id"])

    assert done["state"] == "done"
    assert done["resume_id"] == resume_id
    assert tailor.call_count == 1
    assert seen["data"]["reserve"] == {"skills": ["Go"]}
    [row] = _library(client, posting["account_id"])
    assert row["build_incomplete"] is False
    assert row["has_ai_edited_version"] is True
    assert row["summary"] == "s"


def test_a_build_that_stops_before_anything_finished_is_kept_and_retried_whole(
    posting, monkeypatch
):
    def _busy(*args, **kwargs):
        raise LLMRateLimitedError("gemini-3.5-flash is getting more requests than it can handle")

    monkeypatch.setattr("app.api.resume_build.build_resume_data", _busy)
    client = _client()
    body = {**posting, "name": "For Acme", "selected_skills": ["Python"]}

    first = _wait_done(client, client.post("/api/resume-build/start", json=body).json()["id"])

    resume_id = first["incomplete_resume_id"]
    [row] = _library(client, posting["account_id"])
    assert row["name"] == "For Acme"
    assert not any(step["done"] for step in row["build_progress"])

    again = _wait_done(client, client.post(f"/api/resume-build/retry/{resume_id}").json()["id"])
    assert again["state"] == "error"
    assert again["incomplete_resume_id"] == resume_id
    [row] = _library(client, posting["account_id"])
    assert row["build_attempts"] == 2

    tailor = MagicMock(return_value={"summary": "s", "skills": ["Python"], "projects": []})
    monkeypatch.setattr("app.api.resume_build.build_resume_data", tailor)
    fit = FitResult(tex="x", pdf_bytes=b"%PDF", page_count=1, cuts_made=0)
    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", MagicMock(return_value=fit))

    done = _wait_done(client, client.post(f"/api/resume-build/retry/{resume_id}").json()["id"])

    assert done["state"] == "done"
    assert tailor.call_args.kwargs["selected_skills"] == ["Python"]
    assert len(_library(client, posting["account_id"])) == 1


def test_retry_refuses_a_finished_resume(posting, monkeypatch):
    _fake_build(monkeypatch)
    client = _client()
    state = _wait_done(client, client.post("/api/resume-build/start", json=posting).json()["id"])

    resp = client.post(f"/api/resume-build/retry/{state['resume_id']}")

    assert resp.status_code == 409
    assert client.post("/api/resume-build/retry/9999").status_code == 404
