"""An AI provider that is down, out of budget or missing a key never
blocks or breaks the non-LLM part of a request: signup and uploads answer
at once, the failure is kept on the row with an error_kind, a retry reuses
what was saved, and the same signup sent twice makes one profile."""

import io
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import func, select

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core import jobs
from app.core.app_settings import get_llm_settings
from app.core.db import Account, JobPosting, Resume, get_db, init_db
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMProviderError,
    LLMRateLimitedError,
    LLMUnavailableError,
    _record_health,
    complete,
    error_kind,
    llm_health,
    user_message,
)
from app.core.settings import get_settings

_PDF = b"%PDF-1.4 fake"

_TYPED_ERRORS = [
    (LLMUnavailableError("Gemini is overloaded right now."), "provider_unavailable", 503),
    (LLMRateLimitedError("Gemini rate-limited this key."), "provider_unavailable", 503),
    (ApiKeyMissingError("No Gemini API key is available."), "no_key", 422),
    (LLMProviderError("Gemini rejected the key."), "provider_rejected", 502),
    (BudgetExceededError("The monthly budget is used up."), "budget", 402),
]


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("RESUME_STORAGE_DIR", str(tmp_path / "resumes"))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    init_db()
    _record_health(None)
    yield
    _record_health(None)


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _llm_raises(monkeypatch, error: Exception) -> MagicMock:
    fake = MagicMock(side_effect=error)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake)
    return fake


def _llm_answers(monkeypatch) -> MagicMock:
    response = MagicMock()
    response.parsed = {
        "tags": ["Python"],
        "target_roles": ["Backend Engineer"],
        "summary": "Backend generalist.",
        "experiences": [],
        "education": [],
    }
    fake = MagicMock(return_value=response)
    monkeypatch.setattr("app.profile.resume_extract.complete", fake)
    return fake


def _real_background(monkeypatch) -> None:
    import app.profile.resume_ingest as resume_ingest

    monkeypatch.setattr(
        resume_ingest, "_start_worker", lambda key, work: jobs.start(key, lambda job: work())
    )


def _wait_for_read(resume_id: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = jobs.snapshot(f"resume_read:{resume_id}")
        if state is not None and not state["running"]:
            return
        time.sleep(0.02)
    raise AssertionError("resume read did not finish")


def _count(model) -> int:
    db = get_db()
    try:
        return db.execute(select(func.count()).select_from(model)).scalar_one()
    finally:
        db.close()


def _signup(client, **extra):
    data = {"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"}
    data.update(extra)
    return client.post(
        "/accounts",
        data=data,
        files={"resume": ("resume.pdf", io.BytesIO(_PDF), "application/pdf")},
    )


# --- signup while the provider is down ----------------------------------


def test_signup_answers_at_once_while_the_provider_is_down(monkeypatch):
    _real_background(monkeypatch)
    release = threading.Event()

    def slow_then_unavailable(*args, **kwargs):
        release.wait(5)
        raise LLMUnavailableError("Gemini is overloaded right now. Try again in a few minutes.")

    monkeypatch.setattr("app.profile.resume_extract.complete", slow_then_unavailable)

    started = time.monotonic()
    resp = _signup(_client(), request_id="form-1")
    elapsed = time.monotonic() - started

    assert resp.status_code == 200
    assert elapsed < 2
    body = resp.json()
    assert body["resume_extraction"] == "pending"
    assert _count(Account) == 1

    release.set()
    _wait_for_read(body["resume_id"])

    listed = _client().get(f"/api/resume?account_id={body['id']}").json()
    assert len(listed) == 1
    assert listed[0]["extraction_status"] == "failed"
    assert listed[0]["extraction_error_kind"] == "provider_unavailable"
    assert "overloaded" in listed[0]["extraction_error"]


@pytest.mark.parametrize(("error", "kind", "_status"), _TYPED_ERRORS)
def test_each_llm_error_is_kept_on_the_resume_with_its_kind(monkeypatch, error, kind, _status):
    _llm_raises(monkeypatch, error)

    resp = _signup(_client())

    assert resp.status_code == 200
    assert resp.json()["resume_extraction"] == "failed"
    row = _client().get(f"/api/resume?account_id={resp.json()['id']}").json()[0]
    assert row["extraction_error_kind"] == kind


def test_a_non_llm_failure_has_no_kind(monkeypatch):
    _llm_raises(monkeypatch, ValueError("LLM response was not valid JSON"))

    resp = _signup(_client())

    row = _client().get(f"/api/resume?account_id={resp.json()['id']}").json()[0]
    assert row["extraction_status"] == "failed"
    assert row["extraction_error_kind"] is None


# --- retry ---------------------------------------------------------------


def test_retry_reads_the_saved_file_again_without_new_rows(monkeypatch):
    _llm_raises(monkeypatch, LLMUnavailableError("Gemini is overloaded right now."))
    client = _client()
    created = _signup(client).json()
    resume_id = created["resume_id"]

    fake = _llm_answers(monkeypatch)
    resp = client.post(f"/api/resume/{resume_id}/reprocess")

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == resume_id
    assert body["extraction_status"] == "extracted"
    assert body["extraction_error_kind"] is None
    assert body["tags"] == ["Python"]
    assert fake.call_args.kwargs["bypass_cache"] is True
    assert _count(Account) == 1
    assert _count(Resume) == 1
    stored = list((Path(get_settings().resume_storage_dir) / str(created["id"])).iterdir())
    assert len(stored) == 1


def test_retry_while_a_read_is_running_does_not_start_another(monkeypatch):
    _real_background(monkeypatch)
    release = threading.Event()
    calls: list[int] = []

    def slow(*args, **kwargs):
        calls.append(1)
        release.wait(5)
        raise LLMUnavailableError("still down")

    monkeypatch.setattr("app.profile.resume_extract.complete", slow)
    client = _client()
    resume_id = _signup(client).json()["resume_id"]

    resp = client.post(f"/api/resume/{resume_id}/reprocess")
    release.set()
    _wait_for_read(resume_id)

    assert resp.json()["extraction_status"] == "pending"
    assert len(calls) == 1


def test_a_read_that_crashes_never_stays_pending(monkeypatch):
    _llm_answers(monkeypatch)
    client = _client()
    resume_id = _signup(client).json()["resume_id"]

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("app.profile.resume_ingest.run_extraction", boom)
    resp = client.post(f"/api/resume/{resume_id}/reprocess")

    assert resp.json()["extraction_status"] == "failed"
    assert resp.json()["extraction_error"] == "Internal error, see server logs."


def test_a_read_of_a_deleted_resume_does_nothing():
    from app.profile.resume_ingest import _read_resume

    _read_resume(12345)


def test_restart_fails_uploads_left_pending_but_not_unfinished_builds():
    from app.profile.resume_ingest import fail_interrupted_reads

    db = get_db()
    account = Account(first_name="A", last_name="B", github_username="")
    db.add(account)
    db.commit()
    upload = Resume(
        account_id=account.id, filename="r.pdf", mime_type="application/pdf",
        stored_path="/tmp/r.pdf", extraction_status="pending",
    )
    build = Resume(
        account_id=account.id, filename="b.pdf", mime_type="application/pdf",
        extraction_status="pending", build_state_json={"error": "stopped"},
    )
    db.add_all([upload, build])
    db.commit()
    upload_id, build_id = upload.id, build.id
    db.close()

    assert fail_interrupted_reads() == 1

    db = get_db()
    assert db.get(Resume, upload_id).extraction_status == "failed"
    assert "restarted" in db.get(Resume, upload_id).extraction_error
    assert db.get(Resume, build_id).extraction_status == "pending"
    db.close()


# --- double submit -------------------------------------------------------


def test_the_same_signup_sent_twice_makes_one_profile(monkeypatch):
    _llm_answers(monkeypatch)
    client = _client()

    first = _signup(client, request_id="form-abc")
    second = _signup(client, request_id="form-abc")

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert _count(Account) == 1
    assert _count(Resume) == 1


def test_an_identical_signup_within_a_minute_is_the_same_profile(monkeypatch):
    _llm_answers(monkeypatch)
    client = _client()

    first = _signup(client)
    second = _signup(client)

    assert first.json()["id"] == second.json()["id"]
    assert _count(Account) == 1


def test_an_old_identical_signup_is_a_new_profile(monkeypatch):
    import datetime as dt

    _llm_answers(monkeypatch)
    client = _client()
    first = _signup(client).json()
    db = get_db()
    db.get(Account, first["id"]).created_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    db.commit()
    db.close()

    second = _signup(client).json()

    assert second["id"] != first["id"]
    assert _count(Account) == 2


def test_a_different_person_is_a_new_profile(monkeypatch):
    _llm_answers(monkeypatch)
    client = _client()

    first = _signup(client, request_id="a")
    second = _signup(client, request_id="b", first_name="Ada", last_name="Lovelace")

    assert first.json()["id"] != second.json()["id"]
    assert _count(Account) == 2


def test_a_request_id_that_lost_the_race_answers_with_the_winner(monkeypatch):
    """Two sends of one form running side by side: the second commit hits
    the unique request_id and answers with the first one's profile."""
    import app.api.accounts as accounts

    db = get_db()
    winner = Account(first_name="Grace", last_name="Hopper", github_username="", request_id="r1")
    db.add(winner)
    db.commit()
    winner_id = winner.id
    db.close()

    looked_up: list[str] = []
    real = accounts._already_created

    def misses_first_time(db, request_id, *args):
        looked_up.append(request_id)
        return None if len(looked_up) == 1 else real(db, request_id, *args)

    monkeypatch.setattr(accounts, "_already_created", misses_first_time)
    resp = _client().post(
        "/accounts", data={"first_name": "Other", "last_name": "Name", "request_id": "r1"}
    )

    assert resp.status_code == 200
    assert resp.json()["id"] == winner_id
    assert _count(Account) == 1


def test_signup_without_a_resume_reports_no_extraction():
    resp = _client().post("/accounts", data={"first_name": "Ada", "last_name": "Lovelace"})

    assert resp.status_code == 200
    assert resp.json()["resume_id"] is None
    assert resp.json()["resume_extraction"] is None


# --- error_kind in JSON responses -----------------------------------------


@pytest.mark.parametrize(("error", "kind", "status"), _TYPED_ERRORS)
def test_each_llm_error_maps_to_its_kind_in_the_response(monkeypatch, error, kind, status):
    def raise_it(*args, **kwargs):
        raise error

    monkeypatch.setattr("app.api.resume_build.generate_resume", raise_it)

    resp = _client().post(
        "/api/resume-build/preview", json={"account_id": 1, "job_posting_id": 1}
    )

    assert resp.status_code == status
    body = resp.json()
    assert body["error_kind"] == kind
    assert body["detail"] == str(error)
    assert body["model"]


@pytest.mark.parametrize(("error", "kind", "status"), _TYPED_ERRORS)
def test_an_llm_error_no_route_caught_still_answers_with_its_kind(error, kind, status):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.llm_errors import install

    app = FastAPI()
    install(app)

    @app.get("/boom")
    def boom():
        raise error

    resp = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert resp.status_code == status
    assert resp.json()["error_kind"] == kind
    assert resp.json()["detail"] == str(error)


def test_a_non_llm_error_maps_to_a_502_without_a_kind():
    from app.api.llm_errors import llm_http_error

    mapped = llm_http_error(ValueError("bad"), "resume edit failed")

    assert mapped.status_code == 502
    assert mapped.error_kind is None
    assert mapped.detail == "resume edit failed: bad"


def test_error_kind_of_anything_else_is_none():
    assert error_kind(RuntimeError("x")) is None
    for error, kind, _ in _TYPED_ERRORS:
        assert error_kind(error) == kind


def test_a_job_posting_read_keeps_the_kind(monkeypatch):
    def unavailable(*args, **kwargs):
        raise LLMUnavailableError("Gemini is overloaded right now.")

    monkeypatch.setattr("app.profile.job_extract.complete", unavailable)
    client = _client()
    account_id = client.post("/accounts", data={"first_name": "A", "last_name": "B"}).json()["id"]

    resp = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Backend engineer"}
    )

    assert resp.status_code == 200
    assert resp.json()["extraction_status"] == "failed"
    assert resp.json()["extraction_error_kind"] == "provider_unavailable"

    reprocessed = client.post(f"/api/job-postings/{resp.json()['id']}/reprocess").json()
    assert reprocessed["extraction_error_kind"] == "provider_unavailable"


def test_a_screenshot_only_posting_keeps_the_kind(monkeypatch):
    def unavailable(*args, **kwargs):
        raise LLMUnavailableError("Gemini is overloaded right now.")

    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images", unavailable
    )
    monkeypatch.setenv("JOB_SCREENSHOT_STORAGE_DIR", str(Path(get_settings().resume_storage_dir)))
    get_settings.cache_clear()
    client = _client()
    account_id = client.post("/accounts", data={"first_name": "A", "last_name": "B"}).json()["id"]
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 32

    resp = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", io.BytesIO(png), "image/png")},
    )

    assert resp.status_code == 200
    assert resp.json()["extraction_status"] == "failed"
    assert resp.json()["extraction_error_kind"] == "provider_unavailable"


# --- resume build failures ---------------------------------------------------


def test_a_failed_build_keeps_the_kind_on_its_incomplete_resume(monkeypatch):
    from app.api import resume_build
    from app.resume_build import background

    def unavailable(*args, **kwargs):
        raise LLMUnavailableError("Gemini is overloaded right now.")

    monkeypatch.setattr(resume_build, "build_resume_data", unavailable)
    db = get_db()
    account = Account(first_name="A", last_name="B", github_username="")
    db.add(account)
    db.commit()
    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="hiring", content_hash="h",
    )
    db.add(posting)
    db.commit()
    ids = (account.id, posting.id)
    db.close()

    client = _client()
    build = client.post(
        "/api/resume-build/start", json={"account_id": ids[0], "job_posting_id": ids[1]}
    ).json()
    deadline = time.monotonic() + 5
    while background.get(build["id"]).public()["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    finished = client.get(f"/api/resume-build/builds/{build['id']}").json()
    assert finished["error_kind"] == "provider_unavailable"
    resume = client.get(f"/api/resume?account_id={ids[0]}").json()[0]
    assert resume["build_error_kind"] == "provider_unavailable"


# --- health --------------------------------------------------------------


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list


def _overloaded():
    import litellm

    return litellm.ServiceUnavailableError(
        message="overloaded", llm_provider="gemini", model="gemini-flash-latest"
    )


def test_health_degrades_when_the_provider_is_down_and_clears_on_success(monkeypatch):
    monkeypatch.setattr("app.core.llm._sleep", lambda s: None)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0, raising=False)
    client = _client()
    assert client.get("/api/app-settings/llm/health").json()["degraded"] is False

    def down(**kwargs):
        raise _overloaded()

    with pytest.raises(LLMUnavailableError):
        complete("quality", [user_message("hi")], _completion_fn=down)

    health = client.get("/api/app-settings/llm/health").json()
    assert health["degraded"] is True
    assert health["model"] == get_llm_settings().model_for("quality")

    complete(
        "quality",
        [user_message("hello again")],
        _completion_fn=lambda **kw: _Response([_Choice(_Message("ok"))]),
    )
    assert llm_health()["degraded"] is False


def test_health_forgets_an_old_failure(monkeypatch):
    import app.core.llm as llm

    _record_health("gemini/x", "down")
    monkeypatch.setattr(llm, "_HEALTH_WINDOW_S", 0.0)

    assert llm_health() == {"degraded": False, "model": None, "detail": None}
