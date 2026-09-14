"""app/api/resume_build.py: /api/resume-build/preview and /generate.
generate_resume/build_resume_data/fit_to_page_limit are all mocked at the
router boundary, matching this codebase's existing style for endpoint
tests where the underlying pipeline is covered elsewhere (test_orchestrator.py,
test_pagefit.py, test_compile.py). What's under test here is routing,
request validation, and error-to-status-code mapping.
"""

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.main import app
from app.core.llm import ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError
from app.resume_build.compile import CompileError, TectonicNotInstalledError
from app.resume_build.pagefit import FitResult, PageFitNotAchievedError


def _client() -> TestClient:
    return TestClient(app)


def test_preview_returns_tex(monkeypatch):
    monkeypatch.setattr(
        "app.api.resume_build.generate_resume", MagicMock(return_value=r"\documentclass{article}")
    )

    resp = _client().post(
        "/api/resume-build/preview",
        json={"account_id": 1, "job_posting_id": 2, "template": "onepage"},
    )

    assert resp.status_code == 200
    assert resp.json()["tex"] == r"\documentclass{article}"


def test_preview_rejects_unknown_template():
    resp = _client().post(
        "/api/resume-build/preview",
        json={"account_id": 1, "job_posting_id": 2, "template": "threepage"},
    )
    assert resp.status_code == 422


def test_preview_maps_unknown_account_or_posting_to_404(monkeypatch):
    def _raise(*args, **kwargs):
        raise ValueError("no account with id=999")

    monkeypatch.setattr("app.api.resume_build.generate_resume", _raise)

    resp = _client().post(
        "/api/resume-build/preview",
        json={"account_id": 999, "job_posting_id": 2},
    )
    assert resp.status_code == 404


def test_preview_maps_missing_api_key_to_422(monkeypatch):
    def _raise(*args, **kwargs):
        raise ApiKeyMissingError("no key configured")

    monkeypatch.setattr("app.api.resume_build.generate_resume", _raise)

    resp = _client().post(
        "/api/resume-build/preview", json={"account_id": 1, "job_posting_id": 2}
    )
    assert resp.status_code == 422


def test_preview_maps_budget_exceeded_to_402(monkeypatch):
    def _raise(*args, **kwargs):
        raise BudgetExceededError("budget reached")

    monkeypatch.setattr("app.api.resume_build.generate_resume", _raise)

    resp = _client().post(
        "/api/resume-build/preview", json={"account_id": 1, "job_posting_id": 2}
    )
    assert resp.status_code == 402


def test_preview_maps_rate_limited_to_503(monkeypatch):
    def _raise(*args, **kwargs):
        raise LLMRateLimitedError("rate limited")

    monkeypatch.setattr("app.api.resume_build.generate_resume", _raise)

    resp = _client().post(
        "/api/resume-build/preview", json={"account_id": 1, "job_posting_id": 2}
    )
    assert resp.status_code == 503


def test_generate_returns_pdf_bytes_with_headers(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))
    fit_result = FitResult(tex="x", pdf_bytes=b"%PDF-fake", page_count=1, cuts_made=2)
    monkeypatch.setattr(
        "app.api.resume_build.fit_to_page_limit", MagicMock(return_value=fit_result)
    )

    resp = _client().post(
        "/api/resume-build/generate",
        json={"account_id": 1, "job_posting_id": 2, "template": "onepage"},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["x-page-fit-achieved"] == "true"
    assert resp.headers["x-page-count"] == "1"
    assert resp.content == b"%PDF-fake"


def test_generate_still_returns_best_pdf_on_unfit(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))

    def _raise(*args, **kwargs):
        raise PageFitNotAchievedError(
            "nope", best_tex="x", best_pdf_bytes=b"%PDF-best", best_page_count=2
        )

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )

    assert resp.status_code == 200
    assert resp.headers["x-page-fit-achieved"] == "false"
    assert resp.headers["x-page-count"] == "2"
    assert resp.content == b"%PDF-best"


def test_generate_maps_missing_tectonic_to_501(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))

    def _raise(*args, **kwargs):
        raise TectonicNotInstalledError("no tectonic")

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )
    assert resp.status_code == 501


def test_generate_maps_compile_error_to_502(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))

    def _raise(*args, **kwargs):
        raise CompileError("bad latex")

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )
    assert resp.status_code == 502


def test_generate_maps_unknown_account_to_404(monkeypatch):
    def _raise(*args, **kwargs):
        raise ValueError("no account with id=999")

    monkeypatch.setattr("app.api.resume_build.build_resume_data", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 999, "job_posting_id": 2}
    )
    assert resp.status_code == 404


def test_generate_saves_result_into_resume_library(tmp_path, monkeypatch):
    """Every generated resume lands in the shared library (/resume), see
    app/api/resume_build.py's _save_generated_resume(), not only ever
    returned as a one-off download."""
    import os

    import app.core.db as db_module
    import app.retrieval.vectorstore as vectorstore_module
    from app.core.db import Account, JobPosting, Resume, get_db, init_db
    from app.core.settings import get_settings

    db_module._engine = None
    db_module._SessionLocal = None
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["RESUME_STORAGE_DIR"] = str(tmp_path / "resumes")
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="h", company="Acme",
        title="Backend Engineer", raw_text_quarantined="hiring text", content_hash="h",
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)
    account_id, posting_id = account.id, posting.id
    db.close()

    fake_data = {
        "summary": "A tailored summary.", "skills": ["Python"], "projects": [],
        "experience": [], "education": [], "technologies": [],
    }
    monkeypatch.setattr(
        "app.api.resume_build.build_resume_data", MagicMock(return_value=fake_data)
    )
    fit_result = FitResult(tex="x", pdf_bytes=b"%PDF-fake", page_count=1, cuts_made=0)
    monkeypatch.setattr(
        "app.api.resume_build.fit_to_page_limit", MagicMock(return_value=fit_result)
    )

    resp = _client().post(
        "/api/resume-build/generate",
        json={"account_id": account_id, "job_posting_id": posting_id, "template": "onepage"},
    )

    assert resp.status_code == 200
    resume_id = int(resp.headers["x-resume-id"])

    db = get_db()
    row = db.get(Resume, resume_id)
    assert row is not None
    assert row.job_posting_id == posting_id
    assert row.template == "onepage"
    assert row.content_json == fake_data
    assert row.summary == "A tailored summary."
    assert Path(row.compiled_path).read_bytes() == b"%PDF-fake"
    db.close()


def test_generate_defaults_max_pages_by_template(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))
    captured = {}

    def _fake_fit(data, template, max_pages, account_id=None):
        captured["max_pages"] = max_pages
        return FitResult(tex="x", pdf_bytes=b"%PDF", page_count=max_pages, cuts_made=0)

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _fake_fit)

    _client().post(
        "/api/resume-build/generate",
        json={"account_id": 1, "job_posting_id": 2, "template": "twopage"},
    )

    assert captured["max_pages"] == 2
