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


def test_generate_says_which_step_ran_out_of_keys(monkeypatch):
    """By the time this reaches the route, app/core/llm.py has tried every
    stored key. What the route adds is where in the build that happened,
    so the answer is not just "generation failed"."""
    def _raise(*args, **kwargs):
        raise LLMRateLimitedError("All 2 OpenAI keys failed on this request.")

    monkeypatch.setattr("app.api.resume_build.build_resume_data", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert 'stopped at "tailoring the content to the job"' in detail
    assert "All 2 OpenAI keys failed" in detail
    assert "Nothing had finished yet." in detail


def test_generate_keeps_the_resume_when_the_keys_die_during_the_fit(monkeypatch):
    """The content step is the expensive one and it already succeeded, so
    running out of keys in the page-fit step returns the compiled resume
    with the reason attached, not an error and no file."""
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))

    def _raise(*args, **kwargs):
        raise PageFitNotAchievedError(
            "the trimming step could not run, so the resume is still 2 page(s) against a "
            "target of 1: All 2 OpenAI keys failed on this request.",
            best_tex="x", best_pdf_bytes=b"%PDF-best", best_page_count=2,
        )

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )

    assert resp.status_code == 200
    assert resp.content == b"%PDF-best"
    assert resp.headers["x-page-fit-achieved"] == "false"
    assert "All 2 OpenAI keys failed" in resp.headers["x-page-fit-note"]


def test_generate_note_header_is_one_bounded_line(monkeypatch):
    """The message lists one key per line; a header value cannot carry
    newlines, and an over-long one gets dropped in transit."""
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))

    def _raise(*args, **kwargs):
        raise PageFitNotAchievedError(
            "could not trim:\n- Personal: out of quota\n- Work: rejected\n" + "x" * 900,
            best_tex="x", best_pdf_bytes=b"%PDF-best", best_page_count=2,
        )

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _raise)

    resp = _client().post(
        "/api/resume-build/generate", json={"account_id": 1, "job_posting_id": 2}
    )

    note = resp.headers["x-page-fit-note"]
    assert "\n" not in note
    assert len(note) <= 400
    assert note.startswith("could not trim: - Personal: out of quota - Work: rejected")
    assert note.endswith("...")


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

    db_module.reset_engine()
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
        json={
            "account_id": account_id, "job_posting_id": posting_id, "template": "onepage",
            "name": "  Acme backend  ",
        },
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
    assert row.name == "Acme backend"
    assert Path(row.compiled_path).read_bytes() == b"%PDF-fake"
    db.close()


def test_generate_defaults_max_pages_by_template(monkeypatch):
    monkeypatch.setattr("app.api.resume_build.build_resume_data", MagicMock(return_value={}))
    captured = {}

    def _fake_fit(data, template, max_pages, account_id=None, **kwargs):
        captured["max_pages"] = max_pages
        return FitResult(tex="x", pdf_bytes=b"%PDF", page_count=max_pages, cuts_made=0)

    monkeypatch.setattr("app.api.resume_build.fit_to_page_limit", _fake_fit)

    _client().post(
        "/api/resume-build/generate",
        json={"account_id": 1, "job_posting_id": 2, "template": "twopage"},
    )

    assert captured["max_pages"] == 2


def test_options_endpoint_returns_selections(tmp_path, monkeypatch):
    import os

    import app.core.db as db_module
    from app.core.db import Account, JobPosting, Repository, get_db, init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()

    db = get_db()
    account = Account(
        first_name="Ada",
        last_name="Lovelace",
        github_username="octocat",
        contact_email="ada@example.com",
    )
    db.add(account)
    db.commit()
    db.refresh(account)

    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="opt", company="Acme",
        title="Backend Engineer",
        raw_text_quarantined="Python and FastAPI developer",
        content_hash="opthash",
    )
    db.add(posting)

    repo = Repository(
        account_id=account.id,
        github_id=101,
        name="otw",
        full_name="octocat/otw",
        url="https://github.com/octocat/otw",
    )
    db.add(repo)
    db.commit()

    account_id, posting_id = account.id, posting.id
    db.close()

    resp = _client().get(
        f"/api/resume-build/options?account_id={account_id}&job_posting_id={posting_id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "emails" in data
    assert "phones" in data
    assert "projects" in data
    assert "skills" in data
    assert "experience" in data



def _seed_skills_posting(tmp_path, required, skills):
    import os

    import app.core.db as db_module
    from app.core.db import Account, JobPosting, Skill, get_db, init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()

    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="t", company="Acme",
        title="Backend Engineer", raw_text_quarantined="Go and PostgreSQL",
        content_hash="tiers", extracted_json={"skills_required": required},
        extraction_status="extracted",
    )
    db.add(posting)
    db.add_all([Skill(account_id=account.id, name=n) for n in skills])
    db.commit()
    ids = account.id, posting.id
    db.close()
    return ids


def test_options_tiers_skills_against_the_posting(tmp_path, monkeypatch):
    account_id, posting_id = _seed_skills_posting(
        tmp_path, ["Python", "PostgreSQL", "Haskell"], ["python", "MySQL", "Cooking"]
    )
    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", lambda *a, **k: [])
    monkeypatch.setattr(
        "app.resume_build.skill_match.suggest_related",
        lambda required, have, exclude: {"MySQL": ["PostgreSQL"]},
    )

    data = _client().get(
        f"/api/resume-build/options?account_id={account_id}&job_posting_id={posting_id}"
    ).json()

    tiers = {s["name"]: (s["tier"], s["recommended"]) for s in data["skills"]}
    assert tiers == {
        "python": ("exact", True),
        "MySQL": ("related", False),
        "Cooking": ("other", False),
    }
    assert data["required_skills"] == ["Python", "PostgreSQL", "Haskell"]
    assert data["missing_required"] == ["Haskell"]


def test_skill_review_only_judges_skills_the_account_has(tmp_path, monkeypatch):
    from app.resume_build.skill_match import Verdict

    account_id, posting_id = _seed_skills_posting(tmp_path, ["PostgreSQL"], ["MySQL"])
    seen = {}

    def _fake_review(account_id, job_title, required, suggestions):
        seen["suggestions"] = suggestions
        return {"MySQL": Verdict(True, "Same kind of database.")}

    monkeypatch.setattr("app.resume_build.skill_match.review_related", _fake_review)

    resp = _client().post(
        "/api/resume-build/skill-review",
        json={
            "account_id": account_id, "job_posting_id": posting_id,
            "skills": [
                {"name": "MySQL", "suggested_for": ["PostgreSQL"]},
                {"name": "Kubernetes", "suggested_for": []},
            ],
        },
    )

    assert resp.status_code == 200
    assert seen["suggestions"] == {"MySQL": ["PostgreSQL"]}
    assert resp.json() == [{"name": "MySQL", "keep": True, "reason": "Same kind of database."}]
