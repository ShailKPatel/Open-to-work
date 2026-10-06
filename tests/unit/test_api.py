import io
from pathlib import Path

from github import UnknownObjectException

import app.core.db as db_module
from app.core.db import init_db
from app.core.embeddings import EMBEDDING_MODEL
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["RESUME_STORAGE_DIR"] = str(tmp_path / "resumes")
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def test_health(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_serves_account_picker(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Who's working?" in resp.text
    assert "GitHub username" in resp.text  # inside the create-profile form


def test_home_page_serves_dashboard(tmp_path):
    """The home dashboard renders, including its "Pending" KPI tile (see
    app/web/templates/home.html's homeApp().pendingCount).
    """
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/home")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Pending" in resp.text


def test_portfolio_overview_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Portfolio" in resp.text


def test_sync_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/monitor/sync")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Skill extraction" in resp.text
    assert "GitHub" in resp.text


def test_jobs_analytics_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/jobs/analytics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Insights" in resp.text


def test_resume_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio/resume")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Resume" in resp.text


def test_resume_build_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio/resume/build")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Build a resume" in resp.text


def test_education_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio/education")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Education" in resp.text


def test_contact_links_page_serves_html(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio/contact-links")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Contact &amp; Links" in resp.text


def test_explanation_page_serves_html(tmp_path):
    """Static presentation page: renders with no account, and its diagram
    boxes (the data-node hooks the wire script attaches arrows to) come
    through the macro import intact."""
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/explanation")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Tech stack" in resp.text
    assert 'data-node="llm"' in resp.text
    assert 'data-node="p-ollama"' in resp.text
    assert "Testing and quality" in resp.text
    assert 'aria-label="pytest"' in resp.text
    # Data flow tab: its include renders, and hover cards name the model
    # this instance is configured with, not a hardcoded default.
    assert 'data-wires="flowA"' in resp.text
    assert EMBEDDING_MODEL in resp.text


def test_projects_page_serves_html_not_the_api_endpoint(tmp_path):
    """Regression test: GET /portfolio/projects (the page) and
    GET /api/projects (the JSON list) used to collide at the same path
    before the /api prefix, the API router won route registration order
    and the page was unreachable, returning a 422 JSON body instead of
    HTML. Caught live.
    """
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/portfolio/projects")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Projects" in resp.text


def test_list_accounts_empty(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.get("/accounts")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_account_without_resume(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["first_name"] == "Ada"
    assert body["github_username"] == "octocat"
    assert body["has_resume"] is False

    listed = client.get("/accounts").json()
    assert len(listed) == 1
    assert listed[0]["id"] == body["id"]


def test_create_account_without_github_username(tmp_path):
    # Non-technical signup / no GitHub account, field is optional, no
    # existence check, and no sync source seeded for the fetch-data page.
    _reset_db(tmp_path)
    client = _client()
    resp = client.post(
        "/accounts",
        data={"first_name": "Jane", "last_name": "Doe", "github_username": ""},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["github_username"] == ""

    listed = client.get("/accounts").json()
    assert len(listed) == 1
    assert listed[0]["github_username"] == ""


def test_create_account_with_resume_stores_file(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
        files={"resume": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_resume"] is True

    # Stored filename is prefixed with the Resume row's own id (see
    # app/profile/resume_ingest.py) so a second upload named "resume.pdf"
    # for the same account never collides with this one.
    stored = list((tmp_path / "resumes" / str(body["id"])).iterdir())
    assert len(stored) == 1
    assert stored[0].name.endswith("_resume.pdf")
    assert stored[0].read_bytes() == b"%PDF-1.4 fake"


def test_create_account_with_a_large_resume_asks_first(tmp_path, monkeypatch):
    from app.api import input_limits

    _reset_db(tmp_path)
    monkeypatch.setattr(input_limits, "SOFT_MAX_FILE_MB", 0.001)
    client = _client()
    form = {"first_name": "Grace", "last_name": "Hopper", "github_username": ""}

    big = b"%PDF-1.4 " + b"x" * 4096

    def _send(**extra):
        return client.post(
            "/accounts",
            data={**form, **extra},
            files={"resume": ("resume.pdf", io.BytesIO(big), "application/pdf")},
        )

    asked = _send()
    assert asked.status_code == 409
    assert asked.json()["detail"]["code"] == "large_input"
    # Checked before the account exists, so saying no leaves nothing behind.
    assert client.get("/accounts").json() == []

    assert _send(confirm_large="true").status_code == 200
    assert len(client.get("/accounts").json()) == 1


def test_create_account_sanitizes_resume_filename_path_traversal(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.post(
        "/accounts",
        data={"first_name": "Eve", "last_name": "X", "github_username": "evex"},
        files={
            "resume": ("../../etc/passwd", io.BytesIO(b"not actually passwd"), "text/plain")
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    stored_dir = tmp_path / "resumes" / str(body["id"])
    stored = list(stored_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].name.endswith("_passwd")  # basename only, no path components
    assert stored[0].parent == stored_dir  # never escaped the account's own dir


def test_github_user_exists_true_false_none(monkeypatch):
    from github import BadCredentialsException

    from app.api.accounts import _github_user_exists

    class FakeClientOk:
        def __init__(self, *a, **k):
            pass

        def repo_count_hint(self, username):
            return 5

    class FakeClientNotFound:
        def __init__(self, *a, **k):
            pass

        def repo_count_hint(self, username):
            raise UnknownObjectException(404, "Not Found", {})

    class FakeClientBroken:
        def __init__(self, *a, **k):
            pass

        def repo_count_hint(self, username):
            raise BadCredentialsException(401, "Bad credentials", {})

    monkeypatch.setattr("app.api.accounts.GitHubClient", FakeClientOk)
    assert _github_user_exists("octocat") is True

    monkeypatch.setattr("app.api.accounts.GitHubClient", FakeClientNotFound)
    assert _github_user_exists("nobody-xyz") is False

    monkeypatch.setattr("app.api.accounts.GitHubClient", FakeClientBroken)
    assert _github_user_exists("octocat") is None


def test_create_account_unknown_github_username_rejected(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr("app.api.accounts._github_user_exists", lambda username: False)
    client = _client()

    resp = client.post(
        "/accounts",
        data={
            "first_name": "Nobody",
            "last_name": "Real",
            "github_username": "this-user-does-not-exist-xyz",
        },
    )

    assert resp.status_code == 422
    assert "this-user-does-not-exist-xyz" in resp.json()["detail"]
    assert client.get("/accounts").json() == []  # nothing was created


def test_create_account_proceeds_when_github_check_unavailable(tmp_path, monkeypatch):
    """A dead token / rate limit / network hiccup while verifying shouldn't
    block someone from making a local profile, only a confirmed 404
    should. See _github_user_exists's None case."""
    _reset_db(tmp_path)
    monkeypatch.setattr("app.api.accounts._github_user_exists", lambda username: None)
    client = _client()

    resp = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    )

    assert resp.status_code == 200
    assert resp.json()["github_username"] == "octocat"


def test_multiple_accounts_coexist_independently(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    a = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()
    b = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
    ).json()

    listed = client.get("/accounts").json()
    assert {row["id"] for row in listed} == {a["id"], b["id"]}
    assert a["id"] != b["id"]


def test_delete_account_removes_it_and_its_repos_and_evidence(tmp_path):
    from app.core.db import Repository, SkillEvidence, get_db

    _reset_db(tmp_path)
    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()

    db = get_db()
    repo = Repository(
        account_id=account["id"],
        github_id=1,
        name="proj",
        full_name="octocat/proj",
        url="https://github.com/octocat/proj",
        is_fork=False,
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    db.add(
        SkillEvidence(
            skill="Rust", repo_id=repo.id, evidence_type="declared_dependency",
            weight=0.5, confidence=1.0,
        )
    )
    db.commit()
    repo_id = repo.id
    db.close()

    resp = client.delete(f"/accounts/{account['id']}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True, "account_id": account["id"]}

    assert client.get("/accounts").json() == []

    db = get_db()
    assert db.get(Repository, repo_id) is None
    assert db.query(SkillEvidence).filter_by(repo_id=repo_id).count() == 0
    db.close()


def test_delete_account_removes_resume_file(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
        files={"resume": ("resume.pdf", io.BytesIO(b"fake"), "application/pdf")},
    ).json()
    resume_dir = tmp_path / "resumes" / str(account["id"])
    assert resume_dir.exists()

    client.delete(f"/accounts/{account['id']}")

    assert not resume_dir.exists()


def test_delete_account_removes_job_postings(tmp_path):
    from app.core.db import JobPosting, get_db

    _reset_db(tmp_path)
    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()

    client.post(
        "/api/job-postings", json={"account_id": account["id"], "raw_text": "A job posting"}
    )

    resp = client.delete(f"/accounts/{account['id']}")
    assert resp.status_code == 200

    db = get_db()
    assert db.query(JobPosting).filter_by(account_id=account["id"]).count() == 0
    db.close()


def test_delete_account_removes_qdrant_points_for_both_evidence_types(tmp_path, monkeypatch):
    """Regression test: repo-linked and experience-linked SkillEvidence
    share one Qdrant collection with an id offset for the experience side
    (see app/retrieval/index.py). delete_account must apply that same
    offset when cleaning up, or it either leaves the experience-linked
    points orphaned or, worse, deletes an unrelated repo-linked point that
    happens to share the same raw id.
    """
    import os

    import app.retrieval.vectorstore as vectorstore_module
    from app.core.db import (
        Experience,
        ExperienceSkillEvidence,
        Repository,
        SkillEvidence,
        get_db,
    )
    from app.retrieval.index import (
        COLLECTION,
        index_experience_skill_evidence,
        index_skill_evidence,
    )

    # Set before _reset_db, which caches settings: set after, the client
    # would still point at the default server URL instead of in-memory.
    os.environ["QDRANT_URL"] = ":memory:"
    _reset_db(tmp_path)
    vectorstore_module.get_client.cache_clear()
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts]
    )

    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()

    db = get_db()
    repo = Repository(
        account_id=account["id"], github_id=1, name="proj", full_name="octocat/proj",
        url="https://github.com/octocat/proj", is_fork=False,
    )
    db.add(repo)
    exp = Experience(account_id=account["id"], title="Eng", company="Acme")
    db.add(exp)
    db.commit()
    db.refresh(repo)
    db.refresh(exp)

    # Same raw id (1) for both, which would collide without the offset.
    repo_evidence = SkillEvidence(
        id=1, skill="Rust", repo_id=repo.id, evidence_type="declared_dependency",
        weight=0.5, confidence=1.0,
    )
    exp_evidence = ExperienceSkillEvidence(
        id=1, skill="Leadership", experience_id=exp.id, evidence_type="manual",
        weight=1.0, confidence=1.0,
    )
    db.add_all([repo_evidence, exp_evidence])
    db.commit()

    index_skill_evidence([repo_evidence], account_id=account["id"])
    index_experience_skill_evidence([exp_evidence], account_id=account["id"])
    db.close()

    qdrant = vectorstore_module.get_client()
    assert qdrant.count(COLLECTION).count == 2

    resp = client.delete(f"/accounts/{account['id']}")
    assert resp.status_code == 200

    assert qdrant.count(COLLECTION).count == 0


def test_delete_account_removes_job_screenshot_files(tmp_path, monkeypatch):
    import os

    from app.profile.job_extract import JobExtraction, RequiredSkill
    from app.profile.job_screenshot_extract import ScreenshotExtraction

    _reset_db(tmp_path)
    os.environ["JOB_SCREENSHOT_STORAGE_DIR"] = str(tmp_path / "job_screenshots")
    get_settings.cache_clear()

    fake_result = ScreenshotExtraction(
        raw_text_transcribed="A real job posting, transcribed from the screenshot image.",
        extraction=JobExtraction(
            company="Acme", title="Engineer", location="", salary_range="",
            employment_type="", seniority="", experience_required="",
            skills_required=[RequiredSkill(skill="Python", level="")],
            other_requirements=[], role_summary="",
        ),
    )
    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: fake_result,
    )

    client = _client()
    account = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()
    client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account["id"])},
        files={"file": ("shot.png", io.BytesIO(b"\x89PNG\r\n\x1a\nfake"), "image/png")},
    )
    screenshot_dir = tmp_path / "job_screenshots" / str(account["id"])
    assert screenshot_dir.exists()

    client.delete(f"/accounts/{account['id']}")

    assert not screenshot_dir.exists()


def test_delete_unknown_account_404(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    resp = client.delete("/accounts/999999")
    assert resp.status_code == 404


def test_delete_account_does_not_touch_other_accounts(tmp_path):
    _reset_db(tmp_path)
    client = _client()
    a = client.post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": "octocat"},
    ).json()
    b = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
    ).json()

    client.delete(f"/accounts/{a['id']}")

    remaining = client.get("/accounts").json()
    assert len(remaining) == 1
    assert remaining[0]["id"] == b["id"]
