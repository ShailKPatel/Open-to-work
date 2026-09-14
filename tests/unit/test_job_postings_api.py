from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    # Job posting creation now also resolves a role family (embeds the
    # title, searches Qdrant) and indexes the posting itself. Without
    # pointing this at the in-memory test collection, these tests would
    # make real network calls to whatever QDRANT_URL is configured in .env.
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    return TestClient(app)


def _make_account() -> int:
    from app.core.db import Account

    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def test_create_posting_stores_raw_text(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.post(
        "/api/job-postings",
        json={
            "account_id": account_id,
            "raw_text": "We are hiring a backend engineer with Python experience.",
            "company": "Acme",
            "title": "Backend Engineer",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "Acme"
    assert body["title"] == "Backend Engineer"
    assert body["source"] == "pasted"
    assert body["raw_text"] == "We are hiring a backend engineer with Python experience."


def test_create_posting_defaults_company_and_title_when_missing(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.post(
        "/api/job-postings",
        json={"account_id": account_id, "raw_text": "Some pasted job text with no header."},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "(unspecified)"
    assert body["title"] == "(unspecified)"


def test_create_posting_rejects_blank_text(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "   "}
    )

    assert resp.status_code == 422


def test_create_posting_dedupes_identical_text(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    text = "Identical posting text, pasted twice."

    first = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": text}
    ).json()
    second = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": text}
    ).json()

    assert first["id"] == second["id"]
    assert len(client.get(f"/api/job-postings?account_id={account_id}").json()) == 1


def test_list_postings_scoped_to_account(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account()
    client = _client()
    from app.core.db import Account

    db = get_db()
    b = Account(first_name="Grace", last_name="Hopper", github_username="ghopper")
    db.add(b)
    db.commit()
    db.refresh(b)
    account_b = b.id
    db.close()

    client.post("/api/job-postings", json={"account_id": account_a, "raw_text": "Job for A"})
    client.post("/api/job-postings", json={"account_id": account_b, "raw_text": "Job for B"})

    listed_a = client.get(f"/api/job-postings?account_id={account_a}").json()
    assert len(listed_a) == 1
    assert listed_a[0]["company"] == "(unspecified)"


def test_posting_detail_404_on_unknown_id(tmp_path):
    _reset_db(tmp_path)
    resp = _client().get("/api/job-postings/999999")
    assert resp.status_code == 404


def test_delete_posting(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Delete me"}
    ).json()

    resp = client.delete(f"/api/job-postings/{created['id']}")
    assert resp.status_code == 200
    assert client.get(f"/api/job-postings/{created['id']}").status_code == 404


def test_delete_unknown_posting_404(tmp_path):
    _reset_db(tmp_path)
    resp = _client().delete("/api/job-postings/999999")
    assert resp.status_code == 404


def _fake_extraction(monkeypatch, **overrides):
    from unittest.mock import MagicMock

    fake_response = MagicMock()
    fake_response.parsed = {
        "company": "Acme", "title": "Backend Engineer", "location": "Remote",
        "salary_range": "$120k-$150k", "employment_type": "Full-time",
        "seniority": "Senior", "experience_required": "5+ years",
        "skills_required": [{"skill": "Python", "level": "senior"}], "other_requirements": [],
        "role_summary": "Own the backend.",
        **overrides,
    }
    monkeypatch.setattr(
        "app.profile.job_extract.complete", MagicMock(return_value=fake_response)
    )


def test_create_posting_extracts_structured_fields(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()

    resp = client.post(
        "/api/job-postings",
        json={"account_id": account_id, "raw_text": "We are hiring a backend engineer."},
    )

    body = resp.json()
    assert body["extraction_status"] == "extracted"
    assert body["salary_range"] == "$120k-$150k"
    assert body["skills_required"] == [{"skill": "Python", "level": "senior"}]
    # backfilled since company/title were left unset
    assert body["company"] == "Acme"
    assert body["title"] == "Backend Engineer"


def test_create_posting_extraction_never_overwrites_user_supplied_fields(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch, company="Someone Else Inc", title="Some Other Title")
    client = _client()

    resp = client.post(
        "/api/job-postings",
        json={
            "account_id": account_id,
            "raw_text": "We are hiring a backend engineer.",
            "company": "Acme",
            "title": "Backend Engineer",
        },
    )

    body = resp.json()
    assert body["company"] == "Acme"
    assert body["title"] == "Backend Engineer"


def test_create_posting_extraction_failure_does_not_block_save(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    _reset_db(tmp_path)
    account_id = _make_account()

    def _raise(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr("app.profile.job_extract.complete", MagicMock(side_effect=_raise))
    client = _client()

    resp = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction_status"] == "failed"
    assert body["company"] == "(unspecified)"  # backfill never ran


def test_reprocess_posting(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    ).json()
    assert created["extraction_status"] == "failed"  # no LLM key in tests

    _fake_extraction(monkeypatch)
    resp = client.post(f"/api/job-postings/{created['id']}/reprocess")

    assert resp.status_code == 200
    assert resp.json()["extraction_status"] == "extracted"
    assert resp.json()["salary_range"] == "$120k-$150k"


def test_reprocess_unknown_posting_404(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post("/api/job-postings/999999/reprocess")
    assert resp.status_code == 404


def test_patch_posting_updates_fields(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    ).json()

    resp = client.patch(
        f"/api/job-postings/{created['id']}",
        json={"company": "Acme", "title": "Engineer", "apply_url": "https://acme.example/apply"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["company"] == "Acme"
    assert body["title"] == "Engineer"
    assert body["apply_url"] == "https://acme.example/apply"


def test_patch_unknown_posting_404(tmp_path):
    _reset_db(tmp_path)
    resp = _client().patch("/api/job-postings/999999", json={"company": "Acme"})
    assert resp.status_code == 404


def test_patch_applied_stamps_todays_date_by_default(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    ).json()

    resp = client.patch(f"/api/job-postings/{created['id']}", json={"applied": True})
    body = resp.json()
    assert body["applied"] is True
    assert body["applied_at"] is not None


def test_patch_applied_accepts_explicit_date_and_notes(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    ).json()

    resp = client.patch(
        f"/api/job-postings/{created['id']}",
        json={"applied": True, "applied_at": "2026-01-05", "applied_notes": "Referred by Sam"},
    )
    body = resp.json()
    assert body["applied_at"] == "2026-01-05"
    assert body["applied_notes"] == "Referred by Sam"


def test_unmarking_applied_clears_date_and_notes(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "hiring text"}
    ).json()
    client.patch(
        f"/api/job-postings/{created['id']}",
        json={"applied": True, "applied_at": "2026-01-05", "applied_notes": "note"},
    )

    resp = client.patch(f"/api/job-postings/{created['id']}", json={"applied": False})
    body = resp.json()
    assert body["applied"] is False
    assert body["applied_at"] is None
    assert body["applied_notes"] is None


def test_create_from_url_fetches_and_saves(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    monkeypatch.setattr(
        "app.api.job_postings.fetch_job_url",
        lambda url: ("Backend Engineer at Acme", "We need a backend engineer with Python."),
    )

    client = _client()
    resp = client.post(
        "/api/job-postings/from-url",
        json={"account_id": account_id, "url": "https://acme.example/careers/1"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "url"
    assert body["title"] == "Backend Engineer at Acme"
    assert body["apply_url"] == "https://acme.example/careers/1"
    assert body["raw_text"] == "We need a backend engineer with Python."


def test_create_from_url_maps_fetch_error_to_422(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    from app.ingest.jobs.url_fetch import JobUrlFetchError

    def _raise(url):
        raise JobUrlFetchError("could not reach that page")

    monkeypatch.setattr("app.api.job_postings.fetch_job_url", _raise)

    client = _client()
    resp = client.post(
        "/api/job-postings/from-url",
        json={"account_id": account_id, "url": "https://unreachable.example"},
    )
    assert resp.status_code == 422


def test_create_from_screenshot_saves_transcription_and_image(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    from app.profile.job_extract import JobExtraction, RequiredSkill
    from app.profile.job_screenshot_extract import ScreenshotExtraction

    fake_result = ScreenshotExtraction(
        raw_text_transcribed="Backend Engineer at Acme. Python required.",
        extraction=JobExtraction(
            company="Acme", title="Backend Engineer", location="Remote",
            salary_range="", employment_type="", seniority="", experience_required="",
            skills_required=[RequiredSkill(skill="Python", level="")],
            other_requirements=[], role_summary="",
        ),
    )
    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_image",
        lambda image_bytes, mime_type, account_id=None: fake_result,
    )

    client = _client()
    resp = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", b"fake-bytes", "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "screenshot"
    assert body["company"] == "Acme"
    assert body["raw_text"] == "Backend Engineer at Acme. Python required."
    assert body["screenshot_path"]
    assert Path(body["screenshot_path"]).exists()


def test_create_from_screenshot_unsupported_type_422s(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    from app.profile.job_screenshot_extract import UnsupportedScreenshotType

    def _raise(image_bytes, mime_type, account_id=None):
        raise UnsupportedScreenshotType("can't read this")

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_image", _raise)

    client = _client()
    resp = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
    )
    assert resp.status_code == 422


def test_create_from_authenticated_url_maps_login_failure_to_502(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    from app.ingest.jobs.auth_fetch import AuthLoginFailedError

    def _raise(source_id, url):
        raise AuthLoginFailedError("bad credentials")

    monkeypatch.setattr("app.api.job_postings.fetch_job_url_authenticated", _raise)

    client = _client()
    resp = client.post(
        "/api/job-postings/from-authenticated-url",
        json={"account_id": account_id, "url": "https://site.example/job", "auth_source_id": 1},
    )
    assert resp.status_code == 502


def test_create_from_authenticated_url_success(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    monkeypatch.setattr(
        "app.api.job_postings.fetch_job_url_authenticated",
        lambda source_id, url: ("Role at Site", "Full posting text behind the login wall."),
    )

    client = _client()
    resp = client.post(
        "/api/job-postings/from-authenticated-url",
        json={"account_id": account_id, "url": "https://site.example/job", "auth_source_id": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "authenticated"
    assert body["title"] == "Role at Site"
