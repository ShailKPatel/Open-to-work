import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import Account, get_db, init_db
from app.core.settings import get_settings


@pytest.fixture(autouse=True)
def _fake_embed(monkeypatch):
    """Any successful extraction or manual edit re-indexes into Qdrant
    (app/retrieval/index.py's index_resume), which otherwise needs a real
    embedding model loaded, same fake-embed pattern as test_index.py.
    Autouse: nearly every test in this file ends up going through PATCH or
    a successful upload at some point."""
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["RESUME_STORAGE_DIR"] = str(tmp_path / "resumes")
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


def _fake_extraction(
    monkeypatch,
    tags=None,
    target_roles=None,
    summary="Good generalist.",
    experiences=None,
    education=None,
):
    """Mocks the LLM call underneath extract_resume, same pattern
    test_extract.py uses against app.profile.extract.complete: a resume
    upload's extraction runs synchronously inside POST /api/resume, so
    without this every upload test would otherwise depend on a real
    (missing, in tests) API key and just fail extraction silently."""
    fake_response = MagicMock()
    fake_response.parsed = {
        "tags": tags if tags is not None else ["Python", "FastAPI"],
        "target_roles": target_roles if target_roles is not None else ["Backend Engineer"],
        "summary": summary,
        "experiences": experiences if experiences is not None else [],
        "education": education if education is not None else [],
    }
    monkeypatch.setattr(
        "app.profile.resume_extract.complete", MagicMock(return_value=fake_response)
    )


def test_upload_pdf_extracts_tags_and_summary(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()

    resp = client.post(
        "/api/resume",
        data={
            "account_id": account_id,
            "name": "Backend, 2026 batch",
            "notes": "for backend roles",
        },
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "resume.pdf"
    assert body["name"] == "Backend, 2026 batch"
    assert body["notes"] == "for backend roles"
    assert body["tags"] == ["Python", "FastAPI"]
    assert body["target_roles"] == ["Backend Engineer"]
    assert body["summary"] == "Good generalist."
    assert body["extraction_status"] == "extracted"
    assert body["extraction_error"] is None

    stored = list((tmp_path / "resumes" / str(account_id)).iterdir())
    assert len(stored) == 1
    assert stored[0].name.endswith("_resume.pdf")


def test_upload_merges_tags_and_experience_into_profile(tmp_path, monkeypatch):
    """End to end: an upload's extracted tags and work history land as
    real Skill/Experience rows too, not just on the Resume row itself
    (see app/profile/resume_profile_merge.py)."""

    from app.core.db import Experience, Skill, get_db

    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(
        monkeypatch,
        tags=["Python", "FastAPI"],
        experiences=[
            {
                "company": "Acme Corp",
                "title": "Backend Engineer",
                "start_date": "2021-03-01",
                "end_date": "",
            }
        ],
    )
    client = _client()

    client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )

    db = get_db()
    skill_rows = db.execute(select(Skill).where(Skill.account_id == account_id)).scalars()
    skills = {s.name for s in skill_rows}
    experiences = db.execute(
        select(Experience).where(Experience.account_id == account_id)
    ).scalars().all()
    db.close()

    assert skills == {"Python", "FastAPI"}
    assert len(experiences) == 1
    assert experiences[0].company == "Acme Corp"
    assert experiences[0].title == "Backend Engineer"
    assert experiences[0].start_date == "mar 2021"
    assert experiences[0].end_date is None


def test_upload_shows_and_merges_experience_and_education(tmp_path, monkeypatch):
    """The resume row itself carries what this file said about work
    history and schooling, and the education lands in the Education
    table the same way experience lands in Experience."""

    from app.core.db import Education, get_db

    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(
        monkeypatch,
        experiences=[
            {
                "company": "Acme Corp",
                "title": "Backend Engineer",
                "start_date": "2021-03-01",
                "end_date": "",
                "points": ["Built the billing API"],
            }
        ],
        education=[
            {
                "institution": "Nirma University",
                "degree": "B.Tech in Computer Science",
                "location": "Springfield",
                "start_date": "2022-08-01",
                "end_date": "2026-05-01",
            }
        ],
    )
    client = _client()

    resp = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )

    body = resp.json()
    assert body["experiences"] == [
        {
            "company": "Acme Corp",
            "title": "Backend Engineer",
            "location": None,
            "start_date": "mar 2021",
            "end_date": None,
            "points": ["Built the billing API"],
        }
    ]
    assert body["education"] == [
        {
            "institution": "Nirma University",
            "degree": "B.Tech in Computer Science",
            "location": "Springfield",
            "start_date": "aug 2022",
            "end_date": "may 2026",
            "grade": None,
            "details": [],
        }
    ]

    db = get_db()
    rows = db.execute(select(Education).where(Education.account_id == account_id)).scalars().all()
    db.close()
    assert len(rows) == 1
    assert rows[0].institution == "Nirma University"
    assert rows[0].end_date == "may 2026"


def test_upload_unsupported_type_stores_file_without_tags(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    resp = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.docx", io.BytesIO(b"not a real docx"), "application/msword")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction_status"] == "unsupported_type"
    assert body["tags"] == []
    assert body["extraction_error"]
    assert body["name"] is None  # no name given, stays unset, UI falls back to filename


def test_patch_resume_renames_it(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    resume_id = upload.json()["id"]
    assert upload.json()["name"] is None

    resp = client.patch(f"/api/resume/{resume_id}", json={"name": "For Acme"})

    assert resp.status_code == 200
    assert resp.json()["name"] == "For Acme"
    # filename is untouched by a rename, name and filename are separate fields
    assert resp.json()["filename"] == "resume.pdf"

    listed = client.get(f"/api/resume?account_id={account_id}").json()
    assert listed[0]["name"] == "For Acme"


def test_upload_unknown_account_404s(tmp_path):
    _reset_db(tmp_path)
    client = _client()

    resp = client.post(
        "/api/resume",
        data={"account_id": 999},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert resp.status_code == 404


def test_list_resumes_scoped_to_account(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_a = _make_account(github_username="a")
    account_b = _make_account(github_username="b")
    _fake_extraction(monkeypatch)
    client = _client()

    client.post(
        "/api/resume",
        data={"account_id": account_a},
        files={"file": ("a.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    client.post(
        "/api/resume",
        data={"account_id": account_b},
        files={"file": ("b.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    listed_a = client.get(f"/api/resume?account_id={account_a}").json()
    assert len(listed_a) == 1
    assert listed_a[0]["filename"] == "a.pdf"


def test_patch_resume_updates_tags_roles_summary_notes(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.docx", io.BytesIO(b"not real"), "application/msword")},
    )
    resume_id = upload.json()["id"]

    resp = client.patch(
        f"/api/resume/{resume_id}",
        json={
            "tags": ["Python", " Go "],
            "target_roles": ["Backend Engineer"],
            "summary": "Hand-written summary.",
            "notes": "v2, tailored for Acme",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["tags"] == ["Python", "Go"]  # whitespace stripped
    assert body["target_roles"] == ["Backend Engineer"]
    assert body["summary"] == "Hand-written summary."
    assert body["notes"] == "v2, tailored for Acme"


def test_patch_resume_partial_update_leaves_other_fields(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.docx", io.BytesIO(b"not real"), "application/msword")},
    )
    resume_id = upload.json()["id"]
    client.patch(f"/api/resume/{resume_id}", json={"summary": "first summary"})

    resp = client.patch(f"/api/resume/{resume_id}", json={"notes": "just a note"})

    assert resp.status_code == 200
    assert resp.json()["summary"] == "first summary"
    assert resp.json()["notes"] == "just a note"


def test_patch_unknown_resume_404s(tmp_path):
    _reset_db(tmp_path)
    resp = _client().patch("/api/resume/999", json={"notes": "x"})
    assert resp.status_code == 404


def test_reprocess_resume_reruns_extraction(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    resume_id = upload.json()["id"]
    assert upload.json()["extraction_status"] == "failed"  # no API key configured in tests

    _fake_extraction(monkeypatch, tags=["Rust"], summary="Reprocessed summary.")
    resp = client.post(f"/api/resume/{resume_id}/reprocess")

    assert resp.status_code == 200
    body = resp.json()
    assert body["extraction_status"] == "extracted"
    assert body["tags"] == ["Rust"]
    assert body["summary"] == "Reprocessed summary."
    assert body["is_generated"] is False


def _make_generated_resume(tmp_path, account_id: int) -> tuple[int, int]:
    """A resume as the build flow saves it: compiled PDF only, no upload."""
    from app.core.db import JobPosting, Resume

    compiled = tmp_path / "generated.pdf"
    compiled.write_bytes(b"%PDF-compiled")
    db = get_db()
    posting = JobPosting(
        account_id=account_id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="hiring an engineer", content_hash="h1",
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)
    row = Resume(
        account_id=account_id, filename="Engineer - Acme.pdf", mime_type="application/pdf",
        compiled_path=str(compiled), job_posting_id=posting.id, extraction_status="extracted",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    ids = (row.id, posting.id)
    db.close()
    return ids


def test_generated_resume_is_marked_and_names_its_job(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resume_id, posting_id = _make_generated_resume(tmp_path, account_id)

    listed = _client().get(f"/api/resume?account_id={account_id}").json()

    assert len(listed) == 1
    assert listed[0]["id"] == resume_id
    assert listed[0]["is_generated"] is True
    assert listed[0]["job_posting_id"] == posting_id
    assert listed[0]["job_title"] == "Engineer"
    assert listed[0]["job_company"] == "Acme"


def test_list_resumes_filters_by_job_posting(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()
    client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("mine.pdf", io.BytesIO(b"%PDF-1.4 mine"), "application/pdf")},
    )
    resume_id, posting_id = _make_generated_resume(tmp_path, account_id)

    listed = client.get(
        f"/api/resume?account_id={account_id}&job_posting_id={posting_id}"
    ).json()

    assert [r["id"] for r in listed] == [resume_id]


def test_reprocess_refuses_a_generated_resume(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    resume_id, _ = _make_generated_resume(tmp_path, account_id)

    resp = _client().post(f"/api/resume/{resume_id}/reprocess")

    assert resp.status_code == 409
    assert "built here" in resp.json()["detail"]


def test_download_resume_file_roundtrips_bytes(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake bytes"), "application/pdf")},
    )
    resume_id = upload.json()["id"]

    resp = client.get(f"/api/resume/{resume_id}/file")

    assert resp.status_code == 200
    assert resp.content == b"%PDF-1.4 fake bytes"
    assert "attachment" in resp.headers["content-disposition"]


def test_download_resume_file_inline_for_preview(tmp_path):
    """The View popup requests ?disposition=inline so a PDF/image renders
    straight in an iframe/img instead of the browser forcing a save
    dialog."""
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake bytes"), "application/pdf")},
    )
    resume_id = upload.json()["id"]

    resp = client.get(f"/api/resume/{resume_id}/file?disposition=inline")

    assert resp.status_code == 200
    assert "inline" in resp.headers["content-disposition"]


def test_download_resume_file_rejects_bad_disposition(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    resume_id = upload.json()["id"]

    resp = client.get(f"/api/resume/{resume_id}/file?disposition=nonsense")

    assert resp.status_code == 422


def test_resume_profile_lists_what_it_added_and_follows_deletes(tmp_path, monkeypatch):
    from app.core.db import ResumeProfileLink

    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(
        monkeypatch,
        tags=["Python", "Rust"],
        experiences=[
            {
                "company": "Acme Corp",
                "title": "Backend Engineer",
                "start_date": "2021-03-01",
                "end_date": "",
                "points": ["Built the billing API"],
                "skills": ["Python"],
            }
        ],
        education=[
            {
                "institution": "Nirma University",
                "degree": "B.Tech",
                "location": "",
                "start_date": "",
                "end_date": "",
            }
        ],
    )
    client = _client()
    resume_id = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    ).json()["id"]

    profile = client.get(f"/api/resume/{resume_id}/profile").json()

    [exp] = profile["experiences"]
    assert (exp["company"], exp["title"], exp["created"]) == ("Acme Corp", "Backend Engineer", True)
    assert [p["text"] for p in exp["points"]] == ["Built the billing API"]
    assert [s["skill"] for s in exp["skills"]] == ["Python"]
    assert [e["institution"] for e in profile["education"]] == ["Nirma University"]
    assert [s["name"] for s in profile["skills"]] == ["Rust"]

    client.delete(f"/api/experience/{exp['id']}")
    profile = client.get(f"/api/resume/{resume_id}/profile").json()
    assert profile["experiences"] == []

    client.delete(f"/api/resume/{resume_id}")
    db = get_db()
    assert db.execute(select(ResumeProfileLink)).scalars().all() == []
    db.close()


def test_delete_resume_removes_row_and_file(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    resume_id = upload.json()["id"]

    resp = client.delete(f"/api/resume/{resume_id}")

    assert resp.status_code == 200
    assert client.get(f"/api/resume?account_id={account_id}").json() == []
    assert list((tmp_path / "resumes" / str(account_id)).iterdir()) == []


def test_skill_from_resume_names_its_resume_and_follows_a_rename(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch, tags=["Rust"])
    client = _client()
    client.post("/api/skills", json={"account_id": account_id, "name": "Leadership"})

    upload = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    resume_id = upload.json()["id"]

    def by_name():
        resp = client.get(f"/api/skills?account_id={account_id}")
        return {g["name"]: g for g in resp.json()}

    skills = by_name()
    assert skills["Rust"]["source_resume"] == {"id": resume_id, "name": "resume.pdf"}
    assert skills["Leadership"]["source_resume"] is None

    client.patch(f"/api/resume/{resume_id}", json={"name": "Backend 2026"})
    assert by_name()["Rust"]["source_resume"]["name"] == "Backend 2026"

    client.delete(f"/api/resume/{resume_id}")
    skills = by_name()
    assert "Rust" in skills
    assert skills["Rust"]["source_resume"] is None


def test_delete_account_removes_its_resumes(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()

    client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    resp = client.delete(f"/accounts/{account_id}")

    assert resp.status_code == 200
    assert not (tmp_path / "resumes" / str(account_id)).exists()


def test_signup_resume_upload_also_creates_resume_row(tmp_path, monkeypatch):
    """POST /accounts's optional resume field goes through the same
    app.profile.resume_ingest.ingest_resume as POST /api/resume, so it
    should show up in GET /api/resume too, not just on the account's
    legacy resume_path mirror."""
    _reset_db(tmp_path)
    _fake_extraction(monkeypatch)
    client = _client()

    resp = client.post(
        "/accounts",
        data={"first_name": "Grace", "last_name": "Hopper", "github_username": "ghopper"},
        files={"resume": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )
    account_id = resp.json()["id"]

    listed = client.get(f"/api/resume?account_id={account_id}").json()
    assert len(listed) == 1
    assert listed[0]["filename"] == "resume.pdf"
    assert listed[0]["extraction_status"] == "extracted"


# --- POST /api/resume/{id}/edit ---------------------------------------

def _fake_fit_result():
    from app.resume_build.pagefit import FitResult

    return FitResult(tex="x", pdf_bytes=b"%PDF-edited", page_count=1, cuts_made=0)


def test_edit_adopts_plain_upload_then_applies_edit(tmp_path, monkeypatch):
    """A plain upload (content_json still null) gets seeded via
    build_resume_data_from_seed() before the edit is applied, see
    app/api/resume.py's edit_resume() docstring."""
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch, tags=["Python"], summary="Backend generalist.")
    client = _client()

    uploaded = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    ).json()
    assert uploaded["job_posting_id"] is None
    assert uploaded["template"] is None

    seed_calls = []
    fake_seeded = {
        "name": "Ada", "summary": "Backend generalist.", "experience": [],
        "projects": [], "education": [], "technologies": [], "skills": ["Python"],
    }
    monkeypatch.setattr(
        "app.api.resume.build_resume_data_from_seed",
        lambda account_id, seed_text, template: (seed_calls.append(seed_text), fake_seeded)[1],
    )
    edit_calls = []
    fake_edited = {**fake_seeded, "summary": "Now emphasizes Python and APIs."}
    monkeypatch.setattr(
        "app.api.resume.edit_resume_content",
        lambda account_id, job_text, current, message, template: (
            edit_calls.append((current, message)),
            fake_edited,
        )[1],
    )
    monkeypatch.setattr(
        "app.api.resume.fit_to_page_limit", lambda *a, **k: _fake_fit_result()
    )

    resp = client.post(
        f"/api/resume/{uploaded['id']}/edit",
        json={"message": "Emphasize Python and APIs"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["template"] == "onepage"
    assert body["has_ai_edited_version"] is True
    assert body["has_original_file"] is True  # original upload untouched
    assert body["summary"] == "Now emphasizes Python and APIs."
    assert len(seed_calls) == 1  # adopted exactly once
    assert edit_calls[0] == (fake_seeded, "Emphasize Python and APIs")

    compiled = list((tmp_path / "resumes" / str(account_id)).glob("*_edited.pdf"))
    assert len(compiled) == 1
    assert compiled[0].read_bytes() == b"%PDF-edited"


def test_edit_reuses_existing_content_json_without_reseeding(tmp_path, monkeypatch):
    from app.core.db import Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    row = Resume(
        account_id=account_id, filename="generated.pdf", mime_type="application/pdf",
        template="twopage", content_json={"summary": "Old summary.", "projects": [], "skills": []},
        summary="Old summary.",
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    resume_id = row.id
    db.close()

    seed_mock = MagicMock()
    monkeypatch.setattr("app.api.resume.build_resume_data_from_seed", seed_mock)
    monkeypatch.setattr(
        "app.api.resume.edit_resume_content",
        lambda *a, **k: {"summary": "New summary.", "projects": [], "skills": []},
    )
    monkeypatch.setattr(
        "app.api.resume.fit_to_page_limit", lambda *a, **k: _fake_fit_result()
    )

    resp = _client().post(f"/api/resume/{resume_id}/edit", json={"message": "Update it"})

    assert resp.status_code == 200
    assert resp.json()["summary"] == "New summary."
    assert resp.json()["template"] == "twopage"  # existing template preserved
    seed_mock.assert_not_called()  # already had content_json, never re-seeded


def test_edit_rejects_blank_message(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post("/api/resume/1/edit", json={"message": "   "})
    assert resp.status_code == 422


def test_edit_404_unknown_resume(tmp_path):
    _reset_db(tmp_path)
    resp = _client().post("/api/resume/999999/edit", json={"message": "hi"})
    assert resp.status_code == 404


def test_edit_409_when_never_analyzed(tmp_path, monkeypatch):
    """A resume with no extracted summary/tags and no content_json has no
    seed text to build from: a clear error, not a call to the LLM with
    an empty query."""
    from app.core.db import Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    row = Resume(account_id=account_id, filename="blank.pdf", mime_type="application/pdf")
    db.add(row)
    db.commit()
    db.refresh(row)
    resume_id = row.id
    db.close()

    seed_mock = MagicMock()
    monkeypatch.setattr("app.api.resume.build_resume_data_from_seed", seed_mock)

    resp = _client().post(f"/api/resume/{resume_id}/edit", json={"message": "Update it"})

    assert resp.status_code == 409
    seed_mock.assert_not_called()


# --- GET /api/resume/search ---------------------------------------

def test_search_resumes_for_posting(tmp_path, monkeypatch):
    from app.core.db import JobPosting, Resume
    from app.retrieval.search import Hit

    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    posting = JobPosting(
        account_id=account_id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="hiring an engineer", content_hash="h1",
    )
    resume = Resume(account_id=account_id, filename="resume.pdf", mime_type="application/pdf")
    db.add_all([posting, resume])
    db.commit()
    db.refresh(posting)
    db.refresh(resume)
    posting_id, resume_id = posting.id, resume.id
    db.close()

    monkeypatch.setattr(
        "app.retrieval.search.search_resumes",
        lambda query_text, account_id, top_k=3: [Hit(id=resume_id, score=0.87, payload={})],
    )

    resp = _client().get(
        f"/api/resume/search?account_id={account_id}&job_posting_id={posting_id}"
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["resume"]["id"] == resume_id
    assert body[0]["score"] == 0.87


def test_search_ranks_by_share_of_required_skills(tmp_path, monkeypatch):
    """With extracted required skills, every resume gets a match
    percentage and the list is ranked by it, whatever semantic search
    thinks of the two."""
    from app.core.db import JobPosting, Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    posting = JobPosting(
        account_id=account_id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="Python, SQL, Docker", content_hash="h2",
        extracted_json={"skills_required": ["Python", "SQL", "Docker", "Go"]},
        extraction_status="extracted",
    )
    weak = Resume(
        account_id=account_id, filename="weak.pdf", mime_type="application/pdf",
        tags_json=["Python"],
    )
    strong = Resume(
        account_id=account_id, filename="strong.pdf", mime_type="application/pdf",
        tags_json=["python", "sql"], content_json={"skills": ["Docker"]},
    )
    db.add_all([posting, weak, strong])
    db.commit()
    posting_id, strong_id = posting.id, strong.id
    db.close()

    monkeypatch.setattr("app.retrieval.search.search_resumes", lambda *a, **k: [])
    # Every distinct name on its own axis: nothing is "related", so the
    # ranking below is the exact tier alone.
    axes: dict[str, int] = {}

    def _orthogonal(texts):
        return [
            [1.0 if i == axes.setdefault(t.casefold(), len(axes)) else 0.0 for i in range(16)]
            for t in texts
        ]

    monkeypatch.setattr("app.resume_build.skill_match.embed", _orthogonal)

    body = _client().get(
        f"/api/resume/search?account_id={account_id}&job_posting_id={posting_id}"
    ).json()

    assert [h["resume"]["id"] for h in body][0] == strong_id
    assert body[0]["match_pct"] == 75
    assert body[0]["missing"] == ["Go"]
    assert body[1]["match_pct"] == 25


def test_search_404_unknown_posting(tmp_path):
    _reset_db(tmp_path)
    resp = _client().get("/api/resume/search?account_id=1&job_posting_id=999999")
    assert resp.status_code == 404


# --- GET /api/resume/{id}/compiled-file ---------------------------------------

def test_download_compiled_file(tmp_path):
    from app.core.db import Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    compiled = tmp_path / "generated.pdf"
    compiled.write_bytes(b"%PDF-compiled")

    db = get_db()
    row = Resume(
        account_id=account_id, filename="generated.pdf", mime_type="application/pdf",
        compiled_path=str(compiled),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    resume_id = row.id
    db.close()

    resp = _client().get(f"/api/resume/{resume_id}/compiled-file")

    assert resp.status_code == 200
    assert resp.content == b"%PDF-compiled"


def test_download_compiled_file_404_when_missing(tmp_path):
    _reset_db(tmp_path)
    resp = _client().get("/api/resume/999999/compiled-file")
    assert resp.status_code == 404


def test_uploading_the_same_file_again_is_refused(tmp_path, monkeypatch):
    """Byte-identical file, even under another filename: no second row,
    no second extraction. A different file of the same size still goes in."""
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()

    first = client.post(
        "/api/resume",
        data={"account_id": account_id, "name": "Backend"},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 one"), "application/pdf")},
    )
    assert first.status_code == 200

    again = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume (1).pdf", io.BytesIO(b"%PDF-1.4 one"), "application/pdf")},
    )
    assert again.status_code == 409
    assert '"Backend"' in again.json()["detail"]

    other = client.post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 two"), "application/pdf")},
    )
    assert other.status_code == 200
    assert len(client.get(f"/api/resume?account_id={account_id}").json()) == 2


def _pdf(creator: str) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Creator": creator})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_uploading_a_resume_this_app_built_is_refused(tmp_path, monkeypatch):
    """A generated PDF read back in would fold the LLM's job-tailored
    wording into the profile, so it is turned away before anything is
    saved or sent to the model."""
    _reset_db(tmp_path)
    account_id = _make_account()
    extract = MagicMock()
    monkeypatch.setattr("app.profile.resume_extract.complete", extract)

    resp = _client().post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("built.pdf", io.BytesIO(_pdf("Open to Work")), "application/pdf")},
    )

    assert resp.status_code == 422
    assert "built by Open to Work" in resp.json()["detail"]
    extract.assert_not_called()
    assert _client().get(f"/api/resume?account_id={account_id}").json() == []
    assert not any((tmp_path / "resumes").rglob("*.pdf"))


def test_pdf_made_by_other_latex_tools_still_uploads(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)

    resp = _client().post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("mine.pdf", io.BytesIO(_pdf("LaTeX")), "application/pdf")},
    )

    assert resp.status_code == 200


def test_uploading_a_download_of_an_older_build_is_refused(tmp_path, monkeypatch):
    """Builds compiled before the Creator stamp are caught by comparing
    the upload with the compiled PDFs already in the library."""
    from app.core.db import Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    extract = MagicMock()
    monkeypatch.setattr("app.profile.resume_extract.complete", extract)
    built = tmp_path / "3_generated.pdf"
    built.write_bytes(b"%PDF-1.4 built before the stamp")
    db = get_db()
    db.add(
        Resume(
            account_id=account_id, filename="Acme.pdf", mime_type="application/pdf",
            compiled_path=str(built),
        )
    )
    db.commit()
    db.close()

    resp = _client().post(
        "/api/resume",
        data={"account_id": account_id},
        files={"file": ("download.pdf", io.BytesIO(built.read_bytes()), "application/pdf")},
    )

    assert resp.status_code == 422
    extract.assert_not_called()


def test_signup_with_a_generated_resume_creates_nothing(tmp_path):
    _reset_db(tmp_path)

    resp = _client().post(
        "/accounts",
        data={"first_name": "Ada", "last_name": "Lovelace", "github_username": ""},
        files={"resume": ("built.pdf", io.BytesIO(_pdf("Open to Work")), "application/pdf")},
    )

    assert resp.status_code == 422
    db = get_db()
    assert db.execute(select(Account)).first() is None
    db.close()


def test_large_resume_upload_asks_and_saves_nothing_until_confirmed(tmp_path, monkeypatch):
    from app.api import input_limits

    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    monkeypatch.setattr(input_limits, "SOFT_MAX_FILE_MB", 0.001)
    client = _client()

    big = b"%PDF-1.4 " + b"x" * 4096

    def _send(**extra):
        return client.post(
            "/api/resume",
            data={"account_id": account_id, **extra},
            files={"file": ("resume.pdf", io.BytesIO(big), "application/pdf")},
        )

    asked = _send()
    assert asked.status_code == 409
    assert asked.json()["detail"]["code"] == "large_input"
    assert not list(tmp_path.rglob("*.pdf"))

    confirmed = _send(confirm_large="true")
    assert confirmed.status_code == 200
    assert confirmed.json()["extraction_status"] == "extracted"
