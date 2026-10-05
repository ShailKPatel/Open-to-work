from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import get_db, init_db
from app.core.settings import get_settings

# Uploads are checked by their bytes, so test images carry a real PNG
# signature in front of a tag that tells them apart.
_PNG = b"\x89PNG\r\n\x1a\n"


def _png(tag: bytes) -> bytes:
    return _PNG + tag


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    # Job posting creation now also resolves a role family (embeds the
    # title, searches Qdrant) and indexes the posting itself. Without
    # pointing this at the in-memory test collection, these tests would
    # make real network calls to whatever QDRANT_URL is configured in .env.
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    os.environ["JOB_SCREENSHOT_STORAGE_DIR"] = str(tmp_path / "job_screenshots")
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


def _make_second_account() -> int:
    from app.core.db import Account

    db = get_db()
    account = Account(first_name="Grace", last_name="Hopper", github_username="ghopper")
    db.add(account)
    db.commit()
    account_id = account.id
    db.close()
    return account_id


def test_identical_text_from_another_account_gets_its_own_posting(tmp_path):
    _reset_db(tmp_path)
    account_a = _make_account()
    account_b = _make_second_account()
    client = _client()
    text = "Identical posting text, saved by two profiles."

    a = client.post("/api/job-postings", json={"account_id": account_a, "raw_text": text}).json()
    b = client.post("/api/job-postings", json={"account_id": account_b, "raw_text": text}).json()

    assert a["id"] != b["id"]
    assert [r["id"] for r in client.get(f"/api/job-postings?account_id={account_b}").json()] == [
        b["id"]
    ]
    assert [r["id"] for r in client.get(f"/api/job-postings?account_id={account_a}").json()] == [
        a["id"]
    ]


def test_identical_screenshot_from_another_account_gets_its_own_posting(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_a = _make_account()
    account_b = _make_second_account()
    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: _fake_screenshot_result(),
    )
    client = _client()

    def upload(account_id: int) -> dict:
        return client.post(
            "/api/job-postings/from-screenshot",
            data={"account_id": str(account_id)},
            files={"file": ("shot.png", _png(b"fake-bytes"), "image/png")},
        ).json()

    first_a, second_a, b = upload(account_a), upload(account_a), upload(account_b)

    assert first_a["id"] == second_a["id"]
    assert b["id"] != first_a["id"]


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


def test_delete_posting_unlinks_resumes_built_for_it(tmp_path):
    from app.core.db import Resume

    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Delete me too"}
    ).json()

    db = get_db()
    resume = Resume(
        account_id=account_id,
        filename="tailored.pdf",
        mime_type="application/pdf",
        job_posting_id=created["id"],
    )
    db.add(resume)
    db.commit()
    resume_id = resume.id
    db.close()

    assert client.delete(f"/api/job-postings/{created['id']}").status_code == 200

    db = get_db()
    kept = db.get(Resume, resume_id)
    assert kept is not None
    assert kept.job_posting_id is None
    db.close()


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
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: fake_result,
    )

    client = _client()
    resp = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", _png(b"fake-bytes"), "image/png")},
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

    client = _client()
    resp = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
    )
    assert resp.status_code == 422


def test_extraction_stores_annual_salary_bounds(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(
        monkeypatch, salary_range="1.5 to 1.6 lakh per month", work_mode="Remote"
    )
    client = _client()

    body = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Hiring."}
    ).json()

    assert body["salary_min_annual"] == 1_800_000
    assert body["salary_max_annual"] == 1_920_000
    assert body["salary_currency"] == "INR"
    assert body["work_mode"] == "Remote"


def test_patch_salary_text_reparses_and_numbers_override(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)
    client = _client()
    posting_id = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Hiring."}
    ).json()["id"]

    body = client.patch(
        f"/api/job-postings/{posting_id}",
        json={"salary_range": "12-18 LPA", "employment_type": "Internship"},
    ).json()
    assert (body["salary_min_annual"], body["salary_max_annual"]) == (1_200_000, 1_800_000)
    assert body["salary_range"] == "12-18 LPA"
    assert body["employment_type"] == "Internship"
    # untouched extracted fields survive the edit
    assert body["seniority"] == "Senior"

    body = client.patch(
        f"/api/job-postings/{posting_id}", json={"salary_min_annual": 1_500_000}
    ).json()
    assert (body["salary_min_annual"], body["salary_max_annual"]) == (1_500_000, 1_800_000)


def test_list_filters_by_min_salary(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    ids = {}
    for label, salary in [("low", "5 LPA"), ("high", "20-30 LPA"), ("none", "")]:
        _fake_extraction(monkeypatch, salary_range=salary)
        ids[label] = client.post(
            "/api/job-postings", json={"account_id": account_id, "raw_text": f"Job {label}"}
        ).json()["id"]

    rows = client.get(
        f"/api/job-postings?account_id={account_id}&min_salary=2500000"
    ).json()

    assert [r["id"] for r in rows] == [ids["high"]]


def _fake_screenshot_result(transcription="Backend Engineer at Acme. Python required."):
    from app.profile.job_extract import JobExtraction, RequiredSkill
    from app.profile.job_screenshot_extract import ScreenshotExtraction

    return ScreenshotExtraction(
        raw_text_transcribed=transcription,
        extraction=JobExtraction(
            company="Acme", title="Backend Engineer", location="Remote",
            salary_range="", employment_type="", seniority="", experience_required="",
            skills_required=[RequiredSkill(skill="Python", level="")],
            other_requirements=[], role_summary="",
        ),
    )


def test_compose_text_only_is_a_paste(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id), "text": "Backend engineer, Python, Postgres."},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "pasted"
    assert body["raw_text"] == "Backend engineer, Python, Postgres."
    assert body["warnings"] == []


def test_compose_link_only_asks_for_the_posting_text(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id), "text": "https://acme.example/jobs/1"},
    )

    assert resp.status_code == 422
    assert "Only a link was given" in resp.json()["detail"]


def test_compose_keeps_link_from_text_as_apply_url(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch)

    resp = _client().post(
        "/api/job-postings/compose",
        data={
            "account_id": str(account_id),
            "text": "Backend engineer, Python. Apply at https://acme.example/jobs/1.",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["apply_url"] == "https://acme.example/jobs/1"
    assert body["source"] == "pasted"
    assert body["warnings"] == []


def test_compose_screenshots_with_text_and_link(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    seen = {}

    def _read(images, context_text="", account_id=None):
        seen["count"] = len(images)
        seen["context"] = context_text
        return _fake_screenshot_result()

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_images", _read)

    resp = _client().post(
        "/api/job-postings/compose",
        data={
            "account_id": str(account_id),
            "text": "Referred by a friend, salary 20 LPA. https://acme.example/jobs/9",
        },
        files=[
            ("files", ("one.png", _png(b"1"), "image/png")),
            ("files", ("two.png", _png(b"2"), "image/png")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert seen["count"] == 2
    assert "salary 20 LPA" in seen["context"]
    assert body["source"] == "mixed"
    assert body["company"] == "Acme"
    assert body["apply_url"] == "https://acme.example/jobs/9"
    assert "Python required." in body["raw_text"]
    # The pasted text is kept apart from the transcription joined onto it.
    assert body["source_text"] == "Referred by a friend, salary 20 LPA. https://acme.example/jobs/9"
    assert body["source_links"] == ["https://acme.example/jobs/9"]
    assert body["warnings"] == []
    assert Path(body["screenshot_path"]).exists()
    assert body["image_count"] == 2

    client = _client()
    first = client.get(f"/api/job-postings/{body['id']}/images/0")
    second = client.get(f"/api/job-postings/{body['id']}/images/1")
    assert first.status_code == 200 and first.content == _png(b"1")
    assert second.status_code == 200 and second.content == _png(b"2")
    assert client.get(f"/api/job-postings/{body['id']}/images/2").status_code == 404


def test_compose_keeps_images_the_model_could_not_read(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    from app.profile.job_screenshot_extract import ScreenshotExtractionError

    def _fail(images, context_text="", account_id=None):
        raise ScreenshotExtractionError("model unavailable")

    _fake_extraction(monkeypatch)

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_images", _fail)

    resp = _client().post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id), "text": "Backend engineer, Python."},
        files=[("files", ("one.png", _png(b"1"), "image/png"))],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "pasted"
    assert body["image_count"] == 1
    assert body["extraction_status"] == "extracted"
    assert "model unavailable" in body["extraction_error"]
    assert body["warnings"] == []


def test_delete_posting_removes_its_images(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: _fake_screenshot_result(),
    )
    client = _client()
    body = client.post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id)},
        files=[("files", ("one.png", _png(b"1"), "image/png"))],
    ).json()
    stored = Path(body["screenshot_path"])
    assert stored.exists()

    assert client.delete(f"/api/job-postings/{body['id']}").status_code == 200
    assert not stored.exists()


def test_compose_rejects_non_image_attachment(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id), "text": "Some posting"},
        files=[("files", ("doc.pdf", b"%PDF", "application/pdf"))],
    )

    assert resp.status_code == 422


def test_detail_context_text_lists_extracted_fields_for_an_llm(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_extraction(monkeypatch, other_requirements=["Work permit"])
    client = _client()

    created = client.post(
        "/api/job-postings",
        json={"account_id": account_id, "raw_text": "We are hiring a backend engineer."},
    ).json()
    client.patch(f"/api/job-postings/{created['id']}", json={"applied": True})
    context = client.get(f"/api/job-postings/{created['id']}").json()["context_text"]

    assert context.startswith("# Job posting: Backend Engineer at Acme\n")
    assert "- Company: Acme\n" in context
    assert "- Seniority: Senior\n" in context
    assert "- Salary: 120,000 to 150,000 USD per year (as posted: $120k-$150k)\n" in context
    assert "- Application status: Applied on " in context
    assert "## Role summary\n\nOwn the backend.\n" in context
    assert "## Required skills\n\n- Python (senior)\n" in context
    assert "## Other requirements\n\n- Work permit\n" in context
    # Nothing stated, so no empty line for it.
    assert "Work mode" not in context


def test_pasted_posting_keeps_its_source_text_and_links(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    body = _client().post(
        "/api/job-postings",
        json={
            "account_id": account_id,
            "raw_text": "  Backend role, see https://acme.example/j/1  ",
        },
    ).json()

    assert body["source_text"] == "Backend role, see https://acme.example/j/1"
    assert body["source_links"] == ["https://acme.example/j/1"]


def test_long_pasted_text_asks_before_it_is_processed(tmp_path, monkeypatch):
    from app.api import input_limits

    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(input_limits, "SOFT_MAX_TEXT_CHARS", 20)
    calls = []
    monkeypatch.setattr(
        "app.api.job_postings._run_extraction", lambda posting, db, **_: calls.append(1)
    )
    client = _client()
    body = {"account_id": account_id, "raw_text": "A very long job posting text."}

    asked = client.post("/api/job-postings", json=body)
    confirmed = client.post("/api/job-postings", json={**body, "confirm_large": True})

    assert asked.status_code == 409
    assert asked.json()["detail"]["code"] == "large_input"
    assert "29 characters" in asked.json()["detail"]["message"]
    assert confirmed.status_code == 200
    assert len(calls) == 1
    assert len(client.get(f"/api/job-postings?account_id={account_id}").json()) == 1


def test_large_screenshot_asks_before_the_llm_reads_it(tmp_path, monkeypatch):
    from app.api import input_limits

    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(input_limits, "SOFT_MAX_FILE_MB", 0.001)

    def _unexpected(*args, **kwargs):
        raise AssertionError("the screenshot was sent before the person agreed")

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_images", _unexpected)
    resp = _client().post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", b"x" * 4096, "image/png")},
    )

    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "large_input"


def test_compose_asks_for_large_text_or_screenshots(tmp_path, monkeypatch):
    from app.api import input_limits

    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(input_limits, "SOFT_MAX_TEXT_CHARS", 20)
    monkeypatch.setattr(
        "app.api.job_postings._run_extraction", lambda posting, db, **_: None
    )
    client = _client()
    form = {"account_id": str(account_id), "text": "A very long job posting text."}

    asked = client.post("/api/job-postings/compose", data=form)
    confirmed = client.post(
        "/api/job-postings/compose", data={**form, "confirm_large": "true"}
    )

    assert asked.status_code == 409
    assert asked.json()["detail"]["message"].startswith("This job posting is 29 characters")
    assert confirmed.status_code == 200


def _blocking_extraction(monkeypatch):
    """A fake extraction that waits until the test lets it finish, and
    counts how often it was called."""
    import threading
    from unittest.mock import MagicMock

    release = threading.Event()
    calls: list[int] = []
    response = MagicMock()
    response.parsed = {
        "company": "Acme", "title": "Backend Engineer", "location": "", "salary_range": "",
        "employment_type": "", "seniority": "", "experience_required": "",
        "skills_required": [], "other_requirements": [], "role_summary": "",
    }

    def _complete(*args, **kwargs):
        calls.append(1)
        assert release.wait(10)
        return response

    monkeypatch.setattr("app.profile.job_extract.complete", _complete)
    return release, calls


def _background_reads(monkeypatch):
    import app.api.job_postings as job_postings

    monkeypatch.setattr(job_postings, "_start_extraction", _real_start_extraction)


def _wait_for_read(client, posting_id: int) -> dict:
    """Until the worker thread is gone, not just the status: it still
    resolves the role family and indexes after the status is set."""
    import time

    from app.api.job_postings import _reading

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        body = client.get(f"/api/job-postings/{posting_id}").json()
        if body["extraction_status"] != "pending" and not _reading(posting_id):
            return body
        time.sleep(0.05)
    raise AssertionError("the background read never finished")


from app.api.job_postings import _start_extraction as _real_start_extraction  # noqa: E402


def test_create_posting_returns_before_the_read_finishes(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _background_reads(monkeypatch)
    release, _ = _blocking_extraction(monkeypatch)
    client = _client()

    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Backend role."}
    )

    assert created.status_code == 200
    assert created.json()["extraction_status"] == "pending"
    assert client.get(f"/api/job-postings/{created.json()['id']}").json()[
        "extraction_status"
    ] == "pending"
    release.set()
    done = _wait_for_read(client, created.json()["id"])
    assert done["extraction_status"] == "extracted"
    assert done["company"] == "Acme"


def test_same_posting_sent_twice_while_reading_is_read_once(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _background_reads(monkeypatch)
    release, calls = _blocking_extraction(monkeypatch)
    client = _client()
    body = {"account_id": account_id, "raw_text": "Backend role, sent twice."}

    first = client.post("/api/job-postings", json=body).json()
    second = client.post("/api/job-postings", json=body).json()
    again = client.post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id), "text": "Backend role, sent twice."},
    ).json()

    assert first["id"] == second["id"] == again["id"]
    assert second["extraction_status"] == "pending"
    assert again["warnings"] == ["This posting was already saved, so the saved one was kept."]
    release.set()
    _wait_for_read(client, first["id"])
    assert len(calls) == 1


def test_reprocess_while_reading_does_not_start_a_second_read(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _background_reads(monkeypatch)
    release, calls = _blocking_extraction(monkeypatch)
    client = _client()
    created = client.post(
        "/api/job-postings", json={"account_id": account_id, "raw_text": "Backend role."}
    ).json()

    resp = client.post(f"/api/job-postings/{created['id']}/reprocess")

    assert resp.status_code == 200
    assert resp.json()["extraction_status"] == "pending"
    release.set()
    _wait_for_read(client, created["id"])
    assert len(calls) == 1

    release.clear()
    again = client.post(f"/api/job-postings/{created['id']}/reprocess").json()
    assert again["extraction_status"] == "pending"
    release.set()
    assert _wait_for_read(client, created["id"])["extraction_status"] == "extracted"
    assert len(calls) == 2


def test_screenshot_is_read_in_the_background(tmp_path, monkeypatch):
    import threading

    _reset_db(tmp_path)
    account_id = _make_account()
    _background_reads(monkeypatch)
    release = threading.Event()

    def _read(images, context_text="", account_id=None):
        assert release.wait(10)
        return _fake_screenshot_result()

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_images", _read)
    client = _client()

    created = client.post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", _png(b"shot"), "image/png")},
    ).json()

    assert created["extraction_status"] == "pending"
    assert created["raw_text"] == ""
    assert created["image_count"] == 1
    release.set()
    done = _wait_for_read(client, created["id"])
    assert done["extraction_status"] == "extracted"
    assert done["company"] == "Acme"
    assert done["raw_text"] == "Backend Engineer at Acme. Python required."


def test_unreadable_screenshot_alone_fails_and_reprocess_reads_it_again(tmp_path, monkeypatch):
    from app.profile.job_screenshot_extract import ScreenshotExtractionError

    _reset_db(tmp_path)
    account_id = _make_account()
    reads: list[int] = []

    def _fail(images, context_text="", account_id=None):
        reads.append(len(images))
        raise ScreenshotExtractionError("could not read any job posting text")

    monkeypatch.setattr("app.api.job_postings.extract_job_posting_from_images", _fail)
    client = _client()
    created = client.post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id)},
        files=[("files", ("one.png", _png(b"1"), "image/png"))],
    ).json()
    assert created["extraction_status"] == "failed"
    assert "could not read" in created["extraction_error"]

    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: _fake_screenshot_result(),
    )
    done = client.post(f"/api/job-postings/{created['id']}/reprocess").json()

    assert reads == [1]
    assert done["extraction_status"] == "extracted"
    assert done["source"] == "screenshot"
    assert "Python required." in done["raw_text"]


def test_posting_left_reading_by_a_restart_is_marked_failed(tmp_path):
    from fastapi.testclient import TestClient

    from app.api.job_postings import fail_interrupted_extractions
    from app.api.main import app
    from app.core.db import JobPosting

    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    for i, status in enumerate(("pending", "extracted", "pending")):
        db.add(
            JobPosting(
                account_id=account_id, source="pasted", external_id=str(i), company="Acme",
                title="Engineer", raw_text_quarantined=f"text {i}", content_hash=str(i),
                extraction_status=status,
            )
        )
    db.commit()
    db.close()

    assert fail_interrupted_extractions() == 2
    # The app's startup runs the same thing.
    db = get_db()
    db.add(
        JobPosting(
            account_id=account_id, source="pasted", external_id="3", company="Acme",
            title="Engineer", raw_text_quarantined="text 3", content_hash="3",
            extraction_status="pending",
        )
    )
    db.commit()
    db.close()
    with TestClient(app) as client:
        rows = client.get(f"/api/job-postings?account_id={account_id}").json()

    statuses = sorted(r["extraction_status"] for r in rows)
    assert statuses == ["extracted", "failed", "failed", "failed"]
    failed = [r for r in rows if r["extraction_status"] == "failed"]
    assert all("restarted" in r["extraction_error"] for r in failed)
    assert all("Reprocess" in r["extraction_error"] for r in failed)


def test_concurrent_save_of_the_same_text_returns_the_saved_posting(tmp_path, monkeypatch):
    import app.api.job_postings as job_postings

    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    body = {"account_id": account_id, "raw_text": "Backend role, raced."}
    first = client.post("/api/job-postings", json=body).json()

    # The second request looked before the first one committed.
    real_find = job_postings._find_posting
    looks: list[int] = []

    def _find_late(db, account_id, content_hash):
        looks.append(1)
        return None if len(looks) == 1 else real_find(db, account_id, content_hash)

    monkeypatch.setattr(job_postings, "_find_posting", _find_late)
    second = client.post("/api/job-postings", json=body)

    assert second.status_code == 200
    assert second.json()["id"] == first["id"]


def test_posting_image_is_served_by_its_bytes_not_its_name(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(
        "app.api.job_postings.extract_job_posting_from_images",
        lambda images, context_text="", account_id=None: _fake_screenshot_result(),
    )
    client = _client()
    created = client.post(
        "/api/job-postings/compose",
        data={"account_id": str(account_id)},
        files=[("files", ("page.html", _png(b"<script>alert(1)</script>"), "text/html"))],
    ).json()

    resp = client.get(f"/api/job-postings/{created['id']}/images/0")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"


def test_screenshot_whose_bytes_are_not_an_image_is_refused(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    resp = _client().post(
        "/api/job-postings/from-screenshot",
        data={"account_id": str(account_id)},
        files={"file": ("shot.png", b"<html>not a png</html>", "image/png")},
    )

    assert resp.status_code == 422
    assert "not a PNG" in resp.json()["detail"]
