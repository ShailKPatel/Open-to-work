from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import ExperienceSkillEvidence, JobPosting, RoleFamily, SkillEvidence, init_db
from app.core.settings import get_settings
from app.retrieval.index import (
    COLLECTION,
    JOB_POSTINGS_COLLECTION,
    ROLE_FAMILIES_COLLECTION,
    experience_evidence_point_id,
    index_experience_skill_evidence,
    index_job_posting,
    index_role_family,
    index_skill_evidence,
    job_posting_text,
)


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _evidence(**overrides) -> SkillEvidence:
    defaults = dict(
        id=1,
        skill="React",
        repo_id=1,
        evidence_type="declared_dependency",
        weight=0.7,
        confidence=1.0,
        source_files_json=["package.json"],
    )
    defaults.update(overrides)
    return SkillEvidence(**defaults)


def test_index_empty_list_returns_zero(tmp_path):
    _reset(tmp_path)
    assert index_skill_evidence([]) == 0


def test_index_writes_points_to_qdrant(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )

    rows = [_evidence(id=1, skill="React"), _evidence(id=2, skill="Redux")]
    count = index_skill_evidence(rows)

    assert count == 2
    client = vectorstore_module.get_client()
    stored = client.count(COLLECTION).count
    assert stored == 2


def test_index_payload_carries_skill_metadata(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[0.5, 0.5] for _ in texts])

    index_skill_evidence([_evidence(id=1, skill="Kubernetes", weight=0.42)], account_id=9)

    client = vectorstore_module.get_client()
    point = client.retrieve(COLLECTION, ids=[1])[0]
    assert point.payload["skill"] == "Kubernetes"
    assert point.payload["weight"] == 0.42
    assert point.payload["source_type"] == "repo"
    assert point.payload["account_id"] == 9


def _experience_evidence(**overrides) -> ExperienceSkillEvidence:
    defaults = dict(
        id=1,
        skill="Vector databases",
        experience_id=1,
        evidence_type="manual",
        weight=1.0,
        confidence=1.0,
    )
    defaults.update(overrides)
    return ExperienceSkillEvidence(**defaults)


def test_index_experience_empty_list_returns_zero(tmp_path):
    _reset(tmp_path)
    assert index_experience_skill_evidence([]) == 0


def test_index_experience_writes_points_with_source_type(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )

    count = index_experience_skill_evidence(
        [_experience_evidence(id=1, skill="Vector databases", experience_id=7)],
        account_id=42,
    )

    assert count == 1
    client = vectorstore_module.get_client()
    # id is offset from the raw evidence id; see index.py's
    # _EXPERIENCE_EVIDENCE_ID_OFFSET, which avoids a numeric collision with
    # repo-linked SkillEvidence rows sharing the same collection.
    point = client.retrieve(COLLECTION, ids=[experience_evidence_point_id(1)])[0]
    assert point.payload["skill"] == "Vector databases"
    assert point.payload["experience_id"] == 7
    assert point.payload["source_type"] == "experience"
    assert point.payload["account_id"] == 42


def test_repo_and_experience_evidence_with_same_raw_id_dont_collide(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )

    index_skill_evidence([_evidence(id=1, skill="React")], account_id=1)
    index_experience_skill_evidence(
        [_experience_evidence(id=1, skill="Vector databases", experience_id=7)], account_id=1
    )

    client = vectorstore_module.get_client()
    assert client.count(COLLECTION).count == 2


def test_index_role_family_writes_a_global_point(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    family = RoleFamily(id=5, canonical_name="Machine Learning Engineer")
    index_role_family(family)

    client = vectorstore_module.get_client()
    point = client.retrieve(ROLE_FAMILIES_COLLECTION, ids=[5])[0]
    assert point.payload["canonical_name"] == "Machine Learning Engineer"
    assert "account_id" not in point.payload  # global taxonomy, not per-account


def _extracted_posting(**overrides) -> JobPosting:
    defaults = dict(
        id=1,
        account_id=1,
        source="pasted",
        external_id="hash",
        company="Acme",
        title="Backend Engineer",
        raw_text_quarantined="text",
        content_hash="hash",
        extraction_status="extracted",
        extracted_json={
            "role_summary": "Own the backend.",
            "skills_required": [{"skill": "Python", "level": "senior"}],
        },
    )
    defaults.update(overrides)
    return JobPosting(**defaults)


def test_job_posting_text_includes_title_company_summary_and_skills():
    text = job_posting_text(_extracted_posting())
    assert "Backend Engineer at Acme" in text
    assert "Own the backend." in text
    assert "Python" in text


def test_index_job_posting_skips_unextracted(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    posting = _extracted_posting(extraction_status="pending", extracted_json=None)
    assert index_job_posting(posting, account_id=1) is False


def test_index_job_posting_writes_account_scoped_point(tmp_path, monkeypatch):
    _reset(tmp_path)
    monkeypatch.setattr("app.retrieval.index.embed", lambda texts: [[1.0, 0.0] for _ in texts])

    posting = _extracted_posting(id=3)
    assert index_job_posting(posting, account_id=9) is True

    client = vectorstore_module.get_client()
    point = client.retrieve(JOB_POSTINGS_COLLECTION, ids=[3])[0]
    assert point.payload["account_id"] == 9
    assert point.payload["title"] == "Backend Engineer"
