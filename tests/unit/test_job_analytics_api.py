"""app/api/job_analytics.py: skill-demand aggregation, role-family
breakdown, per-posting gap, and similar-postings search. Skill/portfolio
data seeded directly via the ORM (real DB, no mocking); the similar-
postings case uses a real :memory: Qdrant with a fake embedder, same
convention as test_search.py.
"""

from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import Account, JobPosting, RoleFamily, Skill, get_db, init_db
from app.core.settings import get_settings


def _reset_db(tmp_path: Path):
    import os

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


def _make_account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _make_posting(account_id: int, *, title: str, skills: list[dict], role_family_id=None, n=1):
    db = get_db()
    posting = JobPosting(
        account_id=account_id,
        source="pasted",
        external_id=f"h{n}",
        company="Acme",
        title=title,
        raw_text_quarantined="text",
        content_hash=f"hash-{n}",
        extraction_status="extracted",
        extracted_json={"skills_required": skills, "role_summary": "", "other_requirements": []},
        role_family_id=role_family_id,
    )
    db.add(posting)
    db.commit()
    db.refresh(posting)
    posting_id = posting.id
    db.close()
    return posting_id


def test_skills_demand_counts_and_flags_have_it(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    db = get_db()
    db.add(Skill(account_id=account_id, name="Python"))
    db.commit()
    db.close()

    _make_posting(
        account_id, title="Backend Engineer", n=1,
        skills=[{"skill": "Python", "level": "senior"}, {"skill": "Go", "level": ""}],
    )
    _make_posting(
        account_id, title="Backend Engineer II", n=2,
        skills=[{"skill": "Python", "level": "senior"}],
    )

    client = _client()
    resp = client.get(f"/api/job-analytics/skills-demand?account_id={account_id}")
    assert resp.status_code == 200
    body = {row["skill"]: row for row in resp.json()}

    assert body["Python"]["posting_count"] == 2
    assert body["Python"]["have_it"] is True
    assert body["Python"]["dominant_level"] == "senior"
    assert body["Go"]["posting_count"] == 1
    assert body["Go"]["have_it"] is False


def test_skills_demand_filters_by_role_family(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    db = get_db()
    family = RoleFamily(canonical_name="Backend Engineer")
    db.add(family)
    db.commit()
    db.refresh(family)
    family_id = family.id
    db.close()

    _make_posting(
        account_id, title="Backend Engineer", n=1, role_family_id=family_id,
        skills=[{"skill": "Go", "level": ""}],
    )
    _make_posting(
        account_id, title="Frontend Engineer", n=2, role_family_id=None,
        skills=[{"skill": "React", "level": ""}],
    )

    client = _client()
    resp = client.get(
        f"/api/job-analytics/skills-demand?account_id={account_id}&role_family_id={family_id}"
    )
    skills = {row["skill"] for row in resp.json()}
    assert skills == {"Go"}


def test_role_families_returns_top_skills_per_family(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    db = get_db()
    family = RoleFamily(canonical_name="Backend Engineer")
    db.add(family)
    db.commit()
    db.refresh(family)
    family_id = family.id
    db.close()

    _make_posting(
        account_id, title="Backend Engineer", n=1, role_family_id=family_id,
        skills=[{"skill": "Go", "level": ""}],
    )
    _make_posting(
        account_id, title="Backend Engineer II", n=2, role_family_id=family_id,
        skills=[{"skill": "Go", "level": ""}, {"skill": "Python", "level": ""}],
    )

    client = _client()
    resp = client.get(f"/api/job-analytics/role-families?account_id={account_id}")
    body = resp.json()
    assert len(body) == 1
    assert body[0]["canonical_name"] == "Backend Engineer"
    assert body[0]["posting_count"] == 2
    assert set(body[0]["top_skills"]) == {"Go", "Python"}


def test_gap_splits_matched_and_missing(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    db = get_db()
    db.add(Skill(account_id=account_id, name="Python"))
    db.commit()
    db.close()

    posting_id = _make_posting(
        account_id, title="Backend Engineer", n=1,
        skills=[{"skill": "Python", "level": ""}, {"skill": "Rust", "level": ""}],
    )

    client = _client()
    resp = client.get(f"/api/job-analytics/gap/{posting_id}?account_id={account_id}")
    body = resp.json()
    assert body["matched"] == ["Python"]
    assert body["missing"] == ["Rust"]


def test_gap_unknown_posting_404s(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    client = _client()
    resp = client.get(f"/api/job-analytics/gap/999999?account_id={account_id}")
    assert resp.status_code == 404


def test_similar_postings_uses_semantic_search(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()

    def _fake_embed(texts):
        return [[1.0, 0.0] if "python" in t.lower() else [0.0, 1.0] for t in texts]

    import app.retrieval.index as index_module
    import app.retrieval.search as search_module

    monkeypatch.setattr(index_module, "embed", _fake_embed)
    monkeypatch.setattr(search_module, "embed", _fake_embed)

    p1 = _make_posting(
        account_id, title="Python Backend Engineer", n=1, skills=[{"skill": "Python", "level": ""}]
    )
    p2 = _make_posting(
        account_id, title="Python Platform Engineer", n=2, skills=[{"skill": "Python", "level": ""}]
    )
    p3 = _make_posting(
        account_id, title="Woodworking Instructor", n=3,
        skills=[{"skill": "Woodworking", "level": ""}],
    )

    from app.core.db import JobPosting as JP
    from app.retrieval.index import index_job_posting

    db = get_db()
    for pid in (p1, p2, p3):
        index_job_posting(db.get(JP, pid), account_id=account_id)
    db.close()

    client = _client()
    resp = client.get(f"/api/job-analytics/similar/{p1}?account_id={account_id}")
    body = resp.json()

    ids = [row["id"] for row in body]
    assert p1 not in ids  # never suggests itself
    assert p2 in ids
    # the closer match ranks first when both are present
    assert p3 not in ids or ids.index(p2) < ids.index(p3)
