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

    db_module.reset_engine()
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


def _seed_insights(account_id: int) -> dict:
    """Three backend postings (one a full match, one a skill short) and
    one better-paid ML posting, plus a dollar posting that must stay out
    of rupee pay comparisons."""
    db = get_db()
    db.add_all([Skill(account_id=account_id, name=n) for n in ("Python", "Node.js", "SQL")])
    backend = RoleFamily(canonical_name="Backend Engineer")
    ml = RoleFamily(canonical_name="ML Engineer")
    db.add_all([backend, ml])
    db.commit()
    ids = {"backend": backend.id, "ml": ml.id}
    db.close()

    def _posting(n, title, skills, family, pay, currency="INR", applied=False):
        db = get_db()
        row = JobPosting(
            account_id=account_id, source="pasted", external_id=f"i{n}", company=f"Co{n}",
            title=title, raw_text_quarantined="text", content_hash=f"ins-{n}",
            extraction_status="extracted", role_family_id=family, applied=applied,
            salary_max_annual=pay, salary_currency=currency if pay else None,
            extracted_json={
                "skills_required": [{"skill": s, "level": ""} for s in skills],
                "role_summary": "", "other_requirements": [],
            },
        )
        db.add(row)
        db.commit()
        ids[f"p{n}"] = row.id
        db.close()

    _posting(1, "Backend Engineer", ["Python", "NodeJS", "SQL"], ids["backend"], 1_200_000)
    _posting(2, "Backend Developer", ["Python", "SQL", "Docker"], ids["backend"], 1_400_000)
    _posting(3, "Backend II", ["Python", "Docker", "Kubernetes"], ids["backend"], 1_600_000)
    _posting(4, "ML Engineer", ["Python", "PyTorch", "Docker"], ids["ml"], 3_000_000)
    _posting(5, "ML Engineer", ["Python", "PyTorch", "SQL"], ids["ml"], 2_800_000)
    _posting(6, "Remote Backend", ["Python", "Go"], ids["backend"], 150_000, currency="USD")
    return ids


def test_insights_ranks_roles_jobs_and_skills(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    ids = _seed_insights(account_id)

    resp = _client().get(f"/api/job-analytics/insights?account_id={account_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["posting_count"] == 6
    assert data["currency"] == "INR"
    # The dollar posting is left out of every pay figure.
    assert data["paid_count"] == 5
    assert data["median_pay"] == 1_600_000

    # "NodeJS" in a posting matches "Node.js" in the portfolio.
    best = data["best_matches"][0]
    assert best["id"] == ids["p1"] and best["match_pct"] == 100

    assert data["best_role"]["name"] == "Backend Engineer"
    ml = next(r for r in data["roles"] if r["name"] == "ML Engineer")
    assert ml["median_pay"] == 2_900_000
    assert ml["missing_top"][0] == "PyTorch"

    learn = {s["skill"]: s for s in data["skills_to_learn"]}
    # Docker takes posting 2 from 67% to 100%; postings 3 and 4 only
    # reach 67%, short of a strong match, so one unlock.
    assert learn["Docker"]["posting_count"] == 3
    assert learn["Docker"]["unlocks"] == 1
    assert learn["PyTorch"]["pay_premium_pct"] > 50

    kinds = [a["kind"] for a in data["advice"]]
    assert kinds[0] == "role"
    assert "pay" in kinds and "premium" in kinds
    pay = next(a for a in data["advice"] if a["kind"] == "pay")
    assert pay["headline"].startswith("ML Engineer")
    assert "₹29L" in pay["detail"]

    assert {s["skill"] for s in data["strengths"]} >= {"Python", "SQL"}

    backend = next(r for r in data["roles"] if r["name"] == "Backend Engineer")
    # The dollar posting stays out of the rupee range.
    assert (backend["pay_low"], backend["pay_high"]) == (1_200_000, 1_600_000)
    assert backend["paid_count"] == 3
    points = {p["id"]: p for p in data["pay_points"]}
    assert len(points) == 5 and ids["p6"] not in points
    assert points[ids["p4"]]["role"] == "ML Engineer"
    assert points[ids["p4"]]["years_min"] is None


def test_insights_empty_account(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()

    data = _client().get(f"/api/job-analytics/insights?account_id={account_id}").json()

    assert data["posting_count"] == 0
    assert data["best_role"] is None
    assert data["skills_to_learn"] == []


def test_insights_places_keep_remote_apart_from_cities(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(Skill(account_id=account_id, name="Python"))
    rows = [
        ("Pune, Maharashtra, India", "On-site", ["Python"]),
        ("Pune", "Hybrid", ["Python", "Go"]),
        ("Pune / Mumbai", "", ["Go"]),
        ("Bangalore", "Remote", ["Python"]),
        (None, "", ["Python"]),
    ]
    for n, (location, mode, skills) in enumerate(rows):
        db.add(JobPosting(
            account_id=account_id, source="pasted", external_id=f"loc{n}", company=f"Co{n}",
            title="Engineer", location=location, raw_text_quarantined="text",
            content_hash=f"loc-{n}", extraction_status="extracted",
            extracted_json={
                "skills_required": [{"skill": s, "level": ""} for s in skills],
                "work_mode": mode, "role_summary": "", "other_requirements": [],
            },
        ))
    db.commit()
    db.close()

    data = _client().get(f"/api/job-analytics/insights?account_id={account_id}").json()

    places = {p["name"]: p for p in data["places"]}
    # The remote posting names Bangalore but is not counted there.
    assert "Bengaluru" not in places
    assert places["Remote"]["posting_count"] == 1
    assert places["Pune"]["posting_count"] == 3
    assert places["Pune"]["on_site_count"] == 1
    assert places["Pune"]["hybrid_count"] == 1
    assert places["Pune"]["unstated_count"] == 1
    assert places["Mumbai"]["posting_count"] == 1
    assert data["places"][-1]["name"] == "Not stated"

    modes = {m["kind"]: m["posting_count"] for m in data["work_modes"]}
    assert modes == {"on_site": 1, "hybrid": 1, "remote": 1, "unknown": 2}

    place = next(a for a in data["advice"] if a["kind"] == "place")
    assert place["headline"] == "Most of your jobs are in Pune"
    assert "remote" in place["detail"]
