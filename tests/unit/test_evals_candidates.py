"""app/evals/candidates.py: golden evidence ids turned into skill names and
repository ids, scored against what the resume builder's candidate steps
return. The candidate steps are stubbed; what is under test is the
mapping and the scoring."""

from pathlib import Path

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.core.db import (
    Account,
    Experience,
    ExperienceSkillEvidence,
    Repository,
    SkillEvidence,
    get_db,
    init_db,
)
from app.core.settings import get_settings
from app.evals.candidates import score_candidates
from app.evals.golden import GoldenPair
from app.retrieval.index import experience_evidence_point_id


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _seed() -> dict[str, int]:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    repos = [
        Repository(
            account_id=account.id, github_id=i, name=name, full_name=f"octocat/{name}",
            url=f"https://github.com/octocat/{name}", is_fork=False,
        )
        for i, name in enumerate(("api", "web"), start=1)
    ]
    db.add_all(repos)
    db.commit()
    python = SkillEvidence(
        skill="Python", repo_id=repos[0].id, evidence_type="declared_dependency",
        weight=0.5, confidence=1.0,
    )
    react = SkillEvidence(
        skill="React", repo_id=repos[1].id, evidence_type="declared_dependency",
        weight=0.5, confidence=1.0,
    )
    role = Experience(account_id=account.id, title="Engineer", company="Acme")
    db.add_all([python, react, role])
    db.commit()
    kafka = ExperienceSkillEvidence(skill="Kafka", experience_id=role.id)
    db.add(kafka)
    db.commit()
    ids = {
        "account": account.id, "api": repos[0].id, "web": repos[1].id,
        "python": python.id, "kafka": experience_evidence_point_id(kafka.id),
    }
    db.close()
    return ids


def test_scores_candidate_skills_by_name_and_projects_by_repository(tmp_path, monkeypatch):
    _reset(tmp_path)
    ids = _seed()
    monkeypatch.setattr(
        "app.resume_build.orchestrator._candidate_skills",
        lambda db, account_id, text: ["python", "React", "Go", "Rust", "C", "Kafka"],
    )
    monkeypatch.setattr(
        "app.resume_build.orchestrator._candidate_projects",
        lambda db, account_id, text: [{"repo_id": ids["web"]}, {"repo_id": ids["api"]}],
    )
    pairs = [
        GoldenPair(
            id="p1", account_id=ids["account"], collection="skill_evidence",
            query_text="python streaming", relevant_ids=[ids["python"], ids["kafka"]],
        ),
        GoldenPair(
            id="p2", account_id=ids["account"], collection="experience_points",
            query_text="ignored", relevant_ids=[1],
        ),
    ]

    scores = score_candidates(ids["account"], pairs)

    # Python is in the top 5 (matched case-insensitively), Kafka only in the top 10.
    assert scores["skills"].precision_at_5 == 1 / 5
    assert scores["skills"].recall_at_10 == 1.0
    assert scores["skills"].pairs_scored == 1
    # Only Python's repository is relevant; Kafka is role evidence. Two
    # projects come back, and precision@5 divides by 5 regardless.
    assert scores["projects"].precision_at_5 == 1 / 5
    assert scores["projects"].recall_at_10 == 1.0
    assert scores["projects"].pairs_scored == 1


def test_pair_with_only_role_evidence_scores_skills_but_not_projects(tmp_path, monkeypatch):
    _reset(tmp_path)
    ids = _seed()
    monkeypatch.setattr(
        "app.resume_build.orchestrator._candidate_skills", lambda db, a, t: ["Kafka"]
    )
    monkeypatch.setattr(
        "app.resume_build.orchestrator._candidate_projects", lambda db, a, t: []
    )
    pair = GoldenPair(
        id="p1", account_id=ids["account"], collection="skill_evidence",
        query_text="kafka", relevant_ids=[ids["kafka"]],
    )

    scores = score_candidates(ids["account"], [pair])

    assert scores["skills"].pairs_scored == 1
    assert scores["projects"].pairs_scored == 0
