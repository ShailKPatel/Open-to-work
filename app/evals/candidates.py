"""Candidate-level scoring: what the resume builder actually hands the
model, not the raw search hits run.py scores.

app/resume_build/orchestrator.py does not pass search hits straight
through. _candidate_skills dedupes them by name and drops archived or
excluded skills; _candidate_projects groups them by repository and drops
the profile README and excluded projects. A change to any of that (or to
how candidates are ranked) moves nothing in the dense numbers, so it is
scored here, against the same golden pairs.

A skill_evidence pair's labeled evidence ids are turned into what the
candidates are: the skill names they carry, and the repositories the
repo-linked ones belong to. Experience-point pairs have no candidate step
and are not scored here.
"""

from __future__ import annotations

from sqlalchemy import select

from app.core.db import ExperienceSkillEvidence, SkillEvidence, get_db
from app.evals.golden import GoldenPair
from app.evals.metrics import SystemScore, mean_system_score, precision_at_k, recall_at_k
from app.retrieval.index import experience_evidence_point_id

_PRECISION_K = 5
_RECALL_K = 10


def score_candidates(account_id: int, pairs: list[GoldenPair]) -> dict[str, SystemScore]:
    """Mean precision@5 / recall@10 of candidate skills (by name,
    casefolded) and candidate projects (by repository id) over every
    skill_evidence pair with labels. A pair whose labels name no
    repo-linked evidence counts toward skills only."""
    from app.resume_build.orchestrator import _candidate_projects, _candidate_skills

    offset = experience_evidence_point_id(0)
    skill_points: list[tuple[float, float]] = []
    project_points: list[tuple[float, float]] = []
    db = get_db()
    try:
        for pair in pairs:
            if pair.collection != "skill_evidence" or not pair.relevant_ids:
                continue
            repo_ids = [i for i in pair.relevant_ids if i < offset]
            role_ids = [i - offset for i in pair.relevant_ids if i >= offset]
            repo_rows = list(
                db.execute(select(SkillEvidence).where(SkillEvidence.id.in_(repo_ids))).scalars()
            )
            role_skills = db.execute(
                select(ExperienceSkillEvidence.skill).where(
                    ExperienceSkillEvidence.id.in_(role_ids)
                )
            ).scalars()
            relevant_skills = {e.skill.casefold() for e in repo_rows}
            relevant_skills |= {s.casefold() for s in role_skills}
            relevant_repos = {e.repo_id for e in repo_rows}

            skills = [s.casefold() for s in _candidate_skills(db, account_id, pair.query_text)]
            skill_points.append(
                (
                    precision_at_k(skills, relevant_skills, _PRECISION_K),
                    recall_at_k(skills, relevant_skills, _RECALL_K),
                )
            )
            if relevant_repos:
                projects = [
                    c["repo_id"] for c in _candidate_projects(db, account_id, pair.query_text)
                ]
                project_points.append(
                    (
                        precision_at_k(projects, relevant_repos, _PRECISION_K),
                        recall_at_k(projects, relevant_repos, _RECALL_K),
                    )
                )
    finally:
        db.close()
    return {
        "skills": mean_system_score(skill_points),
        "projects": mean_system_score(project_points),
    }
