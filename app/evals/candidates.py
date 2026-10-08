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
from app.evals.metrics import PairScore, SystemScore, mean_system_score, score_pair
from app.retrieval.index import experience_evidence_point_id
from app.retrieval.search import Query


def score_candidates(account_id: int, pairs: list[GoldenPair]) -> dict[str, SystemScore]:
    """score_pair's metrics, averaged, for candidate skills (by name,
    casefolded) and candidate projects (by repository id) over every
    skill_evidence pair with labels. A pair whose labels name no
    repo-linked evidence counts toward skills only."""
    from app.resume_build.orchestrator import _candidate_projects, _candidate_skills

    offset = experience_evidence_point_id(0)
    skill_points: list[PairScore] = []
    project_points: list[PairScore] = []
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

            query = Query.from_posting_text(pair.query_text)
            skills = [s.casefold() for s in _candidate_skills(db, account_id, query)]
            skill_points.append(score_pair(skills, relevant_skills))
            if relevant_repos:
                projects = [
                    c["repo_id"] for c in _candidate_projects(db, account_id, query)
                ]
                project_points.append(score_pair(projects, relevant_repos))
    finally:
        db.close()
    return {
        "skills": mean_system_score(skill_points),
        "projects": mean_system_score(project_points),
    }
