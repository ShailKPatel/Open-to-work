"""Rebuilds every Qdrant collection from SQLite, which is the source of
truth: skill evidence, experience evidence and points, resumes, role
families and job postings. Used after a restore (app/core/restore.py), or
when the index is lost or out of step. No LLM calls; it only embeds.

    python -m app.retrieval.rebuild           # upsert into what is there
    python -m app.retrieval.rebuild --fresh   # drop the collections first

--fresh also clears points for rows deleted since, which an upsert keeps.
Qdrant is never a backup target: everything in it is derived from here.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from sqlalchemy import select

from app.core.db import (
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    JobPosting,
    Repository,
    Resume,
    RoleFamily,
    SkillEvidence,
    get_db,
)
from app.retrieval.index import (
    COLLECTION,
    EXPERIENCE_POINTS_COLLECTION,
    JOB_POSTINGS_COLLECTION,
    RESUME_COLLECTION,
    ROLE_FAMILIES_COLLECTION,
    index_experience_points,
    index_experience_skill_evidence,
    index_job_posting,
    index_resume,
    index_role_family,
    index_skill_evidence,
)
from app.retrieval.vectorstore import get_client

logger = logging.getLogger(__name__)

COLLECTIONS = (
    COLLECTION,
    EXPERIENCE_POINTS_COLLECTION,
    RESUME_COLLECTION,
    ROLE_FAMILIES_COLLECTION,
    JOB_POSTINGS_COLLECTION,
)


def drop_collections() -> None:
    client = get_client()
    for name in COLLECTIONS:
        if client.collection_exists(name):
            client.delete_collection(name)


def rebuild_index(fresh: bool = False) -> dict[str, int]:
    """Points written per kind. Collections are created on first write, at
    the embedding model's vector size."""
    if fresh:
        drop_collections()
    counts: Counter[str] = Counter()
    db = get_db()
    try:
        for repo in db.execute(select(Repository).order_by(Repository.id)).scalars():
            rows = list(
                db.execute(select(SkillEvidence).where(SkillEvidence.repo_id == repo.id)).scalars()
            )
            counts["skill_evidence"] += index_skill_evidence(rows, account_id=repo.account_id)

        for exp in db.execute(select(Experience).order_by(Experience.id)).scalars():
            evidence = list(
                db.execute(
                    select(ExperienceSkillEvidence).where(
                        ExperienceSkillEvidence.experience_id == exp.id
                    )
                ).scalars()
            )
            counts["experience_evidence"] += index_experience_skill_evidence(
                evidence, account_id=exp.account_id
            )
            points = list(
                db.execute(
                    select(ExperiencePoint).where(ExperiencePoint.experience_id == exp.id)
                ).scalars()
            )
            counts["experience_points"] += index_experience_points(
                points, account_id=exp.account_id
            )

        for resume in db.execute(select(Resume).order_by(Resume.id)).scalars():
            counts["resumes"] += int(index_resume(resume))

        for family in db.execute(select(RoleFamily).order_by(RoleFamily.id)).scalars():
            index_role_family(family)
            counts["role_families"] += 1

        for posting in db.execute(select(JobPosting).order_by(JobPosting.id)).scalars():
            counts["job_postings"] += int(
                index_job_posting(posting, account_id=posting.account_id)
            )
    finally:
        db.close()
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fresh", action="store_true", help="drop the collections first")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    counts = rebuild_index(fresh=args.fresh)
    print(", ".join(f"{name}: {n}" for name, n in sorted(counts.items())) or "nothing to index")


if __name__ == "__main__":
    main()
