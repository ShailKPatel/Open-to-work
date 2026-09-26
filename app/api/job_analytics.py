"""Job-posting analytics: skill demand across everything an account has
collected, rolled up by canonical role family (app/profile/role_family.py)
so "ML Engineer" and "Machine Learning Engineer" postings count together
instead of splitting into two unrelated buckets.

Skill-demand counting is plain SQL aggregation over
JobPosting.extracted_json, not a vector-search feature: semantic search
answers "which postings read as similar to this one" (search_job_postings
below) but cannot correctly answer "how many postings asked for Python";
that needs exact counting over the structured fields every extracted
posting already has, which this file computes directly rather than
approximating through embeddings.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.skills import account_skill_names
from app.core.db import JobPosting, RoleFamily
from app.profile.job_extract import parse_skills_required

router = APIRouter(prefix="/api/job-analytics")


def _extracted_postings(
    db: Session, account_id: int, role_family_id: int | None
) -> list[JobPosting]:
    query = select(JobPosting).where(
        JobPosting.account_id == account_id, JobPosting.extraction_status == "extracted"
    )
    if role_family_id is not None:
        query = query.where(JobPosting.role_family_id == role_family_id)
    return list(db.execute(query).scalars())


class SkillDemand(BaseModel):
    skill: str
    posting_count: int
    dominant_level: str
    level_counts: dict[str, int]
    have_it: bool


@router.get("/skills-demand", response_model=list[SkillDemand])
def skills_demand(
    account_id: int,
    role_family_id: int | None = None,
    *,
    db: DbSession,
) -> list[SkillDemand]:
    """Every skill mentioned across this account's extracted postings
    (optionally narrowed to one role family), most-requested first, each
    flagged with whether the account's own portfolio already demonstrates
    it: "these are your skills, these are missing, these are what's most
    asked for," in one list.
    """
    postings = _extracted_postings(db, account_id, role_family_id)
    have = account_skill_names(account_id)

    counts: Counter[str] = Counter()
    level_counts: dict[str, Counter[str]] = {}
    display: dict[str, str] = {}
    for posting in postings:
        extracted = posting.extracted_json or {}
        for item in parse_skills_required(extracted.get("skills_required", [])):
            key = item["skill"].casefold()
            display.setdefault(key, item["skill"])
            counts[key] += 1
            if item["level"]:
                level_counts.setdefault(key, Counter())[item["level"]] += 1

    results = []
    for key, count in counts.items():
        levels = level_counts.get(key, Counter())
        dominant = levels.most_common(1)[0][0] if levels else ""
        results.append(
            SkillDemand(
                skill=display[key],
                posting_count=count,
                dominant_level=dominant,
                level_counts=dict(levels),
                have_it=key in have,
            )
        )
    results.sort(key=lambda r: (-r.posting_count, r.skill.casefold()))
    return results


class RoleFamilySummary(BaseModel):
    id: int
    canonical_name: str
    posting_count: int
    top_skills: list[str]


@router.get("/role-families", response_model=list[RoleFamilySummary])
def role_families(account_id: int, *, db: DbSession) -> list[RoleFamilySummary]:
    """Every canonical role this account has collected postings under,
    with each one's most-requested skills, the per-role view: "for
    this role, these are the skills that come up most.\""""
    postings = _extracted_postings(db, account_id, None)
    by_family: dict[int, list[JobPosting]] = {}
    for p in postings:
        if p.role_family_id is not None:
            by_family.setdefault(p.role_family_id, []).append(p)

    families = {f.id: f for f in db.execute(select(RoleFamily)).scalars()}
    results = []
    for family_id, rows in by_family.items():
        family = families.get(family_id)
        if family is None:
            continue
        counts: Counter[str] = Counter()
        display: dict[str, str] = {}
        for p in rows:
            extracted = p.extracted_json or {}
            for item in parse_skills_required(extracted.get("skills_required", [])):
                key = item["skill"].casefold()
                display.setdefault(key, item["skill"])
                counts[key] += 1
        top = [display[k] for k, _ in counts.most_common(8)]
        results.append(
            RoleFamilySummary(
                id=family.id,
                canonical_name=family.canonical_name,
                posting_count=len(rows),
                top_skills=top,
            )
        )
    results.sort(key=lambda r: -r.posting_count)
    return results


class SkillGap(BaseModel):
    matched: list[str]
    missing: list[str]


@router.get("/gap/{posting_id}", response_model=SkillGap)
def posting_gap(posting_id: int, account_id: int, *, db: DbSession) -> SkillGap:
    """This one posting's required skills, split into what the account's
    portfolio already demonstrates and what it doesn't: the per-job
    "you have this / you're missing this" view."""
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")
    have = account_skill_names(account_id)
    extracted = posting.extracted_json or {}
    matched: list[str] = []
    missing: list[str] = []
    for item in parse_skills_required(extracted.get("skills_required", [])):
        target = matched if item["skill"].casefold() in have else missing
        target.append(item["skill"])
    return SkillGap(matched=matched, missing=missing)


class SimilarPosting(BaseModel):
    id: int
    title: str
    company: str
    score: float


@router.get("/similar/{posting_id}", response_model=list[SimilarPosting])
def similar_postings(
    posting_id: int,
    account_id: int,
    top_k: int = 5,
    *,
    db: DbSession,
) -> list[SimilarPosting]:
    """Semantic-search companion to the exact-counting endpoints above:
    other postings this account has collected that read as similar to
    this one (app/retrieval/search.py's search_job_postings). The only
    retrieval-based view in this file; everything above is exact
    aggregation."""
    posting = db.get(JobPosting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={posting_id}")

    from app.retrieval.index import job_posting_text
    from app.retrieval.search import search_job_postings

    query_text = job_posting_text(posting)
    if not query_text.strip():
        return []
    hits = search_job_postings(query_text, account_id, top_k=top_k + 1)
    results = []
    for hit in hits:
        if hit.id == posting_id:
            continue
        other = db.get(JobPosting, hit.id)
        if other is None:
            continue
        results.append(
            SimilarPosting(
                id=other.id, title=other.title, company=other.company, score=hit.score
            )
        )
    return results[:top_k]
