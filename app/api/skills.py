"""The /skills aggregate view: one row per skill name, unioned across
three sources for an account: project-linked evidence (SkillEvidence, via
Repository), experience-linked evidence (ExperienceSkillEvidence, via
Experience), and freestanding manual entries (Skill, no evidence at all).
Nothing here writes evidence rows; that's app/api/projects.py's and
app/api/experience.py's job; this router only reads across both and owns
CRUD for the freestanding Skill table.

Grouping key is name.strip().casefold(), so "Python" and "python" land in
one group, displayed using whichever casing was seen first. This is a
display-time union, not a stored/materialized table, so it stays correct
automatically as evidence rows are added/edited/removed elsewhere.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.db import (
    Experience,
    ExperienceSkillEvidence,
    Repository,
    Skill,
    SkillEvidence,
    SkillStar,
    get_db,
)

router = APIRouter(prefix="/api/skills")


class SkillSource(BaseModel):
    type: str  # "project" | "experience"
    evidence_id: int
    ref_id: int  # repo_id or experience_id, for linking to its detail page
    name: str  # repo full_name, or "title @ company"
    evidence_type: str
    weight: float
    confidence: float


class SkillGroup(BaseModel):
    name: str
    manual_skill_id: int | None = None
    sources: list[SkillSource] = []
    starred: bool = False


def _name_key(name: str) -> str:
    return name.strip().casefold()


def _load_groups(db, account_id: int) -> dict[str, SkillGroup]:
    """Builds the name-casefold -> SkillGroup map GET /api/skills returns."""
    groups: dict[str, SkillGroup] = {}

    def _group_for(raw_name: str) -> SkillGroup:
        name = raw_name.strip()
        key = _name_key(name)
        if key not in groups:
            groups[key] = SkillGroup(name=name)
        return groups[key]

    project_rows = db.execute(
        select(SkillEvidence, Repository)
        .join(Repository, SkillEvidence.repo_id == Repository.id)
        .where(Repository.account_id == account_id)
    ).all()
    for evidence, repo in project_rows:
        _group_for(evidence.skill).sources.append(
            SkillSource(
                type="project",
                evidence_id=evidence.id,
                ref_id=repo.id,
                name=repo.full_name,
                evidence_type=evidence.evidence_type,
                weight=evidence.weight,
                confidence=evidence.confidence,
            )
        )

    experience_rows = db.execute(
        select(ExperienceSkillEvidence, Experience)
        .join(Experience, ExperienceSkillEvidence.experience_id == Experience.id)
        .where(Experience.account_id == account_id)
    ).all()
    for evidence, exp in experience_rows:
        _group_for(evidence.skill).sources.append(
            SkillSource(
                type="experience",
                evidence_id=evidence.id,
                ref_id=exp.id,
                name=f"{exp.title} @ {exp.company}",
                evidence_type=evidence.evidence_type,
                weight=evidence.weight,
                confidence=evidence.confidence,
            )
        )

    manual_rows = db.execute(select(Skill).where(Skill.account_id == account_id)).scalars()
    for skill in manual_rows:
        group = _group_for(skill.name)
        group.manual_skill_id = skill.id

    # Stars never create a group, only mark one that exists from the
    # sources above.
    star_keys = db.execute(
        select(SkillStar.name_key).where(SkillStar.account_id == account_id)
    ).scalars()
    for key in star_keys:
        if key in groups:
            groups[key].starred = True

    return groups


@router.get("", response_model=list[SkillGroup])
def list_skills(account_id: int) -> list[SkillGroup]:
    db = get_db()
    try:
        groups = _load_groups(db, account_id)
        return sorted(groups.values(), key=lambda g: (not g.starred, g.name.casefold()))
    finally:
        db.close()


class SkillStarUpdate(BaseModel):
    account_id: int
    name: str
    starred: bool


@router.post("/star", response_model=SkillStarUpdate)
def set_skill_star(body: SkillStarUpdate) -> SkillStarUpdate:
    """Star or unstar a skill by name. Idempotent both ways: starring an
    already-starred skill or unstarring an unstarred one is a no-op, not
    an error. Keyed by casefolded name, so it applies to the whole group
    whatever sources back it (see SkillStar).
    """
    key = _name_key(body.name)
    if not key:
        raise HTTPException(status_code=422, detail="name is required")

    db = get_db()
    try:
        existing = db.execute(
            select(SkillStar).where(
                SkillStar.account_id == body.account_id, SkillStar.name_key == key
            )
        ).scalar_one_or_none()
        if body.starred and existing is None:
            db.add(SkillStar(account_id=body.account_id, name_key=key))
            db.commit()
        elif not body.starred and existing is not None:
            db.delete(existing)
            db.commit()
        return body
    finally:
        db.close()


def account_skill_names(account_id: int) -> dict[str, str]:
    """casefolded skill name -> display name, for callers that need a
    cheap "does this account have skill X" check without an HTTP round
    trip (app/api/job_analytics.py's gap computation). Same union
    _load_groups already builds for GET /api/skills, reshaped. Kept as
    a thin wrapper rather than duplicating the query, so the two
    stay in sync automatically.
    """
    db = get_db()
    try:
        groups = _load_groups(db, account_id)
        return {key: group.name for key, group in groups.items()}
    finally:
        db.close()


class SkillItem(BaseModel):
    id: int
    name: str


class SkillCreate(BaseModel):
    account_id: int
    name: str


@router.post("", response_model=SkillItem)
def create_skill(body: SkillCreate) -> SkillItem:
    """Adds a freestanding skill: one with no project or experience
    behind it. A skill that already has evidence doesn't need this: it
    already shows up in GET /api/skills without a Skill row at all.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")

    db = get_db()
    try:
        row = Skill(account_id=body.account_id, name=name)
        db.add(row)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=409, detail="this account already has a manual skill by that name"
            ) from e
        db.refresh(row)
        return SkillItem(id=row.id, name=row.name)
    finally:
        db.close()


@router.delete("/{skill_id}")
def delete_skill(skill_id: int) -> dict:
    """Removes a freestanding Skill row only. A project- or
    experience-linked skill is deleted through its own evidence endpoint
    (POST/PATCH/DELETE .../skills/{id} on projects.py or experience.py),
    so there is only one delete path for the same underlying row.
    """
    db = get_db()
    try:
        row = db.get(Skill, skill_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no skill with id={skill_id}")
        db.delete(row)
        db.commit()
        return {"deleted": True, "id": skill_id}
    finally:
        db.close()
