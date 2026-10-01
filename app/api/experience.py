"""Work-experience CRUD and its skill evidence: the second evidence
source (alongside app/api/projects.py's repos) that feeds the /skills
aggregate view (app/api/skills.py). Every Experience row is manual; there
is no GitHub-shaped sync for a job history, so unlike projects there's no
is_manual flag or synthetic-id trick, and no LLM extraction pass yet
(evidence_type is "manual" only, see app/core/db/models.py's ExperienceSkillEvidence
docstring for why this is a separate table from SkillEvidence).

Experience itself carries no free-text description; every detail lives as
an ExperiencePoint (see that model's docstring), and every point add/edit
here also gets embedded into Qdrant (index_experience_points) so a later
resume-building pass can pull back whichever points fit a target job.
"""

from __future__ import annotations

import datetime as dt
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.core.db import Experience, ExperiencePoint, ExperienceSkillEvidence
from app.profile.evidence import add_evidence, delete_evidence, update_evidence
from app.profile.resume_profile_merge import unlink_profile_rows
from app.profile.skill_review import approve

router = APIRouter(prefix="/api/experience")
logger = logging.getLogger(__name__)


class SkillEvidenceItem(BaseModel):
    id: int
    skill: str
    evidence_type: str
    weight: float
    confidence: float


class ExperienceSummary(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    start_date: dt.date | None
    end_date: dt.date | None
    skill_count: int

    @classmethod
    def from_experience(cls, exp: Experience, skill_count: int) -> ExperienceSummary:
        return cls(
            id=exp.id,
            title=exp.title,
            company=exp.company,
            location=exp.location,
            start_date=exp.start_date,
            end_date=exp.end_date,
            skill_count=skill_count,
        )


class PointItem(BaseModel):
    id: int
    text: str
    order_index: int

    @classmethod
    def from_point(cls, p: ExperiencePoint) -> PointItem:
        return cls(id=p.id, text=p.text, order_index=p.order_index)


class ExperienceDetail(BaseModel):
    id: int
    title: str
    company: str
    location: str | None
    start_date: dt.date | None
    end_date: dt.date | None
    created_at: dt.datetime
    updated_at: dt.datetime
    skills: list[SkillEvidenceItem]
    points: list[PointItem]

    @classmethod
    def from_experience(
        cls, exp: Experience, skills: list[SkillEvidenceItem], points: list[PointItem]
    ) -> ExperienceDetail:
        return cls(
            id=exp.id,
            title=exp.title,
            company=exp.company,
            location=exp.location,
            start_date=exp.start_date,
            end_date=exp.end_date,
            created_at=exp.created_at,
            updated_at=exp.updated_at,
            skills=skills,
            points=points,
        )


def _skill_counts(db: Session, experience_ids: list[int]) -> dict[int, int]:
    if not experience_ids:
        return {}
    rows = db.execute(
        select(ExperienceSkillEvidence.experience_id, func.count(ExperienceSkillEvidence.id))
        .where(ExperienceSkillEvidence.experience_id.in_(experience_ids))
        .group_by(ExperienceSkillEvidence.experience_id)
    ).all()
    # Row unpacks as a 2-tuple; spelling it out keeps the dict's key/value
    # types checkable instead of landing as Row objects.
    return {group_key: count for group_key, count in rows}


def _skills_for(db: Session, experience_id: int) -> list[SkillEvidenceItem]:
    rows = db.execute(
        select(ExperienceSkillEvidence).where(
            ExperienceSkillEvidence.experience_id == experience_id
        )
    ).scalars()
    skills = [
        SkillEvidenceItem(
            id=r.id, skill=r.skill, evidence_type=r.evidence_type,
            weight=r.weight, confidence=r.confidence,
        )
        for r in rows
    ]
    skills.sort(key=lambda s: s.weight, reverse=True)
    return skills


def _points_for(db: Session, experience_id: int) -> list[PointItem]:
    rows = db.execute(
        select(ExperiencePoint)
        .where(ExperiencePoint.experience_id == experience_id)
        .order_by(ExperiencePoint.order_index, ExperiencePoint.id)
    ).scalars()
    return [PointItem.from_point(r) for r in rows]


def _detail_for(db: Session, exp: Experience) -> ExperienceDetail:
    return ExperienceDetail.from_experience(exp, _skills_for(db, exp.id), _points_for(db, exp.id))


@router.get("", response_model=list[ExperienceSummary])
def list_experience(account_id: int, *, db: DbSession) -> list[ExperienceSummary]:
    rows = list(
        db.execute(
            select(Experience)
            .where(Experience.account_id == account_id)
            .order_by(Experience.start_date.desc().nulls_last())
        ).scalars()
    )
    counts = _skill_counts(db, [r.id for r in rows])
    return [ExperienceSummary.from_experience(r, counts.get(r.id, 0)) for r in rows]


class ExperienceCreate(BaseModel):
    account_id: int
    title: str
    company: str
    location: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None


@router.post("", response_model=ExperienceDetail)
def create_experience(body: ExperienceCreate, *, db: DbSession) -> ExperienceDetail:
    title = body.title.strip()
    company = body.company.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title is required")
    if not company:
        raise HTTPException(status_code=422, detail="company is required")

    exp = Experience(
        account_id=body.account_id,
        title=title,
        company=company,
        location=(body.location or None),
        start_date=body.start_date,
        end_date=body.end_date,
    )
    db.add(exp)
    db.commit()
    db.refresh(exp)
    return _detail_for(db, exp)


@router.get("/{experience_id}", response_model=ExperienceDetail)
def experience_detail(experience_id: int, *, db: DbSession) -> ExperienceDetail:
    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    return _detail_for(db, exp)


class ExperienceUpdate(BaseModel):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None


@router.patch("/{experience_id}", response_model=ExperienceDetail)
def update_experience(
    experience_id: int,
    body: ExperienceUpdate,
    *,
    db: DbSession,
) -> ExperienceDetail:
    fields = body.model_dump(exclude_unset=True)
    for key in ("title", "company"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
            if not fields[key]:
                raise HTTPException(status_code=422, detail=f"{key} is required")

    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    for key, value in fields.items():
        setattr(exp, key, value)
    db.commit()
    db.refresh(exp)
    return _detail_for(db, exp)


def _delete_qdrant_points(evidence_ids: list[int]) -> None:
    """Best-effort, same reasoning as app/api/accounts.py's helper of the
    same name: SQLite is the source of truth, Qdrant is a derived index.
    """
    if not evidence_ids:
        return
    try:
        from app.retrieval.index import COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(COLLECTION):
            client.delete(collection_name=COLLECTION, points_selector=evidence_ids)
    except Exception:
        logger.exception("could not clean up Qdrant points for deleted experience; continuing")


def _delete_point_vectors(point_ids: list[int]) -> None:
    """Twin of _delete_qdrant_points above, against the experience_points
    collection (app/retrieval/index.py) rather than skill_evidence. Kept
    separate rather than parameterized: same reasoning app/api/accounts.py
    already gives for its own twin helpers.
    """
    if not point_ids:
        return
    try:
        from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(EXPERIENCE_POINTS_COLLECTION):
            client.delete(collection_name=EXPERIENCE_POINTS_COLLECTION, points_selector=point_ids)
    except Exception:
        logger.exception(
            "could not clean up point Qdrant vectors for deleted experience; continuing"
        )


@router.delete("/{experience_id}")
def delete_experience(experience_id: int, *, db: DbSession) -> dict:
    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")

    evidence_ids = list(
        db.execute(
            select(ExperienceSkillEvidence.id).where(
                ExperienceSkillEvidence.experience_id == experience_id
            )
        ).scalars()
    )
    from app.retrieval.index import experience_evidence_point_id

    _delete_qdrant_points([experience_evidence_point_id(i) for i in evidence_ids])

    point_ids = list(
        db.execute(
            select(ExperiencePoint.id).where(
                ExperiencePoint.experience_id == experience_id
            )
        ).scalars()
    )
    _delete_point_vectors(point_ids)

    db.execute(
        delete(ExperienceSkillEvidence).where(
            ExperienceSkillEvidence.experience_id == experience_id
        )
    )
    db.execute(
        delete(ExperiencePoint).where(ExperiencePoint.experience_id == experience_id)
    )
    unlink_profile_rows(db, "experience_skill", evidence_ids)
    unlink_profile_rows(db, "experience_point", point_ids)
    unlink_profile_rows(db, "experience", [experience_id])
    db.delete(exp)
    db.commit()
    return {"deleted": True, "id": experience_id}


class PointCreate(BaseModel):
    text: str


class PointUpdate(BaseModel):
    text: str


@router.post("/{experience_id}/points", response_model=ExperienceDetail)
def add_point(experience_id: int, body: PointCreate, *, db: DbSession) -> ExperienceDetail:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    next_order = (
        db.execute(
            select(func.max(ExperiencePoint.order_index)).where(
                ExperiencePoint.experience_id == experience_id
            )
        ).scalar()
        or 0
    ) + 1
    point = ExperiencePoint(experience_id=experience_id, text=text, order_index=next_order)
    db.add(point)
    db.commit()
    db.refresh(point)
    try:
        from app.retrieval.index import index_experience_points

        index_experience_points([point], account_id=exp.account_id)
    except Exception:
        # Same posture as add_skill below: the SQLite write already
        # committed, a Qdrant hiccup here shouldn't fail the request.
        logger.exception("could not index experience point id=%s", point.id)
    db.refresh(exp)
    return _detail_for(db, exp)


@router.patch("/{experience_id}/points/{point_id}", response_model=ExperienceDetail)
def update_point(
    experience_id: int,
    point_id: int,
    body: PointUpdate,
    *,
    db: DbSession,
) -> ExperienceDetail:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")

    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    point = db.execute(
        select(ExperiencePoint).where(
            ExperiencePoint.id == point_id, ExperiencePoint.experience_id == experience_id
        )
    ).scalar_one_or_none()
    if point is None:
        raise HTTPException(status_code=404, detail=f"no point with id={point_id}")
    point.text = text
    db.commit()
    db.refresh(point)
    try:
        from app.retrieval.index import index_experience_points

        index_experience_points([point], account_id=exp.account_id)
    except Exception:
        logger.exception("could not re-index experience point id=%s", point.id)
    db.refresh(exp)
    return _detail_for(db, exp)


@router.delete("/{experience_id}/points/{point_id}", response_model=ExperienceDetail)
def delete_point(experience_id: int, point_id: int, *, db: DbSession) -> ExperienceDetail:
    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    point = db.execute(
        select(ExperiencePoint).where(
            ExperiencePoint.id == point_id, ExperiencePoint.experience_id == experience_id
        )
    ).scalar_one_or_none()
    if point is None:
        raise HTTPException(status_code=404, detail=f"no point with id={point_id}")
    unlink_profile_rows(db, "experience_point", [point_id])
    db.delete(point)
    db.commit()
    _delete_point_vectors([point_id])
    db.refresh(exp)
    return _detail_for(db, exp)


class SkillEvidenceCreate(BaseModel):
    skill: str
    evidence_type: str = "manual"
    weight: float = 1.0
    confidence: float = 1.0


class SkillEvidenceUpdate(BaseModel):
    skill: str | None = None
    evidence_type: str | None = None
    weight: float | None = None
    confidence: float | None = None


@router.post("/{experience_id}/skills", response_model=ExperienceDetail)
def add_skill(experience_id: int, body: SkillEvidenceCreate, *, db: DbSession) -> ExperienceDetail:
    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    row = add_evidence(
        db,
        ExperienceSkillEvidence,
        "experience_id",
        experience_id,
        skill=body.skill,
        evidence_type=body.evidence_type,
        weight=body.weight,
        confidence=body.confidence,
    )
    # Adding it by hand overrules an earlier "not a skill" review verdict.
    approve(db, exp.account_id, row.skill)
    try:
        from app.retrieval.index import index_experience_skill_evidence

        index_experience_skill_evidence([row], account_id=exp.account_id)
    except Exception:
        # Same posture as app/profile/build.py's indexing calls: the
        # SQLite write already committed above, which is the data that
        # matters; a Qdrant hiccup here shouldn't fail the request.
        logger.exception("could not index experience skill evidence id=%s", row.id)
    db.refresh(exp)
    return _detail_for(db, exp)


@router.patch("/{experience_id}/skills/{skill_id}", response_model=ExperienceDetail)
def update_skill(
    experience_id: int,
    skill_id: int,
    body: SkillEvidenceUpdate,
    *,
    db: DbSession,
) -> ExperienceDetail:
    fields = body.model_dump(exclude_unset=True)

    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    row = update_evidence(
        db, ExperienceSkillEvidence, "experience_id", experience_id, skill_id, fields
    )
    if fields.get("skill"):
        approve(db, exp.account_id, row.skill)
    db.refresh(exp)
    return _detail_for(db, exp)


@router.delete("/{experience_id}/skills/{skill_id}", response_model=ExperienceDetail)
def delete_skill(experience_id: int, skill_id: int, *, db: DbSession) -> ExperienceDetail:
    exp = db.get(Experience, experience_id)
    if exp is None:
        raise HTTPException(status_code=404, detail=f"no experience with id={experience_id}")
    unlink_profile_rows(db, "experience_skill", [skill_id])
    delete_evidence(db, ExperienceSkillEvidence, "experience_id", experience_id, skill_id)
    from app.retrieval.index import experience_evidence_point_id

    _delete_qdrant_points([experience_evidence_point_id(skill_id)])
    db.refresh(exp)
    return _detail_for(db, exp)
