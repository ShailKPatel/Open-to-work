"""Education CRUD. Single-table and flat, unlike
app/api/experience.py: an Education row has no points sub-resource, a
degree line doesn't split into independently-retrievable units the way
job-history detail does (see
app/core/db/models.py's Education docstring). Same account-scoping and manual-
entry posture as Experience otherwise.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import DbSession
from app.core.db import Education
from app.profile.resume_profile_merge import unlink_profile_rows

router = APIRouter(prefix="/api/education")


class EducationItem(BaseModel):
    id: int
    institution: str
    degree: str
    location: str | None
    start_date: dt.date | None
    end_date: dt.date | None

    @classmethod
    def from_row(cls, row: Education) -> EducationItem:
        return cls(
            id=row.id, institution=row.institution, degree=row.degree,
            location=row.location, start_date=row.start_date, end_date=row.end_date,
        )


class EducationCreate(BaseModel):
    account_id: int
    institution: str
    degree: str
    location: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None


class EducationUpdate(BaseModel):
    institution: str | None = None
    degree: str | None = None
    location: str | None = None
    start_date: dt.date | None = None
    end_date: dt.date | None = None


@router.get("", response_model=list[EducationItem])
def list_education(account_id: int, *, db: DbSession) -> list[EducationItem]:
    rows = list(
        db.execute(
            select(Education)
            .where(Education.account_id == account_id)
            .order_by(Education.start_date.desc().nulls_last())
        ).scalars()
    )
    return [EducationItem.from_row(r) for r in rows]


@router.post("", response_model=EducationItem)
def create_education(body: EducationCreate, *, db: DbSession) -> EducationItem:
    institution = body.institution.strip()
    degree = body.degree.strip()
    if not institution:
        raise HTTPException(status_code=422, detail="institution is required")
    if not degree:
        raise HTTPException(status_code=422, detail="degree is required")

    row = Education(
        account_id=body.account_id,
        institution=institution,
        degree=degree,
        location=(body.location or None),
        start_date=body.start_date,
        end_date=body.end_date,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return EducationItem.from_row(row)


@router.patch("/{education_id}", response_model=EducationItem)
def update_education(education_id: int, body: EducationUpdate, *, db: DbSession) -> EducationItem:
    fields = body.model_dump(exclude_unset=True)
    if "institution" in fields:
        fields["institution"] = fields["institution"].strip()
        if not fields["institution"]:
            raise HTTPException(status_code=422, detail="institution is required")
    if "degree" in fields:
        fields["degree"] = fields["degree"].strip()
        if not fields["degree"]:
            raise HTTPException(status_code=422, detail="degree is required")

    row = db.get(Education, education_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no education with id={education_id}")
    for key, value in fields.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return EducationItem.from_row(row)


@router.delete("/{education_id}")
def delete_education(education_id: int, *, db: DbSession) -> dict:
    row = db.get(Education, education_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no education with id={education_id}")
    unlink_profile_rows(db, "education", [education_id])
    db.delete(row)
    db.commit()
    return {"deleted": True}
