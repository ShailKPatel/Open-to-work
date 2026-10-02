"""Education CRUD. Single-table and flat, unlike
app/api/experience.py: an Education row has no points sub-resource, a
degree line doesn't split into independently-retrievable units the way
job-history detail does (see
app/core/db/models.py's Education docstring). Same account-scoping and manual-
entry posture as Experience otherwise.

grade and details are optional: blank strings are stored as nothing, so
an entry without them renders on a resume exactly as one that never had
them (see the Education model docstring).

start_date/end_date are stored as "Mar 2026" (or "2026"), whatever form
they arrive in; see app/profile/month_year.py.

exclude_from_resume archives an entry: it stays listed (the page shows it
under Archived) but resume building and the portfolio counts skip it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import DbSession
from app.core.db import Education
from app.profile.month_year import newest_first, normalize
from app.profile.resume_profile_merge import unlink_profile_rows

router = APIRouter(prefix="/api/education")


class EducationItem(BaseModel):
    id: int
    institution: str
    degree: str
    location: str | None
    start_date: str | None
    end_date: str | None
    grade: str | None
    details: list[str]
    exclude_from_resume: bool

    @classmethod
    def from_row(cls, row: Education) -> EducationItem:
        return cls(
            id=row.id, institution=row.institution, degree=row.degree,
            location=row.location, start_date=row.start_date, end_date=row.end_date,
            grade=row.grade, details=list(row.details or []),
            exclude_from_resume=row.exclude_from_resume,
        )


class EducationCreate(BaseModel):
    account_id: int
    institution: str
    degree: str
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    grade: str | None = None
    details: list[str] = []


class EducationUpdate(BaseModel):
    institution: str | None = None
    degree: str | None = None
    location: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    grade: str | None = None
    details: list[str] | None = None
    exclude_from_resume: bool | None = None


def _clean_grade(grade: str | None) -> str | None:
    return (grade or "").strip() or None


def clean_date(field: str, value: str | None) -> str | None:
    try:
        return normalize(value)
    except ValueError:
        raise HTTPException(
            status_code=422, detail=f"{field}: use a month and year like Mar 2026"
        ) from None


def _clean_details(details: list[str] | None) -> list[str]:
    return [line.strip() for line in details or [] if line.strip()]


@router.get("", response_model=list[EducationItem])
def list_education(account_id: int, *, db: DbSession) -> list[EducationItem]:
    rows = newest_first(
        list(db.execute(select(Education).where(Education.account_id == account_id)).scalars())
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
        start_date=clean_date("start_date", body.start_date),
        end_date=clean_date("end_date", body.end_date),
        grade=_clean_grade(body.grade),
        details=_clean_details(body.details),
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
    for key in ("start_date", "end_date"):
        if key in fields:
            fields[key] = clean_date(key, fields[key])
    if "grade" in fields:
        fields["grade"] = _clean_grade(fields["grade"])
    if "details" in fields:
        fields["details"] = _clean_details(fields["details"])
    if fields.get("exclude_from_resume", False) is None:
        del fields["exclude_from_resume"]

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
