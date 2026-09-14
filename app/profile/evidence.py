"""Shared skill-evidence CRUD, used by both app/api/projects.py (against
SkillEvidence, FK'd to Repository) and app/api/experience.py (against
ExperienceSkillEvidence, FK'd to Experience). Same validation, same shape,
same 404 handling. The only thing that differs between the two callers is
which ORM class and which foreign-key column they pass in, so this stays a
plain function taking both as parameters rather than two near-duplicate
routers each hand-rolling the same three handlers.

Callers own the parent-resource 404 check (repo/experience not found) and
whatever response model they build afterward; this module only knows
about the evidence row itself.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def add_evidence(
    db: Any,
    model: type,
    fk_field: str,
    fk_id: int,
    *,
    skill: str,
    evidence_type: str,
    weight: float,
    confidence: float,
) -> Any:
    skill = skill.strip()
    if not skill:
        raise HTTPException(status_code=422, detail="skill is required")
    row = model(
        skill=skill,
        evidence_type=evidence_type,
        weight=weight,
        confidence=confidence,
        **{fk_field: fk_id},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_evidence(
    db: Any, model: type, fk_field: str, fk_id: int, evidence_id: int, fields: dict
) -> Any:
    if "skill" in fields and fields["skill"] is not None:
        fields["skill"] = fields["skill"].strip()
        if not fields["skill"]:
            raise HTTPException(status_code=422, detail="skill is required")

    row = db.get(model, evidence_id)
    if row is None or getattr(row, fk_field) != fk_id:
        raise HTTPException(status_code=404, detail=f"no skill with id={evidence_id}")
    for key, value in fields.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


def delete_evidence(db: Any, model: type, fk_field: str, fk_id: int, evidence_id: int) -> None:
    row = db.get(model, evidence_id)
    if row is None or getattr(row, fk_field) != fk_id:
        raise HTTPException(status_code=404, detail=f"no skill with id={evidence_id}")
    db.delete(row)
    db.commit()
