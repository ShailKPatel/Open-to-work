"""Folds a resume's LLM-extracted skills and work-history entries into an
account's actual profile data (the `skills` and `experiences` tables),
once extraction succeeds (see app/profile/resume_ingest.py's
run_extraction). Two different dedup rules, matching how each table
already treats duplicates elsewhere in this codebase:

Skills: reuses the resume's own `tags` (already "concrete skills, tools,
technologies, and practices" per resume_extract.py's prompt, no separate
"skills" field needed). A tag whose casefolded name already exists
anywhere for this account (a manual `Skill` row, or evidence from a
project or experience, the exact same union GET /api/skills reads) is
left alone, no new row, no new evidence source; a name that's
new becomes a freestanding `Skill` row, same as adding one by hand from
the Skills page. No LLM call, no fuzzy matching, casefold-equal is equal,
the same rule this app already uses everywhere skills get deduplicated.

Experience: matched on (company, title), both casefolded and trimmed, NOT
on dates. Two resumes (or a resume and a hand-entered role) describing
"Engineer at Acme" are the same line item even if one states different
start/end dates or none at all; a promotion at the same company is a
different title, so a legitimately different, second row. A matching
existing row gets its start_date/end_date filled in from the resume ONLY
where that field was previously null, so a date entered by hand (or by an
earlier resume) never gets silently overwritten; a new
(company, title) pair becomes a new Experience row.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Experience, ExperienceSkillEvidence, Repository, Skill, SkillEvidence
from app.profile.resume_extract import ExperienceClaim, ResumeExtraction


@dataclass
class ProfileMergeSummary:
    skills_added: int
    experiences_added: int
    experiences_enriched: int


def _existing_skill_names(db: Session, account_id: int) -> set[str]:
    """Casefolded skill names this account already has, across every
    source GET /api/skills unions (app/api/skills.py's _load_groups):
    project evidence, experience evidence, and freestanding manual rows.
    """
    names: set[str] = set()
    for (skill,) in db.execute(
        select(SkillEvidence.skill)
        .join(Repository, SkillEvidence.repo_id == Repository.id)
        .where(Repository.account_id == account_id)
    ):
        names.add(skill.strip().casefold())
    for (skill,) in db.execute(
        select(ExperienceSkillEvidence.skill)
        .join(Experience, ExperienceSkillEvidence.experience_id == Experience.id)
        .where(Experience.account_id == account_id)
    ):
        names.add(skill.strip().casefold())
    for (name,) in db.execute(select(Skill.name).where(Skill.account_id == account_id)):
        names.add(name.strip().casefold())
    return names


def _merge_skills(db: Session, account_id: int, tags: list[str]) -> int:
    existing = _existing_skill_names(db, account_id)
    added = 0
    for tag in tags:
        name = tag.strip()
        if not name or name.casefold() in existing:
            continue
        db.add(Skill(account_id=account_id, name=name))
        existing.add(name.casefold())  # this pass's own duplicates count once too
        added += 1
    if added:
        db.commit()
    return added


def _merge_experiences(
    db: Session, account_id: int, claims: list[ExperienceClaim]
) -> tuple[int, int]:
    existing_rows = list(
        db.execute(select(Experience).where(Experience.account_id == account_id)).scalars()
    )
    by_key = {
        (row.company.strip().casefold(), row.title.strip().casefold()): row
        for row in existing_rows
    }

    added = 0
    enriched = 0
    for claim in claims:
        key = (claim.company.casefold(), claim.title.casefold())
        existing = by_key.get(key)
        if existing is None:
            row = Experience(
                account_id=account_id,
                title=claim.title,
                company=claim.company,
                start_date=claim.start_date,
                end_date=claim.end_date,
            )
            db.add(row)
            db.flush()  # this pass's own duplicate claims should match, not double-add
            by_key[key] = row
            added += 1
            continue

        changed = False
        if existing.start_date is None and claim.start_date is not None:
            existing.start_date = claim.start_date
            changed = True
        if existing.end_date is None and claim.end_date is not None:
            existing.end_date = claim.end_date
            changed = True
        if changed:
            enriched += 1

    if added or enriched:
        db.commit()
    return added, enriched


def merge_resume_into_profile(
    db: Session, account_id: int, extraction: ResumeExtraction
) -> ProfileMergeSummary:
    """Runs both merges against one just-succeeded extraction. Safe to
    call again on the same extraction (reprocessing a resume re-merges
    it): every duplicate check is against the account's current state,
    so a second pass adds nothing new for what already made it in, and
    can still enrich a date that got filled in elsewhere since.
    """
    skills_added = _merge_skills(db, account_id, extraction.tags)
    experiences_added, experiences_enriched = _merge_experiences(
        db, account_id, extraction.experiences
    )
    return ProfileMergeSummary(
        skills_added=skills_added,
        experiences_added=experiences_added,
        experiences_enriched=experiences_enriched,
    )
