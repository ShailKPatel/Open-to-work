"""Resume library: upload any number of resume files per account, each with
LLM-extracted tags/target-roles/summary/experience/education
(app/profile/resume_extract.py)
plus freeform notes, all independently editable by hand afterward.
`/portfolio/resume` tab: a resume is drawn from the same projects/skills/
experience/education the rest of the Portfolio section holds.

Distinct from Account.resume_filename/resume_path, the single-file mirror
kept for backward compatibility (app/core/db/models.py's Account docstring);
nothing here reads from those two columns, app/profile/resume_ingest.py
writes to them as a side effect of every upload so they stay in sync.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.core.db import (
    Account,
    Education,
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    JobPosting,
    Resume,
    ResumeProfileLink,
    Skill,
)
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMProviderError,
    LLMRateLimitedError,
)
from app.core.settings import get_settings
from app.profile.resume_ingest import DuplicateResumeError, ingest_resume, run_extraction
from app.resume_build.checkpoint import progress_steps
from app.resume_build.compile import CompileError, TectonicNotInstalledError
from app.resume_build.orchestrator import build_resume_data_from_seed, edit_resume_content
from app.resume_build.pagefit import PageFitNotAchievedError, fit_to_page_limit

router = APIRouter(prefix="/api/resume")
logger = logging.getLogger(__name__)

_DEFAULT_MAX_PAGES = {"onepage": 1, "twopage": 2}


def _map_llm_error(e: Exception) -> HTTPException:
    """Same mapping as app/api/resume_build.py's own _map_llm_error, for
    the AI-edit endpoint below, which goes through the same
    orchestrator/pagefit LLM calls."""
    if isinstance(e, ApiKeyMissingError):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, BudgetExceededError):
        return HTTPException(status_code=402, detail=str(e))
    if isinstance(e, LLMRateLimitedError):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, LLMProviderError):
        return HTTPException(status_code=502, detail=str(e))
    return HTTPException(status_code=502, detail=f"resume edit failed: {e}")


class ResumeItem(BaseModel):
    id: int
    account_id: int
    filename: str
    name: str | None
    mime_type: str
    file_size: int
    uploaded_at: dt.datetime
    notes: str | None
    tags: list[str]
    target_roles: list[str]
    summary: str | None
    experiences: list[dict]
    education: list[dict]
    contact: dict
    extraction_status: str
    extraction_error: str | None
    extracted_at: dt.datetime | None
    job_posting_id: int | None
    template: str | None
    has_original_file: bool
    has_ai_edited_version: bool
    compiled_at: dt.datetime | None
    # A generated resume whose build stopped partway: what it had done,
    # step by step (app/resume_build/checkpoint.py), where it stopped and
    # why. POST /api/resume-build/retry/{id} picks it up from there.
    build_incomplete: bool = False
    build_progress: list[dict] = []
    build_stopped_at: str | None = None
    build_error: str | None = None
    build_attempts: int = 0

    @classmethod
    def from_row(cls, row: Resume) -> ResumeItem:
        state = row.build_state_json or None
        return cls(
            id=row.id,
            account_id=row.account_id,
            filename=row.filename,
            name=row.name,
            mime_type=row.mime_type,
            file_size=row.file_size,
            uploaded_at=row.uploaded_at,
            notes=row.notes,
            tags=row.tags_json or [],
            target_roles=row.target_roles_json or [],
            summary=row.summary,
            experiences=row.experiences_json or [],
            education=row.education_json or [],
            contact=row.contact_json or {},
            extraction_status=row.extraction_status,
            extraction_error=row.extraction_error,
            extracted_at=row.extracted_at,
            job_posting_id=row.job_posting_id,
            template=row.template,
            has_original_file=bool(row.stored_path),
            has_ai_edited_version=bool(row.compiled_path),
            compiled_at=row.compiled_at,
            build_incomplete=state is not None,
            build_progress=progress_steps(state),
            build_stopped_at=(state or {}).get("stopped_at"),
            build_error=(state or {}).get("error"),
            build_attempts=int((state or {}).get("attempts") or 0),
        )


@router.get("", response_model=list[ResumeItem])
def list_resumes(account_id: int, *, db: DbSession) -> list[ResumeItem]:
    rows = db.execute(
        select(Resume)
        .where(Resume.account_id == account_id)
        .order_by(Resume.uploaded_at.desc())
    ).scalars()
    return [ResumeItem.from_row(r) for r in rows]


class ResumeSearchHit(BaseModel):
    resume: ResumeItem
    score: float
    # Share of the posting's required skills this resume covers (exact
    # match full credit, close match half, see app/resume_build/
    # skill_match.py). None when the posting has no extracted skills.
    match_pct: int | None = None
    matched: list[str] = []
    related: list[str] = []
    missing: list[str] = []


def _resume_skills(row: Resume) -> list[str]:
    """Skills a resume shows: its extracted tags, plus a generated
    resume's own skills list (the same thing, kept in two places)."""
    seen: dict[str, str] = {}
    content = row.content_json or {}
    for name in [*(row.tags_json or []), *(content.get("skills") or [])]:
        text = str(name).strip()
        if text:
            seen.setdefault(text.casefold(), text)
    return list(seen.values())


@router.get("/search", response_model=list[ResumeSearchHit])
def search_resumes_for_posting(
    account_id: int, job_posting_id: int, top_k: int = 3,
    *,
    db: DbSession,
) -> list[ResumeSearchHit]:
    """"Closest existing resume to this job" for /portfolio/resume/build's
    suggestion panel. When the posting has extracted required skills,
    every resume in the library is scored by how many of them it covers
    (match_pct) and ranked by that, semantic similarity of the posting's
    text to the resume breaking ties. Without extracted skills there is
    no percentage to give, so it falls back to semantic search alone
    (app/retrieval/search.py's search_resumes(), account-scoped). Empty
    list, not an error, when the library is empty.
    """
    from app.profile.job_extract import parse_skills_required
    from app.resume_build.skill_match import coverage_pct, match_requirements
    from app.retrieval.search import search_resumes

    posting = db.get(JobPosting, job_posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={job_posting_id}")

    required = [
        item["skill"]
        for item in parse_skills_required((posting.extracted_json or {}).get("skills_required", []))
    ]
    # An unfinished build has no content to match yet.
    rows = list(
        db.execute(
            select(Resume).where(
                Resume.account_id == account_id, Resume.build_state_json.is_(None)
            )
        ).scalars()
    )
    if not rows:
        return []
    try:
        semantic = {
            h.id: h.score
            for h in search_resumes(posting.raw_text_quarantined, account_id, top_k=len(rows))
        }
    except Exception:
        logger.exception("resume semantic search failed; ranking by skills only")
        semantic = {}

    hits: list[ResumeSearchHit] = []
    for row in rows:
        if required:
            matches = match_requirements(required, _resume_skills(row))
            hit = ResumeSearchHit(
                resume=ResumeItem.from_row(row),
                score=semantic.get(row.id, 0.0),
                match_pct=coverage_pct(matches),
                matched=[m.required for m in matches if m.kind == "exact"],
                related=[m.required for m in matches if m.kind == "related"],
                missing=[m.required for m in matches if m.kind == "missing"],
            )
        elif row.id in semantic:
            hit = ResumeSearchHit(resume=ResumeItem.from_row(row), score=semantic[row.id])
        else:
            continue
        hits.append(hit)

    hits.sort(key=lambda h: (h.match_pct or 0, h.score), reverse=True)
    return hits[:top_k]


@router.post("", response_model=ResumeItem)
def upload_resume(
    account_id: int = Form(...),
    name: str = Form(""),
    notes: str = Form(""),
    file: UploadFile = File(...),
    *,
    db: DbSession,
) -> ResumeItem:
    if not file.filename:
        raise HTTPException(status_code=422, detail="a file is required")

    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
    try:
        row = ingest_resume(db, account_id, file, name=name, notes=notes)
    except DuplicateResumeError as e:
        label = e.existing.name or e.existing.filename
        raise HTTPException(
            status_code=409,
            detail=f'This file is already in your library as "{label}". '
            "Use Reprocess on it to read it again.",
        ) from e
    return ResumeItem.from_row(row)


@router.get("/{resume_id}/file")
def download_resume(
    resume_id: int,
    disposition: str = "attachment",
    *,
    db: DbSession,
) -> FileResponse:
    """`disposition=inline` (the view-in-a-popup path, resume.html's View
    button) lets the browser render a PDF/image straight in an
    iframe/img instead of forcing a save dialog; plain `attachment` (the
    default, an explicit Download click) forces the save dialog every
    browser respects. Anything else is rejected rather than passed
    through to FileResponse's own content_disposition_type unchecked.
    """
    if disposition not in ("attachment", "inline"):
        raise HTTPException(status_code=422, detail="disposition must be attachment or inline")
    row = db.get(Resume, resume_id)
    if row is None or not row.stored_path or not Path(row.stored_path).exists():
        raise HTTPException(status_code=404, detail=f"no resume file with id={resume_id}")
    return FileResponse(
        row.stored_path,
        filename=row.filename,
        media_type=row.mime_type,
        content_disposition_type=disposition,
    )


@router.get("/{resume_id}/compiled-file")
def download_compiled_resume(
    resume_id: int,
    disposition: str = "attachment",
    *,
    db: DbSession,
) -> FileResponse:
    """The AI-edited/generated PDF (Resume.compiled_path), a separate
    file from GET /{resume_id}/file's original upload. Same disposition
    handling as that endpoint, see its docstring."""
    if disposition not in ("attachment", "inline"):
        raise HTTPException(status_code=422, detail="disposition must be attachment or inline")
    row = db.get(Resume, resume_id)
    if row is None or not row.compiled_path or not Path(row.compiled_path).exists():
        raise HTTPException(status_code=404, detail=f"no compiled resume with id={resume_id}")
    return FileResponse(
        row.compiled_path,
        filename=row.filename or "resume.pdf",
        media_type="application/pdf",
        content_disposition_type=disposition,
    )


class ResumeUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    target_roles: list[str] | None = None
    summary: str | None = None


def _reindex_best_effort(row: Resume, resume_id: int) -> None:
    """Manual edits to what's searchable should be reflected in Qdrant too,
    best-effort, same reasoning as accounts.py's _delete_qdrant_points: a
    Qdrant hiccup shouldn't block saving someone's actual edit."""
    try:
        from app.retrieval.index import index_resume

        index_resume(row)
    except Exception:
        logger.exception("could not re-index resume id=%s; continuing", resume_id)


@router.patch("/{resume_id}", response_model=ResumeItem)
def update_resume(resume_id: int, body: ResumeUpdate, *, db: DbSession) -> ResumeItem:
    fields = body.model_dump(exclude_unset=True)
    row = db.get(Resume, resume_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    if "name" in fields:
        row.name = (fields["name"].strip() if fields["name"] else None) or None
    if "notes" in fields:
        row.notes = (fields["notes"].strip() if fields["notes"] else None) or None
    if "tags" in fields:
        row.tags_json = [t.strip() for t in (fields["tags"] or []) if t.strip()]
    if "target_roles" in fields:
        row.target_roles_json = [t.strip() for t in (fields["target_roles"] or []) if t.strip()]
    if "summary" in fields:
        row.summary = (fields["summary"].strip() if fields["summary"] else None) or None
    db.commit()
    db.refresh(row)

    _reindex_best_effort(row, resume_id)
    return ResumeItem.from_row(row)


@router.post("/{resume_id}/reprocess", response_model=ResumeItem)
def reprocess_resume(resume_id: int, *, db: DbSession) -> ResumeItem:
    row = db.get(Resume, resume_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    if not row.stored_path or not Path(row.stored_path).exists():
        raise HTTPException(status_code=409, detail="the resume's file is missing on disk")
    row = run_extraction(db, row)
    return ResumeItem.from_row(row)


class LinkedPoint(BaseModel):
    id: int
    text: str
    created: bool


class LinkedExperienceSkill(BaseModel):
    id: int
    skill: str
    created: bool


class LinkedExperience(BaseModel):
    id: int
    title: str
    company: str
    created: bool
    points: list[LinkedPoint] = []
    skills: list[LinkedExperienceSkill] = []


class LinkedEducation(BaseModel):
    id: int
    institution: str
    degree: str
    created: bool


class LinkedSkill(BaseModel):
    id: int
    name: str
    created: bool


class ResumeProfile(BaseModel):
    """What this resume put into (or matched in) the profile, read from
    its ResumeProfileLink rows. created is true for a row this resume
    added and false for one that already existed. Freestanding skills are
    only the resume's skills not tied to any role; a skill tied to a role
    is under that experience.
    """

    experiences: list[LinkedExperience]
    education: list[LinkedEducation]
    skills: list[LinkedSkill]


@router.get("/{resume_id}/profile", response_model=ResumeProfile)
def resume_profile(resume_id: int, *, db: DbSession) -> ResumeProfile:
    if db.get(Resume, resume_id) is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    created: dict[str, dict[int, bool]] = {}
    for link in db.execute(
        select(ResumeProfileLink)
        .where(ResumeProfileLink.resume_id == resume_id)
        .order_by(ResumeProfileLink.id)
    ).scalars():
        created.setdefault(link.kind, {})[link.ref_id] = link.created

    def ids(kind: str) -> list[int]:
        return list(created.get(kind, {}))

    experiences: dict[int, LinkedExperience] = {}
    for exp in db.execute(
        select(Experience).where(Experience.id.in_(ids("experience")))
    ).scalars():
        experiences[exp.id] = LinkedExperience(
            id=exp.id, title=exp.title, company=exp.company,
            created=created["experience"][exp.id],
        )
    for point in db.execute(
        select(ExperiencePoint)
        .where(ExperiencePoint.id.in_(ids("experience_point")))
        .order_by(ExperiencePoint.order_index)
    ).scalars():
        if point.experience_id in experiences:
            experiences[point.experience_id].points.append(
                LinkedPoint(
                    id=point.id, text=point.text,
                    created=created["experience_point"][point.id],
                )
            )
    for evidence in db.execute(
        select(ExperienceSkillEvidence).where(
            ExperienceSkillEvidence.id.in_(ids("experience_skill"))
        )
    ).scalars():
        if evidence.experience_id in experiences:
            experiences[evidence.experience_id].skills.append(
                LinkedExperienceSkill(
                    id=evidence.id, skill=evidence.skill,
                    created=created["experience_skill"][evidence.id],
                )
            )

    education = [
        LinkedEducation(
            id=row.id, institution=row.institution, degree=row.degree,
            created=created["education"][row.id],
        )
        for row in db.execute(
            select(Education).where(Education.id.in_(ids("education")))
        ).scalars()
    ]
    skills = [
        LinkedSkill(id=row.id, name=row.name, created=created["skill"][row.id])
        for row in db.execute(
            select(Skill).where(Skill.id.in_(ids("skill"))).order_by(Skill.name)
        ).scalars()
    ]
    order = {ref_id: i for i, ref_id in enumerate(ids("experience"))}
    return ResumeProfile(
        experiences=sorted(experiences.values(), key=lambda e: order[e.id]),
        education=education,
        skills=skills,
    )


class ResumeEditRequest(BaseModel):
    message: str


def _job_text_for_edit(resume: Resume, db: Session) -> str:
    """The text this resume's content is (or, for the first edit, will
    be) grounded against: a linked JobPosting's own text when this
    resume came from /portfolio/resume/build, otherwise the resume's own current
    content_json summary/skills once it has one (kept representative as
    it evolves edit to edit), otherwise (the first edit of a plain upload,
    before content_json exists) the extracted
    summary/tags/target_roles from its own upload (app/profile/
    resume_extract.py), same fields app/retrieval/index.py's
    _resume_text() already embeds this row under.
    """
    if resume.job_posting_id is not None:
        posting = db.get(JobPosting, resume.job_posting_id)
        if posting is not None:
            return posting.raw_text_quarantined
    if resume.content_json:
        c = resume.content_json
        parts = [c.get("summary") or ""]
        if c.get("skills"):
            parts.append(", ".join(c["skills"]))
        return "\n".join(p for p in parts if p)
    parts = [resume.summary or ""]
    if resume.tags_json:
        parts.append(", ".join(resume.tags_json))
    if resume.target_roles_json:
        parts.append(", ".join(resume.target_roles_json))
    return "\n".join(p for p in parts if p)


@router.post("/{resume_id}/edit", response_model=ResumeItem)
def edit_resume(resume_id: int, body: ResumeEditRequest, *, db: DbSession) -> ResumeItem:
    """Prompt-driven edit: the account holder writes a message describing
    what to change, an LLM updates the resume's summary/projects/skills
    to match (app/resume_build/orchestrator.py's edit_resume_content()),
    grounded against this account's own real projects/skills exactly
    like generating one from scratch. experience/education/header are
    never touched (rebuilt fresh from the account's own data inside
    edit_resume_content(), not sent to the LLM at all), and the compiled
    template is fixed (whatever this resume already used, "onepage" by
    default for a plain upload's first edit), so "facts change, layout
    doesn't" holds structurally, not just by prompt instruction.

    A plain uploaded file that's never been edited before (content_json
    still null) gets adopted first: build_resume_data_from_seed() builds
    an initial structured version from its own extraction, seeded from
    the account's real data the same grounded way, before this message's
    edit is applied on top. The original uploaded file (stored_path) is
    never touched by any of this; only compiled_path, a separate
    artifact, changes.
    """
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    row = db.get(Resume, resume_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    if row.build_state_json:
        raise HTTPException(
            status_code=409, detail="this resume's build has not finished; retry it first"
        )

    job_text = _job_text_for_edit(row, db)
    template = row.template or "onepage"

    try:
        if row.content_json is None:
            if not job_text.strip():
                raise HTTPException(
                    status_code=409,
                    detail="this resume hasn't been analyzed yet; reprocess it first",
                )
            base_content = build_resume_data_from_seed(row.account_id, job_text, template)
        else:
            base_content = row.content_json

        new_content = edit_resume_content(
            row.account_id, job_text, base_content, message, template=template
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (
        ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError
    ) as e:
        raise _map_llm_error(e) from e

    max_pages = _DEFAULT_MAX_PAGES[template]
    try:
        result = fit_to_page_limit(new_content, template, max_pages, account_id=row.account_id)
        pdf_bytes = result.pdf_bytes
        new_content = result.data or new_content
        if not result.fit_exact:
            logger.warning(
                "page-fit came up short for resume id=%s: %s page(s) against a target of %s",
                resume_id, result.page_count, max_pages,
            )
    except PageFitNotAchievedError as e:
        logger.warning("page-fit did not reach target for resume id=%s: %s", resume_id, e)
        pdf_bytes = e.best_pdf_bytes
        new_content = e.best_data or new_content
    except TectonicNotInstalledError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except CompileError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except (
        ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError
    ) as e:
        raise _map_llm_error(e) from e

    # The held-back reserve content the page-fit loop draws on is
    # working state, never part of the resume's saved content.
    new_content.pop("reserve", None)

    account_dir = Path(get_settings().resume_storage_dir) / str(row.account_id)
    account_dir.mkdir(parents=True, exist_ok=True)
    dest = account_dir / f"{row.id}_edited.pdf"
    dest.write_bytes(pdf_bytes)

    row.content_json = new_content
    row.template = template
    row.compiled_path = str(dest)
    row.compiled_at = dt.datetime.now(dt.UTC)
    row.summary = new_content.get("summary")
    row.tags_json = new_content.get("skills", [])
    if row.extraction_status != "extracted":
        row.extraction_status = "extracted"
    db.commit()
    db.refresh(row)

    _reindex_best_effort(row, resume_id)
    return ResumeItem.from_row(row)


def _delete_qdrant_point(resume_id: int) -> None:
    """Best-effort, same reasoning as accounts.py's _delete_qdrant_points:
    SQLite is the source of truth, Qdrant is a derived index."""
    try:
        from app.retrieval.index import RESUME_COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(RESUME_COLLECTION):
            client.delete(collection_name=RESUME_COLLECTION, points_selector=[resume_id])
    except Exception:
        logger.exception("could not clean up Qdrant point for deleted resume; continuing")


@router.delete("/{resume_id}")
def delete_resume(resume_id: int, *, db: DbSession) -> dict:
    row = db.get(Resume, resume_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    _delete_qdrant_point(resume_id)
    if row.stored_path:
        Path(row.stored_path).unlink(missing_ok=True)
    if row.compiled_path:
        Path(row.compiled_path).unlink(missing_ok=True)
    # SQLite runs without foreign key enforcement here, so the ON DELETE
    # SET NULL on skills.source_resume_id has to be done by hand. The
    # skills themselves stay; they just stop naming a resume.
    db.execute(
        update(Skill).where(Skill.source_resume_id == resume_id).values(source_resume_id=None)
    )
    db.execute(delete(ResumeProfileLink).where(ResumeProfileLink.resume_id == resume_id))
    db.delete(row)
    db.commit()
    return {"deleted": True, "id": resume_id}
