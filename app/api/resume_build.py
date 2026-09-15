"""Generate-a-resume endpoints: paste a job posting in, get a compiled PDF
tailored against it, back to app/resume_build/orchestrator.py (data
selection + LLM content), app/resume_build/pagefit.py (compile + page-fit
loop), app/resume_build/compile.py (Tectonic). Separate router/prefix
from app/api/resume.py: that one is the resume library (documents an
account already has); this generates a new one, a different resource,
following the same one-resource-per-router split as app/api/sources.py
vs app/api/accounts.py.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from app.core.db import JobPosting, Resume, get_db
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMProviderError,
    LLMRateLimitedError,
)
from app.core.settings import get_settings
from app.resume_build.compile import CompileError, TectonicNotInstalledError
from app.resume_build.orchestrator import build_resume_data, generate_resume
from app.resume_build.pagefit import PageFitNotAchievedError, fit_to_page_limit

router = APIRouter(prefix="/api/resume-build")
logger = logging.getLogger(__name__)


def _save_generated_resume(
    account_id: int, job_posting_id: int, template: str, data: dict, pdf_bytes: bytes
) -> int | None:
    """Lands every generated resume in the shared library (/resume), so
    it's viewable/editable there like any uploaded file, rather than the
    PDF only ever existing as a one-off download. Best-effort: a failure
    here (disk, DB) shouldn't turn a successful generation into an error
    response, the caller already has the PDF bytes to hand back either
    way. Same posture as index_resume()'s other callers.
    """
    db = get_db()
    try:
        posting = db.get(JobPosting, job_posting_id)
        title_bits = [
            b for b in [posting.title if posting else None, posting.company if posting else None]
            if b
        ]
        filename = (" - ".join(title_bits) or "resume") + ".pdf"

        row = Resume(
            account_id=account_id,
            filename=filename,
            mime_type="application/pdf",
            file_size=len(pdf_bytes),
            job_posting_id=job_posting_id,
            template=template,
            content_json=data,
            # Mirrors content_json's own summary/skills into the same
            # columns an uploaded file's extraction would populate, so
            # this row is indistinguishable from an uploaded one to
            # index_resume() (app/retrieval/index.py, embeds
            # summary/tags_json/target_roles_json, not content_json) and
            # to the rest of /resume's UI.
            summary=data.get("summary"),
            tags_json=data.get("skills", []),
            extraction_status="extracted",
            extracted_at=dt.datetime.now(dt.UTC),
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        account_dir = Path(get_settings().resume_storage_dir) / str(account_id)
        account_dir.mkdir(parents=True, exist_ok=True)
        dest = account_dir / f"{row.id}_generated.pdf"
        dest.write_bytes(pdf_bytes)
        row.compiled_path = str(dest)
        row.compiled_at = dt.datetime.now(dt.UTC)
        db.commit()

        try:
            from app.retrieval.index import index_resume

            index_resume(row)
        except Exception:
            logger.exception("could not index generated resume id=%s; continuing", row.id)

        return row.id
    except Exception:
        logger.exception("could not save generated resume to the library; continuing")
        return None
    finally:
        db.close()

_VALID_TEMPLATES = {"onepage", "twopage"}
_DEFAULT_MAX_PAGES = {"onepage": 1, "twopage": 2}


def _validate_template(template: str) -> None:
    if template not in _VALID_TEMPLATES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown template {template!r}, must be one of {sorted(_VALID_TEMPLATES)}",
        )


def _map_llm_error(e: Exception) -> HTTPException:
    if isinstance(e, ApiKeyMissingError):
        return HTTPException(status_code=422, detail=str(e))
    if isinstance(e, BudgetExceededError):
        return HTTPException(status_code=402, detail=str(e))
    if isinstance(e, LLMRateLimitedError):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, LLMProviderError):
        return HTTPException(status_code=502, detail=str(e))
    return HTTPException(status_code=502, detail=f"resume generation failed: {e}")


class ResumeBuildOptionsResponse(BaseModel):
    emails: list[dict[str, Any]]
    phones: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    experience: list[dict[str, Any]]


@router.get("/options", response_model=ResumeBuildOptionsResponse)
def get_resume_build_options(account_id: int, job_posting_id: int | None = None) -> ResumeBuildOptionsResponse:
    from sqlalchemy import select
    from app.core.db import Account, ContactEmail, ContactPhone, JobPosting, Repository, get_db
    from app.resume_build.context import build_experience_context
    from app.resume_build.orchestrator import _candidate_projects, _candidate_skills

    db = get_db()
    try:
        account = db.get(Account, account_id)
        if account is None:
            raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
        
        posting = db.get(JobPosting, job_posting_id) if job_posting_id is not None else None

        emails_rows = list(
            db.execute(
                select(ContactEmail)
                .where(ContactEmail.account_id == account_id)
                .order_by(ContactEmail.created_at)
            ).scalars()
        )
        emails = [{"id": e.id, "email": e.email, "is_primary": e.is_primary} for e in emails_rows]
        if not emails and account.contact_email:
            emails = [{"id": 0, "email": account.contact_email, "is_primary": True}]

        phones_rows = list(
            db.execute(
                select(ContactPhone)
                .where(ContactPhone.account_id == account_id)
                .order_by(ContactPhone.created_at)
            ).scalars()
        )
        phones = [{"id": p.id, "phone": p.phone, "is_primary": p.is_primary} for p in phones_rows]
        if not phones and account.contact_phone:
            phones = [{"id": 0, "phone": account.contact_phone, "is_primary": True}]

        cand_projects = _candidate_projects(db, account_id, posting.raw_text_quarantined) if posting else []
        recommended_repo_ids = {c["repo_id"] for c in cand_projects[:4]}

        all_repos = list(
            db.execute(
                select(Repository).where(Repository.account_id == account_id).order_by(Repository.starred.desc(), Repository.name)
            ).scalars()
        )
        projects = []
        for r in all_repos:
            is_rec = r.id in recommended_repo_ids
            projects.append({
                "repo_id": r.id,
                "name": r.name,
                "description": r.description or "",
                "url": r.url,
                "starred": r.starred,
                "recommended": is_rec,
            })

        cand_skills = _candidate_skills(account_id, posting.raw_text_quarantined) if posting else []
        rec_skills_set = set(cand_skills[:12])

        from app.api.skills import list_skills
        all_skills_data = list_skills(account_id)
        skills = []
        seen_skill_names = set()
        for sk_obj in all_skills_data:
            sk_name = sk_obj.get("name") if isinstance(sk_obj, dict) else getattr(sk_obj, "name", "")
            if sk_name and sk_name.casefold() not in seen_skill_names:
                seen_skill_names.add(sk_name.casefold())
                skills.append({
                    "name": sk_name,
                    "recommended": sk_name in rec_skills_set or sk_name.casefold() in {s.casefold() for s in cand_skills},
                })
        for c_sk in cand_skills:
            if c_sk.casefold() not in seen_skill_names:
                seen_skill_names.add(c_sk.casefold())
                skills.append({
                    "name": c_sk,
                    "recommended": True,
                })

        experience = build_experience_context(db, account_id)

        return ResumeBuildOptionsResponse(
            emails=emails,
            phones=phones,
            projects=projects,
            skills=skills,
            experience=experience,
        )
    finally:
        db.close()



class PreviewRequest(BaseModel):
    account_id: int
    job_posting_id: int
    template: str = "onepage"


class PreviewResponse(BaseModel):
    tex: str


@router.post("/preview", response_model=PreviewResponse)
def preview(body: PreviewRequest) -> PreviewResponse:
    """Fast path: LLM content generation only, no compile, no page-fit
    loop. Lets the UI show the tailored content (or a raw .tex download)
    without needing a working Tectonic install in every environment this
    runs in.
    """
    _validate_template(body.template)
    try:
        tex = generate_resume(body.account_id, body.job_posting_id, template=body.template)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError) as e:
        raise _map_llm_error(e) from e
    return PreviewResponse(tex=tex)


class GenerateRequest(BaseModel):
    account_id: int
    job_posting_id: int
    template: str = "onepage"
    max_pages: int | None = None
    selected_email: str | None = None
    selected_phone: str | None = None
    selected_project_ids: list[int] | None = None
    project_instructions: dict[int, str] | None = None
    selected_skills: list[str] | None = None
    selected_experience_ids: list[int] | None = None
    custom_instruction: str | None = None


@router.post("/generate")
def generate(body: GenerateRequest) -> Response:
    """Full pipeline: orchestrator content, then compile + the page-fit
    loop, returns the PDF bytes directly (Content-Type: application/pdf),
    same "serve the raw file" convention app/api/resume.py's
    GET /{id}/file already uses.
    """
    _validate_template(body.template)
    max_pages = body.max_pages or _DEFAULT_MAX_PAGES[body.template]

    try:
        data = build_resume_data(
            body.account_id,
            body.job_posting_id,
            template=body.template,
            selected_email=body.selected_email,
            selected_phone=body.selected_phone,
            selected_project_ids=body.selected_project_ids,
            project_instructions=body.project_instructions,
            selected_skills=body.selected_skills,
            selected_experience_ids=body.selected_experience_ids,
            custom_instruction=body.custom_instruction,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError) as e:
        raise _map_llm_error(e) from e

    try:
        result = fit_to_page_limit(data, body.template, max_pages, account_id=body.account_id)
        pdf_bytes = result.pdf_bytes
        page_count = result.page_count
        achieved = True
    except PageFitNotAchievedError as e:
        logger.warning("page-fit did not reach target for account_id=%s: %s", body.account_id, e)
        pdf_bytes = e.best_pdf_bytes
        page_count = e.best_page_count
        achieved = False
    except TectonicNotInstalledError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except CompileError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError) as e:
        raise _map_llm_error(e) from e

    resume_id = _save_generated_resume(
        body.account_id, body.job_posting_id, body.template, data, pdf_bytes
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=resume.pdf",
            "X-Page-Fit-Achieved": "true" if achieved else "false",
            "X-Page-Count": str(page_count),
            "X-Resume-Id": str(resume_id) if resume_id is not None else "",
        },
    )

