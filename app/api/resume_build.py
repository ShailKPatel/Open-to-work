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

import copy
import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.api.llm_errors import llm_http_error
from app.api.skills import active_skill_groups
from app.core.db import JobPosting, Resume, get_db
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMProviderError,
    LLMRateLimitedError,
)
from app.core.pipeline import Pipeline
from app.core.settings import get_settings
from app.resume_build import background
from app.resume_build.compile import CompileError, TectonicNotInstalledError
from app.resume_build.orchestrator import build_resume_data, generate_resume
from app.resume_build.pagefit import PageFitNotAchievedError, fit_to_page_limit
from app.retrieval.search import query_for_posting

router = APIRouter(prefix="/api/resume-build")
logger = logging.getLogger(__name__)


def _library_filename(db: Session, job_posting_id: int) -> str:
    posting = db.get(JobPosting, job_posting_id)
    title_bits = [
        b for b in [posting.title if posting else None, posting.company if posting else None]
        if b
    ]
    return (" - ".join(title_bits) or "resume") + ".pdf"


def _save_generated_resume(
    account_id: int,
    job_posting_id: int,
    template: str,
    data: dict,
    pdf_bytes: bytes,
    name: str | None = None,
    resume_id: int | None = None,
) -> int | None:
    """Lands every generated resume in the shared library (/resume), so
    it's viewable/editable there like any uploaded file, rather than the
    PDF only ever existing as a one-off download. Best-effort: a failure
    here (disk, DB) shouldn't turn a successful generation into an error
    response, the caller already has the PDF bytes to hand back either
    way. Same posture as index_resume()'s other callers.

    resume_id is the incomplete row a retried build finishes: it is
    filled in and marked finished rather than joined by a second row.
    """
    db = get_db()
    try:
        row = db.get(Resume, resume_id) if resume_id is not None else None
        if row is None:
            row = Resume(
                account_id=account_id,
                filename=_library_filename(db, job_posting_id),
                name=name,
                mime_type="application/pdf",
                job_posting_id=job_posting_id,
            )
            db.add(row)
        row.file_size = len(pdf_bytes)
        row.template = template
        row.content_json = data
        # Mirrors content_json's own summary/skills into the same
        # columns an uploaded file's extraction would populate, so
        # this row is indistinguishable from an uploaded one to
        # index_resume() (app/retrieval/index.py, embeds
        # summary/tags_json/target_roles_json, not content_json) and
        # to the rest of /resume's UI.
        row.summary = data.get("summary")
        row.tags_json = data.get("skills", [])
        row.extraction_status = "extracted"
        row.extraction_error = None
        row.extracted_at = dt.datetime.now(dt.UTC)
        row.build_state_json = None
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


def _save_incomplete_resume(
    body: GenerateRequest,
    checkpoint: dict[str, Any],
    stopped_at: str,
    error: str,
    resume_id: int | None = None,
    error_kind: str | None = None,
) -> int | None:
    """Keeps a build that stopped partway in the library as an incomplete
    resume, holding what it had done (Resume.build_state_json), so a
    retry starts from there instead of from nothing. A retry that stops
    again updates the same row. Best-effort like _save_generated_resume:
    failing to save the checkpoint must not hide the build's own error.
    """
    db = get_db()
    try:
        row = db.get(Resume, resume_id) if resume_id is not None else None
        if row is None:
            row = Resume(
                account_id=body.account_id,
                filename=_library_filename(db, body.job_posting_id),
                name=(body.name or "").strip() or None,
                mime_type="application/pdf",
                job_posting_id=body.job_posting_id,
                template=body.template,
            )
            db.add(row)
        attempts = int((row.build_state_json or {}).get("attempts") or 0) + 1
        row.extraction_status = "pending"
        row.extraction_error = None
        row.build_state_json = {
            "request": body.model_dump(mode="json"),
            "data": checkpoint.get("data"),
            "reworded": bool(checkpoint.get("reworded")),
            "cuts_made": int(checkpoint.get("cuts_made") or 0),
            "stopped_at": stopped_at,
            "error": error,
            "error_kind": error_kind,
            "stopped_time": dt.datetime.now(dt.UTC).isoformat(),
            "attempts": attempts,
        }
        db.commit()
        return row.id
    except Exception:
        logger.exception("could not save the unfinished build to the library; continuing")
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


_MAX_HEADER_CHARS = 400


def _header_safe(text: str) -> str:
    """One line, bounded, for an HTTP response header. A header value
    cannot carry the newlines these messages use to list one key per
    line, and an over-long one gets dropped by proxies."""
    flat = " ".join(text.split())
    return flat if len(flat) <= _MAX_HEADER_CHARS else flat[: _MAX_HEADER_CHARS - 3] + "..."


def _map_llm_error(e: Exception) -> HTTPException:
    return llm_http_error(e, "resume generation failed")


class ResumeBuildOptionsResponse(BaseModel):
    emails: list[dict[str, Any]]
    phones: list[dict[str, Any]]
    projects: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    experience: list[dict[str, Any]]
    education: list[dict[str, Any]]
    # The posting's own required skills, and the ones this account has
    # neither an exact nor a related skill for.
    required_skills: list[str] = []
    missing_required: list[str] = []


def _skill_options(
    db: Session, account_id: int, posting: JobPosting | None
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Every skill the account has, each tagged with a tier saying why it
    is (or is not) preselected for this posting:

    - "exact": the posting asks for it by name (app/resume_build/
      skill_match.py's skill_key folding). Preselected.
    - "related": close to a requirement the account has no exact match
      for, or among the account's skill evidence that reads closest to the
      posting's text. Not preselected on its own: the page sends these to
      POST /skill-review and selects the ones the model keeps.
    - "other": everything else, available to pick by hand.

    A posting with no extracted skills has nothing to match exactly, so
    the evidence search alone decides, preselected as before.
    """
    from app.profile.job_extract import parse_skills_required, skill_key
    from app.resume_build.orchestrator import _candidate_skills
    from app.resume_build.skill_match import suggest_related

    # Archived skills, by hand or because every project and role behind
    # them is archived, never reach the picker.
    names: dict[str, str] = {}
    for group in active_skill_groups(db, account_id):
        names.setdefault(group.name.casefold(), group.name)
    have = list(names.values())

    required: list[str] = []
    cand_skills: list[str] = []
    if posting is not None:
        required = [
            item["skill"]
            for item in parse_skills_required(
                (posting.extracted_json or {}).get("skills_required", [])
            )
        ]
        cand_skills = _candidate_skills(db, account_id, query_for_posting(posting))
    for name in cand_skills:
        names.setdefault(name.casefold(), name)

    required_keys = {skill_key(r) for r in required}
    exact = {n for n in names.values() if skill_key(n) in required_keys}
    suggestions = suggest_related(required, have, exclude=exact) if required else {}
    for name in cand_skills:
        if name not in exact:
            suggestions.setdefault(name, [])

    skills: list[dict[str, Any]] = []
    for name in names.values():
        if name in exact:
            tier = "exact"
        elif name in suggestions:
            tier = "related"
        else:
            tier = "other"
        skills.append({
            "name": name,
            "tier": tier,
            "suggested_for": suggestions.get(name, []),
            "recommended": tier == "exact" or (tier == "related" and not required),
        })
    order = {"exact": 0, "related": 1, "other": 2}
    skills.sort(key=lambda sk: order[sk["tier"]])

    covered = {skill_key(n) for n in exact}
    for reqs in suggestions.values():
        covered.update(skill_key(r) for r in reqs)
    missing = [r for r in required if skill_key(r) not in covered]
    return skills, required, missing


@router.get("/options", response_model=ResumeBuildOptionsResponse)
def get_resume_build_options(
    account_id: int, job_posting_id: int | None = None
,
    *,
    db: DbSession,) -> ResumeBuildOptionsResponse:
    from sqlalchemy import select

    from app.core.db import Account, ContactEmail, ContactPhone, JobPosting, Repository
    from app.resume_build.context import build_education_context, build_experience_context
    from app.resume_build.orchestrator import _candidate_projects

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

    cand_projects = (
        _candidate_projects(db, account_id, query_for_posting(posting)) if posting else []
    )
    recommended_repo_ids = {c["repo_id"] for c in cand_projects[:4]}

    all_repos = list(
        db.execute(
            select(Repository)
            .where(
                Repository.account_id == account_id,
                Repository.is_profile_readme.is_(False),
                Repository.exclude_from_resume.is_(False),
            )
            .order_by(Repository.starred.desc(), Repository.name)
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

    skills, required_skills, missing_required = _skill_options(db, account_id, posting)
    experience = build_experience_context(db, account_id)

    return ResumeBuildOptionsResponse(
        emails=emails,
        phones=phones,
        projects=projects,
        skills=skills,
        experience=experience,
        education=build_education_context(db, account_id),
        required_skills=required_skills,
        missing_required=missing_required,
    )


class SkillReviewItem(BaseModel):
    name: str
    suggested_for: list[str] = []


class SkillReviewRequest(BaseModel):
    account_id: int
    job_posting_id: int
    skills: list[SkillReviewItem]


class SkillVerdictOut(BaseModel):
    name: str
    keep: bool
    reason: str


@router.post("/skill-review", response_model=list[SkillVerdictOut])
def skill_review(body: SkillReviewRequest, *, db: DbSession) -> list[SkillVerdictOut]:
    """The model's keep/drop call on every "related" skill GET /options
    suggested, so a skill that is only similar by name ("Java" for a
    "JavaScript" job) does not get preselected. Only names the account
    actually has are sent, whatever the page posts. A skill the model
    gave no verdict for is simply absent from the reply.
    """
    from app.profile.job_extract import parse_skills_required
    from app.resume_build.skill_match import review_related

    posting = db.get(JobPosting, body.job_posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={body.job_posting_id}")
    known = {g.name.casefold() for g in active_skill_groups(db, body.account_id)}
    suggestions = {
        item.name: item.suggested_for
        for item in body.skills
        if item.name.casefold() in known
    }
    required = [
        item["skill"]
        for item in parse_skills_required((posting.extracted_json or {}).get("skills_required", []))
    ]
    try:
        verdicts = review_related(body.account_id, posting.title, required, suggestions)
    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError) as e:
        raise _map_llm_error(e) from e
    return [SkillVerdictOut(name=n, keep=v.keep, reason=v.reason) for n, v in verdicts.items()]



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
    selected_education_ids: list[int] | None = None
    custom_instruction: str | None = None
    # Most experience points per role; None picks by page count.
    points_per_role: int | None = None
    # What the library calls this resume. Left empty, the library shows
    # the "Title - Company.pdf" filename instead.
    name: str | None = None


@dataclass
class _BuildOutcome:
    pdf_bytes: bytes
    page_count: int
    max_pages: int
    achieved: bool
    fit_note: str | None
    resume_id: int | None
    match: dict[str, Any]


def _content_match(job_posting_id: int, skills: list[str]) -> dict[str, Any]:
    """How much of the posting's required skills the finished resume
    lists, scored the same way GET /api/resume/search scores the library
    (app/resume_build/skill_match.py), so a new resume and an old one
    read on one scale. match_pct is None when the posting names no
    skills."""
    from app.profile.job_extract import parse_skills_required
    from app.resume_build.skill_match import coverage_pct, match_requirements

    db = get_db()
    try:
        posting = db.get(JobPosting, job_posting_id)
        raw = (posting.extracted_json or {}).get("skills_required", []) if posting else []
    finally:
        db.close()
    required = [item["skill"] for item in parse_skills_required(raw)]
    matches = match_requirements(required, skills) if required else []
    return {
        "match_pct": coverage_pct(matches),
        "matched": [m.required for m in matches if m.kind == "exact"],
        "related": [m.required for m in matches if m.kind == "related"],
        "missing": [m.required for m in matches if m.kind == "missing"],
    }


_TAILOR_STEP = "tailoring the content to the job"


def _run_build(
    body: GenerateRequest,
    on_stage: Callable[[str], None] = lambda _: None,
    checkpoint: dict[str, Any] | None = None,
    resume_id: int | None = None,
) -> _BuildOutcome:
    """The whole build, shared by POST /generate (inside the request) and
    POST /start (in a worker thread). Failures come out as the
    HTTPException the caller would answer with; on_stage hears each
    step's name as it starts.

    checkpoint is updated in place as model calls finish ("data": the
    tailored content with its reserve, then "reworded"/"cuts_made" from
    the page fit), so a caller can save it when the build stops. One
    passed in already holding "data" skips the tailoring call, which is
    how a retry picks up where the last try stopped. resume_id is the
    incomplete library row a retry finishes.
    """
    _validate_template(body.template)
    max_pages = body.max_pages or _DEFAULT_MAX_PAGES[body.template]
    checkpoint = checkpoint if checkpoint is not None else {}

    # Two LLM steps, and the second one is worth finishing even if it
    # cannot: the content step's work is already paid for by the time the
    # page-fit step runs. Naming the steps means a stop says which one
    # stopped and what had already finished, instead of one flat message.
    run = Pipeline("The resume build", account_id=body.account_id)
    fit_note: str | None = None

    if checkpoint.get("data") is not None:
        data = copy.deepcopy(checkpoint["data"])
        run.completed.append(_TAILOR_STEP)
    else:
        try:
            on_stage("Tailoring the content to the job")
            with run.stage(_TAILOR_STEP):
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
                    selected_education_ids=body.selected_education_ids,
                    custom_instruction=body.custom_instruction,
                    points_per_role=body.points_per_role,
                )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (
            ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError
        ) as e:
            raise _map_llm_error(e) from e
        checkpoint["data"] = copy.deepcopy(data)

    try:
        on_stage("Fitting the resume to the page count")
        with run.stage("fitting the resume to the page count"):
            result = fit_to_page_limit(
                data,
                body.template,
                max_pages,
                account_id=body.account_id,
                reworded=bool(checkpoint.get("reworded")),
                cuts_made=int(checkpoint.get("cuts_made") or 0),
                on_progress=checkpoint.update,
            )
        pdf_bytes = result.pdf_bytes
        page_count = result.page_count
        achieved = result.fit_exact
        content = result.data or data
        if not achieved:
            logger.warning(
                "page-fit came up short for account_id=%s: %s page(s) against a target of %s",
                body.account_id, page_count, max_pages,
            )
    except PageFitNotAchievedError as e:
        # Includes the case where the page-fit step lost every key
        # partway: the content is written and compiled, so the resume is
        # saved and returned with the reason attached rather than lost.
        logger.warning("page-fit did not reach target for account_id=%s: %s", body.account_id, e)
        pdf_bytes = e.best_pdf_bytes
        page_count = e.best_page_count
        achieved = False
        content = e.best_data or data
        fit_note = str(e)
    except TectonicNotInstalledError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except CompileError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError, LLMProviderError) as e:
        raise _map_llm_error(e) from e

    # What gets saved is the content that was actually rendered, after
    # the page-fit loop's cuts and additions, not the pre-fit draft: the
    # library row and the PDF beside it have to describe the same resume.
    content.pop("reserve", None)
    on_stage("Saving to the resume library")
    resume_id = _save_generated_resume(
        body.account_id,
        body.job_posting_id,
        body.template,
        content,
        pdf_bytes,
        name=(body.name or "").strip() or None,
        resume_id=resume_id,
    )
    try:
        match = _content_match(body.job_posting_id, content.get("skills") or [])
    except Exception:
        logger.exception("could not score the new resume against the posting; continuing")
        match = {"match_pct": None, "matched": [], "related": [], "missing": []}

    return _BuildOutcome(
        pdf_bytes=pdf_bytes,
        page_count=page_count,
        max_pages=max_pages,
        achieved=achieved,
        fit_note=fit_note,
        resume_id=resume_id,
        match=match,
    )


@router.post("/generate")
def generate(body: GenerateRequest) -> Response:
    """Full pipeline: orchestrator content, then compile + the page-fit
    loop, returns the PDF bytes directly (Content-Type: application/pdf),
    same "serve the raw file" convention app/api/resume.py's
    GET /{id}/file already uses. The build page uses POST /start instead,
    which runs the same build in the background.

    The page count the caller picks (one page or two, or an explicit
    max_pages) is an exact target, not a ceiling: the page-fit loop
    tightens or opens up the layout, and adds back the account's own
    held-back content, to land on it in both directions. X-Page-Fit-
    Achieved reports whether it got there, X-Page-Count what it actually
    produced.
    """
    out = _run_build(body)
    return Response(
        content=out.pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline; filename=resume.pdf",
            "X-Page-Fit-Achieved": "true" if out.achieved else "false",
            # Only set when the fit fell short for a reason worth telling
            # the person (ran out of safe cuts, or ran out of keys).
            # Header-safe: newlines flattened, length bounded.
            **({"X-Page-Fit-Note": _header_safe(out.fit_note)} if out.fit_note else {}),
            "X-Page-Count": str(out.page_count),
            "X-Page-Target": str(out.max_pages),
            "X-Resume-Id": str(out.resume_id) if out.resume_id is not None else "",
            **(
                {"X-Match-Pct": str(out.match["match_pct"])}
                if out.match["match_pct"] is not None
                else {}
            ),
        },
    )


def _start_in_background(
    body: GenerateRequest,
    label: str,
    checkpoint: dict[str, Any] | None = None,
    retry_of: int | None = None,
) -> background.Build:
    """Runs _run_build in a worker thread. A build that stops partway, for
    whatever reason, is kept in the library as an incomplete resume with
    what it had done; the failed build names that row as
    incomplete_resume_id so the page can offer to retry it."""
    state: dict[str, Any] = dict(checkpoint or {})

    def run(build: background.Build) -> None:
        try:
            out = _run_build(
                body, on_stage=build.set_stage, checkpoint=state, resume_id=retry_of
            )
        except HTTPException as e:
            detail = str(e.detail)
            kind = getattr(e, "error_kind", None)
        except Exception:
            logger.exception("resume build %s crashed", build.id)
            detail = "Internal error, see server logs."
            kind = None
        else:
            build.finish(
                out.pdf_bytes,
                {
                    "resume_id": out.resume_id,
                    "page_fit_achieved": out.achieved,
                    "page_fit_note": out.fit_note,
                    "page_count": out.page_count,
                    "page_target": out.max_pages,
                    **out.match,
                },
            )
            return
        saved = _save_incomplete_resume(
            body, state, build.stage, detail, resume_id=retry_of, error_kind=kind
        )
        build.fail(detail, {"incomplete_resume_id": saved, "error_kind": kind})

    return background.start(
        body.account_id, body.job_posting_id, label, run, retry_of=retry_of
    )


def _build_label(body: GenerateRequest, posting: JobPosting) -> str:
    return (body.name or "").strip() or " - ".join(
        b for b in [posting.title, posting.company] if b
    ) or "Resume"


@router.post("/start")
def start_build(body: GenerateRequest, *, db: DbSession) -> dict[str, Any]:
    """Starts the same build as POST /generate in a worker thread and
    answers at once with its id. Closing the tab or moving to another
    page does not stop it; GET /builds reports its progress and result
    to whichever page asks. A bad template or an unknown posting is
    refused here rather than turning up later as a failed build.
    """
    _validate_template(body.template)
    posting = db.get(JobPosting, body.job_posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail=f"no job posting with id={body.job_posting_id}")
    return _start_in_background(body, _build_label(body, posting)).public()


@router.post("/retry/{resume_id}")
def retry_build(resume_id: int, *, db: DbSession) -> dict[str, Any]:
    """Picks an incomplete resume's build up where it stopped, with the
    same choices it was started with: a build whose content was already
    tailored goes straight to the page fit, so the model call that
    finished is not paid for twice. Asking again while a retry is running
    answers with that retry rather than starting a second one.
    """
    row = db.get(Resume, resume_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no resume with id={resume_id}")
    state = row.build_state_json
    if not state:
        raise HTTPException(status_code=409, detail="this resume is already finished")
    running = background.running_retry_of(resume_id)
    if running is not None:
        return running.public()
    body = GenerateRequest(**state["request"])
    posting = db.get(JobPosting, body.job_posting_id)
    if posting is None:
        raise HTTPException(
            status_code=404,
            detail="the job posting this resume was built for has been deleted",
        )
    checkpoint = {k: state[k] for k in ("data", "reworded", "cuts_made") if k in state}
    label = row.name or _build_label(body, posting)
    return _start_in_background(body, label, checkpoint, retry_of=resume_id).public()


def _get_build(build_id: str) -> background.Build:
    build = background.get(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail=f"no resume build with id={build_id}")
    return build


@router.get("/builds")
def list_builds(account_id: int, job_posting_id: int | None = None) -> list[dict[str, Any]]:
    """Builds for the account this process still remembers, newest first,
    dismissed ones left out. The header reminder polls this on every page."""
    return [b.public() for b in background.list_for_account(account_id, job_posting_id)]


@router.get("/builds/{build_id}")
def get_build(build_id: str) -> dict[str, Any]:
    return _get_build(build_id).public()


@router.get("/builds/{build_id}/pdf")
def get_build_pdf(build_id: str) -> Response:
    """The finished PDF straight from memory, so the result opens even
    when saving it to the library failed."""
    build = _get_build(build_id)
    if build.pdf_bytes is None:
        raise HTTPException(status_code=404, detail="this build has no PDF yet")
    return Response(
        content=build.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline; filename=resume.pdf"},
    )


@router.post("/builds/{build_id}/dismiss")
def dismiss_build(build_id: str) -> dict[str, bool]:
    _get_build(build_id)
    return {"dismissed": background.dismiss(build_id)}
