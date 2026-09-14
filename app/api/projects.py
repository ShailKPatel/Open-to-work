"""Detected-projects page backend: list synced repos with skill-extraction
status, and manual (re)process triggers.

Extraction isn't bundled into POST /sync/github. A sync just fetches; this
router is the separate, explicit step that turns fetched repos into skill
evidence. Keeps /sync/github's duration predictable regardless of how many
repos need an LLM call, and matches the page split: /sync fetches, /projects
shows and manages extraction.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.core.db import ProjectLink, Repository, SkillEvidence, get_db
from app.profile.build import build_profile, reprocess_repo, skill_evidence_for_repos
from app.profile.evidence import add_evidence, delete_evidence, update_evidence
from app.profile.jobs import extraction_stream, start_extraction

# /api prefix, not just style: a JSON endpoint at the same path as an HTML
# page silently wins the route (routes are matched in registration order)
# and makes the page unreachable.
router = APIRouter(prefix="/api/projects")
logger = logging.getLogger(__name__)


def _delete_qdrant_points(evidence_ids: list[int]) -> None:
    """Twin of app/api/experience.py's own _delete_qdrant_points, against
    the same skill_evidence collection. SQLite is the source of truth here
    too; a Qdrant cleanup failure is logged, never blocks the request.
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
        logger.exception("could not clean up Qdrant points for deleted skill evidence; continuing")


class ProjectSummary(BaseModel):
    id: int
    name: str
    full_name: str
    url: str
    description: str | None
    has_readme: bool
    is_fork: bool
    primary_language: str | None
    skill_extraction_status: str
    skill_extraction_error: str | None
    skills_extracted_at: dt.datetime | None
    skill_count: int
    # Manual entries never touch GitHub: no synced github_id, so we mark
    # them with a synthetic negative one instead of a schema migration
    # (see _next_manual_github_id). Surfaced here so the list page can
    # skip "GitHub ↗" for a project that never had a GitHub repo.
    is_manual: bool
    # Curation: optional, purely for "which projects lead a resume" (see
    # Repository.starred). Drives list ordering below; no other behavior
    # reads it yet.
    starred: bool

    @classmethod
    def from_repo(cls, repo: Repository, skill_count: int) -> ProjectSummary:
        return cls(
            id=repo.id,
            name=repo.name,
            full_name=repo.full_name,
            url=repo.url,
            description=repo.description,
            has_readme=bool(repo.readme and repo.readme.strip()),
            is_fork=repo.is_fork,
            primary_language=repo.primary_language,
            skill_extraction_status=repo.skill_extraction_status,
            skill_extraction_error=repo.skill_extraction_error,
            skills_extracted_at=repo.skills_extracted_at,
            skill_count=skill_count,
            is_manual=repo.github_id < 0,
            starred=repo.starred,
        )


def _skill_counts(db, repo_ids: list[int]) -> dict[int, int]:
    if not repo_ids:
        return {}
    rows = db.execute(
        select(SkillEvidence.repo_id, func.count(SkillEvidence.id))
        .where(SkillEvidence.repo_id.in_(repo_ids))
        .group_by(SkillEvidence.repo_id)
    ).all()
    return dict(rows)


@router.get("", response_model=list[ProjectSummary])
def list_projects(account_id: int) -> list[ProjectSummary]:
    db = get_db()
    try:
        repos = list(
            db.execute(
                select(Repository)
                .where(Repository.account_id == account_id)
                .order_by(Repository.full_name)
            ).scalars()
        )
        # Starred first, then alphabetical: the same tie-break
        # order_by(full_name) already gave everything else.
        repos.sort(key=lambda r: (not r.starred, r.full_name))
        counts = _skill_counts(db, [r.id for r in repos])
        return [ProjectSummary.from_repo(r, counts.get(r.id, 0)) for r in repos]
    finally:
        db.close()


class ProjectLinkItem(BaseModel):
    id: int
    label: str
    url: str
    # "manual" | "readme_extracted", surfaced so the UI can show where an
    # extracted link came from; not itself editable (see update_link).
    source: str


class SkillEvidenceItem(BaseModel):
    id: int
    skill: str
    evidence_type: str
    weight: float
    confidence: float


class ProjectDetail(BaseModel):
    id: int
    name: str
    full_name: str
    url: str
    description: str | None
    is_fork: bool
    primary_language: str | None
    stars: int
    has_readme: bool
    # capped, not the full README: this is a detail page, not a reader.
    # None when there's no README at all (see has_readme).
    readme_preview: str | None
    manifests: dict
    commits_authored: int
    last_commit_at: dt.datetime | None
    pushed_at: dt.datetime | None
    fetched_at: dt.datetime
    skill_extraction_status: str
    skill_extraction_error: str | None
    skills_extracted_at: dt.datetime | None
    skills: list[SkillEvidenceItem]
    links: list[ProjectLinkItem]
    is_manual: bool
    starred: bool

    @classmethod
    def from_repo(
        cls, repo: Repository, skills: list[SkillEvidenceItem], links: list[ProjectLinkItem]
    ) -> ProjectDetail:
        readme = repo.readme
        return cls(
            id=repo.id,
            name=repo.name,
            full_name=repo.full_name,
            url=repo.url,
            description=repo.description,
            is_fork=repo.is_fork,
            primary_language=repo.primary_language,
            stars=repo.stars,
            has_readme=bool(readme and readme.strip()),
            readme_preview=readme[:4000] if readme else None,
            manifests=repo.manifests_json or {},
            commits_authored=repo.commits_authored,
            last_commit_at=repo.last_commit_at,
            pushed_at=repo.pushed_at,
            fetched_at=repo.fetched_at,
            skill_extraction_status=repo.skill_extraction_status,
            skill_extraction_error=repo.skill_extraction_error,
            skills_extracted_at=repo.skills_extracted_at,
            skills=skills,
            links=links,
            is_manual=repo.github_id < 0,
            starred=repo.starred,
        )


def _links_for_repo(db, repo_id: int) -> list[ProjectLinkItem]:
    rows = list(
        db.execute(
            select(ProjectLink).where(ProjectLink.repo_id == repo_id).order_by(ProjectLink.id)
        ).scalars()
    )
    return [ProjectLinkItem(id=r.id, label=r.label, url=r.url, source=r.source) for r in rows]


def _detail_for(db, repo: Repository) -> ProjectDetail:
    evidence = skill_evidence_for_repos([repo.id])
    skills = [
        SkillEvidenceItem(
            id=e.id,
            skill=e.skill,
            evidence_type=e.evidence_type,
            weight=e.weight,
            confidence=e.confidence,
        )
        for e in evidence
    ]
    # highest-weight evidence first, so the most confident claims read first
    skills.sort(key=lambda s: s.weight, reverse=True)
    links = _links_for_repo(db, repo.id)
    return ProjectDetail.from_repo(repo, skills, links)


@router.get("/{repo_id}", response_model=ProjectDetail)
def project_detail(repo_id: int) -> ProjectDetail:
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        return _detail_for(db, repo)
    finally:
        db.close()


def _next_manual_github_id(db) -> int:
    """Manually-added projects have no GitHub repo behind them, but
    `Repository.github_id` is a NOT NULL unique column (no migration
    tooling in this project to make it nullable). Real GitHub ids are always
    positive, so a strictly-decreasing negative counter is unique forever
    without touching the schema; `ProjectSummary.is_manual` /
    `ProjectDetail.is_manual` key off `github_id < 0` to tell these apart
    from synced repos.
    """
    current_min = db.execute(select(func.min(Repository.github_id))).scalar()
    return min(current_min or 0, 0) - 1


class ProjectCreate(BaseModel):
    account_id: int
    name: str
    full_name: str | None = None
    url: str = ""
    description: str | None = None
    primary_language: str | None = None
    readme: str | None = None
    is_fork: bool = False


@router.post("", response_model=ProjectDetail)
def create_project(body: ProjectCreate) -> ProjectDetail:
    """Manual counterpart to GitHub sync, for a project that isn't (or
    isn't yet) a GitHub repo. Starts at `skill_extraction_status="pending"`
    same as a freshly-synced repo, so Reprocess / process-pending pick it
    up normally once it has a README or description to extract from.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")

    db = get_db()
    try:
        repo = Repository(
            account_id=body.account_id,
            github_id=_next_manual_github_id(db),
            name=name,
            full_name=(body.full_name or name).strip(),
            url=body.url.strip(),
            description=(body.description or None),
            primary_language=(body.primary_language or None),
            readme=(body.readme or None),
            is_fork=body.is_fork,
        )
        db.add(repo)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=409, detail="a project with that full name already exists"
            ) from e
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


MAX_STARRED_PROJECTS = 3


class ProjectUpdate(BaseModel):
    name: str | None = None
    full_name: str | None = None
    url: str | None = None
    description: str | None = None
    primary_language: str | None = None
    readme: str | None = None
    is_fork: bool | None = None
    starred: bool | None = None


@router.patch("/{repo_id}", response_model=ProjectDetail)
def update_project(repo_id: int, body: ProjectUpdate) -> ProjectDetail:
    """Everything on the detail page is editable in place; this backs the
    single save action for name/full_name/url/description/language/readme/
    fork flag. Fields the client didn't send are left untouched
    (`exclude_unset`), so a save always round-trips the whole visible form
    without clobbering anything not shown there (stars, commit stats,
    extraction status stay sync/extraction-owned).
    """
    fields = body.model_dump(exclude_unset=True)
    for key in ("name", "full_name", "url"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
    if "name" in fields and not fields["name"]:
        raise HTTPException(status_code=422, detail="name is required")
    if "full_name" in fields and not fields["full_name"]:
        raise HTTPException(status_code=422, detail="full_name is required")

    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        if fields.get("starred") is True and not repo.starred:
            starred_count = db.execute(
                select(func.count(Repository.id)).where(
                    Repository.account_id == repo.account_id,
                    Repository.starred.is_(True),
                )
            ).scalar()
            if (starred_count or 0) >= MAX_STARRED_PROJECTS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"only {MAX_STARRED_PROJECTS} starred projects allowed"
                        "; unstar one first"
                    ),
                )
        for key, value in fields.items():
            setattr(repo, key, value)
        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            raise HTTPException(
                status_code=409, detail="a project with that full name already exists"
            ) from e
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


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


@router.post("/{repo_id}/skills", response_model=ProjectDetail)
def add_skill(repo_id: int, body: SkillEvidenceCreate) -> ProjectDetail:
    """Hand-added skill claim: same row shape LLM extraction writes
    (`skill_evidence`), just entered by a person instead. Defaults to
    `evidence_type="manual"` so the UI's evidence label ("from README",
    "from dependencies", …) can show "added manually" for these without
    guessing.
    """
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        row = add_evidence(
            db,
            SkillEvidence,
            "repo_id",
            repo_id,
            skill=body.skill,
            evidence_type=body.evidence_type,
            weight=body.weight,
            confidence=body.confidence,
        )
        try:
            from app.retrieval.index import index_skill_evidence

            index_skill_evidence([row], account_id=repo.account_id)
        except Exception:
            logger.exception("could not index skill evidence id=%s", row.id)
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


@router.patch("/{repo_id}/skills/{skill_id}", response_model=ProjectDetail)
def update_skill(repo_id: int, skill_id: int, body: SkillEvidenceUpdate) -> ProjectDetail:
    fields = body.model_dump(exclude_unset=True)

    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        update_evidence(db, SkillEvidence, "repo_id", repo_id, skill_id, fields)
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


@router.delete("/{repo_id}/skills/{skill_id}", response_model=ProjectDetail)
def delete_skill(repo_id: int, skill_id: int) -> ProjectDetail:
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        delete_evidence(db, SkillEvidence, "repo_id", repo_id, skill_id)
        _delete_qdrant_points([skill_id])
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


class ProjectLinkCreate(BaseModel):
    label: str
    url: str


class ProjectLinkUpdate(BaseModel):
    label: str | None = None
    url: str | None = None


@router.post("/{repo_id}/links", response_model=ProjectDetail)
def add_link(repo_id: int, body: ProjectLinkCreate) -> ProjectDetail:
    """Hand-added link: same manual/extracted split as add_skill above.
    Always `source="manual"`: only build.py's first-pass README extraction
    writes `readme_extracted` rows.
    """
    label = body.label.strip()
    url = body.url.strip()
    if not label:
        raise HTTPException(status_code=422, detail="label is required")
    if not url:
        raise HTTPException(status_code=422, detail="url is required")

    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        db.add(ProjectLink(repo_id=repo_id, label=label, url=url, source="manual"))
        db.commit()
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


@router.patch("/{repo_id}/links/{link_id}", response_model=ProjectDetail)
def update_link(repo_id: int, link_id: int, body: ProjectLinkUpdate) -> ProjectDetail:
    """Editing an extracted link's label/url locks it to `source="manual"`;
    otherwise the next Reprocess would silently wipe the correction (see
    build.py's _process_repo, which only re-derives `readme_extracted` rows).
    """
    fields = body.model_dump(exclude_unset=True)
    if "label" in fields:
        fields["label"] = (fields["label"] or "").strip()
        if not fields["label"]:
            raise HTTPException(status_code=422, detail="label is required")
    if "url" in fields:
        fields["url"] = (fields["url"] or "").strip()
        if not fields["url"]:
            raise HTTPException(status_code=422, detail="url is required")

    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        link = db.get(ProjectLink, link_id)
        if link is None or link.repo_id != repo_id:
            raise HTTPException(status_code=404, detail=f"no link with id={link_id}")
        if fields:
            fields["source"] = "manual"
        for key, value in fields.items():
            setattr(link, key, value)
        db.commit()
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


@router.delete("/{repo_id}/links/{link_id}", response_model=ProjectDetail)
def delete_link(repo_id: int, link_id: int) -> ProjectDetail:
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail=f"no repo with id={repo_id}")
        link = db.get(ProjectLink, link_id)
        if link is None or link.repo_id != repo_id:
            raise HTTPException(status_code=404, detail=f"no link with id={link_id}")
        db.delete(link)
        db.commit()
        db.refresh(repo)
        return _detail_for(db, repo)
    finally:
        db.close()


@router.post("/{repo_id}/reprocess", response_model=ProjectSummary)
def reprocess(repo_id: int) -> ProjectSummary:
    """The UI's manual retry: always forces re-extraction, whatever the
    current status. Works whether the repo previously failed, had no
    signal, or already succeeded and someone just wants to redo it.
    """
    try:
        repo = reprocess_repo(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    db = get_db()
    try:
        count = _skill_counts(db, [repo_id]).get(repo_id, 0)
        return ProjectSummary.from_repo(repo, count)
    finally:
        db.close()


class ProcessPendingResponse(BaseModel):
    considered: int


@router.post("/process-pending", response_model=ProcessPendingResponse)
def process_pending(account_id: int) -> ProcessPendingResponse:
    """Bulk-triggers extraction for every one of this account's repos that
    isn't already done. build_profile's own skip logic means repos
    already `extracted`/`no_signal` cost nothing to include here, so this
    is safe to call every time the projects page loads.
    """
    db = get_db()
    try:
        repos = list(
            db.execute(select(Repository).where(Repository.account_id == account_id)).scalars()
        )
    finally:
        db.close()
    build_profile(repos)
    return ProcessPendingResponse(considered=len(repos))


class StartExtractionResponse(BaseModel):
    started: bool


@router.post("/process-pending/start", response_model=StartExtractionResponse)
def start_process_pending(account_id: int) -> StartExtractionResponse:
    """Backgrounded twin of POST /process-pending: kicks off extraction in
    its own thread and returns immediately instead of blocking for the
    whole batch. `started=False` just means one was already running for
    this account (not an error); either way, GET .../process-pending/stream
    picks it up. Safe to call on every page load / every sync tick; a
    second call while one's in flight is a no-op.
    """
    return StartExtractionResponse(started=start_extraction(account_id))


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@router.get("/process-pending/stream")
def process_pending_stream(account_id: int) -> StreamingResponse:
    """SSE progress for the background extraction job: "N of M repos
    processed", one event per repo. Starts the job if it isn't already
    running (so a client can just open this directly without a separate
    /start call), but the job itself is NOT tied to this connection: it
    keeps running in its own thread even if this stream is never opened, or
    is closed/abandoned (tab closed, navigated away) partway through. This
    is what lets the caller start extraction, then go look at other pages,
    and reconnect here later to see how far it's gotten.
    """
    start_extraction(account_id)

    def events():
        for state in extraction_stream(account_id):
            yield _sse(state)

    return StreamingResponse(events(), media_type="text/event-stream")
