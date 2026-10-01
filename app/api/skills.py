"""The /skills aggregate view: one row per skill name, unioned across
three sources for an account: project-linked evidence (SkillEvidence, via
Repository), experience-linked evidence (ExperienceSkillEvidence, via
Experience), and freestanding entries (Skill, no evidence at all), either
typed in by hand or pulled from a resume.
Evidence rows are added and edited by app/api/projects.py and
app/api/experience.py; this router reads across both, owns CRUD for the
freestanding Skill table, and has the one delete that clears a skill
from every source at once (POST /api/skills/remove).

Grouping key is name.strip().casefold(), so "Python" and "python" land in
one group, displayed using whichever casing was seen first. This is a
display-time union, not a stored/materialized table, so it stays correct
automatically as evidence rows are added/edited/removed elsewhere.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.core.db import (
    Account,
    Experience,
    ExperienceSkillEvidence,
    Repository,
    Resume,
    Skill,
    SkillEvidence,
    SkillStar,
    SkillVerdict,
    get_db,
)
from app.profile.resume_profile_merge import unlink_profile_rows
from app.profile.skill_map import (
    build_layout,
    layout_fingerprint,
    load_cached,
    skill_contexts,
    skill_text,
    store_cached,
)
from app.profile.skill_review import approve, reject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skills")


class SkillSource(BaseModel):
    type: str  # "project" | "github_profile" | "experience"
    evidence_id: int
    ref_id: int  # repo_id or experience_id, for linking to its detail page
    name: str  # repo name without its owner, or "title @ company"
    evidence_type: str
    weight: float
    confidence: float


class SkillResumeRef(BaseModel):
    """The resume a freestanding skill was extracted from. The name is
    read from the resume row on every request, so a rename shows up here
    without touching the skill.
    """

    id: int
    name: str  # the resume's label, or its filename when it has none


class SkillGroup(BaseModel):
    name: str
    manual_skill_id: int | None = None
    source_resume: SkillResumeRef | None = None
    sources: list[SkillSource] = []
    starred: bool = False


def _name_key(name: str) -> str:
    return name.strip().casefold()


def _load_groups(db: Session, account_id: int) -> dict[str, SkillGroup]:
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
                type="github_profile" if repo.is_profile_readme else "project",
                evidence_id=evidence.id,
                ref_id=repo.id,
                # The repo's own name, not owner/name: every repo here is
                # the account's own, so the owner is noise on the page.
                name=repo.name or repo.full_name.rsplit("/", 1)[-1],
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
    for exp_evidence, exp in experience_rows:
        _group_for(exp_evidence.skill).sources.append(
            SkillSource(
                type="experience",
                evidence_id=exp_evidence.id,
                ref_id=exp.id,
                name=f"{exp.title} @ {exp.company}",
                evidence_type=exp_evidence.evidence_type,
                weight=exp_evidence.weight,
                confidence=exp_evidence.confidence,
            )
        )

    manual_rows = db.execute(
        select(Skill, Resume)
        .outerjoin(Resume, Skill.source_resume_id == Resume.id)
        .where(Skill.account_id == account_id)
    ).all()
    for skill, resume in manual_rows:
        group = _group_for(skill.name)
        group.manual_skill_id = skill.id
        if resume is not None:
            group.source_resume = SkillResumeRef(
                id=resume.id, name=(resume.name or "").strip() or resume.filename
            )

    # Stars never create a group, only mark one that exists from the
    # sources above.
    star_keys = db.execute(
        select(SkillStar.name_key).where(SkillStar.account_id == account_id)
    ).scalars()
    for key in star_keys:
        if key in groups:
            groups[key].starred = True

    return groups


def skill_groups_for_account(db: Session, account_id: int) -> list[SkillGroup]:
    """The body of GET /api/skills, callable with a session the caller
    already holds. app/api/resume_build.py needs this same list while
    building its options payload; going through the route function would
    open a second session inside the first one's transaction.
    """
    groups = _load_groups(db, account_id)
    return sorted(groups.values(), key=lambda g: (not g.starred, g.name.casefold()))


@router.get("", response_model=list[SkillGroup])
def list_skills(account_id: int, *, db: DbSession) -> list[SkillGroup]:
    return skill_groups_for_account(db, account_id)


class SkillMapNode(BaseModel):
    name: str
    x: float
    y: float
    cluster_id: int
    cluster_label: str
    cluster_color: str
    starred: bool
    manual_skill_id: int | None = None
    source_resume: SkillResumeRef | None = None
    sources: list[SkillSource] = []


class SkillClusterInfo(BaseModel):
    id: int
    label: str
    color: str
    count: int


class SkillMapResponse(BaseModel):
    clusters: list[SkillClusterInfo]
    nodes: list[SkillMapNode]


# Cluster colours are values of one ink, not a rainbow: the map has to
# read as part of the same page as the rest of the app, and eight
# saturated hues on a near-black background said "chart demo" more than
# they said "these skills are related". Ordered light to dark so a
# cluster stays distinguishable at any zoom.
CLUSTER_COLORS = [
    "#f2f2f0",
    "#c9c9c6",
    "#a3a3a6",
    "#8f8f97",
    "#7a7a85",
    "#6a6a74",
    "#5a5a63",
    "#4c4c54",
    "#3e3e45",
]


def warm_skill_maps() -> None:
    """Builds every account's map layout if its cache is stale, so the
    embedding model is loaded once at startup instead of inside whoever
    opens the map first. Called from the app lifespan on a daemon thread;
    a failure here must never stop the app from serving, so it is logged
    and dropped.
    """
    db = get_db()
    try:
        account_ids = list(db.execute(select(Account.id)).scalars())
        for account_id in account_ids:
            groups = list(_load_groups(db, account_id).values())
            if not groups:
                continue
            contexts = skill_contexts(account_id)
            texts = [skill_text(g.name, contexts.get(_name_key(g.name))) for g in groups]
            fingerprint = layout_fingerprint(texts)
            if load_cached(account_id, fingerprint) is not None:
                continue
            store_cached(account_id, fingerprint, build_layout(account_id, groups))
    except Exception:
        logger.exception("skill map warm start failed")
    finally:
        db.close()


@router.get("/map", response_model=SkillMapResponse)
def get_skills_map(account_id: int, *, db: DbSession) -> SkillMapResponse:
    """The account's skills as points on a plane, grouped into clusters.

    The layout itself is built by app/profile/skill_map.py and cached
    against a fingerprint of the skills that went into it, so this
    endpoint normally does no embedding work at all. It rebuilds when
    the fingerprint misses, which is the first request after a skill is
    added, removed, or given new evidence.

    Colour is assigned here rather than stored in the cache: it is a
    presentation choice that can change without every cached layout
    becoming wrong.
    """
    groups_by_key = _load_groups(db, account_id)
    groups = list(groups_by_key.values())
    if not groups:
        return SkillMapResponse(clusters=[], nodes=[])

    contexts = skill_contexts(account_id)
    texts = [skill_text(g.name, contexts.get(_name_key(g.name))) for g in groups]
    fingerprint = layout_fingerprint(texts)

    payload = load_cached(account_id, fingerprint)
    if payload is None:
        payload = build_layout(account_id, groups)
        store_cached(account_id, fingerprint, payload)

    def color_for(cluster_id: int) -> str:
        return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

    return SkillMapResponse(
        clusters=[
            SkillClusterInfo(
                id=c["id"], label=c["label"], color=color_for(c["id"]), count=c["count"]
            )
            for c in payload["clusters"]
        ],
        nodes=[
            SkillMapNode(
                cluster_color=color_for(n["cluster_id"]),
                **{**n, **_live_fields(groups_by_key.get(_name_key(n["name"])))},
            )
            for n in payload["nodes"]
        ],
    )


def _live_fields(group: SkillGroup | None) -> dict:
    """What a cached map node shows that is not part of its layout: a
    star, a resume rename or a new source never changes the fingerprint,
    so these are read fresh instead of from the cache.
    """
    if group is None:
        return {}
    return {
        "starred": group.starred,
        "manual_skill_id": group.manual_skill_id,
        "source_resume": group.source_resume,
        "sources": group.sources,
    }


class SkillStarUpdate(BaseModel):
    account_id: int
    name: str
    starred: bool


@router.post("/star", response_model=SkillStarUpdate)
def set_skill_star(body: SkillStarUpdate, *, db: DbSession) -> SkillStarUpdate:
    """Star or unstar a skill by name. Idempotent both ways: starring an
    already-starred skill or unstarring an unstarred one is a no-op, not
    an error. Keyed by casefolded name, so it applies to the whole group
    whatever sources back it (see SkillStar).
    """
    key = _name_key(body.name)
    if not key:
        raise HTTPException(status_code=422, detail="name is required")

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
def create_skill(body: SkillCreate, *, db: DbSession) -> SkillItem:
    """Adds a freestanding skill: one with no project or experience
    behind it. A skill that already has evidence doesn't need this: it
    already shows up in GET /api/skills without a Skill row at all.
    """
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")

    row = Skill(account_id=body.account_id, name=name)
    db.add(row)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="this account already has a manual skill by that name"
        ) from e
    # Adding it by hand overrules an earlier "not a skill" review verdict.
    approve(db, body.account_id, name)
    db.refresh(row)
    return SkillItem(id=row.id, name=row.name)


class RejectedSkill(BaseModel):
    name: str
    decided_by: str


@router.get("/rejected", response_model=list[RejectedSkill])
def list_rejected_skills(account_id: int, *, db: DbSession) -> list[RejectedSkill]:
    """Names review filtered out (app/profile/skill_review.py), for the
    Skills page's "filtered out" list, where any can be added back."""
    rows = db.execute(
        select(SkillVerdict).where(
            SkillVerdict.account_id == account_id, SkillVerdict.verdict == "rejected"
        )
    ).scalars()
    return sorted(
        (RejectedSkill(name=r.name, decided_by=r.decided_by) for r in rows),
        key=lambda r: r.name.casefold(),
    )


class SkillRemove(BaseModel):
    account_id: int
    name: str


@router.post("/remove")
def remove_skill_everywhere(body: SkillRemove, *, db: DbSession) -> dict:
    """Deletes a whole skill group, the Skills page's bin button: every
    project and experience evidence row with this name, the freestanding
    Skill row, and the tag on each resume it came from. The name is then
    marked rejected by the user, so the next GitHub sync or resume
    reprocess does not quietly add it back; it shows up under "filtered
    out" instead, where "Add back" undoes all of this except the removed
    evidence rows.
    """
    key = _name_key(body.name)
    if not key:
        raise HTTPException(status_code=422, detail="name is required")

    project_ids: list[int] = []
    project_rows = db.execute(
        select(SkillEvidence)
        .join(Repository, SkillEvidence.repo_id == Repository.id)
        .where(Repository.account_id == body.account_id)
    ).scalars()
    for evidence in project_rows:
        if _name_key(evidence.skill) == key:
            project_ids.append(evidence.id)
            db.delete(evidence)

    experience_ids: list[int] = []
    experience_rows = db.execute(
        select(ExperienceSkillEvidence)
        .join(Experience, ExperienceSkillEvidence.experience_id == Experience.id)
        .where(Experience.account_id == body.account_id)
    ).scalars()
    for evidence in experience_rows:
        if _name_key(evidence.skill) == key:
            experience_ids.append(evidence.id)
            db.delete(evidence)

    resume_ids: set[int] = set()
    skill_ids: list[int] = []
    for skill in db.execute(select(Skill).where(Skill.account_id == body.account_id)).scalars():
        if _name_key(skill.name) != key:
            continue
        if skill.source_resume_id is not None:
            resume_ids.add(skill.source_resume_id)
        skill_ids.append(skill.id)
        db.delete(skill)
    unlink_profile_rows(db, "experience_skill", experience_ids)
    unlink_profile_rows(db, "skill", skill_ids)
    for resume in db.execute(select(Resume).where(Resume.id.in_(resume_ids))).scalars():
        resume.tags_json = [t for t in resume.tags_json or [] if _name_key(str(t)) != key]
    db.commit()

    from app.api.experience import _delete_qdrant_points as delete_experience_points
    from app.api.projects import _delete_qdrant_points as delete_project_points
    from app.retrieval.index import experience_evidence_point_id

    delete_project_points(project_ids)
    delete_experience_points([experience_evidence_point_id(i) for i in experience_ids])

    reject(db, body.account_id, body.name)
    return {
        "deleted": True,
        "project_evidence": len(project_ids),
        "experience_evidence": len(experience_ids),
        "resumes": len(resume_ids),
    }


@router.delete("/{skill_id}")
def delete_skill(skill_id: int, *, db: DbSession) -> dict:
    """Removes a freestanding Skill row only. A project- or
    experience-linked skill is deleted through its own evidence endpoint
    (POST/PATCH/DELETE .../skills/{id} on projects.py or experience.py),
    so there is only one delete path for the same underlying row.
    """
    row = db.get(Skill, skill_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no skill with id={skill_id}")
    unlink_profile_rows(db, "skill", [skill_id])
    db.delete(row)
    db.commit()
    return {"deleted": True, "id": skill_id}
