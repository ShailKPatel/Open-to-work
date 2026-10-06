"""Writers for every Qdrant collection the app searches: skill evidence,
experience points, resumes, role families and job postings.

What is embedded is the reduced record, not the source: a skill claim
with its context rather than a README, a resume's summary and tags rather
than the file. Each experience point is its own vector, so resume
building can retrieve the points that match a job.

Every payload carries account_id, which app/retrieval/search.py filters on.
"""

from __future__ import annotations

from app.core.db import (
    ExperiencePoint,
    ExperienceSkillEvidence,
    JobPosting,
    Resume,
    RoleFamily,
    SkillEvidence,
)
from app.core.llm import embed
from app.retrieval.vectorstore import ensure_collection, get_client

COLLECTION = "skill_evidence"
RESUME_COLLECTION = "resumes"
EXPERIENCE_POINTS_COLLECTION = "experience_points"
ROLE_FAMILIES_COLLECTION = "role_families"
JOB_POSTINGS_COLLECTION = "job_postings"

# SkillEvidence (repo-linked) and ExperienceSkillEvidence (experience-linked)
# share the skill_evidence collection but each has its own independent
# autoincrement id sequence, so a repo-evidence id and an experience-evidence
# id can collide numerically. Harmless as long as nothing queries the
# collection; once search.py started doing that, a collision would silently
# overwrite one row's point with the other's (Qdrant upsert-by-id) and one
# of the two would vanish from every future search with no error anywhere.
# Fixed by offsetting every experience-linked point id into a range no
# repo-linked autoincrement id will ever reach.
_EXPERIENCE_EVIDENCE_ID_OFFSET = 1_000_000_000


def evidence_text(evidence: SkillEvidence | ExperienceSkillEvidence) -> str:
    """Public (not just an internal helper): app/evals/run.py's BM25
    baseline builds its corpus from the exact same text the dense system
    embeds, via this function, so the two are actually comparable rather
    than scored against two different representations of the same row."""
    return f"Skill: {evidence.skill}. Evidence: {evidence.evidence_type}."


def index_skill_evidence(evidence_rows: list[SkillEvidence], account_id: int | None = None) -> int:
    if not evidence_rows:
        return 0

    from qdrant_client.models import PointStruct

    texts = [evidence_text(e) for e in evidence_rows]
    vectors = embed(texts)
    ensure_collection(COLLECTION, vector_size=len(vectors[0]))

    points = [
        PointStruct(
            id=evidence.id,
            vector=vector,
            payload={
                "skill": evidence.skill,
                "repo_id": evidence.repo_id,
                "evidence_type": evidence.evidence_type,
                "weight": evidence.weight,
                "confidence": evidence.confidence,
                "source_type": "repo",
                "account_id": account_id,
            },
        )
        for evidence, vector in zip(evidence_rows, vectors, strict=True)
    ]
    get_client().upsert(collection_name=COLLECTION, points=points)
    return len(points)


def index_experience_skill_evidence(
    evidence_rows: list[ExperienceSkillEvidence], account_id: int | None = None
) -> int:
    """Twin of index_skill_evidence() for experience-linked evidence, same
    collection, same embed/upsert shape, kept as a separate function rather
    than a branch inside the other one so neither caller has to know about
    the other's model (same split as sync_account/sync_account_progress).
    Point id is offset, see
    _EXPERIENCE_EVIDENCE_ID_OFFSET above; deleting these points (e.g.
    app/api/experience.py's delete_skill) must apply the same offset.
    """
    if not evidence_rows:
        return 0

    from qdrant_client.models import PointStruct

    texts = [evidence_text(e) for e in evidence_rows]
    vectors = embed(texts)
    ensure_collection(COLLECTION, vector_size=len(vectors[0]))

    points = [
        PointStruct(
            id=evidence.id + _EXPERIENCE_EVIDENCE_ID_OFFSET,
            vector=vector,
            payload={
                "skill": evidence.skill,
                "experience_id": evidence.experience_id,
                "evidence_type": evidence.evidence_type,
                "weight": evidence.weight,
                "confidence": evidence.confidence,
                "source_type": "experience",
                "account_id": account_id,
            },
        )
        for evidence, vector in zip(evidence_rows, vectors, strict=True)
    ]
    get_client().upsert(collection_name=COLLECTION, points=points)
    return len(points)


def experience_evidence_point_id(evidence_id: int) -> int:
    """The offset id an ExperienceSkillEvidence row's Qdrant point actually
    has, see _EXPERIENCE_EVIDENCE_ID_OFFSET. Callers deleting a point by
    evidence_id (app/api/experience.py) must go through this rather than
    passing the raw evidence_id straight to Qdrant.
    """
    return evidence_id + _EXPERIENCE_EVIDENCE_ID_OFFSET


def index_experience_points(
    point_rows: list[ExperiencePoint], account_id: int | None = None
) -> int:
    """Embeds each ExperiencePoint's raw text (not a reduced claim like the
    two functions above; a point already is the unit, nothing to reduce)
    into its own collection, id == ExperiencePoint.id, so the same id can
    upsert-overwrite on an edit (see app/api/experience.py's update_point).
    This is the collection resume building queries per job
    target to pull back a person's most relevant points instead of reading
    one fixed paragraph.
    """
    if not point_rows:
        return 0

    from qdrant_client.models import PointStruct

    texts = [p.text for p in point_rows]
    vectors = embed(texts)
    ensure_collection(EXPERIENCE_POINTS_COLLECTION, vector_size=len(vectors[0]))

    points = [
        PointStruct(
            id=point.id,
            vector=vector,
            payload={
                "experience_id": point.experience_id,
                "text": point.text,
                "order_index": point.order_index,
                "account_id": account_id,
            },
        )
        for point, vector in zip(point_rows, vectors, strict=True)
    ]
    get_client().upsert(collection_name=EXPERIENCE_POINTS_COLLECTION, points=points)
    return len(points)


def _resume_text(resume: Resume) -> str:
    parts = [resume.summary or ""]
    if resume.tags_json:
        parts.append("Tags: " + ", ".join(resume.tags_json))
    if resume.target_roles_json:
        parts.append("Target roles: " + ", ".join(resume.target_roles_json))
    return "\n".join(p for p in parts if p)


def index_resume(resume: Resume) -> bool:
    """Embeds this resume's extracted summary/tags/target_roles (not the
    raw file, same "index the reduced claim, not the source text"
    reasoning as index_skill_evidence above) into its own Qdrant
    collection, one point per resume, id == Resume.id. Returns False
    without calling the embedder when there's nothing extracted yet to
    embed (a resume still "pending" or one that failed extraction); called
    again once extraction succeeds, or after a manual edit to any of the
    three fields (see PATCH /api/resume/{id}).
    """
    text = _resume_text(resume)
    if not text.strip():
        return False

    from qdrant_client.models import PointStruct

    vector = embed([text])[0]
    ensure_collection(RESUME_COLLECTION, vector_size=len(vector))
    get_client().upsert(
        collection_name=RESUME_COLLECTION,
        points=[
            PointStruct(
                id=resume.id,
                vector=vector,
                payload={
                    "account_id": resume.account_id,
                    "filename": resume.filename,
                    "tags": resume.tags_json,
                    "target_roles": resume.target_roles_json,
                },
            )
        ],
    )
    return True


def index_role_family(role_family: RoleFamily) -> None:
    """Embeds a RoleFamily's canonical_name into its own collection, id ==
    RoleFamily.id, no account_id payload: this taxonomy is global, not
    per-account. app/profile/role_family.py's resolve_role_family() searches
    this collection before ever creating a new row, so most titles after
    the first few dozen postings should resolve without a new LLM call.
    """
    from qdrant_client.models import PointStruct

    vector = embed([role_family.canonical_name])[0]
    ensure_collection(ROLE_FAMILIES_COLLECTION, vector_size=len(vector))
    get_client().upsert(
        collection_name=ROLE_FAMILIES_COLLECTION,
        points=[
            PointStruct(
                id=role_family.id,
                vector=vector,
                payload={"canonical_name": role_family.canonical_name},
            )
        ],
    )


def job_posting_text(posting: JobPosting) -> str:
    """Public: also used by app/api/job_analytics.py's similar-postings
    endpoint to build the query text for an existing posting, so both the
    write side (indexing) and that read side embed the exact same
    reduced representation of a posting."""
    extracted = posting.extracted_json or {}
    from app.profile.job_extract import parse_skills_required

    skills = [s["skill"] for s in parse_skills_required(extracted.get("skills_required", []))]
    parts = [f"{posting.title} at {posting.company}"]
    if extracted.get("role_summary"):
        parts.append(extracted["role_summary"])
    if skills:
        parts.append("Skills: " + ", ".join(skills))
    return "\n".join(parts)


def index_job_posting(posting: JobPosting, account_id: int | None = None) -> bool:
    """Embeds a job posting's title/company/role_summary/skills (the
    reduced extracted claim, not raw_text_quarantined, same "index the
    reduced claim, not the source text" reasoning as index_skill_evidence)
    into its own collection, id == JobPosting.id, account-scoped. Powers a
    "postings like this one you've already seen" lookup
    (app/retrieval/search.py's search_job_postings), separate from the
    plain SQL aggregation app/api/job_analytics.py uses for skill-demand
    counting. Semantic search finds similar postings, it can't correctly
    answer "how many postings mention Python," which is exactly what
    counting real extracted_json rows can. Returns False without embedding
    when there's nothing extracted yet.
    """
    text = job_posting_text(posting)
    if not text.strip() or posting.extraction_status != "extracted":
        return False

    from qdrant_client.models import PointStruct

    vector = embed([text])[0]
    ensure_collection(JOB_POSTINGS_COLLECTION, vector_size=len(vector))
    get_client().upsert(
        collection_name=JOB_POSTINGS_COLLECTION,
        points=[
            PointStruct(
                id=posting.id,
                vector=vector,
                payload={
                    "account_id": account_id,
                    "title": posting.title,
                    "company": posting.company,
                },
            )
        ],
    )
    return True
