"""Retrieval read path: semantic search over the Qdrant collections
app/retrieval/index.py writes into.

This is the query layer: given a chunk of text (typically a pasted job
description), pull back the account's best-matching skill evidence,
experience points, or resumes.

Account-scoped by construction. Every hit is filtered on the payload's
account_id (see index.py) so an account's search never surfaces another
account's data: this app already supports more than one local profile.

Experience-linked skill evidence is stored at an offset point id so it
can't collide with repo-linked rows in the shared skill_evidence
collection (see index.py's _EXPERIENCE_EVIDENCE_ID_OFFSET).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.llm import embed
from app.retrieval.vectorstore import get_client


@dataclass
class Hit:
    id: int
    score: float
    payload: dict[str, Any]


def _search(
    collection: str,
    query_text: str,
    account_id: int | None,
    top_k: int,
    must: list[Any] | None = None,
) -> list[Hit]:
    """account_id=None skips the account filter entirely; only correct
    for a global, not-per-account collection (role_families today, see
    search_role_families below). Every account-scoped collection must
    keep passing a real account_id; there is no separate safety check
    here beyond callers using the right wrapper function.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = get_client()
    if not client.collection_exists(collection):
        return []

    vector = embed([query_text])[0]
    conditions: list[Any] = []
    if account_id is not None:
        conditions.append(FieldCondition(key="account_id", match=MatchValue(value=account_id)))
    if must:
        conditions.extend(must)

    result = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=Filter(must=conditions) if conditions else None,
        limit=top_k,
    )
    return [Hit(id=p.id, score=p.score, payload=p.payload or {}) for p in result.points]


def search_skill_evidence(
    query_text: str, account_id: int, top_k: int = 10, source_type: str | None = None
) -> list[Hit]:
    """Searches the skill_evidence collection (both repo-linked and
    experience-linked rows share it, see index.py). source_type narrows to
    "repo" or "experience" when the caller only wants one kind; omitted,
    both are searched together.
    """
    from app.retrieval.index import COLLECTION

    must = None
    if source_type is not None:
        from qdrant_client.models import FieldCondition, MatchValue

        must = [FieldCondition(key="source_type", match=MatchValue(value=source_type))]
    return _search(COLLECTION, query_text, account_id, top_k, must)


def search_experience_points(
    query_text: str, account_id: int, top_k: int = 10, experience_id: int | None = None
) -> list[Hit]:
    """Best-matching ExperiencePoint rows for a target job, the piece a
    resume-building pass needs: pull back whichever points actually match
    the job, either across every role (experience_id omitted) or scoped
    to one role (experience_id given, app/resume_build/orchestrator.py's
    per-role point selection uses this so a role with few strong matches
    doesn't lose out to another role's points crowding an account-wide
    top_k).
    """
    from app.retrieval.index import EXPERIENCE_POINTS_COLLECTION

    must = None
    if experience_id is not None:
        from qdrant_client.models import FieldCondition, MatchValue

        must = [FieldCondition(key="experience_id", match=MatchValue(value=experience_id))]
    return _search(EXPERIENCE_POINTS_COLLECTION, query_text, account_id, top_k, must)


def search_resumes(query_text: str, account_id: int, top_k: int = 5) -> list[Hit]:
    from app.retrieval.index import RESUME_COLLECTION

    return _search(RESUME_COLLECTION, query_text, account_id, top_k)


def search_role_families(title_text: str, top_k: int = 3) -> list[Hit]:
    """The one search in this module that is not account-scoped in this module: role
    families are a global taxonomy (see app/core/db/models.py's RoleFamily
    docstring), not per-account data, so account_id is omitted entirely
    rather than passed as None-meaning-unrestricted by accident; _search
    only skips its account filter when explicitly asked to.
    """
    from app.retrieval.index import ROLE_FAMILIES_COLLECTION

    return _search(ROLE_FAMILIES_COLLECTION, title_text, None, top_k)


def search_job_postings(query_text: str, account_id: int, top_k: int = 5) -> list[Hit]:
    """Postings similar to a given piece of text (typically another
    posting's own title+skills) that this account has already seen:
    "you've looked at roles like this before." Distinct from
    app/api/job_analytics.py's skill-demand counting, which needs exact
    aggregation over extracted_json, not semantic similarity.
    """
    from app.retrieval.index import JOB_POSTINGS_COLLECTION

    return _search(JOB_POSTINGS_COLLECTION, query_text, account_id, top_k)
