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

Skill evidence and experience points are searched hybrid by default
(_hybrid_search): the full query plus one query per skill the posting
asks for, each run both as a dense search and as BM25 keyword matching,
fused by reciprocal rank. That combination was chosen by measurement
(scripts/compare_retrieval.py): on the real-text eval set it lifted
precision@5 from 0.515 for a single dense query to 0.823, and nDCG@10 from
0.481 to 0.747, while a cross-encoder reranker over the same candidates
made every metric worse. A single vector for a whole posting has to sit
near every skill it names at once, and a skill document is a short label;
one query per skill, plus exact keyword matching, fixes both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.llm import embed
from app.retrieval.vectorstore import get_client

# The BGE models are trained with this instruction in front of short
# search queries and nothing in front of the documents they search. Adding
# it lifted single-query dense precision@5 on both eval sets.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Reciprocal rank fusion constant, the standard value from the original
# RRF paper; it damps the difference between adjacent ranks so no single
# run dominates.
_RRF_K = 60

# How deep each run ranks before fusion. Profiles hold hundreds of
# evidence rows at most, so this covers a meaningful slice of any of them.
_RUN_DEPTH = 50

# Sub-queries beyond the full query. A posting can name dozens of skills;
# past this many, each extra run mostly adds noise and embedding time.
_MAX_SKILL_QUERIES = 12

SearchMode = Literal["hybrid", "dense"]


@dataclass
class Hit:
    """One search result. For hybrid search, `score` is the fused
    reciprocal rank score scaled so a document ranked first by every run
    scores 1.0; for dense search it is cosine similarity."""

    id: int
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class Query:
    """What to search for: the posting as text, and the skills it asks for
    by name. `skills` drives one sub-query each; empty, only the text is
    searched."""

    text: str
    skills: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_posting_text(cls, text: str) -> Query:
        """A Query from text in app/retrieval/index.py's job_posting_text
        shape, whose last line lists the skills ("Skills: a, b"). Text
        without that line becomes a query with no skills."""
        lines = text.rstrip().split("\n")
        if lines and lines[-1].startswith("Skills: "):
            skills = tuple(s.strip() for s in lines[-1][len("Skills: "):].split(",") if s.strip())
            return cls(text=text, skills=skills)
        return cls(text=text)


def query_for_posting(posting: Any) -> Query:
    """The retrieval query for a saved JobPosting: its extracted title,
    summary and skills when extraction has run, the raw text otherwise.

    The extracted form matters. Raw posting text runs to thousands of
    characters, and the embedding model reads only the first 512 tokens,
    which in most postings are the company's introduction rather than the
    role; the extracted fields are the role."""
    from app.profile.job_extract import parse_skills_required
    from app.retrieval.index import job_posting_text

    extracted = posting.extracted_json or {}
    if not extracted:
        return Query(text=posting.raw_text_quarantined or "")
    skills = tuple(s["skill"] for s in parse_skills_required(extracted.get("skills_required", [])))
    return Query(text=job_posting_text(posting), skills=skills)


def _as_query(query: str | Query) -> Query:
    return query if isinstance(query, Query) else Query(text=query)


def _sub_queries(query: Query) -> list[str]:
    """The full text first, then each named skill once, case-insensitively."""
    out = [query.text]
    seen = {query.text.casefold()}
    for skill in query.skills:
        key = skill.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(skill.strip())
        if len(out) > _MAX_SKILL_QUERIES:
            break
    return out


def _fuse(runs: list[list[int]]) -> list[tuple[int, float]]:
    """Reciprocal rank fusion, scaled by the best achievable score so the
    result sits in (0, 1]. Ties break by id, so equal documents always come
    back in the same order."""
    scores: dict[int, float] = {}
    for run in runs:
        for rank, doc_id in enumerate(run):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    best = len(runs) / (_RRF_K + 1)
    return sorted(
        ((doc_id, score / best) for doc_id, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _document_text(payload: dict[str, Any]) -> str:
    """The text a stored point was embedded from, rebuilt from its payload:
    an experience point carries its text, a skill claim its two fields."""
    from app.retrieval.index import evidence_document

    if "text" in payload:
        return str(payload["text"])
    return evidence_document(str(payload.get("skill", "")), str(payload.get("evidence_type", "")))


def _hybrid_search(
    collection: str,
    query: Query,
    account_id: int,
    top_k: int,
    must: Sequence[Any] | None = None,
) -> list[Hit]:
    """Dense and keyword runs for every sub-query over the account's
    points in this collection, fused. The keyword corpus is the account's
    own points, read back from Qdrant with their payloads, so it is
    exactly the set the dense side searches and needs no database."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.retrieval.keyword import Bm25Corpus

    client = get_client()
    if not client.collection_exists(collection):
        return []
    conditions: list[Any] = [FieldCondition(key="account_id", match=MatchValue(value=account_id))]
    conditions.extend(must or [])
    scope = Filter(must=conditions)

    payloads: dict[int, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scope,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payloads[int(point.id)] = point.payload or {}
        if offset is None:
            break
    if not payloads:
        return []

    sub_queries = _sub_queries(query)
    vectors = embed([QUERY_INSTRUCTION + q for q in sub_queries])
    keyword = Bm25Corpus([(pid, _document_text(p)) for pid, p in sorted(payloads.items())])
    runs: list[list[int]] = []
    for text, vector in zip(sub_queries, vectors, strict=True):
        dense = client.query_points(
            collection_name=collection, query=vector, query_filter=scope, limit=_RUN_DEPTH
        )
        runs.append([int(p.id) for p in dense.points])
        runs.append(keyword.top_k(text, _RUN_DEPTH))
    return [
        Hit(id=doc_id, score=score, payload=payloads[doc_id])
        for doc_id, score in _fuse(runs)[:top_k]
        if doc_id in payloads
    ]


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
    query: str | Query,
    account_id: int,
    top_k: int = 10,
    source_type: str | None = None,
    mode: SearchMode = "hybrid",
) -> list[Hit]:
    """Searches the skill_evidence collection (both repo-linked and
    experience-linked rows share it, see index.py). source_type narrows to
    "repo" or "experience" when the caller only wants one kind; omitted,
    both are searched together. mode="dense" is the single-vector search
    this replaced, kept so the eval harness can report it as a reference.
    """
    from app.retrieval.index import COLLECTION

    must = None
    if source_type is not None:
        from qdrant_client.models import FieldCondition, MatchValue

        must = [FieldCondition(key="source_type", match=MatchValue(value=source_type))]
    if mode == "dense":
        return _search(COLLECTION, _as_query(query).text, account_id, top_k, must)
    return _hybrid_search(COLLECTION, _as_query(query), account_id, top_k, must)


def search_experience_points(
    query: str | Query,
    account_id: int,
    top_k: int = 10,
    experience_id: int | None = None,
    mode: SearchMode = "hybrid",
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
    if mode == "dense":
        return _search(EXPERIENCE_POINTS_COLLECTION, _as_query(query).text, account_id, top_k, must)
    return _hybrid_search(EXPERIENCE_POINTS_COLLECTION, _as_query(query), account_id, top_k, must)


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
