"""Canonical job-title clustering: "ML Engineer", "Machine Learning
Engineer", "Applied ML Engineer" all resolve to the same RoleFamily row,
so job-posting analytics (app/api/job_analytics.py) roll up by what a role
actually is, not by exact title string. Global across accounts, same
posture as JobPosting.content_hash's shared-cache reasoning: a title's
canonical family doesn't depend on which local profile pasted the posting.

Retrieval-first, not an LLM call on every title: resolve_role_family()
embeds the raw title and searches the `role_families` Qdrant collection
(app/retrieval/index.py's index_role_family) for a close-enough existing
cluster before ever asking an LLM. Only a new cluster costs one
cheap bulk-tier call to produce a clean canonical name. This is
a nearest-neighbor lookup, not string matching: "ML Engineer"
and "Machine Learning Engineer" don't share enough characters for a
fuzzy-string match to reliably group them, but they embed close together.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.db import RoleFamily, get_db
from app.core.llm import complete, system_message, user_message

logger = logging.getLogger(__name__)

# Cosine similarity floor for "this is the same role family", picked
# conservatively (favors creating a new family over silently merging two
# different roles); not tuned against a labeled set, worth
# revisiting once real job-posting data accumulates.
_SIMILARITY_THRESHOLD = 0.86

_CANONICALIZE_SCHEMA = {
    "type": "object",
    "properties": {"canonical_name": {"type": "string"}},
    "required": ["canonical_name"],
}

_CANONICALIZE_SYSTEM_PROMPT = (
    "You turn a raw job title into a clean, canonical role name for "
    "grouping near-duplicate titles (e.g. \"ML Engineer\", \"Machine "
    "Learning Engineer\", \"Applied ML Engineer\" should all map to the "
    "same canonical name). Return the most standard, widely recognized "
    "industry title that this specific posting's title actually means, "
    "not a broader category (don't collapse \"Backend Engineer\" into "
    "\"Software Engineer\"), not a narrower guess than the title supports. "
    "Title case, no company-specific branding or internal leveling codes "
    "(\"Senior Software Engineer\" stays as given; strip something like "
    "\"L5\" or \"IC4\" that only means something inside one company)."
)


class RoleFamilyResolutionError(Exception):
    """Raised internally when the canonicalization LLM call fails or
    returns nothing usable; never escapes resolve_role_family() itself,
    which always degrades to returning None rather than raising, since
    role-family resolution is always a best-effort enrichment (see
    app/api/job_postings.py's call site); a posting must still save fine
    with role_family_id left null."""


def resolve_role_family(title: str, account_id: int | None = None) -> RoleFamily | None:
    """Given a raw job title, returns the RoleFamily it belongs to: reuses
    an existing close-enough one when the similarity search finds one,
    otherwise asks the LLM for a clean canonical name and creates a new
    row. Returns None (never raises) for a blank title or if resolution
    couldn't complete for any reason (embedder unavailable, Qdrant
    unreachable, LLM call failed); always best-effort, never something
    that should block saving a job posting.
    """
    title = title.strip()
    if not title:
        return None

    from app.retrieval.search import search_role_families

    try:
        hits = search_role_families(title, top_k=1)
    except Exception:
        logger.warning("role family search failed for title=%r", title, exc_info=True)
        hits = []

    if hits and hits[0].score >= _SIMILARITY_THRESHOLD:
        db = get_db()
        try:
            existing = db.get(RoleFamily, hits[0].id)
            if existing is not None:
                return existing
        finally:
            db.close()

    try:
        response = complete(
            "bulk",
            [
                system_message(_CANONICALIZE_SYSTEM_PROMPT),
                user_message(f"Job title: {title}"),
            ],
            schema=_CANONICALIZE_SCHEMA,
            account_id=account_id,
            purpose="role_family",
        )
        if response.parsed is None:
            raise RoleFamilyResolutionError("LLM response was not valid JSON")
        canonical_name = str(response.parsed.get("canonical_name", "")).strip()
        if not canonical_name:
            raise RoleFamilyResolutionError("LLM returned an empty canonical_name")
    except Exception:
        logger.warning("role family canonicalization failed for title=%r", title, exc_info=True)
        return None

    db = get_db()
    try:
        row = RoleFamily(canonical_name=canonical_name)
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            # Lost a race with a concurrent resolution creating the exact
            # same canonical_name; reuse the winner's row instead of
            # erroring out of what's still a best-effort enrichment.
            db.rollback()
            return db.execute(
                select(RoleFamily).where(RoleFamily.canonical_name == canonical_name)
            ).scalar_one_or_none()
        db.refresh(row)
    finally:
        db.close()

    try:
        from app.retrieval.index import index_role_family

        index_role_family(row)
    except Exception:
        logger.warning("could not index new role family id=%s", row.id, exc_info=True)

    return row


def list_role_families() -> list[RoleFamily]:
    db = get_db()
    try:
        return list(
            db.execute(select(RoleFamily).order_by(RoleFamily.canonical_name)).scalars()
        )
    finally:
        db.close()
