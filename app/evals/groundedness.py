"""Groundedness scoring, computed through this app's own LLM client
rather than the `ragas` package.

RAGAS makes its own LLM-judge calls through LangChain, outside
app/core/llm.py's cache, budget, attribution, and key-resolution layer, and
pulls in a large dependency chain (langchain, datasets, pandas). The judge
calls here go through complete() like every other LLM call instead.

Precision@10 needs no LLM call at all: app/evals/run.py computes it
directly from precision_at_k against the golden set's hand-labeled ground
truth. Groundedness is the one number that needs a judge call: whether a
generated resume bullet is supported by the project evidence it was
generated from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Repository, Resume, get_db
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMRateLimitedError,
    complete,
    system_message,
    user_message,
)
from app.resume_build.orchestrator import format_project_evidence, project_skills

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}},
    "required": ["grounded"],
}

_SYSTEM_PROMPT = (
    "You judge whether a single resume bullet point is factually "
    "supported by the evidence provided about the project it describes. "
    "Answer `grounded: true` only if every concrete claim in the bullet "
    "(technology used, what was built, scale or impact stated) is either "
    "directly stated in the evidence or a reasonable, conservative "
    "paraphrase of it. Answer `grounded: false` if the bullet states "
    "anything the evidence doesn't support, even if it sounds plausible. "
    "A user note is the account holder's own statement about the project "
    "and counts as evidence."
)

_DEFAULT_SAMPLE_SIZE = 5
_DEFAULT_MAX_CHECKS = 20


@dataclass
class GroundednessResult:
    score: float | None
    checked: int
    skipped_reason: str | None = None


def _project_evidence_text(
    db: Session, account_id: int, repo_id: int, user_note: str | None
) -> str:
    """The project as the resume writer was shown it, read back from the
    account's current data: the same truncated description, the same
    non-archived skills and the note saved with the project. No README,
    since the writer never saw one."""
    repo = db.get(Repository, repo_id)
    if repo is None or repo.account_id != account_id:
        return ""
    skills = project_skills(db, account_id, [repo_id]).get(repo_id, [])
    return format_project_evidence(repo.name, repo.description or "", skills, user_note)


def judge_bullet(evidence: str, bullet: str, account_id: int | None) -> bool | None:
    """The judge's verdict on one bullet against its project's evidence:
    True grounded, False not, None when the reply was not usable JSON.
    Provider errors (no key, budget, rate limit) propagate to the caller.
    One function so the eval that validates the judge against human labels
    (app/evals/llm_evals.py) runs exactly the judge this module uses."""
    response = complete(
        "bulk",
        [
            system_message(_SYSTEM_PROMPT),
            user_message(f"Evidence:\n{evidence}\n\nBullet: {bullet}"),
        ],
        schema=_SCHEMA,
        account_id=account_id,
        purpose="groundedness_eval",
    )
    if response.parsed is None:
        return None
    return bool(response.parsed.get("grounded", False))


def score_groundedness(
    account_id: int,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    max_checks: int = _DEFAULT_MAX_CHECKS,
) -> GroundednessResult:
    """Checks up to `max_checks` project bullets, drawn from the
    `sample_size` most-recently-generated resumes for this account,
    against their own project's real evidence (the description, skills and
    user note the writer was given, see _project_evidence_text()), one
    bulk-tier LLM judge call per bullet, a true/false factual check being
    the kind of classification the bulk tier is for. `max_checks` bounds
    worst-case cost/latency regardless of how many resumes or bullets
    exist; it is not a claim that a larger sample wouldn't be more
    reliable, just a ceiling on what one eval run spends.

    Returns score=None (not 0.0) when nothing could be checked at all
    (no generated resumes exist yet, or every judge call failed before one
    succeeded), so app/evals/run.py can tell "measured as ungrounded"
    apart from "never measured." A budget cap or missing API key hit
    mid-run stops the check early and still returns whatever was measured
    before that point, same "keep what's already good" posture
    app/profile/build.py's rate-limit handling uses.
    """
    db = get_db()
    try:
        resumes = list(
            db.execute(
                select(Resume)
                .where(Resume.account_id == account_id, Resume.content_json.is_not(None))
                .order_by(Resume.uploaded_at.desc())
                .limit(sample_size)
            ).scalars()
        )
        if not resumes:
            return GroundednessResult(
                score=None, checked=0,
                skipped_reason="no generated resumes with content_json for this account yet",
            )

        outcomes: list[bool] = []
        for resume in resumes:
            projects = (resume.content_json or {}).get("projects", [])
            for project in projects:
                if len(outcomes) >= max_checks:
                    break
                repo_id = project.get("repo_id")
                points = project.get("points") or []
                if repo_id is None or not points:
                    continue
                evidence = _project_evidence_text(
                    db, account_id, repo_id, project.get("user_note")
                )
                if not evidence.strip():
                    continue
                for bullet in points:
                    if len(outcomes) >= max_checks:
                        break
                    try:
                        verdict = judge_bullet(evidence, bullet, account_id)
                    except (ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError) as e:
                        logger.info("groundedness check stopped early: %s", e)
                        if outcomes:
                            return GroundednessResult(
                                score=sum(outcomes) / len(outcomes),
                                checked=len(outcomes),
                                skipped_reason=f"stopped early: {e}",
                            )
                        return GroundednessResult(score=None, checked=0, skipped_reason=str(e))
                    if verdict is not None:
                        outcomes.append(verdict)

        if not outcomes:
            return GroundednessResult(
                score=None, checked=0,
                skipped_reason="no project bullets with checkable evidence found",
            )
        return GroundednessResult(score=sum(outcomes) / len(outcomes), checked=len(outcomes))
    finally:
        db.close()
