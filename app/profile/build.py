"""build_profile(repos) -> Profile, the skill-profile entrypoint.

Per repo: manifest-declared skills (free, deterministic) + README/description
skills (one bulk-tier LLM call, see extract.py for the fallback chain and
NoSourceTextError). Each claim gets a weight from `weighting.compute_weight`
(fork status, recency, commit volume, evidence type). Writes one
`SkillEvidence` row per claim, then aggregates across repos into
`Profile.skills_json` via noisy-OR: multiple independent pieces of evidence
for the same skill raise confidence with diminishing returns, rather than a
plain sum that could exceed 1.0 or a max that ignores corroboration.

Fault isolation: one repo's LLM call failing (auth revoked, malformed
input, an unexpected error specific to that repo) marks that repo
"failed" and moves on without losing any other repo's already-computed
evidence. Each repo commits independently, so a single exception can't
lose the entire batch.

Rate limit / budget exhaustion is not treated as "this repo failed"; it
means every remaining repo in this batch is about to fail the same way.
Continuing would spend one doomed call per remaining repo and fill each
with the same raw error. Instead, `BudgetExceededError`/`LLMRateLimitedError`
(app/core/llm.py) are caught separately: that one repo is marked
"rate_limited" with a clean message (not the raw exception), and the whole
batch stops. Every repo not yet attempted is left as it was (usually still
"pending"), so the next process-pending call picks up where this one
stopped without re-spending on repos that already succeeded.

Idempotent by default: a repo already `extracted` (or `no_signal`: no
README/description, nothing changed since) is skipped on a normal call,
so calling `build_profile` again doesn't re-spend LLM budget on repos
that already succeeded. `force=True` (or `reprocess_repo`) bypasses that,
for the UI's manual retry button.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import reduce

from sqlalchemy import delete, select

from app.core.db import Profile, ProjectLink, Repository, SkillEvidence, get_db
from app.core.llm import BudgetExceededError, LLMRateLimitedError
from app.profile.claims import LinkClaim, SkillClaim
from app.profile.extract import (
    NoSourceTextError,
    extract_links_from_repo,
    extract_skills_from_repo,
)
from app.profile.manifest_skills import skills_from_manifests
from app.profile.skill_review import name_key, rejected_keys, review_names
from app.profile.weighting import compute_weight

logger = logging.getLogger(__name__)

_SKIP_STATUSES = {"extracted", "no_signal"}
_ERROR_MESSAGE_LIMIT = 2000


def _write_claims(db, repo: Repository, claims: list[SkillClaim], now: dt.datetime) -> None:
    # Names already reviewed and rejected for this account never get written
    # again; new names are reviewed in one batch after extraction, see
    # review_skill_evidence.
    rejected = rejected_keys(db, repo.account_id)
    for claim in claims:
        if name_key(claim.skill) in rejected:
            continue
        weight = compute_weight(
            claim.evidence_type,
            claim.confidence,
            is_fork=repo.is_fork,
            last_commit_at=repo.last_commit_at,
            commits_authored=repo.commits_authored,
            now=now,
        )
        db.add(
            SkillEvidence(
                skill=claim.skill,
                repo_id=repo.id,
                evidence_type=claim.evidence_type,
                weight=weight,
                confidence=claim.confidence,
                source_files_json=claim.source_files,
            )
        )


def _write_links(db, repo: Repository, claims: list[LinkClaim]) -> None:
    # Dedupe against repo.url itself (already shown as the project's primary
    # link elsewhere in the UI, so a README that just links back to its own
    # GitHub repo shouldn't produce a redundant row) and against duplicates
    # within this same extraction pass.
    seen = {repo.url.strip().rstrip("/")} if repo.url else set()
    for claim in claims:
        key = claim.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        db.add(
            ProjectLink(
                repo_id=repo.id, label=claim.label, url=claim.url, source="readme_extracted"
            )
        )


def _process_repo(db, repo: Repository, now: dt.datetime) -> bool:
    """Mutates repo's status and this repo's SkillEvidence rows in place.
    Never raises: every failure mode ends in a status, not an exception,
    so the caller can commit unconditionally after this returns.

    Returns True if the caller should stop the whole batch now (rate
    limit / budget exhaustion hit on this repo, since every repo after it
    would fail the identical way) rather than continue to the next repo.
    False otherwise, including the normal "this repo failed for its own
    reasons, others might still succeed" case.

    Commits partway through, before calling extract_skills_from_repo. Not
    optional. That call reaches core.llm.complete(), which opens its own
    independent session to record the LLMCall row. If this function's own
    session still had an uncommitted delete/insert pending at that point,
    SQLite would deadlock by construction: the inner session waits on a
    write lock this (outer) session holds, and this session can't release
    it until the inner call returns.
    """
    # Manual evidence_type rows are hand-added on the project detail page
    # (app/api/projects.py add_skill): a person's own claim, not something
    # extraction produced, so Reprocess must not wipe it out from under them.
    db.execute(
        delete(SkillEvidence).where(
            SkillEvidence.repo_id == repo.id, SkillEvidence.evidence_type != "manual"
        )
    )
    # Same manual-vs-derived split as SkillEvidence above: a link someone
    # typed in by hand (or edited, see app/api/projects.py update_link) is
    # "manual" and survives Reprocess; only "readme_extracted" rows get
    # wiped and re-derived here.
    db.execute(
        delete(ProjectLink).where(ProjectLink.repo_id == repo.id, ProjectLink.source != "manual")
    )
    manifest_claims = skills_from_manifests(repo.manifests_json)  # never raises
    _write_claims(db, repo, manifest_claims, now)
    db.commit()  # release the write lock before the nested LLM-call session runs

    stop_batch = False
    try:
        llm_claims = extract_skills_from_repo(repo)
    except NoSourceTextError:
        repo.skill_extraction_status = "no_signal"
        repo.skill_extraction_error = None
        llm_claims = []
    except (BudgetExceededError, LLMRateLimitedError) as e:
        logger.warning("rate limit / budget hit on %s: %s; stopping batch", repo.full_name, e)
        repo.skill_extraction_status = "rate_limited"
        repo.skill_extraction_error = (
            f"{e} Processing stopped here so the remaining projects don't fail "
            "the same way; run it again later to continue."
        )
        llm_claims = []
        stop_batch = True
    except Exception as e:  # this repo's own problem, not the batch's
        logger.warning("skill extraction failed for %s: %s", repo.full_name, e)
        repo.skill_extraction_status = "failed"
        repo.skill_extraction_error = str(e)[:_ERROR_MESSAGE_LIMIT]
        llm_claims = []
    else:
        repo.skill_extraction_status = "extracted"
        repo.skill_extraction_error = None

    repo.skills_extracted_at = now
    _write_claims(db, repo, llm_claims, now)

    # Same first pass, separate LLM call (see extract.py's
    # extract_links_from_repo module docstring for why it's not folded into
    # the skills call above). Skipped when stop_batch is already set;
    # another call would just hit the same rate limit / budget cap.
    # Best-effort otherwise: a link-extraction failure is not this repo's
    # skill_extraction_status problem, so it's caught and dropped rather
    # than overwriting a status the skills block above already decided.
    if not stop_batch:
        try:
            link_claims = extract_links_from_repo(repo)
        except NoSourceTextError:
            link_claims = []
        except (BudgetExceededError, LLMRateLimitedError) as e:
            logger.warning("rate limit / budget hit extracting links for %s: %s", repo.full_name, e)
            link_claims = []
        except Exception as e:
            logger.warning("link extraction failed for %s: %s", repo.full_name, e)
            link_claims = []
        _write_links(db, repo, link_claims)

    return stop_batch


def _index_repo_evidence(db, repo: Repository) -> None:
    """Best-effort re-embed of this repo's current SkillEvidence rows into
    Qdrant, called right after each repo's write commits. Never raises: a
    Qdrant hiccup here must not lose or roll back the SQLite write that
    already committed, same posture as every other indexing call site in
    this codebase (app/api/experience.py, app/api/projects.py).
    """
    try:
        from app.retrieval.index import index_skill_evidence

        rows = list(
            db.execute(select(SkillEvidence).where(SkillEvidence.repo_id == repo.id)).scalars()
        )
        index_skill_evidence(rows, account_id=repo.account_id)
    except Exception:
        logger.exception("could not index skill evidence for repo id=%s", repo.id)


def _aggregate(evidence: list[SkillEvidence]) -> dict:
    by_skill: dict[str, list[tuple[float, str]]] = {}
    for row in evidence:
        by_skill.setdefault(row.skill, []).append((row.weight, row.evidence_type))

    skills_json = {}
    for skill, entries in by_skill.items():
        weights = [w for w, _ in entries]
        combined = 1.0 - reduce(lambda acc, w: acc * (1.0 - w), weights, 1.0)
        skills_json[skill] = {
            "weight": round(combined, 4),
            "repo_count": len(entries),
            "evidence_types": sorted({et for _, et in entries}),
        }
    return skills_json


def build_profile(
    repos: list[Repository], now: dt.datetime | None = None, force: bool = False
) -> Profile:
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    try:
        for i, repo_ref in enumerate(repos):
            # repo_ref may be detached (loaded in an earlier, now-closed
            # session); mutating it directly wouldn't be tracked by this
            # function's own session and would silently fail to persist.
            # Re-fetch within this session so status changes actually commit.
            repo = db.get(Repository, repo_ref.id)
            if repo is None:
                continue
            if not force and repo.skill_extraction_status in _SKIP_STATUSES:
                continue
            try:
                stop_batch = _process_repo(db, repo, now)
                db.commit()
            except Exception:
                # _process_repo shouldn't raise (it catches its own LLM
                # errors); this is a last-resort net so a truly unexpected
                # exception (e.g. a DB error) still doesn't lose the rest
                # of the batch's already-committed repos.
                logger.exception("unexpected error processing %s", repo.full_name)
                db.rollback()
                continue
            _index_repo_evidence(db, repo)
            if stop_batch:
                # Rate limit / budget hit: every repo after this one
                # would fail the identical way right now. Leave them
                # untouched (still "pending" in the common case) rather
                # than burn more doomed calls; the next process-pending
                # call picks up exactly where this one stopped.
                logger.warning(
                    "stopping skill-extraction batch early at %s; %d repo(s) left untouched",
                    repo.full_name,
                    len(repos) - (i + 1),
                )
                break

        db.commit()  # end any open read before review's nested LLM-call sessions
        review_skill_evidence([r.account_id for r in repos])
        evidence = list(
            db.execute(
                select(SkillEvidence).where(
                    SkillEvidence.repo_id.in_([r.id for r in repos])
                )
            ).scalars()
        )
        profile = Profile(skills_json=_aggregate(evidence))
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile
    finally:
        db.close()


def build_profile_progress(
    repos: list[Repository], now: dt.datetime | None = None, force: bool = False
):
    """Generator twin of build_profile(): yields progress per repo instead
    of only returning once the whole batch is done, for the background
    extraction job's SSE stream (see app/profile/jobs.py). Duplicated rather
    than shared with build_profile(), same reasoning as
    sync_account_progress() vs sync_account() in app/ingest/github/sync.py:
    keeps the plain (non-streaming) path used by tests/CLI untouched.

    Stages yielded, in order:
      {"stage": "repo_progress", "index": int, "total": int, "name": str,
       "status": str}  (repeated; status is the repo's
       skill_extraction_status after processing: "extracted"/"no_signal"/
       "failed", or "skipped" if it was already done and force=False)
      {"stage": "rate_limited", "index": int, "total": int} instead of the
      next repo_progress, same stop-the-batch reasoning as build_profile's
      stop_batch: every repo after this one would fail the same way
      right now, so they're left untouched for a later run to pick up.
      {"stage": "done", "total": int, "processed": int}; processed counts
      only repos actually sent to _process_repo, not skipped ones.
    """
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    total = len(repos)
    processed = 0
    try:
        for i, repo_ref in enumerate(repos, start=1):
            repo = db.get(Repository, repo_ref.id)
            if repo is None:
                continue
            if not force and repo.skill_extraction_status in _SKIP_STATUSES:
                yield {
                    "stage": "repo_progress",
                    "index": i,
                    "total": total,
                    "name": repo.full_name,
                    "status": "skipped",
                }
                continue
            try:
                stop_batch = _process_repo(db, repo, now)
                db.commit()
            except Exception:
                logger.exception("unexpected error processing %s", repo.full_name)
                db.rollback()
                yield {
                    "stage": "repo_progress",
                    "index": i,
                    "total": total,
                    "name": repo.full_name,
                    "status": "failed",
                }
                continue
            _index_repo_evidence(db, repo)
            processed += 1
            yield {
                "stage": "repo_progress",
                "index": i,
                "total": total,
                "name": repo.full_name,
                "status": repo.skill_extraction_status,
            }
            if stop_batch:
                yield {"stage": "rate_limited", "index": i, "total": total}
                db.commit()
                review_skill_evidence([r.account_id for r in repos])
                return

        db.commit()  # end any open read before review's nested LLM-call sessions
        review_skill_evidence([r.account_id for r in repos])
        evidence = list(
            db.execute(
                select(SkillEvidence).where(
                    SkillEvidence.repo_id.in_([r.id for r in repos])
                )
            ).scalars()
        )
        profile = Profile(skills_json=_aggregate(evidence))
        db.add(profile)
        db.commit()
    finally:
        db.close()
    yield {"stage": "done", "total": total, "processed": processed}


def reprocess_repo(repo_id: int, now: dt.datetime | None = None) -> Repository:
    """Manual retry for one repo (the UI's "Reprocess" button). Always
    forces re-extraction, regardless of current status.
    """
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise ValueError(f"no repository with id={repo_id}")
        _process_repo(db, repo, now)
        db.commit()
        _index_repo_evidence(db, repo)
        db.commit()
        review_skill_evidence([repo.account_id])
        db.refresh(repo)
        return repo
    finally:
        db.close()


def _delete_index_points(evidence_ids: list[int]) -> None:
    """Best-effort Qdrant cleanup for evidence rows this module deleted,
    same posture as app/api/projects.py's _delete_qdrant_points."""
    if not evidence_ids:
        return
    try:
        from app.retrieval.index import COLLECTION
        from app.retrieval.vectorstore import get_client

        client = get_client()
        if client.collection_exists(COLLECTION):
            client.delete(collection_name=COLLECTION, points_selector=evidence_ids)
    except Exception:
        logger.exception("could not clean up Qdrant points for replaced manifest evidence")


def review_skill_evidence(account_ids: list[int | None]) -> int:
    """Runs every not-yet-reviewed, automatically found skill name for these
    accounts past the LLM review (app/profile/skill_review.py), then deletes
    the evidence rows whose name was rejected. Hand-added ("manual") rows are
    never reviewed or deleted. Never raises; returns rows removed.
    """
    removed = 0
    db = get_db()
    try:
        for account_id in {a for a in account_ids if a is not None}:
            try:
                rows = db.execute(
                    select(SkillEvidence.id, SkillEvidence.skill)
                    .join(Repository, SkillEvidence.repo_id == Repository.id)
                    .where(
                        Repository.account_id == account_id,
                        SkillEvidence.evidence_type != "manual",
                    )
                ).all()
                rejected = review_names(db, account_id, [skill for _, skill in rows])
                stale_ids = [row_id for row_id, skill in rows if name_key(skill) in rejected]
                if not stale_ids:
                    continue
                db.execute(delete(SkillEvidence).where(SkillEvidence.id.in_(stale_ids)))
                db.commit()
                removed += len(stale_ids)
                _delete_index_points(stale_ids)
            except Exception:
                logger.exception("skill review failed for account id=%s", account_id)
                db.rollback()
        return removed
    finally:
        db.close()


def rebuild_manifest_evidence(now: dt.datetime | None = None) -> tuple[int, int]:
    """Re-derives every repo's declared_dependency rows from the manifests
    already stored on it, with no LLM call and no GitHub fetch. For when
    manifest_skills.py's package-to-skill mapping changes: README-derived
    and manual rows, extraction status, and links are all left alone.

    Returns (rows removed, rows written).
    """
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    removed = written = 0
    try:
        repos = list(db.execute(select(Repository)).scalars())
        for repo in repos:
            stale = SkillEvidence.repo_id == repo.id, SkillEvidence.evidence_type == "declared_dependency"
            old_ids = list(db.execute(select(SkillEvidence.id).where(*stale)).scalars())
            db.execute(delete(SkillEvidence).where(*stale))
            claims = skills_from_manifests(repo.manifests_json or {})
            _write_claims(db, repo, claims, now)
            db.commit()
            removed += len(old_ids)
            written += len(claims)
            _delete_index_points(old_ids)
            _index_repo_evidence(db, repo)
        return removed, written
    finally:
        db.close()


def skill_evidence_for_repos(repo_ids: list[int]) -> list[SkillEvidence]:
    """Fresh query rather than returning the rows built inside
    `build_profile`, since those get detached the moment that function's session
    closes. Used to hand evidence off to `app.retrieval.index` for
    embedding; kept here rather than in `retrieval` since it's a plain
    `core`-only read, no embedding/Qdrant involved.
    """
    if not repo_ids:
        return []
    db = get_db()
    try:
        return list(
            db.execute(
                select(SkillEvidence).where(SkillEvidence.repo_id.in_(repo_ids))
            ).scalars()
        )
    finally:
        db.close()
