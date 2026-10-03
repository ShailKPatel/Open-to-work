"""Builds the skill profile from synced repos.

Per repo: manifest skills (deterministic, free) plus README skills and
links (one bulk-tier call, see extract.py). _prefetch_facts runs the
batched pass first; any repo it misses gets its own call, so nothing
depends on the batch succeeding.

Each claim is weighted by weighting.compute_weight (fork, recency, commit
volume, evidence type) and stored as a SkillEvidence row. Evidence for
the same skill across repos combines by noisy-OR into
Profile.skills_json: corroboration raises confidence with diminishing
returns and never past 1.0.

A failure specific to one repo marks it "failed" and the batch moves on;
each repo commits on its own. Running out of budget, rate limit or keys
is different, since every remaining repo would fail the same way: that
repo is marked "rate_limited" and the batch stops, leaving the rest
pending for the next run.

Repos already extracted, or with nothing to extract, are skipped unless
force=True (the UI's retry), so a rerun does not spend twice.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from functools import reduce
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.db import (
    Profile,
    ProjectLink,
    Repository,
    SkillEvidence,
    get_db,
    is_profile_repo,
)
from app.core.llm import BudgetExceededError, LLMRateLimitedError, is_out_of_keys
from app.profile.claims import LinkClaim, SkillClaim
from app.profile.extract import (
    NoSourceTextError,
    RepoFacts,
    extract_repo_facts,
    prefetch_repo_facts,
)
from app.profile.manifest_skills import skills_from_manifests
from app.profile.skill_review import name_key, rejected_keys, review_names
from app.profile.weighting import compute_weight

logger = logging.getLogger(__name__)

_SKIP_STATUSES = {"extracted", "no_signal"}
_ERROR_MESSAGE_LIMIT = 2000


def _write_claims(
    db: Session, repo: Repository, claims: list[SkillClaim], now: dt.datetime
) -> None:
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


def _write_links(db: Session, repo: Repository, claims: list[LinkClaim]) -> None:
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


def _process_repo(
    db: Session,
    repo: Repository,
    now: dt.datetime,
    prefetched: dict[int, RepoFacts] | None = None,
    batch: bool = True,
) -> bool:
    """Extracts one repo's skills and links, updating its status and evidence
    rows. Never raises; every outcome ends as a status, so the caller can
    commit afterwards. Returns True when the batch should stop (budget, rate
    limit or keys exhausted), since every later repo would fail the same way.

    prefetched holds what the batched pass already got; a repo it missed
    gets its own call. batch=False (a single Reprocess) leaves out the
    note about stopping the remaining projects, since there are none.

    Commits before calling the LLM: complete() records the call in its own
    session, and SQLite would deadlock on a write lock this session still
    held.
    """
    # A profile README repo saved as a project before the flag existed
    # moves out of the projects list the moment it is (re)processed.
    repo.is_profile_readme = is_profile_repo(repo.full_name)
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
    facts = (prefetched or {}).get(repo.id)
    if facts is not None:
        # Already answered by the batched pass, no call of its own.
        repo.skill_extraction_status = "extracted"
        repo.skill_extraction_error = None
    else:
        try:
            facts = extract_repo_facts(repo)
        except NoSourceTextError:
            repo.skill_extraction_status = "no_signal"
            repo.skill_extraction_error = None
            facts = RepoFacts()
        except (BudgetExceededError, LLMRateLimitedError) as e:
            logger.warning("rate limit / budget hit on %s: %s; stopping batch", repo.full_name, e)
            repo.skill_extraction_status = "rate_limited"
            repo.skill_extraction_error = (
                f"{e} Processing stopped here so the remaining projects don't fail "
                "the same way; run it again later to continue."
                if batch
                else str(e)
            )
            facts = RepoFacts()
            stop_batch = True
        except Exception as e:
            if is_out_of_keys(e):
                # Every key for the provider is spent (see
                # app/core/llm.py), not something wrong with this repo.
                # Same posture as the rate-limit case above: stop, so the
                # repos after it keep their pending status and the next
                # run continues from here.
                logger.warning("out of usable keys on %s: %s; stopping batch", repo.full_name, e)
                repo.skill_extraction_status = "rate_limited"
                repo.skill_extraction_error = (
                    f"{e} Processing stopped here so the remaining projects don't fail "
                    "the same way; run it again once a key is available to continue."
                    if batch
                    else str(e)
                )[:_ERROR_MESSAGE_LIMIT]
                stop_batch = True
            else:  # this repo's own problem, not the batch's
                logger.warning("skill extraction failed for %s: %s", repo.full_name, e)
                repo.skill_extraction_status = "failed"
                repo.skill_extraction_error = str(e)[:_ERROR_MESSAGE_LIMIT]
            facts = RepoFacts()
        else:
            repo.skill_extraction_status = "extracted"
            repo.skill_extraction_error = None

    repo.skills_extracted_at = now
    _write_claims(db, repo, facts.skills, now)
    # Same call, no second dispatch: skills and links come back together
    # (see extract.py's module docstring).
    _write_links(db, repo, facts.links)

    return stop_batch


def _index_repo_evidence(db: Session, repo: Repository) -> None:
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


def _prefetch_facts(db: Session, repos: list[Repository], force: bool) -> dict[int, RepoFacts]:
    """One batched extraction pass, ahead of the per-repo loop, over the
    repos this run will actually process (the same skip rule the loop
    itself applies, so a batch of already-extracted repos spends nothing).

    Below two repos there is nothing to amortize, so the batched prompt's
    extra instructions would cost more than the call it saves; the loop's
    own per-repo calls handle those.

    Commits first for the same reason the loop body does: the batched call
    opens its own session to record its LLMCall row, and an open write
    transaction on this session would deadlock against it (see
    _process_repo).
    """
    pending: list[Repository] = []
    for repo_ref in repos:
        repo = db.get(Repository, repo_ref.id)
        if repo is None:
            continue
        if not force and repo.skill_extraction_status in _SKIP_STATUSES:
            continue
        pending.append(repo)
    if len(pending) < 2:
        return {}
    db.commit()
    return prefetch_repo_facts(pending)


def build_profile(
    repos: list[Repository], now: dt.datetime | None = None, force: bool = False
) -> Profile:
    """Blocking form of build_profile_progress; returns the profile snapshot."""
    profile_id = None
    for event in build_profile_progress(repos, now, force):
        profile_id = event.get("profile_id", profile_id)
    db = get_db()
    try:
        profile = db.get(Profile, profile_id)
        assert profile is not None
        return profile
    finally:
        db.close()


def build_profile_progress(
    repos: list[Repository], now: dt.datetime | None = None, force: bool = False
) -> Iterator[dict[str, Any]]:
    """Extracts skills for each repo, yielding progress as it goes:

      {"stage": "repo_progress", "index", "total", "name", "status"}  (repeated)
      {"stage": "done", "total", "processed", "profile_id"}

    status is the repo's extraction status afterwards, or "skipped" when it
    was already done and force is False. When the budget, rate limit or keys
    run out, the batch stops and ends with {"stage": "rate_limited",
    "index", "total", "profile_id"} instead; the repos after it stay pending
    for the next run. Either way the profile snapshot is rebuilt from the
    evidence written so far.
    """
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    total = len(repos)
    processed = 0
    stopped_at: int | None = None
    try:
        prefetched = _prefetch_facts(db, repos, force)
        for i, repo_ref in enumerate(repos, start=1):
            # repo_ref may belong to a closed session; re-read it here so
            # status changes are tracked and committed.
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
                stop_batch = _process_repo(db, repo, now, prefetched)
                db.commit()
            except Exception:
                # _process_repo catches its own LLM errors; this keeps an
                # unexpected one (a database error, say) from losing the batch.
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
                logger.warning(
                    "stopping skill-extraction batch early at %s; %d repo(s) left untouched",
                    repo.full_name,
                    total - i,
                )
                stopped_at = i
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
        profile_id = profile.id
    finally:
        db.close()
    if stopped_at is not None:
        yield {
            "stage": "rate_limited",
            "index": stopped_at,
            "total": total,
            "profile_id": profile_id,
        }
    else:
        yield {
            "stage": "done",
            "total": total,
            "processed": processed,
            "profile_id": profile_id,
        }


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
        _process_repo(db, repo, now, batch=False)
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
            stale = (
                SkillEvidence.repo_id == repo.id,
                SkillEvidence.evidence_type == "declared_dependency",
            )
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


def refresh_repo_evidence(repo_id: int, now: dt.datetime | None = None) -> None:
    """For a repo that was pushed to without its README changing: the
    LLM-derived rows still hold, but the manifests may have changed and
    the commit count and recency that weight every row have. Rewrites the
    declared_dependency rows from the stored manifests and reweights the
    README/description rows from their stored confidence. No LLM call, no
    GitHub fetch; manual rows and links are left alone."""
    now = now or dt.datetime.now(dt.UTC)
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            return
        stale = (
            SkillEvidence.repo_id == repo.id,
            SkillEvidence.evidence_type == "declared_dependency",
        )
        old_ids = list(db.execute(select(SkillEvidence.id).where(*stale)).scalars())
        db.execute(delete(SkillEvidence).where(*stale))
        _write_claims(db, repo, skills_from_manifests(repo.manifests_json or {}), now)
        described = db.execute(
            select(SkillEvidence).where(
                SkillEvidence.repo_id == repo.id,
                SkillEvidence.evidence_type.in_(("readme_described", "description_described")),
            )
        ).scalars()
        for row in described:
            row.weight = compute_weight(
                row.evidence_type,  # type: ignore[arg-type]
                row.confidence,
                is_fork=repo.is_fork,
                last_commit_at=repo.last_commit_at,
                commits_authored=repo.commits_authored,
                now=now,
            )
        db.commit()
        _delete_index_points(old_ids)
        _index_repo_evidence(db, repo)
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
