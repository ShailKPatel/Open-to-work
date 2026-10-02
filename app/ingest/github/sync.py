"""sync_account(username) -> list[Repository]

Fetches repos, README, manifests, and authorship stats; upserts into SQLite
by github_id. Skips README/manifest/stats refetch when pushed_at is
unchanged since the last sync (cache hit); the cheap repo-list call still runs
every time to detect what changed, and is also what finds new repos.

A pushed repo only goes back to skill extraction when its extraction
source changed: the README (judged by its git blob SHA, which the root
listing carries for free) or, for a repo without one, its description.
Any other push refreshes manifests and commit stats and reweights the
existing evidence in code (build.py's refresh_repo_evidence), no LLM call.

A changed repo costs two REST calls: the root listing (README and manifest
bodies then come from raw download URLs, outside the API quota) and one
commits query for authorship. Without a token GitHub allows 60 calls an
hour, so every call per repo matters.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from github import GithubException, RateLimitExceededException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Repository, get_db, is_profile_repo
from app.ingest.github.cancellation import clear as clear_cancel
from app.ingest.github.cancellation import is_cancelled
from app.ingest.github.client import GitHubClient
from app.ingest.github.manifests import MANIFEST_FILENAMES, parse_dependencies

logger = logging.getLogger(__name__)


@dataclass
class SyncSummary:
    total_repos: int
    fetched: int
    cache_hits: int


def _readme_rank(name: str) -> int | None:
    """Root README candidates, README.md first. None if not a README."""
    lower = name.lower()
    if lower == "readme.md":
        return 0
    if lower == "readme" or lower.startswith("readme."):
        return 1
    return None


def git_blob_sha(text: str) -> str:
    """The SHA git (and GitHub's contents listing) gives a file with this
    content, so a stored README can be checked against the listing without
    downloading it again."""
    data = text.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _known_readme_sha(repo: Repository | None) -> str | None:
    if repo is None:
        return None
    if repo.readme_sha:
        return repo.readme_sha
    # synced before readme_sha existed
    return git_blob_sha(repo.readme) if repo.readme else None


def _fetch_readme_and_manifests(
    client: GitHubClient, gh_repo: Any, existing: Repository | None = None
) -> tuple[str | None, str | None, dict]:
    """(readme, readme_sha, manifests). A root README whose SHA matches
    the stored one is not downloaded again; the stored text is reused."""
    manifests: dict[str, dict] = {}
    readme_entry = None
    readme_rank = 0
    for entry in client.root_contents(gh_repo):
        if entry.type != "file":
            continue
        rank = _readme_rank(entry.name)
        if rank is not None:
            if readme_entry is None or rank < readme_rank:
                readme_entry, readme_rank = entry, rank
            continue
        if entry.name not in MANIFEST_FILENAMES:
            continue
        content = client.entry_text(gh_repo, entry)
        if content is None:
            continue
        manifests[entry.name] = {
            "ecosystem": MANIFEST_FILENAMES[entry.name],
            "dependencies": parse_dependencies(entry.name, content),
        }
    if readme_entry is not None:
        sha = getattr(readme_entry, "sha", None)
        if sha and existing is not None and sha == _known_readme_sha(existing):
            return existing.readme, sha, manifests
        readme = client.entry_text(gh_repo, readme_entry)
    else:
        # GitHub also finds a README under docs/ or .github/
        sha = None
        readme = client.readme_text(gh_repo)
    if readme is not None and not sha:
        sha = git_blob_sha(readme)
    return readme, sha, manifests


def _source_changed(existing: Repository, readme: str | None, description: str | None) -> bool:
    """Whether what skill extraction reads (extract.py: the README, else
    the description) is different from what it last read."""
    if (readme or "").strip() != (existing.readme or "").strip():
        return True
    if (readme or "").strip():
        return False
    return (description or "").strip() != (existing.description or "").strip()


# _upsert outcomes
CACHE_HIT = "cache_hit"  # not pushed since the last sync, nothing fetched
REFRESHED = "refreshed"  # pushed, README unchanged: stats and manifests only
FETCHED = "fetched"  # new repo or new README: goes back to skill extraction


def _upsert(
    db: Session, gh_repo: Any, client: GitHubClient, username: str, account_id: int | None
) -> str:
    """Returns CACHE_HIT, REFRESHED or FETCHED."""
    existing = db.execute(
        select(Repository).where(Repository.github_id == gh_repo.id)
    ).scalar_one_or_none()

    pushed_at = gh_repo.pushed_at
    if pushed_at and pushed_at.tzinfo is None:
        pushed_at = pushed_at.replace(tzinfo=dt.UTC)

    # SQLite drops tzinfo on round-trip (DateTime(timezone=True) is best-effort
    # there), so an aware-vs-naive `==` would silently read as "changed" every
    # time. Compare on naive UTC instants instead.
    def _naive_utc(value: dt.datetime | None) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(dt.UTC).replace(tzinfo=None)
        return value

    if existing is not None and _naive_utc(existing.pushed_at) == _naive_utc(pushed_at):
        existing.stars = gh_repo.stargazers_count
        existing.is_profile_readme = is_profile_repo(gh_repo.full_name)
        existing.fetched_at = dt.datetime.now(dt.UTC)
        return CACHE_HIT

    readme, readme_sha, manifests = _fetch_readme_and_manifests(client, gh_repo, existing)
    authored = client.authored_commits(gh_repo, username)

    if existing is None:
        changed = True
        existing = Repository(github_id=gh_repo.id, account_id=account_id)
        db.add(existing)
    else:
        changed = _source_changed(existing, readme, gh_repo.description)
        if account_id is not None:
            existing.account_id = account_id

    existing.name = gh_repo.name
    existing.full_name = gh_repo.full_name
    existing.is_profile_readme = is_profile_repo(gh_repo.full_name)
    existing.url = gh_repo.html_url
    existing.is_fork = gh_repo.fork
    existing.primary_language = gh_repo.language
    existing.stars = gh_repo.stargazers_count
    existing.readme = readme
    existing.readme_sha = readme_sha
    existing.description = gh_repo.description
    existing.manifests_json = manifests
    # None means GitHub couldn't say; don't clobber a known value with zero.
    if authored is not None:
        existing.commits_authored, existing.last_commit_at = authored
    existing.pushed_at = pushed_at
    existing.fetched_at = dt.datetime.now(dt.UTC)
    if not changed:
        return REFRESHED
    # source changed (or this is a new repo): any prior extraction is
    # stale, whether it previously succeeded, failed, or had no signal.
    existing.skill_extraction_status = "pending"
    existing.skill_extraction_error = None
    return FETCHED


def _save(
    db: Session, gh_repo: Any, client: GitHubClient, username: str, account_id: int | None
) -> bool:
    """Upserts and commits one repo. Returns True on a cache hit."""
    outcome = _upsert(db, gh_repo, client, username, account_id)
    db.commit()
    if outcome == REFRESHED:
        repo_id = db.execute(
            select(Repository.id).where(Repository.github_id == gh_repo.id)
        ).scalar_one()
        db.commit()  # end the read before refresh opens its own session
        # imported here to keep the LLM stack out of this module's import
        from app.profile.build import refresh_repo_evidence

        refresh_repo_evidence(repo_id)
    return outcome == CACHE_HIT


def _rate_limited_event(e: GithubException, completed: int, total_hint: int) -> dict[str, Any]:
    """reset_at is set when the hourly quota is spent, so the UI can say
    when a retry will work instead of just "later"."""
    reset_at = getattr(e, "reset_at", None)
    if reset_at is not None:
        detail = (
            f"GitHub's hourly request limit is used up: {completed} of "
            f"{total_hint} repos saved. Sync again after the limit resets "
            "to pick up the rest; saved repos are skipped."
        )
    else:
        detail = (
            f"GitHub may be rate-limiting us: {completed} of "
            f"{total_hint} repos saved. Try syncing again "
            "later to pick up the rest."
        )
    return {
        "stage": "rate_limited",
        "completed": completed,
        "total_hint": total_hint,
        "detail": detail,
        "reset_at": reset_at.isoformat() if reset_at is not None else None,
    }


def sync_account_progress(
    username: str,
    include_forks: bool = True,
    client: GitHubClient | None = None,
    account_id: int | None = None,
    run_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Generator twin of sync_account() for the SSE progress endpoint:
    yields dicts as each stage happens instead of returning once at the end.
    Same upsert logic, duplicated rather than shared with sync_account()
    below so the plain (non-streaming) path used by the CLI and by existing
    mocked tests stays untouched.

    Stages yielded, in order:
      {"stage": "checking_profile", "username": ...}
      {"stage": "listing_repos", "total_hint": int}
      {"stage": "repo_progress", "index": int, "total_hint": int, "name": str,
       "cache_hit": bool}  (repeated)
      {"stage": "done", "total_repos": int, "fetched": int, "cache_hits": int}

    Or, if GitHub starts blocking us partway through a big account (rate
    limit / secondary rate limit hit mid-batch, after some repos already
    committed): {"stage": "rate_limited", "completed": int, "total_hint":
    int, "detail": str} instead of "done". The repos processed before the
    block are already committed to SQLite (per-repo db.commit() above, not
    batched), so nothing already fetched is lost; this just tells the
    caller how much of the batch actually landed so it can say "3 of 8
    saved, try again later" instead of a bare "sync failed". A 404/401/etc
    mid-batch is not this (those aren't "try again later", they're real
    errors), so only rate-limit-shaped exceptions are caught here;
    everything else still propagates to the caller's existing error
    handling.

    Or, if the caller asks to stop mid-batch (run_id given, and
    app.ingest.github.cancellation.request_cancel(run_id) gets called from
    another request while this generator is running): {"stage":
    "cancelled", "completed": int, "total_hint": int} instead of "done".
    Checked once per repo, before starting the next one, same granularity
    as the rate-limit path, so the repo currently in flight still finishes
    and stays committed, but no further repos start.

    Caller's responsibility: pass a run_id that's unique to THIS attempt,
    not reused across separate syncs (e.g. a fresh UUID generated
    client-side per click of "Sync", not a fixed id like a repo's own id).
    The cancellation registry is a bare id->flag set with no notion of
    "which attempt": if a cancel flag is left set after this generator
    already finished on its own (a request_cancel arriving just after the
    last check, too late to matter) and the SAME run_id gets reused for a
    brand new sync later, that new sync will see the leftover flag and
    cancel itself immediately. A fresh id per attempt makes this
    structurally impossible; reusing one is a real footgun.
    """
    client = client or GitHubClient()
    db = get_db()
    fetched = 0
    cache_hits = 0
    total = 0
    yield {"stage": "checking_profile", "username": username}
    try:
        total_hint = client.repo_count_hint(username)
        yield {"stage": "listing_repos", "total_hint": total_hint}
        try:
            for gh_repo in client.user_repos(username, include_forks=include_forks):
                if run_id and is_cancelled(run_id):
                    clear_cancel(run_id)
                    yield {
                        "stage": "cancelled",
                        "completed": total,
                        "total_hint": total_hint,
                    }
                    return
                total += 1
                hit = _save(db, gh_repo, client, username, account_id)
                if hit:
                    cache_hits += 1
                else:
                    fetched += 1
                yield {
                    "stage": "repo_progress",
                    "index": total,
                    "total_hint": total_hint,
                    "name": gh_repo.full_name,
                    "cache_hit": hit,
                }
        except (RateLimitExceededException, GithubException) as e:
            if isinstance(e, RateLimitExceededException) or (
                isinstance(e, GithubException) and e.status in (403, 429)
            ):
                yield _rate_limited_event(e, total, total_hint)
                return
            raise
    finally:
        db.close()
    yield {"stage": "done", "total_repos": total, "fetched": fetched, "cache_hits": cache_hits}


def sync_single_repo(
    repo_full_name: str,
    attribution_username: str,
    client: GitHubClient | None = None,
    account_id: int | None = None,
) -> SyncSummary:
    """Fetches and upserts exactly one repo ("owner/name"), for a project
    someone contributed to but doesn't own, where syncing the whole owning
    account would pull in repos that aren't theirs. Commit attribution
    (commits_authored/last_commit_at) is credited to attribution_username,
    not the repo's owner; pass the account's own GitHub username here.
    """
    client = client or GitHubClient()
    db = get_db()
    try:
        gh_repo = client.get_repo(repo_full_name)
        hit = _save(db, gh_repo, client, attribution_username, account_id)
    finally:
        db.close()
    return SyncSummary(total_repos=1, fetched=0 if hit else 1, cache_hits=1 if hit else 0)


def sync_single_repo_progress(
    repo_full_name: str,
    attribution_username: str,
    client: GitHubClient | None = None,
    account_id: int | None = None,
    run_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Generator twin of sync_single_repo(): same event vocabulary as
    sync_account_progress() (checking_profile / listing_repos /
    repo_progress / done) so the sync-sources UI can drive one progress
    component for both a whole-account sync and a single-repo sync,
    total_hint is always 1 here. Only one unit of work here (one repo), so
    "stop" only has one moment to take effect: before that fetch starts,
    not after, since there's nothing partial to preserve once it's
    underway. Same run_id-uniqueness caveat as
    sync_account_progress(); see its docstring.
    """
    client = client or GitHubClient()
    yield {"stage": "checking_profile", "username": attribution_username}
    if run_id and is_cancelled(run_id):
        clear_cancel(run_id)
        yield {"stage": "cancelled", "completed": 0, "total_hint": 1}
        return
    db = get_db()
    try:
        gh_repo = client.get_repo(repo_full_name)
        yield {"stage": "listing_repos", "total_hint": 1}
        hit = _save(db, gh_repo, client, attribution_username, account_id)
        yield {
            "stage": "repo_progress",
            "index": 1,
            "total_hint": 1,
            "name": gh_repo.full_name,
            "cache_hit": hit,
        }
    finally:
        db.close()
    yield {
        "stage": "done",
        "total_repos": 1,
        "fetched": 0 if hit else 1,
        "cache_hits": 1 if hit else 0,
    }


def sync_account(
    username: str,
    include_forks: bool = True,
    client: GitHubClient | None = None,
    account_id: int | None = None,
) -> SyncSummary:
    """account_id is optional: a sync isn't required to belong to an
    account (e.g. looking up someone else's public profile out of
    curiosity). When given, every repo this run touches gets stamped with
    it, so the projects page can filter to "my repos" only.
    """
    client = client or GitHubClient()
    db = get_db()
    fetched = 0
    cache_hits = 0
    total = 0
    try:
        for gh_repo in client.user_repos(username, include_forks=include_forks):
            total += 1
            hit = _save(db, gh_repo, client, username, account_id)
            if hit:
                cache_hits += 1
            else:
                fetched += 1
            logger.info("%s %s", "cache-hit" if hit else "fetched", gh_repo.full_name)
    finally:
        db.close()
    return SyncSummary(total_repos=total, fetched=fetched, cache_hits=cache_hits)
