"""GitHub sync: fetches repos, README, manifests, and authorship stats,
and upserts them into SQLite by github_id. Skips README/manifest/stats refetch when pushed_at is
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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from github import GithubException, RateLimitExceededException
from sqlalchemy import ColumnElement, select
from sqlalchemy.orm import Session

from app.core.db import Account, Repository, get_db, is_profile_repo
from app.ingest.github.cancellation import clear as clear_cancel
from app.ingest.github.cancellation import is_cancelled
from app.ingest.github.client import GitHubClient
from app.ingest.github.manifests import MANIFEST_FILENAMES, parse_dependencies


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


def _owned_by(account_id: int | None) -> ColumnElement[bool]:
    """Repos are unique per account, so every lookup by github_id is
    scoped to the syncing account (or to unowned rows when there is none)."""
    if account_id is None:
        return Repository.account_id.is_(None)
    return Repository.account_id == account_id


# _upsert outcomes
CACHE_HIT = "cache_hit"  # not pushed since the last sync, nothing fetched
REFRESHED = "refreshed"  # pushed, README unchanged: stats and manifests only
FETCHED = "fetched"  # new repo or new README: goes back to skill extraction


def _upsert(
    db: Session,
    gh_repo: Any,
    client: GitHubClient,
    username: str,
    account_id: int | None,
    force: bool = False,
) -> str:
    """Returns CACHE_HIT, REFRESHED or FETCHED. force skips the pushed_at
    cache check and refetches README, manifests and stats regardless."""
    existing = db.execute(
        select(Repository).where(_owned_by(account_id), Repository.github_id == gh_repo.id)
    ).scalar_one_or_none()
    if existing is None and account_id is not None:
        # An unowned row (synced from the command line) is claimed, not copied.
        existing = db.execute(
            select(Repository).where(_owned_by(None), Repository.github_id == gh_repo.id)
        ).scalar_one_or_none()
        if existing is not None:
            existing.account_id = account_id

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

    if (
        not force
        and existing is not None
        and _naive_utc(existing.pushed_at) == _naive_utc(pushed_at)
    ):
        existing.stars = gh_repo.stargazers_count
        existing.is_profile_readme = is_profile_repo(gh_repo.full_name)
        existing.fetched_at = dt.datetime.now(dt.UTC)
        # Editing the About text on GitHub is not a push, so pushed_at stays
        # put; the listing carries the description anyway, so take it here.
        description_changed = _source_changed(existing, existing.readme, gh_repo.description)
        existing.description = gh_repo.description
        if description_changed:
            existing.skill_extraction_status = "pending"
            existing.skill_extraction_error = None
            return FETCHED
        return CACHE_HIT

    readme, readme_sha, manifests = _fetch_readme_and_manifests(client, gh_repo, existing)
    authored = client.authored_commits(gh_repo, username)

    if existing is None:
        changed = True
        existing = Repository(github_id=gh_repo.id, account_id=account_id)
        db.add(existing)
    else:
        changed = _source_changed(existing, readme, gh_repo.description)

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
            select(Repository.id).where(_owned_by(account_id), Repository.github_id == gh_repo.id)
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
    """Syncs every repo of `username`, yielding a progress event per stage:

      {"stage": "checking_profile", "username": ...}
      {"stage": "listing_repos", "total_hint": int}
      {"stage": "repo_progress", "index", "total_hint", "name", "cache_hit"}  (repeated)
      {"stage": "done", "total_repos", "fetched", "cache_hits"}

    A rate limit partway through ends with {"stage": "rate_limited", ...}
    instead of "done"; every repo before it is already committed, so the
    caller can report "3 of 8 saved". Other GitHub errors propagate.

    With a run_id, a cancel requested through
    app/ingest/github/cancellation.py stops the run before the next repo
    and ends with {"stage": "cancelled", ...}. The run_id must be unique
    per attempt: a leftover flag would cancel a later run that reused it.
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


def sync_single_repo_progress(
    repo_full_name: str,
    attribution_username: str,
    client: GitHubClient | None = None,
    account_id: int | None = None,
    run_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Syncs one repo ("owner/name") with the same events as
    sync_account_progress, for a project the person contributed to but does
    not own. Commits are credited to attribution_username, not the owner.
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


def refetch_repo(repo_id: int, client: GitHubClient | None = None) -> None:
    """Pulls one stored repo fresh from GitHub (description, README,
    manifests, stats), skipping the pushed_at cache. The Reprocess button
    calls this first so extraction reads what GitHub has now. Raises
    ValueError for an unknown or hand-added repo; GitHub errors propagate.
    """
    db = get_db()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            raise ValueError(f"no repository with id={repo_id}")
        if repo.github_id < 0:
            raise ValueError(f"{repo.full_name} was added by hand, not synced from GitHub")
        account = db.get(Account, repo.account_id) if repo.account_id is not None else None
        username = account.github_username if account else repo.full_name.split("/")[0]
        account_id = repo.account_id
        full_name = repo.full_name
        db.commit()  # end the read before the GitHub calls

        client = client or GitHubClient()
        gh_repo = client.get_repo(full_name)
        _upsert(db, gh_repo, client, username, account_id, force=True)
        db.commit()
    finally:
        db.close()


class SyncRateLimitedError(Exception):
    """GitHub stopped a blocking sync partway; the repos before it are saved."""


def _run_to_end(events: Iterator[dict[str, Any]]) -> SyncSummary:
    for event in events:
        if event["stage"] == "rate_limited":
            raise SyncRateLimitedError(event["detail"])
        if event["stage"] == "done":
            return SyncSummary(
                total_repos=event["total_repos"],
                fetched=event["fetched"],
                cache_hits=event["cache_hits"],
            )
    raise RuntimeError("sync ended without a result")


def sync_account(
    username: str,
    include_forks: bool = True,
    client: GitHubClient | None = None,
    account_id: int | None = None,
) -> SyncSummary:
    """Blocking form of sync_account_progress, for the command line. When
    account_id is given, every repo the run touches is stamped with it."""
    return _run_to_end(
        sync_account_progress(username, include_forks, client=client, account_id=account_id)
    )


def sync_single_repo(
    repo_full_name: str,
    attribution_username: str,
    client: GitHubClient | None = None,
    account_id: int | None = None,
) -> SyncSummary:
    """Blocking form of sync_single_repo_progress."""
    return _run_to_end(
        sync_single_repo_progress(
            repo_full_name, attribution_username, client=client, account_id=account_id
        )
    )
