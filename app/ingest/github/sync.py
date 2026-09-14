"""sync_account(username) -> list[Repository]

Fetches repos, README, manifests, and authorship stats; upserts into SQLite
by github_id. Skips README/manifest/stats refetch when pushed_at is
unchanged since the last sync (cache hit); the cheap repo-list call still runs
every time to detect what changed.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from github import GithubException, RateLimitExceededException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import Repository, get_db
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


def _last_authored_commit(stats: list, username: str) -> tuple[int, dt.datetime | None]:
    """(commits_authored, last_commit_at) for username from contributor stats.
    Returns (0, None) if stats absent or username has no entry.
    """
    username_lower = username.lower()
    for entry in stats:
        author = getattr(entry, "author", None)
        if author is None or (author.login or "").lower() != username_lower:
            continue
        total = getattr(entry, "total", 0) or 0
        last: dt.datetime | None = None
        for week in getattr(entry, "weeks", []) or []:
            if getattr(week, "c", 0):
                w = week.w
                if isinstance(w, dt.datetime):
                    candidate = w if w.tzinfo else w.replace(tzinfo=dt.UTC)
                else:
                    candidate = dt.datetime.fromtimestamp(w, tz=dt.UTC)
                if last is None or candidate > last:
                    last = candidate
        return total, last
    return 0, None


def _fetch_manifests(client: GitHubClient, gh_repo) -> dict:
    manifests: dict[str, dict] = {}
    for entry in client.root_contents(gh_repo):
        if entry.type != "file" or entry.name not in MANIFEST_FILENAMES:
            continue
        content = client.file_text(gh_repo, entry.path)
        if content is None:
            continue
        manifests[entry.name] = {
            "ecosystem": MANIFEST_FILENAMES[entry.name],
            "dependencies": parse_dependencies(entry.name, content),
        }
    return manifests


def _upsert(
    db: Session, gh_repo, client: GitHubClient, username: str, account_id: int | None
) -> bool:
    """Returns True if this repo was a cache hit (no refetch of readme/manifests/stats)."""
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
        existing.fetched_at = dt.datetime.now(dt.UTC)
        return True

    readme = client.readme_text(gh_repo)
    manifests = _fetch_manifests(client, gh_repo)
    stats = client.contributor_stats(gh_repo)
    commits_authored, last_commit_at = _last_authored_commit(stats, username)

    if existing is None:
        existing = Repository(github_id=gh_repo.id, account_id=account_id)
        db.add(existing)
    elif account_id is not None:
        existing.account_id = account_id

    existing.name = gh_repo.name
    existing.full_name = gh_repo.full_name
    existing.url = gh_repo.html_url
    existing.is_fork = gh_repo.fork
    existing.primary_language = gh_repo.language
    existing.stars = gh_repo.stargazers_count
    existing.readme = readme
    existing.description = gh_repo.description
    existing.manifests_json = manifests
    # stats/contributors can return [] while GitHub is still computing;
    # don't clobber a previously known value with zero in that case.
    if stats:
        existing.commits_authored = commits_authored
        existing.last_commit_at = last_commit_at
    existing.pushed_at = pushed_at
    existing.fetched_at = dt.datetime.now(dt.UTC)
    # content changed (or this is a new repo): any prior extraction is
    # stale, whether it previously succeeded, failed, or had no signal.
    existing.skill_extraction_status = "pending"
    existing.skill_extraction_error = None
    return False


def sync_account_progress(
    username: str,
    include_forks: bool = True,
    client: GitHubClient | None = None,
    account_id: int | None = None,
    run_id: str | None = None,
):
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
                hit = _upsert(db, gh_repo, client, username, account_id)
                db.commit()
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
                yield {
                    "stage": "rate_limited",
                    "completed": total,
                    "total_hint": total_hint,
                    "detail": (
                        f"GitHub may be rate-limiting us: {total} of "
                        f"{total_hint} repos saved. Try syncing again "
                        "later to pick up the rest."
                    ),
                }
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
        hit = _upsert(db, gh_repo, client, attribution_username, account_id)
        db.commit()
    finally:
        db.close()
    return SyncSummary(total_repos=1, fetched=0 if hit else 1, cache_hits=1 if hit else 0)


def sync_single_repo_progress(
    repo_full_name: str,
    attribution_username: str,
    client: GitHubClient | None = None,
    account_id: int | None = None,
    run_id: str | None = None,
):
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
        hit = _upsert(db, gh_repo, client, attribution_username, account_id)
        db.commit()
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
            hit = _upsert(db, gh_repo, client, username, account_id)
            db.commit()
            if hit:
                cache_hits += 1
            else:
                fetched += 1
            logger.info("%s %s", "cache-hit" if hit else "fetched", gh_repo.full_name)
    finally:
        db.close()
    return SyncSummary(total_repos=total, fetched=fetched, cache_hits=cache_hits)
