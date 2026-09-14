"""Thin PyGithub wrapper: auth, rate-limit backoff, pagination stays hidden
inside PyGithub's PaginatedList (callers just iterate).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

from github import Auth, Github, GithubException, RateLimitExceededException
from github.Repository import Repository as GHRepository
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.rate_limits import record_event
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

# The real GitHub rate-limit reset window can be up to an hour away.
# Sleeping the actual remaining time inside a synchronous HTTP request
# handler (account creation, single-repo sync, the "checking profile"
# stage of a sync stream; every one of these calls _call()) means that
# request just hangs from the caller's point of view for up to an hour.
# Creating an account while rate-limited would otherwise hang instead of
# failing fast. Only sleep through a short remaining window; past this
# cap, skip the sleep and let it re-raise. tenacity's
# own (much shorter, exponential-backoff-capped) retry already handles the
# short-but-real "wait a few seconds" case, and the caller's existing
# RateLimitExceededException handling (already built, everywhere this is
# called from) surfaces the error quickly instead of hanging.
_MAX_RATE_LIMIT_WAIT_SECONDS = 10.0


class GitHubClient:
    def __init__(self, token: str | None = None):
        settings = get_settings()
        token = token or settings.github_token
        auth = Auth.Token(token) if token else None
        # retry=None: PyGithub defaults to its own GithubRetry (total=10),
        # a urllib3 Retry that reads GitHub's rate-limit headers and sleeps
        # the real reset window *inside the HTTP call itself*, below our
        # own _call()/tenacity retry layer, invisible to it, and not
        # subject to the cap in _wait_for_rate_limit_reset above. That
        # produced hangs of ~740s. _call() already retries (short backoff,
        # only on the right status codes), so a second, uncapped retry
        # layer underneath it is redundant.
        self._gh = Github(auth=auth, per_page=100, retry=None)

    def _wait_for_rate_limit_reset(self) -> None:
        core = self._gh.get_rate_limit().resources.core
        reset_at = core.reset.timestamp()
        sleep_for = max(0.0, reset_at - time.time()) + 1
        if sleep_for > _MAX_RATE_LIMIT_WAIT_SECONDS:
            logger.warning(
                "rate limit exhausted, real reset is %.0fs away; too long to "
                "block a request on, not sleeping (retry/backoff still applies)",
                sleep_for,
            )
            return
        logger.warning("rate limit exhausted, sleeping %.0fs", sleep_for)
        time.sleep(sleep_for)

    @retry(
        # Only retry rate-limit-shaped errors (403/429) and real server
        # trouble (5xx), not 404 (unknown user/repo), 401 (bad token),
        # 422, etc. retry_if_exception_type(GithubException) would match
        # every PyGithub exception, since UnknownObjectException and
        # BadCredentialsException both subclass it, so a plain "user
        # doesn't exist" 404 was getting retried 5x with exponential
        # backoff (~30s wasted) before finally surfacing, which read as a
        # hang on the sync-progress UI instead of an instant error.
        retry=retry_if_exception(
            lambda e: isinstance(e, RateLimitExceededException)
            or (isinstance(e, GithubException) and (e.status in (403, 429) or e.status >= 500))
        ),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededException as e:
            record_event("github", "rate_limited", str(e), context=getattr(fn, "__name__", None))
            self._wait_for_rate_limit_reset()
            raise
        except GithubException as e:
            if e.status in (403, 429):
                record_event(
                    "github", "rate_limited", str(e), context=getattr(fn, "__name__", None)
                )
                self._wait_for_rate_limit_reset()
            raise

    def user_repos(self, username: str, include_forks: bool = True) -> Iterator[GHRepository]:
        user = self._call(self._gh.get_user, username)
        for repo in self._call(user.get_repos):
            if not include_forks and repo.fork:
                continue
            yield repo

    def get_repo(self, full_name: str) -> GHRepository:
        """full_name is "owner/repo", for a sync source that's a single
        project rather than a whole account."""
        return self._call(self._gh.get_repo, full_name)

    def repo_count_hint(self, username: str) -> int:
        """Cheap approximate total for progress UI (one extra /users/{username}
        call). Counts forks too even when include_forks=False, and can drift
        if repos change mid-sync: a hint, not a guarantee."""
        user = self._call(self._gh.get_user, username)
        return user.public_repos

    def readme_text(self, repo: GHRepository) -> str | None:
        try:
            content_file = self._call(repo.get_readme)
        except GithubException as e:
            if e.status == 404:
                return None
            raise
        try:
            return content_file.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            return None

    def root_contents(self, repo: GHRepository) -> list[Any]:
        try:
            contents = self._call(repo.get_contents, "")
        except GithubException as e:
            if e.status == 404:
                return []
            raise
        return contents if isinstance(contents, list) else [contents]

    def file_text(self, repo: GHRepository, path: str) -> str | None:
        try:
            content_file = self._call(repo.get_contents, path)
        except GithubException as e:
            if e.status == 404:
                return None
            raise
        if isinstance(content_file, list):
            return None
        try:
            return content_file.decoded_content.decode("utf-8", errors="replace")
        except Exception:
            return None

    def contributor_stats(self, repo: GHRepository) -> list[Any]:
        """GET /repos/{owner}/{repo}/stats/contributors. Returns [] while
        GitHub is still computing (202) after retries; caller should treat
        that as 'unknown, try again next sync' rather than 'zero commits'.
        """
        for _ in range(3):
            stats = self._call(repo.get_stats_contributors)
            if stats is not None:
                return stats
            time.sleep(2)
        return []
