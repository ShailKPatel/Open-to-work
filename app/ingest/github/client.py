"""Thin PyGithub wrapper: auth, rate-limit backoff, pagination stays hidden
inside PyGithub's PaginatedList (callers just iterate).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
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

# raw.githubusercontent.com serves file bodies outside the REST API's
# quota (60 requests an hour per IP without a token), so README and
# manifest text is read from there and only the listing costs quota.
_RAW_TIMEOUT_SECONDS = 15.0

_LAST_PAGE = re.compile(r'[?&]page=(\d+)[^>]*>;\s*rel="last"')


class RateLimitWindowExhausted(RateLimitExceededException):
    """The hourly quota is spent and resets too far out to wait for.
    Retrying before reset_at cannot succeed, so _call() raises this
    straight away instead of backing off five times first."""

    def __init__(self, source: GithubException, reset_at: dt.datetime):
        super().__init__(source.status, source.data, source.headers)
        self.reset_at = reset_at


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
        self._token = token or None
        self._users: dict[str, Any] = {}

    def _wait_for_rate_limit_reset(self) -> dt.datetime | None:
        """Sleeps through a short reset window. Returns the reset time
        instead when the hourly quota is spent and the reset is too far
        away to wait for; a secondary limit (quota left, still blocked)
        returns None so the normal backoff retries it."""
        core = self._gh.get_rate_limit().resources.core
        reset_at = core.reset.timestamp()
        sleep_for = max(0.0, reset_at - time.time()) + 1
        if sleep_for > _MAX_RATE_LIMIT_WAIT_SECONDS:
            logger.warning(
                "rate limit exhausted, real reset is %.0fs away; too long to "
                "block a request on, not sleeping",
                sleep_for,
            )
            if getattr(core, "remaining", 0) == 0:
                return core.reset
            return None
        logger.warning("rate limit exhausted, sleeping %.0fs", sleep_for)
        time.sleep(sleep_for)
        return None

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
            lambda e: not isinstance(e, RateLimitWindowExhausted)
            and (
                isinstance(e, RateLimitExceededException)
                or (isinstance(e, GithubException) and (e.status in (403, 429) or e.status >= 500))
            )
        ),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except GithubException as e:
            if isinstance(e, RateLimitWindowExhausted):
                raise
            if isinstance(e, RateLimitExceededException) or e.status in (403, 429):
                record_event(
                    "github", "rate_limited", str(e), context=getattr(fn, "__name__", None)
                )
                reset_at = self._wait_for_rate_limit_reset()
                if reset_at is not None:
                    raise RateLimitWindowExhausted(e, reset_at) from e
            raise

    def _user(self, username: str) -> Any:
        """One /users/{username} call per sync, shared by the count hint
        and the repo listing."""
        key = username.lower()
        if key not in self._users:
            self._users[key] = self._call(self._gh.get_user, username)
        return self._users[key]

    def user_repos(self, username: str, include_forks: bool = True) -> Iterator[GHRepository]:
        user = self._user(username)
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
        return self._user(username).public_repos

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

    def entry_text(self, repo: GHRepository, entry: Any) -> str | None:
        """Text of a file from a root_contents() listing. Read from its
        raw download URL first, which costs no API quota; only falls back
        to the contents API if that fails."""
        url = getattr(entry, "download_url", None)
        if url:
            headers = {"Authorization": f"token {self._token}"} if self._token else {}
            try:
                resp = httpx.get(
                    url, headers=headers, timeout=_RAW_TIMEOUT_SECONDS, follow_redirects=True
                )
                if resp.status_code == 200:
                    return resp.content.decode("utf-8", errors="replace")
            except httpx.HTTPError:
                pass
        return self.file_text(repo, entry.path)

    def authored_commits(
        self, repo: GHRepository, username: str
    ) -> tuple[int, dt.datetime | None] | None:
        """(commits by username on the default branch, date of the newest
        one) from a single GET /commits?author=...&per_page=1: the page
        holds the newest commit and the Link header's last page number is
        the count. stats/contributors used to answer this, but it replies
        202 while GitHub computes and needed several polls per repo.
        None means unknown (don't overwrite a known value); an empty repo
        is (0, None)."""

        def list_authored_commits() -> tuple[dict, Any]:
            return repo._requester.requestJsonAndCheck(
                "GET",
                f"{repo.url}/commits",
                parameters={"author": username, "per_page": 1},
            )

        try:
            headers, data = self._call(list_authored_commits)
        except GithubException as e:
            if e.status == 409:  # empty repository
                return 0, None
            if e.status == 404:
                return None
            raise
        if not isinstance(data, list) or not data:
            return 0, None
        link = {k.lower(): v for k, v in (headers or {}).items()}.get("link", "")
        match = _LAST_PAGE.search(link)
        count = int(match.group(1)) if match else len(data)
        committed = ((data[0].get("commit") or {}).get("author") or {}).get("date")
        last = None
        if committed:
            last = dt.datetime.fromisoformat(committed.replace("Z", "+00:00"))
        return count, last
