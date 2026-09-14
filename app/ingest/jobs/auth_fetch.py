"""Automated-login job-posting fetch for sites that require being signed
in to see a posting (Wellfound, LinkedIn, and similar). Real headless
Chromium via Playwright, driven by the CSS selectors and credentials
stored in an AuthSource row (app/core/db.py, app/core/auth_sources_store.py).

Intended for a person's own job search with their own credentials. It is
not a scraping-at-scale tool: one fetch is one
login, one page, one close. Read this before wiring a caller to it:

- Automating a login is against most sites' terms of service. Real risk
  of the signed-in account being rate-limited, flagged, or permanently
  banned, especially with repeated use. Use an alternate account, never a
  primary one. This is surfaced as a UI warning
  (app/web/templates/auth_sources.html) and an explicit acknowledgment
  flag at setup (AuthSource.acknowledged_risk), not by this module.
- This is fragile by construction: any change to a site's login form
  (new field, new step, a CAPTCHA, 2FA) breaks the stored selectors with
  no advance warning, and this module cannot detect or work around a
  CAPTCHA/2FA challenge; it will fail with AuthLoginFailedError.
- No session is ever kept between calls. Every fetch launches a fresh
  browser context, logs in from scratch, fetches one page, and closes
  everything (cookies included) before returning, success or failure.
  Nothing about a login is cached or reused.
- Credentials are never logged, never included in any exception message,
  and only ever exist in memory for the duration of one fetch call (see
  app/core/auth_sources_store.py's resolve_credentials, the only place
  they're decrypted).

`_playwright_fn` is a test-injection point (matching
app/resume_build/compile.py's `_run_fn` convention): production callers
never pass it, it defaults to the real playwright.sync_api.sync_playwright.
The unit tests cover the selector-fill/submit/navigate sequence against a
fake Page/Browser; they do not run against a real login-walled site.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.auth_sources_store import record_check_outcome, resolve_credentials
from app.core.db import AuthSource

logger = logging.getLogger(__name__)

_NAV_TIMEOUT_MS = 30_000
_MAX_CHARS = 20_000


class AuthSourceNotFoundError(Exception):
    """No enabled AuthSource with the given id."""


class AuthLoginFailedError(Exception):
    """The login attempt itself didn't succeed: a bad selector, bad
    credentials, a CAPTCHA/2FA step this module can't drive, or (when
    configured) a post_login_wait_selector that never appeared."""


class AuthFetchError(Exception):
    """Login succeeded but the target page fetch failed, or came back
    with nothing usable once stripped of markup."""


def _default_playwright_fn() -> Any:
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _perform_login(page: Any, source: AuthSource, credentials: dict) -> None:
    """Fill + submit the login form, then wait for a signal that it
    worked. Raises whatever Playwright raises (a missing selector, a
    navigation timeout); the caller wraps this into AuthLoginFailedError
    with context, this function stays a thin, directly-testable step.
    """
    page.goto(source.login_url)
    page.fill(source.username_selector, credentials["username"])
    page.fill(source.password_selector, credentials["password"])
    page.click(source.submit_selector)
    if source.post_login_wait_selector:
        page.wait_for_selector(source.post_login_wait_selector)
    else:
        page.wait_for_load_state("networkidle")


def test_login(source_id: int, _playwright_fn: Any = None) -> None:
    """Logs in and nothing else; no target page fetched. Lets someone
    verify their stored selectors actually work (app/web/templates/
    auth_sources.html's "Test login" button) without spending a real
    fetch on a specific job URL. Raises AuthSourceNotFoundError /
    AuthLoginFailedError; records the outcome either way via
    record_check_outcome, same as a real fetch does.
    """
    resolved = resolve_credentials(source_id)
    if resolved is None:
        raise AuthSourceNotFoundError(f"no enabled authenticated source with id={source_id}")
    source, credentials = resolved

    playwright_fn = _playwright_fn or _default_playwright_fn()
    with playwright_fn() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_context().new_page()
            page.set_default_timeout(_NAV_TIMEOUT_MS)
            try:
                _perform_login(page, source, credentials)
            except Exception as e:
                record_check_outcome(source.id, ok=False, detail=f"login failed: {e}")
                raise AuthLoginFailedError(f"could not log in via '{source.label}': {e}") from e
        finally:
            browser.close()

    record_check_outcome(source.id, ok=True, detail="test login succeeded")


def fetch_job_url_authenticated(
    source_id: int, target_url: str, _playwright_fn: Any = None
) -> tuple[str, str]:
    """Logs into `source_id`'s site and returns (title_guess, text) from
    target_url, same return shape as app/ingest/jobs/url_fetch.py's
    fetch_job_url, so both feed the same caller
    (app/api/job_postings.py). See this module's docstring for the full
    risk framing before wiring a new caller to this.
    """
    resolved = resolve_credentials(source_id)
    if resolved is None:
        raise AuthSourceNotFoundError(f"no enabled authenticated source with id={source_id}")
    source, credentials = resolved

    playwright_fn = _playwright_fn or _default_playwright_fn()

    with playwright_fn() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_context().new_page()
            page.set_default_timeout(_NAV_TIMEOUT_MS)
            try:
                _perform_login(page, source, credentials)
            except Exception as e:
                record_check_outcome(source.id, ok=False, detail=f"login failed: {e}")
                raise AuthLoginFailedError(f"could not log in via '{source.label}': {e}") from e

            record_check_outcome(source.id, ok=True, detail="last login succeeded")

            try:
                page.goto(target_url)
                page.wait_for_load_state("networkidle")
                html = page.content()
            except Exception as e:
                raise AuthFetchError(f"logged in, but could not load {target_url}: {e}") from e
        finally:
            browser.close()

    from app.ingest.jobs.url_fetch import _strip_html

    title, text = _strip_html(html)
    if len(text) < 200:
        raise AuthFetchError(
            "logged in, but that page had no readable text once stripped of markup"
        )
    return title, text[:_MAX_CHARS]
