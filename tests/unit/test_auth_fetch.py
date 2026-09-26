"""app/ingest/jobs/auth_fetch.py: the selector-fill/submit/navigate
sequence, against a fake Playwright Page/Browser injected via
`_playwright_fn` (same test-injection convention as
app/resume_build/compile.py's `_run_fn`). Not run against a
real login-walled site.
"""

from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.auth_sources_store import add_source
from app.core.db import Account, get_db, init_db
from app.core.settings import get_settings
from app.ingest.jobs.auth_fetch import (
    AuthFetchError,
    AuthLoginFailedError,
    AuthSourceNotFoundError,
    fetch_job_url_authenticated,
)
from app.ingest.jobs.auth_fetch import test_login as auth_test_login


def _reset(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _make_account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _add_source(account_id: int, **overrides) -> int:
    defaults = dict(
        account_id=account_id,
        label="Test site",
        site_domain="example.com",
        login_url="https://example.com/login",
        username_selector="#user",
        password_selector="#pass",
        submit_selector="#submit",
        post_login_wait_selector=None,
        username="me@example.com",
        password="hunter2",
        acknowledged_risk=True,
    )
    defaults.update(overrides)
    return add_source(**defaults)["id"]


class _FakePage:
    def __init__(self, fail_login: bool = False, fail_fetch: bool = False):
        self.calls: list[tuple] = []
        self.fail_login = fail_login
        self.fail_fetch = fail_fetch

    def set_default_timeout(self, ms):
        self.calls.append(("set_default_timeout", ms))

    def goto(self, url):
        self.calls.append(("goto", url))
        if self.fail_fetch and url == "https://example.com/jobs/1":
            raise RuntimeError("could not load target page")

    def fill(self, selector, value):
        self.calls.append(("fill", selector, value))
        if self.fail_login and selector == "#user":
            raise RuntimeError("selector not found")

    def click(self, selector):
        self.calls.append(("click", selector))

    def wait_for_selector(self, selector):
        self.calls.append(("wait_for_selector", selector))

    def wait_for_load_state(self, state):
        self.calls.append(("wait_for_load_state", state))

    def content(self):
        return (
            "<html><body><main><h1>Job</h1>"
            "<p>Real posting text that is definitely long enough to pass the "
            "minimum-content check this module enforces before accepting a fetch "
            "as successful rather than a login wall or empty page. Padding this "
            "out further with a second sentence of detail about the role, the "
            "team, and the responsibilities involved, so the stripped visible "
            "text comfortably clears the two-hundred-character floor.</p>"
            "</main></body></html>"
        )


class _FakeContext:
    def __init__(self, page):
        self._page = page

    def new_page(self):
        return self._page


class _FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.closed = False

    def new_context(self):
        return _FakeContext(self._page)

    def close(self):
        self.closed = True


class _FakeChromium:
    def __init__(self, page):
        self._page = page
        self.browser = None

    def launch(self, headless=True):
        self.browser = _FakeBrowser(self._page)
        return self.browser


class _FakePlaywright:
    def __init__(self, page):
        self.chromium = _FakeChromium(page)


def _playwright_fn_for(page):
    """Returns a zero-arg callable usable as a context manager, matching
    playwright.sync_api.sync_playwright()'s own shape (`with
    sync_playwright() as p:`). Reuses one _FakePlaywright/_FakeChromium
    instance across calls so a test can inspect chromium.browser.closed
    afterward to prove the browser was actually closed, success or
    failure."""
    playwright = _FakePlaywright(page)

    class _CM:
        def __enter__(self):
            return playwright

        def __exit__(self, *exc):
            return False

    fn = lambda: _CM()  # noqa: E731
    fn.playwright = playwright
    return fn


def test_login_success_fills_and_submits_then_records_status(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    page = _FakePage()

    auth_test_login(source_id, _playwright_fn=_playwright_fn_for(page))

    assert ("fill", "#user", "me@example.com") in page.calls
    assert ("fill", "#pass", "hunter2") in page.calls
    assert ("click", "#submit") in page.calls

    from app.core.auth_sources_store import list_sources

    row = list_sources(account_id)[0]
    assert row["status"] == "valid"


def test_login_failure_raises_and_records_invalid(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    page = _FakePage(fail_login=True)

    with pytest.raises(AuthLoginFailedError):
        auth_test_login(source_id, _playwright_fn=_playwright_fn_for(page))

    from app.core.auth_sources_store import list_sources

    row = list_sources(account_id)[0]
    assert row["status"] == "invalid"


def test_unknown_source_raises_not_found(tmp_path):
    _reset(tmp_path)
    with pytest.raises(AuthSourceNotFoundError):
        auth_test_login(999999, _playwright_fn=_playwright_fn_for(_FakePage()))


def test_disabled_source_raises_not_found(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    from app.core.auth_sources_store import set_enabled

    set_enabled(source_id, False)

    with pytest.raises(AuthSourceNotFoundError):
        auth_test_login(source_id, _playwright_fn=_playwright_fn_for(_FakePage()))


def test_fetch_after_successful_login_returns_stripped_text(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    page = _FakePage()

    title, text = fetch_job_url_authenticated(
        source_id, "https://example.com/jobs/1", _playwright_fn=_playwright_fn_for(page)
    )

    assert "Real posting text" in text
    assert ("goto", "https://example.com/jobs/1") in page.calls


def test_fetch_wraps_target_page_failure(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    page = _FakePage(fail_fetch=True)

    with pytest.raises(AuthFetchError):
        fetch_job_url_authenticated(
            source_id, "https://example.com/jobs/1", _playwright_fn=_playwright_fn_for(page)
        )


def test_browser_always_closed_even_on_login_failure(tmp_path):
    _reset(tmp_path)
    account_id = _make_account()
    source_id = _add_source(account_id)
    page = _FakePage(fail_login=True)
    fn = _playwright_fn_for(page)

    with pytest.raises(AuthLoginFailedError):
        auth_test_login(source_id, _playwright_fn=fn)

    assert fn.playwright.chromium.browser.closed is True
