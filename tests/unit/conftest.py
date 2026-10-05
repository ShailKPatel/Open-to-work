from types import SimpleNamespace

import pytest

import app.core.db as db_module


@pytest.fixture(autouse=True)
def _no_real_skill_review(monkeypatch):
    """Extraction and resume merges now end in an LLM review of new skill
    names (app/profile/skill_review.py). Default every test to a review that
    removes nothing, so no test makes a real LLM call; tests of the review
    itself patch this again with their own fake."""
    monkeypatch.setattr(
        "app.profile.skill_review.complete",
        lambda *args, **kwargs: SimpleNamespace(parsed={"remove": []}),
    )


@pytest.fixture(autouse=True)
def _dispose_db_engine():
    """Tests swap in a fresh SQLite file per test via reset_engine().
    Reset again on the way out so the engine a test leaves behind has its
    pooled connections closed instead of leaking until garbage
    collection."""
    yield
    db_module.reset_engine()


@pytest.fixture(autouse=True)
def _default_github_username_exists(monkeypatch):
    """POST /accounts validates github_username against a real GitHub call
    via app.api.accounts.GitHubClient (see _github_user_exists). Patch the
    client class itself, not _github_user_exists, so its real True/False/
    None branching still runs in every test; only the network call
    underneath is faked. Default: every username "exists", so throwaway
    usernames used across the suite (octocat, ghopper, evex, ...) don't
    need real network access. Tests that care about the not-found or
    verification-unavailable paths override this (or patch
    _github_user_exists directly when they don't care about its internals).
    """

    class _FakeGitHubClient:
        def __init__(self, *args, **kwargs):
            pass

        def repo_count_hint(self, username: str) -> int:
            return 1

    monkeypatch.setattr("app.api.accounts.GitHubClient", _FakeGitHubClient)


@pytest.fixture(autouse=True)
def _no_background_startup_work(monkeypatch):
    """The app lifespan starts three background threads: one builds missing
    skill-map layouts (app/api/skills.py's warm_skill_maps), which loads a
    real embedding model, one rechecks keys whose quota cooldown has
    elapsed (app/core/key_refresh.py), which calls providers, and one
    re-syncs stale GitHub sources (app/ingest/github/auto_sync.py). Every
    test that builds a TestClient would pay for all three and hit the
    network, so they are off unless a test asks for them."""
    from app.core.settings import get_settings

    monkeypatch.setenv("SKILL_MAP_WARM_START", "0")
    monkeypatch.setenv("KEY_REFRESH_ON_START", "0")
    monkeypatch.setenv("GITHUB_AUTO_SYNC_ON_START", "0")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _job_posting_reads_inline(monkeypatch):
    """Saving a job posting starts its LLM read in a worker thread
    (app/api/job_postings.py's _start_extraction). Run it inline instead,
    so a test sees the finished read in the response and no thread
    outlives its temp database. Tests of the background path itself
    restore the real starter."""
    import app.api.job_postings as job_postings

    def run_inline(posting_id: int, bypass_cache: bool = False) -> bool:
        job_postings._read_posting(posting_id, bypass_cache=bypass_cache)
        return True

    monkeypatch.setattr(job_postings, "_start_extraction", run_inline)


@pytest.fixture(autouse=True)
def _resume_reads_inline(monkeypatch):
    """Same as _job_posting_reads_inline, for an uploaded resume's read
    (app/profile/resume_ingest.py's start_extraction). Tests of the
    background path itself restore the real starter."""
    import app.profile.resume_ingest as resume_ingest

    def run_inline(key, work) -> bool:
        work()
        return True

    monkeypatch.setattr(resume_ingest, "_start_worker", run_inline)


@pytest.fixture(autouse=True)
def _test_client_host_allowed(monkeypatch):
    """TestClient sends Host: testserver, which the app's host check
    (app/api/security.py) refuses like any other non-loopback name. Let it
    through so every API test can keep its plain TestClient(app)."""
    from app.api import security

    monkeypatch.setattr(security, "ALLOWED_HOSTS", security.ALLOWED_HOSTS | {"testserver"})
