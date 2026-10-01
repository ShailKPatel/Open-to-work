"""Real GitHub API calls through app/ingest/github/client.py and the sync
pipeline. Not part of `make test`; run with `make test-live`.

Skipped unless LIVE_GITHUB=1, so the normal test run never depends on the
network or spends GitHub rate limit. Unauthenticated requests are limited
to 60 per hour; set GITHUB_TOKEN to raise that. Targets octocat and
octocat/Hello-World, GitHub's own long-lived demo account and repository.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from github import UnknownObjectException
from sqlalchemy import select

import app.core.db as db_module
from app.core.db import Repository, get_db, init_db
from app.core.settings import get_settings
from app.ingest.github.client import GitHubClient
from app.ingest.github.sync import sync_single_repo

_USER = "octocat"
_REPO = "octocat/Hello-World"
_MISSING_USER = "otw-no-such-user-7f3c2a91e4"

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVE_GITHUB") != "1",
    reason="LIVE_GITHUB=1 not set; live GitHub tests are opt-in (network, rate limit)",
)


@pytest.fixture(scope="module")
def client() -> GitHubClient:
    get_settings.cache_clear()
    gh = GitHubClient()
    core = gh._gh.get_rate_limit().resources.core
    print(f"\n[live:github] rate limit {core.remaining}/{core.limit}")
    if core.remaining < 20:
        pytest.skip(f"only {core.remaining} GitHub requests left this hour")
    return gh


def test_authenticated_state_matches_settings(client):
    core = client._gh.get_rate_limit().resources.core
    if get_settings().github_token:
        assert core.limit > 60, "GITHUB_TOKEN is set but GitHub treated the request as anonymous"
    else:
        assert core.limit == 60


def test_known_user_exists_and_has_public_repos(client):
    assert client.repo_count_hint(_USER) > 0


def test_unknown_user_fails_fast_with_404(client):
    start = time.monotonic()
    with pytest.raises(UnknownObjectException):
        client.repo_count_hint(_MISSING_USER)
    assert time.monotonic() - start < 10  # a 404 is never retried


def test_user_repos_lists_the_demo_repo(client):
    names = {repo.full_name for repo in client.user_repos(_USER)}
    assert _REPO in names


def test_repo_readme_and_contents(client):
    repo = client.get_repo(_REPO)
    assert repo.full_name == _REPO

    readme = client.readme_text(repo)
    assert readme and readme.strip()

    entries = client.root_contents(repo)
    assert entries
    assert all(entry.type in ("file", "dir", "symlink", "submodule") for entry in entries)

    assert client.file_text(repo, "definitely/not/a/real/path.txt") is None


def test_authored_commits_and_raw_file_text(client):
    repo = client.get_repo(_REPO)

    authored = client.authored_commits(repo, _USER)
    assert authored is not None
    count, last = authored
    assert count >= 0
    assert last is None or last.tzinfo is not None

    readme = next(e for e in client.root_contents(repo) if e.name.lower().startswith("readme"))
    assert client.entry_text(repo, readme)


def test_single_repo_sync_writes_to_the_database_then_hits_cache(client, tmp_path: Path):
    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()

    first = sync_single_repo(_REPO, _USER, client=client)
    assert (first.total_repos, first.fetched, first.cache_hits) == (1, 1, 0)

    db = get_db()
    try:
        row = db.execute(select(Repository).where(Repository.full_name == _REPO)).scalar_one()
        assert row.url == f"https://github.com/{_REPO}"
        assert row.readme
    finally:
        db.close()

    second = sync_single_repo(_REPO, _USER, client=client)
    assert (second.fetched, second.cache_hits) == (0, 1)
