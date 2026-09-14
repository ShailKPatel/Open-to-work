import pytest


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
