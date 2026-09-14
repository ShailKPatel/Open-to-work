"""Regression coverage for two retry/rate-limit bugs:

1. retry_if_exception_type matched every PyGithub exception
   (UnknownObjectException and BadCredentialsException both subclass
   GithubException), so a plain 404 "user doesn't exist" or a bad token
   got retried 5x with exponential backoff before finally surfacing:
   ~30s of apparent hang on the sync-progress UI instead of an instant
   error.
2. _wait_for_rate_limit_reset() slept the *real* remaining reset window
   (up to an hour) inside a synchronous HTTP request handler, so creating
   an account while rate-limited just hung indefinitely instead of
   failing fast.

See app/ingest/github/client.py::GitHubClient._call and
_wait_for_rate_limit_reset for the fixes.
"""

import datetime as dt
from types import SimpleNamespace

from github import (
    BadCredentialsException,
    GithubException,
    RateLimitExceededException,
    UnknownObjectException,
)

from app.ingest.github.client import GitHubClient


def _client() -> GitHubClient:
    return GitHubClient(token="fake-token")


def test_call_does_not_retry_404():
    client = _client()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise UnknownObjectException(404, "Not Found", {})

    try:
        client._call(fn)
    except UnknownObjectException:
        pass
    else:
        raise AssertionError("expected UnknownObjectException")

    assert calls["n"] == 1  # no retries: a 404 is not transient


def test_call_does_not_retry_401():
    client = _client()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise BadCredentialsException(401, "Bad credentials", {})

    try:
        client._call(fn)
    except BadCredentialsException:
        pass
    else:
        raise AssertionError("expected BadCredentialsException")

    assert calls["n"] == 1  # no retries: a bad token won't fix itself


def test_call_retries_403_then_succeeds(monkeypatch):
    client = _client()
    monkeypatch.setattr(client, "_wait_for_rate_limit_reset", lambda: None)
    monkeypatch.setattr("time.sleep", lambda *_: None)  # skip tenacity's real backoff wait
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GithubException(403, "rate limited", {})
        return "ok"

    result = client._call(fn)

    assert result == "ok"
    assert calls["n"] == 3  # retried twice, then succeeded


def test_call_retries_5xx(monkeypatch):
    client = _client()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise GithubException(502, "bad gateway", {})
        return "ok"

    result = client._call(fn)

    assert result == "ok"
    assert calls["n"] == 2


def test_wait_for_rate_limit_reset_skips_sleep_when_reset_is_far_away(monkeypatch):
    client = _client()
    far_future = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=45)
    client._gh.get_rate_limit = lambda: SimpleNamespace(
        resources=SimpleNamespace(core=SimpleNamespace(reset=far_future))
    )
    slept = {"called": False}
    monkeypatch.setattr("time.sleep", lambda *_: slept.__setitem__("called", True))

    client._wait_for_rate_limit_reset()

    assert slept["called"] is False  # would've hung ~45 minutes before the fix


def test_wait_for_rate_limit_reset_sleeps_when_reset_is_soon(monkeypatch):
    client = _client()
    soon = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=3)
    client._gh.get_rate_limit = lambda: SimpleNamespace(
        resources=SimpleNamespace(core=SimpleNamespace(reset=soon))
    )
    slept = {"seconds": None}
    monkeypatch.setattr("time.sleep", lambda s: slept.__setitem__("seconds", s))

    client._wait_for_rate_limit_reset()

    assert slept["seconds"] is not None
    assert slept["seconds"] <= 10  # short waits still actually wait


def test_call_does_not_retry_422():
    client = _client()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise GithubException(422, "Unprocessable Entity", {})

    try:
        client._call(fn)
    except GithubException:
        pass
    else:
        raise AssertionError("expected GithubException")

    assert calls["n"] == 1


def test_constructor_uses_settings_token_and_disables_pygithub_retry(monkeypatch):
    captured = {}
    monkeypatch.setattr("app.ingest.github.client.Github", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        "app.ingest.github.client.get_settings",
        lambda: SimpleNamespace(github_token="from-settings"),
    )

    GitHubClient()

    assert captured["auth"].token == "from-settings"
    assert captured["retry"] is None
    assert captured["per_page"] == 100


def test_constructor_without_any_token_is_anonymous(monkeypatch):
    captured = {}
    monkeypatch.setattr("app.ingest.github.client.Github", lambda **kw: captured.update(kw))
    monkeypatch.setattr(
        "app.ingest.github.client.get_settings", lambda: SimpleNamespace(github_token="")
    )

    GitHubClient()

    assert captured["auth"] is None


def test_rate_limit_exception_is_recorded_waited_on_and_retried(monkeypatch):
    client = _client()
    events = []
    waits = []
    monkeypatch.setattr(
        "app.ingest.github.client.record_event",
        lambda source, kind, detail, context=None: events.append((source, kind, context)),
    )
    monkeypatch.setattr(client, "_wait_for_rate_limit_reset", lambda: waits.append(1))
    monkeypatch.setattr("time.sleep", lambda *_: None)
    calls = {"n": 0}

    def get_user():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitExceededException(403, "API rate limit exceeded", {})
        return "ok"

    assert client._call(get_user) == "ok"
    assert events == [("github", "rate_limited", "get_user")]
    assert waits == [1]


def test_404_is_not_recorded_as_a_rate_limit_event(monkeypatch):
    client = _client()
    events = []
    monkeypatch.setattr(
        "app.ingest.github.client.record_event", lambda *a, **kw: events.append(a)
    )

    def fn():
        raise UnknownObjectException(404, "Not Found", {})

    try:
        client._call(fn)
    except UnknownObjectException:
        pass

    assert events == []


def _repo(full_name: str, fork: bool = False, **methods) -> SimpleNamespace:
    return SimpleNamespace(full_name=full_name, fork=fork, **methods)


def _not_found():
    raise UnknownObjectException(404, "Not Found", {})


def _unauthorized():
    raise BadCredentialsException(401, "Bad credentials", {})


def test_user_repos_includes_forks_by_default_and_can_skip_them():
    client = _client()
    repos = [_repo("octocat/a"), _repo("octocat/b", fork=True), _repo("octocat/c")]
    requested = []

    def get_user(username):
        requested.append(username)
        return SimpleNamespace(get_repos=lambda: iter(repos))

    client._gh = SimpleNamespace(get_user=get_user)

    assert [r.full_name for r in client.user_repos("octocat")] == [
        "octocat/a",
        "octocat/b",
        "octocat/c",
    ]
    assert [r.full_name for r in client.user_repos("octocat", include_forks=False)] == [
        "octocat/a",
        "octocat/c",
    ]
    assert requested == ["octocat", "octocat"]


def test_get_repo_and_repo_count_hint_delegate_to_pygithub():
    client = _client()
    client._gh = SimpleNamespace(
        get_repo=lambda full_name: _repo(full_name),
        get_user=lambda username: SimpleNamespace(public_repos=42),
    )

    assert client.get_repo("octocat/Hello-World").full_name == "octocat/Hello-World"
    assert client.repo_count_hint("octocat") == 42


def test_readme_text_decodes_and_replaces_invalid_utf8():
    client = _client()
    repo = _repo("o/r", get_readme=lambda: SimpleNamespace(decoded_content=b"caf\xe9 # Title"))

    assert client.readme_text(repo) == "caf� # Title"


def test_readme_text_missing_readme_is_none():
    client = _client()
    assert client.readme_text(_repo("o/r", get_readme=_not_found)) is None


def test_readme_text_undecodable_payload_is_none():
    client = _client()
    repo = _repo("o/r", get_readme=lambda: SimpleNamespace(decoded_content=None))

    assert client.readme_text(repo) is None


def test_readme_text_other_errors_propagate():
    client = _client()
    try:
        client.readme_text(_repo("o/r", get_readme=_unauthorized))
    except BadCredentialsException:
        pass
    else:
        raise AssertionError("expected BadCredentialsException")


def test_root_contents_returns_a_list_for_dirs_files_and_empty_repos():
    client = _client()
    entries = [SimpleNamespace(name="README.md"), SimpleNamespace(name="pyproject.toml")]
    single = SimpleNamespace(name="only-file")

    assert client.root_contents(_repo("o/r", get_contents=lambda path: entries)) == entries
    assert client.root_contents(_repo("o/r", get_contents=lambda path: single)) == [single]
    assert client.root_contents(_repo("o/r", get_contents=lambda path: _not_found())) == []


def test_root_contents_other_errors_propagate():
    client = _client()
    try:
        client.root_contents(_repo("o/r", get_contents=lambda path: _unauthorized()))
    except BadCredentialsException:
        pass
    else:
        raise AssertionError("expected BadCredentialsException")


def test_file_text_file_directory_missing_and_undecodable():
    client = _client()
    requested = []

    def get_contents(path):
        requested.append(path)
        return SimpleNamespace(decoded_content=b'{"name": "x"}')

    assert client.file_text(_repo("o/r", get_contents=get_contents), "package.json") == (
        '{"name": "x"}'
    )
    assert requested == ["package.json"]
    assert client.file_text(_repo("o/r", get_contents=lambda p: [1, 2]), "src") is None
    assert client.file_text(_repo("o/r", get_contents=lambda p: _not_found()), "x") is None
    undecodable = _repo("o/r", get_contents=lambda p: SimpleNamespace(decoded_content=None))
    assert client.file_text(undecodable, "x") is None


def test_file_text_other_errors_propagate():
    client = _client()
    try:
        client.file_text(_repo("o/r", get_contents=lambda p: _unauthorized()), "x")
    except BadCredentialsException:
        pass
    else:
        raise AssertionError("expected BadCredentialsException")


def test_contributor_stats_waits_while_github_is_computing(monkeypatch):
    client = _client()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    answers = iter([None, None, ["stats"]])

    repo = _repo("o/r", get_stats_contributors=lambda: next(answers))

    assert client.contributor_stats(repo) == ["stats"]


def test_contributor_stats_gives_up_with_empty_list(monkeypatch):
    client = _client()
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def still_computing():
        calls["n"] += 1
        return None

    assert client.contributor_stats(_repo("o/r", get_stats_contributors=still_computing)) == []
    assert calls["n"] == 3
    assert len(sleeps) == 3
