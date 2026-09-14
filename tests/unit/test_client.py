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

from github import BadCredentialsException, GithubException, UnknownObjectException

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
