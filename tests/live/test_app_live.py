"""Smoke checks against a running instance, such as one started with
`make up`. Not part of `make test`; run with `make test-live`.

Skipped unless LIVE_APP_URL is set (for example
LIVE_APP_URL=http://localhost:8000). Read-only: nothing here creates,
changes, or deletes data in the target instance.
"""

from __future__ import annotations

import os

import pytest
import requests

_BASE = os.environ.get("LIVE_APP_URL", "").rstrip("/")

pytestmark = pytest.mark.skipif(
    not _BASE,
    reason="LIVE_APP_URL not set; live app checks are opt-in (need a running instance)",
)

_PAGES = [
    "/",
    "/home",
    "/portfolio",
    "/portfolio/projects",
    "/portfolio/experience",
    "/portfolio/education",
    "/portfolio/skills",
    "/portfolio/contact-links",
    "/portfolio/resume",
    "/portfolio/resume/build",
    "/jobs",
    "/jobs/analytics",
    "/monitor",
    "/explanation",
    "/settings",
    "/monitor/sync",
    "/apis",
]


def _get(path: str) -> requests.Response:
    return requests.get(f"{_BASE}{path}", timeout=30)


def test_health():
    resp = _get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.parametrize("path", _PAGES)
def test_page_renders(path):
    resp = _get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_openapi_schema():
    resp = _get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["paths"]


@pytest.mark.parametrize("path", ["/accounts", "/api/api-keys/providers", "/api/monitor/events"])
def test_json_endpoint(path):
    resp = _get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


def test_monitor_status_reaches_github_and_reports_llm_budget():
    """The running app's own GitHub rate-limit check and LLM budget readout,
    so this confirms the deployed container can reach GitHub."""
    resp = _get("/api/monitor/status")
    assert resp.status_code == 200
    body = resp.json()
    print(f"\n[live:app] github={body['github']} llm={body['llm']}")
    assert body["github"]
    assert body["llm"]["monthly_budget_usd"] >= 0
