"""Route-level smoke tests across the whole app: every page renders, every
JSON list endpoint answers on an empty account, the OpenAPI schema builds,
and unknown paths or wrong methods fail cleanly. Catches a broken template,
a bad import, or a route that errors before any feature-specific test would.

Routes are discovered from the app itself, so a new page or endpoint is
covered here without editing this file.
"""

import re

import pytest
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import app.core.db as db_module
import app.retrieval.vectorstore as vectorstore_module
from app.api.main import app
from app.core.db import Account, get_db
from app.core.settings import get_settings

# GET endpoints left out of the generic sweep, each covered with fakes in
# its own test module.
_SKIP_JSON_GETS = {
    "/api/monitor/status": "calls the live GitHub rate-limit endpoint",
    "/api/resume/search": "loads the local embedding model",
    "/api/projects/process-pending/stream": "long-lived SSE stream",
    "/sync/github/stream": "SSE stream that syncs from GitHub",
    "/sync/github/status": "needs a username, covered in test_api.py",
}


def _page_paths() -> list[str]:
    paths = set()
    for route in app.routes:
        if (
            isinstance(route, APIRoute)
            and "GET" in route.methods
            and route.response_class is HTMLResponse
        ):
            paths.add(re.sub(r"\{[^}]+\}", "1", route.path))
    return sorted(paths)


def _json_get_paths() -> list[str]:
    pages = set(_page_paths())
    return sorted(
        path
        for path, operations in app.openapi()["paths"].items()
        if "get" in operations
        and "{" not in path
        and path not in pages
        and path not in _SKIP_JSON_GETS
    )


PAGE_PATHS = _page_paths()
JSON_GET_PATHS = _json_get_paths()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("RESUME_STORAGE_DIR", str(tmp_path / "resumes"))
    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    get_settings.cache_clear()
    with TestClient(app) as test_client:  # runs the lifespan, which creates the schema
        yield test_client
    get_settings.cache_clear()


@pytest.fixture
def account_id(client) -> int:
    db = get_db()
    try:
        account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
        db.add(account)
        db.commit()
        return account.id
    finally:
        db.close()


def test_route_discovery_found_the_app_surface():
    """Guards the sweeps below against silently parametrizing over nothing."""
    assert len(PAGE_PATHS) >= 20
    assert {"/", "/home", "/portfolio", "/jobs", "/explanation", "/apis"} <= set(PAGE_PATHS)
    assert len(JSON_GET_PATHS) >= 10
    assert {"/accounts", "/api/projects", "/api/job-postings"} <= set(JSON_GET_PATHS)


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_every_page_renders(client, path):
    resp = client.get(path)

    assert resp.status_code == 200, resp.text[:500]
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text.lower()
    assert "<title>" in body
    assert "</html>" in body


@pytest.mark.parametrize("path", JSON_GET_PATHS)
def test_every_json_list_endpoint_answers_on_an_empty_account(client, account_id, path):
    resp = client.get(path, params={"account_id": account_id})

    assert resp.status_code == 200, resp.text[:500]
    assert resp.headers["content-type"].startswith("application/json")
    resp.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openapi_schema_documents_every_api_area(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    for prefix in (
        "/accounts",
        "/api/api-keys",
        "/api/auth-sources",
        "/api/accounts/{account_id}/contact",
        "/api/education",
        "/api/experience",
        "/api/job-analytics",
        "/api/job-postings",
        "/api/monitor",
        "/api/projects",
        "/api/resume",
        "/api/resume-build",
        "/api/skills",
        "/api/sources",
        "/sync/github",
    ):
        assert any(p.startswith(prefix) for p in paths), prefix


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_interactive_api_docs_serve(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_unknown_path_is_a_json_404(client):
    resp = client.get("/definitely-not-a-route")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not Found"}


def test_wrong_method_is_a_405(client):
    assert client.delete("/health").status_code == 405


def test_malformed_path_parameter_is_a_422(client):
    assert client.get("/api/projects/not-an-int").status_code == 422
