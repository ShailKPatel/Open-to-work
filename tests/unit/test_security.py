"""app/api/security.py: the Host check, the cross-site request check and
the response headers, through the real app."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.security import refusal

_OWN = "http://localhost:8001"


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    """Requests that get past the middleware reach routes that open the
    database; point it at a throwaway file, never the one in .env."""
    import app.core.db as db_module
    from app.core.db import init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    get_settings.cache_clear()
    init_db()
    yield
    db_module.reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app, base_url=_OWN)


def test_same_origin_post_is_let_through(client):
    resp = client.post(
        "/api/job-postings/999999/reprocess",
        headers={"Origin": _OWN, "Sec-Fetch-Site": "same-origin"},
    )

    # Past the middleware: the route itself answers.
    assert resp.status_code == 404


def test_cross_origin_post_is_refused(client):
    resp = client.post(
        "/api/job-postings/999999/reprocess", headers={"Origin": "https://evil.example"}
    )

    assert resp.status_code == 403
    assert resp.json() == {"detail": "Cross-site request refused."}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_every_state_changing_method_is_checked(client, method):
    resp = client.request(method, "/accounts/1", headers={"Origin": "https://evil.example"})

    assert resp.status_code == 403


def test_another_local_port_is_a_different_origin(client):
    resp = client.post(
        "/api/job-postings/999999/reprocess",
        headers={"Origin": "http://localhost:3000", "Sec-Fetch-Site": "same-site"},
    )

    assert resp.status_code == 403


def test_null_origin_is_refused(client):
    resp = client.post("/api/job-postings/999999/reprocess", headers={"Origin": "null"})

    assert resp.status_code == 403


def test_sec_fetch_site_cross_site_is_refused_without_origin(client):
    resp = client.post(
        "/api/job-postings/999999/reprocess", headers={"Sec-Fetch-Site": "cross-site"}
    )

    assert resp.status_code == 403


def test_referer_is_used_when_origin_is_missing(client):
    own = client.post(
        "/api/job-postings/999999/reprocess", headers={"Referer": f"{_OWN}/jobs/1"}
    )
    other = client.post(
        "/api/job-postings/999999/reprocess",
        headers={"Referer": "https://evil.example/page"},
    )

    assert own.status_code == 404
    assert other.status_code == 403


def test_request_with_no_browser_headers_is_let_through(client):
    # curl or a script: no Origin, Referer or Sec-Fetch-Site. It can reach
    # the app directly anyway, so there is nothing to protect against.
    resp = client.post("/api/job-postings/999999/reprocess")

    assert resp.status_code == 404


def test_cross_origin_get_is_not_blocked(client):
    resp = client.get("/health", headers={"Origin": "https://evil.example"})

    assert resp.status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8001", "192.168.1.5:8001"])
def test_unknown_host_is_refused(host):
    # DNS rebinding: a page on evil.example whose name now points at
    # 127.0.0.1 still sends its own name as Host.
    resp = TestClient(app, base_url=f"http://{host}").get("/health")

    assert resp.status_code == 400
    assert "Unknown host" in resp.json()["detail"]


@pytest.mark.parametrize(
    "host", ["localhost", "localhost:8000", "127.0.0.1:8123", "[::1]:8000", "LOCALHOST:8000"]
)
def test_loopback_hosts_on_any_port_are_allowed(host):
    assert refusal("GET", {"host": host}, "http") is None


def test_responses_carry_security_headers(client):
    resp = client.get("/health")

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "SAMEORIGIN"
    assert resp.headers["referrer-policy"] == "same-origin"
    assert "frame-ancestors 'self'" in resp.headers["content-security-policy"]


def test_refused_responses_carry_security_headers_too(client):
    resp = client.post("/accounts", headers={"Origin": "https://evil.example"})

    assert resp.status_code == 403
    assert resp.headers["x-content-type-options"] == "nosniff"
