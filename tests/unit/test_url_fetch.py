"""app/ingest/jobs/url_fetch.py: public-URL job-posting fetch. httpx is
mocked (no real network call); real BeautifulSoup does the HTML stripping,
so that part is proven against real parsing, not a stand-in.
"""

from unittest.mock import MagicMock

import httpx
import pytest

from app.ingest.jobs.url_fetch import JobUrlFetchError, fetch_job_url


def _fake_response(status_code=200, text="", raise_error=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    return resp


def test_fetch_strips_html_and_returns_visible_text(monkeypatch):
    html = """
    <html><head><title>Backend Engineer at Acme</title></head>
    <body>
      <script>trackingCode();</script>
      <style>.hidden{}</style>
      <nav>Home | About</nav>
      <main>
        <h1>Backend Engineer</h1>
        <p>We are looking for someone with 5+ years of Python experience.</p>
        <p>Salary: $120k-$150k. Apply now to join our growing team of engineers
           who love building reliable, well-tested backend services at scale.</p>
      </main>
    </body></html>
    """
    monkeypatch.setattr(
        "httpx.get", MagicMock(return_value=_fake_response(200, html))
    )

    title, text = fetch_job_url("https://acme.example/careers/backend")

    assert title == "Backend Engineer at Acme"
    assert "Python experience" in text
    assert "trackingCode" not in text  # script content stripped


def test_fetch_raises_on_http_error(monkeypatch):
    monkeypatch.setattr("httpx.get", MagicMock(return_value=_fake_response(404, "")))
    with pytest.raises(JobUrlFetchError):
        fetch_job_url("https://acme.example/gone")


def test_fetch_raises_on_network_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr("httpx.get", _raise)
    with pytest.raises(JobUrlFetchError):
        fetch_job_url("https://unreachable.example/x")


def test_fetch_raises_when_page_has_too_little_text(monkeypatch):
    login_wall_html = "<html><body>Sign in</body></html>"
    monkeypatch.setattr("httpx.get", MagicMock(return_value=_fake_response(200, login_wall_html)))
    with pytest.raises(JobUrlFetchError):
        fetch_job_url("https://loginwalled.example/job")


def test_fetch_rejects_blank_url():
    with pytest.raises(JobUrlFetchError):
        fetch_job_url("   ")


def test_fetch_truncates_very_long_text(monkeypatch):
    long_paragraph = "Requirement detail line. " * 2000
    html = f"<html><body><p>{long_paragraph}</p></body></html>"
    monkeypatch.setattr("httpx.get", MagicMock(return_value=_fake_response(200, html)))

    _, text = fetch_job_url("https://acme.example/long")

    assert len(text) <= 20_000
