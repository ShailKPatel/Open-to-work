"""Request checks and response headers.

Run locally the app has no login and trusts whoever can reach it, which is
meant to be the person at this machine (ports bind to 127.0.0.1). With
DEMO_MODE on it is public: app/api/demo.py handles sign-in and ownership,
and the checks below accept only the demo's own host name. Two browser
tricks would let a web page elsewhere act through a person's browser:

- Cross-site requests (CSRF): another site's page posting a form or a
  fetch here. State-changing methods are refused unless Sec-Fetch-Site,
  Origin or Referer say the request came from this app's own pages. A
  request with none of the three is not from a browser page (curl, a
  script) and is let through, since that caller could reach the app
  anyway.
- DNS rebinding: a site whose name is pointed at 127.0.0.1, so its pages
  count as same-origin with this app. Refused by the Host check, which
  accepts only loopback names. The port is not checked: the app runs on
  whatever free port `make start` picked, and a page cannot make a
  browser send a loopback Host for a name it does not own.

Every response also gets headers against content sniffing, framing by
other sites, and leaking URLs in Referer. No script-src policy: the
pages use inline scripts and Alpine.js evaluates its attribute
expressions, which would need 'unsafe-inline' and 'unsafe-eval', so it
would not stop much. Scripts and styles are served by the app itself
(see the assets stage in the Dockerfile), not a CDN. See docs/ARCHITECTURE.md, "Security model".

Plain ASGI rather than BaseHTTPMiddleware, so streamed responses (the
sync and extraction progress streams) pass through untouched.
"""

from __future__ import annotations

from http.cookies import SimpleCookie
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.settings import get_settings

ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# "none" is a request the person started themselves (typed URL, bookmark).
_OWN_FETCH_SITES = {"same-origin", "none"}

# Framing stays allowed for the app's own pages: the resume library and
# the build page show PDFs in an iframe.
_RESPONSE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "frame-ancestors 'self'; base-uri 'self'; form-action 'self'",
}


def _hostname(host: str) -> str:
    """The name part of a Host header: "localhost:8001" -> "localhost",
    "[::1]:8000" -> "::1"."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1 : host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _origin(url: str) -> str | None:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}".lower()


def refusal(
    method: str, headers: dict[str, str], scheme: str, public_host: str | None = None
) -> tuple[int, str] | None:
    """(status, reason) for a request to refuse, or None to let it
    through. headers are keyed by lowercase name. public_host is the demo's
    own name: then it is the only Host accepted, and its origin is https
    whatever scheme the proxy in front used to reach this process."""
    host = headers.get("host", "")
    allowed = {public_host.lower()} if public_host else ALLOWED_HOSTS
    if _hostname(host) not in allowed:
        return 400, "Unknown host. Open the app at http://localhost with its port."
    if method.upper() not in _UNSAFE_METHODS:
        return None

    cross_site = (403, "Cross-site request refused.")
    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.lower() not in _OWN_FETCH_SITES:
        return cross_site
    own = f"https://{public_host}".lower() if public_host else f"{scheme}://{host}".lower()
    if "origin" in headers:
        return None if headers["origin"].lower() == own else cross_site
    if "referer" in headers:
        return None if _origin(headers["referer"]) == own else cross_site
    return None


_DEMO_MAX_BODY_BYTES = 12 * 1024 * 1024
_DEMO_CLOSED_PATHS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def _demo_refusal(method: str, path: str, headers: dict[str, str]) -> tuple[int, str] | None:
    """What can be refused before routing. FastAPI's documentation routes
    never reach demo_gate; and a request that changes something has to be
    signed in before its body, possibly a large upload, is read."""
    from app.api.demo import session_account

    if path in _DEMO_CLOSED_PATHS:
        return 404, "Not found."
    length = headers.get("content-length", "")
    if length.isdigit() and int(length) > _DEMO_MAX_BODY_BYTES:
        return 413, "That upload is too large for the demo."
    if method.upper() in _UNSAFE_METHODS and not path.startswith("/auth/"):
        cookie = SimpleCookie(headers.get("cookie", "")).get("otw_session")
        if not cookie or session_account(cookie.value) is None:
            return 401, "Sign in first."
    return None


class SecurityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        public_host = settings.demo_public_host if settings.demo_mode else None
        path = scope["path"]

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if public_host:
                    headers["Strict-Transport-Security"] = "max-age=31536000"
                for name, value in _RESPONSE_HEADERS.items():
                    if name not in headers:
                        headers[name] = value
            await send(message)

        headers = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope["headers"]
        }
        refused = refusal(scope["method"], headers, scope.get("scheme", "http"), public_host)
        if public_host:
            refused = refused or _demo_refusal(scope["method"], path, headers)
            # The platform's health probe calls by IP, with no public Host.
            if path == "/health":
                refused = None
        if refused is not None:
            status, reason = refused
            await JSONResponse({"detail": reason}, status_code=status)(
                scope, receive, send_with_headers
            )
            return
        await self.app(scope, receive, send_with_headers)
