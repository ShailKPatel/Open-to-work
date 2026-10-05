"""Request checks and response headers for an app with no login.

The app trusts whoever can reach it, which is meant to be the person at
this machine (ports bind to 127.0.0.1). Two browser tricks would let a
web page elsewhere act through that person's browser:

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
pages load Tailwind and Alpine.js from a CDN and use inline scripts,
which would need 'unsafe-inline' and 'unsafe-eval', so it would not
stop much. See docs/ARCHITECTURE.md, "Security model".

Plain ASGI rather than BaseHTTPMiddleware, so streamed responses (the
sync and extraction progress streams) pass through untouched.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


def refusal(method: str, headers: dict[str, str], scheme: str) -> tuple[int, str] | None:
    """(status, reason) for a request to refuse, or None to let it
    through. headers are keyed by lowercase name."""
    host = headers.get("host", "")
    if _hostname(host) not in ALLOWED_HOSTS:
        return 400, "Unknown host. Open the app at http://localhost with its port."
    if method.upper() not in _UNSAFE_METHODS:
        return None

    cross_site = (403, "Cross-site request refused.")
    fetch_site = headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site.lower() not in _OWN_FETCH_SITES:
        return cross_site
    own = f"{scheme}://{host}".lower()
    if "origin" in headers:
        return None if headers["origin"].lower() == own else cross_site
    if "referer" in headers:
        return None if _origin(headers["referer"]) == own else cross_site
    return None


class SecurityMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in _RESPONSE_HEADERS.items():
                    if name not in headers:
                        headers[name] = value
            await send(message)

        headers = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in scope["headers"]
        }
        refused = refusal(scope["method"], headers, scope.get("scheme", "http"))
        if refused is not None:
            status, reason = refused
            await JSONResponse({"detail": reason}, status_code=status)(
                scope, receive, send_with_headers
            )
            return
        await self.app(scope, receive, send_with_headers)
