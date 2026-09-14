"""Public-URL job-posting fetch: no login, no ToS problem, works for most
company career pages and public ATS listings (Greenhouse/Lever/Ashby-style
pages, and plenty of others). A real HTTP GET, HTML stripped down to
visible text, handed to app/profile/job_extract.py exactly like pasted
text is. Login-walled pages belong to app/ingest/jobs/auth_fetch.py
instead; this module makes no attempt to authenticate anywhere and simply
fails informatively if the page it gets back doesn't look like a real
posting (a login wall, a bot-check interstitial, an empty SPA shell).
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# A generic browser UA: some career-page CDNs 403 a bare "python-httpx/x.y"
# user agent outright even for public pages.
_USER_AGENT = (
    "Mozilla/5.0 (compatible; OpenToWorkBot/1.0; personal job-tracking tool)"
)
_TIMEOUT_S = 15.0
# Hard cap on how much extracted text gets handed to the LLM: a real
# posting page is a few KB of real content; anything past this is either
# an unusually long posting or boilerplate that slipped through the strip,
# and either way the extraction prompt doesn't need more than this to work
# with. Keeps token cost bounded regardless of what a URL returns.
_MAX_CHARS = 20_000


class JobUrlFetchError(Exception):
    """The URL couldn't be fetched, or came back with nothing usable."""


def _strip_html(html: str) -> tuple[str, str]:
    """(title, visible_text). BeautifulSoup, not a regex tag-stripper: a
    regex can't correctly handle nested tags, script/style content, or
    malformed real-world HTML, and a job page is exactly the kind of page
    likely to have all three.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return title, text


def fetch_job_url(url: str) -> tuple[str, str]:
    """Returns (title_guess, text). Raises JobUrlFetchError on anything
    that isn't a usable page: unreachable, non-2xx, or effectively empty
    once stripped (a common symptom of a login wall or a JS-only SPA that
    renders nothing server-side; this function does not run a browser,
    see auth_fetch.py for that).
    """
    url = url.strip()
    if not url:
        raise JobUrlFetchError("url is required")

    try:
        response = httpx.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT_S,
            follow_redirects=True,
        )
    except httpx.RequestError as e:
        raise JobUrlFetchError(f"could not reach {url}: {e}") from e

    if response.status_code >= 400:
        raise JobUrlFetchError(f"{url} returned HTTP {response.status_code}")

    title, text = _strip_html(response.text)
    if len(text) < 200:
        raise JobUrlFetchError(
            "that page didn't have enough readable text once stripped of markup; "
            "it may require login, or render its content with JavaScript this "
            "fetch doesn't run. Try pasting the text directly, or a screenshot."
        )

    return title, text[:_MAX_CHARS]
