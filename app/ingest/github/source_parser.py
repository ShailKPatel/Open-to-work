"""Parses whatever someone pastes into the "fetch data from" box on the
sync-sources page: a bare username, a github.com profile URL, or a
github.com repo URL. Pure: no network call, doesn't check whether the
thing actually exists on GitHub (that happens at sync time, same 404
handling already built for the plain username case).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

SourceKind = Literal["user", "repo"]

# GitHub's own username rule: alphanumeric or single hyphens, no leading/
# trailing hyphen, max 39 chars.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


class InvalidSourceError(ValueError):
    """Raised on input that isn't a recognizable username or github.com
    link, never on a well-formed input that just doesn't exist on GitHub.
    """


@dataclass
class ParsedSource:
    kind: SourceKind
    username: str
    repo_full_name: str | None = None  # "owner/repo", set only when kind == "repo"


def parse_source(raw: str) -> ParsedSource:
    text = raw.strip()
    if not text:
        raise InvalidSourceError("empty input")

    # Bare username: no slash, no "github.com" mention at all.
    if "/" not in text and "github.com" not in text.lower():
        username = text.lstrip("@")
        _validate_username(username, raw)
        return ParsedSource(kind="user", username=username)

    candidate = text if "://" in text else f"https://{text}"
    parsed = urlparse(candidate)
    host = (parsed.netloc or "").lower()
    if host not in ("github.com", "www.github.com"):
        raise InvalidSourceError(f"not a github.com link: {raw!r}")

    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        raise InvalidSourceError(f"no username in link: {raw!r}")

    username = parts[0]
    _validate_username(username, raw)

    if len(parts) == 1:
        return ParsedSource(kind="user", username=username)

    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo:
        raise InvalidSourceError(f"no repo name in link: {raw!r}")
    return ParsedSource(kind="repo", username=username, repo_full_name=f"{username}/{repo}")


def _validate_username(username: str, raw: str) -> None:
    if not _USERNAME_RE.match(username):
        raise InvalidSourceError(f"not a valid GitHub username in {raw!r}: {username!r}")
