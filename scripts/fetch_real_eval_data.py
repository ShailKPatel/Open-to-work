"""Downloads the real-text eval set listed in evals/real/sources.yaml into
evals/real/cache/ (gitignored): each pinned repository's README, About
line and root manifests, and each job posting's text. Already cached
files are kept, so a posting that has since closed stays readable.

Posting text is scrubbed before it is written (app/evals/real.py's
scrub_pii): email addresses, phone numbers and named contact lines are
replaced, since none of it is needed for the eval and none of it should
sit on disk.

No login and no GitHub API calls: repository files come from
raw.githubusercontent.com at the pinned commit, and the About line,
language and star count are recorded in sources.yaml. Postings come from
the public Greenhouse job board API.

Usage: .venv/bin/python -m scripts.fetch_real_eval_data [--refresh]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re

import httpx

from app.evals.real import CACHE_DIR, load_sources, posting_cache_path, repo_cache_path, scrub_pii
from app.ingest.github.manifests import MANIFEST_FILENAMES, parse_dependencies

_README_NAMES = ("README.md", "README.rst", "README.markdown", "README.txt", "README")
_TIMEOUT = 30.0


def _html_to_text(raw: str) -> str:
    """Greenhouse returns the description as escaped HTML. Block tags
    become line breaks so lists keep one item per line."""
    text = html.unescape(raw)
    text = re.sub(r"(?i)<\s*(br|/p|/li|/h[1-6]|/div|/ul|/ol)\s*/?>", "\n", text)
    text = re.sub(r"(?i)<\s*li[^>]*>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _fetch_repo(client: httpx.Client, item: dict) -> dict:
    repo, sha = item["repo"], item["sha"]
    base = f"https://raw.githubusercontent.com/{repo}/{sha}"
    readme = None
    for name in _README_NAMES:
        response = client.get(f"{base}/{name}")
        if response.status_code == 200:
            readme = response.text
            # A symlinked README comes back as its target path, not text.
            target = readme.strip()
            if "\n" not in target and target.lower().endswith((".md", ".rst")):
                linked = client.get(f"{base}/{target}")
                if linked.status_code == 200:
                    readme = linked.text
            break
    manifests = {}
    for filename, ecosystem in MANIFEST_FILENAMES.items():
        response = client.get(f"{base}/{filename}")
        if response.status_code == 200:
            manifests[filename] = {
                "ecosystem": ecosystem,
                "dependencies": parse_dependencies(filename, response.text),
            }
    return {
        "repo": repo,
        "sha": sha,
        "description": item.get("description") or "",
        "language": item.get("language"),
        "stars": item.get("stars") or 0,
        "readme": readme,
        "manifests": manifests,
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def _fetch_posting(client: httpx.Client, board: str, job_id: int) -> dict:
    response = client.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}")
    response.raise_for_status()
    job = response.json()
    company = (job.get("company_name") or board).strip()
    return {
        "board": board,
        "id": job_id,
        "company": company,
        "title": job.get("title", "").strip(),
        "location": (job.get("location") or {}).get("name", ""),
        "url": job.get("absolute_url", ""),
        "text": scrub_pii(_html_to_text(job.get("content", ""))),
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the real-text eval set.")
    parser.add_argument(
        "--refresh", action="store_true", help="re-download files that are already cached"
    )
    args = parser.parse_args()

    sources = load_sources()
    (CACHE_DIR / "repos").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "postings").mkdir(parents=True, exist_ok=True)
    fetched = kept = failed = 0
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
        for item in sources["repos"]:
            path = repo_cache_path(item["repo"])
            if path.exists() and not args.refresh:
                kept += 1
                continue
            try:
                path.write_text(json.dumps(_fetch_repo(client, item), indent=1))
                fetched += 1
            except httpx.HTTPError as e:
                print(f"repo {item['repo']}: {e}")
                failed += 1
        for item in sources["postings"]:
            path = posting_cache_path(item["board"], item["id"])
            if path.exists() and not args.refresh:
                kept += 1
                continue
            try:
                posting = _fetch_posting(client, item["board"], item["id"])
                path.write_text(json.dumps(posting, indent=1))
                fetched += 1
            except httpx.HTTPError as e:
                print(f"posting {item['board']}/{item['id']}: {e}")
                failed += 1
    print(f"fetched {fetched}, kept {kept} cached, failed {failed}; cache at {CACHE_DIR}")


if __name__ == "__main__":
    main()
