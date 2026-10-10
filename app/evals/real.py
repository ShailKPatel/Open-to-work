"""The real-text eval set: public job postings and public open-source
repositories, labeled by hand, scored the same way as the synthetic set
(app/evals/synthetic.py) and inside the same throwaway environment.

evals/real/sources.yaml pins what is used, evals/real/labels.yaml holds
the labels, and the text itself is downloaded into evals/real/cache/
(gitignored) by scripts/fetch_real_eval_data.py. DATA_SOURCES.md next to
them covers provenance, privacy and what these numbers cannot tell you.

The repositories together play one developer's portfolio. There is no
real resume behind it: no source of real work history exists that is
both consented and free to read, so experience points are not part of
this set and stay synthetic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

REAL_DIR = Path(__file__).resolve().parents[2] / "evals" / "real"
CACHE_DIR = REAL_DIR / "cache"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
# Seven or more digits once separators are ignored, optionally led by a
# country code: long enough to skip years, salaries and requisition ids.
_PHONE = re.compile(r"(?<![\w$])\+?\(?\d[\d ().-]{7,}\d(?![\w%])")
# A labeled contact line ("Recruiter: Jane Doe"). Deliberately narrow:
# a looser rule ("contact" anywhere near two capitalized words) matched
# accessibility and pay-range boilerplate, never a person.
_CONTACT_LINE = re.compile(
    r"(?im)^\s*(recruiter|hiring manager|contact|point of contact)\s*[:\-]\s*"
    r"[A-Z][a-z]+(?: [A-Z][a-z'.-]+)+\s*$"
)


def scrub_pii(text: str) -> str:
    """Removes what could identify or reach a person from posting text:
    email addresses, phone numbers, and lines that name a contact. Company
    names, which are the point of a posting, stay."""
    text = _EMAIL.sub("[email removed]", text)
    text = _PHONE.sub(_phone_or_number, text)
    return _CONTACT_LINE.sub("[contact line removed]", text)


def _phone_or_number(match: re.Match[str]) -> str:
    """Keeps a pay figure or a range like "2020-2024" that merely looks
    like a phone number; replaces anything with phone-style grouping."""
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 7 or "," in raw:
        return raw
    if re.fullmatch(r"\d{4}\s*-\s*\d{4}", raw.strip()):
        return raw
    return "[phone removed]"


def load_sources(path: Path | None = None) -> dict[str, Any]:
    return dict(yaml.safe_load((path or REAL_DIR / "sources.yaml").read_text(encoding="utf-8")))


def repo_cache_path(repo: str) -> Path:
    return CACHE_DIR / "repos" / f"{repo.replace('/', '__')}.json"


def posting_cache_path(board: str, job_id: int) -> Path:
    return CACHE_DIR / "postings" / f"{board}-{job_id}.json"


def read_cached(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


# Inside the synthetic id range (app/evals/synthetic.py), so it shares that
# module's isolation and id-block rules, and far from every persona there.
REAL_ACCOUNT_ID = 9190


class CacheMissingError(Exception):
    """The cache has not been built yet; scripts/fetch_real_eval_data.py
    builds it."""


def load_labels(path: Path | None = None) -> dict[str, Any]:
    return dict(yaml.safe_load((path or REAL_DIR / "labels.yaml").read_text(encoding="utf-8")))


def posting_key(board: str, job_id: int) -> str:
    return f"{board}-{job_id}"


def build_portfolio(
    sources: dict[str, Any] | None = None, labels: dict[str, Any] | None = None
) -> Any:
    """The pinned repositories as one synthetic-shaped Persona: README and
    manifests from the cache, skills from the labels. No roles, since no
    real work history is used (see the module docstring)."""
    from app.evals.synthetic import Persona, Repo

    sources = sources or load_sources()
    labels = labels or load_labels()
    repos = []
    for item in sources["repos"]:
        cached = read_cached(repo_cache_path(item["repo"]))
        if cached is None:
            raise CacheMissingError(
                f"{item['repo']} is not cached; run python -m scripts.fetch_real_eval_data"
            )
        repos.append(
            Repo(
                key=item["repo"],
                name=item["repo"].split("/", 1)[1],
                language=item.get("language"),
                stars=int(item.get("stars") or 0),
                description=item.get("description") or "",
                readme=cached.get("readme"),
                manifests={
                    name: list(manifest.get("dependencies", []))
                    for name, manifest in (cached.get("manifests") or {}).items()
                },
                readme_skills=list(labels["repos"][item["repo"]]["skills"]),
            )
        )
    return Persona(
        key="real-portfolio",
        account_id=REAL_ACCOUNT_ID,
        first_name="Portfolio",
        last_name="Composite",
        github_username="real-portfolio",
        headline="Composite open-source portfolio",
        location="",
        email="portfolio@example.invalid",
        phone="",
        links=[],
        resume_style="classic",
        date_style="mon_year",
        covers=["real-text"],
        repos=repos,
        roles=[],
        education=[],
    )


def build_jobs(
    sources: dict[str, Any] | None = None, labels: dict[str, Any] | None = None
) -> list[Any]:
    """Every pinned posting as a Job: cached text, labeled expectations,
    and the portfolio as its only retrieval target when any portfolio
    skill is relevant to it."""
    from app.evals.synthetic import Job

    sources = sources or load_sources()
    labels = labels or load_labels()
    jobs = []
    for item in sources["postings"]:
        key = posting_key(item["board"], item["id"])
        cached = read_cached(posting_cache_path(item["board"], item["id"]))
        if cached is None:
            raise CacheMissingError(
                f"posting {key} is not cached; run python -m scripts.fetch_real_eval_data"
            )
        label = labels["postings"][key]
        relevant = list(label.get("relevant_skills") or [])
        jobs.append(
            Job(
                key=key,
                for_personas=["real-portfolio"] if relevant else [],
                covers=[item.get("role", "")],
                text=posting_text(cached),
                expected={**label["expected"], "summary": label.get("summary", "")},
                relevant_skills=relevant,
            )
        )
    return jobs


def posting_text(cached: dict[str, Any]) -> str:
    """The posting as someone pasting it from the job board would copy it:
    the header the board shows (title, company, location) and then the
    description. Greenhouse returns those header fields apart from the
    description body, so without this the extraction eval would be asking
    the model for a title and location the text never shows."""
    header = [cached.get("title", ""), cached.get("company", ""), cached.get("location", "")]
    lines = [line for line in header if line]
    return "\n".join([*lines, "", cached.get("text", "")]).strip()


def run_real_retrieval_eval() -> list[Any]:
    """Scores the portfolio against the labeled postings in the same
    throwaway environment the synthetic set uses."""
    from app.evals.synthetic import run_retrieval_eval

    return run_retrieval_eval([build_portfolio()], build_jobs())
