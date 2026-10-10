"""The public-dataset eval set: job postings from the LinkedIn Job
Postings (2023-2024) dataset (CC BY-SA 4.0), scored for extraction against
the structured fields each poster filled in on LinkedIn's own form.

Those fields are the labels, so nobody on this project wrote them. A field
is scored only when the posting's text gives the model a fair chance at
it; the rules are in expected_fields() and evals/public/DATA_SOURCES.md.

The software postings in the sample double as a retrieval test set for
the composite open-source portfolio (app/evals/real.py). Their relevance
labels are judgments, made before the system was first run on them and
kept in evals/public/retrieval_labels.yaml.

evals/public/sources.yaml pins the sampled job ids and their split; the
text is downloaded into evals/public/cache/ (gitignored) by
scripts/fetch_public_eval_data.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from app.evals.splits import TEST, split_for

PUBLIC_DIR = Path(__file__).resolve().parents[2] / "evals" / "public"
CACHE_DIR = PUBLIC_DIR / "cache"

DATASET_URL = (
    "https://huggingface.co/datasets/datastax/linkedin_job_listings/resolve/main/postings.csv"
)

# LinkedIn's work types in the app's employment_type vocabulary. Temporary,
# Volunteer and Other have no counterpart and are not scored.
_WORK_TYPES = {
    "Full-time": "Full-time",
    "Part-time": "Part-time",
    "Contract": "Contract",
    "Internship": "Internship",
}

# How a posting's text says it, per work type. Only postings whose text
# says it are scored on employment type: the form value alone is not
# something a model reading the text could know.
_TYPE_WORDS = {
    "Full-time": ("full-time", "full time"),
    "Part-time": ("part-time", "part time"),
    "Contract": ("contract", "contractor"),
    "Internship": ("intern", "internship"),
}

# Any way a posting states pay: a dollar amount, a figure in thousands
# ("160k"), or a rate per hour ("13.07 / Hour", "50HR"). Used only to
# decide that a posting states no pay at all.
_PAY_STATED = re.compile(
    r"\$\s?\d"
    r"|\b\d[\d,.]*\s?k\b"
    r"|\b\d[\d,.]*\s?(?:/\s?(?:hr|hour)\b|per hour|an hour|hourly|hr\b)",
    re.IGNORECASE,
)

SOFTWARE_TITLE = re.compile(
    r"(?i)\b(software|developer|programmer|data scientist|data engineer|devops|"
    r"machine learning|ml engineer|back[- ]?end|front[- ]?end|full[- ]?stack|"
    r"site reliability|cloud engineer|platform engineer|android|ios|web developer|"
    r"python|java|golang|kubernetes|sre)\b"
)


def posting_key(job_id: int | str) -> str:
    return f"linkedin-{job_id}"


def cache_path(job_id: int | str) -> Path:
    return CACHE_DIR / f"{posting_key(job_id)}.json"


# Software postings drawn later as a fresh retrieval test set, after the
# first test set was spent (scripts/fetch_public_eval_data.py --add-fresh).
# Not part of the extraction set, whose sample stays the original 200.
FRESH_GROUP = "software-fresh"
# A second fresh sample, larger, labeled by a validated LLM annotator
# (evals/public/annotator_prompt.md) rather than by hand. Its labels live
# in retrieval_labels_llm.yaml, apart from the human ones.
FRESH2_GROUP = "software-fresh-2"


def item_split(item: dict[str, Any]) -> str:
    """A posting's split: pinned in sources.yaml for fresh postings, by
    hash for the rest."""
    return str(item.get("split") or split_for(posting_key(item["id"])))


def load_sources(path: Path | None = None) -> dict[str, Any]:
    return dict(yaml.safe_load((path or PUBLIC_DIR / "sources.yaml").read_text(encoding="utf-8")))


def read_cached(job_id: int | str) -> dict[str, Any] | None:
    path = cache_path(job_id)
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _amount(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _money_forms(amount: float) -> set[str]:
    """How an amount appears in posting text: 120000 as "120,000",
    "120000" or "120k"; 25.5 as "25.50" or "25.5"."""
    forms = {f"{amount:,.2f}", f"{amount:.2f}"}
    if amount == int(amount):
        whole = int(amount)
        forms |= {f"{whole:,}", str(whole)}
        if whole % 1000 == 0 and whole >= 1000:
            forms.add(f"{whole // 1000}k")
    else:
        forms.add(f"{amount:g}")
    return forms


def _states(text: str, amount: float) -> bool:
    lowered = text.casefold()
    return any(form.casefold() in lowered for form in _money_forms(amount))


def expected_fields(row: dict[str, Any]) -> dict[str, Any]:
    """The labels for one posting, from its form fields. A field set to
    None is not scored.

    title, company, location: as the poster entered them.
    salary_range: the form's minimum and maximum when the description
      states both; an empty string when the form has no salary and the
      description states no pay in any form, which checks that the
      model does not invent one; otherwise not scored.
    employment_type: the form's work type when the description says it
      in words (and the type has a counterpart in the app); otherwise not
      scored.
    work_mode: "Remote" when the form allows remote work and the
      description says "remote"; otherwise not scored, since the form has
      no way to say on-site or hybrid.
    """
    text = row.get("description") or ""
    lowered = text.casefold()
    low, high = _amount(row.get("min_salary")), _amount(row.get("max_salary"))
    salary: str | None = None
    if low is not None and high is not None:
        if _states(text, low) and _states(text, high):
            salary = f"{low:g}" if low == high else f"{low:g}-{high:g}"
    elif low is None and high is None and _amount(row.get("med_salary")) is None:
        if not _PAY_STATED.search(text):
            salary = ""
    work_type = _WORK_TYPES.get(row.get("formatted_work_type") or "")
    employment = None
    if work_type and any(
        re.search(rf"\b{re.escape(w)}\b", lowered) for w in _TYPE_WORDS[work_type]
    ):
        employment = work_type
    remote = str(row.get("remote_allowed") or "") in ("1", "1.0")
    work_mode = "Remote" if remote and re.search(r"\bremote\b", lowered) else None
    return {
        "company": row.get("company_name") or None,
        "title": row.get("title") or None,
        "location": row.get("location") or None,
        "salary_range": salary,
        "employment_type": employment,
        "work_mode": work_mode,
    }


def posting_text(row: dict[str, Any]) -> str:
    """The posting as copied from its LinkedIn page: the header the page
    shows (title, company, location) and then the description. Salary and
    work type are left out of the header, so scoring them tests reading the
    description, not copying a label."""
    header = [row.get("title") or "", row.get("company_name") or "", row.get("location") or ""]
    lines = [line for line in header if line]
    return "\n".join([*lines, "", (row.get("description") or "").strip()]).strip()


def build_jobs(split: str | None = TEST, sources: dict[str, Any] | None = None) -> list[Any]:
    """The sampled postings as Jobs with form-derived labels, limited to
    one split (None for every posting)."""
    from app.evals.real import CacheMissingError
    from app.evals.synthetic import Job

    sources = sources or load_sources()
    jobs = []
    for item in sources["postings"]:
        if item.get("group") in (FRESH_GROUP, FRESH2_GROUP):
            continue
        if split is not None and item_split(item) != split:
            continue
        row = read_cached(item["id"])
        if row is None:
            raise CacheMissingError(
                f"{posting_key(item['id'])} is not cached; "
                "run python -m scripts.fetch_public_eval_data"
            )
        jobs.append(
            Job(
                key=posting_key(item["id"]),
                for_personas=[],
                covers=[item.get("group", "")],
                text=posting_text(row),
                expected=expected_fields(row),
            )
        )
    return jobs


def load_retrieval_labels(path: Path | None = None) -> dict[str, list[str]]:
    raw = yaml.safe_load((path or PUBLIC_DIR / "retrieval_labels.yaml").read_text("utf-8"))
    return {key: list(skills or []) for key, skills in dict(raw).items()}


def extraction_path(key: str) -> Path:
    return CACHE_DIR / "extractions" / f"{key}.json"


def build_retrieval_jobs(
    split: str | None = TEST,
    sources: dict[str, Any] | None = None,
    groups: tuple[str, ...] = ("software",),
    labels_path: Path | None = None,
) -> list[Any]:
    """The software postings as retrieval queries against the composite
    portfolio, searched the way the app searches a saved posting: with the
    title, role summary and skills the app's own extraction produced
    (saved by the extraction eval, scripts/run_llm_evals.py --only public
    public-dev), not with hand-written queries."""
    from app.evals.real import CacheMissingError
    from app.evals.synthetic import Job

    sources = sources or load_sources()
    labels = load_retrieval_labels(labels_path)
    jobs = []
    for item in sources["postings"]:
        key = posting_key(item["id"])
        if item.get("group") not in groups or key not in labels:
            continue
        if split is not None and item_split(item) != split:
            continue
        row = read_cached(item["id"])
        path = extraction_path(key)
        if not path.exists():
            raise CacheMissingError(
                f"{key} has no saved extraction; run "
                "python -m scripts.run_llm_evals --only public public-dev"
            )
        extracted = json.loads(path.read_text(encoding="utf-8"))
        relevant = labels[key]
        jobs.append(
            Job(
                key=key,
                for_personas=["real-portfolio"] if relevant else [],
                covers=["software"],
                text=posting_text(row) if row else "",
                expected={
                    "title": extracted.get("title", ""),
                    "company": extracted.get("company", ""),
                    "summary": extracted.get("role_summary", ""),
                    "skills": [s["skill"] for s in extracted.get("skills_required", [])],
                },
                relevant_skills=relevant,
            )
        )
    return jobs
