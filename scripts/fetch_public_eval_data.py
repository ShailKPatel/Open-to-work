"""Downloads the public-dataset eval set listed in evals/public/sources.yaml
into evals/public/cache/ (gitignored), from the LinkedIn Job Postings
(2023-2024) dataset's CSV (about 500 MB, spooled to a temporary file and deleted after reading).
Description text is scrubbed of email addresses, phone numbers and named
contact lines before it is written (app/evals/real.py's scrub_pii).

--select draws the sample again and rewrites sources.yaml. It is how the
committed sample was made and is not part of a normal run: re-selecting
changes the test set, which makes earlier results incomparable.

Sampling is deterministic: within each group, the eligible postings with
the lowest SHA-256 of "<salt>:<job id>" are taken, one per company.
Eligible means a description of 600 to 8000 characters that is almost
all ASCII (an English-language proxy). Groups:
  software        titles matching app/evals/public.py's SOFTWARE_TITLE
  general-salary  any other title, with a salary on the form
  general         any other title, without one

--add-fresh N appends N software postings drawn under a new salt (group
software-fresh by default, or --group; all in test), for when the existing test set has been
spent: once a change is made after reading test results, those postings
are dev, and the claim needs postings nothing was tuned on.

Usage:
  .venv/bin/python -m scripts.fetch_public_eval_data [--csv PATH] [--select | --add-fresh N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.evals.public import CACHE_DIR, DATASET_URL, PUBLIC_DIR, SOFTWARE_TITLE, cache_path
from app.evals.real import scrub_pii

_SALT = "open-to-work-eval-2026"
_GROUP_SIZES = {"software": 80, "general-salary": 60, "general": 60}
_KEEP = (
    "job_id",
    "company_name",
    "title",
    "location",
    "description",
    "min_salary",
    "max_salary",
    "med_salary",
    "pay_period",
    "formatted_work_type",
    "remote_allowed",
    "formatted_experience_level",
    "skills_desc",
)

_SOURCES_HEADER = """\
# Pinned sample of the LinkedIn Job Postings (2023-2024) dataset, by
# Kaggle user arshkon, CC BY-SA 4.0: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
# (read from its Hugging Face mirror, datastax/linkedin_job_listings).
# Drawn by scripts/fetch_public_eval_data.py --select; see DATA_SOURCES.md.
# Do not redraw: the test split is defined by these ids.
"""


def _rows(csv_path: Path | None) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    if csv_path is not None:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
        return
    with tempfile.TemporaryFile("w+", encoding="utf-8", newline="") as spool:
        with httpx.stream("GET", DATASET_URL, follow_redirects=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                spool.write(chunk)
        spool.seek(0)
        yield from csv.DictReader(spool)


def _eligible(row: dict[str, Any]) -> bool:
    text = row.get("description") or ""
    if not 600 <= len(text) <= 8000 or not row.get("company_name"):
        return False
    ascii_share = sum(1 for ch in text if ord(ch) < 128) / len(text)
    return ascii_share > 0.98


def _group(row: dict[str, Any]) -> str:
    if SOFTWARE_TITLE.search(row.get("title") or ""):
        return "software"
    return "general-salary" if row.get("max_salary") or row.get("med_salary") else "general"


def _rank(job_id: str) -> str:
    return hashlib.sha256(f"{_SALT}:{job_id}".encode()).hexdigest()


def _job_id(row: dict[str, Any]) -> str:
    return str(int(float(row["job_id"])))


def _select(csv_path: Path | None) -> list[dict[str, Any]]:
    candidates: dict[str, list[tuple[str, dict[str, Any]]]] = {g: [] for g in _GROUP_SIZES}
    for row in _rows(csv_path):
        if _eligible(row):
            candidates[_group(row)].append((_rank(_job_id(row)), row))
    chosen = []
    for group, size in _GROUP_SIZES.items():
        companies: set[str] = set()
        for _, row in sorted(candidates[group], key=lambda pair: pair[0]):
            company = row["company_name"].strip().casefold()
            if company in companies:
                continue
            companies.add(company)
            chosen.append({"id": int(_job_id(row)), "group": group})
            if len(companies) == size:
                break
    return chosen


def _select_fresh(
    csv_path: Path | None,
    size: int,
    salt: str,
    taken: list[dict[str, Any]],
    group: str = "software-fresh",
) -> list[dict[str, Any]]:
    """`size` more software postings, ranked under a new salt, skipping
    every posting and company already in the sample. All go to test: they
    replace a test set that was spent by changing the system after it was
    read."""
    taken_ids = {int(item["id"]) for item in taken}
    taken_companies: set[str] = set()
    rows = []
    for row in _rows(csv_path):
        if int(_job_id(row)) in taken_ids:
            taken_companies.add((row.get("company_name") or "").strip().casefold())
        elif _eligible(row) and _group(row) == "software":
            rows.append(row)
    ranked = sorted(rows, key=lambda r: hashlib.sha256(f"{salt}:{_job_id(r)}".encode()).hexdigest())
    chosen: list[dict[str, Any]] = []
    for row in ranked:
        company = row["company_name"].strip().casefold()
        if company in taken_companies:
            continue
        taken_companies.add(company)
        chosen.append({"id": int(_job_id(row)), "group": group, "split": "test"})
        if len(chosen) == size:
            break
    return chosen


def _cache(csv_path: Path | None, ids: set[int]) -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in _rows(csv_path):
        job_id = int(_job_id(row))
        if job_id not in ids:
            continue
        kept = {name: row.get(name) or None for name in _KEEP}
        kept["job_id"] = job_id
        kept["description"] = scrub_pii(row.get("description") or "")
        if kept["skills_desc"]:
            kept["skills_desc"] = scrub_pii(kept["skills_desc"])
        cache_path(job_id).write_text(json.dumps(kept, indent=1), encoding="utf-8")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, help="a local copy of postings.csv")
    parser.add_argument("--select", action="store_true", help="redraw the sample")
    parser.add_argument(
        "--add-fresh", type=int, metavar="N",
        help="append N new software postings as a fresh test set",
    )
    parser.add_argument("--salt", default="open-to-work-eval-2026-fresh-1")
    parser.add_argument(
        "--group", default="software-fresh", help="group name for --add-fresh postings"
    )
    args = parser.parse_args()

    sources_path = PUBLIC_DIR / "sources.yaml"
    if args.select:
        postings = _select(args.csv)
        body = yaml.safe_dump({"postings": postings}, sort_keys=False)
        sources_path.write_text(_SOURCES_HEADER + "\n" + body, encoding="utf-8")
        print(f"selected {len(postings)} postings")
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    if args.add_fresh:
        fresh = _select_fresh(
            args.csv, args.add_fresh, args.salt, sources["postings"], args.group
        )
        sources["postings"].extend(fresh)
        body = yaml.safe_dump(sources, sort_keys=False)
        header = sources_path.read_text(encoding="utf-8").split("\npostings:", 1)[0]
        sources_path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")
        print(f"added {len(fresh)} fresh test postings (salt {args.salt!r})")
    ids = {int(item["id"]) for item in sources["postings"]}
    missing = {i for i in ids if not cache_path(i).exists()}
    if not missing:
        print(f"all {len(ids)} postings already cached in {CACHE_DIR}")
        return
    written = _cache(args.csv, missing)
    print(f"cached {written} of {len(missing)} missing postings in {CACHE_DIR}")


if __name__ == "__main__":
    main()
