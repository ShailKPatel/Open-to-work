"""Interactive golden-set labeling: walks a person through building the
hand-labeled pairs app/evals/run.py measures retrieval quality against
(target: 50 hand-labeled job-to-profile pairs).

Judging "is this piece of evidence relevant to this job posting" needs a
human. This script makes that labeling fast: it runs your account's own real search against your
own real query, shows you the actual candidate results, and asks you to
mark each one relevant or not, using an unlabeled query id you choose
yourself, so it round-trips through the exact same
app/retrieval/search.py functions the eval harness itself measures.

Usage: .venv/bin/python -m scripts.label_golden_set --account 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import select

from app.core.db import JobPosting, get_db, init_db
from app.core.settings import get_settings
from app.evals.golden import GoldenPair, load_golden_set, save_golden_set, upsert_pair
from app.retrieval.index import job_posting_text
from app.retrieval.search import search_experience_points, search_skill_evidence

_COLLECTIONS = ("skill_evidence", "experience_points")


def _choose(prompt: str, options: tuple[str, ...]) -> str | None:
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {option}")
    raw = input("> ").strip()
    if not raw:
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    return raw if raw in options else None


def _pick_query_text(account_id: int) -> tuple[str, int | None]:
    """Returns (query_text, job_posting_id or None)."""
    mode = _choose(
        "Query source:", ("paste text directly", "use an existing job posting from /jobs")
    )
    if mode == "use an existing job posting from /jobs":
        db = get_db()
        try:
            postings = list(
                db.execute(
                    select(JobPosting).where(JobPosting.account_id == account_id)
                ).scalars()
            )
        finally:
            db.close()
        if not postings:
            print("No job postings saved for this account yet; falling back to pasted text.")
        else:
            print("Job postings on file:")
            for row in postings:
                print(f"  {row.id}: {row.title} @ {row.company}")
            raw_id = input("Job posting id to use as the query: ").strip()
            if raw_id.isdigit():
                db = get_db()
                try:
                    posting = db.get(JobPosting, int(raw_id))
                finally:
                    db.close()
                if posting is not None:
                    return job_posting_text(posting), posting.id
            print("Not a valid id; falling back to pasted text.")

    print("Paste the query text (job title/skills/summary), then an empty line to finish:")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines), None


def _run_search(collection: str, query_text: str, account_id: int) -> list[tuple[int, str, float]]:
    """(id, display_text, score) for the real top-10 results this
    account's own search would actually return today."""
    if collection == "skill_evidence":
        hits = search_skill_evidence(query_text, account_id, top_k=10)
        return [
            (h.id, f"{h.payload.get('skill')} ({h.payload.get('evidence_type')})", h.score)
            for h in hits
        ]
    hits = search_experience_points(query_text, account_id, top_k=10)
    return [(h.id, str(h.payload.get("text", ""))[:80], h.score) for h in hits]


def label_one(account_id: int, golden_path: Path) -> None:
    collection = _choose("Which collection is this query about?", _COLLECTIONS)
    if collection is None:
        print("Not a valid choice, skipping.")
        return

    query_text, job_posting_id = _pick_query_text(account_id)
    if not query_text.strip():
        print("Empty query, skipping.")
        return

    candidates = _run_search(collection, query_text, account_id)
    if not candidates:
        print(f"Nothing indexed yet in {collection!r} for this account; nothing to label.")
        return

    print(f"\n{len(candidates)} candidate result(s):")
    relevant_ids: list[int] = []
    for candidate_id, display, score in candidates:
        prompt = f"  [{score:.3f}] {display}  relevant? (y/N/q to stop labeling this query) "
        answer = input(prompt)
        if answer.strip().lower() == "q":
            break
        if answer.strip().lower() == "y":
            relevant_ids.append(candidate_id)

    if not relevant_ids:
        print("No relevant results marked; this pair won't help the eval, not saved.")
        return

    pairs = load_golden_set(golden_path)
    existing_ids = {
        p.id for p in pairs if p.account_id == account_id and p.collection == collection
    }
    n = 1
    while f"{account_id}-{collection}-{n}" in existing_ids:
        n += 1
    pair_id = f"{account_id}-{collection}-{n}"
    notes = input("Notes for this pair (optional): ").strip()

    pair = GoldenPair(
        id=pair_id, account_id=account_id, collection=collection, query_text=query_text,
        relevant_ids=relevant_ids, job_posting_id=job_posting_id, notes=notes,
    )
    pairs = upsert_pair(pairs, pair)
    save_golden_set(pairs, golden_path)

    total_for_account = sum(1 for p in pairs if p.account_id == account_id)
    print(f"Saved as {pair_id!r} ({len(relevant_ids)} relevant of {len(candidates)} shown).")
    print(
        f"Total labeled pairs for this account: {total_for_account} "
        "(target is 50)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively build the eval harness's golden set."
    )
    parser.add_argument("--account", type=int, required=True, help="account id to label pairs for")
    args = parser.parse_args()

    init_db()
    golden_path = Path(get_settings().evals_golden_dir) / "golden_set.yaml"

    while True:
        label_one(args.account, golden_path)
        again = input("\nLabel another query? (Y/n) ").strip().lower()
        if again == "n":
            break


if __name__ == "__main__":
    main()
