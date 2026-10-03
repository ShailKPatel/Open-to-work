"""Runs the LLM skill review (app/profile/skill_review.py) over skills
already in the database, for every account, and deletes the evidence rows
for names it rejects. Extraction already reviews new names as it goes; this
covers skills stored before a run that did not review them.

Only names without a verdict are sent, in batches of 50, so running it again
costs nothing unless new names appeared. Hand-added skills are never touched.

Usage: .venv/bin/python -m scripts.review_skills
"""

from __future__ import annotations

from sqlalchemy import select

from app.core.db import Account, get_db, init_db
from app.profile.build import review_skill_evidence


def main() -> None:
    init_db()
    db = get_db()
    try:
        account_ids = list(db.execute(select(Account.id)).scalars())
    finally:
        db.close()
    removed = review_skill_evidence(account_ids)
    print(f"removed {removed} skill row(s) rejected by review")


if __name__ == "__main__":
    main()
