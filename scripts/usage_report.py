"""The numbers a running instance has earned: users, resumes built, what
the judge did, and what a resume costs. Point DATABASE_URL at the instance's
database. Average cost is every build-related model call divided by resumes
finished, so an abandoned build's spend counts against the ones that landed.

Usage:
  .venv/bin/python -m scripts.usage_report
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from app.core.db import Account, LLMCall, UsageEvent, get_db
from app.core.usage import RESUME_BUILT

_BUILD_PURPOSES = (
    "resume_build", "resume_revise", "resume_edit", "groundedness_eval",
    "pagefit_rewrite", "pagefit_trim",
)


def report() -> dict[str, Any]:
    db = get_db()
    try:
        built = list(
            db.execute(select(UsageEvent).where(UsageEvent.kind == RESUME_BUILT)).scalars()
        )
        spend = db.execute(
            select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                LLMCall.purpose.in_(_BUILD_PURPOSES)
            )
        ).scalar_one()
        accounts = db.execute(select(func.count(Account.id))).scalar_one()
    finally:
        db.close()
    judged = [e.data_json or {} for e in built]
    return {
        "accounts": accounts,
        "users_with_a_resume": len({e.account_id for e in built}),
        "resumes_built": len(built),
        "bullets_rejected": sum(j.get("rejected", 0) for j in judged),
        "bullets_saved_by_revision": sum(j.get("saved", 0) for j in judged),
        "bullets_dropped": sum(j.get("dropped", 0) for j in judged),
        "build_spend_usd": round(float(spend), 4),
        "avg_cost_per_resume_usd": round(float(spend) / len(built), 4) if built else None,
    }


def main() -> None:
    for name, value in report().items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
