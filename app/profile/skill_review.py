"""LLM review of skill names found automatically (dependency files, README
extraction, resume tags), so junk like transitive packages, type stubs, or a
class inside a library doesn't land on the skills list.

Each account keeps a verdict per name (SkillVerdict in app/core/db/models.py):
"rejected" or "approved". Only names with no verdict yet go to the LLM,
deduplicated and split into batches of BATCH_SIZE, and the model is asked
for just the names to remove: a short list back is cheaper than a label for
every name. Anything not returned counts as approved. After that a name is
never sent again: rejected names are dropped on write without another call,
approved ones pass straight through.

A person always wins. Adding a skill by hand calls approve(), which flips a
"rejected" verdict to "approved" and marks it user-decided; deleting one from
the Skills page calls reject(), the same flip the other way.

Callers must commit their own pending writes before review_names(): each
LLM call records itself through its own session, and SQLite would deadlock
on a write lock the caller still holds (same constraint as
app/profile/build.py's _process_repo).
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.db import SkillVerdict
from app.core.llm import (
    BudgetExceededError,
    LLMRateLimitedError,
    complete,
    is_out_of_keys,
    system_message,
    user_message,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 50

_SCHEMA = {
    "type": "object",
    "properties": {"remove": {"type": "array", "items": {"type": "string"}}},
    "required": ["remove"],
}

_SYSTEM_PROMPT = (
    "You review skill names pulled automatically from a developer's software "
    "projects (dependency files, READMEs) and resume, before they appear on "
    "the skills list of their resume. Return the names that do not belong "
    "there: helper or transitive packages nobody would list as a skill (e.g. "
    "'certifi', 'blinker', 'six'), type stubs, fonts, and build plugins (e.g. "
    "'@types/uuid'), individual classes, functions, or estimators inside a "
    "library (e.g. 'ElasticNetCV', 'RobustScaler'), code editors, words too "
    "generic to mean a skill, and anything garbled or meaningless. Keep real "
    "programming languages, frameworks, libraries, databases, platforms, "
    "tools, and engineering practices, including niche ones. When unsure, "
    "keep it. Copy each returned name exactly as it was given. If every name "
    "belongs, return an empty list."
)


def name_key(name: str) -> str:
    """Same key GET /api/skills groups by, with inner whitespace collapsed."""
    return " ".join(name.split()).casefold()


def rejected_keys(db: Session, account_id: int | None) -> set[str]:
    if account_id is None:
        return set()
    return set(
        db.execute(
            select(SkillVerdict.name_key).where(
                SkillVerdict.account_id == account_id, SkillVerdict.verdict == "rejected"
            )
        ).scalars()
    )


def approve(db: Session, account_id: int, name: str) -> None:
    """Records a person's own "this is a skill". Commits."""
    _decide(db, account_id, name, "approved")


def reject(db: Session, account_id: int, name: str) -> None:
    """Records a person's own "this is not a skill", so a later sync or
    resume extraction does not bring back a skill they deleted. Commits.
    """
    _decide(db, account_id, name, "rejected")


def _decide(db: Session, account_id: int, name: str, verdict: str) -> None:
    key = name_key(name)
    if not key:
        return
    row = db.execute(
        select(SkillVerdict).where(
            SkillVerdict.account_id == account_id, SkillVerdict.name_key == key
        )
    ).scalar_one_or_none()
    if row is None:
        db.add(
            SkillVerdict(
                account_id=account_id, name_key=key, name=name.strip(),
                verdict=verdict, decided_by="user",
            )
        )
    else:
        row.verdict = verdict
        row.decided_by = "user"
        row.decided_at = dt.datetime.now(dt.UTC)
    db.commit()


def _names_to_remove(names: list[str], account_id: int) -> set[str]:
    listing = "\n".join(names)
    messages = [
        system_message(_SYSTEM_PROMPT),
        user_message(f"Skill names, one per line:\n{listing}"),
    ]
    response = complete(
        "bulk", messages, schema=_SCHEMA, account_id=account_id, purpose="skill_review"
    )
    if response.parsed is None:
        raise ValueError("skill review returned no parseable JSON")
    asked = {name_key(n) for n in names}
    # Only names we actually sent count; the model can't invent a removal.
    return {name_key(str(n)) for n in response.parsed.get("remove", [])} & asked


def review_names(db: Session, account_id: int, names: list[str]) -> set[str]:
    """Judges every name in `names` that has no verdict yet, stores the
    verdicts, and returns the keys of all rejected names among `names`
    (new and previously rejected alike).

    Never raises. A batch that fails is left without verdicts so a later
    run retries it; a rate limit or budget cap stops the remaining batches
    for the same reason build.py stops its batch.
    """
    unique: dict[str, str] = {}
    for raw in names:
        key = name_key(raw)
        if key and key not in unique:
            unique[key] = " ".join(raw.split())
    if not unique:
        return set()

    known = {
        row.name_key: row.verdict
        for row in db.execute(
            select(SkillVerdict).where(
                SkillVerdict.account_id == account_id,
                SkillVerdict.name_key.in_(list(unique)),
            )
        ).scalars()
    }
    pending = [name for key, name in unique.items() if key not in known]
    db.commit()  # release any read transaction before the nested LLM-call session

    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        try:
            removed = _names_to_remove(batch, account_id)
        except (BudgetExceededError, LLMRateLimitedError) as e:
            logger.warning("skill review stopped for account %s: %s", account_id, e)
            break
        except Exception as e:
            if is_out_of_keys(e):
                logger.warning(
                    "skill review stopped for account %s, no key left: %s", account_id, e
                )
                break
            logger.exception("skill review batch failed for account %s", account_id)
            continue
        for name in batch:
            key = name_key(name)
            verdict = "rejected" if key in removed else "approved"
            db.add(
                SkillVerdict(
                    account_id=account_id, name_key=key, name=name,
                    verdict=verdict, decided_by="llm",
                )
            )
            known[key] = verdict
        db.commit()

    return {key for key, verdict in known.items() if verdict == "rejected"}
