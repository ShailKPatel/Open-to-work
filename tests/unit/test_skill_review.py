import json
import os
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

import app.core.db as db_module
from app.core.db import Account, Repository, SkillEvidence, SkillVerdict, get_db, init_db
from app.core.llm import LLMRateLimitedError
from app.core.settings import get_settings
from app.profile.build import review_skill_evidence
from app.profile.skill_review import approve, rejected_keys, review_names


def _reset_db(tmp_path: Path):
    import app.retrieval.vectorstore as vectorstore_module

    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["QDRANT_URL"] = ":memory:"
    get_settings.cache_clear()
    init_db()


def _make_account() -> int:
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    account_id = account.id
    db.close()
    return account_id


def _fake_review(monkeypatch, remove=(), fail_with=None):
    """Patches the LLM call; records the names sent in each batch."""
    batches: list[list[str]] = []

    def fake_complete(tier, messages, schema=None, account_id=None, purpose=None):
        if fail_with is not None:
            raise fail_with
        # The names travel as a JSON array, so that a name containing
        # newlines or quotes stays one name (see _names_to_remove). Parsed
        # back the same way a model would read it, rather than by splitting
        # on newlines, which would make this fake disagree with the prompt.
        names = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
        batches.append(names)
        return SimpleNamespace(parsed={"remove": [n for n in names if n in remove]})

    monkeypatch.setattr("app.profile.skill_review.complete", fake_complete)
    return batches


def test_names_go_out_deduplicated_in_batches_of_fifty(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    batches = _fake_review(monkeypatch)
    names = [f"skill-{i}" for i in range(155)] + ["SKILL-0", " skill-1 "]

    db = get_db()
    review_names(db, account_id, names)
    db.close()

    assert [len(b) for b in batches] == [50, 50, 50, 5]


def test_only_returned_names_are_rejected_and_verdicts_are_remembered(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    batches = _fake_review(monkeypatch, remove={"blinker", "@types/uuid"})

    db = get_db()
    rejected = review_names(db, account_id, ["React", "blinker", "@types/uuid"])
    assert rejected == {"blinker", "@types/uuid"}

    # Second pass: nothing new to ask about, same answer from the stored verdicts.
    assert review_names(db, account_id, ["blinker", "React"]) == {"blinker"}
    assert len(batches) == 1
    db.close()


def test_model_cannot_reject_a_name_it_was_not_sent(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    monkeypatch.setattr(
        "app.profile.skill_review.complete",
        lambda *a, **k: SimpleNamespace(parsed={"remove": ["Python", "made-up"]}),
    )
    db = get_db()
    assert review_names(db, account_id, ["Python"]) == {"python"}
    assert rejected_keys(db, account_id) == {"python"}
    db.close()


def test_failed_review_leaves_names_unjudged_for_next_time(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_review(monkeypatch, fail_with=LLMRateLimitedError("quota"))

    db = get_db()
    assert review_names(db, account_id, ["blinker"]) == set()
    assert db.execute(select(SkillVerdict)).scalars().all() == []
    db.close()


def test_review_stops_asking_once_the_keys_are_gone(tmp_path, monkeypatch):
    """A spent key set reaches here as a provider error, not a 429, so
    is_out_of_keys() is what stops it. 155 names would otherwise cost
    four identical doomed calls instead of one."""
    from app.core.llm import LLMProviderError

    _reset_db(tmp_path)
    account_id = _make_account()
    calls: list = []

    def every_key_spent(tier, messages, schema=None, account_id=None, purpose=None):
        calls.append(messages)
        error = LLMProviderError("All 2 OpenAI keys failed on this request.")
        error.blames_key = True
        raise error

    monkeypatch.setattr("app.profile.skill_review.complete", every_key_spent)

    db = get_db()
    assert review_names(db, account_id, [f"skill-{i}" for i in range(155)]) == set()
    # Nothing judged, so the names come back round next time unchanged.
    assert db.execute(select(SkillVerdict)).scalars().all() == []
    db.close()
    assert len(calls) == 1


def test_manual_approval_overrules_rejection(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_review(monkeypatch, remove={"Pug"})

    db = get_db()
    review_names(db, account_id, ["Pug"])
    approve(db, account_id, "pug")
    assert rejected_keys(db, account_id) == set()
    row = db.execute(select(SkillVerdict)).scalar_one()
    assert (row.verdict, row.decided_by) == ("approved", "user")
    db.close()


def test_review_skill_evidence_deletes_rejected_rows_but_not_manual_ones(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_review(monkeypatch, remove={"blinker"})
    db = get_db()
    repo = Repository(account_id=account_id, github_id=1, name="p", full_name="o/p", url="")
    db.add(repo)
    db.commit()
    for skill, evidence_type in [
        ("blinker", "declared_dependency"),
        ("Flask", "declared_dependency"),
        ("blinker", "manual"),
    ]:
        db.add(
            SkillEvidence(
                skill=skill, repo_id=repo.id, evidence_type=evidence_type,
                weight=0.5, confidence=1.0,
            )
        )
    db.commit()
    db.close()

    assert review_skill_evidence([account_id, None]) == 1

    db = get_db()
    left = {(e.skill, e.evidence_type) for e in db.execute(select(SkillEvidence)).scalars()}
    db.close()
    assert left == {("Flask", "declared_dependency"), ("blinker", "manual")}


def test_adding_a_rejected_skill_by_hand_approves_it(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.main import app

    _reset_db(tmp_path)
    account_id = _make_account()
    _fake_review(monkeypatch, remove={"Pug"})
    db = get_db()
    review_names(db, account_id, ["Pug"])
    db.close()

    client = TestClient(app)
    assert client.get(f"/api/skills/rejected?account_id={account_id}").json() == [
        {"name": "Pug", "decided_by": "llm"}
    ]
    client.post("/api/skills", json={"account_id": account_id, "name": "Pug"})
    assert client.get(f"/api/skills/rejected?account_id={account_id}").json() == []
