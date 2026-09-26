from pathlib import Path

import pytest
from fastapi import HTTPException

import app.core.db as db_module
from app.core.db import Account, Repository, SkillEvidence, get_db, init_db
from app.core.settings import get_settings
from app.profile.evidence import add_evidence, delete_evidence, update_evidence


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _make_repo(tmp_path: Path) -> int:
    _reset_db(tmp_path)
    db = get_db()
    account = Account(first_name="Ada", last_name="Lovelace", github_username="octocat")
    db.add(account)
    db.commit()
    db.refresh(account)
    repo = Repository(
        account_id=account.id, github_id=1, name="proj", full_name="octocat/proj", url=""
    )
    db.add(repo)
    db.commit()
    db.refresh(repo)
    repo_id = repo.id
    db.close()
    return repo_id


def test_add_evidence_strips_and_writes(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    row = add_evidence(
        db, SkillEvidence, "repo_id", repo_id,
        skill="  Python  ", evidence_type="manual", weight=1.0, confidence=1.0,
    )
    assert row.skill == "Python"
    assert row.repo_id == repo_id
    db.close()


def test_add_evidence_rejects_blank_skill(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    with pytest.raises(HTTPException) as exc:
        add_evidence(
            db, SkillEvidence, "repo_id", repo_id,
            skill="   ", evidence_type="manual", weight=1.0, confidence=1.0,
        )
    assert exc.value.status_code == 422
    db.close()


def test_update_evidence_wrong_parent_404s(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    row = add_evidence(
        db, SkillEvidence, "repo_id", repo_id,
        skill="Python", evidence_type="manual", weight=1.0, confidence=1.0,
    )
    with pytest.raises(HTTPException) as exc:
        update_evidence(db, SkillEvidence, "repo_id", repo_id + 999, row.id, {"skill": "Rust"})
    assert exc.value.status_code == 404
    db.close()


def test_update_evidence_rejects_blank_skill(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    row = add_evidence(
        db, SkillEvidence, "repo_id", repo_id,
        skill="Python", evidence_type="manual", weight=1.0, confidence=1.0,
    )
    with pytest.raises(HTTPException) as exc:
        update_evidence(db, SkillEvidence, "repo_id", repo_id, row.id, {"skill": "  "})
    assert exc.value.status_code == 422
    db.close()


def test_delete_evidence_removes_row(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    row = add_evidence(
        db, SkillEvidence, "repo_id", repo_id,
        skill="Python", evidence_type="manual", weight=1.0, confidence=1.0,
    )
    evidence_id = row.id
    delete_evidence(db, SkillEvidence, "repo_id", repo_id, evidence_id)
    assert db.get(SkillEvidence, evidence_id) is None
    db.close()


def test_delete_evidence_missing_404s(tmp_path):
    repo_id = _make_repo(tmp_path)
    db = get_db()
    with pytest.raises(HTTPException) as exc:
        delete_evidence(db, SkillEvidence, "repo_id", repo_id, 99999)
    assert exc.value.status_code == 404
    db.close()
