import datetime as dt
from pathlib import Path

from sqlalchemy import select

import app.core.db as db_module
from app.core.db import (
    Account,
    Experience,
    ExperienceSkillEvidence,
    Repository,
    Skill,
    SkillEvidence,
    get_db,
    init_db,
)
from app.core.settings import get_settings
from app.profile.resume_extract import ExperienceClaim, ResumeExtraction
from app.profile.resume_profile_merge import merge_resume_into_profile


def _reset_db(tmp_path: Path):
    import os

    db_module._engine = None
    db_module._SessionLocal = None
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def _make_account(**overrides) -> int:
    db = get_db()
    defaults = dict(first_name="Ada", last_name="Lovelace", github_username="octocat")
    defaults.update(overrides)
    account = Account(**defaults)
    db.add(account)
    db.commit()
    db.refresh(account)
    account_id = account.id
    db.close()
    return account_id


def _extraction(tags=None, experiences=None) -> ResumeExtraction:
    return ResumeExtraction(
        tags=tags or [],
        target_roles=[],
        summary="",
        experiences=experiences or [],
    )


def test_new_skill_gets_added(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    summary = merge_resume_into_profile(db, account_id, _extraction(tags=["Rust", "gRPC"]))

    assert summary.skills_added == 2
    rows = db.execute(select(Skill).where(Skill.account_id == account_id)).scalars()
    assert {s.name for s in rows} == {"Rust", "gRPC"}
    db.close()


def test_skill_already_manual_is_not_duplicated(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(Skill(account_id=account_id, name="python"))  # different casing on purpose
    db.commit()

    summary = merge_resume_into_profile(db, account_id, _extraction(tags=["Python", "Go"]))

    assert summary.skills_added == 1  # only "Go" is new, "Python" already there (casefold)
    rows = db.execute(select(Skill).where(Skill.account_id == account_id)).scalars().all()
    assert len(rows) == 2
    db.close()


def test_skill_already_backed_by_project_evidence_is_not_duplicated(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    repo = Repository(account_id=account_id, github_id=1, name="a", full_name="o/a", url="u")
    db.add(repo)
    db.commit()
    db.refresh(repo)
    db.add(
        SkillEvidence(
            skill="Kubernetes", repo_id=repo.id, evidence_type="readme_described",
            weight=0.5, confidence=0.5,
        )
    )
    db.commit()

    summary = merge_resume_into_profile(db, account_id, _extraction(tags=["kubernetes"]))

    assert summary.skills_added == 0  # already covered by project evidence
    assert db.execute(select(Skill).where(Skill.account_id == account_id)).scalars().all() == []
    db.close()


def test_skill_already_backed_by_experience_evidence_is_not_duplicated(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    exp = Experience(account_id=account_id, title="Eng", company="Acme")
    db.add(exp)
    db.commit()
    db.refresh(exp)
    db.add(
        ExperienceSkillEvidence(skill="Terraform", experience_id=exp.id, evidence_type="manual")
    )
    db.commit()

    summary = merge_resume_into_profile(db, account_id, _extraction(tags=["Terraform"]))

    assert summary.skills_added == 0
    db.close()


def test_new_experience_gets_added(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    claim = ExperienceClaim(
        company="Acme Corp",
        title="Software Engineer",
        start_date=dt.date(2020, 1, 1),
        end_date=None,
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 1
    assert summary.experiences_enriched == 0
    rows = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars().all()
    assert len(rows) == 1
    assert rows[0].company == "Acme Corp"
    assert rows[0].title == "Software Engineer"
    assert rows[0].start_date == dt.date(2020, 1, 1)
    assert rows[0].end_date is None
    db.close()


def test_matching_company_and_title_is_not_duplicated(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(Experience(account_id=account_id, title="engineer", company="acme corp"))
    db.commit()

    claim = ExperienceClaim(
        company="Acme Corp",  # different casing, same company
        title="Engineer",  # different casing, same title
        start_date=dt.date(2020, 1, 1),
        end_date=dt.date(2021, 1, 1),
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 0
    rows = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars().all()
    assert len(rows) == 1
    db.close()


def test_matching_experience_gets_dates_enriched_only_when_missing(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    existing = Experience(
        account_id=account_id, title="Engineer", company="Acme", start_date=None, end_date=None
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)
    existing_id = existing.id

    claim = ExperienceClaim(
        company="Acme",
        title="Engineer",
        start_date=dt.date(2019, 6, 1),
        end_date=dt.date(2022, 3, 1),
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 0
    assert summary.experiences_enriched == 1
    refreshed = db.get(Experience, existing_id)
    assert refreshed.start_date == dt.date(2019, 6, 1)
    assert refreshed.end_date == dt.date(2022, 3, 1)
    db.close()


def test_matching_experience_never_overwrites_an_existing_date(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    existing = Experience(
        account_id=account_id,
        title="Engineer",
        company="Acme",
        start_date=dt.date(2018, 1, 1),  # already set, by hand or an earlier resume
        end_date=None,
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)
    existing_id = existing.id

    claim = ExperienceClaim(
        company="Acme",
        title="Engineer",
        start_date=dt.date(2020, 1, 1),
        end_date=dt.date(2021, 1, 1),
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_enriched == 1  # end_date was empty, got filled
    refreshed = db.get(Experience, existing_id)
    assert refreshed.start_date == dt.date(2018, 1, 1)  # untouched
    assert refreshed.end_date == dt.date(2021, 1, 1)  # filled in
    db.close()


def test_same_company_different_title_is_a_second_row(tmp_path):
    """A promotion at the same company: different title, legitimately a
    second Experience row, not merged into the first."""
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(Experience(account_id=account_id, title="Engineer", company="Acme"))
    db.commit()

    claim = ExperienceClaim(
        company="Acme", title="Senior Engineer", start_date=None, end_date=None
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 1
    rows = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars().all()
    assert len(rows) == 2
    db.close()


def test_duplicate_claims_within_one_resume_only_add_once(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    claim = ExperienceClaim(company="Acme", title="Engineer", start_date=None, end_date=None)
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim, claim]))

    assert summary.experiences_added == 1
    rows = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars().all()
    assert len(rows) == 1
    db.close()
