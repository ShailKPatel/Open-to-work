from pathlib import Path

from sqlalchemy import select

import app.core.db as db_module
from app.core.db import (
    Account,
    ContactEmail,
    ContactPhone,
    Education,
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    Repository,
    ResumeProfileLink,
    Skill,
    SkillEvidence,
    SocialLink,
    get_db,
    init_db,
)
from app.core.settings import get_settings
from app.profile.resume_extract import (
    ContactClaim,
    EducationClaim,
    ExperienceClaim,
    LinkClaim,
    ResumeExtraction,
)
from app.profile.resume_profile_merge import merge_resume_into_profile


def _reset_db(tmp_path: Path):
    import os

    db_module.reset_engine()
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


def _extraction(tags=None, experiences=None, education=None, contact=None) -> ResumeExtraction:
    return ResumeExtraction(
        tags=tags or [],
        target_roles=[],
        summary="",
        experiences=experiences or [],
        education=education or [],
        contact=contact or ContactClaim(),
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


def test_new_skill_records_the_resume_it_came_from(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    merge_resume_into_profile(db, account_id, _extraction(tags=["Rust"]), resume_id=7)

    row = db.execute(select(Skill).where(Skill.account_id == account_id)).scalar_one()
    assert row.source_resume_id == 7
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
        start_date="jan 2020",
        end_date=None,
        location="Berlin, Germany",
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 1
    assert summary.experiences_enriched == 0
    rows = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars().all()
    assert len(rows) == 1
    assert rows[0].company == "Acme Corp"
    assert rows[0].title == "Software Engineer"
    assert rows[0].location == "Berlin, Germany"
    assert rows[0].start_date == "jan 2020"
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
        start_date="jan 2020",
        end_date="jan 2021",
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
        start_date="jun 2019",
        end_date="mar 2022",
        location="Remote",
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 0
    assert summary.experiences_enriched == 1
    refreshed = db.get(Experience, existing_id)
    assert refreshed.location == "Remote"
    assert refreshed.start_date == "jun 2019"
    assert refreshed.end_date == "mar 2022"
    db.close()


def test_matching_experience_never_overwrites_an_existing_date(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    existing = Experience(
        account_id=account_id,
        title="Engineer",
        company="Acme",
        start_date="jan 2018",  # already set, by hand or an earlier resume
        end_date=None,
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)
    existing_id = existing.id

    claim = ExperienceClaim(
        company="Acme",
        title="Engineer",
        start_date="jan 2020",
        end_date="jan 2021",
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_enriched == 1  # end_date was empty, got filled
    refreshed = db.get(Experience, existing_id)
    assert refreshed.start_date == "jan 2018"  # untouched
    assert refreshed.end_date == "jan 2021"  # filled in
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


def test_experience_points_added_and_deduplicated(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()

    claim1 = ExperienceClaim(
        company="Acme Corp",
        title="Software Engineer",
        start_date=None,
        end_date=None,
        points=["Built REST APIs", "Wrote unit tests"],
    )
    summary1 = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim1]))

    assert summary1.experiences_added == 1
    assert summary1.experience_points_added == 2

    # Second pass with same role: one duplicate point, one new point ("Led code reviews")
    claim2 = ExperienceClaim(
        company="Acme Corp",
        title="Software Engineer",
        start_date=None,
        end_date=None,
        points=["built rest apis", "Led code reviews"],
    )
    summary2 = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim2]))

    assert summary2.experiences_added == 0
    assert summary2.experience_points_added == 1

    exp = db.execute(select(Experience).where(Experience.account_id == account_id)).scalar_one()
    points = list(
        db.execute(
            select(ExperiencePoint)
            .where(ExperiencePoint.experience_id == exp.id)
            .order_by(ExperiencePoint.order_index)
        ).scalars()
    )

    assert [p.text for p in points] == ["Built REST APIs", "Wrote unit tests", "Led code reviews"]
    assert [p.order_index for p in points] == [1, 2, 3]
    db.close()


def test_experience_skills_become_evidence_and_deduplicate(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()

    claim = ExperienceClaim(
        company="Acme Corp",
        title="Software Engineer",
        start_date=None,
        end_date=None,
        skills=["Python", "Docker", "python"],
    )
    merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))
    claim.skills = ["docker", "Kubernetes"]
    merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    rows = list(db.execute(select(ExperienceSkillEvidence)).scalars())
    assert sorted(r.skill for r in rows) == ["Docker", "Kubernetes", "Python"]
    assert {r.evidence_type for r in rows} == {"resume"}
    db.close()


def test_skill_tied_to_a_role_is_not_also_freestanding(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()

    claim = ExperienceClaim(
        company="Acme Corp", title="Engineer", start_date=None, end_date=None,
        skills=["Python"],
    )
    summary = merge_resume_into_profile(
        db, account_id, _extraction(tags=["Python", "Rust"], experiences=[claim]), resume_id=1
    )

    assert summary.skills_added == 1
    assert [s.name for s in db.execute(select(Skill)).scalars()] == ["Rust"]
    assert [e.skill for e in db.execute(select(ExperienceSkillEvidence)).scalars()] == ["Python"]
    db.close()


def test_resume_only_skill_is_retired_once_a_role_backs_it(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()
    db.add(Skill(account_id=account_id, name="Docker"))  # typed in by hand
    db.commit()

    merge_resume_into_profile(db, account_id, _extraction(tags=["Python"]), resume_id=1)
    claim = ExperienceClaim(
        company="Acme Corp", title="Engineer", start_date=None, end_date=None,
        skills=["python", "Docker"],
    )
    merge_resume_into_profile(
        db, account_id, _extraction(tags=["python", "Docker"], experiences=[claim]), resume_id=1
    )

    assert [s.name for s in db.execute(select(Skill)).scalars()] == ["Docker"]
    assert sorted(e.skill for e in db.execute(select(ExperienceSkillEvidence)).scalars()) == [
        "Docker", "python",
    ]
    kinds = {link.kind for link in db.execute(select(ResumeProfileLink)).scalars()}
    assert kinds == {"experience", "experience_skill", "skill"}  # skill: the manual Docker
    db.close()


def test_links_record_what_each_resume_created_or_matched(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()
    extraction = _extraction(
        tags=["Rust"],
        experiences=[
            ExperienceClaim(
                company="Acme Corp", title="Engineer", start_date=None, end_date=None,
                points=["Shipped X"], skills=["Python"],
            )
        ],
        education=[
            EducationClaim(
                institution="MIT", degree="BS", location=None, start_date=None, end_date=None
            )
        ],
    )

    merge_resume_into_profile(db, account_id, extraction, resume_id=1)
    merge_resume_into_profile(db, account_id, extraction, resume_id=1)  # reprocess
    merge_resume_into_profile(db, account_id, extraction, resume_id=2)

    def links(resume_id):
        return {
            (link.kind, link.created)
            for link in db.execute(
                select(ResumeProfileLink).where(ResumeProfileLink.resume_id == resume_id)
            ).scalars()
        }

    kinds = ["experience", "experience_point", "experience_skill", "education", "skill"]
    assert links(1) == {(kind, True) for kind in kinds}
    assert links(2) == {(kind, False) for kind in kinds}
    assert len(list(db.execute(select(ResumeProfileLink)).scalars())) == 10
    db.close()


def test_new_education_gets_added(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    claim = EducationClaim(
        institution="Nirma University",
        degree="B.Tech in Computer Science",
        location="Springfield",
        start_date="aug 2022",
        end_date=None,
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(education=[claim, claim]))

    assert summary.education_added == 1  # in-pass duplicate counted once
    rows = db.execute(select(Education).where(Education.account_id == account_id)).scalars().all()
    assert len(rows) == 1
    assert rows[0].institution == "Nirma University"
    assert rows[0].degree == "B.Tech in Computer Science"
    assert rows[0].location == "Springfield"
    assert rows[0].start_date == "aug 2022"
    assert rows[0].end_date is None
    db.close()


def test_matching_education_only_fills_missing_fields(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    existing = Education(
        account_id=account_id,
        institution="nirma university",
        degree="b.tech in computer science",
        location=None,
        start_date="jan 2021",  # set by hand, must survive
        end_date=None,
    )
    db.add(existing)
    db.commit()
    db.refresh(existing)
    existing_id = existing.id

    claim = EducationClaim(
        institution="Nirma University",
        degree="B.Tech in Computer Science",
        location="Springfield",
        start_date="aug 2022",
        end_date="may 2026",
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(education=[claim]))

    assert summary.education_added == 0
    assert summary.education_enriched == 1
    refreshed = db.get(Education, existing_id)
    assert refreshed.location == "Springfield"
    assert refreshed.start_date == "jan 2021"  # untouched
    assert refreshed.end_date == "may 2026"
    assert len(db.execute(select(Education)).scalars().all()) == 1
    db.close()


def test_education_grade_and_details_fill_only_when_empty(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    kept = Education(
        account_id=account_id, institution="MIT", degree="BS Physics",
        grade="GPA 4.0", details=["Typed by hand"],
    )
    bare = Education(account_id=account_id, institution="MIT", degree="MS Physics")
    db.add_all([kept, bare])
    db.commit()
    kept_id, bare_id = kept.id, bare.id

    claims = [
        EducationClaim(
            institution="MIT", degree=degree, location=None, start_date=None,
            end_date=None, grade="GPA 3.5", details=["From the resume"],
        )
        for degree in ("BS Physics", "MS Physics", "PhD Physics")
    ]
    merge_resume_into_profile(db, account_id, _extraction(education=claims))

    assert db.get(Education, kept_id).grade == "GPA 4.0"
    assert db.get(Education, kept_id).details == ["Typed by hand"]
    assert db.get(Education, bare_id).grade == "GPA 3.5"
    assert db.get(Education, bare_id).details == ["From the resume"]
    new = db.execute(select(Education).where(Education.degree == "PhD Physics")).scalar_one()
    assert new.grade == "GPA 3.5"
    assert new.details == ["From the resume"]
    db.close()


def test_same_institution_different_degree_is_a_second_row(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(Education(account_id=account_id, institution="MIT", degree="BS Physics"))
    db.commit()

    claim = EducationClaim(
        institution="MIT", degree="MS Physics", location=None, start_date=None, end_date=None
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(education=[claim]))

    assert summary.education_added == 1
    rows = db.execute(select(Education).where(Education.account_id == account_id)).scalars().all()
    assert len(rows) == 2
    db.close()


def test_contact_adds_everything_new_and_sets_primaries(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    contact = ContactClaim(
        name="Someone Else",
        location="London, UK",
        emails=["ada@example.com", "ada@work.example"],
        phones=["+44 20 7946 0958", "+1 (555) 123-4567"],
        links=[
            LinkClaim(platform="website", url="https://ada.dev"),
            LinkClaim(platform="website", url="https://blog.ada.dev"),
            LinkClaim(platform="other", url="https://leetcode.com/ada", label="LeetCode"),
        ],
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(contact=contact))

    assert (summary.emails_added, summary.phones_added, summary.links_added) == (2, 2, 3)
    assert summary.location_filled
    account = db.get(Account, account_id)
    assert account.contact_location == "London, UK"
    assert account.contact_email == "ada@example.com"
    assert account.contact_phone == "+44 20 7946 0958"
    assert (account.first_name, account.last_name) == ("Ada", "Lovelace")  # name never merged
    emails = db.execute(select(ContactEmail).order_by(ContactEmail.id)).scalars().all()
    assert [(e.email, e.is_primary) for e in emails] == [
        ("ada@example.com", True),
        ("ada@work.example", False),
    ]
    links = db.execute(select(SocialLink).order_by(SocialLink.id)).scalars().all()
    assert [(link.platform, link.label) for link in links] == [
        ("website", None),
        ("website", None),
        ("other", "LeetCode"),
    ]
    db.close()


def test_contact_already_known_is_ignored(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account(contact_location="Paris")
    db = get_db()
    db.add(ContactEmail(account_id=account_id, email="Ada@Example.com", is_primary=True))
    db.add(ContactPhone(account_id=account_id, phone="(555) 123-4567", is_primary=True))
    db.add(SocialLink(account_id=account_id, platform="linkedin", url="https://www.linkedin.com/in/ada/"))
    db.commit()

    contact = ContactClaim(
        location="London, UK",
        emails=["ada@example.com"],
        phones=["+1 555 123 4567"],
        links=[
            LinkClaim(platform="linkedin", url="http://linkedin.com/in/ada"),
            LinkClaim(platform="github", url="https://github.com/OctoCat"),  # sync identity
        ],
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(contact=contact))

    assert (summary.emails_added, summary.phones_added, summary.links_added) == (0, 0, 0)
    assert not summary.location_filled
    assert db.get(Account, account_id).contact_location == "Paris"
    assert len(db.execute(select(ContactEmail)).scalars().all()) == 1
    assert len(db.execute(select(ContactPhone)).scalars().all()) == 1
    assert len(db.execute(select(SocialLink)).scalars().all()) == 1
    db.close()


def test_new_contact_email_does_not_steal_primary(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account(contact_email="old@example.com")
    db = get_db()
    db.add(ContactEmail(account_id=account_id, email="old@example.com", is_primary=True))
    db.commit()

    contact = ContactClaim(emails=["new@example.com"])
    merge_resume_into_profile(db, account_id, _extraction(contact=contact))

    assert db.get(Account, account_id).contact_email == "old@example.com"
    new_row = db.execute(
        select(ContactEmail).where(ContactEmail.email == "new@example.com")
    ).scalar_one()
    assert new_row.is_primary is False
    db.close()


def test_contact_names_an_unnamed_saved_link(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()
    db.add(SocialLink(account_id=account_id, platform="website", url="https://ada.dev/"))
    db.add(
        SocialLink(account_id=account_id, platform="other", url="https://ada.blog", label="Notes")
    )
    db.commit()

    contact = ContactClaim(
        links=[
            LinkClaim(platform="other", url="https://ada.dev", label="Portfolio"),
            LinkClaim(platform="other", url="https://ada.blog", label="Blog"),
            LinkClaim(platform="other", url="https://credly.com/ada", label="Certificates"),
        ],
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(contact=contact))

    assert summary.links_added == 1
    links = db.execute(select(SocialLink).order_by(SocialLink.id)).scalars().all()
    assert [(link.platform, link.label) for link in links] == [
        ("other", "Portfolio"),  # unnamed site takes the resume's name
        ("other", "Notes"),  # a name set by hand is kept
        ("other", "Certificates"),
    ]
    db.close()


def test_role_matches_despite_company_descriptor_and_suffix(tmp_path, monkeypatch):
    """One extraction reads the bare company, the next adds a descriptor
    in parentheses or a legal suffix: still the same role, enriched, not
    a second row."""
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()

    first = ExperienceClaim(
        company="Northwind Labs",
        title="Founding Machine Learning Engineer",
        start_date="nov 2025",
        end_date=None,
    )
    merge_resume_into_profile(db, account_id, _extraction(experiences=[first]))

    again = [
        ExperienceClaim(
            company="Northwind Labs (Logistics Tech)",
            title="Founding Machine-Learning Engineer",
            start_date="nov 2025",
            end_date=None,
            location="Lisbon, Portugal (Remote)",
            points=["Built the forecasting service"],
        ),
        ExperienceClaim(
            company="northwind labs, Inc.",
            title="founding machine learning engineer",
            start_date=None,
            end_date=None,
        ),
    ]
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=again))

    assert summary.experiences_added == 0
    assert summary.experiences_enriched == 1
    (row,) = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars()
    assert row.company == "Northwind Labs"
    assert row.location == "Lisbon, Portugal (Remote)"
    points = db.execute(select(ExperiencePoint).where(ExperiencePoint.experience_id == row.id))
    assert [p.text for p in points.scalars()] == ["Built the forecasting service"]
    db.close()


def test_role_left_off_resumes_still_absorbs_a_resume_naming_it(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    monkeypatch.setattr(
        "app.retrieval.index.embed", lambda texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    account_id = _make_account()
    db = get_db()
    db.add(
        Experience(
            account_id=account_id, title="Engineer", company="Acme", exclude_from_resume=True
        )
    )
    db.commit()

    claim = ExperienceClaim(
        company="Acme", title="Engineer", start_date=None, end_date=None, points=["Did a thing"]
    )
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=[claim]))

    assert summary.experiences_added == 0
    assert summary.experience_points_added == 1
    (row,) = db.execute(select(Experience).where(Experience.account_id == account_id)).scalars()
    assert row.exclude_from_resume is True
    db.close()


def test_different_companies_sharing_a_prefix_stay_apart(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    claims = [
        ExperienceClaim(company="Meta", title="Engineer", start_date=None, end_date=None),
        ExperienceClaim(company="Metaflow", title="Engineer", start_date=None, end_date=None),
    ]
    summary = merge_resume_into_profile(db, account_id, _extraction(experiences=claims))

    assert summary.experiences_added == 2
    db.close()


def test_education_matches_despite_punctuation_and_descriptor(tmp_path):
    _reset_db(tmp_path)
    account_id = _make_account()
    db = get_db()

    merge_resume_into_profile(
        db,
        account_id,
        _extraction(
            education=[
                EducationClaim(
                    institution="Nirma University",
                    degree="B.Tech in Computer Science",
                    location=None,
                    start_date=None,
                    end_date=None,
                )
            ]
        ),
    )
    summary = merge_resume_into_profile(
        db,
        account_id,
        _extraction(
            education=[
                EducationClaim(
                    institution="Nirma University (Ahmedabad)",
                    degree="BTech in Computer Science",
                    location=None,
                    start_date="jul 2020",
                    end_date=None,
                )
            ]
        ),
    )

    assert summary.education_added == 0
    assert summary.education_enriched == 1
    rows = db.execute(select(Education).where(Education.account_id == account_id)).scalars()
    assert len(list(rows)) == 1
    db.close()
