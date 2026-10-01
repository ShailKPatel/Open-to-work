"""Folds a resume's LLM-extracted skills, work-history and education
entries into an account's actual profile data (the `skills`,
`experiences` and `education` tables), once extraction succeeds (see
app/profile/resume_ingest.py's run_extraction). Different dedup rules, matching how each table
already treats duplicates elsewhere in this codebase:

Skills: reuses the resume's own `tags` (already "concrete skills, tools,
technologies, and practices" per resume_extract.py's prompt, no separate
"skills" field needed). A tag whose casefolded name already exists
anywhere for this account (a manual `Skill` row, or evidence from a
project or experience, the exact same union GET /api/skills reads) is
left alone, no new row, no new evidence source; a name that's
new becomes a freestanding `Skill` row pointing back at the resume it
came from, so the Skills page can say where it was found, unless the
skill review (app/profile/skill_review.py) rejects it first. No fuzzy
matching, casefold-equal is equal, the same rule this app already uses
everywhere skills get deduplicated.

Experience: matched on (company, title), both casefolded and trimmed, NOT
on dates. Two resumes (or a resume and a hand-entered role) describing
"Engineer at Acme" are the same line item even if one states different
start/end dates or none at all; a promotion at the same company is a
different title, so a legitimately different, second row. A matching
existing row gets its location/start_date/end_date filled in from the resume ONLY
where that field was previously null, so a date entered by hand (or by an
earlier resume) never gets silently overwritten; a new
(company, title) pair becomes a new Experience row. Skills the resume
ties to a role become that row's experience evidence (evidence_type
"resume"), skipping names it already has or the skill review rejects.
Roles merge before the tags do, so a skill tied to a role is never also
added as freestanding; a freestanding skill an earlier resume added is
removed once a role's evidence carries the same name.

Education: same shape as Experience, matched on (institution, degree),
casefolded and trimmed, not on dates. A matching row only gets its
location/start_date/end_date filled where previously null; a new
(institution, degree) pair becomes a new Education row.

Every experience, bullet point, experience skill, education entry and
freestanding skill a resume creates or matches is recorded as a
ResumeProfileLink row, so what a resume contributed can be read back
later without re-extracting it.

Contact: every extracted email, phone and link that the account doesn't
already have is added; one it already has is skipped, never edited.
Emails match casefolded; phones match on their digits alone, ignoring a
country-code prefix present on only one side; links match on the URL
with scheme, "www.", query/fragment and trailing slash stripped,
casefolded, and a GitHub link to Account.github_username counts as
already present. The first email/phone an account ever gets becomes its
primary, same as adding one on the Links page. Location fills
Account.contact_location only when it's empty. The name is never
merged: first/last name are set at signup and only change by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

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
)
from app.profile.resume_extract import (
    ContactClaim,
    EducationClaim,
    ExperienceClaim,
    ResumeExtraction,
)
from app.profile.skill_review import name_key, review_names


@dataclass
class ProfileMergeSummary:
    skills_added: int
    experiences_added: int
    experiences_enriched: int
    experience_points_added: int = 0
    education_added: int = 0
    education_enriched: int = 0
    emails_added: int = 0
    phones_added: int = 0
    links_added: int = 0
    location_filled: bool = False


def _existing_skill_names(db: Session, account_id: int) -> set[str]:
    """Casefolded skill names this account already has, across every
    source GET /api/skills unions (app/api/skills.py's _load_groups):
    project evidence, experience evidence, and freestanding manual rows.
    """
    names: set[str] = set()
    for (skill,) in db.execute(
        select(SkillEvidence.skill)
        .join(Repository, SkillEvidence.repo_id == Repository.id)
        .where(Repository.account_id == account_id)
    ):
        names.add(skill.strip().casefold())
    for (skill,) in db.execute(
        select(ExperienceSkillEvidence.skill)
        .join(Experience, ExperienceSkillEvidence.experience_id == Experience.id)
        .where(Experience.account_id == account_id)
    ):
        names.add(skill.strip().casefold())
    for (name,) in db.execute(select(Skill.name).where(Skill.account_id == account_id)):
        names.add(name.strip().casefold())
    return names


class _LinkRecorder:
    """Writes the ResumeProfileLink rows for one merge: which profile rows
    this resume created or matched. A no-op when there's no resume id
    (a merge run outside any stored resume has nothing to link to).
    Callers flush before recording so every new row already has its id.
    """

    def __init__(self, db: Session, resume_id: int | None) -> None:
        self.db = db
        self.resume_id = resume_id
        self.seen: set[tuple[str, int]] = set()
        if resume_id is not None:
            self.seen = {
                (kind, ref_id)
                for kind, ref_id in db.execute(
                    select(ResumeProfileLink.kind, ResumeProfileLink.ref_id).where(
                        ResumeProfileLink.resume_id == resume_id
                    )
                )
            }

    def record(self, kind: str, ref_id: int, created: bool) -> None:
        if self.resume_id is None or (kind, ref_id) in self.seen:
            return
        self.db.add(
            ResumeProfileLink(resume_id=self.resume_id, kind=kind, ref_id=ref_id, created=created)
        )
        self.seen.add((kind, ref_id))


def unlink_profile_rows(db: Session, kind: str, ref_ids: list[int]) -> None:
    """Drops the ResumeProfileLink rows pointing at profile rows that are
    being deleted, so a later row reusing the same id is never mistaken
    for something a resume added. Leaves committing to the caller.
    """
    if ref_ids:
        db.execute(
            delete(ResumeProfileLink).where(
                ResumeProfileLink.kind == kind, ResumeProfileLink.ref_id.in_(ref_ids)
            )
        )


def _merge_skills(
    db: Session, account_id: int, tags: list[str], links: _LinkRecorder
) -> int:
    existing = _existing_skill_names(db, account_id)
    new_names = [t.strip() for t in tags if t.strip() and t.strip().casefold() not in existing]
    # Same LLM review as project skills, run before anything is written.
    rejected = review_names(db, account_id, new_names)
    added = 0
    for name in new_names:
        if name.casefold() in existing or name_key(name) in rejected:
            continue
        row = Skill(account_id=account_id, name=name, source_resume_id=links.resume_id)
        db.add(row)
        db.flush()
        links.record("skill", row.id, created=True)
        existing.add(name.casefold())  # this pass's own duplicates count once too
        added += 1

    # A tag that already sits as a freestanding skill still belongs to
    # this resume; one backed by evidence is linked through that evidence.
    tag_keys = {t.strip().casefold() for t in tags if t.strip()}
    for row in db.execute(select(Skill).where(Skill.account_id == account_id)).scalars():
        if row.name.strip().casefold() in tag_keys:
            links.record("skill", row.id, created=False)

    db.commit()
    return added


def _retire_resume_only_skills(db: Session, account_id: int, names: set[str]) -> None:
    """A skill a resume once added as freestanding (no role to tie it to
    yet) that now has experience evidence is the same skill twice; the
    evidence carries it from here on, so the freestanding row goes.
    Skills typed in by hand (no source resume) stay as they are.
    """
    if not names:
        return
    stale = [
        row
        for row in db.execute(
            select(Skill).where(
                Skill.account_id == account_id, Skill.source_resume_id.is_not(None)
            )
        ).scalars()
        if row.name.strip().casefold() in names
    ]
    unlink_profile_rows(db, "skill", [row.id for row in stale])
    for row in stale:
        db.delete(row)


def _merge_experiences(
    db: Session, account_id: int, claims: list[ExperienceClaim], links: _LinkRecorder
) -> tuple[int, int, int]:
    existing_rows = list(
        db.execute(select(Experience).where(Experience.account_id == account_id)).scalars()
    )
    by_key = {
        (row.company.strip().casefold(), row.title.strip().casefold()): row
        for row in existing_rows
    }

    added = 0
    enriched = 0
    points_added = 0
    new_points_to_index: list[ExperiencePoint] = []
    new_evidence_to_index: list[ExperienceSkillEvidence] = []
    evidenced_names: set[str] = set()
    # Same LLM review as the resume's top-level skills, run before anything is written.
    rejected_skills = review_names(
        db, account_id, [name for claim in claims for name in claim.skills]
    )

    for claim in claims:
        key = (claim.company.casefold(), claim.title.casefold())
        existing = by_key.get(key)
        target_row: Experience

        if existing is None:
            target_row = Experience(
                account_id=account_id,
                title=claim.title,
                company=claim.company,
                location=claim.location,
                start_date=claim.start_date,
                end_date=claim.end_date,
            )
            db.add(target_row)
            db.flush()  # this pass's own duplicate claims should match, not double-add
            by_key[key] = target_row
            added += 1
            links.record("experience", target_row.id, created=True)
        else:
            target_row = existing
            changed = False
            for attr in ("location", "start_date", "end_date"):
                value = getattr(claim, attr)
                if getattr(target_row, attr) is None and value is not None:
                    setattr(target_row, attr, value)
                    changed = True
            if changed:
                enriched += 1
            links.record("experience", target_row.id, created=False)

        if claim.points:
            existing_points = {
                p.text.strip().casefold(): p
                for p in db.execute(
                    select(ExperiencePoint).where(
                        ExperiencePoint.experience_id == target_row.id
                    )
                ).scalars()
            }
            current_max_order = max(
                (p.order_index for p in existing_points.values()), default=0
            )

            for p_text in claim.points:
                clean_text = p_text.strip()
                if not clean_text:
                    continue
                matched = existing_points.get(clean_text.casefold())
                if matched is not None:
                    links.record("experience_point", matched.id, created=False)
                    continue
                current_max_order += 1
                new_point = ExperiencePoint(
                    experience_id=target_row.id,
                    text=clean_text,
                    order_index=current_max_order,
                )
                db.add(new_point)
                db.flush()
                existing_points[clean_text.casefold()] = new_point
                links.record("experience_point", new_point.id, created=True)
                new_points_to_index.append(new_point)
                points_added += 1

        if claim.skills:
            known_skills = {
                row.skill.strip().casefold(): row
                for row in db.execute(
                    select(ExperienceSkillEvidence).where(
                        ExperienceSkillEvidence.experience_id == target_row.id
                    )
                ).scalars()
            }
            for raw_skill in claim.skills:
                skill = raw_skill.strip()
                if not skill:
                    continue
                matched_evidence = known_skills.get(skill.casefold())
                if matched_evidence is not None:
                    links.record("experience_skill", matched_evidence.id, created=False)
                    evidenced_names.add(skill.casefold())
                    continue
                if name_key(skill) in rejected_skills:
                    continue
                evidence = ExperienceSkillEvidence(
                    skill=skill, experience_id=target_row.id, evidence_type="resume"
                )
                db.add(evidence)
                db.flush()
                known_skills[skill.casefold()] = evidence
                links.record("experience_skill", evidence.id, created=True)
                evidenced_names.add(skill.casefold())
                new_evidence_to_index.append(evidence)

    _retire_resume_only_skills(db, account_id, evidenced_names)
    db.commit()

    if new_points_to_index:
        try:
            from app.retrieval.index import index_experience_points

            index_experience_points(new_points_to_index, account_id=account_id)
        except Exception:
            pass

    if new_evidence_to_index:
        try:
            from app.retrieval.index import index_experience_skill_evidence

            index_experience_skill_evidence(new_evidence_to_index, account_id=account_id)
        except Exception:
            pass

    return added, enriched, points_added


def _merge_education(
    db: Session, account_id: int, claims: list[EducationClaim], links: _LinkRecorder
) -> tuple[int, int]:
    by_key = {
        (row.institution.strip().casefold(), row.degree.strip().casefold()): row
        for row in db.execute(
            select(Education).where(Education.account_id == account_id)
        ).scalars()
    }

    added = 0
    enriched = 0
    for claim in claims:
        key = (claim.institution.casefold(), claim.degree.casefold())
        existing = by_key.get(key)
        if existing is None:
            row = Education(
                account_id=account_id,
                institution=claim.institution,
                degree=claim.degree,
                location=claim.location,
                start_date=claim.start_date,
                end_date=claim.end_date,
            )
            db.add(row)
            db.flush()
            by_key[key] = row  # this pass's own duplicate claims match, not double-add
            links.record("education", row.id, created=True)
            added += 1
            continue

        links.record("education", existing.id, created=False)
        changed = False
        for attr in ("location", "start_date", "end_date"):
            value = getattr(claim, attr)
            if getattr(existing, attr) is None and value is not None:
                setattr(existing, attr, value)
                changed = True
        if changed:
            enriched += 1

    db.commit()
    return added, enriched


def _phone_digits(phone: str) -> str:
    return re.sub(r"\D", "", phone)


def _same_phone(a: str, b: str) -> bool:
    """Equal digits, or one is the other plus a country-code prefix of at
    most three digits ("+1 555 123 4567" vs "(555) 123-4567")."""
    if not a or not b:
        return False
    short, long = sorted((a, b), key=len)
    return long == short or (
        len(short) >= 7 and len(long) - len(short) <= 3 and long.endswith(short)
    )


def _link_key(url: str) -> str:
    raw = url.strip()
    parts = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = parts.netloc.casefold().removeprefix("www.")
    return f"{host}{parts.path.rstrip('/')}".casefold()


def _merge_contact(db: Session, account_id: int, claim: ContactClaim) -> tuple[int, int, int, bool]:
    account = db.get(Account, account_id)
    if account is None:
        return 0, 0, 0, False

    emails = list(
        db.execute(select(ContactEmail).where(ContactEmail.account_id == account_id)).scalars()
    )
    known_emails = {e.email.strip().casefold() for e in emails}
    if account.contact_email:
        known_emails.add(account.contact_email.strip().casefold())
    emails_added = 0
    for email in claim.emails:
        key = email.strip().casefold()
        if not key or key in known_emails:
            continue
        make_primary = not emails and not emails_added and not account.contact_email
        db.add(ContactEmail(account_id=account_id, email=email.strip(), is_primary=make_primary))
        if make_primary:
            account.contact_email = email.strip()
        known_emails.add(key)
        emails_added += 1

    phones = list(
        db.execute(select(ContactPhone).where(ContactPhone.account_id == account_id)).scalars()
    )
    known_phones = [_phone_digits(p.phone) for p in phones]
    if account.contact_phone:
        known_phones.append(_phone_digits(account.contact_phone))
    phones_added = 0
    for phone in claim.phones:
        digits = _phone_digits(phone)
        if not digits or any(_same_phone(digits, k) for k in known_phones):
            continue
        make_primary = not phones and not phones_added and not account.contact_phone
        db.add(ContactPhone(account_id=account_id, phone=phone.strip(), is_primary=make_primary))
        if make_primary:
            account.contact_phone = phone.strip()
        known_phones.append(digits)
        phones_added += 1

    known_links = {
        _link_key(url)
        for (url,) in db.execute(select(SocialLink.url).where(SocialLink.account_id == account_id))
    }
    if account.github_username:
        known_links.add(f"github.com/{account.github_username.strip().casefold()}")
    links_added = 0
    for link in claim.links:
        key = _link_key(link.url)
        if not key or key in known_links:
            continue
        db.add(
            SocialLink(
                account_id=account_id, platform=link.platform, url=link.url, label=link.label
            )
        )
        known_links.add(key)
        links_added += 1

    location_filled = False
    if not (account.contact_location or "").strip() and claim.location:
        account.contact_location = claim.location
        location_filled = True

    if emails_added or phones_added or links_added or location_filled:
        db.commit()
    return emails_added, phones_added, links_added, location_filled


def merge_resume_into_profile(
    db: Session,
    account_id: int,
    extraction: ResumeExtraction,
    resume_id: int | None = None,
) -> ProfileMergeSummary:
    """Runs every merge against one just-succeeded extraction. Safe to
    call again on the same extraction (reprocessing a resume re-merges
    it): every duplicate check is against the account's current state,
    so a second pass adds nothing new for what already made it in, and
    can still enrich a date or point that got filled in elsewhere since.
    """
    links = _LinkRecorder(db, resume_id)
    # Roles first, so a skill the resume ties to a role lands as that
    # role's evidence and the top-level tags pass then sees it as already
    # present instead of adding the same name again as freestanding.
    experiences_added, experiences_enriched, points_added = _merge_experiences(
        db, account_id, extraction.experiences, links
    )
    skills_added = _merge_skills(db, account_id, extraction.tags, links)
    education_added, education_enriched = _merge_education(
        db, account_id, extraction.education, links
    )
    emails_added, phones_added, links_added, location_filled = _merge_contact(
        db, account_id, extraction.contact
    )
    return ProfileMergeSummary(
        skills_added=skills_added,
        experiences_added=experiences_added,
        experiences_enriched=experiences_enriched,
        experience_points_added=points_added,
        education_added=education_added,
        education_enriched=education_enriched,
        emails_added=emails_added,
        phones_added=phones_added,
        links_added=links_added,
        location_filled=location_filled,
    )
