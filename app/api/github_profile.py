"""The account's GitHub profile README (the "hi/hi" repo for user "hi"),
for the portfolio overview card and the home page suggestion.

It is synced and skill-extracted like any repo, but it is not a project:
app/api/projects.py leaves it out of the projects list, and this router is
where it shows up instead. When the account's GitHub has been synced and
no such repo came back, the state is "missing", and the UI suggests
creating one, since a profile README is the first thing a recruiter sees
on the GitHub profile.

Re-extracting after the person edits it needs nothing special here: an
edit moves the repo's pushed_at, the next sync marks it pending, and the
extraction worker picks it up like any changed repo. The card's Reprocess
button goes through POST /api/projects/{id}/reprocess.
"""

from __future__ import annotations

import datetime as dt
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from app.api.deps import DbSession
from app.core.db import Account, GitHubSyncRun, Repository, SkillEvidence
from app.ingest.github.background import account_key

router = APIRouter(prefix="/api/github-profile")


class ProfileReadme(BaseModel):
    id: int
    full_name: str
    url: str
    has_readme: bool
    skill_extraction_status: str
    skill_extraction_error: str | None
    skills_extracted_at: dt.datetime | None
    pushed_at: dt.datetime | None


class GitHubProfileState(BaseModel):
    github_username: str
    # "present": the profile repo is synced. "missing": the account's
    # GitHub was synced and has no profile repo. "not_synced": nothing
    # synced yet, so there is no telling.
    state: str
    repo: ProfileReadme | None = None
    skills: list[str] = []
    # GitHub's new-repo form with the name already filled in
    create_url: str


def _synced(db: DbSession, account_id: int, username: str) -> bool:
    owned = db.execute(
        select(func.count(Repository.id)).where(
            Repository.account_id == account_id,
            func.lower(Repository.full_name).like(f"{username.lower()}/%"),
        )
    ).scalar()
    if owned:
        return True
    run_state = db.execute(
        select(GitHubSyncRun.state).where(GitHubSyncRun.key == account_key(username))
    ).scalar_one_or_none()
    return run_state == "done"


@router.get("", response_model=GitHubProfileState)
def get_profile_readme(account_id: int, *, db: DbSession) -> GitHubProfileState:
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
    username = (account.github_username or "").strip()
    create_url = f"https://github.com/new?name={quote(username)}"

    candidates = list(
        db.execute(
            select(Repository).where(
                Repository.account_id == account_id, Repository.is_profile_readme.is_(True)
            )
        ).scalars()
    )
    # The account's own profile first; one synced from another source
    # (a second GitHub login on the same account) is still better than none.
    own = f"{username}/{username}".lower()
    candidates.sort(key=lambda r: r.full_name.lower() != own)
    repo = candidates[0] if candidates else None

    if repo is None:
        state = "missing" if username and _synced(db, account_id, username) else "not_synced"
        return GitHubProfileState(github_username=username, state=state, create_url=create_url)

    skills = sorted(
        set(
            db.execute(
                select(SkillEvidence.skill).where(SkillEvidence.repo_id == repo.id)
            ).scalars()
        ),
        key=str.casefold,
    )
    return GitHubProfileState(
        github_username=username,
        state="present",
        repo=ProfileReadme(
            id=repo.id,
            full_name=repo.full_name,
            url=repo.url,
            has_readme=bool(repo.readme and repo.readme.strip()),
            skill_extraction_status=repo.skill_extraction_status,
            skill_extraction_error=repo.skill_extraction_error,
            skills_extracted_at=repo.skills_extracted_at,
            pushed_at=repo.pushed_at,
        ),
        skills=skills,
        create_url=create_url,
    )
