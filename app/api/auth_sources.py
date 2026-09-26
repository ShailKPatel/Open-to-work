"""Authenticated job-source management (/settings/auth-sources): stored
login profiles for fetching a posting from a login-walled site via a real
automated browser login (app/ingest/jobs/auth_fetch.py, Playwright).

Read app/core/db/models.py's AuthSource docstring and
app/ingest/jobs/auth_fetch.py's module docstring before touching this
file. Automated login is opt-in and intended for a person's own job
search with their own credentials. Every create requires acknowledged_risk=True, checked here
(422 without it), not just a UI checkbox someone could bypass by calling
this endpoint directly.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.auth_sources_store import (
    add_source,
    delete_source,
    list_sources,
    set_enabled,
)
from app.ingest.jobs.auth_fetch import AuthLoginFailedError, AuthSourceNotFoundError, test_login

router = APIRouter(prefix="/api/auth-sources")


class AuthSourceOut(BaseModel):
    id: int
    account_id: int
    label: str
    site_domain: str
    login_url: str
    username_selector: str
    password_selector: str
    submit_selector: str
    post_login_wait_selector: str | None
    masked_username: str
    enabled: bool
    status: str
    last_checked_at: str | None
    last_check_detail: str | None
    created_at: str


@router.get("", response_model=list[AuthSourceOut])
def list_auth_sources(account_id: int) -> list[dict]:
    return list_sources(account_id)


class AuthSourceCreate(BaseModel):
    account_id: int
    label: str
    site_domain: str
    login_url: str
    username_selector: str
    password_selector: str
    submit_selector: str
    post_login_wait_selector: str | None = None
    username: str
    password: str
    acknowledged_risk: bool = False


@router.post("", response_model=AuthSourceOut)
def create_auth_source(body: AuthSourceCreate) -> dict:
    if not body.acknowledged_risk:
        raise HTTPException(
            status_code=422,
            detail=(
                "you must acknowledge the ban-risk warning to add an authenticated "
                "source; use an alternate account, never your primary one"
            ),
        )
    try:
        return add_source(
            account_id=body.account_id,
            label=body.label,
            site_domain=body.site_domain,
            login_url=body.login_url,
            username_selector=body.username_selector,
            password_selector=body.password_selector,
            submit_selector=body.submit_selector,
            post_login_wait_selector=body.post_login_wait_selector,
            username=body.username,
            password=body.password,
            acknowledged_risk=body.acknowledged_risk,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


class EnabledUpdate(BaseModel):
    enabled: bool


@router.patch("/{source_id}/enabled", response_model=AuthSourceOut)
def update_enabled(source_id: int, body: EnabledUpdate) -> dict:
    row = set_enabled(source_id, body.enabled)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no authenticated source with id={source_id}")
    return row


@router.delete("/{source_id}")
def delete_auth_source(source_id: int) -> dict:
    delete_source(source_id)
    return {"deleted": True}


@router.post("/{source_id}/test-login")
def test_auth_source_login(source_id: int) -> dict:
    """Logs in and nothing else, so selectors can be verified before
    they're ever used against a real target job posting. See
    app/ingest/jobs/auth_fetch.py's test_login."""
    try:
        test_login(source_id)
    except AuthSourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AuthLoginFailedError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"ok": True}
