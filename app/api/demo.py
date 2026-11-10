"""Public demo mode: GitHub sign-in, and a gate on every route.

The app was built for one person on one machine, so its routes trust the
account_id they are given and fetch rows by id without asking whose they
are. demo_gate is what makes that safe to put on the internet. It is
deny-by-default: a route is reachable only if it is listed in _ALLOWED, so
a route added later stays closed until someone decides otherwise. Every id
a listed route receives, from the path, query string, JSON body or form, is
checked against the signed-in account in _OWNERS.

Off unless DEMO_MODE is set, and then nothing here runs: the local app is
unchanged. It is an app-level dependency rather than middleware because a
dependency sees the matched route's path template, so the table below names
routes, not URL patterns that could be matched differently from the router.

One account, the one whose GitHub id is DEMO_ADMIN_GITHUB_ID, skips the
gate. That is how the owner adds the LLM key and sets the budget.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.db import (
    Account,
    ApiKey,
    ContactEmail,
    ContactPhone,
    Education,
    Experience,
    ExperiencePoint,
    ExperienceSkillEvidence,
    GitHubSyncRun,
    JobPosting,
    ProjectLink,
    Repository,
    Resume,
    SkillEvidence,
    SocialLink,
    SyncSource,
    get_db,
)
from app.core.settings import Settings, get_settings
from app.core.usage import BUILD_STARTED, DEMO_INTAKE, DEMO_WRITE, count_today, record
from app.resume_build import background

router = APIRouter()

_SESSION_COOKIE = "otw_session"
_STATE_COOKIE = "otw_oauth_state"
_SESSION_SECONDS = 7 * 24 * 3600
_MIN_SECRET_CHARS = 32

_PUBLIC = {
    "/health", "/explanation", "/welcome", "/auth/login", "/auth/callback", "/auth/logout",
    "/auth/sample",
}

# The sample profile is read-only, and these reads stay closed to it too:
# its keys, and the original resume file, which may carry real contact
# details. Streams are closed because some of them start work.
_SAMPLE_CLOSED = {
    "/apis",
    "/api/api-keys",
    "/api/api-keys/providers",
    "/api/api-keys/default-provider",
    "/api/resume/{resume_id}/file",
}

_ALLOWED_PAGES = {
    "/", "/home", "/portfolio", "/portfolio/projects", "/portfolio/projects/{repo_id:int}",
    "/portfolio/education", "/portfolio/experience", "/portfolio/experience/{experience_id:int}",
    "/portfolio/skills", "/portfolio/contact-links", "/portfolio/resume",
    "/portfolio/resume/build", "/monitor/sync", "/jobs", "/jobs/{posting_id:int}", "/apis",
}

_ALLOWED = {
    *(("GET", page) for page in _ALLOWED_PAGES),
    ("GET", "/accounts"),
    ("DELETE", "/accounts/{account_id}"),
    ("GET", "/api/accounts/{account_id}/contact"),
    ("PATCH", "/api/accounts/{account_id}/contact"),
    *(
        (method, f"/api/accounts/{{account_id}}/{kind}")
        for kind in ("social-links", "emails", "phones")
        for method in ("GET", "POST")
    ),
    *(
        (method, path)
        for path in (
            "/api/accounts/{account_id}/social-links/{link_id}",
            "/api/accounts/{account_id}/emails/{email_id}",
            "/api/accounts/{account_id}/phones/{phone_id}",
            "/api/experience/{experience_id}",
            "/api/experience/{experience_id}/points/{point_id}",
            "/api/experience/{experience_id}/skills/{skill_id}",
            "/api/projects/{repo_id}",
            "/api/projects/{repo_id}/skills/{skill_id}",
            "/api/projects/{repo_id}/links/{link_id}",
            "/api/education/{education_id}",
        )
        for method in ("PATCH", "DELETE")
    ),
    ("POST", "/api/experience"),
    ("POST", "/api/experience/{experience_id}/points"),
    ("POST", "/api/experience/{experience_id}/skills"),
    ("POST", "/api/education"),
    ("POST", "/api/projects"),
    ("POST", "/api/projects/{repo_id}/skills"),
    ("POST", "/api/projects/{repo_id}/links"),
    ("POST", "/api/projects/{repo_id}/reprocess"),
    ("POST", "/api/skills"),
    ("POST", "/api/skills/star"),
    ("POST", "/api/skills/archive"),
    ("POST", "/api/skills/remove"),
    ("GET", "/api/skills/rejected"),
    ("GET", "/api/github-profile"),
    ("PATCH", "/api/resume/{resume_id}"),
    ("POST", "/api/resume/{resume_id}/reprocess"),
    ("PATCH", "/api/job-postings/{posting_id}"),
    ("POST", "/api/job-postings/{posting_id}/reprocess"),
    ("POST", "/api/job-postings/compose"),
    ("POST", "/api/sources/sync-all"),
    ("GET", "/api/projects"),
    ("GET", "/api/projects/{repo_id}"),
    ("POST", "/api/projects/process-pending/start"),
    ("GET", "/api/projects/process-pending/status"),
    ("GET", "/api/projects/process-pending/stream"),
    ("GET", "/api/experience"),
    ("GET", "/api/experience/{experience_id}"),
    ("GET", "/api/education"),
    ("GET", "/api/skills"),
    ("GET", "/api/skills/map"),
    ("GET", "/api/resume"),
    ("POST", "/api/resume"),
    ("GET", "/api/resume/search"),
    ("GET", "/api/resume/{resume_id}/file"),
    ("GET", "/api/resume/{resume_id}/compiled-file"),
    ("GET", "/api/resume/{resume_id}/profile"),
    ("POST", "/api/resume/{resume_id}/edit"),
    ("DELETE", "/api/resume/{resume_id}"),
    ("GET", "/api/job-postings"),
    ("POST", "/api/job-postings"),
    ("GET", "/api/job-postings/{posting_id}"),
    ("GET", "/api/job-postings/{posting_id}/images/{index}"),
    ("DELETE", "/api/job-postings/{posting_id}"),
    ("GET", "/api/job-analytics/gap/{posting_id}"),
    # /generate and /preview are absent: they skip the approval step and
    # the per-day build limit.
    ("GET", "/api/resume-build/options"),
    ("POST", "/api/resume-build/skill-review"),
    ("POST", "/api/resume-build/start"),
    ("POST", "/api/resume-build/retry/{resume_id}"),
    ("GET", "/api/resume-build/builds"),
    ("GET", "/api/resume-build/builds/{build_id}"),
    ("GET", "/api/resume-build/builds/{build_id}/pdf"),
    ("POST", "/api/resume-build/builds/{build_id}/decision"),
    ("POST", "/api/resume-build/builds/{build_id}/dismiss"),
    ("GET", "/api/sources"),
    ("POST", "/api/sources"),
    ("DELETE", "/api/sources/{source_id}"),
    ("GET", "/api/sources/sync-all/status"),
    ("GET", "/api/sources/{source_id}/sync/stream"),
    ("POST", "/api/sources/{source_id}/sync/cancel"),
    ("GET", "/api/github-sync"),
    ("POST", "/api/github-sync/{run_row_id}/resume"),
    ("POST", "/api/github-sync/{run_row_id}/dismiss"),
    ("GET", "/api/job-analytics/skills-demand"),
    # Each person's own LLM keys. The model choice and budget stay admin-only.
    ("GET", "/api/api-keys"),
    ("POST", "/api/api-keys"),
    ("GET", "/api/api-keys/providers"),
    ("GET", "/api/api-keys/default-provider"),
    ("POST", "/api/api-keys/recheck"),
    ("PATCH", "/api/api-keys/{key_id}"),
    ("DELETE", "/api/api-keys/{key_id}"),
    ("PUT", "/api/api-keys/{key_id}/credentials"),
    *(
        ("POST", f"/api/api-keys/{{key_id}}/{action}")
        for action in ("check", "activate", "enable", "disable")
    ),
}

# Every kind of id a listed route takes, and the table that says who owns it.
_OWNERS: dict[str, Any] = {
    "resume_id": Resume,
    "posting_id": JobPosting,
    "job_posting_id": JobPosting,
    "repo_id": Repository,
    "experience_id": Experience,
    "source_id": SyncSource,
    "run_row_id": GitHubSyncRun,
    "education_id": Education,
    "key_id": ApiKey,
}

# A child id in the path must belong to the parent beside it, which _OWNERS
# or account_id has already checked: (param, table, column, parent param).
_CHILDREN: dict[str, tuple[str, Any, str, str]] = {
    "/api/accounts/{account_id}/social-links/{link_id}": (
        "link_id", SocialLink, "account_id", "account_id"
    ),
    "/api/accounts/{account_id}/emails/{email_id}": (
        "email_id", ContactEmail, "account_id", "account_id"
    ),
    "/api/accounts/{account_id}/phones/{phone_id}": (
        "phone_id", ContactPhone, "account_id", "account_id"
    ),
    "/api/experience/{experience_id}/points/{point_id}": (
        "point_id", ExperiencePoint, "experience_id", "experience_id"
    ),
    "/api/experience/{experience_id}/skills/{skill_id}": (
        "skill_id", ExperienceSkillEvidence, "experience_id", "experience_id"
    ),
    "/api/projects/{repo_id}/skills/{skill_id}": ("skill_id", SkillEvidence, "repo_id", "repo_id"),
    "/api/projects/{repo_id}/links/{link_id}": ("link_id", ProjectLink, "repo_id", "repo_id"),
}

# These send the input to a model and store it.
_INTAKE_ROUTES = {
    "/api/resume",
    "/api/resume/{resume_id}/reprocess",
    "/api/job-postings",
    "/api/job-postings/compose",
    "/api/job-postings/{posting_id}/reprocess",
    "/api/projects/{repo_id}/reprocess",
}


@dataclass
class DemoUser:
    account_id: int
    first_name: str
    admin: bool
    # A visitor looking around the sample profile without signing in.
    sample: bool = False


def require_valid_demo_settings(settings: Settings) -> None:
    """Refuses to start a public instance that could not protect itself."""
    if not settings.demo_mode:
        return
    missing = [
        name
        for name, value in (
            ("DEMO_PUBLIC_HOST", settings.demo_public_host),
            ("DEMO_GITHUB_CLIENT_ID", settings.demo_github_client_id),
            ("DEMO_GITHUB_CLIENT_SECRET", settings.demo_github_client_secret),
        )
        if not value
    ]
    if len(settings.demo_session_secret) < _MIN_SECRET_CHARS:
        missing.append(f"DEMO_SESSION_SECRET (at least {_MIN_SECRET_CHARS} characters)")
    if missing:
        raise RuntimeError("DEMO_MODE is on but these are not set: " + ", ".join(missing))


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(get_settings().demo_session_secret, salt="otw-session")


def session_account(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    try:
        data = _serializer().loads(token, max_age=_SESSION_SECONDS)
    except BadSignature:
        return None
    return data if isinstance(data, dict) else None


def _denied() -> HTTPException:
    # 404 rather than 403, so an id cannot be probed for existence.
    return HTTPException(status_code=404, detail="Not found.")


def _load_user(session: dict[str, Any]) -> DemoUser | None:
    db = get_db()
    try:
        account = db.get(Account, session.get("account_id"))
        if account is None:
            return None
        sample = bool(session.get("sample"))
        if sample and account.id != get_settings().demo_sample_account_id:
            return None  # the sample was switched to another account, or off
        admin = bool(session.get("admin")) and not sample
        return DemoUser(account.id, account.first_name, admin, sample)
    finally:
        db.close()


async def _submitted_values(request: Request) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = [*request.path_params.items()]
    values.extend(request.query_params.multi_items())
    content_type = request.headers.get("content-type", "")
    try:
        if content_type.startswith("application/json"):
            body = await request.json()
            if isinstance(body, dict):
                values.extend(body.items())
        elif content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            form = await request.form()
            values.extend(form.multi_items())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Unreadable request body.") from e
    return values


def _check_child(db: Session, route_path: str, params: dict[str, Any]) -> None:
    child = _CHILDREN.get(route_path)
    if child is None:
        return
    param, table, column, parent = child
    owner = db.execute(
        select(getattr(table, column)).where(table.id == int(params[param]))
    ).scalar_one_or_none()
    if owner is None or owner != int(params[parent]):
        raise _denied()


def _check_owned(
    user: DemoUser, values: list[tuple[str, Any]], route_path: str, params: dict[str, Any]
) -> None:
    db = get_db()
    try:
        for key, value in values:
            if value is None:
                continue
            if key == "build_id":
                build = background.get(str(value))
                if build is None or build.account_id != user.account_id:
                    raise _denied()
                continue
            if key != "account_id" and key not in _OWNERS:
                continue
            try:
                row_id = int(value)
            except (TypeError, ValueError):
                raise _denied() from None
            if key == "account_id":
                if row_id != user.account_id:
                    raise _denied()
                continue
            owner = db.execute(
                select(_OWNERS[key].account_id).where(_OWNERS[key].id == row_id)
            ).scalar_one_or_none()
            if owner != user.account_id:
                raise _denied()
        _check_child(db, route_path, params)
    finally:
        db.close()


def _check_build_limits(user: DemoUser) -> None:
    settings = get_settings()
    active = [
        b for b in background.list_for_account(user.account_id) if b.state == "running"
    ]
    if active:
        raise HTTPException(status_code=429, detail="You already have a resume build running.")
    if background.running_count() >= settings.demo_max_running_builds:
        raise HTTPException(
            status_code=429, detail="The demo is busy right now. Try again in a minute."
        )
    if count_today(user.account_id, BUILD_STARTED) >= settings.demo_daily_resume_builds:
        raise HTTPException(
            status_code=429,
            detail=f"The demo allows {settings.demo_daily_resume_builds} resume builds a day. "
            "Try again tomorrow.",
        )


def _too_many(detail: str) -> HTTPException:
    return HTTPException(status_code=429, detail=f"{detail} Try again tomorrow.")


def _check_write_limits(user: DemoUser, route_path: str) -> None:
    """Stops one person filling the disk: model spend has its own cap in
    app/core/llm.py, but saving rows and files costs no model call."""
    settings = get_settings()
    if count_today(user.account_id, DEMO_WRITE) >= settings.demo_daily_writes:
        raise _too_many("You have reached today's demo limit for changes.")
    if route_path in _INTAKE_ROUTES:
        if count_today(user.account_id, DEMO_INTAKE) >= settings.demo_daily_intake:
            raise _too_many("You have reached today's demo limit for uploads.")
        record(user.account_id, DEMO_INTAKE)
    if route_path == "/api/sources":
        db = get_db()
        try:
            sources = db.execute(
                select(func.count(SyncSource.id)).where(SyncSource.account_id == user.account_id)
            ).scalar_one()
        finally:
            db.close()
        if sources >= settings.demo_max_sources:
            raise HTTPException(
                status_code=429, detail="The demo allows a few GitHub sources per person."
            )
    record(user.account_id, DEMO_WRITE)


async def demo_gate(request: Request) -> None:
    if not get_settings().demo_mode:
        return
    route_path: str = request.scope["route"].path
    if route_path in _PUBLIC:
        # A signed-in visitor to a public page still gets their account.
        session = session_account(request.cookies.get(_SESSION_COOKIE))
        user = await run_in_threadpool(_load_user, session) if session else None
        if user is not None:
            request.state.demo_user = user
        return
    is_page = not route_path.startswith("/api") and route_path != "/accounts"

    session = session_account(request.cookies.get(_SESSION_COOKIE))
    user = await run_in_threadpool(_load_user, session) if session else None
    if user is None:
        if is_page:
            raise HTTPException(status_code=303, headers={"Location": "/welcome"})
        raise HTTPException(status_code=401, detail="Sign in first.")
    request.state.demo_user = user
    if user.admin:
        return

    if user.sample and (
        request.method != "GET"
        or route_path in _SAMPLE_CLOSED
        or route_path.endswith("/stream")
    ):
        raise _denied()
    if (request.method, route_path) not in _ALLOWED:
        raise _denied()
    if route_path == "/":
        raise HTTPException(status_code=303, headers={"Location": "/home"})
    await run_in_threadpool(
        _check_owned,
        user,
        await _submitted_values(request),
        route_path,
        dict(request.path_params),
    )
    if route_path == "/api/resume-build/start":
        await run_in_threadpool(_check_build_limits, user)
    if request.method != "GET":
        await run_in_threadpool(_check_write_limits, user, route_path)


def _require_demo() -> Settings:
    settings = get_settings()
    if not settings.demo_mode:
        raise HTTPException(status_code=404, detail="Not found.")
    return settings


def _callback_url(settings: Settings) -> str:
    return f"https://{settings.demo_public_host}/auth/callback"


@router.get("/auth/login")
def login() -> Response:
    settings = _require_demo()
    state = secrets.token_urlsafe(32)
    # No scope: only the public profile is needed, and it is what is read.
    query = urlencode(
        {
            "client_id": settings.demo_github_client_id,
            "redirect_uri": _callback_url(settings),
            "state": state,
            "scope": "",
        }
    )
    response = RedirectResponse(f"https://github.com/login/oauth/authorize?{query}", 303)
    response.set_cookie(
        _STATE_COOKIE, state, max_age=600, httponly=True, secure=True, samesite="lax", path="/auth"
    )
    return response


def _github_profile(settings: Settings, code: str) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=10) as client:
            token = client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": settings.demo_github_client_id,
                    "client_secret": settings.demo_github_client_secret,
                    "code": code,
                    "redirect_uri": _callback_url(settings),
                },
            ).json().get("access_token")
            if not token:
                raise HTTPException(status_code=400, detail="GitHub sign-in failed. Try again.")
            response = client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as e:
        raise HTTPException(status_code=502, detail="Could not reach GitHub.") from e


def _account_for(settings: Settings, profile: dict[str, Any]) -> int:
    """The account for this GitHub identity, matched on GitHub's numeric id:
    a login can be renamed and then claimed by somebody else."""
    db = get_db()
    try:
        github_id = int(profile["id"])
        account = db.execute(
            select(Account).where(Account.github_user_id == github_id)
        ).scalar_one_or_none()
        if account is not None:
            return account.id
        total = db.execute(select(func.count(Account.id))).scalar_one()
        if total >= settings.demo_max_accounts and github_id != settings.demo_admin_github_id:
            raise HTTPException(status_code=403, detail="The demo is full right now.")
        login_name = str(profile["login"])
        first, _, last = (profile.get("name") or login_name).strip().partition(" ")
        account = Account(
            first_name=first or login_name,
            last_name=last,
            github_username=login_name,
            github_user_id=github_id,
        )
        db.add(account)
        db.commit()
        return account.id
    finally:
        db.close()


@router.get("/auth/callback")
def callback(request: Request, code: str = "", state: str = "") -> Response:
    settings = _require_demo()
    expected = request.cookies.get(_STATE_COOKIE, "")
    if not code or not state or not expected or not secrets.compare_digest(state, expected):
        raise HTTPException(status_code=400, detail="Sign-in expired. Start again.")
    profile = _github_profile(settings, code)
    account_id = _account_for(settings, profile)
    token = _serializer().dumps(
        {"account_id": account_id, "admin": int(profile["id"]) == settings.demo_admin_github_id}
    )
    response = RedirectResponse("/home", 303)
    response.delete_cookie(_STATE_COOKIE, path="/auth")
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=_SESSION_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@router.post("/auth/sample")
def sample() -> Response:
    """Opens the sample profile read-only, with no GitHub sign-in."""
    settings = _require_demo()
    db = get_db()
    try:
        exists = db.get(Account, settings.demo_sample_account_id) is not None
    finally:
        db.close()
    if not exists:
        raise HTTPException(status_code=404, detail="Not found.")
    token = _serializer().dumps(
        {"account_id": settings.demo_sample_account_id, "admin": False, "sample": True}
    )
    response = RedirectResponse("/home", 303)
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=24 * 3600,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


@router.post("/auth/logout")
def logout() -> Response:
    _require_demo()
    response = RedirectResponse("/welcome", 303)
    response.delete_cookie(_SESSION_COOKIE)
    return response
