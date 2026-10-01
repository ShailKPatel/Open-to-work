import json
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from github import BadCredentialsException, RateLimitExceededException, UnknownObjectException
from pydantic import BaseModel

from app.api.accounts import router as accounts_router
from app.api.api_keys import router as api_keys_router
from app.api.app_settings import router as app_settings_router
from app.api.auth_sources import router as auth_sources_router
from app.api.contact import router as contact_router
from app.api.education import router as education_router
from app.api.experience import router as experience_router
from app.api.github_profile import router as github_profile_router
from app.api.github_sync import router as github_sync_router
from app.api.job_analytics import router as job_analytics_router
from app.api.job_postings import router as job_postings_router
from app.api.monitor import router as monitor_router
from app.api.projects import router as projects_router
from app.api.resume import router as resume_router
from app.api.resume_build import router as resume_build_router
from app.api.skills import router as skills_router
from app.api.skills import warm_skill_maps
from app.api.sources import router as sources_router
from app.core.app_settings import get_llm_settings
from app.core.db import init_db
from app.core.embeddings import EMBEDDING_MODEL
from app.core.key_refresh import start_key_refresh
from app.core.settings import get_settings
from app.ingest.github import background as github_background
from app.ingest.github.cancellation import request_cancel
from app.ingest.github.sync import SyncSummary, sync_account


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    # The skill map needs an embedding model in memory, which takes a few
    # seconds to load. Doing it here, off the request path, means the
    # first person to open the map gets a cached layout instead of a
    # spinner (see app/profile/skill_map.py). Daemon so it never holds up
    # shutdown, and started only if the layout is actually stale.
    if get_settings().skill_map_warm_start:
        threading.Thread(target=warm_skill_maps, name="skill-map-warm", daemon=True).start()
    # A key that ran out of quota yesterday is usually fine today, so the
    # keys waiting on a reset are rechecked here and then on an interval
    # (app/core/key_refresh.py). Restarting the app is the moment a person
    # most expects that to have happened.
    if get_settings().key_refresh_on_start:
        start_key_refresh()
    # A GitHub sync cut off by the hourly limit resumes itself once the
    # limit resets; the timers for that live in memory, so they are set
    # again here from what the database remembers.
    github_background.restore_after_restart()
    yield


app = FastAPI(title="Open to Work", lifespan=_lifespan)
app.include_router(accounts_router)
app.include_router(api_keys_router)
app.include_router(app_settings_router)
app.include_router(auth_sources_router)
app.include_router(contact_router)
app.include_router(education_router)
app.include_router(experience_router)
app.include_router(github_profile_router)
app.include_router(github_sync_router)
app.include_router(job_analytics_router)
app.include_router(job_postings_router)
app.include_router(monitor_router)
app.include_router(projects_router)
app.include_router(resume_router)
app.include_router(resume_build_router)
app.include_router(skills_router)
app.include_router(sources_router)
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "web" / "templates"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """First page: who's using this device. Picks an existing local
    account or creates a new one, no password, client-side selection
    only (see app/core/db/models.py's Account docstring)."""
    return templates.TemplateResponse(request, "accounts.html")


@app.get("/home", response_class=HTMLResponse)
def home_page(request: Request) -> HTMLResponse:
    """The base page once logged in: KPI summary (projects, unprocessed
    count, skills, LLM key status for whichever provider the bulk tier
    currently uses) plus a reprocess-all action. Only "Portfolio" has
    anything to summarize today; as the job-collection and resume-building
    sections ship, their own KPIs join this page. Reads against
    GET /api/projects, GET /api/skills, GET /api/api-keys/default-provider,
    and GET /api/api-keys client-side, same thin-page split as everywhere
    else."""
    return templates.TemplateResponse(request, "home.html")


@app.get("/sync", include_in_schema=False)
def sync_page(request: Request) -> RedirectResponse:
    """Old home of GitHub syncing, now /monitor/sync. Keeps its query
    string, so signup's ?new=1 (start syncing straight away) still works."""
    query = f"?{request.url.query}" if request.url.query else ""
    return RedirectResponse(f"/monitor/sync{query}", status_code=307)


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_overview_page(request: Request) -> HTMLResponse:
    """"Portfolio" section landing page: projects/skills/experience/
    education/contact/resume, one layer deeper than the /home KPI row.
    GitHub sources live under /monitor/sync instead, with the rest of the
    background work, rather than with portfolio content. The nested pages
    below share this section's tab strip
    (app/web/templates/_portfolio_subnav.html)."""
    return templates.TemplateResponse(request, "portfolio_overview.html")


@app.get("/portfolio/projects", response_class=HTMLResponse)
def projects_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "projects.html")


@app.get("/portfolio/projects/{repo_id}", response_class=HTMLResponse)
def project_detail_page(request: Request, repo_id: int) -> HTMLResponse:
    """Per-project detail: full repo metadata, README preview, manifests,
    and every extracted skill with its evidence type/weight/confidence.
    Data comes from GET /api/projects/{repo_id} (app/api/projects.py);
    this route just serves the page shell, same split as every other page
    here. repo_id is typed int in the path so a non-numeric segment 404s
    at the routing layer rather than reaching the template."""
    return templates.TemplateResponse(request, "project_detail.html")


@app.get("/portfolio/education", response_class=HTMLResponse)
def education_page(request: Request) -> HTMLResponse:
    """Degree/school list, same split as /portfolio/experience: this route
    just serves the page shell, data comes from GET /api/education
    (app/api/education.py). No detail sub-page, unlike experience:
    Education has no points sub-resource to drill into."""
    return templates.TemplateResponse(request, "education.html")


@app.get("/portfolio/experience", response_class=HTMLResponse)
def experience_page(request: Request) -> HTMLResponse:
    """Work-experience list, same split as /portfolio/projects: this route
    just serves the page shell, data comes from GET /api/experience
    (app/api/experience.py)."""
    return templates.TemplateResponse(request, "experience.html")


@app.get("/portfolio/experience/{experience_id}", response_class=HTMLResponse)
def experience_detail_page(request: Request, experience_id: int) -> HTMLResponse:
    """Per-experience detail, mirroring /portfolio/projects/{repo_id}: full
    fields plus its skill evidence. Data from
    GET /api/experience/{experience_id}."""
    return templates.TemplateResponse(request, "experience_detail.html")


@app.get("/portfolio/skills", response_class=HTMLResponse)
def skills_page(request: Request) -> HTMLResponse:
    """One row per skill, unioned across projects, experience, and
    freestanding manual entries. Data from GET /api/skills
    (app/api/skills.py)."""
    return templates.TemplateResponse(request, "skills.html")


@app.get("/portfolio/contact", response_class=HTMLResponse)
def contact_page(request: Request) -> HTMLResponse:
    """Contact info + social links. Data from GET /api/accounts/{id}/contact
    and GET /api/accounts/{id}/social-links (app/api/contact.py)."""
    return templates.TemplateResponse(request, "contact.html")


@app.get("/monitor", response_class=HTMLResponse)
def monitor_page(request: Request) -> HTMLResponse:
    """Rate-limit/budget monitor: live GitHub headroom + LLM monthly
    spend, plus the append-only event log of every past hit
    (app/core/rate_limits.py). Data from GET /api/monitor/status and
    GET /api/monitor/events (app/api/monitor.py)."""
    return templates.TemplateResponse(request, "monitor.html")


@app.get("/monitor/sync", response_class=HTMLResponse)
def monitor_sync_page(request: Request) -> HTMLResponse:
    """One place for the background work that feeds the portfolio: the
    GitHub accounts/repos this profile pulls from (with their syncs, see
    app/api/sources.py) and skill extraction over what they bring in
    (GET /api/projects/process-pending/status). Under Monitor, next to
    usage, since both are long-running jobs bounded by API limits.
    ?new=1 (from signup) syncs every source straight away."""
    return templates.TemplateResponse(request, "sources.html")


@app.get("/explanation", response_class=HTMLResponse)
def explanation_page(request: Request) -> HTMLResponse:
    """Presentation page: what the project does and how it is built, meant
    to be shown to someone as-is. Tabbed (what it does, tech stack, how it
    works). No API calls and no account required, so it renders fine
    logged out. The embedding model and monthly budget are passed in so
    the page names what this instance actually runs. LLM model names are
    deliberately not shown: they are whatever LiteLLM model is picked on
    /apis. Diagram boxes are laid out by CSS grid and the right-angle
    connectors between them are drawn client-side from real box positions
    (see the script at the bottom of app/web/templates/explanation.html)."""
    models = {
        "embedding": EMBEDDING_MODEL,
        "budget": get_llm_settings().monthly_budget_usd,
    }
    return templates.TemplateResponse(request, "explanation.html", {"models": models})


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request) -> HTMLResponse:
    """Job posting record: paste a posting, get it back as a card (role,
    company, salary, experience/skills required, LLM-extracted, see
    app/profile/job_extract.py) alongside every posting pasted before.
    Top-level section like /home, /portfolio, and /monitor: a job posting
    isn't part of the account's own portfolio, it's a third-party record
    the account is considering applying against. "Build resume for this
    job" on a posting's page (/jobs/{id}) is the only link into
    /portfolio/resume/build; that
    page has no posting picker of its own. Data from GET/POST/PATCH/
    DELETE /api/job-postings and POST /api/job-postings/{id}/reprocess
    (app/api/job_postings.py)."""
    return templates.TemplateResponse(request, "jobs.html")


@app.get("/jobs/analytics", response_class=HTMLResponse)
def jobs_analytics_page(request: Request) -> HTMLResponse:
    """Skill-demand analytics over every job posting an account has
    collected: most-requested skills (with your own have/missing status),
    a per-canonical-role-family breakdown (app/profile/role_family.py),
    and the missing-skills-first framing /home's "Recommended skills" tile
    links into. Data from GET /api/job-analytics/skills-demand and
    GET /api/job-analytics/role-families (app/api/job_analytics.py)."""
    return templates.TemplateResponse(request, "jobs_analytics.html")


@app.get("/jobs/{posting_id}", response_class=HTMLResponse)
def job_detail_page(request: Request, posting_id: int) -> HTMLResponse:
    """One job posting on its own page: pay, type, experience, where the
    account's skills stand against it, the applied tracker, and the jump
    into resume building. Data from GET/PATCH/DELETE
    /api/job-postings/{posting_id} and GET /api/job-analytics/gap/{id}."""
    return templates.TemplateResponse(request, "job_detail.html")


@app.get("/portfolio/resume", response_class=HTMLResponse)
def resume_page(request: Request) -> HTMLResponse:
    """Resume library: every resume the account has, uploaded files and
    ones this app generated alike. Uploads get LLM-extracted tags/
    target-roles/summary (app/profile/resume_extract.py) plus freeform
    notes, all editable by hand afterward; any resume can also be edited
    by writing a prompt to an LLM (POST /api/resume/{id}/edit), which
    updates its summary/projects/skills only, never its work history or
    its layout, see app/resume_build/orchestrator.py's
    edit_resume_content() docstring for how that's enforced structurally,
    not just by prompt instruction. A /portfolio/* tab, since a resume is
    drawn from the same projects/skills/experience/education the rest of
    this section holds, sharing this section's
    tab strip. Data from GET/POST/PATCH/DELETE /api/resume
    (app/api/resume.py)."""
    return templates.TemplateResponse(request, "resume.html")


@app.get("/portfolio/resume/build", response_class=HTMLResponse)
def resume_build_page(request: Request) -> HTMLResponse:
    """Pick a job posting (via ?job_posting_id=, linked from a card on
    /jobs; this page has no posting picker of its own), pick a
    length, generate a tailored resume; a "closest existing resumes"
    panel (GET /api/resume/search) offers a semantic-search shortcut to
    an existing resume instead of generating a new one. Distinct from
    /portfolio/resume (the resume library): this is the page that
    produces a new document from the account's own data; every document
    it produces is also saved into that library (see
    app/api/resume_build.py's generate()) rather than only ever existing
    as a one-off download. Data from GET /api/job-postings/{id}
    (app/api/job_postings.py), GET /api/resume/search (app/api/resume.py),
    and POST /api/resume-build/preview, POST /api/resume-build/generate
    (app/api/resume_build.py)."""
    return templates.TemplateResponse(request, "resume_build.html")


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request) -> HTMLResponse:
    """Delete-account (+ log out). Links out to the /apis page (API keys
    are device-wide, not per-profile) and to /monitor/sync (GitHub sources this
    profile pulls evidence from) below. Data from DELETE /accounts/{id}
    (app/api/accounts.py)."""
    return templates.TemplateResponse(request, "settings.html")


@app.get("/settings/sources", include_in_schema=False)
def old_sources_page() -> RedirectResponse:
    """Moved to /monitor/sync."""
    return RedirectResponse("/monitor/sync", status_code=307)


@app.get("/settings/auth-sources", response_class=HTMLResponse)
def auth_sources_page(request: Request) -> HTMLResponse:
    """Authenticated job-source login profiles: real automated-browser
    logins used to fetch a posting from a site that requires being signed
    in. See app/core/db/models.py's AuthSource docstring and
    app/ingest/jobs/auth_fetch.py's module docstring for the risks. Data from GET/POST/PATCH/DELETE
    /api/auth-sources and POST /api/auth-sources/{id}/test-login
    (app/api/auth_sources.py)."""
    return templates.TemplateResponse(request, "auth_sources.html")


@app.get("/apis", response_class=HTMLResponse)
def apis_page(request: Request) -> HTMLResponse:
    """API key management for every supported LLM provider
    (app/core/llm_providers.py), any number of labeled keys per provider, one
    marked active per provider (app/core/db/models.py's ApiKey model). Device-
    wide, not per-profile, so this page is reachable from two
    entry points rather than duplicating the widget in each: the top of
    the pre-login account picker (app/web/templates/accounts.html) and a
    link on the post-login Settings page. One page, one bit of JS, one
    backend (app/api/api_keys.py, app/core/api_keys_store.py); a new
    provider is a registry entry in app/core/llm_providers.py, not a
    second copy of this page.

    The stored keys show up in three groups, which is the difference the
    page exists to make plain: the ones working, the ones waiting on a
    quota that resets by itself (rechecked automatically, see
    app/core/key_refresh.py), and the ones the provider rejected or
    blocked, which nothing rechecks until someone asks."""
    return templates.TemplateResponse(request, "apis.html")


class GithubSyncRequest(BaseModel):
    username: str
    account_id: int | None = None


class GithubSyncResponse(BaseModel):
    total_repos: int
    fetched: int
    cache_hits: int

    @classmethod
    def from_summary(cls, summary: SyncSummary) -> "GithubSyncResponse":
        return cls(
            total_repos=summary.total_repos,
            fetched=summary.fetched,
            cache_hits=summary.cache_hits,
        )


@app.post("/sync/github", response_model=GithubSyncResponse)
def sync_github(payload: GithubSyncRequest) -> GithubSyncResponse:
    """Not tied to any one account: takes whatever username the caller
    (the web form, or anyone else's client) sends. Runs synchronously in a
    threadpool worker (this def is sync, not async, so FastAPI offloads it),
    fine for one person's account on their own instance; would need a
    background job queue if this ever ran against many accounts at once.
    """
    username = payload.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="username is required")
    try:
        summary = sync_account(username, account_id=payload.account_id)
    except UnknownObjectException as e:
        raise HTTPException(status_code=404, detail=f"GitHub user '{username}' not found") from e
    except BadCredentialsException as e:
        raise HTTPException(
            status_code=502, detail="GitHub rejected the configured token/credentials"
        ) from e
    except RateLimitExceededException as e:
        raise HTTPException(
            status_code=503, detail="GitHub rate limit exhausted; try again shortly"
        ) from e
    return GithubSyncResponse.from_summary(summary)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/sync/github/stream")
def sync_github_stream(
    username: str,
    account_id: int | None = None,
    run_id: str | None = None,
    attach: bool = False,
) -> StreamingResponse:
    """Same sync as POST /sync/github, but in the background with its
    progress as Server-Sent Events (checking profile, listing repos, N of
    M, repo name). GET + query params because the browser's EventSource
    can't send a POST body or custom headers. The sync runs in its own
    thread (app/ingest/github/background.py), so closing this stream does
    not stop it; a second call for the same username joins the running
    sync instead of starting another. attach=true only follows a running
    sync. run_id is an opaque client-generated token for
    POST /sync/github/cancel?run_id=...; events carry the run_id of the
    sync actually running, which is the one to cancel."""
    username = username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="username is required")

    target = github_background.SyncTarget(
        kind="user",
        github_username=username,
        attribution_username=username,
        account_id=account_id,
    )
    key = target.key
    if not attach:
        github_background.start_sync(target, run_id or str(uuid.uuid4()))

    def events() -> Iterator[str]:
        for event in github_background.follow(key):
            yield _sse(event)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/sync/github/status")
def sync_github_status(username: str) -> dict:
    """Whether a sync of this account is running, so /sync can pick its
    progress back up after the page was left mid-sync."""
    return {"running": github_background.is_running(github_background.account_key(username))}


@app.post("/sync/github/cancel")
def sync_github_cancel(run_id: str) -> dict:
    """Flags the in-flight /sync/github/stream call with this run_id to
    stop before its next repo, see app/ingest/github/cancellation.py. No
    error if run_id doesn't match anything in flight (already finished,
    typo, or the stream was never given a run_id): cancelling something
    that isn't running is a no-op, not a failure."""
    request_cancel(run_id)
    return {"cancel_requested": True, "run_id": run_id}
