import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
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
from app.api.job_analytics import router as job_analytics_router
from app.api.job_postings import router as job_postings_router
from app.api.monitor import router as monitor_router
from app.api.projects import router as projects_router
from app.api.resume import router as resume_router
from app.api.resume_build import router as resume_build_router
from app.api.skills import router as skills_router
from app.api.sources import router as sources_router
from app.core.app_settings import get_llm_settings
from app.core.db import init_db
from app.core.embeddings import EMBEDDING_MODEL
from app.ingest.github.cancellation import request_cancel
from app.ingest.github.sync import SyncSummary, sync_account, sync_account_progress


@asynccontextmanager
async def _lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Open to Work", lifespan=_lifespan)
app.include_router(accounts_router)
app.include_router(api_keys_router)
app.include_router(app_settings_router)
app.include_router(auth_sources_router)
app.include_router(contact_router)
app.include_router(education_router)
app.include_router(experience_router)
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
    only (see app/core/db.py's Account docstring)."""
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


@app.get("/sync", response_class=HTMLResponse)
def sync_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "sync.html")


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_overview_page(request: Request) -> HTMLResponse:
    """"Portfolio" section landing page: projects/skills/experience/
    education/contact/resume, one layer deeper than the /home KPI row.
    GitHub sources live under /settings/sources instead, since they are
    profile setup rather than portfolio content. The nested pages
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
    job" on a card is the only link into /portfolio/resume/build; that
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
    are device-wide, not per-profile) and to /settings/sources (GitHub sources this
    profile pulls evidence from) below. Data from DELETE /accounts/{id}
    (app/api/accounts.py)."""
    return templates.TemplateResponse(request, "settings.html")


@app.get("/settings/sources", response_class=HTMLResponse)
def sources_page(request: Request) -> HTMLResponse:
    """Fetch-data page: manage the list of GitHub accounts/repos this
    profile pulls evidence from, separate from Account.github_username,
    which stays identity-only. Lives under Settings because which accounts
    to pull evidence from is profile setup, not portfolio content. See
    app/api/sources.py."""
    return templates.TemplateResponse(request, "sources.html")


@app.get("/settings/auth-sources", response_class=HTMLResponse)
def auth_sources_page(request: Request) -> HTMLResponse:
    """Authenticated job-source login profiles: real automated-browser
    logins used to fetch a posting from a site that requires being signed
    in. See app/core/db.py's AuthSource docstring and
    app/ingest/jobs/auth_fetch.py's module docstring for the risks. Data from GET/POST/PATCH/DELETE
    /api/auth-sources and POST /api/auth-sources/{id}/test-login
    (app/api/auth_sources.py)."""
    return templates.TemplateResponse(request, "auth_sources.html")


@app.get("/apis", response_class=HTMLResponse)
def apis_page(request: Request) -> HTMLResponse:
    """API key management for every supported LLM provider
    (app/core/llm_providers.py), any number of labeled keys per provider, one
    marked active per provider (app/core/db.py's ApiKey model). Device-
    wide, not per-profile, so this page is reachable from two
    entry points rather than duplicating the widget in each: the top of
    the pre-login account picker (app/web/templates/accounts.html) and a
    link on the post-login Settings page. One page, one bit of JS, one
    backend (app/api/api_keys.py, app/core/api_keys_store.py); a new
    provider is a registry entry in app/core/llm_providers.py, not a
    second copy of this page."""
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
    username: str, account_id: int | None = None, run_id: str | None = None
) -> StreamingResponse:
    """Same sync as POST /sync/github, but as Server-Sent Events so the UI
    can show live progress (checking profile, listing repos, N of M,
    repo name). GET + query params because the browser's EventSource can't
    send a POST body or custom headers. account_id is optional, same as the
    JSON endpoint. run_id is an opaque client-generated token, pass one
    to be able to stop this specific sync mid-way via
    POST /sync/github/cancel?run_id=... (see app/ingest/github/
    cancellation.py); omit it and the sync just can't be stopped early."""
    username = username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="username is required")

    def events():
        try:
            for event in sync_account_progress(
                username, account_id=account_id, run_id=run_id
            ):
                yield _sse(event)
        except UnknownObjectException:
            yield _sse({"stage": "error", "detail": f"GitHub user '{username}' not found"})
        except BadCredentialsException:
            yield _sse(
                {"stage": "error", "detail": "GitHub rejected the configured token/credentials"}
            )
        except RateLimitExceededException:
            yield _sse(
                {"stage": "error", "detail": "GitHub rate limit exhausted; try again shortly"}
            )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/sync/github/cancel")
def sync_github_cancel(run_id: str) -> dict:
    """Flags the in-flight /sync/github/stream call with this run_id to
    stop before its next repo, see app/ingest/github/cancellation.py. No
    error if run_id doesn't match anything in flight (already finished,
    typo, or the stream was never given a run_id): cancelling something
    that isn't running is a no-op, not a failure."""
    request_cancel(run_id)
    return {"cancel_requested": True, "run_id": run_id}
