import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.accounts import router as accounts_router
from app.api.api_keys import router as api_keys_router
from app.api.app_settings import router as app_settings_router
from app.api.contact import router as contact_router
from app.api.education import router as education_router
from app.api.experience import router as experience_router
from app.api.github_profile import router as github_profile_router
from app.api.github_sync import router as github_sync_router
from app.api.job_analytics import router as job_analytics_router
from app.api.job_postings import fail_interrupted_extractions
from app.api.job_postings import router as job_postings_router
from app.api.monitor import router as monitor_router
from app.api.projects import router as projects_router
from app.api.resume import router as resume_router
from app.api.resume_build import router as resume_build_router
from app.api.security import SecurityMiddleware
from app.api.skills import router as skills_router
from app.api.skills import warm_skill_maps
from app.api.sources import router as sources_router
from app.core.app_settings import get_llm_settings
from app.core.db import init_db
from app.core.embeddings import EMBEDDING_MODEL
from app.core.key_refresh import start_key_refresh
from app.core.settings import get_settings
from app.ingest.github import background as github_background
from app.ingest.github.auto_sync import start_auto_sync


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_db()
    # Job posting reads run in memory, so one cut off by the restart would
    # otherwise show as in progress forever.
    fail_interrupted_extractions()
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
    # Synced sources are synced again once they go stale (a week), so new
    # repos and README changes reach the profile without a click.
    if get_settings().github_auto_sync_on_start:
        start_auto_sync()
    yield


app = FastAPI(title="Open to Work", lifespan=_lifespan)
# Host check, cross-site request check and security headers; see
# app/api/security.py.
app.add_middleware(SecurityMiddleware)
app.include_router(accounts_router)
app.include_router(api_keys_router)
app.include_router(app_settings_router)
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


# Page routes only serve the template; every page loads its data from the
# JSON API client-side.


def _page(name: str) -> Callable[[Request], HTMLResponse]:
    def render(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request, name)

    return render


for _path, _template in (
    ("/", "accounts.html"),  # profile picker and first-run setup
    ("/home", "home.html"),
    ("/portfolio", "portfolio_overview.html"),
    ("/portfolio/projects", "projects.html"),
    ("/portfolio/projects/{repo_id:int}", "project_detail.html"),
    ("/portfolio/education", "education.html"),
    ("/portfolio/experience", "experience.html"),
    ("/portfolio/experience/{experience_id:int}", "experience_detail.html"),
    ("/portfolio/skills", "skills.html"),
    ("/portfolio/contact-links", "contact_links.html"),
    ("/portfolio/resume", "resume.html"),
    ("/portfolio/resume/build", "resume_build.html"),
    ("/monitor", "monitor.html"),
    ("/monitor/sync", "sources.html"),
    ("/jobs", "jobs.html"),
    ("/jobs/analytics", "jobs_analytics.html"),
    ("/jobs/{posting_id:int}", "job_detail.html"),
    ("/settings", "settings.html"),
    ("/apis", "apis.html"),
):
    app.add_api_route(_path, _page(_template), methods=["GET"], response_class=HTMLResponse)


@app.get("/explanation", response_class=HTMLResponse)
def explanation_page(request: Request) -> HTMLResponse:
    """Project overview meant to be shown as-is, no account needed. Names
    the embedding model and budget this instance actually runs."""
    models = {
        "embedding": EMBEDDING_MODEL,
        "budget": get_llm_settings().monthly_budget_usd,
    }
    return templates.TemplateResponse(request, "explanation.html", {"models": models})
