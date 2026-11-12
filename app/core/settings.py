"""App settings. No secrets in code.

The app runs locally by default, or as a public demo with DEMO_MODE set
(docs/DEPLOY.md). The only value read from `.env` is GITHUB_TOKEN. Everything the user picks
(LLM models, the monthly budget, provider keys) lives in the database and
is edited in the app, see app/core/app_settings.py and the /apis page.

The remaining fields are fixed local paths. They can still be overridden
through the process environment, which is how tests point at a temporary
database and how Docker Compose points the app at its Qdrant container,
but a `.env` file never changes them.
"""

from functools import lru_cache
from typing import Any

from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

_DOTENV_FIELDS = {"github_token"}


class _DotEnvAllowlist(DotEnvSettingsSource):
    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in super().__call__().items() if k in _DOTENV_FIELDS}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional. Raises the GitHub API rate limit from 60/hr to 5,000/hr;
    # public data is readable without it. Which accounts to sync is chosen
    # in the app, not configured here.
    github_token: str = ""

    database_url: str = "sqlite:///./data/open_to_work.db"

    # Docker Compose sets this to the qdrant service. ":memory:" runs an
    # in-process Qdrant for tests.
    qdrant_url: str = "http://localhost:6333"

    # LangGraph's own SQLite file (app/resume_build/graph.py): where a resume
    # build waits for approval and resumes from after a restart. Separate
    # from database_url so graph writes never hold up the app's database.
    resume_graph_checkpoint_path: str = "./data/resume_graph.db"

    # Whether a build started from the page stops for approval before the
    # page fit. POST /generate has nobody to ask, so it never stops.
    resume_approval_required: bool = True

    # Sends LangGraph and LLM traces, resume content included, to LangSmith
    # (app/core/tracing.py). Needs LANGSMITH_API_KEY too.
    tracing_enabled: bool = False

    # Lets the profile agent (app/agent/mcp_hub.py) read the profile's own
    # repos live through GitHub's remote MCP server, next to the app's own
    # MCP servers. Off by default; needs GITHUB_TOKEN as well, sent to that
    # server as a bearer token. The URL asks for the repos toolset in
    # read-only mode, and the hub still allows only a fixed list of read
    # tools on repos that belong to the asking profile.
    mcp_github_enabled: bool = False
    mcp_github_url: str = "https://api.githubcopilot.com/mcp/x/repos/readonly"

    # Public demo (app/api/demo.py, docs/DEPLOY.md). Off for the local app.
    demo_mode: bool = False
    demo_public_host: str = ""
    demo_github_client_id: str = ""
    demo_github_client_secret: str = ""
    demo_session_secret: str = ""
    demo_admin_github_id: int = 0
    # An account anyone may look around without signing in, read-only.
    # 0 turns the "sample profile" button off.
    demo_sample_account_id: int = 0
    demo_max_accounts: int = 50
    demo_daily_resume_builds: int = 5
    demo_daily_llm_calls: int = 200
    demo_daily_cost_usd: float = 0.5
    demo_daily_writes: int = 500
    demo_daily_intake: int = 30
    demo_max_sources: int = 5
    demo_max_running_builds: int = 3

    # Nightly backups to S3-compatible storage (app/core/backup.py). Off
    # until a bucket is named. Its keys are read from the environment.
    backup_bucket: str = ""
    backup_endpoint: str = ""
    backup_region: str = ""
    backup_keep: int = 7

    # Uploaded resumes, stored on local disk in a per-account subfolder.
    resume_storage_dir: str = "./data/resumes"

    # Job-posting screenshots, same per-account layout as resume_storage_dir
    # (see app/api/job_postings.py's from_screenshot).
    job_screenshot_storage_dir: str = "./data/job_screenshots"

    # Builds any missing skill-map layout in a background thread at
    # startup (app/api/skills.py's warm_skill_maps), so the first person
    # to open the map does not wait for an embedding model to load.
    # Turned off by tests, which must not load a model at all.
    skill_map_warm_start: bool = True

    # Rechecks keys whose quota cooldown has elapsed, at startup and on an
    # interval, in a background thread (app/core/key_refresh.py). Turned
    # off by tests, which must not make provider calls; the /apis page can
    # still run the same pass on demand.
    key_refresh_on_start: bool = True

    # Re-syncs every GitHub source that hasn't been synced for a while, in
    # a background thread (app/ingest/github/auto_sync.py). Turned off by
    # tests, which must not call GitHub.
    github_auto_sync_on_start: bool = True

    # Eval harness (app/evals/). Kept outside data/, which is gitignored,
    # so golden labels and result reports can be committed.
    evals_golden_dir: str = "./evals/golden"
    evals_results_dir: str = "./evals/results"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, _DotEnvAllowlist(settings_cls)


@lru_cache
def get_settings() -> Settings:
    return Settings()
