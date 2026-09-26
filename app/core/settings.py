"""App settings. No secrets in code.

This app runs locally, so almost nothing here is deployment config. The
only value read from `.env` is GITHUB_TOKEN. Everything the user picks
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

    # Optional. Raises the GitHub API rate limit from 60/hr to 5,000/hr.
    # Public repo, README, manifest, and commit-stats data is readable
    # without auth. There is no GITHUB_USERNAME setting: the username is a
    # per-request input (POST /sync/github body, or a CLI arg), not
    # deployment config.
    github_token: str = ""

    database_url: str = "sqlite:///./data/open_to_work.db"

    # Docker Compose sets this to the qdrant service. ":memory:" runs an
    # in-process Qdrant for tests.
    qdrant_url: str = "http://localhost:6333"

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
