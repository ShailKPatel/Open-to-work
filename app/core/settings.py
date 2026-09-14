"""App settings, sourced from environment / .env. No secrets in code."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Optional. Raises the GitHub API rate limit from 60/hr to 5,000/hr.
    # Public repo, README, manifest, and commit-stats data is readable
    # without auth. There is no GITHUB_USERNAME setting: the username is a
    # per-request input (POST /sync/github body, or a CLI arg), not
    # deployment config.
    github_token: str = ""
    database_url: str = "sqlite:///./data/open_to_work.db"

    # LLM model strings in LiteLLM's "<provider>/<model>" form. Change the
    # tiers here and make sure a key for that provider is active on the
    # /apis page (llm_providers.provider_of_model() reads the prefix).
    # API keys are not read from the environment; they are stored encrypted
    # in the database (app/core/api_keys_store.py).
    llm_bulk_model: str = "openai/gpt-4o-mini"
    llm_quality_model: str = "openai/gpt-4o"
    monthly_budget_usd: float = 20.0

    # Local sentence-transformers embeddings, no external API.
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    qdrant_url: str = "http://localhost:6333"

    # Uploaded resumes, stored on local disk in a per-account subfolder.
    resume_storage_dir: str = "./data/resumes"

    # Job-posting screenshots, same per-account layout as resume_storage_dir
    # (see app/api/job_postings.py's from_screenshot).
    job_screenshot_storage_dir: str = "./data/job_screenshots"

    # Eval harness (app/evals/). Kept outside data/, which is gitignored,
    # so golden labels and result reports can be committed.
    evals_golden_dir: str = "./evals/golden"
    evals_results_dir: str = "./evals/results"


@lru_cache
def get_settings() -> Settings:
    return Settings()
