"""LLM model and budget settings, backing the "Models and budget" section
of the /apis page. Stored in the database (app/core/app_settings.py), not
in `.env`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import app_settings
from app.core.llm_providers import PROVIDER_LABELS, PROVIDERS, model_prefix_for_provider

router = APIRouter(prefix="/api/app-settings")


class LlmSettingsOut(BaseModel):
    bulk_model: str
    quality_model: str
    monthly_budget_usd: float
    default_bulk_model: str
    default_quality_model: str
    default_monthly_budget_usd: float


class LlmSettingsUpdate(BaseModel):
    bulk_model: str | None = None
    quality_model: str | None = None
    monthly_budget_usd: float | None = None


class ProviderModelsOut(BaseModel):
    provider: str
    label: str
    prefix: str
    models: list[str]


def _out(current: app_settings.LlmSettings) -> LlmSettingsOut:
    return LlmSettingsOut(
        **current.as_dict(),
        default_bulk_model=app_settings.DEFAULT_BULK_MODEL,
        default_quality_model=app_settings.DEFAULT_QUALITY_MODEL,
        default_monthly_budget_usd=app_settings.DEFAULT_MONTHLY_BUDGET_USD,
    )


@router.get("/llm", response_model=LlmSettingsOut)
def get_llm_settings() -> LlmSettingsOut:
    return _out(app_settings.get_llm_settings())


@router.put("/llm", response_model=LlmSettingsOut)
def update_llm_settings(payload: LlmSettingsUpdate) -> LlmSettingsOut:
    try:
        updated = app_settings.update_llm_settings(**payload.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _out(updated)


@router.get("/llm/models", response_model=list[ProviderModelsOut])
def list_models() -> list[ProviderModelsOut]:
    suggestions = app_settings.suggested_models()
    return [
        ProviderModelsOut(
            provider=provider,
            label=PROVIDER_LABELS[provider],
            prefix=model_prefix_for_provider(provider),
            models=suggestions.get(provider, []),
        )
        for provider in PROVIDERS
    ]
