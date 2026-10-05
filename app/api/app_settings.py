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
    # Models selected here that LiteLLM cannot price, so no budget can see
    # what they spend (app/core/app_settings.py's is_model_priced). Empty is
    # the healthy case. Returned on GET and on PUT, so the page can warn at
    # the moment someone picks one rather than leaving it to be noticed on
    # /monitor a month later.
    unpriced_models: list[str] = []
    budget_warning: str | None = None


class LlmSettingsUpdate(BaseModel):
    bulk_model: str | None = None
    quality_model: str | None = None
    monthly_budget_usd: float | None = None


class ProviderModelsOut(BaseModel):
    provider: str
    label: str
    prefix: str
    models: list[str]


def _budget_warning(unpriced: list[str]) -> str | None:
    """The one sentence the /apis page shows next to the budget when a
    chosen model cannot be priced. Names the models, says which limits stop
    working, and does not pretend the model is invalid."""
    if not unpriced:
        return None
    names = ", ".join(f'"{m}"' for m in unpriced)
    plural = "these models" if len(unpriced) > 1 else "this model"
    return (
        f"AI calls are being recorded as $0.00, because pricing for {names} is not "
        f"known here. The monthly budget and every per-key budget add up recorded "
        f"cost, so neither will stop spending while {plural} is selected, and the "
        "usage figures will read as zero. The model still works; only the spending "
        "limits are blind to it."
    )


def _out(current: app_settings.LlmSettings) -> LlmSettingsOut:
    unpriced = app_settings.unpriced_models_in_use(current)
    return LlmSettingsOut(
        **current.as_dict(),
        default_bulk_model=app_settings.DEFAULT_BULK_MODEL,
        default_quality_model=app_settings.DEFAULT_QUALITY_MODEL,
        default_monthly_budget_usd=app_settings.DEFAULT_MONTHLY_BUDGET_USD,
        unpriced_models=unpriced,
        budget_warning=_budget_warning(unpriced),
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


class LlmHealthOut(BaseModel):
    degraded: bool
    model: str | None = None
    detail: str | None = None


@router.get("/llm/health", response_model=LlmHealthOut)
def get_llm_health() -> LlmHealthOut:
    """Whether the AI provider was found unavailable in the last few
    minutes with no successful call since (app/core/llm.py's llm_health).
    The nav polls this to show its "AI degraded" pill."""
    from app.core.llm import llm_health

    return LlmHealthOut(**llm_health())
