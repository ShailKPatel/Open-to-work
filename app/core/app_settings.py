"""User-picked LLM settings, stored in the `app_settings` table and edited
on the /apis page: the model for each tier and the global monthly budget.

Defaults point at Gemini, whose API keys are free to create in Google AI
Studio. The "-latest" aliases follow Google's current Flash models, so a
fresh install does not break when a pinned model version is retired.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Literal

from app.core.db import AppSetting, get_db
from app.core.llm_providers import PROVIDERS, model_prefix_for_provider, provider_of_model

logger = logging.getLogger(__name__)

Tier = Literal["bulk", "quality"]

DEFAULT_BULK_MODEL = "gemini/gemini-flash-lite-latest"
DEFAULT_QUALITY_MODEL = "gemini/gemini-flash-latest"
DEFAULT_MONTHLY_BUDGET_USD = 20.0

_BULK_MODEL = "llm_bulk_model"
_QUALITY_MODEL = "llm_quality_model"
_MONTHLY_BUDGET = "monthly_budget_usd"

_DEFAULTS: dict[str, object] = {
    _BULK_MODEL: DEFAULT_BULK_MODEL,
    _QUALITY_MODEL: DEFAULT_QUALITY_MODEL,
    _MONTHLY_BUDGET: DEFAULT_MONTHLY_BUDGET_USD,
}


@dataclass(frozen=True)
class LlmSettings:
    bulk_model: str
    quality_model: str
    monthly_budget_usd: float

    def model_for(self, tier: Tier) -> str:
        return self.bulk_model if tier == "bulk" else self.quality_model

    def as_dict(self) -> dict:
        return asdict(self)


def get_llm_settings() -> LlmSettings:
    db = get_db()
    try:
        stored = {row.key: row.value for row in db.query(AppSetting).all()}
    finally:
        db.close()
    values = {**_DEFAULTS, **{k: v for k, v in stored.items() if k in _DEFAULTS}}
    return LlmSettings(
        bulk_model=_usable_model(values[_BULK_MODEL], DEFAULT_BULK_MODEL),
        quality_model=_usable_model(values[_QUALITY_MODEL], DEFAULT_QUALITY_MODEL),
        monthly_budget_usd=float(values[_MONTHLY_BUDGET]),  # type: ignore[arg-type]
    )


def _usable_model(value: object, default: str) -> str:
    """A stored model that no longer passes validate_model() (saved before
    a rule existed) is never dispatched; the default is used instead. The
    value itself isn't logged, since it may be a pasted API key."""
    try:
        return validate_model(str(value))
    except ValueError:
        logger.warning("ignoring an invalid stored model setting; using %s", default)
        return default


# Provider API keys start with recognizable prefixes. One pasted into a
# model field would be stored in plain text and sent out as a model name.
_API_KEY_PREFIXES = ("AIza", "sk-", "gsk_", "xai-", "hf_", "AKIA", "ya29.")


def validate_model(model: str) -> str:
    """Returns the trimmed model string, or raises ValueError. A model
    must name its provider ("gemini/gemini-flash-latest") so dispatch
    knows which stored key to use."""
    model = model.strip()
    prefix, _, name = model.partition("/")
    if name.strip().startswith(_API_KEY_PREFIXES) or model.startswith(_API_KEY_PREFIXES):
        raise ValueError(
            "That looks like an API key, not a model name. Keys go under \"Add a key\"; "
            "this field takes a model name such as gemini-flash-latest."
        )
    if not prefix or not name.strip():
        raise ValueError(f"model {model!r} must look like <provider>/<model>")
    if provider_of_model(model) not in PROVIDERS:
        raise ValueError(f"unknown provider {prefix!r} in model {model!r}")
    return model


def update_llm_settings(
    *,
    bulk_model: str | None = None,
    quality_model: str | None = None,
    monthly_budget_usd: float | None = None,
) -> LlmSettings:
    """None leaves a setting unchanged. Validates everything before
    writing anything, so a bad value never leaves a half-applied update."""
    updates: dict[str, object] = {}
    if bulk_model is not None:
        updates[_BULK_MODEL] = validate_model(bulk_model)
    if quality_model is not None:
        updates[_QUALITY_MODEL] = validate_model(quality_model)
    if monthly_budget_usd is not None:
        if monthly_budget_usd < 0:
            raise ValueError("monthly budget can't be negative")
        updates[_MONTHLY_BUDGET] = float(monthly_budget_usd)

    if updates:
        db = get_db()
        try:
            for key, value in updates.items():
                row = db.get(AppSetting, key)
                if row is None:
                    db.add(AppSetting(key=key, value=value))
                else:
                    row.value = value
            db.commit()
        finally:
            db.close()
    return get_llm_settings()


@lru_cache
def suggested_models() -> dict[str, list[str]]:
    """Chat models LiteLLM knows for each provider, as full "<prefix>/<name>"
    strings, for the /apis model picker. Retired models (a deprecation
    date in the past) are left out. Only suggestions: any model string
    with a known provider prefix can still be saved, which is how Ollama
    models and new releases are picked."""
    import litellm

    litellm_providers = {
        provider: {model_prefix_for_provider(provider)} for provider in PROVIDERS
    }
    litellm_providers["bedrock"].add("bedrock_converse")
    today = dt.date.today().isoformat()

    out: dict[str, set[str]] = {provider: set() for provider in PROVIDERS}
    for name, info in litellm.model_cost.items():
        if info.get("mode") != "chat" or "*" in name:
            continue
        deprecation = info.get("deprecation_date")
        if isinstance(deprecation, str) and deprecation <= today:
            continue
        for provider, names in litellm_providers.items():
            if info.get("litellm_provider") in names:
                prefix = model_prefix_for_provider(provider)
                bare = name.split("/", 1)[1] if name.startswith(f"{prefix}/") else name
                out[provider].add(f"{prefix}/{bare}")
    return {provider: sorted(models) for provider, models in out.items()}
