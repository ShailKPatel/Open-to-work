"""Multi-provider LLM key management, backing the /apis page. A raw key
only ever flows IN here (add); GET responses always carry `masked`
previews from app/core/api_keys_store.py, never the real value.
See app/core/db.py's ApiKey docstring for the multi-provider/multi-key/
per-account data model this backs.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import api_keys_store
from app.core.app_settings import get_llm_settings
from app.core.llm_providers import PROVIDER_LABELS, PROVIDERS, provider_of_model

router = APIRouter(prefix="/api/api-keys")


class ProviderFieldOut(BaseModel):
    name: str
    label: str
    secret: bool
    placeholder: str


class ProviderOut(BaseModel):
    provider: str
    label: str
    fields: list[ProviderFieldOut]


class ApiKeyOut(BaseModel):
    id: int
    provider: str
    label: str
    masked: dict
    status: str
    last_checked_at: str | None
    last_check_detail: str | None
    budget_cap_usd: float | None
    is_active: bool
    enabled: bool
    allowed_account_ids: list[int]


class AddKeyRequest(BaseModel):
    provider: str
    label: str = ""
    credentials: dict[str, str]
    budget_cap_usd: float | None = None
    # Empty (default) = every profile on this device may use this key.
    # Non-empty = restricted to just these Account ids.
    allowed_account_ids: list[int] = []


class UpdateKeyRequest(BaseModel):
    label: str | None = None
    budget_cap_usd: float | None = None
    has_budget_cap: bool = True  # false clears budget_cap_usd to None
    allowed_account_ids: list[int] = []


class DefaultProviderOut(BaseModel):
    bulk: str
    quality: str
    bulk_label: str
    quality_label: str


@router.get("/providers", response_model=list[ProviderOut])
def list_providers() -> list[ProviderOut]:
    """Drives the /apis "add a key" provider dropdown and its field list;
    the frontend never hardcodes a provider's shape, it asks here."""
    return [
        ProviderOut(
            provider=provider,
            label=PROVIDER_LABELS[provider],
            fields=[
                ProviderFieldOut(
                    name=f.name, label=f.label, secret=f.secret, placeholder=f.placeholder
                )
                for f in fields
            ],
        )
        for provider, fields in PROVIDERS.items()
    ]


@router.get("/default-provider", response_model=DefaultProviderOut)
def default_provider() -> DefaultProviderOut:
    """Which provider the bulk/quality tiers currently point at (the models
    picked on /apis, app/core/app_settings.py). Drives onboarding's
    "connect your key" step so its wording and target provider follow
    whatever is picked."""
    settings = get_llm_settings()
    bulk = provider_of_model(settings.bulk_model)
    quality = provider_of_model(settings.quality_model)
    return DefaultProviderOut(
        bulk=bulk,
        quality=quality,
        bulk_label=PROVIDER_LABELS.get(bulk, bulk),
        quality_label=PROVIDER_LABELS.get(quality, quality),
    )


@router.get("", response_model=list[ApiKeyOut])
def list_keys() -> list[ApiKeyOut]:
    return [ApiKeyOut(**row) for row in api_keys_store.list_keys()]


@router.post("", response_model=ApiKeyOut)
def add_key(payload: AddKeyRequest) -> ApiKeyOut:
    if payload.provider not in PROVIDERS:
        raise HTTPException(status_code=422, detail=f"unknown provider {payload.provider!r}")
    row, detail = api_keys_store.add_key(
        payload.provider,
        payload.label,
        payload.credentials,
        payload.budget_cap_usd,
        allowed_account_ids=payload.allowed_account_ids,
    )
    if row is None:
        raise HTTPException(status_code=422, detail=detail)
    return ApiKeyOut(**row)


@router.patch("/{key_id}", response_model=ApiKeyOut)
def update_key(key_id: int, payload: UpdateKeyRequest) -> ApiKeyOut:
    row = api_keys_store.update_key(
        key_id,
        label=payload.label,
        budget_cap_usd=payload.budget_cap_usd if payload.has_budget_cap else None,
        allowed_account_ids=payload.allowed_account_ids,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/check", response_model=ApiKeyOut)
def check_key(key_id: int) -> ApiKeyOut:
    row = api_keys_store.check_key(key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/activate", response_model=ApiKeyOut)
def activate_key(key_id: int) -> ApiKeyOut:
    row = api_keys_store.activate_key(key_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/enable", response_model=ApiKeyOut)
def enable_key(key_id: int) -> ApiKeyOut:
    row = api_keys_store.set_enabled(key_id, True)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/disable", response_model=ApiKeyOut)
def disable_key(key_id: int) -> ApiKeyOut:
    row = api_keys_store.set_enabled(key_id, False)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.delete("/{key_id}")
def delete_key(key_id: int) -> dict:
    api_keys_store.delete_key(key_id)
    return {"deleted": True}
