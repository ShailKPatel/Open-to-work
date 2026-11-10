"""Multi-provider LLM key management, backing the /apis page. A raw key
only ever flows IN here (add); GET responses always carry `masked`
previews from app/core/api_keys_store.py, never the real value.
See app/core/db/models.py's ApiKey docstring for the multi-provider/multi-key/
per-account data model this backs.

Every key route takes account_id as a query parameter: a key belongs to one
account, and another account's key answers 404. In the public demo,
app/api/demo.py has already checked that account_id is the signed-in one.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.deps import DbSession
from app.core import api_keys_store
from app.core.app_settings import get_llm_settings
from app.core.db import Account
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
    # Quota bookkeeping, set only while `status` is "rate_limited": when
    # the key ran out, which limit it hit, and when it is worth looking at
    # again. See app/core/key_cooldown.py.
    exhausted_at: str | None
    retry_at: str | None
    exhaustion_kind: str | None
    # Whether this key is in the automatic recheck pass at all (quota
    # exhaustion is; a rejected or blocked key is not), and whether its
    # wait is already over.
    auto_rechecked: bool
    recheck_due: bool
    budget_cap_usd: float | None
    is_active: bool
    enabled: bool
    account_id: int | None


class AddKeyRequest(BaseModel):
    provider: str
    label: str = ""
    credentials: dict[str, str]
    budget_cap_usd: float | None = None


class UpdateKeyRequest(BaseModel):
    label: str | None = None
    budget_cap_usd: float | None = None
    has_budget_cap: bool = True  # false clears budget_cap_usd to None


class VerifyKeyRequest(BaseModel):
    provider: str
    credentials: dict[str, str]


class VerifyKeyOut(BaseModel):
    status: str
    detail: str


class ReplaceCredentialsRequest(BaseModel):
    credentials: dict[str, str]


class RecheckRequest(BaseModel):
    # "due"       exhausted keys whose cooldown has elapsed (what the
    #             background pass does, app/core/key_refresh.py)
    # "exhausted" every exhausted key, cooldown elapsed or not
    # "blocked"   the rejected/blocked keys, which nothing checks on its own
    scope: str = "due"


class RecheckOut(BaseModel):
    scope: str
    checked: int
    recovered: int
    keys: list[ApiKeyOut]


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


@router.post("/verify", response_model=VerifyKeyOut)
def verify_key(payload: VerifyKeyRequest) -> VerifyKeyOut:
    """Checks a credential without storing it: the first-run wizard checks
    the key before a profile exists, then sends it with the new profile
    (POST /accounts). A rejected key is a 422."""
    if payload.provider not in PROVIDERS:
        raise HTTPException(status_code=422, detail=f"unknown provider {payload.provider!r}")
    # Through the store's import, the one place credential checks are made.
    status, detail = api_keys_store.validate_credentials(payload.provider, payload.credentials)
    if status == "invalid":
        raise HTTPException(status_code=422, detail=detail)
    return VerifyKeyOut(status=status, detail=detail)


@router.get("", response_model=list[ApiKeyOut])
def list_keys(account_id: int) -> list[ApiKeyOut]:
    return [ApiKeyOut(**row) for row in api_keys_store.list_keys(account_id)]


def _require_account(db: DbSession, account_id: int) -> None:
    if db.get(Account, account_id) is None:
        raise HTTPException(status_code=404, detail="no such profile")


@router.post("", response_model=ApiKeyOut)
def add_key(payload: AddKeyRequest, account_id: int, *, db: DbSession) -> ApiKeyOut:
    _require_account(db, account_id)
    if payload.provider not in PROVIDERS:
        raise HTTPException(status_code=422, detail=f"unknown provider {payload.provider!r}")
    row, detail = api_keys_store.add_key(
        payload.provider,
        payload.label,
        payload.credentials,
        payload.budget_cap_usd,
        account_id=account_id,
    )
    if row is None:
        raise HTTPException(status_code=422, detail=detail)
    return ApiKeyOut(**row)


@router.patch("/{key_id}", response_model=ApiKeyOut)
def update_key(key_id: int, payload: UpdateKeyRequest, account_id: int) -> ApiKeyOut:
    row = api_keys_store.update_key(
        key_id,
        label=payload.label,
        budget_cap_usd=payload.budget_cap_usd if payload.has_budget_cap else None,
        account_id=account_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.put("/{key_id}/credentials", response_model=ApiKeyOut)
def replace_credentials(
    key_id: int, payload: ReplaceCredentialsRequest, account_id: int
) -> ApiKeyOut:
    """Swaps the secret on an existing key. Same check as adding one; a
    rejected credential is a 422 and the stored one stays as it was."""
    try:
        row, detail = api_keys_store.replace_credentials(
            key_id, payload.credentials, account_id
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="no such key") from None
    if row is None:
        raise HTTPException(status_code=422, detail=detail)
    return ApiKeyOut(**row)


@router.post("/{key_id}/check", response_model=ApiKeyOut)
def check_key(key_id: int, account_id: int) -> ApiKeyOut:
    row = api_keys_store.check_key(key_id, account_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/recheck", response_model=RecheckOut)
def recheck_keys(account_id: int, payload: RecheckRequest | None = None) -> RecheckOut:
    """Runs the recheck pass now, for one group of unhappy keys. Same code
    the background refresh runs (app/core/key_refresh.py), so the buttons
    on /apis and the startup pass cannot drift apart.

    `scope` is the whole point: "due" and "exhausted" are about quotas,
    which come back on their own, while "blocked" is the explicit ask for
    keys the provider has shut off, which are never touched otherwise.
    """
    scope = (payload or RecheckRequest()).scope
    try:
        rows = api_keys_store.recheck_keys(scope, account_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return RecheckOut(
        scope=scope,
        checked=len(rows),
        recovered=sum(1 for row in rows if row["status"] == "valid"),
        keys=[ApiKeyOut(**row) for row in rows],
    )


@router.post("/{key_id}/activate", response_model=ApiKeyOut)
def activate_key(key_id: int, account_id: int) -> ApiKeyOut:
    row = api_keys_store.activate_key(key_id, account_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/enable", response_model=ApiKeyOut)
def enable_key(key_id: int, account_id: int) -> ApiKeyOut:
    row = api_keys_store.set_enabled(key_id, True, account_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.post("/{key_id}/disable", response_model=ApiKeyOut)
def disable_key(key_id: int, account_id: int) -> ApiKeyOut:
    row = api_keys_store.set_enabled(key_id, False, account_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such key")
    return ApiKeyOut(**row)


@router.delete("/{key_id}")
def delete_key(key_id: int, account_id: int) -> dict:
    if not api_keys_store.delete_key(key_id, account_id):
        raise HTTPException(status_code=404, detail="no such key")
    return {"deleted": True}
