"""Multi-provider LLM credential store. Nothing outside this module ever
decrypts a credential for display: listing keys reads
`masked_preview` only; app/core/llm.py is the one caller that gets a real
decrypted dict back (via resolve_dispatch_key()), to hand straight to
litellm and never log or return.

Multiple rows per provider are allowed (rotation, differently-labeled
keys); at most one is `is_active` per provider at a time, enforced here,
not by a DB constraint (see app/core/db.py's ApiKey docstring). Each
provider's credential shape lives in app/core/llm_providers.py, not here:
this module just encrypts/decrypts/masks whatever dict that module hands
it.

Dispatch key resolution (resolve_dispatch_key) is per (provider,
account_id): a key can be `enabled=False` (manually switched off,
regardless of is_active) or restricted to a list of accounts
(allowed_account_ids, empty = every account). The active key is preferred
but skipped if it doesn't cover this account or is disabled; the next
enabled, covering key (by id) is used instead; None means nothing usable
exists for this provider+account.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import select

from app.core.crypto import decrypt, encrypt
from app.core.db import ApiKey, get_db
from app.core.llm_providers import fields_for, validate_credentials


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _mask(value: str) -> str:
    value = value.strip()
    if len(value) <= 6:
        return "•" * max(len(value), 3)
    return f"{value[:3]}{'•' * 6}{value[-3:]}"


def _build_masked_preview(provider: str, credentials: dict) -> dict:
    return {
        f.name: _mask(credentials[f.name]) if f.secret else credentials[f.name]
        for f in fields_for(provider)
    }


def _to_out(row: ApiKey) -> dict:
    return {
        "id": row.id,
        "provider": row.provider,
        "label": row.label,
        "masked": row.masked_preview,
        "status": row.status,
        "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
        "last_check_detail": row.last_check_detail,
        "budget_cap_usd": row.budget_cap_usd,
        "is_active": row.is_active,
        "enabled": row.enabled,
        "allowed_account_ids": row.allowed_account_ids or [],
    }


def list_keys() -> list[dict]:
    db = get_db()
    try:
        rows = db.execute(select(ApiKey).order_by(ApiKey.provider, ApiKey.id)).scalars().all()
        return [_to_out(r) for r in rows]
    finally:
        db.close()


def add_key(
    provider: str,
    label: str,
    credentials: dict,
    budget_cap_usd: float | None,
    allowed_account_ids: list[int] | None = None,
) -> tuple[dict | None, str]:
    status, detail = validate_credentials(provider, credentials)
    if status == "invalid":
        return None, detail
    db = get_db()
    try:
        # First key stored for a provider goes active automatically; a
        # second one for the same provider is added inactive, waiting for
        # activate_key() rather than silently taking over dispatch.
        has_active = db.execute(
            select(ApiKey.id).where(ApiKey.provider == provider, ApiKey.is_active.is_(True))
        ).first()
        row = ApiKey(
            provider=provider,
            label=label.strip() or "Untitled",
            encrypted_credentials=encrypt(json.dumps(credentials)),
            masked_preview=_build_masked_preview(provider, credentials),
            is_active=has_active is None,
            enabled=True,
            allowed_account_ids=sorted(set(allowed_account_ids or [])),
            status=status,
            last_checked_at=_now(),
            last_check_detail=detail,
            budget_cap_usd=budget_cap_usd,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _to_out(row), detail
    finally:
        db.close()


def update_key(
    key_id: int,
    *,
    label: str | None = None,
    budget_cap_usd: float | None = ...,  # type: ignore[assignment]
    allowed_account_ids: list[int] | None = ...,  # type: ignore[assignment]
) -> dict | None:
    """Partial update for the fields that don't need re-validating a
    credential (label, budget cap, the allowed-accounts list). Swapping
    the credentials themselves means adding a new key and deleting the old
    one, so this never touches encrypted_credentials.
    `budget_cap_usd`/`allowed_account_ids` use `...` as "leave unchanged"
    since `None`/`[]` are both meaningful values (no cap, no restriction).
    """
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return None
        if label is not None:
            row.label = label.strip() or "Untitled"
        if budget_cap_usd is not ...:
            row.budget_cap_usd = budget_cap_usd
        if allowed_account_ids is not ...:
            row.allowed_account_ids = sorted(set(allowed_account_ids or []))
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def set_enabled(key_id: int, enabled: bool) -> dict | None:
    """The manual on/off switch, independent of is_active/delete. A
    disabled key is skipped by resolve_dispatch_key() entirely, even if
    it's the active one for its provider; re-enabling needs no
    re-activation."""
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return None
        row.enabled = enabled
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def delete_key(key_id: int) -> None:
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return
        was_active, provider = row.is_active, row.provider
        db.delete(row)
        db.commit()
        if was_active:
            # promote the next-oldest remaining key for that provider (if
            # any) so it doesn't silently end up with none active
            next_row = (
                db.execute(select(ApiKey).where(ApiKey.provider == provider).order_by(ApiKey.id))
                .scalars()
                .first()
            )
            if next_row is not None:
                next_row.is_active = True
                db.commit()
    finally:
        db.close()


def activate_key(key_id: int) -> dict | None:
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return None
        others = (
            db.execute(select(ApiKey).where(ApiKey.provider == row.provider, ApiKey.id != row.id))
            .scalars()
            .all()
        )
        for other in others:
            other.is_active = False
        row.is_active = True
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def check_key(key_id: int) -> dict | None:
    """Re-validates the already-stored credentials in place; the caller
    never resends them."""
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return None
        credentials = json.loads(decrypt(row.encrypted_credentials))
        status, detail = validate_credentials(row.provider, credentials)
        row.status = status
        row.last_checked_at = _now()
        row.last_check_detail = detail
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def _allows_account(row: ApiKey, account_id: int | None) -> bool:
    if not row.allowed_account_ids:  # empty/None = every account allowed
        return True
    if account_id is None:
        return False  # restricted key, no account context to check against
    return account_id in row.allowed_account_ids


def _resolve_row(db, provider: str, account_id: int | None) -> ApiKey | None:
    rows = (
        db.execute(
            select(ApiKey).where(ApiKey.provider == provider, ApiKey.enabled.is_(True))
        )
        .scalars()
        .all()
    )
    active = next((r for r in rows if r.is_active), None)
    if active is not None and _allows_account(active, account_id):
        return active
    for row in sorted(rows, key=lambda r: r.id):
        if _allows_account(row, account_id):
            return row
    return None


def resolve_dispatch_key(
    provider: str, account_id: int | None
) -> tuple[int, dict, float | None] | None:
    """The one function app/core/llm.py calls before dispatch: (key_id,
    decrypted credentials, budget_cap_usd) for whichever key should serve
    this provider+account, or None if nothing usable is configured:
    either no key at all, or every key for this provider is disabled or
    restricted to other accounts.
    """
    db = get_db()
    try:
        row = _resolve_row(db, provider, account_id)
        if row is None:
            return None
        credentials = json.loads(decrypt(row.encrypted_credentials))
        return row.id, credentials, row.budget_cap_usd
    finally:
        db.close()


def get_active_status(provider: str) -> dict:
    """Status of whichever key WOULD serve an unrestricted (account_id=
    None) call for this provider. Used by the /home KPI tile, not by real
    dispatch."""
    db = get_db()
    try:
        row = _resolve_row(db, provider, None)
        if row is None:
            return {"configured": False, "status": "unknown", "last_checked_at": None}
        return {
            "configured": True,
            "status": row.status,
            "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
        }
    finally:
        db.close()


def record_dispatch_outcome(
    key_id: int, *, ok: bool, rate_limited: bool = False, detail: str | None = None
) -> None:
    """Called by app/core/llm.py right after a real dispatch that used
    this exact key (the id resolve_dispatch_key() returned). Only ever
    downgrades: a provider failure updates status between manual checks,
    same as a manual 'check status' would; a success doesn't need a write
    on every call."""
    if ok:
        return
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return
        row.status = "rate_limited" if rate_limited else "invalid"
        row.last_checked_at = _now()
        if detail:
            row.last_check_detail = detail
        db.commit()
    finally:
        db.close()

