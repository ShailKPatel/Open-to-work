"""Multi-provider LLM credential store. Nothing outside this module ever
decrypts a credential for display: listing keys reads
`masked_preview` only; app/core/llm.py is the one caller that gets a real
decrypted dict back (via resolve_dispatch_keys()), to hand straight to
litellm and never log or return.

Multiple rows per provider are allowed (rotation, differently-labeled
keys); at most one is `is_active` per provider at a time, enforced here,
not by a DB constraint (see app/core/db/models.py's ApiKey docstring). Each
provider's credential shape lives in app/core/llm_providers.py, not here:
this module just encrypts/decrypts/masks whatever dict that module hands
it.

Dispatch key resolution (resolve_dispatch_keys) is per (provider,
account_id): a key can be `enabled=False` (manually switched off,
regardless of is_active) or restricted to a list of accounts
(allowed_account_ids, empty = every account). Disabled keys and keys
restricted to other accounts are dropped outright; what's left comes
back as an ordered list, best first, and app/core/llm.py walks down it,
trying the next one whenever the provider blames the key it just used
(quota exhausted, credential rejected, model not permitted). An empty
list means nothing usable exists for this provider+account.

The order is: keys the provider hasn't recently complained about first,
then the `is_active` one, then by id. Status is a hint, not a filter: a
key marked `rate_limited` an hour ago is very likely fine now, so it
goes last rather than getting dropped, and one that answers again is
marked `valid` again by record_dispatch_outcome().

Exhaustion is tracked with a clock, not just a flag. A key refused on
quota grounds gets `exhausted_at`, the `exhaustion_kind` the provider
named, and a `retry_at` planned by app/core/key_cooldown.py, and it sorts
behind a key whose cooldown has already elapsed so dispatch spends its
first attempt on the one more likely to answer. recheck_keys() is the
pass that comes back for them, run at startup and on an interval by
app/core/key_refresh.py and from the buttons on /apis. Keys the provider
rejected or blocked outright are never in that pass: no amount of waiting
un-revokes a key, so they wait for someone to ask (scope="blocked").
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.crypto import decrypt, encrypt
from app.core.db import ApiKey, get_db
from app.core.key_cooldown import plan_cooldown, postpone
from app.core.llm_providers import fields_for, validate_credentials

logger = logging.getLogger(__name__)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _as_utc(value: dt.datetime | None) -> dt.datetime | None:
    """SQLite hands back naive datetimes even for a timezone=True column,
    so everything read off a row goes through here before it is compared
    with _now() or serialized. Stored values are always UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=dt.UTC)


def _iso(value: dt.datetime | None) -> str | None:
    aware = _as_utc(value)
    return aware.isoformat() if aware else None


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


# Statuses that mean the key filled a quota window. Temporary by
# definition: the window rolls over, so these are the ones recheck_keys()
# comes back for on its own.
_EXHAUSTED_STATUSES = {"rate_limited"}

# Statuses no waiting fixes: the credential was rejected, or the provider
# has shut the key off. Rechecked only when someone asks
# (recheck_keys(scope="blocked") or the per-key check button).
_MANUAL_ONLY_STATUSES = {"invalid", "blocked"}

# Statuses that mean the provider complained about this key the last
# time it was used. Not a reason to skip the key (quotas reset, and a
# manual check can't observe quota at all), only a reason to try a
# quieter one first.
_DEGRADED_STATUSES = _EXHAUSTED_STATUSES | _MANUAL_ONLY_STATUSES


def _is_due(row: ApiKey, now: dt.datetime) -> bool:
    """Whether an exhausted key's cooldown has elapsed. A missing
    `retry_at` counts as due: that is a key marked exhausted before the
    cooldown columns existed, or by something that didn't plan one, and
    leaving it waiting forever would be worse than one extra check."""
    if row.status not in _EXHAUSTED_STATUSES:
        return False
    retry_at = _as_utc(row.retry_at)
    return retry_at is None or retry_at <= now


def _health_rank(row: ApiKey, now: dt.datetime) -> int:
    """Where a key sits in the dispatch queue, by how likely it is to
    answer: a key nobody has complained about, then one whose cooldown has
    passed, then one still inside its cooldown, then one the provider
    rejected or blocked. Nothing is excluded, so a device with a single
    unhappy key still gets to try it."""
    if row.status in _MANUAL_ONLY_STATUSES:
        return 3
    if row.status in _EXHAUSTED_STATUSES:
        return 1 if _is_due(row, now) else 2
    return 0


def _mark_exhausted(row: ApiKey, detail: str | None, now: dt.datetime) -> None:
    cooldown = plan_cooldown(row.provider, detail, now)
    row.status = "rate_limited"
    row.exhausted_at = now
    row.exhaustion_kind = cooldown.kind
    row.retry_at = cooldown.retry_at


def _clear_exhaustion(row: ApiKey) -> None:
    row.exhausted_at = None
    row.retry_at = None
    row.exhaustion_kind = None


def _to_out(row: ApiKey) -> dict:
    return {
        "id": row.id,
        "provider": row.provider,
        "label": row.label,
        "masked": row.masked_preview,
        "status": row.status,
        "last_checked_at": _iso(row.last_checked_at),
        "last_check_detail": row.last_check_detail,
        "exhausted_at": _iso(row.exhausted_at),
        "retry_at": _iso(row.retry_at),
        "exhaustion_kind": row.exhaustion_kind,
        "auto_rechecked": row.status in _EXHAUSTED_STATUSES,
        "recheck_due": _is_due(row, _now()),
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
        if status in _EXHAUSTED_STATUSES:
            # Stored while out of quota: it goes in with a cooldown so the
            # refresh pass picks it up rather than it sitting there unhappy.
            _mark_exhausted(row, detail, _now())
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
    one, so this never touches encrypted_credentials (replace_credentials()
    does that).
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


def replace_credentials(key_id: int, credentials: dict) -> tuple[dict | None, str]:
    """Swaps the secret behind an existing key in place, keeping its label,
    cap, allow-list and place in rotation. Validated exactly like
    add_key(): a rejected credential leaves the stored one untouched. The
    old status and quota clock belonged to the old secret, so both are
    replaced by the new check's outcome. Raises LookupError for an unknown
    id so the caller can tell that apart from a rejected credential."""
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            raise LookupError(key_id)
        status, detail = validate_credentials(row.provider, credentials)
        if status == "invalid":
            return None, detail
        row.encrypted_credentials = encrypt(json.dumps(credentials))
        row.masked_preview = _build_masked_preview(row.provider, credentials)
        row.status = status
        row.last_checked_at = _now()
        row.last_check_detail = detail
        _clear_exhaustion(row)
        if status in _EXHAUSTED_STATUSES:
            _mark_exhausted(row, detail, _now())
        db.commit()
        db.refresh(row)
        return _to_out(row), detail
    finally:
        db.close()


def set_enabled(key_id: int, enabled: bool) -> dict | None:
    """The manual on/off switch, independent of is_active/delete. A
    disabled key is skipped by resolve_dispatch_keys() entirely, even if
    it's the active one for its provider, so it is not used as failover
    either; re-enabling needs no re-activation."""
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
    never resends them. This is the manual "check" button, so it runs
    whatever the key's current status is, including a blocked one that the
    automatic pass deliberately leaves alone."""
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return None
        _apply_check(row, _now())
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def _apply_check(row: ApiKey, now: dt.datetime) -> None:
    """One credential check, written onto the row, with the cooldown
    bookkeeping that goes with each outcome. Shared by check_key() (one
    key, on request) and recheck_keys() (a whole group).

    The check lists models rather than generating, so a "valid" answer
    proves the credential is live, not that its generation quota has
    refilled (see app/core/llm_providers.py). For a key that was waiting
    on a quota, that is still the moment to stop treating it as dead: its
    cooldown has elapsed, the credential still works, so it goes back into
    normal rotation and the next real call is what confirms the quota. The
    detail says exactly that rather than claiming more than was checked.
    """
    was_exhausted = row.status in _EXHAUSTED_STATUSES
    credentials = json.loads(decrypt(row.encrypted_credentials))
    status, detail = validate_credentials(row.provider, credentials)
    row.last_checked_at = now

    if status == "rate_limited":
        # Still throttled, and the provider just said so again: re-plan the
        # wait from this fresh refusal rather than keeping the old one.
        _mark_exhausted(row, detail, now)
        row.last_check_detail = detail
        return

    if status == "unknown" and was_exhausted:
        # Nothing was learned (the provider was unreachable), so the key
        # stays where it was and we come back later.
        row.retry_at = postpone(now)
        row.last_check_detail = detail
        return

    row.status = status
    row.last_check_detail = detail
    if status != "rate_limited":
        _clear_exhaustion(row)
    if status == "valid" and was_exhausted:
        row.last_check_detail = (
            "Back in rotation: the wait is over and this key still answers. Its quota is assumed "
            "reset; the next real call confirms it. " + detail
        )


def recheck_keys(scope: str = "due") -> list[dict]:
    """Rechecks a group of unhappy keys and returns the ones it touched.

    `scope` picks the group, and the difference between them is the whole
    point of splitting the statuses:

    "due"       every exhausted key whose cooldown has elapsed. This is
                the automatic pass (app/core/key_refresh.py, at startup
                and on an interval) and the "recheck now" button.
    "exhausted" every exhausted key, cooldown elapsed or not, for someone
                who does not want to wait out the timer.
    "blocked"   the rejected and blocked keys, which nothing else ever
                touches on its own. Only ever from an explicit request.

    Keys that are switched off are skipped in every scope: they are not
    dispatched with, so spending a request to learn their status is waste.
    """
    now = _now()
    db = get_db()
    try:
        rows = db.execute(select(ApiKey).where(ApiKey.enabled.is_(True))).scalars().all()
        if scope == "blocked":
            targets = [r for r in rows if r.status in _MANUAL_ONLY_STATUSES]
        elif scope == "exhausted":
            targets = [r for r in rows if r.status in _EXHAUSTED_STATUSES]
        elif scope == "due":
            targets = [r for r in rows if _is_due(r, now)]
        else:
            raise ValueError(f"unknown recheck scope {scope!r}")

        for row in targets:
            try:
                _apply_check(row, now)
            except Exception:
                # One unreadable or unreachable key must not stop the rest
                # of the pass, which often runs unattended at startup.
                logger.exception("could not recheck key id=%s; leaving it as it was", row.id)
        db.commit()
        # db.get re-reads each row after the commit expired it. None means
        # the key was deleted while the pass was running (the /apis page
        # stays usable during the background recheck), so it is dropped
        # from the report rather than reloaded into an error.
        return [_to_out(k) for r in targets if (k := db.get(ApiKey, r.id)) is not None]
    finally:
        db.close()


def _allows_account(row: ApiKey, account_id: int | None) -> bool:
    if not row.allowed_account_ids:  # empty/None = every account allowed
        return True
    if account_id is None:
        return False  # restricted key, no account context to check against
    return account_id in row.allowed_account_ids


def _dispatch_order(row: ApiKey) -> tuple[int, int, int]:
    return (_health_rank(row, _now()), 0 if row.is_active else 1, row.id)


def _resolve_rows(db: Session, provider: str, account_id: int | None) -> list[ApiKey]:
    """Every key that may serve this provider+account, best first. See
    the module docstring for what "best" means."""
    rows = (
        db.execute(
            select(ApiKey).where(ApiKey.provider == provider, ApiKey.enabled.is_(True))
        )
        .scalars()
        .all()
    )
    return sorted((r for r in rows if _allows_account(r, account_id)), key=_dispatch_order)


def _resolve_row(db: Session, provider: str, account_id: int | None) -> ApiKey | None:
    rows = _resolve_rows(db, provider, account_id)
    return rows[0] if rows else None


@dataclass(frozen=True)
class DispatchKey:
    """One usable key, as app/core/llm.py needs it at dispatch time.
    `label` is carried along so a failure can name the key that failed
    without that module having to read the row back."""

    id: int
    label: str
    credentials: dict
    budget_cap_usd: float | None


def resolve_dispatch_keys(provider: str, account_id: int | None) -> list[DispatchKey]:
    """Every key app/core/llm.py may try for this provider+account, in
    the order to try them. Empty means nothing usable is configured:
    either no key at all, or every key for this provider is disabled or
    restricted to other accounts.

    Decrypts each one up front rather than lazily, so dispatch never
    reopens the store mid-request. That is one small decrypt per stored
    key on a provider, and only the keys that could actually serve this
    account.
    """
    db = get_db()
    try:
        return [
            DispatchKey(
                id=row.id,
                label=row.label,
                credentials=json.loads(decrypt(row.encrypted_credentials)),
                budget_cap_usd=row.budget_cap_usd,
            )
            for row in _resolve_rows(db, provider, account_id)
        ]
    finally:
        db.close()


def resolve_dispatch_key(
    provider: str, account_id: int | None
) -> tuple[int, dict, float | None] | None:
    """The single best key as a plain tuple, for callers that only want
    one and don't do failover (the /home status tile, and tests). Real
    dispatch uses resolve_dispatch_keys() and walks the whole list.
    """
    keys = resolve_dispatch_keys(provider, account_id)
    if not keys:
        return None
    first = keys[0]
    return first.id, first.credentials, first.budget_cap_usd


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
            "last_checked_at": _iso(row.last_checked_at),
        }
    finally:
        db.close()


def record_dispatch_outcome(
    key_id: int,
    *,
    ok: bool,
    rate_limited: bool = False,
    blocked: bool = False,
    detail: str | None = None,
    provider_detail: str | None = None,
) -> None:
    """Called by app/core/llm.py right after a real dispatch that used
    this exact key (one of the ids resolve_dispatch_keys() returned). A
    provider failure updates status between manual checks, same as a
    manual 'check status' would.

    This is the only place that learns about quota from a real call, since
    the cheap check cannot observe it, so a `rate_limited` outcome is also
    where the cooldown gets planned. `detail` is the message shown on
    /apis; `provider_detail` is the provider's own untranslated text,
    passed separately because that is what app/core/key_cooldown.py reads
    to work out which window was hit and when it rolls over.

    `blocked` is for the provider refusing the credential itself
    (suspended, revoked, its API not enabled). It is kept apart from
    `invalid` and from a quota because it is the one outcome that will not
    fix itself, so nothing rechecks it until someone asks.

    A success only writes when there is something to undo: a key that
    was marked `rate_limited`, `invalid` or `blocked` and has just
    answered anyway goes back to `valid` with its cooldown cleared,
    because the quota reset or the outage passed and the /apis page
    shouldn't keep showing a dead key that works. A key already `valid`
    needs no write on every call.
    """
    now = _now()
    db = get_db()
    try:
        row = db.get(ApiKey, key_id)
        if row is None:
            return
        if ok:
            if row.status not in _DEGRADED_STATUSES:
                return
            row.status = "valid"
            row.last_checked_at = now
            row.last_check_detail = "Working again: the last request through this key went through."
            _clear_exhaustion(row)
        elif rate_limited:
            _mark_exhausted(row, provider_detail or detail, now)
            row.last_checked_at = now
            if detail:
                row.last_check_detail = detail
        else:
            row.status = "blocked" if blocked else "invalid"
            row.last_checked_at = now
            _clear_exhaustion(row)
            if detail:
                row.last_check_detail = detail
        db.commit()
    finally:
        db.close()

