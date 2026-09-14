"""Encrypted credential store for AuthSource rows (app/core/db.py): the
login profiles app/ingest/jobs/auth_fetch.py uses to fetch a job posting
from a login-walled site via a real automated browser login. Same
shape/reasoning as app/core/api_keys_store.py: nothing outside this module
ever decrypts credentials for display, only resolve_credentials() (called
by auth_fetch.py right before a real login attempt) gets the real
username/password back.

Intended for a person's own job search with their own credentials. Every
row requires acknowledged_risk=True at creation, checked here rather than
only in a UI checkbox that could be bypassed by calling the API directly.
"""

from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import select

from app.core.crypto import decrypt, encrypt
from app.core.db import AuthSource, get_db


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _mask_username(username: str) -> str:
    username = username.strip()
    if len(username) <= 4:
        return "•" * max(len(username), 3)
    return f"{username[:2]}{'•' * 6}{username[-2:]}"


def _to_out(row: AuthSource) -> dict:
    return {
        "id": row.id,
        "account_id": row.account_id,
        "label": row.label,
        "site_domain": row.site_domain,
        "login_url": row.login_url,
        "username_selector": row.username_selector,
        "password_selector": row.password_selector,
        "submit_selector": row.submit_selector,
        "post_login_wait_selector": row.post_login_wait_selector,
        "masked_username": row.masked_username,
        "enabled": row.enabled,
        "status": row.status,
        "last_checked_at": row.last_checked_at.isoformat() if row.last_checked_at else None,
        "last_check_detail": row.last_check_detail,
        "created_at": row.created_at.isoformat(),
    }


def list_sources(account_id: int) -> list[dict]:
    db = get_db()
    try:
        rows = (
            db.execute(
                select(AuthSource)
                .where(AuthSource.account_id == account_id)
                .order_by(AuthSource.id)
            )
            .scalars()
            .all()
        )
        return [_to_out(r) for r in rows]
    finally:
        db.close()


def add_source(
    *,
    account_id: int,
    label: str,
    site_domain: str,
    login_url: str,
    username_selector: str,
    password_selector: str,
    submit_selector: str,
    post_login_wait_selector: str | None,
    username: str,
    password: str,
    acknowledged_risk: bool,
) -> dict:
    if not acknowledged_risk:
        raise ValueError("acknowledged_risk must be true to store an authenticated source")
    for field_name, value in (
        ("label", label),
        ("site_domain", site_domain),
        ("login_url", login_url),
        ("username_selector", username_selector),
        ("password_selector", password_selector),
        ("submit_selector", submit_selector),
        ("username", username),
        ("password", password),
    ):
        if not value.strip():
            raise ValueError(f"{field_name} is required")

    db = get_db()
    try:
        row = AuthSource(
            account_id=account_id,
            label=label.strip(),
            site_domain=site_domain.strip(),
            login_url=login_url.strip(),
            username_selector=username_selector.strip(),
            password_selector=password_selector.strip(),
            submit_selector=submit_selector.strip(),
            post_login_wait_selector=(post_login_wait_selector or "").strip() or None,
            encrypted_credentials=encrypt(json.dumps({"username": username, "password": password})),
            masked_username=_mask_username(username),
            acknowledged_risk=True,
            enabled=True,
            status="unknown",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def set_enabled(source_id: int, enabled: bool) -> dict | None:
    db = get_db()
    try:
        row = db.get(AuthSource, source_id)
        if row is None:
            return None
        row.enabled = enabled
        db.commit()
        db.refresh(row)
        return _to_out(row)
    finally:
        db.close()


def delete_source(source_id: int) -> None:
    db = get_db()
    try:
        row = db.get(AuthSource, source_id)
        if row is None:
            return
        db.delete(row)
        db.commit()
    finally:
        db.close()


def record_check_outcome(source_id: int, *, ok: bool, detail: str) -> None:
    db = get_db()
    try:
        row = db.get(AuthSource, source_id)
        if row is None:
            return
        row.status = "valid" if ok else "invalid"
        row.last_checked_at = _now()
        row.last_check_detail = detail
        db.commit()
    finally:
        db.close()


def resolve_credentials(source_id: int) -> tuple[AuthSource, dict] | None:
    """(row, {"username", "password"}) for a real fetch attempt
    (app/ingest/jobs/auth_fetch.py): the one place in this codebase that
    gets a real decrypted credential back for this table. None if the
    source doesn't exist or is disabled.
    """
    db = get_db()
    try:
        row = db.get(AuthSource, source_id)
        if row is None or not row.enabled:
            return None
        credentials = json.loads(decrypt(row.encrypted_credentials))
        return row, credentials
    finally:
        db.close()
