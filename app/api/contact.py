"""Contact info and social links: resume-header material that isn't part
of account creation itself (see app/api/accounts.py) and doesn't belong
crammed into that router as it grows. Kept as its own small router,
same reasoning app/api/sources.py gives for staying separate
from accounts.py: one clear resource per file, not a shared mega-file of
everything account-adjacent.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from app.core.db import Account, SocialLink, get_db

router = APIRouter(prefix="/api/accounts")


class ContactInfo(BaseModel):
    account_id: int
    first_name: str
    last_name: str
    contact_email: str | None
    contact_phone: str | None
    contact_location: str | None
    # Account.github_username is identity, not something this router owns
    # (see app/api/sources.py's SyncSource docstring), but the contact page
    # wants to show it as a quick "GitHub ↗" link alongside the social
    # links it does own, so it's surfaced here read-only.
    github_username: str | None

    @classmethod
    def from_account(cls, account: Account) -> ContactInfo:
        return cls(
            account_id=account.id,
            first_name=account.first_name,
            last_name=account.last_name,
            contact_email=account.contact_email,
            contact_phone=account.contact_phone,
            contact_location=account.contact_location,
            github_username=account.github_username or None,
        )


@router.get("/{account_id}/contact", response_model=ContactInfo)
def get_contact(account_id: int) -> ContactInfo:
    db = get_db()
    try:
        account = db.get(Account, account_id)
        if account is None:
            raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
        return ContactInfo.from_account(account)
    finally:
        db.close()


class ContactUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    contact_location: str | None = None


@router.patch("/{account_id}/contact", response_model=ContactInfo)
def update_contact(account_id: int, body: ContactUpdate) -> ContactInfo:
    fields = body.model_dump(exclude_unset=True)
    for key in ("contact_email", "contact_phone", "contact_location"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip() or None
    for key in ("first_name", "last_name"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
            if not fields[key]:
                raise HTTPException(status_code=422, detail=f"{key} is required")

    db = get_db()
    try:
        account = db.get(Account, account_id)
        if account is None:
            raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
        for key, value in fields.items():
            setattr(account, key, value)
        db.commit()
        db.refresh(account)
        return ContactInfo.from_account(account)
    finally:
        db.close()


class SocialLinkItem(BaseModel):
    id: int
    platform: str
    url: str
    label: str | None

    @classmethod
    def from_row(cls, row: SocialLink) -> SocialLinkItem:
        return cls(id=row.id, platform=row.platform, url=row.url, label=row.label)


@router.get("/{account_id}/social-links", response_model=list[SocialLinkItem])
def list_social_links(account_id: int) -> list[SocialLinkItem]:
    db = get_db()
    try:
        rows = db.execute(
            select(SocialLink)
            .where(SocialLink.account_id == account_id)
            .order_by(SocialLink.created_at)
        ).scalars()
        return [SocialLinkItem.from_row(r) for r in rows]
    finally:
        db.close()


class SocialLinkCreate(BaseModel):
    platform: str
    url: str
    label: str | None = None


@router.post("/{account_id}/social-links", response_model=SocialLinkItem)
def add_social_link(account_id: int, body: SocialLinkCreate) -> SocialLinkItem:
    platform = body.platform.strip()
    url = body.url.strip()
    if not platform:
        raise HTTPException(status_code=422, detail="platform is required")
    if not url:
        raise HTTPException(status_code=422, detail="url is required")

    db = get_db()
    try:
        account = db.get(Account, account_id)
        if account is None:
            raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
        row = SocialLink(
            account_id=account_id,
            platform=platform,
            url=url,
            label=(body.label.strip() if body.label else None) or None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return SocialLinkItem.from_row(row)
    finally:
        db.close()


class SocialLinkUpdate(BaseModel):
    platform: str | None = None
    url: str | None = None
    label: str | None = None


@router.patch("/{account_id}/social-links/{link_id}", response_model=SocialLinkItem)
def update_social_link(account_id: int, link_id: int, body: SocialLinkUpdate) -> SocialLinkItem:
    fields = body.model_dump(exclude_unset=True)
    for key in ("platform", "url"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
            if not fields[key]:
                raise HTTPException(status_code=422, detail=f"{key} is required")

    db = get_db()
    try:
        row = db.get(SocialLink, link_id)
        if row is None or row.account_id != account_id:
            raise HTTPException(status_code=404, detail=f"no social link with id={link_id}")
        for key, value in fields.items():
            setattr(row, key, value)
        db.commit()
        db.refresh(row)
        return SocialLinkItem.from_row(row)
    finally:
        db.close()


@router.delete("/{account_id}/social-links/{link_id}")
def delete_social_link(account_id: int, link_id: int) -> dict:
    db = get_db()
    try:
        row = db.get(SocialLink, link_id)
        if row is None or row.account_id != account_id:
            raise HTTPException(status_code=404, detail=f"no social link with id={link_id}")
        db.delete(row)
        db.commit()
        return {"deleted": True, "id": link_id}
    finally:
        db.close()
