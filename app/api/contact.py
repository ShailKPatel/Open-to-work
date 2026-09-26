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

from app.api.deps import DbSession
from app.core.db import Account, ContactEmail, ContactPhone, SocialLink

router = APIRouter(prefix="/api/accounts")


class ContactEmailItem(BaseModel):
    id: int
    email: str
    is_primary: bool

    @classmethod
    def from_row(cls, row: ContactEmail) -> ContactEmailItem:
        return cls(id=row.id, email=row.email, is_primary=row.is_primary)


class ContactPhoneItem(BaseModel):
    id: int
    phone: str
    is_primary: bool

    @classmethod
    def from_row(cls, row: ContactPhone) -> ContactPhoneItem:
        return cls(id=row.id, phone=row.phone, is_primary=row.is_primary)


class ContactInfo(BaseModel):
    account_id: int
    first_name: str
    last_name: str
    contact_email: str | None
    contact_phone: str | None
    contact_location: str | None
    emails: list[ContactEmailItem] = []
    phones: list[ContactPhoneItem] = []
    # Account.github_username is identity, not something this router owns
    # (see app/api/sources.py's SyncSource docstring), but the contact page
    # wants to show it as a quick "GitHub ↗" link alongside the social
    # links it does own, so it's surfaced here read-only.
    github_username: str | None

    @classmethod
    def from_account(
        cls,
        account: Account,
        emails: list[ContactEmail] | None = None,
        phones: list[ContactPhone] | None = None,
    ) -> ContactInfo:
        email_items = [ContactEmailItem.from_row(e) for e in (emails or [])]
        phone_items = [ContactPhoneItem.from_row(p) for p in (phones or [])]

        primary_email = next(
            (e.email for e in (emails or []) if e.is_primary), account.contact_email
        )
        if not primary_email and email_items:
            primary_email = email_items[0].email

        primary_phone = next(
            (p.phone for p in (phones or []) if p.is_primary), account.contact_phone
        )
        if not primary_phone and phone_items:
            primary_phone = phone_items[0].phone

        return cls(
            account_id=account.id,
            first_name=account.first_name,
            last_name=account.last_name,
            contact_email=primary_email,
            contact_phone=primary_phone,
            contact_location=account.contact_location,
            emails=email_items,
            phones=phone_items,
            github_username=account.github_username or None,
        )


@router.get("/{account_id}/contact", response_model=ContactInfo)
def get_contact(account_id: int, *, db: DbSession) -> ContactInfo:
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
    emails = list(
        db.execute(
            select(ContactEmail)
            .where(ContactEmail.account_id == account_id)
            .order_by(ContactEmail.created_at)
        ).scalars()
    )
    phones = list(
        db.execute(
            select(ContactPhone)
            .where(ContactPhone.account_id == account_id)
            .order_by(ContactPhone.created_at)
        ).scalars()
    )
    return ContactInfo.from_account(account, emails=emails, phones=phones)


class ContactUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    contact_location: str | None = None


@router.patch("/{account_id}/contact", response_model=ContactInfo)
def update_contact(account_id: int, body: ContactUpdate, *, db: DbSession) -> ContactInfo:
    fields = body.model_dump(exclude_unset=True)
    for key in ("contact_email", "contact_phone", "contact_location"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip() or None
    for key in ("first_name", "last_name"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
            if not fields[key]:
                raise HTTPException(status_code=422, detail=f"{key} is required")

    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")
    for key, value in fields.items():
        setattr(account, key, value)

    # Sync single contact_email into contact_emails table if missing
    if "contact_email" in fields:
        new_email = fields["contact_email"]
        if new_email:
            existing_email = db.execute(
                select(ContactEmail).where(
                    ContactEmail.account_id == account_id,
                    ContactEmail.email == new_email,
                )
            ).scalar_one_or_none()
            if not existing_email:
                # Clear primary flag on others. Done through the ORM
                # rather than a bulk UPDATE so already-loaded rows in
                # this session see the change too.
                for e in db.execute(
                    select(ContactEmail).where(ContactEmail.account_id == account_id)
                ).scalars():
                    e.is_primary = False
                db.add(ContactEmail(account_id=account_id, email=new_email, is_primary=True))

    if "contact_phone" in fields:
        new_phone = fields["contact_phone"]
        if new_phone:
            existing_phone = db.execute(
                select(ContactPhone).where(
                    ContactPhone.account_id == account_id,
                    ContactPhone.phone == new_phone,
                )
            ).scalar_one_or_none()
            if not existing_phone:
                for p in db.execute(
                    select(ContactPhone).where(ContactPhone.account_id == account_id)
                ).scalars():
                    p.is_primary = False
                db.add(ContactPhone(account_id=account_id, phone=new_phone, is_primary=True))

    db.commit()
    db.refresh(account)
    emails = list(
        db.execute(
            select(ContactEmail)
            .where(ContactEmail.account_id == account_id)
            .order_by(ContactEmail.created_at)
        ).scalars()
    )
    phones = list(
        db.execute(
            select(ContactPhone)
            .where(ContactPhone.account_id == account_id)
            .order_by(ContactPhone.created_at)
        ).scalars()
    )
    return ContactInfo.from_account(account, emails=emails, phones=phones)



class SocialLinkItem(BaseModel):
    id: int
    platform: str
    url: str
    label: str | None

    @classmethod
    def from_row(cls, row: SocialLink) -> SocialLinkItem:
        return cls(id=row.id, platform=row.platform, url=row.url, label=row.label)


@router.get("/{account_id}/social-links", response_model=list[SocialLinkItem])
def list_social_links(account_id: int, *, db: DbSession) -> list[SocialLinkItem]:
    rows = db.execute(
        select(SocialLink)
        .where(SocialLink.account_id == account_id)
        .order_by(SocialLink.created_at)
    ).scalars()
    return [SocialLinkItem.from_row(r) for r in rows]


class SocialLinkCreate(BaseModel):
    platform: str
    url: str
    label: str | None = None


@router.post("/{account_id}/social-links", response_model=SocialLinkItem)
def add_social_link(account_id: int, body: SocialLinkCreate, *, db: DbSession) -> SocialLinkItem:
    platform = body.platform.strip()
    url = body.url.strip()
    if not platform:
        raise HTTPException(status_code=422, detail="platform is required")
    if not url:
        raise HTTPException(status_code=422, detail="url is required")

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


class SocialLinkUpdate(BaseModel):
    platform: str | None = None
    url: str | None = None
    label: str | None = None


@router.patch("/{account_id}/social-links/{link_id}", response_model=SocialLinkItem)
def update_social_link(
    account_id: int,
    link_id: int,
    body: SocialLinkUpdate,
    *,
    db: DbSession,
) -> SocialLinkItem:
    fields = body.model_dump(exclude_unset=True)
    for key in ("platform", "url"):
        if key in fields and fields[key] is not None:
            fields[key] = fields[key].strip()
            if not fields[key]:
                raise HTTPException(status_code=422, detail=f"{key} is required")

    row = db.get(SocialLink, link_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no social link with id={link_id}")
    for key, value in fields.items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return SocialLinkItem.from_row(row)


@router.delete("/{account_id}/social-links/{link_id}")
def delete_social_link(account_id: int, link_id: int, *, db: DbSession) -> dict:
    row = db.get(SocialLink, link_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no social link with id={link_id}")
    db.delete(row)
    db.commit()
    return {"deleted": True, "id": link_id}


class ContactEmailCreate(BaseModel):
    email: str
    is_primary: bool = False


@router.get("/{account_id}/emails", response_model=list[ContactEmailItem])
def list_contact_emails(account_id: int, *, db: DbSession) -> list[ContactEmailItem]:
    rows = list(
        db.execute(
            select(ContactEmail)
            .where(ContactEmail.account_id == account_id)
            .order_by(ContactEmail.created_at)
        ).scalars()
    )
    return [ContactEmailItem.from_row(r) for r in rows]


@router.post("/{account_id}/emails", response_model=ContactEmailItem)
def add_contact_email(
    account_id: int,
    body: ContactEmailCreate,
    *,
    db: DbSession,
) -> ContactEmailItem:
    email = body.email.strip()
    if not email:
        raise HTTPException(status_code=422, detail="email is required")

    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")

    existing = list(
        db.execute(select(ContactEmail).where(ContactEmail.account_id == account_id)).scalars()
    )
    is_first = len(existing) == 0
    make_primary = body.is_primary or is_first

    if make_primary:
        for e in existing:
            e.is_primary = False
        account.contact_email = email

    row = ContactEmail(account_id=account_id, email=email, is_primary=make_primary)
    db.add(row)
    db.commit()
    db.refresh(row)
    return ContactEmailItem.from_row(row)


class ContactEmailUpdate(BaseModel):
    email: str | None = None
    is_primary: bool | None = None


@router.patch("/{account_id}/emails/{email_id}", response_model=ContactEmailItem)
def update_contact_email(
    account_id: int, email_id: int, body: ContactEmailUpdate
,
    *,
    db: DbSession,) -> ContactEmailItem:
    row = db.get(ContactEmail, email_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no contact email with id={email_id}")

    if body.email is not None:
        email = body.email.strip()
        if not email:
            raise HTTPException(status_code=422, detail="email is required")
        row.email = email

    if body.is_primary is True:
        for e in db.execute(
            select(ContactEmail).where(ContactEmail.account_id == account_id)
        ).scalars():
            e.is_primary = False
        row.is_primary = True
        account = db.get(Account, account_id)
        if account:
            account.contact_email = row.email

    db.commit()
    db.refresh(row)
    return ContactEmailItem.from_row(row)


@router.delete("/{account_id}/emails/{email_id}")
def delete_contact_email(account_id: int, email_id: int, *, db: DbSession) -> dict:
    row = db.get(ContactEmail, email_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no contact email with id={email_id}")
    was_primary = row.is_primary
    db.delete(row)
    db.commit()

    if was_primary:
        remaining = list(
            db.execute(
                select(ContactEmail)
                .where(ContactEmail.account_id == account_id)
                .order_by(ContactEmail.created_at)
            ).scalars()
        )
        account = db.get(Account, account_id)
        if remaining:
            remaining[0].is_primary = True
            if account:
                account.contact_email = remaining[0].email
        elif account:
            account.contact_email = None
        db.commit()

    return {"deleted": True, "id": email_id}


class ContactPhoneCreate(BaseModel):
    phone: str
    is_primary: bool = False


@router.get("/{account_id}/phones", response_model=list[ContactPhoneItem])
def list_contact_phones(account_id: int, *, db: DbSession) -> list[ContactPhoneItem]:
    rows = list(
        db.execute(
            select(ContactPhone)
            .where(ContactPhone.account_id == account_id)
            .order_by(ContactPhone.created_at)
        ).scalars()
    )
    return [ContactPhoneItem.from_row(r) for r in rows]


@router.post("/{account_id}/phones", response_model=ContactPhoneItem)
def add_contact_phone(
    account_id: int,
    body: ContactPhoneCreate,
    *,
    db: DbSession,
) -> ContactPhoneItem:
    phone = body.phone.strip()
    if not phone:
        raise HTTPException(status_code=422, detail="phone is required")

    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"no account with id={account_id}")

    existing = list(
        db.execute(select(ContactPhone).where(ContactPhone.account_id == account_id)).scalars()
    )
    is_first = len(existing) == 0
    make_primary = body.is_primary or is_first

    if make_primary:
        for p in existing:
            p.is_primary = False
        account.contact_phone = phone

    row = ContactPhone(account_id=account_id, phone=phone, is_primary=make_primary)
    db.add(row)
    db.commit()
    db.refresh(row)
    return ContactPhoneItem.from_row(row)


class ContactPhoneUpdate(BaseModel):
    phone: str | None = None
    is_primary: bool | None = None


@router.patch("/{account_id}/phones/{phone_id}", response_model=ContactPhoneItem)
def update_contact_phone(
    account_id: int, phone_id: int, body: ContactPhoneUpdate
,
    *,
    db: DbSession,) -> ContactPhoneItem:
    row = db.get(ContactPhone, phone_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no contact phone with id={phone_id}")

    if body.phone is not None:
        phone = body.phone.strip()
        if not phone:
            raise HTTPException(status_code=422, detail="phone is required")
        row.phone = phone

    if body.is_primary is True:
        for p in db.execute(
            select(ContactPhone).where(ContactPhone.account_id == account_id)
        ).scalars():
            p.is_primary = False
        row.is_primary = True
        account = db.get(Account, account_id)
        if account:
            account.contact_phone = row.phone

    db.commit()
    db.refresh(row)
    return ContactPhoneItem.from_row(row)


@router.delete("/{account_id}/phones/{phone_id}")
def delete_contact_phone(account_id: int, phone_id: int, *, db: DbSession) -> dict:
    row = db.get(ContactPhone, phone_id)
    if row is None or row.account_id != account_id:
        raise HTTPException(status_code=404, detail=f"no contact phone with id={phone_id}")
    was_primary = row.is_primary
    db.delete(row)
    db.commit()

    if was_primary:
        remaining = list(
            db.execute(
                select(ContactPhone)
                .where(ContactPhone.account_id == account_id)
                .order_by(ContactPhone.created_at)
            ).scalars()
        )
        account = db.get(Account, account_id)
        if remaining:
            remaining[0].is_primary = True
            if account:
                account.contact_phone = remaining[0].phone
        elif account:
            account.contact_phone = None
        db.commit()

    return {"deleted": True, "id": phone_id}

