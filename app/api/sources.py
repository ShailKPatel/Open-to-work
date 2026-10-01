"""Sync sources: the "fetch data from" list. Kept separate from
Account.github_username, which stays pure identity (who this profile is,
used for commit attribution); a SyncSource is where evidence gets
pulled from, and an account can have any number of them.

Accepts a bare username, a github.com profile URL, or a github.com repo
URL; see app/ingest/github/source_parser.py for what's recognized. A
username/profile-URL source syncs the whole account (sync_account_progress,
already built for /sync); a repo-URL source syncs just that one project
(sync_single_repo_progress), for something contributed to but not owned,
where pulling the owner's entire account would mix in repos that aren't
this person's.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import DbSession
from app.core.db import Account, SyncSource, get_db
from app.ingest.github import background
from app.ingest.github.cancellation import request_cancel
from app.ingest.github.source_parser import InvalidSourceError, parse_source

router = APIRouter(prefix="/api/sources")
logger = logging.getLogger(__name__)


class SourceSummary(BaseModel):
    id: int
    raw_input: str
    kind: str
    github_username: str
    repo_full_name: str | None
    last_synced_at: dt.datetime | None
    # a background sync of this target is in flight, possibly started
    # from another page; the fetch-data page reattaches to it on load
    syncing: bool = False

    @classmethod
    def from_row(cls, row: SyncSource) -> SourceSummary:
        return cls(
            id=row.id,
            raw_input=row.raw_input,
            kind=row.kind,
            github_username=row.github_username,
            repo_full_name=row.repo_full_name,
            last_synced_at=row.last_synced_at,
            syncing=background.is_running(
                background.repo_key(row.repo_full_name)
                if row.kind == "repo" and row.repo_full_name
                else background.account_key(row.github_username)
            ),
        )


@router.get("", response_model=list[SourceSummary])
def list_sources(account_id: int, *, db: DbSession) -> list[SourceSummary]:
    rows = list(
        db.execute(
            select(SyncSource)
            .where(SyncSource.account_id == account_id)
            .order_by(SyncSource.created_at)
        ).scalars()
    )
    return [SourceSummary.from_row(r) for r in rows]


class CreateSourceRequest(BaseModel):
    account_id: int
    raw_input: str


@router.post("", response_model=SourceSummary)
def create_source(payload: CreateSourceRequest, *, db: DbSession) -> SourceSummary:
    account = db.get(Account, payload.account_id)
    if account is None:
        raise HTTPException(
            status_code=404, detail=f"no account with id={payload.account_id}"
        )

    try:
        parsed = parse_source(payload.raw_input)
    except InvalidSourceError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Same account, same kind, same target: reject rather than
    # silently create a second tile that just duplicates the first
    # one's work. Compared on the *parsed* target, not raw_input, so
    # "octocat" and "https://github.com/octocat" collide too.
    existing = db.execute(
        select(SyncSource).where(
            SyncSource.account_id == payload.account_id,
            SyncSource.kind == parsed.kind,
            SyncSource.github_username == parsed.username,
            SyncSource.repo_full_name == parsed.repo_full_name,
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=409, detail="This is already in your fetch-data list."
        )

    row = SyncSource(
        account_id=payload.account_id,
        raw_input=payload.raw_input.strip(),
        kind=parsed.kind,
        github_username=parsed.username,
        repo_full_name=parsed.repo_full_name,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return SourceSummary.from_row(row)


@router.delete("/{source_id}")
def delete_source(source_id: int, *, db: DbSession) -> dict:
    """Removes it from the fetch list only; does not touch any repos or
    skill evidence a previous sync of it already pulled in. Those stay
    with the account like anything else synced; delete-account is still
    the only "erase everything" action.
    """
    row = db.get(SyncSource, source_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no source with id={source_id}")
    db.delete(row)
    db.commit()
    return {"deleted": True, "source_id": source_id}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _target(db: Session, row: SyncSource) -> background.SyncTarget:
    account = db.get(Account, row.account_id)
    return background.SyncTarget(
        kind=row.kind,
        github_username=row.github_username,
        # commits are credited to the profile's own login, not a repo's owner
        attribution_username=account.github_username if account else row.github_username,
        account_id=row.account_id,
        repo_full_name=row.repo_full_name if row.kind == "repo" else None,
    )


@router.post("/sync-all")
def sync_all_sources(account_id: int, *, db: DbSession) -> dict:
    """Syncs every source of this account in turn, server-side
    (app/ingest/github/background.py's start_all), so leaving the page
    doesn't stop the list part way. started=False: already going."""
    rows = list(
        db.execute(
            select(SyncSource)
            .where(SyncSource.account_id == account_id)
            .order_by(SyncSource.created_at)
        ).scalars()
    )
    targets = [_target(db, row) for row in rows]
    return {"started": background.start_all(account_id, targets), "sources": len(targets)}


@router.get("/sync-all/status")
def sync_all_status(account_id: int) -> dict:
    return {"running": background.is_running(background.all_sources_key(account_id))}


@router.get("/{source_id}/sync/stream")
def sync_source_stream(
    source_id: int, run_id: str | None = None, attach: bool = False
) -> StreamingResponse:
    """Starts this source's sync in the background (or joins the one
    already running for the same target) and streams its progress. The
    sync does not depend on this connection: closing it stops the
    progress updates, not the fetching. attach=true only follows a sync
    that is already running, for a page coming back to one.

    run_id should be a fresh id generated by the client for each Sync
    click (see sync_account_progress()'s docstring for why). Every event
    carries the run_id of the sync actually running, which differs from
    the one sent when this call joined an earlier sync; Stop must use the
    one from the events."""
    db = get_db()
    try:
        row = db.get(SyncSource, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no source with id={source_id}")
        target = _target(db, row)
    finally:
        db.close()

    key = target.key
    if not attach:
        background.start_sync(target, run_id or str(source_id))

    def events() -> Iterator[str]:
        for event in background.follow(key):
            yield _sse(event)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/{source_id}/sync/cancel")
def cancel_source_sync(source_id: int, run_id: str | None = None) -> dict:
    """Flags this source's in-flight sync to stop before its next repo;
    see app/ingest/github/cancellation.py. Pass the SAME run_id the
    matching GET .../sync/stream call used (falls back to the source's
    own id if that call didn't supply one either, see that endpoint's
    docstring). No error if nothing is running: cancelling
    something that isn't in flight is a no-op, not a failure.
    """
    cancel_key = run_id or str(source_id)
    request_cancel(cancel_key)
    return {"cancel_requested": True, "source_id": source_id}
