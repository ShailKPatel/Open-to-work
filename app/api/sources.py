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
from github import BadCredentialsException, RateLimitExceededException, UnknownObjectException
from pydantic import BaseModel
from sqlalchemy import select

from app.api.deps import DbSession
from app.core.db import Account, SyncSource, get_db
from app.ingest.github.cancellation import request_cancel
from app.ingest.github.source_parser import InvalidSourceError, parse_source
from app.ingest.github.sync import sync_account_progress, sync_single_repo_progress

router = APIRouter(prefix="/api/sources")
logger = logging.getLogger(__name__)


class SourceSummary(BaseModel):
    id: int
    raw_input: str
    kind: str
    github_username: str
    repo_full_name: str | None
    last_synced_at: dt.datetime | None

    @classmethod
    def from_row(cls, row: SyncSource) -> SourceSummary:
        return cls(
            id=row.id,
            raw_input=row.raw_input,
            kind=row.kind,
            github_username=row.github_username,
            repo_full_name=row.repo_full_name,
            last_synced_at=row.last_synced_at,
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


def _mark_synced(source_id: int) -> None:
    db = get_db()
    try:
        row = db.get(SyncSource, source_id)
        if row is not None:
            row.last_synced_at = dt.datetime.now(dt.UTC)
            db.commit()
    finally:
        db.close()


@router.get("/{source_id}/sync/stream")
def sync_source_stream(source_id: int, run_id: str | None = None) -> StreamingResponse:
    """run_id should be a fresh id generated by the client for each Sync
    click (not reused across attempts; see sync_account_progress()'s
    docstring for why that matters for the "Stop" button to be reliable).
    Falls back to the source's own id if the caller doesn't supply one,
    which is fine for a one-off call but reintroduces the reuse caveat if
    that source gets synced (and possibly cancelled) more than once."""
    db = get_db()
    try:
        row = db.get(SyncSource, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no source with id={source_id}")
        account = db.get(Account, row.account_id)
        attribution_username = account.github_username if account else row.github_username
        kind = row.kind
        username = row.github_username
        repo_full_name = row.repo_full_name
        account_id = row.account_id
        raw_input = row.raw_input
    finally:
        db.close()

    def events() -> Iterator[str]:
        # Only a real "done" counts as synced, not merely "the generator
        # didn't raise", since a mid-batch rate-limit now stops the
        # generator cleanly (yields "rate_limited", returns) instead of
        # raising. A partial batch should stay retryable, not get marked
        # as if it fully succeeded.
        done = False
        try:
            cancel_key = run_id or str(source_id)
            if kind == "repo":
                assert repo_full_name is not None
                gen = sync_single_repo_progress(
                    repo_full_name,
                    attribution_username,
                    account_id=account_id,
                    run_id=cancel_key,
                )
            else:
                gen = sync_account_progress(
                    username, account_id=account_id, run_id=cancel_key
                )
            for event in gen:
                yield _sse(event)
                if event.get("stage") == "done":
                    done = True
        except UnknownObjectException:
            yield _sse({"stage": "error", "detail": f"'{raw_input}' not found on GitHub"})
        except BadCredentialsException:
            yield _sse(
                {"stage": "error", "detail": "GitHub rejected the configured token/credentials"}
            )
        except RateLimitExceededException:
            yield _sse(
                {"stage": "error", "detail": "GitHub rate limit exhausted; try again shortly"}
            )
        if done:
            _mark_synced(source_id)

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
