"""The GitHub sync reminder: syncs that are running or were cut off part
way, for the widget in the page header, plus its Resume and Dismiss
buttons. The syncs themselves live in app/ingest/github/background.py.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ingest.github import background

router = APIRouter(prefix="/api/github-sync")


class PendingSync(BaseModel):
    id: int
    kind: str
    label: str
    state: str
    running: bool
    completed: int
    total_hint: int | None
    detail: str | None
    reset_at: dt.datetime | None
    resume_at: dt.datetime | None


@router.get("", response_model=list[PendingSync])
def list_pending(account_id: int) -> list[PendingSync]:
    return [PendingSync(**run) for run in background.pending_runs(account_id)]


@router.post("/{run_row_id}/resume")
def resume_sync(run_row_id: int) -> dict:
    """Runs the sync again now, without waiting for the automatic resume.
    Repos already saved are cache hits, so it continues where it stopped."""
    run_id = background.resume(run_row_id)
    if run_id is None:
        raise HTTPException(status_code=404, detail=f"no sync with id={run_row_id}")
    return {"resumed": True, "run_id": run_id}


@router.post("/{run_row_id}/dismiss")
def dismiss_sync(run_row_id: int) -> dict:
    if not background.dismiss(run_row_id):
        raise HTTPException(status_code=404, detail=f"no sync with id={run_row_id}")
    return {"dismissed": True}
