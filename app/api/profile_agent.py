"""POST /api/profile-agent/ask: one question about the profile, answered by
the agent in app/agent/profile_agent.py.

Typed LLM failures (no key, budget used up, provider down) are left to
app/api/llm_errors.py's handlers, like every other LLM-backed route.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent import profile_agent
from app.core.db import Account, get_db

router = APIRouter(prefix="/api/profile-agent")

MAX_QUESTION_CHARS = 1000


class AskRequest(BaseModel):
    account_id: int
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    steps: int
    cost_usd: float
    # MCP servers the agent's tools came from, and configured outside ones
    # that could not be reached (app/agent/mcp_hub.py).
    servers: list[str] = []
    unavailable: list[str] = []


def _account_exists(account_id: int) -> bool:
    db = get_db()
    try:
        return db.get(Account, account_id) is not None
    finally:
        db.close()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Ask a question first.")
    if not await run_in_threadpool(_account_exists, body.account_id):
        raise HTTPException(status_code=404, detail=f"no profile with id={body.account_id}")
    result = await profile_agent.ask(body.account_id, question)
    return AskResponse(
        answer=result.answer,
        tools_used=result.tools_used,
        steps=result.steps,
        cost_usd=result.cost_usd,
        servers=result.servers,
        unavailable=result.unavailable,
    )
