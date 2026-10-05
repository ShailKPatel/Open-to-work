"""How an LLM failure reaches the browser: JSON with the usual `detail`
plus an `error_kind` (app/core/llm.py's error_kind()) the pages turn into
a plain message and a link to Manage APIs. One mapping for every route,
and a handler for any typed LLM error a route lets through, so none of
them ever comes back as a bare 500.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.app_settings import get_llm_settings
from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMDispatchError,
    error_kind,
    llm_health,
)

_STATUS = {
    "no_key": 422,
    "budget": 402,
    "provider_unavailable": 503,
    "provider_rejected": 502,
}


class LLMRequestError(HTTPException):
    def __init__(self, status_code: int, detail: str, kind: str | None) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_kind = kind


def llm_http_error(e: Exception, fallback: str = "the AI request failed") -> LLMRequestError:
    """The HTTP error a route raises for an exception from an LLM call.
    Anything that is not a typed LLM error becomes a 502 with `fallback`
    in front of its message."""
    kind = error_kind(e)
    if kind is None:
        return LLMRequestError(502, f"{fallback}: {e}", None)
    return LLMRequestError(_STATUS[kind], str(e), kind)


def failing_model() -> str:
    """The model the person should know is down: the one that last failed,
    or the quality-tier model when nothing has been recorded."""
    return llm_health()["model"] or get_llm_settings().model_for("quality")


def _body(detail: str, kind: str | None) -> dict:
    body: dict = {"detail": detail, "error_kind": kind}
    if kind is not None:
        body["model"] = failing_model()
    return body


def install(app: FastAPI) -> None:
    async def on_llm_request_error(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, LLMRequestError)
        return JSONResponse(
            _body(str(exc.detail), exc.error_kind),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    async def on_llm_error(_: Request, exc: Exception) -> JSONResponse:
        mapped = llm_http_error(exc)
        return JSONResponse(_body(str(mapped.detail), mapped.error_kind), mapped.status_code)

    app.add_exception_handler(LLMRequestError, on_llm_request_error)
    for error_type in (ApiKeyMissingError, BudgetExceededError, LLMDispatchError):
        app.add_exception_handler(error_type, on_llm_error)
