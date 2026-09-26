"""Step tracking for the multi-step flows that call app/core/llm.py more
than once (the resume build: tailor the content, then fit it to the page
count).

Why this exists: when such a flow stops partway, "resume generation
failed: <provider message>" doesn't say which step stopped, what had
already finished, or whether re-running costs that work again. This adds
the missing half of that sentence, and nothing else. The heavy lifting is
elsewhere: app/core/llm.py switches keys under a step so a dying key
doesn't stop the flow at all, and each step commits its own work, so the
only thing left to say is where it stopped.

The re-raised exception keeps the original's type, so every caller that
maps an LLM error to an HTTP status (app/api/resume_build.py's
_map_llm_error) keeps mapping it the same way; only the message grows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMDispatchError,
)
from app.core.rate_limits import record_event

logger = logging.getLogger(__name__)

# Only failures that are about reaching a model at all get a step label.
# A bug in a step (a ValueError, a missing row) is not something "which
# step, and what already finished" helps with, and wrapping it would hide
# the traceback that does help.
_ANNOTATED = (ApiKeyMissingError, BudgetExceededError, LLMDispatchError)


class Pipeline:
    """One run of a multi-step flow. `stage()` wraps each step; steps that
    complete are remembered by name so a later failure can list them.

    Not persistence: what a finished step actually saved is that step's
    own business (a repo's status row, a committed resume). This only
    reports.
    """

    def __init__(self, name: str, account_id: int | None = None) -> None:
        self.name = name
        self.account_id = account_id
        self.completed: list[str] = []

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        logger.info("%s: starting %s", self.name, label)
        try:
            yield
        except _ANNOTATED as e:
            detail = self._stopped_detail(label, e)
            logger.warning("%s", detail)
            record_event(
                "llm", "stage_failed", detail, context=self.name, account_id=self.account_id
            )
            raise type(e)(detail) from e
        self.completed.append(label)
        logger.info("%s: finished %s", self.name, label)

    def _stopped_detail(self, label: str, error: Exception) -> str:
        if self.completed:
            done = "Already finished: " + ", ".join(self.completed) + "."
        else:
            done = "Nothing had finished yet."
        return f"{self.name} stopped at \"{label}\". {error} {done}"
