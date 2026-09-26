"""app/core/pipeline.py: the half of a failure message that says where a
multi-step run stopped.

The other half (not retried here) is app/core/llm.py's key failover,
which is what usually stops a step from failing in the first place. What
matters here is that when a run does stop, the message names the step and
what had already finished, and that it stays the same kind of exception
so the API routes keep mapping it to the status they always did.
"""

import os
from pathlib import Path

import pytest

import app.core.db as db_module
from app.core.db import init_db
from app.core.llm import ApiKeyMissingError, BudgetExceededError, LLMRateLimitedError
from app.core.pipeline import Pipeline
from app.core.settings import get_settings


def _reset_db(tmp_path: Path) -> None:
    db_module.reset_engine()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    get_settings.cache_clear()
    init_db()


def test_a_stop_names_the_step_and_what_already_finished(tmp_path):
    _reset_db(tmp_path)
    run = Pipeline("The resume build")

    with run.stage("tailoring the content to the job"):
        pass

    with pytest.raises(LLMRateLimitedError) as excinfo:
        with run.stage("fitting the resume to the page count"):
            raise LLMRateLimitedError("Every OpenAI key is out of quota.")

    message = str(excinfo.value)
    assert 'The resume build stopped at "fitting the resume to the page count".' in message
    assert "Every OpenAI key is out of quota." in message
    assert "Already finished: tailoring the content to the job." in message


def test_a_stop_in_the_first_step_says_so_rather_than_listing_nothing(tmp_path):
    _reset_db(tmp_path)
    run = Pipeline("The resume build")

    with pytest.raises(ApiKeyMissingError, match="Nothing had finished yet."):
        with run.stage("tailoring the content to the job"):
            raise ApiKeyMissingError("No OpenAI API key is available.")


def test_the_exception_type_survives_so_routes_map_it_the_same_way(tmp_path):
    """app/api/resume_build.py turns a budget error into 402 and a rate
    limit into 503 by type. Annotating the message must not cost that."""
    _reset_db(tmp_path)
    run = Pipeline("The resume build")

    with pytest.raises(BudgetExceededError):
        with run.stage("tailoring the content to the job"):
            raise BudgetExceededError("The monthly budget is used up.")


def test_a_bug_in_a_step_is_left_exactly_as_it_was(tmp_path):
    """"Which step, and what had finished" is the answer to "I ran out of
    keys", not to "this code is broken". Wrapping the second kind would
    replace a traceback that helps with a sentence that does not."""
    _reset_db(tmp_path)
    run = Pipeline("The resume build")

    with pytest.raises(ValueError, match="^no account with id=9$"):
        with run.stage("tailoring the content to the job"):
            raise ValueError("no account with id=9")


def test_a_stop_is_written_to_the_event_log(tmp_path):
    _reset_db(tmp_path)
    from app.core.rate_limits import list_events

    run = Pipeline("The resume build", account_id=3)
    with pytest.raises(LLMRateLimitedError):
        with run.stage("tailoring the content to the job"):
            raise LLMRateLimitedError("Every key is out of quota.")

    events = [e for e in list_events(source="llm") if e.kind == "stage_failed"]
    assert len(events) == 1
    assert events[0].context == "The resume build"
    assert events[0].account_id == 3
