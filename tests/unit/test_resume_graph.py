"""app/resume_build/graph.py, run for real against its SQLite checkpointer.
Only the model-facing functions and the page fit are faked, so routing, the
revise limit, the interrupt and every checkpoint round trip are the real
thing. API-level approval tests are at the bottom.
"""

import datetime as dt
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.llm import BudgetExceededError
from app.core.pipeline import Pipeline
from app.resume_build import graph as graph_module
from app.resume_build import orchestrator
from app.resume_build.compile import CompileError
from app.resume_build.graph import BuildHooks, BuildInputError, draft_preview, run_graph
from app.resume_build.grounding import TechVocabulary, bullet_problem
from app.resume_build.pagefit import FitResult, PageFitNotAchievedError

GOOD = "Built a parser in Python"
BAD = "Served 900 customers"
FIXED = "Built a customer-facing parser"


def _context():
    return orchestrator._DeterministicContext(
        header={"name": "A"},
        experience=[],
        education=[],
        candidates=[
            {
                "repo_id": 7,
                "name": "parser",
                "description": "A parser",
                "href": "u",
                "skills": ["Python"],
                # A date and integer-keyed maps below must survive the checkpoint.
                "date": dt.date(2025, 1, 2),
            }
        ],
        candidate_skill_names=["Python"],
        experience_pool=[],
        experience_all_points={1: ["p"]},
        technologies=["Python"],
        repo_strength={7: 1.0},
    )


def _draft(points):
    return {
        "summary": "s",
        "experience": [],
        "projects": [
            {"repo_id": 7, "name": "parser", "tagline": None, "href": "u", "url_display": "parser",
             "date_range": "", "points": list(points), "note": None, "user_note": None}
        ],
        "education": [],
        "technologies": [],
        "skills": ["Python"],
    }


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(graph_module, "load_context", lambda *a, **k: _context())
    monkeypatch.setattr(graph_module, "finish_draft", orchestrator.finish_draft)
    fit = FitResult(tex="x", pdf_bytes=b"%PDF", page_count=1, cuts_made=0, data={"summary": "s"})
    fit_mock = MagicMock(return_value=fit)
    monkeypatch.setattr(graph_module, "fit_to_page_limit", fit_mock)
    write = MagicMock(return_value=_draft([GOOD, BAD]))
    monkeypatch.setattr(graph_module, "write_draft", write)
    judge = MagicMock(side_effect=lambda evidence, bullet, account_id: bullet != "judge-fails")
    monkeypatch.setattr(graph_module, "judge_bullet", judge)
    return SimpleNamespace(fit=fit_mock, write=write, judge=judge)


REQUEST = {"account_id": 1, "job_posting_id": 2, "template": "onepage"}


def _run(thread="t", approval=False, decision=None, progress=None, stages=None):
    hooks = BuildHooks(
        pipeline=Pipeline("test"),
        on_stage=(stages.append if stages is not None else lambda _: None),
        progress=progress if progress is not None else {},
    )
    return run_graph(REQUEST, 1, hooks, thread, approval, decision=decision)


def _rewriter(monkeypatch, text):
    mock = MagicMock(
        return_value=SimpleNamespace(
            parsed={"rewrites": [{"repo_id": 7, "original": BAD, "text": text}]}
        )
    )
    monkeypatch.setattr(orchestrator, "complete", mock)
    return mock


def test_clean_draft_goes_straight_to_the_page_fit(fakes):
    fakes.write.return_value = _draft([GOOD])

    result = _run()

    assert result.awaiting_approval is False
    assert result.values["fit"]["pdf_bytes"] == b"%PDF"
    assert result.values["judge_stats"]["rejected"] == 0
    assert result.values["revision_count"] == 0
    assert result.values["draft"]["reserve"]["known_technologies"] == ["Python"]


def test_rejected_bullet_is_fixed_by_the_writer_and_counted_as_saved(fakes, monkeypatch):
    rewriter = _rewriter(monkeypatch, FIXED)
    stages = []

    result = _run(stages=stages)

    assert result.values["draft"]["projects"][0]["points"] == [GOOD, FIXED]
    stats = result.values["judge_stats"]
    assert (stats["rejected"], stats["saved"], stats["dropped"]) == (1, 1, 0)
    assert rewriter.call_count == 1
    sent = rewriter.call_args.args[1][1]["content"]
    assert BAD in str(sent) and "900" in str(sent) and GOOD not in str(sent)
    assert rewriter.call_args.kwargs["purpose"] == "resume_revise"
    assert "Revising the rejected bullets" in stages


def test_bullet_still_failing_after_two_rounds_is_dropped(fakes, monkeypatch):
    rewriter = _rewriter(monkeypatch, BAD)

    result = _run()

    assert rewriter.call_count == graph_module.MAX_REVISION_ROUNDS
    assert result.values["draft"]["projects"][0]["points"] == [GOOD]
    stats = result.values["judge_stats"]
    assert (stats["rejected"], stats["saved"], stats["dropped"]) == (1, 0, 1)
    assert stats["dropped_bullets"][0]["reason"].startswith("states a number")


def test_llm_judge_rejection_carries_its_reason_to_the_writer(fakes, monkeypatch):
    fakes.write.return_value = _draft(["judge-fails"])
    rewriter = MagicMock(
        return_value=SimpleNamespace(
            parsed={"rewrites": [{"repo_id": 7, "original": "judge-fails", "text": GOOD}]}
        )
    )
    monkeypatch.setattr(orchestrator, "complete", rewriter)

    result = _run()

    assert "independent fact-check" in str(rewriter.call_args.args[1][1]["content"])
    assert result.values["judge_stats"]["saved"] == 1


def test_project_left_with_nothing_falls_back_to_its_description(fakes, monkeypatch):
    fakes.write.return_value = _draft([BAD])
    _rewriter(monkeypatch, BAD)

    result = _run()

    assert result.values["draft"]["projects"][0]["points"] == ["A parser"]


def test_unreachable_judge_keeps_rule_passing_bullets(fakes):
    fakes.write.return_value = _draft([GOOD, "Wrote docs"])
    fakes.judge.side_effect = BudgetExceededError("spent")

    result = _run()

    assert result.values["draft"]["projects"][0]["points"] == [GOOD, "Wrote docs"]
    assert fakes.judge.call_count == 1
    assert {v["checked_by"] for v in result.values["verdicts"]} == {"rule"}


def test_unusable_judge_reply_does_not_reject(fakes):
    fakes.write.return_value = _draft([GOOD])
    fakes.judge.side_effect = None
    fakes.judge.return_value = None

    assert _run().values["draft"]["projects"][0]["points"] == [GOOD]


def test_approval_stops_before_page_fit_then_approve_continues(fakes, monkeypatch):
    fakes.write.return_value = _draft([GOOD])
    progress = {}

    waiting = _run(approval=True, progress=progress)

    assert waiting.awaiting_approval is True
    fakes.fit.assert_not_called()
    assert "data" not in progress  # unapproved text is never what a retry fits
    preview = draft_preview(waiting.values)
    assert preview["draft"]["projects"][0]["points"] == [GOOD]
    assert "reserve" not in preview["draft"]

    # Asked again with no decision (after a restart): same draft, no work.
    again = _run(approval=True, progress=progress)
    assert again.awaiting_approval is True
    assert fakes.write.call_count == 1

    done = _run(approval=True, decision={"action": "approve"}, progress=progress)
    assert done.awaiting_approval is False
    assert done.values["approval_status"] == "approved"
    fakes.fit.assert_called_once()


def test_plain_instruction_is_applied_and_judged_before_asking_again(fakes, monkeypatch):
    fakes.write.return_value = _draft([GOOD])
    edited = {**_draft(["Edited line"]), "reserve": {}}
    edit = MagicMock(return_value=edited)
    monkeypatch.setattr(graph_module, "edit_resume_content", edit)
    _run(approval=True)

    again = _run(
        approval=True, decision={"action": "revise", "instruction": "shorter please"}
    )

    assert again.awaiting_approval is True
    assert edit.call_args.args[3] == "shorter please"
    assert again.values["instruction"] is None
    assert again.values["draft"]["projects"][0]["points"] == ["Edited line"]
    assert ("Edited line") in [c.args[1] for c in fakes.judge.call_args_list]
    assert edit.call_args.kwargs["context"].experience_all_points == {1: ["p"]}


def test_context_survives_the_checkpoint_with_its_types(fakes):
    fakes.write.return_value = _draft([GOOD])
    result = _run(approval=True)
    context = result.values["context"]
    assert context["candidates"][0]["date"] == dt.date(2025, 1, 2)
    assert context["repo_strength"] == {7: 1.0}


def test_failed_page_fit_retries_from_that_node_only(fakes):
    fakes.write.return_value = _draft([GOOD])
    fit = fakes.fit.return_value
    fakes.fit.side_effect = CompileError("timed out")
    progress = {}

    with pytest.raises(CompileError):
        _run(progress=progress)

    fakes.fit.side_effect = None
    fakes.fit.return_value = fit
    done = _run(progress=progress)

    assert done.values["fit"]["page_count"] == 1
    assert fakes.write.call_count == 1


def test_page_fit_not_achieved_keeps_the_best_result(fakes):
    fakes.write.return_value = _draft([GOOD])
    fakes.fit.side_effect = PageFitNotAchievedError(
        "no cuts left", "tex", b"%PDF-best", 2, {"summary": "best"}
    )

    fit = _run().values["fit"]

    assert (fit["pdf_bytes"], fit["achieved"], fit["note"]) == (b"%PDF-best", False, "no cuts left")
    assert fit["content"] == {"summary": "best"}


def test_page_fit_coming_up_short_is_not_exact(fakes):
    fakes.write.return_value = _draft([GOOD])
    fakes.fit.return_value = FitResult(
        tex="x", pdf_bytes=b"%PDF", page_count=1, cuts_made=0, fit_exact=False, data={"a": 1}
    )
    assert _run().values["fit"]["achieved"] is False


def test_draft_saved_before_the_graph_goes_straight_to_the_page_fit(fakes):
    result = _run(thread="legacy", progress={"data": _draft([GOOD])})

    assert result.values["fit"]["pdf_bytes"] == b"%PDF"
    fakes.write.assert_not_called()


def test_finished_run_leaves_no_checkpoint(fakes):
    fakes.write.return_value = _draft([GOOD])
    _run(thread="gone")

    fakes.write.reset_mock()
    fakes.write.return_value = _draft([GOOD])
    _run(thread="gone")
    assert fakes.write.call_count == 1  # a fresh run, nothing was left to resume


def test_unknown_posting_is_an_input_error(fakes, monkeypatch):
    def _missing(*a, **k):
        raise ValueError("no job posting with id=2")

    monkeypatch.setattr(graph_module, "load_context", _missing)
    with pytest.raises(BuildInputError):
        _run()


# ---- orchestrator seams


def test_revise_ignores_bullets_it_was_not_asked_about_and_empty_text(monkeypatch):
    reply = {
        "rewrites": [
            {"repo_id": 7, "original": GOOD, "text": "sneaky change"},
            {"repo_id": 7, "original": BAD, "text": "   "},
            "not a dict",
        ]
    }
    monkeypatch.setattr(
        orchestrator, "complete", MagicMock(return_value=SimpleNamespace(parsed=reply))
    )
    feedback = [{"repo_id": 7, "bullet": BAD, "reason": "r", "evidence": "e"}]

    draft = orchestrator.revise_bullets(_context(), _draft([GOOD, BAD]), feedback, 1)

    assert draft["projects"][0]["points"] == [GOOD, BAD]


def test_revise_survives_an_unparsed_reply(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "complete", MagicMock(return_value=SimpleNamespace(parsed=None))
    )
    feedback = [{"repo_id": 7, "bullet": BAD, "reason": "r", "evidence": "e"}]
    draft = orchestrator.revise_bullets(_context(), _draft([BAD]), feedback, 1)
    assert draft["projects"][0]["points"] == [BAD]


def test_bullet_problem_names_the_technology():
    vocabulary = TechVocabulary(["Python", "Rust"])
    assert "rust" in bullet_problem("Used Rust", "evidence", ["Python"], vocabulary)
    assert bullet_problem("Used Python", "evidence", ["Python"], vocabulary) is None


def test_load_context_and_posting_reject_unknown_ids(tmp_path, monkeypatch):
    import app.core.db as db_module
    from app.core.db import Account, JobPosting, get_db, init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    get_settings.cache_clear()
    init_db()
    with pytest.raises(ValueError, match="no job posting"):
        orchestrator.load_posting(99)

    db = get_db()
    account = Account(first_name="A", last_name="B", github_username="")
    db.add(account)
    db.commit()
    posting = JobPosting(
        account_id=account.id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="hiring", content_hash="h",
    )
    db.add(posting)
    db.commit()
    posting_id = posting.id
    db.close()

    assert orchestrator.load_posting(posting_id)[0] == "hiring"
    with pytest.raises(ValueError, match="no account"):
        orchestrator.load_context(99, posting_id)
    get_settings.cache_clear()


# ---- the API's approval step


@pytest.fixture
def posting(tmp_path, monkeypatch):
    import app.core.db as db_module
    from app.core.db import Account, JobPosting, get_db, init_db
    from app.core.settings import get_settings

    db_module.reset_engine()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("RESUME_STORAGE_DIR", str(tmp_path / "resumes"))
    get_settings.cache_clear()
    init_db()
    db = get_db()
    account = Account(first_name="A", last_name="B", github_username="")
    db.add(account)
    db.commit()
    row = JobPosting(
        account_id=account.id, source="pasted", external_id="x", company="Acme",
        title="Engineer", raw_text_quarantined="hiring", content_hash="h",
    )
    db.add(row)
    db.commit()
    body = {"account_id": account.id, "job_posting_id": row.id}
    db.close()
    monkeypatch.setattr("app.api.resume_build._approval_required", lambda: True)
    monkeypatch.setattr(
        "app.retrieval.index.index_resume", lambda *a, **k: None
    )
    yield body
    get_settings.cache_clear()


def _settled(client, build_id, want):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = client.get(f"/api/resume-build/builds/{build_id}").json()
        if state["state"] in want:
            return state
        time.sleep(0.02)
    raise AssertionError(f"build never reached {want}: {state}")


def test_api_waits_shows_the_draft_then_approval_finishes(posting, fakes):
    fakes.write.return_value = _draft([GOOD])
    client = TestClient(app)

    waiting = _settled(
        client, client.post("/api/resume-build/start", json=posting).json()["id"],
        {"awaiting_approval"},
    )

    assert waiting["running"] is False
    assert waiting["draft"]["projects"][0]["points"] == [GOOD]
    assert waiting["retry_of"] is not None
    url = f"/api/resume-build/builds/{waiting['id']}"
    assert client.post(f"{url}/dismiss").json() == {"dismissed": False}

    resp = client.post(f"{url}/decision", json={"action": "approve"})
    assert resp.status_code == 200

    done = _settled(client, waiting["id"], {"done", "error"})
    assert done["state"] == "done"
    assert done["resume_id"] == waiting["retry_of"]
    assert client.post(f"{url}/decision", json={"action": "approve"}).status_code == 409


def test_api_retry_of_a_waiting_build_returns_it(posting, fakes):
    fakes.write.return_value = _draft([GOOD])
    client = TestClient(app)
    waiting = _settled(
        client, client.post("/api/resume-build/start", json=posting).json()["id"],
        {"awaiting_approval"},
    )

    retry = client.post(f"/api/resume-build/retry/{waiting['retry_of']}").json()

    assert retry["id"] == waiting["id"]
    assert fakes.write.call_count == 1


def test_api_after_restart_retry_shows_the_saved_draft_again(posting, fakes):
    from app.resume_build import background

    fakes.write.return_value = _draft([GOOD])
    client = TestClient(app)
    waiting = _settled(
        client, client.post("/api/resume-build/start", json=posting).json()["id"],
        {"awaiting_approval"},
    )
    background._builds.pop(waiting["id"])  # what a process restart does

    retry = client.post(f"/api/resume-build/retry/{waiting['retry_of']}").json()
    again = _settled(client, retry["id"], {"awaiting_approval"})

    assert again["id"] != waiting["id"]
    assert again["draft"]["projects"][0]["points"] == [GOOD]
    assert fakes.write.call_count == 1


def test_api_revise_needs_an_instruction_and_a_known_action(posting, fakes):
    fakes.write.return_value = _draft([GOOD])
    client = TestClient(app)
    waiting = _settled(
        client, client.post("/api/resume-build/start", json=posting).json()["id"],
        {"awaiting_approval"},
    )
    url = f"/api/resume-build/builds/{waiting['id']}/decision"

    assert client.post(url, json={"action": "revise", "instruction": "  "}).status_code == 422
    assert client.post(url, json={"action": "explode"}).status_code == 422
    assert client.post(url, json={"action": "revise", "instruction": "x" * 2001}).status_code == 422
    assert _settled(client, waiting["id"], {"awaiting_approval"})["id"] == waiting["id"]


def test_double_click_only_starts_one_run():
    from app.resume_build.background import Build

    build = Build(id="x", account_id=1, job_posting_id=2, label="l")
    calls = []
    build.await_approval({}, calls.append, 5)
    first, second = build.take_continuation(), build.take_continuation()
    assert first is not None and second is None
