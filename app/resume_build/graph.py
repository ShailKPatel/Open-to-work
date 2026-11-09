"""The resume build as a LangGraph graph:

    retrieve -> write -> judge -> human_approval -> page_fit
                 ^  ^      |            |
                 |  +-- revise          |
                 +----------------------+ (a plain instruction)

The nodes only sequence work that lives elsewhere: retrieval and writing
(orchestrator.py), the rule filter (grounding.py), the LLM judge
(app/evals/groundedness.py) and the page fit (pagefit.py). Every model call
still goes through app/core/llm.py's complete(), so budget, cache, cost
tracking and failover apply as they do everywhere else. LangChain's chat
wrappers are deliberately not used: they would call the provider around it.

A bullet the judge rejects goes back to the writer with the reason, for up
to MAX_REVISION_ROUNDS rounds. One still failing after that is dropped.

The state is checkpointed to its own SQLite file after every node, so a
build waiting for approval, or stopped by a failure, carries on after a
restart. The posting's text is not in the state: nodes read it from its row
when they need it, so quarantined text is not copied into a second file.
"""

from __future__ import annotations

import copy
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.core.llm import (
    ApiKeyMissingError,
    BudgetExceededError,
    LLMDispatchError,
)
from app.core.pipeline import Pipeline
from app.core.settings import get_settings
from app.evals.groundedness import judge_bullet
from app.resume_build.grounding import TechVocabulary, bullet_problem
from app.resume_build.orchestrator import (
    _DeterministicContext,
    bullet_evidence,
    context_from_dict,
    context_to_dict,
    edit_resume_content,
    finish_draft,
    load_context,
    load_posting,
    revise_bullets,
    write_draft,
)
from app.resume_build.pagefit import PageFitNotAchievedError, fit_to_page_limit

logger = logging.getLogger(__name__)

MAX_REVISION_ROUNDS = 2

TAILOR_STEP = "tailoring the content to the job"
_FIT_STEP = "fitting the resume to the page count"
_TAILOR_STAGE = "Tailoring the content to the job"
_JUDGE_STAGE = "Fact-checking the bullets"
_REVISE_STAGE = "Revising the rejected bullets"
_INSTRUCTION_STAGE = "Applying your instruction"
_FIT_STAGE = "Fitting the resume to the page count"

# The judge answers only true or false, and its prompt is what the human
# labels validated (app/evals/llm_evals.py), so it is not asked for more.
_JUDGE_REASON = (
    "An independent fact-check found a claim here the project evidence does "
    "not state. Keep only what the evidence says."
)

_LLM_UNREACHABLE = (ApiKeyMissingError, BudgetExceededError, LLMDispatchError)


class BuildInputError(ValueError):
    """The build was asked for something that cannot be built: an unknown
    account or posting, or a writer reply that was not usable."""


class ResumeState(TypedDict, total=False):
    request: dict[str, Any]
    max_pages: int
    require_approval: bool
    context: dict[str, Any]
    draft: dict[str, Any]
    verdicts: list[dict[str, Any]]
    feedback: list[dict[str, Any]]
    revision_count: int
    first_rejected: int | None
    judge_stats: dict[str, Any]
    approval_status: str
    instruction: str | None
    fit: dict[str, Any]


@dataclass
class BuildHooks:
    """Per-run reporting. Passed by closure, not through the state, because
    none of it can or should be checkpointed. `progress` is the same
    in-place dict app/api/resume_build.py saves when a build stops."""

    pipeline: Pipeline
    on_stage: Callable[[str], None]
    progress: dict[str, Any]


@dataclass
class GraphResult:
    awaiting_approval: bool
    values: dict[str, Any]


@contextmanager
def _as_input_error() -> Iterator[None]:
    try:
        yield
    except ValueError as e:
        raise BuildInputError(str(e)) from e


def _judge_draft(
    ctx: _DeterministicContext, draft: dict[str, Any], account_id: int
) -> list[dict[str, Any]]:
    """One verdict per bullet: the free rule filter first, then the LLM
    judge on what passes it. Text identical to a bullet already judged is
    answered from complete()'s cache, so a round only pays for rewrites.

    A judge that cannot be reached leaves the remaining bullets checked by
    the rules alone, the bar every bullet met before the judge was added:
    a spent key should not cost the person the resume already paid for.
    """
    vocabulary = TechVocabulary(ctx.technologies)
    reachable = True
    verdicts: list[dict[str, Any]] = []
    for project in draft.get("projects", []):
        source = bullet_evidence(ctx, project)
        if source is None:
            continue
        evidence, skills, description = source
        for bullet in project["points"]:
            grounded, reason, checked_by = True, None, "none"
            if bullet.strip() != description:
                reason = bullet_problem(bullet, evidence, skills, vocabulary)
                checked_by = "rule"
                grounded = reason is None
                if grounded and reachable:
                    try:
                        answer = judge_bullet(evidence, bullet, account_id)
                    except _LLM_UNREACHABLE as e:
                        logger.warning("judge unreachable, rules only from here: %s", e)
                        reachable = False
                    else:
                        checked_by = "judge"
                        if answer is False:
                            grounded, reason = False, _JUDGE_REASON
            verdicts.append(
                {
                    "repo_id": project["repo_id"],
                    "project": project.get("name"),
                    "bullet": bullet,
                    "grounded": grounded,
                    "reason": reason,
                    "checked_by": checked_by,
                }
            )
    return verdicts


def _rejected(state: ResumeState) -> list[dict[str, Any]]:
    return [v for v in state.get("verdicts", []) if not v["grounded"]]


def _build_nodes_graph(hooks: BuildHooks) -> StateGraph:
    def retrieve(state: ResumeState) -> dict[str, Any]:
        request = state["request"]
        hooks.on_stage(_TAILOR_STAGE)
        with _as_input_error():
            ctx = load_context(
                request["account_id"],
                request["job_posting_id"],
                selected_email=request.get("selected_email"),
                selected_phone=request.get("selected_phone"),
                selected_project_ids=request.get("selected_project_ids"),
                selected_skills=request.get("selected_skills"),
            )
        return {"context": context_to_dict(ctx)}

    def write(state: ResumeState) -> dict[str, Any]:
        request = state["request"]
        account_id = request["account_id"]
        ctx = context_from_dict(state["context"])

        if state.get("instruction"):
            hooks.on_stage(_INSTRUCTION_STAGE)
            with hooks.pipeline.stage("applying your instruction"), _as_input_error():
                job_text, query = load_posting(request["job_posting_id"])
                draft = edit_resume_content(
                    account_id,
                    job_text,
                    state["draft"],
                    state["instruction"] or "",
                    template=request["template"],
                    retrieval_query=query,
                    context=ctx,
                )
            return {
                "draft": draft,
                "instruction": None,
                "feedback": [],
                "verdicts": [],
                "revision_count": 0,
                "first_rejected": None,
                "approval_status": "pending",
            }

        if state.get("feedback"):
            hooks.on_stage(_REVISE_STAGE)
            with hooks.pipeline.stage("revising the rejected bullets"):
                draft = revise_bullets(ctx, state["draft"], state["feedback"], account_id)
            return {"draft": draft, "feedback": []}

        hooks.on_stage(_TAILOR_STAGE)
        with hooks.pipeline.stage(TAILOR_STEP), _as_input_error():
            job_text, _ = load_posting(request["job_posting_id"])
            draft = write_draft(
                ctx,
                account_id,
                job_text,
                request["template"],
                project_instructions=request.get("project_instructions"),
                selected_skills=request.get("selected_skills"),
                selected_experience_ids=request.get("selected_experience_ids"),
                selected_education_ids=request.get("selected_education_ids"),
                custom_instruction=request.get("custom_instruction"),
                points_per_role=request.get("points_per_role"),
            )
        return {"draft": draft}

    def judge(state: ResumeState) -> dict[str, Any]:
        hooks.on_stage(_JUDGE_STAGE)
        account_id = state["request"]["account_id"]
        ctx = context_from_dict(state["context"])
        verdicts = _judge_draft(ctx, state["draft"], account_id)
        rejected = [v for v in verdicts if not v["grounded"]]
        first_rejected = state.get("first_rejected")
        if first_rejected is None:
            first_rejected = len(rejected)
        update: dict[str, Any] = {"verdicts": verdicts, "first_rejected": first_rejected}
        if rejected and state.get("revision_count", 0) < MAX_REVISION_ROUNDS:
            return update

        draft = finish_draft(ctx, state["draft"], {(v["repo_id"], v["bullet"]) for v in rejected})
        stats = {
            "rejected": first_rejected,
            "saved": max(0, first_rejected - len(rejected)),
            "dropped": len(rejected),
            "rounds": state.get("revision_count", 0),
            "dropped_bullets": [
                {"project": v["project"], "bullet": v["bullet"], "reason": v["reason"]}
                for v in rejected
            ],
        }
        logger.info(
            "resume build for account %s: judge rejected %d bullet(s), %d saved, %d dropped",
            account_id, stats["rejected"], stats["saved"], stats["dropped"],
        )
        return {**update, "draft": draft, "judge_stats": stats}

    def revise(state: ResumeState) -> dict[str, Any]:
        ctx = context_from_dict(state["context"])
        projects = {p["repo_id"]: p for p in state["draft"]["projects"]}
        feedback = []
        for verdict in _rejected(state):
            source = bullet_evidence(ctx, projects[verdict["repo_id"]])
            feedback.append(
                {
                    "repo_id": verdict["repo_id"],
                    "bullet": verdict["bullet"],
                    "reason": verdict["reason"],
                    "evidence": source[0] if source else "",
                }
            )
        return {"feedback": feedback, "revision_count": state.get("revision_count", 0) + 1}

    def human_approval(state: ResumeState) -> dict[str, Any]:
        if not state.get("require_approval"):
            return {"approval_status": "approved"}
        # Nothing before this line may have a side effect: LangGraph runs
        # the node again from the top when the decision arrives.
        decision = interrupt({"kind": "resume_approval"})
        if decision.get("action") == "revise":
            return {
                "approval_status": "changes_requested",
                "instruction": decision.get("instruction"),
            }
        return {"approval_status": "approved"}

    def page_fit(state: ResumeState) -> dict[str, Any]:
        request = state["request"]
        progress = hooks.progress
        # A retry finds the fit's own partial work here, newer than the
        # approved draft, and does not pay for those model calls again.
        if progress.get("data") is None:
            progress["data"] = copy.deepcopy(state["draft"])
        data = copy.deepcopy(progress["data"])
        hooks.on_stage(_FIT_STAGE)
        try:
            with hooks.pipeline.stage(_FIT_STEP):
                result = fit_to_page_limit(
                    data,
                    request["template"],
                    state["max_pages"],
                    account_id=request["account_id"],
                    reworded=bool(progress.get("reworded")),
                    cuts_made=int(progress.get("cuts_made") or 0),
                    on_progress=progress.update,
                )
        except PageFitNotAchievedError as e:
            # Includes the page fit losing every key partway: the content
            # is written and compiled, so it is kept with the reason.
            logger.warning(
                "page-fit did not reach target for account_id=%s: %s", request["account_id"], e
            )
            return {
                "fit": {
                    "pdf_bytes": e.best_pdf_bytes,
                    "page_count": e.best_page_count,
                    "achieved": False,
                    "note": str(e),
                    "content": e.best_data or data,
                }
            }
        if not result.fit_exact:
            logger.warning(
                "page-fit came up short for account_id=%s: %s page(s) against a target of %s",
                request["account_id"], result.page_count, state["max_pages"],
            )
        return {
            "fit": {
                "pdf_bytes": result.pdf_bytes,
                "page_count": result.page_count,
                "achieved": result.fit_exact,
                "note": None,
                "content": result.data or data,
            }
        }

    def route_start(state: ResumeState) -> str:
        # A build stopped before this graph existed has an approved draft
        # and no checkpoint, and goes straight to the page fit.
        return "page_fit" if state.get("approval_status") == "approved" else "retrieve"

    def route_judge(state: ResumeState) -> str:
        if _rejected(state) and state.get("revision_count", 0) < MAX_REVISION_ROUNDS:
            return "revise"
        return "human_approval"

    def route_approval(state: ResumeState) -> str:
        return "page_fit" if state.get("approval_status") == "approved" else "write"

    graph = StateGraph(ResumeState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("write", write)
    graph.add_node("judge", judge)
    graph.add_node("revise", revise)
    graph.add_node("human_approval", human_approval)
    graph.add_node("page_fit", page_fit)
    graph.add_conditional_edges(START, route_start, ["retrieve", "page_fit"])
    graph.add_edge("retrieve", "write")
    graph.add_edge("write", "judge")
    graph.add_conditional_edges("judge", route_judge, ["revise", "human_approval"])
    graph.add_edge("revise", "write")
    graph.add_conditional_edges("human_approval", route_approval, ["page_fit", "write"])
    graph.add_edge("page_fit", END)
    return graph


def _checkpoint_path() -> Path:
    return Path(get_settings().resume_graph_checkpoint_path)


@contextmanager
def _open_checkpointer() -> Iterator[SqliteSaver]:
    path = _checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One connection per run. Graph steps hand it between threads, and
    # SqliteSaver serializes access itself.
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()


def run_graph(
    request: dict[str, Any],
    max_pages: int,
    hooks: BuildHooks,
    thread_id: str,
    require_approval: bool,
    decision: dict[str, Any] | None = None,
) -> GraphResult:
    """Starts the build, or carries on thread_id's: with `decision` if it is
    waiting for one, from the failed node if it stopped. A thread still
    waiting comes straight back so its draft can be shown again. A finished
    build's checkpoints are deleted: the resume is in the library by then,
    and this file should not keep every draft ever written."""
    with _open_checkpointer() as saver:
        graph: Any = _build_nodes_graph(hooks).compile(checkpointer=saver)
        config: Any = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)

        graph_input: Any
        if decision is not None:
            graph_input = Command(resume=decision)
        elif snapshot.interrupts:
            return GraphResult(awaiting_approval=True, values=dict(snapshot.values))
        elif snapshot.next:
            graph_input = None
        else:
            graph_input = {
                "request": request,
                "max_pages": max_pages,
                "require_approval": require_approval,
                "revision_count": 0,
                "approval_status": "pending",
            }
            if hooks.progress.get("data") is not None:
                graph_input["draft"] = hooks.progress["data"]
                graph_input["approval_status"] = "approved"

        graph.invoke(graph_input, config)
        snapshot = graph.get_state(config)
        values = dict(snapshot.values)
        if snapshot.interrupts:
            return GraphResult(awaiting_approval=True, values=values)
        saver.delete_thread(thread_id)
        return GraphResult(awaiting_approval=False, values=values)


def forget_thread(thread_id: str) -> None:
    """Deletes one unfinished build's checkpoints."""
    with _open_checkpointer() as saver:
        saver.delete_thread(thread_id)


def draft_preview(values: dict[str, Any]) -> dict[str, Any]:
    """What the person reads before approving. The reserve is left out: it
    is held-back material, not part of this resume."""
    draft = values.get("draft") or {}
    return {
        "draft": {
            "summary": draft.get("summary"),
            "projects": [
                {"name": p.get("name"), "tagline": p.get("tagline"), "points": p.get("points", [])}
                for p in draft.get("projects", [])
            ],
            "experience": [
                {
                    "company": r.get("company"),
                    "title": r.get("title"),
                    "points": r.get("points", []),
                }
                for r in draft.get("experience", [])
            ],
            "skills": draft.get("skills", []),
        },
        "judge": values.get("judge_stats") or {},
    }
