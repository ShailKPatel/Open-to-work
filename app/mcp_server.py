"""MCP server: one profile's evidence, projects, job postings and resume
builds as tools an MCP client (Claude Desktop, Claude Code, Cursor) can
call.

Every tool wraps code the web app already runs; nothing here decides
anything the routes do not. The profile is fixed when the server is built
(--account-id or OTW_MCP_ACCOUNT_ID), so no tool takes an account_id and
none can reach another profile's rows: an id from a client is checked
against that profile first, and one that belongs elsewhere reads as not
found, as app/api/demo.py answers it.

With DEMO_MODE on, the same per-profile rules as the hosted demo apply:
the sample profile is read-only, and writes and builds count against the
demo's daily limits.

read_only=True leaves out the tools that write or spend (adding a posting,
starting a build). The profile agent (app/agent/) builds its server that
way, so text inside a posting or a README cannot steer it into either.

Run over stdio:  make mcp ACCOUNT=1
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any

import anyio
from fastapi import HTTPException
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from sqlalchemy import select

from app.core.db import Account, Experience, JobPosting, Repository, get_db
from app.core.settings import get_settings

_MAX_RESULTS = 25

_INSTRUCTIONS = (
    "Tools for one Open to Work profile: search the skill evidence drawn "
    "from its GitHub projects and work experience, list its projects, save "
    "a job posting, and start or follow a resume build tailored to one."
)


def _not_found(kind: str, ident: object) -> ToolError:
    return ToolError(f"No {kind} with id {ident} in this profile.")


def _detail(e: HTTPException) -> str:
    detail = e.detail
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail)


def _demo_user(account_id: int) -> Any:
    """The demo's view of this profile, or None when DEMO_MODE is off."""
    settings = get_settings()
    if not settings.demo_mode:
        return None
    from app.api.demo import DemoUser

    sample = account_id == settings.demo_sample_account_id
    return DemoUser(account_id, "", admin=False, sample=sample)


def _check_write(account_id: int, route_path: str) -> None:
    """Applies the hosted demo's rules for a write to this profile."""
    user = _demo_user(account_id)
    if user is None:
        return
    if user.sample:
        raise ToolError("The sample profile is read-only.")
    from app.api import demo

    try:
        if route_path == "/api/resume-build/start":
            demo._check_build_limits(user)
        demo._check_write_limits(user, route_path)
    except HTTPException as e:
        raise ToolError(_detail(e)) from e


def _call[T](fn: Callable[[], T]) -> T:
    """Runs a wrapped route function, turning its HTTP errors into tool
    errors with the same message."""
    try:
        return fn()
    except HTTPException as e:
        raise ToolError(_detail(e)) from e


def _owned_posting(account_id: int, posting_id: int) -> None:
    db = get_db()
    try:
        owner = db.execute(
            select(JobPosting.account_id).where(JobPosting.id == posting_id)
        ).scalar_one_or_none()
    finally:
        db.close()
    if owner != account_id:
        raise _not_found("job posting", posting_id)


def search_evidence(
    account_id: int, query: str, top_k: int = 10, source: str | None = None
) -> list[dict[str, Any]]:
    from app.retrieval.search import search_skill_evidence

    if source not in (None, "repo", "experience"):
        raise ToolError("source must be 'repo', 'experience' or left out.")
    if not query.strip():
        raise ToolError("query is empty.")
    top_k = max(1, min(top_k, _MAX_RESULTS))
    hits = search_skill_evidence(query, account_id, top_k=top_k, source_type=source)

    repo_ids = sorted({h.payload["repo_id"] for h in hits if h.payload.get("repo_id")})
    experience_ids = sorted(
        {h.payload["experience_id"] for h in hits if h.payload.get("experience_id")}
    )
    repos: dict[int, str] = {}
    roles: dict[int, str] = {}
    db = get_db()
    try:
        if repo_ids:
            for repo in db.execute(
                select(Repository).where(
                    Repository.id.in_(repo_ids), Repository.account_id == account_id
                )
            ).scalars():
                repos[repo.id] = repo.full_name
        if experience_ids:
            for role in db.execute(
                select(Experience).where(
                    Experience.id.in_(experience_ids), Experience.account_id == account_id
                )
            ).scalars():
                roles[role.id] = f"{role.title} at {role.company}"
    finally:
        db.close()

    results = []
    for hit in hits:
        p = hit.payload
        if p.get("account_id") != account_id:
            continue  # the search filters on it; this is a second lock
        source_type = p.get("source_type")
        results.append(
            {
                "skill": p.get("skill"),
                "score": round(hit.score, 4),
                "source": source_type,
                "project": repos.get(p.get("repo_id") or 0) if source_type == "repo" else None,
                "project_id": p.get("repo_id"),
                "role": roles.get(p.get("experience_id") or 0),
                "experience_id": p.get("experience_id"),
                "evidence_type": p.get("evidence_type"),
                "confidence": p.get("confidence"),
            }
        )
    return results


def list_projects(account_id: int) -> list[dict[str, Any]]:
    from app.api import projects

    db = get_db()
    try:
        rows = projects.list_projects(account_id, db=db)
    finally:
        db.close()
    return [
        {
            "id": r.id,
            "name": r.full_name,
            "url": r.url,
            "description": r.description,
            "language": r.primary_language,
            "skill_count": r.skill_count,
            "starred": r.starred,
            "exclude_from_resume": r.exclude_from_resume,
            "extraction_status": r.skill_extraction_status,
        }
        for r in rows
    ]


def add_job_posting(
    account_id: int,
    text: str,
    company: str | None = None,
    title: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    from app.api import job_postings

    _check_write(account_id, "/api/job-postings")
    body = job_postings.JobPostingCreate(
        account_id=account_id,
        raw_text=text,
        company=company,
        title=title,
        location=location,
    )
    db = get_db()
    try:
        posting = _call(lambda: job_postings.create_posting(body, db=db))
    finally:
        db.close()
    return {
        "id": posting.id,
        "company": posting.company,
        "title": posting.title,
        "extraction_status": posting.extraction_status,
        "extraction_error": posting.extraction_error,
    }


def start_resume_build(
    account_id: int,
    job_posting_id: int,
    template: str = "onepage",
    name: str | None = None,
) -> dict[str, Any]:
    from app.api import resume_build

    _owned_posting(account_id, job_posting_id)
    _check_write(account_id, "/api/resume-build/start")
    body = resume_build.GenerateRequest(
        account_id=account_id, job_posting_id=job_posting_id, template=template, name=name
    )
    db = get_db()
    try:
        return _call(lambda: resume_build.start_build(body, db=db))
    finally:
        db.close()


def get_build_status(account_id: int, build_id: str) -> dict[str, Any]:
    from app.resume_build import background

    build = background.get(build_id)
    if build is None or build.account_id != account_id:
        raise _not_found("resume build", build_id)
    return build.public()


def build_server(account_id: int, *, read_only: bool = False) -> FastMCP:
    """An MCP server bound to one profile. read_only leaves out the tools
    that write or spend."""
    server = FastMCP("open-to-work", instructions=_INSTRUCTIONS)

    async def _run[T](fn: Callable[[], T]) -> T:
        # The wrapped code is synchronous (SQLAlchemy, the embedding model),
        # so it runs off the event loop the MCP session reads from.
        return await anyio.to_thread.run_sync(fn)

    @server.tool(name="search_evidence")
    async def _search_evidence(
        query: str, top_k: int = 10, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Searches this profile's skill evidence (claims drawn from its
        GitHub projects and work experience) for a skill, technology or
        job requirement. source narrows to 'repo' or 'experience'. Returns
        the best matches, each with the project or role it comes from."""
        return await _run(lambda: search_evidence(account_id, query, top_k, source))

    @server.tool(name="list_projects")
    async def _list_projects() -> list[dict[str, Any]]:
        """Lists this profile's projects (synced and hand-added repos), starred
        first, with language, skill count and extraction status."""
        return await _run(lambda: list_projects(account_id))

    @server.tool(name="get_build_status")
    async def _get_build_status(build_id: str) -> dict[str, Any]:
        """Reports a resume build's stage and state ('running',
        'awaiting_approval', 'done' or 'error'), and its result once done."""
        return await _run(lambda: get_build_status(account_id, build_id))

    if read_only:
        return server

    @server.tool(name="add_job_posting")
    async def _add_job_posting(
        text: str,
        company: str | None = None,
        title: str | None = None,
        location: str | None = None,
    ) -> dict[str, Any]:
        """Saves a job posting's text to this profile. The posting is read
        (skills, seniority, salary) in the background; the same text saved
        twice returns the first copy."""
        return await _run(lambda: add_job_posting(account_id, text, company, title, location))

    @server.tool(name="start_resume_build")
    async def _start_resume_build(
        job_posting_id: int, template: str = "onepage", name: str | None = None
    ) -> dict[str, Any]:
        """Starts a resume build tailored to one of this profile's job
        postings and returns its build id at once. template is 'onepage' or
        'twopage'. Follow it with get_build_status. The build runs inside
        this server's process, so it stops if the server is closed."""
        return await _run(
            lambda: start_resume_build(account_id, job_posting_id, template, name)
        )

    return server


def _account_exists(account_id: int) -> bool:
    db = get_db()
    try:
        return db.get(Account, account_id) is not None
    finally:
        db.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--account-id",
        type=int,
        default=int(os.environ.get("OTW_MCP_ACCOUNT_ID") or 0) or None,
        help="profile the tools act on (default: $OTW_MCP_ACCOUNT_ID)",
    )
    parser.add_argument(
        "--read-only", action="store_true", help="leave out add_job_posting and start_resume_build"
    )
    args = parser.parse_args(argv)
    if args.account_id is None:
        parser.error("--account-id or OTW_MCP_ACCOUNT_ID is required")

    from app.core.db import init_db
    from app.core.tracing import configure_tracing

    configure_tracing()
    init_db()
    if not _account_exists(args.account_id):
        parser.error(f"no profile with id {args.account_id}")
    # stdout is the MCP channel; logs go to stderr, as they already do.
    build_server(args.account_id, read_only=args.read_only).run("stdio")


if __name__ == "__main__":
    main()
