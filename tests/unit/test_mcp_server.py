"""app/mcp_server.py: each tool against a temp database, both called
directly and over a real (in-memory) MCP session, and the per-profile
rules: another profile's rows read as not found, the sample profile is
read-only in demo mode, and read_only drops the writing tools."""

from types import SimpleNamespace

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session

import app.core.db as db_module
import app.mcp_server as mcp_server
from app.core.db import Account, Experience, JobPosting, Repository, get_db, init_db
from app.core.settings import get_settings
from app.core.usage import BUILD_STARTED, count_today
from app.resume_build import background
from app.retrieval.search import Hit


@pytest.fixture(autouse=True)
def _db(tmp_path, monkeypatch):
    import app.retrieval.vectorstore as vectorstore_module

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    get_settings.cache_clear()
    init_db()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_posting_reads(monkeypatch):
    """Saving a posting starts its LLM read; not what these tests check."""
    monkeypatch.setattr("app.api.job_postings._start_extraction", lambda *a, **k: True)


def _add(row):
    db = get_db()
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()


def _account(name="Ada"):
    return _add(Account(first_name=name, last_name="L", github_username=name.lower()))


def _repo(account_id, name="ada/app", github_id=1):
    return _add(
        Repository(
            account_id=account_id,
            github_id=github_id,
            name=name.split("/")[-1],
            full_name=name,
            url=f"https://github.com/{name}",
            primary_language="Python",
        )
    )


def _posting(account_id, text="Senior Python engineer"):
    return _add(
        JobPosting(
            account_id=account_id,
            source="pasted",
            external_id=f"x-{account_id}-{text}",
            company="Acme",
            title="Engineer",
            raw_text_quarantined=text,
            content_hash=f"h-{account_id}-{text}",
            extraction_status="extracted",
        )
    )


def _call(server, name, args=None):
    async def run():
        async with create_connected_server_and_client_session(server) as session:
            return await session.call_tool(name, args or {})

    return anyio.run(run)


def _payload(result):
    """The tool's return value, from the structured copy FastMCP sends
    beside the text (lists are wrapped as {"result": [...]})."""
    assert not result.isError, result.content
    data = result.structuredContent
    return data["result"] if set(data) == {"result"} else data


def _tools(server):
    async def run():
        async with create_connected_server_and_client_session(server) as session:
            return {t.name for t in (await session.list_tools()).tools}

    return anyio.run(run)


# --- tool list -----------------------------------------------------------


def test_full_server_lists_every_tool():
    assert _tools(mcp_server.build_server(1)) == {
        "search_evidence",
        "list_projects",
        "list_job_postings",
        "add_job_posting",
        "start_resume_build",
        "get_build_status",
    }


def test_read_only_server_leaves_out_writes():
    assert _tools(mcp_server.build_server(1, read_only=True)) == {
        "search_evidence",
        "list_projects",
        "list_job_postings",
        "get_build_status",
    }


def test_profile_and_jobs_servers_split_the_tools():
    assert _tools(mcp_server.build_profile_server(1)) == {"search_evidence", "list_projects"}
    assert _tools(mcp_server.build_jobs_server(1)) == {
        "list_job_postings",
        "add_job_posting",
        "start_resume_build",
        "get_build_status",
    }
    assert _tools(mcp_server.build_jobs_server(1, read_only=True)) == {
        "list_job_postings",
        "get_build_status",
    }


@pytest.mark.parametrize(
    "name, expected",
    [("profile", 2), ("jobs", 2), ("all", 4)],
)
def test_build_named_picks_the_server(name, expected):
    assert len(_tools(mcp_server.build_named(name, 1, read_only=True))) == expected


def test_build_named_refuses_an_unknown_server():
    with pytest.raises(ValueError):
        mcp_server.build_named("billing", 1)


# --- list_job_postings ---------------------------------------------------


def test_list_job_postings_is_this_profile_only():
    mine, other = _account("Ada"), _account("Bob")
    _posting(mine, "Senior Python engineer")
    _posting(other, "Staff Go engineer")
    rows = _payload(_call(mcp_server.build_jobs_server(mine, read_only=True), "list_job_postings"))
    assert [(r["company"], r["title"]) for r in rows] == [("Acme", "Engineer")]
    assert set(rows[0]) >= {"id", "seniority", "work_mode", "salary_range", "saved_at"}
    assert "raw_text" not in rows[0]


# --- command line --------------------------------------------------------


def test_main_runs_the_named_server_over_http(monkeypatch):
    account = _account()
    ran = {}

    def fake_run(self, transport="stdio", mount_path=None):
        ran.update(name=self.name, transport=transport, host=self.settings.host,
                   port=self.settings.port)

    monkeypatch.setattr(mcp_server.FastMCP, "run", fake_run)
    monkeypatch.setattr("app.core.tracing.configure_tracing", lambda: None)
    mcp_server.main(
        ["--account-id", str(account), "--server", "profile", "--transport", "streamable-http",
         "--port", "9001"]
    )
    assert ran == {"name": "open-to-work-profile", "transport": "streamable-http",
                   "host": "127.0.0.1", "port": 9001}


# --- search_evidence -----------------------------------------------------


def _fake_search(monkeypatch, hits):
    seen = {}

    def fake(query, account_id, top_k=10, source_type=None):
        seen.update(query=query, account_id=account_id, top_k=top_k, source_type=source_type)
        return hits

    monkeypatch.setattr("app.retrieval.search.search_skill_evidence", fake)
    return seen


def test_search_evidence_names_projects_and_roles(monkeypatch):
    me = _account()
    repo_id = _repo(me)
    exp_id = _add(Experience(account_id=me, title="SRE", company="Initech"))
    seen = _fake_search(
        monkeypatch,
        [
            Hit(1, 0.91234, {"skill": "Python", "repo_id": repo_id, "source_type": "repo",
                             "evidence_type": "code", "confidence": 0.9, "account_id": me}),
            Hit(2, 0.5, {"skill": "Kubernetes", "experience_id": exp_id,
                         "source_type": "experience", "account_id": me}),
        ],
    )
    server = mcp_server.build_server(me)
    out = _payload(_call(server, "search_evidence", {"query": "python", "top_k": 99}))
    assert seen == {"query": "python", "account_id": me, "top_k": 25, "source_type": None}
    assert out[0]["skill"] == "Python"
    assert out[0]["project"] == "ada/app"
    assert out[0]["score"] == 0.9123
    assert out[1]["role"] == "SRE at Initech"
    assert out[1]["project"] is None


def test_search_evidence_drops_hits_from_another_profile(monkeypatch):
    me, other = _account(), _account("Bob")
    other_repo = _repo(other, "bob/x", github_id=2)
    _fake_search(
        monkeypatch,
        [Hit(1, 0.9, {"skill": "Go", "repo_id": other_repo, "source_type": "repo",
                      "account_id": other})],
    )
    assert mcp_server.search_evidence(me, "go") == []


def test_search_evidence_with_no_hits_skips_lookups(monkeypatch):
    seen = _fake_search(monkeypatch, [])
    assert mcp_server.search_evidence(1, "rust", top_k=0, source="repo") == []
    assert seen["top_k"] == 1 and seen["source_type"] == "repo"


@pytest.mark.parametrize("query, source", [("  ", None), ("python", "github")])
def test_search_evidence_refuses_bad_input(query, source):
    with pytest.raises(ToolError):
        mcp_server.search_evidence(1, query, source=source)


def test_tool_error_reaches_the_client_as_an_error_result():
    result = _call(mcp_server.build_server(1), "search_evidence", {"query": " "})
    assert result.isError
    assert "query is empty" in result.content[0].text


# --- list_projects -------------------------------------------------------


def test_list_projects_is_scoped_to_the_profile():
    me, other = _account(), _account("Bob")
    _repo(me, "ada/one", github_id=1)
    _repo(other, "bob/two", github_id=2)
    out = _payload(_call(mcp_server.build_server(me), "list_projects"))
    assert [p["name"] for p in out] == ["ada/one"]
    assert out[0]["language"] == "Python"
    assert out[0]["skill_count"] == 0


# --- add_job_posting -----------------------------------------------------


def test_add_job_posting_saves_to_this_profile():
    me = _account()
    out = _payload(
        _call(
            mcp_server.build_server(me),
            "add_job_posting",
            {"text": "Backend role, Go and Postgres", "company": "Acme", "title": "SWE"},
        )
    )
    db = get_db()
    try:
        row = db.get(JobPosting, out["id"])
        assert row.account_id == me
        assert row.company == "Acme"
    finally:
        db.close()
    assert out["extraction_status"] == row.extraction_status


def test_add_job_posting_turns_http_errors_into_tool_errors():
    me = _account()
    with pytest.raises(ToolError, match="no readable text"):
        mcp_server.add_job_posting(me, "   ")


# --- start_resume_build / get_build_status -------------------------------


def test_start_resume_build_and_follow_it(monkeypatch):
    me = _account()
    posting_id = _posting(me)
    started = {}

    def fake_start(body, label, checkpoint=None, retry_of=None):
        started.update(body=body, label=label)
        return background.Build(id="b1", account_id=body.account_id,
                                job_posting_id=body.job_posting_id, label=label)

    monkeypatch.setattr("app.api.resume_build._start_in_background", fake_start)
    server = mcp_server.build_server(me)
    out = _payload(_call(server, "start_resume_build", {"job_posting_id": posting_id}))
    assert out["id"] == "b1"
    assert started["body"].account_id == me
    assert started["body"].template == "onepage"
    assert started["label"] == "Engineer - Acme"
    assert count_today(me, BUILD_STARTED) == 1


def test_start_resume_build_refuses_another_profiles_posting():
    me, other = _account(), _account("Bob")
    theirs = _posting(other)
    with pytest.raises(ToolError, match="No job posting"):
        mcp_server.start_resume_build(me, theirs)


def test_start_resume_build_refuses_a_bad_template():
    me = _account()
    with pytest.raises(ToolError):
        mcp_server.start_resume_build(me, _posting(me), template="nope")


def test_get_build_status_only_for_this_profile(monkeypatch):
    build = background.Build(id="b2", account_id=7, job_posting_id=1, label="L")
    monkeypatch.setitem(background._builds, "b2", build)
    out = _payload(_call(mcp_server.build_server(7), "get_build_status", {"build_id": "b2"}))
    assert out["state"] == "running" and out["id"] == "b2"
    with pytest.raises(ToolError, match="No resume build"):
        mcp_server.get_build_status(8, "b2")
    with pytest.raises(ToolError):
        mcp_server.get_build_status(7, "missing")


# --- demo mode -----------------------------------------------------------


def _demo(monkeypatch, sample_id=0, **limits):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_SAMPLE_ACCOUNT_ID", str(sample_id))
    for key, value in limits.items():
        monkeypatch.setenv(key.upper(), str(value))
    get_settings.cache_clear()


def test_sample_profile_is_read_only_in_demo_mode(monkeypatch):
    sample = _account()
    _demo(monkeypatch, sample_id=sample)
    with pytest.raises(ToolError, match="read-only"):
        mcp_server.add_job_posting(sample, "Python role")
    with pytest.raises(ToolError, match="read-only"):
        mcp_server.start_resume_build(sample, _posting(sample))
    assert mcp_server.list_projects(sample) == []


def test_demo_daily_build_limit_applies(monkeypatch):
    me = _account()
    posting_id = _posting(me)
    _demo(monkeypatch, demo_daily_resume_builds=0)
    with pytest.raises(ToolError, match="resume builds a day"):
        mcp_server.start_resume_build(me, posting_id)


def test_demo_write_counts_against_daily_limit(monkeypatch):
    me = _account()
    _demo(monkeypatch, demo_daily_writes=0)
    with pytest.raises(ToolError, match="limit for changes"):
        mcp_server.add_job_posting(me, "Python role")


def test_demo_write_is_recorded_when_allowed(monkeypatch):
    from app.core.usage import DEMO_INTAKE, DEMO_WRITE

    me = _account()
    _demo(monkeypatch)
    mcp_server.add_job_posting(me, "Python role")
    assert count_today(me, DEMO_WRITE) == 1
    assert count_today(me, DEMO_INTAKE) == 1


# --- error detail and entry point ----------------------------------------


def test_detail_reads_dict_details():
    from fastapi import HTTPException

    assert mcp_server._detail(HTTPException(413, {"message": "too big"})) == "too big"
    assert mcp_server._detail(HTTPException(400, "plain")) == "plain"


def test_main_requires_an_account(monkeypatch):
    monkeypatch.delenv("OTW_MCP_ACCOUNT_ID", raising=False)
    with pytest.raises(SystemExit):
        mcp_server.main([])


def test_main_refuses_an_unknown_profile(monkeypatch):
    monkeypatch.setattr("app.core.tracing.configure_tracing", lambda: None)
    with pytest.raises(SystemExit):
        mcp_server.main(["--account-id", "999"])


def test_main_runs_stdio_for_the_profile(monkeypatch):
    me = _account()
    ran = {}

    def fake_build(account_id, *, read_only=False):
        ran.update(account_id=account_id, read_only=read_only)
        return SimpleNamespace(run=lambda transport: ran.update(transport=transport))

    monkeypatch.setattr("app.core.tracing.configure_tracing", lambda: None)
    monkeypatch.setattr(mcp_server, "build_server", fake_build)
    monkeypatch.setenv("OTW_MCP_ACCOUNT_ID", str(me))
    mcp_server.main(["--read-only"])
    assert ran == {"account_id": me, "read_only": True, "transport": "stdio"}
