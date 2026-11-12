"""app/agent/mcp_hub.py: the policy every tool call passes, which outside
servers a deployment connects to, and the agent working across the app's
own servers and an in-memory stand-in for GitHub's MCP server. Nothing
here reaches the network: the one "remote" server that is not in memory
points at a closed local port."""

import json

import anyio
import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

import app.core.db as db_module
from app.agent import mcp_hub, profile_agent
from app.core.chat_model import CompleteChatModel
from app.core.db import Account, Repository, get_db, init_db
from app.core.llm import LLMResponse
from app.core.settings import Settings, get_settings

# --- ToolPolicy ----------------------------------------------------------


def _policy():
    return mcp_hub.ToolPolicy(
        allowed={"profile": None, "github": frozenset({"get_file_contents"})},
        repos=frozenset({"ada/kube"}),
    )


def test_policy_lets_own_servers_through_whole():
    assert _policy().check("profile", "anything_it_lists", {}) is None


def test_policy_refuses_unknown_servers_and_unlisted_tools():
    policy = _policy()
    assert "not available" in policy.check("shell", "run", {})
    assert "not available" in policy.check("github", "push_files", {"owner": "ada", "repo": "kube"})


def test_policy_keeps_github_on_the_profiles_own_repos():
    policy = _policy()
    assert policy.check("github", "get_file_contents", {"owner": "Ada", "repo": "KUBE"}) is None
    reason = policy.check("github", "get_file_contents", {"owner": "evil", "repo": "kube"})
    assert "not one of this profile's projects" in reason
    assert policy.check("github", "get_file_contents", {}) is not None


def test_interceptor_refuses_without_calling_the_server():
    policy = _policy()
    called = []

    async def handler(request):
        called.append(request)
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    async def run(server, name, args):
        request = MCPToolCallRequest(name=name, args=args, server_name=server)
        return await policy(request, handler)

    refused = anyio.run(run, "github", "get_file_contents", {"owner": "evil", "repo": "x"})
    assert refused.isError and called == []
    assert policy.refused == ["github_get_file_contents"]

    allowed = anyio.run(run, "github", "get_file_contents", {"owner": "ada", "repo": "kube"})
    assert not allowed.isError and len(called) == 1


# --- remote_servers ------------------------------------------------------


def test_no_outside_server_unless_enabled_with_a_token():
    assert mcp_hub.remote_servers(Settings(mcp_github_enabled=False, github_token="t")) == {}
    assert mcp_hub.remote_servers(Settings(mcp_github_enabled=True, github_token="")) == {}


def test_github_server_is_read_only_streamable_http():
    servers = mcp_hub.remote_servers(Settings(mcp_github_enabled=True, github_token="tok"))
    github = servers["github"]
    assert github["transport"] == "streamable_http"
    assert github["url"] == "https://api.githubcopilot.com/mcp/x/repos/readonly"
    assert github["headers"] == {"Authorization": "Bearer tok", "X-MCP-Readonly": "true"}


# --- the hub with a stand-in GitHub server -------------------------------


def _fake_github():
    server = FastMCP("fake-github")
    reads = []

    @server.tool()
    def get_file_contents(owner: str, repo: str, path: str = "") -> str:
        """Reads a file."""
        reads.append(f"{owner}/{repo}/{path}")
        return f"contents of {owner}/{repo}/{path}: uses Helm charts"

    @server.tool()
    def push_files(owner: str, repo: str, message: str) -> str:
        """Writes files."""
        raise AssertionError("a write tool must never be called")

    @server.tool()
    def search_code(query: str) -> str:
        """Searches all of GitHub."""
        raise AssertionError("search reaches beyond the profile")

    server.reads = reads  # type: ignore[attr-defined]
    return server


@pytest.fixture
def profile(tmp_path, monkeypatch):
    import app.retrieval.vectorstore as vectorstore_module

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    db_module.reset_engine()
    vectorstore_module.get_client.cache_clear()
    get_settings.cache_clear()
    init_db()
    db = get_db()
    try:
        account = Account(first_name="Ada", last_name="L", github_username="ada")
        db.add(account)
        db.commit()
        db.add(Repository(account_id=account.id, github_id=1, name="kube", full_name="ada/kube",
                          url="https://github.com/ada/kube"))
        db.commit()
        account_id = account.id
    finally:
        db.close()
    yield account_id
    get_settings.cache_clear()


def test_hub_offers_only_allowed_tools_named_by_server(profile):
    async def run():
        async with mcp_hub.connect(profile, remote={"github": _fake_github()}) as hub:
            return sorted(t.name for t in hub.tools), hub.servers

    names, servers = anyio.run(run)
    assert servers == ["profile", "jobs", "github"]
    assert names == [
        "github_get_file_contents",
        "jobs_get_build_status",
        "jobs_list_job_postings",
        "profile_list_projects",
        "profile_search_evidence",
    ]


def test_hub_connects_no_outside_server_by_default(profile):
    async def run():
        async with mcp_hub.connect(profile) as hub:
            return hub.servers

    assert anyio.run(run) == ["profile", "jobs"]


class _Script:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, tier, messages, schema=None, account_id=None, purpose=None):
        self.calls.append(messages)
        return self.replies.pop(0)


def _decision(tool="", args=None, answer=""):
    parsed = {"tool": tool, "arguments_json": json.dumps(args or {}), "answer": answer}
    return LLMResponse(content=json.dumps(parsed), parsed=parsed, cost_usd=0.0, cached=False,
                       model="fake")


def test_agent_reads_own_repo_and_is_refused_someone_elses(profile):
    github = _fake_github()
    fake = _Script(
        _decision("github_get_file_contents", {"owner": "ada", "repo": "kube", "path": "a.yaml"}),
        _decision("github_get_file_contents", {"owner": "evil", "repo": "x", "path": "s"}),
        _decision(answer="ada/kube uses Helm."),
    )
    model = CompleteChatModel(account_id=profile, complete_fn=fake)
    result = anyio.run(
        lambda: profile_agent.ask(profile, "Helm?", model=model, remote={"github": github})
    )

    assert result.answer == "ada/kube uses Helm."
    assert github.reads == ["ada/kube/a.yaml"]
    own = fake.calls[1][-1]["content"]
    assert "uses Helm charts" in own
    refused = fake.calls[2][-1]["content"]
    assert "not one of this profile's projects" in refused
    assert result.servers == ["profile", "jobs", "github"]


def test_unreachable_outside_server_is_skipped(profile):
    from langchain_mcp_adapters.sessions import StreamableHttpConnection

    down = StreamableHttpConnection(transport="streamable_http", url="http://127.0.0.1:9/mcp")
    fake = _Script(_decision(answer="From the profile only."))
    model = CompleteChatModel(account_id=profile, complete_fn=fake)
    result = anyio.run(
        lambda: profile_agent.ask(profile, "Hi", model=model, remote={"github": down})
    )
    assert result.answer == "From the profile only."
    assert result.servers == ["profile", "jobs"]
    assert result.unavailable == ["github"]
