"""The MCP hub the profile agent works through: every tool it can call
comes from an MCP server, the app's own and outside ones alike, behind one
policy.

Servers:

- profile, jobs: the app's own (app/mcp_server.py), built read-only and
  bound to the asking profile. Connected in memory: each question gets
  servers for its own profile, which a long-running HTTP server fixed to
  one profile at startup could not give, and the protocol is the same
  either way.
- github: GitHub's remote MCP server over streamable HTTP, only when
  MCP_GITHUB_ENABLED and GITHUB_TOKEN are both set. Live reads of the
  profile's repos (files, commits, releases) beyond what the last sync
  stored.

Tool names carry their server ("profile_search_evidence",
"github_get_file_contents"), so the model and the answer's trail both say
where a fact came from.

Policy (ToolPolicy), applied twice: a tool not on the server's allowlist
is never handed to the model, and every call is checked again as it
happens (a langchain-mcp-adapters interceptor), so a model that names a
tool or a repo it was not given gets an error back instead of a result.
For github, the call must name a repo of the asking profile: the agent
reads text other people wrote, and nothing in it may point the agent at
someone else's code. An outside server that is down or refuses is
skipped; the agent still runs on the app's own servers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.sessions import Connection, StreamableHttpConnection
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent
from sqlalchemy import select

from app.core.db import Repository, get_db
from app.core.settings import Settings, get_settings
from app.mcp_server import build_jobs_server, build_profile_server

logger = logging.getLogger(__name__)

# GitHub tools the agent may use: reads scoped to one repo, each taking
# owner and repo. Intersected with what the server actually lists, so a
# name GitHub drops or renames is simply not offered. Search tools are
# left out on purpose: they reach beyond the profile's own repos.
GITHUB_READ_TOOLS = frozenset(
    {
        "get_file_contents",
        "list_commits",
        "get_commit",
        "list_branches",
        "list_tags",
        "list_releases",
        "get_latest_release",
    }
)

Source = FastMCP | Connection


@dataclass
class ToolPolicy:
    """What the agent may call. `allowed` maps a server to the tool names
    it may use there (None: every tool that server lists, for the app's
    own read-only servers). `repos` are the asking profile's repos as
    "owner/name", casefolded."""

    allowed: dict[str, frozenset[str] | None]
    repos: frozenset[str] = frozenset()
    refused: list[str] = field(default_factory=list)

    def lists(self, server: str, tool: str) -> bool:
        if server not in self.allowed:
            return False
        names = self.allowed[server]
        return names is None or tool in names

    def check(self, server: str, tool: str, args: Mapping[str, Any]) -> str | None:
        """None when the call may go ahead, otherwise why not."""
        if not self.lists(server, tool):
            return f"{server}_{tool} is not available to this agent."
        if server == "github":
            owner, repo = str(args.get("owner") or ""), str(args.get("repo") or "")
            if f"{owner}/{repo}".casefold() not in self.repos:
                return (
                    f"{owner}/{repo} is not one of this profile's projects. "
                    "Use list_projects for the repos you may read."
                )
        return None

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        reason = self.check(request.server_name, request.name, request.args)
        if reason is not None:
            self.refused.append(f"{request.server_name}_{request.name}")
            logger.info("mcp hub refused %s_%s: %s", request.server_name, request.name, reason)
            return CallToolResult(content=[TextContent(type="text", text=reason)], isError=True)
        return await handler(request)


def remote_servers(settings: Settings) -> dict[str, Connection]:
    """The outside servers this deployment is configured for."""
    servers: dict[str, Connection] = {}
    if settings.mcp_github_enabled and settings.github_token:
        servers["github"] = StreamableHttpConnection(
            transport="streamable_http",
            url=settings.mcp_github_url,
            headers={
                "Authorization": f"Bearer {settings.github_token}",
                # Also asked for in the URL; both, so a URL edited to the
                # default toolset still gets read-only tools.
                "X-MCP-Readonly": "true",
            },
        )
    return servers


def _profile_repos(account_id: int) -> frozenset[str]:
    db = get_db()
    try:
        names = db.execute(
            select(Repository.full_name).where(Repository.account_id == account_id)
        ).scalars()
        return frozenset(n.casefold() for n in names if n)
    finally:
        db.close()


@dataclass
class Hub:
    tools: list[BaseTool]
    servers: list[str]
    unavailable: list[str]
    policy: ToolPolicy


async def _load(
    stack: AsyncExitStack, name: str, source: Source, policy: ToolPolicy
) -> list[BaseTool]:
    common: dict[str, Any] = {
        "server_name": name,
        "tool_name_prefix": True,
        "tool_interceptors": [policy],
    }
    if isinstance(source, FastMCP):
        session = await stack.enter_async_context(
            create_connected_server_and_client_session(source)
        )
        tools = await load_mcp_tools(session, **common)
    else:
        # No session: the adapter opens one per call, which suits a remote
        # HTTP server and keeps nothing open between questions.
        tools = await load_mcp_tools(None, connection=source, **common)
    prefix = f"{name}_"
    return [t for t in tools if policy.lists(name, t.name.removeprefix(prefix))]


@asynccontextmanager
async def connect(
    account_id: int, *, remote: Mapping[str, Source] | None = None
) -> AsyncIterator[Hub]:
    """Opens the hub for one question about `account_id`. `remote`
    replaces the configured outside servers (tests pass an in-memory
    stand-in)."""
    outside = dict(remote_servers(get_settings()) if remote is None else remote)
    sources: dict[str, Source] = {
        "profile": build_profile_server(account_id),
        "jobs": build_jobs_server(account_id, read_only=True),
        **outside,
    }
    policy = ToolPolicy(
        allowed={
            "profile": None,
            "jobs": None,
            **{name: GITHUB_READ_TOOLS if name == "github" else frozenset() for name in outside},
        },
        repos=_profile_repos(account_id) if "github" in outside else frozenset(),
    )
    async with AsyncExitStack() as stack:
        tools: list[BaseTool] = []
        servers: list[str] = []
        unavailable: list[str] = []
        for name, source in sources.items():
            try:
                tools.extend(await _load(stack, name, source, policy))
            except Exception:
                if name not in outside:
                    raise  # the app's own servers failing is a bug, not an outage
                logger.warning("mcp hub: outside server %s unavailable", name, exc_info=True)
                unavailable.append(name)
                continue
            servers.append(name)
        yield Hub(tools=tools, servers=servers, unavailable=unavailable, policy=policy)
