"""The "ask about my profile" agent: a LangGraph loop over the MCP server's
read-only tools (app/mcp_server.py), with every model call made through
app/core/llm.py's complete()."""
