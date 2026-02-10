"""MCP tools for context-efficient cluster and project summaries."""

from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    """Register summary tools with the MCP server.

    Placeholder: no tools registered. Add lightweight summary tools here
    (e.g. cluster/project overviews) to reduce token usage for agents.
    """
    pass
