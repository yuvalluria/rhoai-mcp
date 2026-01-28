"""Evaluation plugin for RHOAI MCP server.

This module provides the EvaluationPlugin class that integrates
the evaluation harness with the MCP server.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager
from rhoai_mcp.evaluation.validation import create_default_validator
from rhoai_mcp.hooks import hookimpl
from rhoai_mcp.plugin import BasePlugin, PluginMetadata

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


class EvaluationPlugin(BasePlugin):
    """Plugin for agent evaluation and metrics collection.

    Provides MCP tools for starting/ending evaluation sessions,
    recording expectations, and generating evaluation reports.
    Also implements hooks to capture tool calls for metrics.
    """

    def __init__(self) -> None:
        """Initialize the evaluation plugin."""
        super().__init__(
            PluginMetadata(
                name="evaluation",
                version="1.0.0",
                description="Agent evaluation and metrics collection",
                maintainer="rhoai-mcp@redhat.com",
                requires_crds=[],
            )
        )
        self._session_manager = EvaluationSessionManager()
        self._validator = create_default_validator()

    @property
    def session_manager(self) -> EvaluationSessionManager:
        """Get the session manager instance."""
        return self._session_manager

    def get_active_session_id(self) -> str | None:
        """Get the currently active session ID.

        This is used by the instrumentation to associate tool
        calls with evaluation sessions.
        """
        return self._session_manager.active_session_id

    @hookimpl
    def rhoai_register_tools(self, mcp: FastMCP, server: RHOAIServer) -> None:
        """Register evaluation MCP tools."""
        from rhoai_mcp.domains.evaluation.tools import register_tools

        register_tools(mcp, server, self._session_manager, self._validator)
        logger.info("Registered evaluation tools")

    @hookimpl
    def rhoai_health_check(self, server: RHOAIServer) -> tuple[bool, str]:  # noqa: ARG002
        """Check plugin health.

        The evaluation plugin has no external dependencies and
        is always healthy.
        """
        return True, "Evaluation plugin ready"

    @hookimpl
    def rhoai_before_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],  # noqa: ARG002
        session_id: str | None,
    ) -> None:
        """Handle before tool call hook.

        Currently just logs for debugging. Could be extended to
        support pre-call validation or modification.
        """
        if session_id:
            logger.debug(f"Before tool call: {tool_name} (session: {session_id})")

    @hookimpl
    def rhoai_after_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        duration_ms: float,
        success: bool,
        error: str | None,
        session_id: str | None,
    ) -> None:
        """Handle after tool call hook.

        Records the tool call to the active session for metrics
        calculation.
        """
        # Skip recording evaluation tools to avoid infinite recursion
        if tool_name.startswith("evaluation_") or tool_name.startswith("eval_"):
            return

        if session_id:
            self._session_manager.record_tool_call(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                duration_ms=duration_ms,
                success=success,
                error=error,
                session_id=session_id,
            )
