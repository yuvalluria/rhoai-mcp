"""MCP tools for NeuralNav agent recommendation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def _base_url(server: "RHOAIServer") -> str | None:
    url = getattr(server.config, "neuralnav_backend_url", None)
    if url:
        return url.rstrip("/")
    return None


def register_tools(mcp: FastMCP, server: "RHOAIServer") -> None:
    @mcp.tool()
    def get_agent_recommendation(use_case: str) -> dict[str, Any]:
        """Get agent recommendation for a use case: agent name, single vs multi-agent, tools needed, and system prompt.

        Use this to see what kind of agent (single or multi-agent), which tools are needed, and the
        recommended system prompt for a given use case (e.g. chatbot_conversational, code_completion,
        document_analysis_rag) without running a full deployment recommendation.

        Args:
            use_case: Use case key (e.g. chatbot_conversational, code_completion, translation,
                document_analysis_rag, research_legal_analysis, long_document_summarization).

        Returns:
            Dict with recommended_agent, agent_type (single_agent | multi_agent), tools_needed (list),
            agent_explanation, recommended_system_prompt, eval_dataset_path.
        """
        base = _base_url(server)
        if not base:
            return {
                "error": "NeuralNav service URL not configured",
                "hint": "Set RHOAI_MCP_NEURALNAV_BACKEND_URL to the NeuralNav backend URL",
            }
        try:
            with httpx.Client(timeout=30) as client:
                r = client.get(f"{base}/api/agent-recommendation", params={"use_case": use_case})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Agent recommendation HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            try:
                err = e.response.json()
                return {"error": err.get("detail", str(e)), "status_code": e.response.status_code}
            except Exception:
                return {"error": e.response.text or str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Agent recommendation failed")
            return {"error": str(e)}
