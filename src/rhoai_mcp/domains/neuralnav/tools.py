"""MCP tools for NeuralNav deployment recommendation (model + GPU + agent + system prompt)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from rhoai_mcp.server import RHOAIServer

logger = logging.getLogger(__name__)


def _base_url(server: "RHOAIServer") -> str | None:
    """NeuralNav backend base URL from config (reuses Opik service URL)."""
    url = getattr(server.config, "opik_service_url", None)
    if url:
        return url.rstrip("/")
    return None


def register_tools(
    mcp: FastMCP,
    server: "RHOAIServer",
    tools_set: str = "deployment_only",
) -> None:
    """Register NeuralNav tools with the MCP server.

    tools_set: "deployment_only" = only get_deployment_recommendation;
               "agent_only" = only get_agent_recommendation.
    """

    if tools_set != "agent_only":

        @mcp.tool()
        def get_deployment_recommendation(
            use_case: str,
            user_count: int,
            prompt_tokens: int = 512,
            output_tokens: int = 256,
            expected_qps: float = 10.0,
            ttft_target_ms: int = 500,
            itl_target_ms: int = 100,
            e2e_target_ms: int = 5000,
            percentile: str = "p95",
            preferred_gpu_types: list[str] | None = None,
            min_accuracy: int | None = None,
            max_cost: float | None = None,
            include_near_miss: bool = True,
        ) -> dict[str, Any]:
            """Get ranked deployment recommendations (model + GPU + agent + system prompt) from NeuralNav.

            Returns five ranked views: balanced, best_accuracy, lowest_cost, lowest_latency, simplest.
            Each list includes model name, GPU config, scores, and the response includes
            recommended_agent and recommended_system_prompt for the use case. Use this before
            deploying a model so the agent can recommend the right model and prompt.

            Args:
                use_case: Use case key (e.g. chatbot_conversational, code_completion, translation).
                user_count: Expected number of users (drives QPS and capacity).
                prompt_tokens: Typical prompt length (default 512).
                output_tokens: Typical output length (default 256).
                expected_qps: Expected queries per second (default 10).
                ttft_target_ms: Target time-to-first-token in ms (default 500).
                itl_target_ms: Target inter-token latency in ms (default 100).
                e2e_target_ms: Target end-to-end latency in ms (default 5000).
                percentile: Latency percentile (mean, p90, p95, p99; default p95).
                preferred_gpu_types: Optional list of GPU types to prefer (e.g. ["NVIDIA-H100", "NVIDIA-A100-80GB"]).
                min_accuracy: Optional minimum accuracy score filter.
                max_cost: Optional maximum monthly cost filter (USD).
                include_near_miss: Include configs that nearly meet SLOs (default True).

            Returns:
                Dict with balanced, best_accuracy, lowest_cost, lowest_latency, simplest (lists of configs),
                recommended_agent, recommended_system_prompt, prompt_eval_dataset_path, specification, and stats.
            """
            base = _base_url(server)
            if not base:
                return {
                    "error": "NeuralNav service URL not configured",
                    "hint": "Set RHOAI_MCP_OPIK_SERVICE_URL to the NeuralNav backend base URL (e.g. http://backend.neuralnav.svc.cluster.local:8000)",
                }
            payload: dict[str, Any] = {
                "use_case": use_case,
                "user_count": user_count,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "expected_qps": expected_qps,
                "ttft_target_ms": ttft_target_ms,
                "itl_target_ms": itl_target_ms,
                "e2e_target_ms": e2e_target_ms,
                "percentile": percentile,
                "include_near_miss": include_near_miss,
            }
            if preferred_gpu_types is not None:
                payload["preferred_gpu_types"] = preferred_gpu_types
            if min_accuracy is not None:
                payload["min_accuracy"] = min_accuracy
            if max_cost is not None:
                payload["max_cost"] = max_cost
            try:
                with httpx.Client(timeout=60) as client:
                    r = client.post(f"{base}/api/ranked-recommend-from-spec", json=payload)
                r.raise_for_status()
                result = r.json()
                if "error" not in result:
                    try:
                        from rhoai_mcp.utils.opik_tracer import send_mcp_tool_trace
                        send_mcp_tool_trace("get_deployment_recommendation", payload, result)
                    except Exception:  # noqa: S110
                        pass
                return result
            except httpx.HTTPStatusError as e:
                logger.warning("Deployment recommendation HTTP error: %s %s", e.response.status_code, e.response.text[:200])
                try:
                    err = e.response.json()
                    return {"error": err.get("detail", str(e)), "status_code": e.response.status_code}
                except Exception:
                    return {"error": e.response.text or str(e), "status_code": e.response.status_code}
            except Exception as e:
                logger.exception("Deployment recommendation failed")
                return {"error": str(e)}

    if tools_set != "deployment_only":

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
                    "hint": "Set RHOAI_MCP_OPIK_SERVICE_URL to the NeuralNav backend base URL",
                }
            try:
                with httpx.Client(timeout=30) as client:
                    r = client.get(f"{base}/api/agent-recommendation", params={"use_case": use_case})
                r.raise_for_status()
                result = r.json()
                if "error" not in result:
                    try:
                        from rhoai_mcp.utils.opik_tracer import send_mcp_tool_trace
                        send_mcp_tool_trace("get_agent_recommendation", {"use_case": use_case}, result)
                    except Exception:  # noqa: S110
                        pass
                return result
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
