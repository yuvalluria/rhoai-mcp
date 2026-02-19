"""MCP tools for prompt evaluation and optimization via NeuralNav backend."""

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
    def run_prompt_evaluation(
        prompt_text: str,
        dataset_url: str | None = None,
        dataset_path: str | None = None,
        limit: int | None = None,
        use_prompt_as_system: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a system prompt on a Q&A dataset and return metrics (e.g. accuracy score).

        Calls the NeuralNav backend to score the prompt. For Q&A datasets, set
        use_prompt_as_system=True so the prompt is used as the system message.

        Args:
            prompt_text: The system prompt to evaluate (or prompt template).
            dataset_url: Optional URL to a JSON dataset.
            dataset_path: Optional path to a JSON dataset (relative to service root).
            limit: Optional max number of examples to evaluate.
            use_prompt_as_system: If True, use prompt as system message (recommended for chatbot Q&A).

        Returns:
            Dict with metrics (e.g. overall_accuracy), count, and engine.
        """
        base = _base_url(server)
        if not base:
            return {
                "error": "NeuralNav service URL not configured",
                "hint": "Set RHOAI_MCP_NEURALNAV_BACKEND_URL to the NeuralNav backend URL",
            }
        payload: dict[str, Any] = {
            "prompt_text": prompt_text,
            "use_prompt_as_system": use_prompt_as_system,
        }
        if dataset_url:
            payload["dataset_url"] = dataset_url
        if dataset_path:
            payload["dataset_path"] = dataset_path
        if limit is not None:
            payload["limit"] = limit
        try:
            with httpx.Client(timeout=120) as client:
                r = client.post(f"{base}/api/evaluate-prompt", json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Prompt evaluation HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            try:
                err = e.response.json()
                return {"error": err.get("detail", str(e)), "status_code": e.response.status_code}
            except Exception:
                return {"error": e.response.text or str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Prompt evaluation failed")
            return {"error": str(e)}

    @mcp.tool()
    def run_prompt_optimization(
        prompt_text: str,
        dataset_url: str | None = None,
        dataset_path: str | None = None,
        optimization_target: str = "accuracy",
        max_iterations: int = 3,
        questions_answers: dict[str, Any] | None = None,
        use_prompt_as_system: bool = True,
        use_long_prompt_mode: bool = False,
        include_code_examples: bool = False,
    ) -> dict[str, Any]:
        """Optimize a system prompt using NeuralNav backend (tone, length, audience).

        Calls the NeuralNav backend. Optional questions_answers can bias the optimization
        (e.g. tone, response_length, domain).

        Args:
            prompt_text: The system prompt to optimize.
            dataset_url: Optional URL to a JSON dataset.
            dataset_path: Optional path to a JSON dataset.
            optimization_target: Target metric (default accuracy).
            max_iterations: Max optimization steps (default 3).
            questions_answers: Optional dict e.g. {"tone": "polite", "response_length": "brief", "domain": "retail"}.
            use_prompt_as_system: If True, treat prompt as system message (recommended for Q&A).
            use_long_prompt_mode: If True and prompt is long, use RLM-style (decompose/summarize then optimize).
            include_code_examples: If True, ask optimizer to include short code examples in the improved prompt.

        Returns:
            Dict with optimized_prompt, best_score, and optional optimized_prompt_path.
        """
        base = _base_url(server)
        if not base:
            return {
                "error": "NeuralNav service URL not configured",
                "hint": "Set RHOAI_MCP_NEURALNAV_BACKEND_URL to the NeuralNav backend URL",
            }
        payload: dict[str, Any] = {
            "prompt_text": prompt_text,
            "optimization_target": optimization_target,
            "max_iterations": max_iterations,
            "use_prompt_as_system": use_prompt_as_system,
            "use_long_prompt_mode": use_long_prompt_mode,
            "include_code_examples": include_code_examples,
        }
        if dataset_url:
            payload["dataset_url"] = dataset_url
        if dataset_path:
            payload["dataset_path"] = dataset_path
        if questions_answers:
            payload["questions_answers"] = questions_answers
        try:
            with httpx.Client(timeout=300) as client:
                r = client.post(f"{base}/api/optimize-prompt", json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Prompt optimization HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            try:
                err = e.response.json()
                return {"error": err.get("detail", err.get("error", str(e))), "hint": err.get("hint"), "status_code": e.response.status_code}
            except Exception:
                return {"error": e.response.text or str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Prompt optimization failed")
            return {"error": str(e)}

    @mcp.tool()
    def run_agent_optimization(
        agent_config: dict[str, Any],
        dataset_url: str | None = None,
        dataset_path: str | None = None,
        optimization_target: str = "accuracy",
        max_iterations: int = 3,
        questions_answers: dict[str, Any] | None = None,
        use_prompt_as_system: bool = True,
        use_long_prompt_mode: bool = False,
        include_code_examples: bool = False,
    ) -> dict[str, Any]:
        """Optimize an agent's system prompt. agent_config must have system_prompt; tools passed through.

        Calls the NeuralNav backend to optimize the agent's main prompt. Returns optimized_agent_config
        with improved system_prompt, best_score, and optimized_prompts.

        Args:
            agent_config: Dict with system_prompt (required); tools (optional, passed through).
            dataset_url: Optional URL to a JSON dataset.
            dataset_path: Optional path to a JSON dataset.
            optimization_target: Target metric (default accuracy).
            max_iterations: Max optimization steps (default 3).
            questions_answers: Optional dict e.g. {"tone": "polite", "domain": "retail"}.
            use_prompt_as_system: If True, treat prompt as system message.
            use_long_prompt_mode: If True and prompt is long, use RLM-style optimization.
            include_code_examples: If True, ask optimizer to include code examples.

        Returns:
            Dict with optimized_agent_config, best_score, optimized_prompts.
        """
        base = _base_url(server)
        if not base:
            return {
                "error": "NeuralNav service URL not configured",
                "hint": "Set RHOAI_MCP_NEURALNAV_BACKEND_URL to the NeuralNav backend URL",
            }
        payload: dict[str, Any] = {
            "agent_config": agent_config,
            "optimization_target": optimization_target,
            "max_iterations": max_iterations,
            "use_prompt_as_system": use_prompt_as_system,
            "use_long_prompt_mode": use_long_prompt_mode,
            "include_code_examples": include_code_examples,
        }
        if dataset_url:
            payload["dataset_url"] = dataset_url
        if dataset_path:
            payload["dataset_path"] = dataset_path
        if questions_answers:
            payload["questions_answers"] = questions_answers
        try:
            with httpx.Client(timeout=300) as client:
                r = client.post(f"{base}/api/optimize-agent", json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Agent optimization HTTP error: %s %s", e.response.status_code, e.response.text[:200])
            try:
                err = e.response.json()
                return {"error": err.get("detail", err.get("error", str(e))), "hint": err.get("hint"), "status_code": e.response.status_code}
            except Exception:
                return {"error": e.response.text or str(e), "status_code": e.response.status_code}
        except Exception as e:
            logger.exception("Agent optimization failed")
            return {"error": str(e)}
