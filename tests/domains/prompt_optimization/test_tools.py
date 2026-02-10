"""Tests for prompt_optimization MCP tools (run_prompt_evaluation, run_prompt_optimization)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestPromptOptimizationTools:
    """Test run_prompt_evaluation and run_prompt_optimization tools."""

    @pytest.fixture
    def mock_server(self) -> MagicMock:
        """Server with opik_service_url set."""
        server = MagicMock()
        server.config.opik_service_url = "http://127.0.0.1:8000"
        return server

    @pytest.fixture
    def mock_server_no_url(self) -> MagicMock:
        """Server without opik_service_url (tools should return error)."""
        server = MagicMock()
        server.config.opik_service_url = None
        return server

    def _capture_tools(self, mcp: MagicMock) -> dict[str, Any]:
        """Register tools and return callable dict."""
        registered: dict[str, Any] = {}

        def capture() -> Any:
            def deco(f: Any) -> Any:
                registered[f.__name__] = f
                return f
            return deco
        mcp.tool = capture
        return registered

    def test_run_prompt_evaluation_returns_error_when_url_not_configured(
        self, mock_server_no_url: MagicMock
    ) -> None:
        """When RHOAI_MCP_OPIK_SERVICE_URL is not set, evaluation returns error."""
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools

        mcp = MagicMock()
        registered = self._capture_tools(mcp)
        register_tools(mcp, mock_server_no_url)

        result = registered["run_prompt_evaluation"](
            prompt_text="You are a helpful assistant.",
            dataset_path="scripts/opik_optimization/test_dataset_small.json",
        )
        assert "error" in result
        assert "not configured" in result["error"].lower() or "OPIK" in result.get("hint", "")

    def test_run_prompt_evaluation_success(self, mock_server: MagicMock) -> None:
        """When backend is configured and returns 200, evaluation returns result."""
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools

        mcp = MagicMock()
        registered = self._capture_tools(mcp)
        register_tools(mcp, mock_server)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "overall_accuracy": 0.85,
            "count": 5,
            "engine": "rule_based",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rhoai_mcp.domains.prompt_optimization.tools.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_client)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_httpx.Client.return_value = mock_ctx

            result = registered["run_prompt_evaluation"](
                prompt_text="You are a helpful assistant. Answer briefly.",
                dataset_path="scripts/opik_optimization/test_dataset_small.json",
                limit=2,
                use_prompt_as_system=True,
            )

        assert "error" not in result
        assert result.get("overall_accuracy") == 0.85
        assert result.get("count") == 5
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:8000/api/evaluate-prompt"
        body = call_args[1]["json"]
        assert body["prompt_text"] == "You are a helpful assistant. Answer briefly."
        assert body["dataset_path"] == "scripts/opik_optimization/test_dataset_small.json"
        assert body["limit"] == 2
        assert body["use_prompt_as_system"] is True

    def test_run_prompt_evaluation_http_error(self, mock_server: MagicMock) -> None:
        """When backend returns 4xx/5xx, evaluation returns error dict."""
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools
        import httpx

        mcp = MagicMock()
        registered = self._capture_tools(mcp)
        register_tools(mcp, mock_server)

        with patch("rhoai_mcp.domains.prompt_optimization.tools.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_httpx.Client.return_value.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_resp.text = "Dataset not found"
            mock_resp.json.return_value = {"detail": "Dataset file not found"}
            mock_client.post.return_value = mock_resp
            mock_client.post.return_value.raise_for_status.side_effect = (
                httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)
            )

            result = registered["run_prompt_evaluation"](
                prompt_text="You are helpful.",
                dataset_path="scripts/opik_optimization/nonexistent.json",
            )

        assert "error" in result
        assert result.get("status_code") == 404

    def test_run_prompt_optimization_returns_error_when_url_not_configured(
        self, mock_server_no_url: MagicMock
    ) -> None:
        """When RHOAI_MCP_OPIK_SERVICE_URL is not set, optimization returns error."""
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools

        mcp = MagicMock()
        registered = self._capture_tools(mcp)
        register_tools(mcp, mock_server_no_url)

        result = registered["run_prompt_optimization"](
            prompt_text="You are a helpful assistant.",
            dataset_path="scripts/opik_optimization/test_dataset_small.json",
        )
        assert "error" in result

    def test_run_prompt_optimization_success(self, mock_server: MagicMock) -> None:
        """When backend is configured and returns 200, optimization returns result."""
        from rhoai_mcp.domains.prompt_optimization.tools import register_tools

        mcp = MagicMock()
        registered = self._capture_tools(mcp)
        register_tools(mcp, mock_server)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "optimized_prompt": "You are a concise, helpful assistant. Answer briefly and accurately.",
            "best_score": 0.92,
            "optimized_prompts": [{"prompt": "You are a concise...", "score": 0.92}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("rhoai_mcp.domains.prompt_optimization.tools.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_client)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_httpx.Client.return_value = mock_ctx
            mock_client.post.return_value = mock_response

            result = registered["run_prompt_optimization"](
                prompt_text="You are a helpful assistant.",
                dataset_path="scripts/opik_optimization/test_dataset_small.json",
                max_iterations=2,
                use_prompt_as_system=True,
            )

        assert "error" not in result
        assert "optimized_prompt" in result or "best_score" in result
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:8000/api/optimize-prompt"
        body = call_args[1]["json"]
        assert body["prompt_text"] == "You are a helpful assistant."
        assert body["dataset_path"] == "scripts/opik_optimization/test_dataset_small.json"
        assert body["max_iterations"] == 2
        assert body["use_prompt_as_system"] is True
