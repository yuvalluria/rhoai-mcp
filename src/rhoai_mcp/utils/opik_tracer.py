"""Optional Opik observability: send MCP tool calls as traces to Opik (A) integration.

When RHOAI_MCP_OPIK_TRACE_API_URL and RHOAI_MCP_OPIK_TRACE_API_KEY are set, successful
tool calls are sent to Opik as traces. Which tools are traced depends on tool branch:
- deployment_only: get_deployment_recommendation
- agent_prompt: run_prompt_evaluation, run_prompt_optimization, get_agent_recommendation

No Opik SDK dependency; uses httpx to POST to Opik REST API. Fire-and-forget in a
daemon thread so tool latency is not affected. Failures are logged only.

Usage (e.g. when testing MCP from Claude Code):
  - Set RHOAI_MCP_OPIK_TRACE_API_KEY (and optionally RHOAI_MCP_OPIK_TRACE_API_URL,
    RHOAI_MCP_OPIK_TRACE_PROJECT) in your MCP server env (e.g. Cursor MCP config).
  - Run tools from chat; on success a trace is sent in the background.
  - In Opik (Comet), open the project (default "rhoai-mcp") to see traces: tool name,
    input/output, and timing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_TRACE_API_URL = os.environ.get("RHOAI_MCP_OPIK_TRACE_API_URL", "https://www.comet.com/opik/api").rstrip("/")
_TRACE_API_KEY = os.environ.get("RHOAI_MCP_OPIK_TRACE_API_KEY", "")
_TRACE_PROJECT = (os.environ.get("RHOAI_MCP_OPIK_TRACE_PROJECT") or "rhoai-mcp").strip()

_MAX_PAYLOAD_CHARS = 4000  # truncate input/output so we don't send huge bodies


def _truncate(obj: Any, max_chars: int = _MAX_PAYLOAD_CHARS) -> Any:
    """Truncate string values in dict for trace payloads; keep types JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:max_chars] + ("..." if len(obj) > max_chars else "")
    if isinstance(obj, dict):
        return {str(k): _truncate(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_truncate(x, max_chars) for x in obj][:50]
    return str(obj)[:max_chars]


def send_mcp_tool_trace(
    tool_name: str,
    input_payload: dict[str, Any],
    output_payload: dict[str, Any],
    project_name: str | None = None,
) -> None:
    """Send a single MCP tool call as a trace to Opik. No-op if env not set. Runs in background."""
    if not _TRACE_API_KEY or not _TRACE_API_URL:
        return
    project = (project_name or _TRACE_PROJECT).strip() or "rhoai-mcp"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    trace = {
        "name": tool_name,
        "start_time": now,
        "end_time": now,
        "input": _truncate(input_payload),
        "output": _truncate(output_payload),
        "project_name": project,
    }
    url = f"{_TRACE_API_URL}/v1/private/traces/batch"
    headers = {"Authorization": f"Bearer {_TRACE_API_KEY}", "Content-Type": "application/json"}
    body = {"traces": [trace]}

    def _post() -> None:
        try:
            import httpx
            raw = json.dumps(body, default=str)
            with httpx.Client(timeout=10) as client:
                r = client.post(url, content=raw, headers=headers)
            if r.status_code == 204:
                return
            if r.status_code in (401, 403):
                logger.warning(
                    "Opik trace auth failed (%s). Check RHOAI_MCP_OPIK_TRACE_API_KEY and URL.",
                    r.status_code,
                )
                return
            if r.status_code >= 400:
                logger.warning("Opik trace failed: %s %s", r.status_code, (r.text or "")[:300])
        except json.JSONEncodeError as e:
            logger.debug("Opik trace payload not JSON-serializable: %s", e)
        except Exception as e:
            logger.debug("Opik trace send failed: %s", e)

    t = threading.Thread(target=_post, daemon=True)
    t.start()
