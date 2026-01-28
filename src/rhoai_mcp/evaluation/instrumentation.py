"""Tool instrumentation for the evaluation harness.

This module provides functions to instrument MCP tools with
evaluation hooks for capturing tool call metrics.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def create_instrumented_tool_wrapper(
    original_func: F,
    tool_name: str,
    hook_caller: Any,
    session_provider: Callable[[], str | None],
) -> F:
    """Create an instrumented wrapper for a tool function.

    The wrapper emits before/after hooks for evaluation tracking.

    Args:
        original_func: The original tool function to wrap.
        tool_name: Name of the tool.
        hook_caller: The pluggy hook caller for emitting hooks.
        session_provider: Callable that returns the current session ID.

    Returns:
        A wrapped function that emits evaluation hooks.
    """
    if inspect.iscoroutinefunction(original_func):

        @functools.wraps(original_func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = session_provider()

            # Emit before hook
            try:
                hook_caller.rhoai_before_tool_call(
                    tool_name=tool_name,
                    arguments=kwargs,
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"Error in before_tool_call hook: {e}")

            # Execute the tool
            start_time = time.perf_counter()
            success = True
            error_msg: str | None = None
            result: Any = None

            try:
                result = await original_func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Emit after hook
                try:
                    hook_caller.rhoai_after_tool_call(
                        tool_name=tool_name,
                        arguments=kwargs,
                        result=result,
                        duration_ms=duration_ms,
                        success=success,
                        error=error_msg,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.debug(f"Error in after_tool_call hook: {e}")

        return async_wrapper  # type: ignore[return-value]
    else:

        @functools.wraps(original_func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = session_provider()

            # Emit before hook
            try:
                hook_caller.rhoai_before_tool_call(
                    tool_name=tool_name,
                    arguments=kwargs,
                    session_id=session_id,
                )
            except Exception as e:
                logger.debug(f"Error in before_tool_call hook: {e}")

            # Execute the tool
            start_time = time.perf_counter()
            success = True
            error_msg: str | None = None
            result: Any = None

            try:
                result = original_func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Emit after hook
                try:
                    hook_caller.rhoai_after_tool_call(
                        tool_name=tool_name,
                        arguments=kwargs,
                        result=result,
                        duration_ms=duration_ms,
                        success=success,
                        error=error_msg,
                        session_id=session_id,
                    )
                except Exception as e:
                    logger.debug(f"Error in after_tool_call hook: {e}")

        return sync_wrapper  # type: ignore[return-value]


class InstrumentedToolDecorator:
    """A decorator factory that creates instrumented tool decorators.

    This replaces the standard mcp.tool() decorator to add
    evaluation instrumentation to all tools.
    """

    def __init__(
        self,
        original_tool_decorator: Callable[..., Callable[[F], F]],
        hook_caller: Any,
        session_provider: Callable[[], str | None],
    ) -> None:
        """Initialize the instrumented decorator.

        Args:
            original_tool_decorator: The original mcp.tool() decorator.
            hook_caller: The pluggy hook caller for emitting hooks.
            session_provider: Callable that returns the current session ID.
        """
        self._original_tool = original_tool_decorator
        self._hook_caller = hook_caller
        self._session_provider = session_provider

    def __call__(self, *args: Any, **kwargs: Any) -> Callable[[F], F]:
        """Create an instrumented tool decorator.

        Supports both @tool and @tool() syntax.
        """
        # Handle @tool (no parentheses) case
        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            tool_name = func.__name__
            wrapped = create_instrumented_tool_wrapper(
                func, tool_name, self._hook_caller, self._session_provider
            )
            return self._original_tool(wrapped)  # type: ignore[return-value]

        # Handle @tool() or @tool(name="...", ...) case
        def decorator(func: F) -> F:
            tool_name = kwargs.get("name", func.__name__)
            wrapped = create_instrumented_tool_wrapper(
                func, tool_name, self._hook_caller, self._session_provider
            )
            return self._original_tool(*args, **kwargs)(wrapped)  # type: ignore[return-value, arg-type]

        return decorator  # type: ignore[return-value]


def instrument_mcp_tools(
    mcp: Any,
    hook_caller: Any,
    session_provider: Callable[[], str | None],
) -> None:
    """Replace mcp.tool with an instrumented version.

    This patches the FastMCP instance to use an instrumented
    tool decorator that emits evaluation hooks.

    Args:
        mcp: The FastMCP instance to instrument.
        hook_caller: The pluggy hook caller.
        session_provider: Callable that returns the current session ID.
    """
    original_tool = mcp.tool
    instrumented = InstrumentedToolDecorator(original_tool, hook_caller, session_provider)
    mcp.tool = instrumented
    logger.info("Instrumented MCP tools for evaluation tracking")
