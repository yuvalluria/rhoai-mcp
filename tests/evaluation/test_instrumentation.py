"""Tests for tool instrumentation."""

from unittest.mock import MagicMock, patch
import time

import pytest

from rhoai_mcp.evaluation.instrumentation import (
    InstrumentedToolDecorator,
    create_instrumented_tool_wrapper,
    instrument_mcp_tools,
)


class TestCreateInstrumentedToolWrapper:
    """Test create_instrumented_tool_wrapper function."""

    def test_wraps_sync_function(self) -> None:
        """Test wrapping a synchronous function."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        def original_func(arg1: str, arg2: int = 10) -> dict:
            return {"arg1": arg1, "arg2": arg2}

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        result = wrapped(arg1="hello", arg2=42)

        assert result == {"arg1": "hello", "arg2": 42}
        hook_caller.rhoai_before_tool_call.assert_called_once()
        hook_caller.rhoai_after_tool_call.assert_called_once()

    def test_captures_success(self) -> None:
        """Test that successful calls are captured correctly."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        def original_func() -> str:
            return "success"

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        wrapped()

        call_args = hook_caller.rhoai_after_tool_call.call_args
        assert call_args.kwargs["success"] is True
        assert call_args.kwargs["error"] is None
        assert call_args.kwargs["result"] == "success"

    def test_captures_errors(self) -> None:
        """Test that errors are captured correctly."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        def original_func() -> None:
            raise ValueError("Something went wrong")

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        with pytest.raises(ValueError):
            wrapped()

        call_args = hook_caller.rhoai_after_tool_call.call_args
        assert call_args.kwargs["success"] is False
        assert "Something went wrong" in call_args.kwargs["error"]

    def test_captures_duration(self) -> None:
        """Test that duration is captured."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        def original_func() -> None:
            time.sleep(0.01)  # Sleep 10ms

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        wrapped()

        call_args = hook_caller.rhoai_after_tool_call.call_args
        assert call_args.kwargs["duration_ms"] >= 10.0

    def test_passes_session_id(self) -> None:
        """Test that session ID is passed to hooks."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="my-session-id")

        def original_func() -> None:
            pass

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        wrapped()

        before_args = hook_caller.rhoai_before_tool_call.call_args
        after_args = hook_caller.rhoai_after_tool_call.call_args

        assert before_args.kwargs["session_id"] == "my-session-id"
        assert after_args.kwargs["session_id"] == "my-session-id"

    def test_handles_no_session(self) -> None:
        """Test that None session ID is handled."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value=None)

        def original_func() -> str:
            return "ok"

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        result = wrapped()

        assert result == "ok"
        after_args = hook_caller.rhoai_after_tool_call.call_args
        assert after_args.kwargs["session_id"] is None

    def test_hook_errors_dont_break_function(self) -> None:
        """Test that hook errors don't break the wrapped function."""
        hook_caller = MagicMock()
        hook_caller.rhoai_before_tool_call.side_effect = Exception("Hook error")
        session_provider = MagicMock(return_value="session-123")

        def original_func() -> str:
            return "success"

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "test_tool",
            hook_caller,
            session_provider,
        )

        # Should not raise despite hook error
        result = wrapped()
        assert result == "success"


class TestInstrumentedToolDecorator:
    """Test InstrumentedToolDecorator class."""

    def test_decorator_with_parentheses(self) -> None:
        """Test @tool() syntax."""
        original_tool = MagicMock()
        original_tool.return_value = lambda f: f

        hook_caller = MagicMock()
        session_provider = MagicMock(return_value=None)

        decorator = InstrumentedToolDecorator(
            original_tool,
            hook_caller,
            session_provider,
        )

        @decorator(name="custom_name")
        def my_tool() -> str:
            return "result"

        # Original decorator should have been called
        original_tool.assert_called()

    def test_decorator_without_parentheses(self) -> None:
        """Test @tool syntax (no parentheses)."""
        original_tool = MagicMock()
        original_tool.return_value = lambda f: f

        hook_caller = MagicMock()
        session_provider = MagicMock(return_value=None)

        decorator = InstrumentedToolDecorator(
            original_tool,
            hook_caller,
            session_provider,
        )

        def my_tool() -> str:
            return "result"

        decorated = decorator(my_tool)

        # Should work as a decorator
        assert callable(decorated)


class TestInstrumentMcpTools:
    """Test instrument_mcp_tools function."""

    def test_replaces_tool_decorator(self) -> None:
        """Test that mcp.tool is replaced."""
        mcp = MagicMock()
        original_tool = mcp.tool

        hook_caller = MagicMock()
        session_provider = MagicMock(return_value=None)

        instrument_mcp_tools(mcp, hook_caller, session_provider)

        # The tool attribute should be replaced
        assert mcp.tool != original_tool
        assert isinstance(mcp.tool, InstrumentedToolDecorator)


class TestAsyncFunctionWrapper:
    """Test wrapping async functions."""

    @pytest.mark.asyncio
    async def test_wraps_async_function(self) -> None:
        """Test wrapping an async function."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        async def original_func(value: int) -> int:
            return value * 2

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "async_tool",
            hook_caller,
            session_provider,
        )

        result = await wrapped(value=21)

        assert result == 42
        hook_caller.rhoai_before_tool_call.assert_called_once()
        hook_caller.rhoai_after_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_captures_errors(self) -> None:
        """Test that async errors are captured."""
        hook_caller = MagicMock()
        session_provider = MagicMock(return_value="session-123")

        async def original_func() -> None:
            raise RuntimeError("Async error")

        wrapped = create_instrumented_tool_wrapper(
            original_func,
            "async_tool",
            hook_caller,
            session_provider,
        )

        with pytest.raises(RuntimeError):
            await wrapped()

        call_args = hook_caller.rhoai_after_tool_call.call_args
        assert call_args.kwargs["success"] is False
        assert "Async error" in call_args.kwargs["error"]
