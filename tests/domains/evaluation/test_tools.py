"""Tests for evaluation domain tools."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager
from rhoai_mcp.evaluation.validation import create_default_validator


class TestEvaluationTools:
    """Test evaluation MCP tools."""

    @pytest.fixture
    def mcp(self) -> MagicMock:
        """Create a mock FastMCP instance."""
        mcp = MagicMock()
        # Store registered tools
        mcp._tools = {}

        def mock_tool():
            def decorator(func):
                mcp._tools[func.__name__] = func
                return func
            return decorator

        mcp.tool = mock_tool
        return mcp

    @pytest.fixture
    def server(self) -> MagicMock:
        """Create a mock server."""
        return MagicMock()

    @pytest.fixture
    def session_manager(self) -> EvaluationSessionManager:
        """Create a session manager."""
        return EvaluationSessionManager()

    @pytest.fixture
    def validator(self):
        """Create a validator."""
        return create_default_validator()

    @pytest.fixture
    def registered_tools(self, mcp, server, session_manager, validator) -> dict:
        """Register tools and return them."""
        from rhoai_mcp.domains.evaluation.tools import register_tools

        register_tools(mcp, server, session_manager, validator)
        return mcp._tools

    def test_eval_start_session(self, registered_tools, session_manager) -> None:
        """Test starting an evaluation session."""
        result = registered_tools["eval_start_session"](
            name="Test Session",
            task_definition="Complete the task",
            expected_outcome="Task is completed",
        )

        assert "session_id" in result
        assert result["name"] == "Test Session"
        assert result["status"] == "active"
        assert session_manager.active_session_id is not None

    def test_eval_start_session_when_active(self, registered_tools, session_manager) -> None:
        """Test starting a session when one is already active."""
        registered_tools["eval_start_session"](
            name="First",
            task_definition="Task 1",
            expected_outcome="Outcome 1",
        )

        result = registered_tools["eval_start_session"](
            name="Second",
            task_definition="Task 2",
            expected_outcome="Outcome 2",
        )

        assert "error" in result
        assert "already active" in result["error"]

    def test_eval_end_session(self, registered_tools, session_manager) -> None:
        """Test ending an evaluation session."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_end_session"](
            task_completed=True,
            notes="All done",
        )

        assert result["status"] == "completed"
        assert result["task_completed"] is True
        assert session_manager.active_session_id is None

    def test_eval_end_session_no_active(self, registered_tools) -> None:
        """Test ending when no session is active."""
        result = registered_tools["eval_end_session"]()

        assert "error" in result

    def test_eval_cancel_session(self, registered_tools, session_manager) -> None:
        """Test cancelling a session."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_cancel_session"]()

        assert result["status"] == "cancelled"
        assert session_manager.active_session_id is None

    def test_eval_get_report(self, registered_tools, session_manager) -> None:
        """Test getting an evaluation report."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        # Record a tool call
        session_manager.record_tool_call(
            tool_name="test_tool",
            arguments={"arg": "value"},
            result={"status": "ok"},
            duration_ms=100.0,
            success=True,
        )

        result = registered_tools["eval_get_report"]()

        assert result["session_name"] == "Test"
        assert result["tool_count"] == 1
        assert result["success_count"] == 1

    def test_eval_get_composite_score(self, registered_tools, session_manager) -> None:
        """Test getting composite evaluation score."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        # Record some tool calls
        for i in range(3):
            session_manager.record_tool_call(
                tool_name=f"tool_{i}",
                arguments={},
                result={"ok": True},
                duration_ms=100.0,
                success=True,
            )

        result = registered_tools["eval_get_composite_score"]()

        assert "scores" in result
        assert "overall_score" in result
        assert "grade" in result

    def test_eval_list_sessions(self, registered_tools, session_manager) -> None:
        """Test listing sessions."""
        registered_tools["eval_start_session"](
            name="Session 1",
            task_definition="Task 1",
            expected_outcome="Outcome 1",
        )
        registered_tools["eval_end_session"]()

        registered_tools["eval_start_session"](
            name="Session 2",
            task_definition="Task 2",
            expected_outcome="Outcome 2",
        )

        result = registered_tools["eval_list_sessions"]()

        assert result["count"] == 2
        assert len(result["sessions"]) == 2

    def test_eval_list_sessions_by_status(self, registered_tools, session_manager) -> None:
        """Test listing sessions filtered by status."""
        registered_tools["eval_start_session"](
            name="Session 1",
            task_definition="Task 1",
            expected_outcome="Outcome 1",
        )
        registered_tools["eval_end_session"]()

        registered_tools["eval_start_session"](
            name="Session 2",
            task_definition="Task 2",
            expected_outcome="Outcome 2",
        )

        active_result = registered_tools["eval_list_sessions"](status="active")
        completed_result = registered_tools["eval_list_sessions"](status="completed")

        assert active_result["count"] == 1
        assert completed_result["count"] == 1

    def test_eval_get_session_status_active(self, registered_tools) -> None:
        """Test getting status of active session."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_get_session_status"]()

        assert result["active"] is True
        assert result["name"] == "Test"

    def test_eval_get_session_status_none(self, registered_tools) -> None:
        """Test getting status when no session is active."""
        result = registered_tools["eval_get_session_status"]()

        assert result["active"] is False

    def test_eval_add_expected_result(self, registered_tools, session_manager) -> None:
        """Test adding an expected result."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_add_expected_result"](
            tool_name="test_tool",
            required_fields=["name", "status"],
            field_values={"status": "ok"},
        )

        assert result["added"] is True
        session = session_manager.get_active_session()
        assert len(session.expected_results) == 1

    def test_eval_set_expected_trajectory(self, registered_tools, session_manager) -> None:
        """Test setting expected trajectory."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_set_expected_trajectory"](
            required_tools=["tool_a", "tool_b"],
            forbidden_tools=["tool_x"],
        )

        assert result["set"] is True
        session = session_manager.get_active_session()
        assert session.expected_sequence is not None
        assert session.expected_sequence.required_tools == ["tool_a", "tool_b"]

    def test_eval_set_trajectory_spec(self, registered_tools, session_manager) -> None:
        """Test setting trajectory specification."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_set_trajectory_spec"](
            goal_description="Complete the task",
            optimal_trajectory=["step1", "step2", "step3"],
            max_steps=10,
        )

        assert result["set"] is True
        session = session_manager.get_active_session()
        assert session.trajectory_spec is not None
        assert session.trajectory_spec.goal_description == "Complete the task"

    def test_eval_set_parameter_specs(self, registered_tools, session_manager) -> None:
        """Test setting parameter specifications."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        result = registered_tools["eval_set_parameter_specs"](
            tool_name="my_tool",
            parameters=[
                {"name": "arg1", "required": True, "expected_type": "string"},
                {"name": "arg2", "required": False, "expected_type": "int"},
            ],
        )

        assert result["set"] is True
        assert result["parameter_count"] == 2

    def test_eval_validate_session_results(self, registered_tools, session_manager) -> None:
        """Test validating session results."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        # Add expected result
        registered_tools["eval_add_expected_result"](
            tool_name="test_tool",
            required_fields=["status"],
        )

        # Record matching tool call
        session_manager.record_tool_call(
            tool_name="test_tool",
            arguments={},
            result={"status": "ok"},
            duration_ms=100.0,
            success=True,
        )

        result = registered_tools["eval_validate_session_results"]()

        assert result["validated_count"] == 1
        assert result["passed_count"] == 1
        assert result["all_passed"] is True

    def test_eval_get_stability_metrics(self, registered_tools, session_manager) -> None:
        """Test getting stability metrics."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        session_manager.record_tool_call(
            tool_name="test_tool",
            arguments={},
            result={"ok": True},
            duration_ms=100.0,
            success=True,
        )

        result = registered_tools["eval_get_stability_metrics"]()

        assert "metrics" in result
        assert result["metrics"]["total_calls"] == 1
        assert result["metrics"]["stability_score"] == 1.0

    def test_eval_get_performance_metrics(self, registered_tools, session_manager) -> None:
        """Test getting performance metrics."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        session_manager.record_tool_call(
            tool_name="test_tool",
            arguments={},
            result={"ok": True},
            duration_ms=150.0,
            success=True,
        )

        result = registered_tools["eval_get_performance_metrics"]()

        assert "metrics" in result
        assert result["metrics"]["avg_duration_ms"] == 150.0

    def test_eval_get_trajectory_analysis(self, registered_tools, session_manager) -> None:
        """Test getting trajectory analysis."""
        registered_tools["eval_start_session"](
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        session_manager.record_tool_call(
            tool_name="step1",
            arguments={},
            result={},
            duration_ms=100.0,
            success=True,
        )
        session_manager.record_tool_call(
            tool_name="step2",
            arguments={},
            result={},
            duration_ms=100.0,
            success=True,
        )

        result = registered_tools["eval_get_trajectory_analysis"]()

        assert "metrics" in result
        assert result["metrics"]["total_steps"] == 2
