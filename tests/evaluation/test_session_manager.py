"""Tests for the evaluation session manager."""

import pytest

from rhoai_mcp.evaluation.models import SessionStatus
from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager


class TestEvaluationSessionManager:
    """Test EvaluationSessionManager class."""

    def test_start_session(self, session_manager) -> None:
        """Test starting a new session."""
        session = session_manager.start_session(
            name="Test Session",
            task_definition="Do something",
            expected_outcome="Something done",
        )

        assert session.name == "Test Session"
        assert session.task_definition == "Do something"
        assert session.status == SessionStatus.ACTIVE
        assert session_manager.active_session_id == session.session_id

    def test_start_session_when_active_raises(self, session_manager) -> None:
        """Test that starting a session when one is active raises."""
        session_manager.start_session(
            name="First",
            task_definition="Task 1",
            expected_outcome="Outcome 1",
        )

        with pytest.raises(ValueError, match="already active"):
            session_manager.start_session(
                name="Second",
                task_definition="Task 2",
                expected_outcome="Outcome 2",
            )

    def test_end_session(self, session_manager) -> None:
        """Test ending a session."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        ended = session_manager.end_session(
            task_completed=True,
            notes="All done",
        )

        assert ended.status == SessionStatus.COMPLETED
        assert ended.task_completed is True
        assert ended.notes == "All done"
        assert ended.end_time is not None
        assert session_manager.active_session_id is None

    def test_end_session_no_active_raises(self, session_manager) -> None:
        """Test ending when no session is active raises."""
        with pytest.raises(ValueError, match="No active session"):
            session_manager.end_session()

    def test_cancel_session(self, session_manager) -> None:
        """Test cancelling a session."""
        session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        cancelled = session_manager.cancel_session()

        assert cancelled.status == SessionStatus.CANCELLED
        assert session_manager.active_session_id is None

    def test_record_tool_call(self, session_manager) -> None:
        """Test recording a tool call."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        call = session_manager.record_tool_call(
            tool_name="test_tool",
            arguments={"arg": "value"},
            result={"status": "ok"},
            duration_ms=100.0,
            success=True,
        )

        assert call is not None
        assert call.tool_name == "test_tool"
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0] == call

    def test_record_tool_call_no_session(self, session_manager) -> None:
        """Test recording when no session is active returns None."""
        call = session_manager.record_tool_call(
            tool_name="test",
            arguments={},
            result=None,
            duration_ms=10.0,
            success=True,
        )

        assert call is None

    def test_add_expected_result(self, session_manager) -> None:
        """Test adding an expected result."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        expected = session_manager.add_expected_result(
            tool_name="test_tool",
            required_fields=["field1", "field2"],
            field_values={"status": "ok"},
        )

        assert expected.tool_name == "test_tool"
        assert len(session.expected_results) == 1

    def test_set_expected_trajectory(self, session_manager) -> None:
        """Test setting expected trajectory."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        seq = session_manager.set_expected_trajectory(
            required_tools=["tool_a", "tool_b"],
            forbidden_tools=["tool_x"],
        )

        assert seq.required_tools == ["tool_a", "tool_b"]
        assert session.expected_sequence == seq

    def test_set_trajectory_spec(self, session_manager) -> None:
        """Test setting trajectory specification."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        spec = session_manager.set_trajectory_spec(
            goal_description="Complete the task",
            optimal_trajectory=["step1", "step2"],
            max_steps=5,
        )

        assert spec.goal_description == "Complete the task"
        assert session.trajectory_spec == spec

    def test_set_parameter_specs(self, session_manager) -> None:
        """Test setting parameter specifications."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        specs = session_manager.set_parameter_specs(
            tool_name="my_tool",
            parameters=[
                {"name": "arg1", "required": True, "expected_type": "string"},
                {"name": "arg2", "required": False, "expected_type": "int"},
            ],
        )

        assert len(specs) == 2
        assert "my_tool" in session.parameter_specs

    def test_get_session(self, session_manager) -> None:
        """Test getting a session by ID."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        retrieved = session_manager.get_session(session.session_id)

        assert retrieved == session

    def test_get_session_not_found(self, session_manager) -> None:
        """Test getting a non-existent session."""
        retrieved = session_manager.get_session("nonexistent")

        assert retrieved is None

    def test_list_sessions(self, session_manager) -> None:
        """Test listing sessions."""
        s1 = session_manager.start_session(
            name="Session 1",
            task_definition="Task 1",
            expected_outcome="Outcome 1",
        )
        session_manager.end_session()

        s2 = session_manager.start_session(
            name="Session 2",
            task_definition="Task 2",
            expected_outcome="Outcome 2",
        )

        all_sessions = session_manager.list_sessions()
        active = session_manager.list_sessions(status=SessionStatus.ACTIVE)
        completed = session_manager.list_sessions(status=SessionStatus.COMPLETED)

        assert len(all_sessions) == 2
        assert len(active) == 1
        assert len(completed) == 1

    def test_generate_report(self, session_manager) -> None:
        """Test generating an evaluation report."""
        session = session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        session_manager.record_tool_call(
            tool_name="tool1",
            arguments={},
            result={"ok": True},
            duration_ms=100.0,
            success=True,
        )

        report = session_manager.generate_report()

        assert report.session_id == session.session_id
        assert report.tool_count == 1
        assert report.success_count == 1

    def test_get_current_session_status_active(self, session_manager) -> None:
        """Test getting status of active session."""
        session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        status = session_manager.get_current_session_status()

        assert status["active"] is True
        assert status["name"] == "Test"

    def test_get_current_session_status_none(self, session_manager) -> None:
        """Test getting status when no session is active."""
        status = session_manager.get_current_session_status()

        assert status["active"] is False

    def test_clear_sessions(self, session_manager) -> None:
        """Test clearing all sessions."""
        session_manager.start_session(
            name="Test",
            task_definition="Task",
            expected_outcome="Outcome",
        )

        session_manager.clear_sessions()

        assert session_manager.active_session_id is None
        assert len(session_manager.list_sessions()) == 0
