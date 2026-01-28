"""Session manager for evaluation harness.

This module provides the EvaluationSessionManager class that manages
evaluation sessions and tracks tool calls.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from rhoai_mcp.evaluation.models import (
    EvaluationReport,
    EvaluationSession,
    ExpectedResult,
    ExpectedToolSequence,
    ParameterSpec,
    SessionStatus,
    ToolCall,
    TrajectorySpec,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class EvaluationSessionManager:
    """Manages evaluation sessions and tracks tool calls.

    Provides methods to start/end sessions, record tool calls,
    and generate evaluation reports.
    """

    def __init__(self) -> None:
        """Initialize the session manager."""
        self._sessions: dict[str, EvaluationSession] = {}
        self._active_session_id: str | None = None

    @property
    def active_session_id(self) -> str | None:
        """Get the currently active session ID."""
        return self._active_session_id

    def get_active_session(self) -> EvaluationSession | None:
        """Get the currently active session."""
        if self._active_session_id:
            return self._sessions.get(self._active_session_id)
        return None

    def start_session(
        self,
        name: str,
        task_definition: str,
        expected_outcome: str,
    ) -> EvaluationSession:
        """Start a new evaluation session.

        Args:
            name: Human-readable session name.
            task_definition: Description of the task being evaluated.
            expected_outcome: Description of the expected outcome.

        Returns:
            The newly created session.

        Raises:
            ValueError: If there's already an active session.
        """
        if self._active_session_id:
            raise ValueError(
                f"Session '{self._active_session_id}' is already active. "
                "End it before starting a new one."
            )

        session_id = str(uuid.uuid4())
        session = EvaluationSession(
            session_id=session_id,
            name=name,
            task_definition=task_definition,
            expected_outcome=expected_outcome,
            status=SessionStatus.ACTIVE,
            start_time=datetime.now(timezone.utc),
        )

        self._sessions[session_id] = session
        self._active_session_id = session_id

        logger.info(f"Started evaluation session: {session_id} ({name})")
        return session

    def end_session(
        self,
        session_id: str | None = None,
        task_completed: bool = False,
        notes: str = "",
    ) -> EvaluationSession:
        """End an evaluation session.

        Args:
            session_id: Session ID to end (defaults to active session).
            task_completed: Whether the task was successfully completed.
            notes: Additional notes about the session.

        Returns:
            The ended session.

        Raises:
            ValueError: If the session doesn't exist or is not active.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session to end")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        if session.status != SessionStatus.ACTIVE:
            raise ValueError(f"Session '{sid}' is not active")

        session.status = SessionStatus.COMPLETED
        session.end_time = datetime.now(timezone.utc)
        session.task_completed = task_completed
        session.notes = notes

        if self._active_session_id == sid:
            self._active_session_id = None

        logger.info(
            f"Ended evaluation session: {sid} "
            f"(completed={task_completed}, tools={session.tool_count()})"
        )
        return session

    def cancel_session(self, session_id: str | None = None) -> EvaluationSession:
        """Cancel an evaluation session.

        Args:
            session_id: Session ID to cancel (defaults to active session).

        Returns:
            The cancelled session.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session to cancel")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        session.status = SessionStatus.CANCELLED
        session.end_time = datetime.now(timezone.utc)

        if self._active_session_id == sid:
            self._active_session_id = None

        logger.info(f"Cancelled evaluation session: {sid}")
        return session

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        duration_ms: float,
        success: bool,
        error: str | None = None,
        session_id: str | None = None,
    ) -> ToolCall | None:
        """Record a tool call to the session.

        Args:
            tool_name: Name of the tool that was called.
            arguments: Arguments passed to the tool.
            result: Result returned by the tool.
            duration_ms: Execution time in milliseconds.
            success: Whether the tool call succeeded.
            error: Error message if the call failed.
            session_id: Session to record to (defaults to active session).

        Returns:
            The recorded ToolCall, or None if no session is active.
        """
        sid = session_id or self._active_session_id
        if not sid:
            return None

        session = self._sessions.get(sid)
        if not session or session.status != SessionStatus.ACTIVE:
            return None

        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
            success=success,
            error=error,
        )

        session.tool_calls.append(tool_call)
        logger.debug(
            f"Recorded tool call: {tool_name} (success={success}, duration={duration_ms:.2f}ms)"
        )
        return tool_call

    def add_expected_result(
        self,
        tool_name: str,
        required_fields: list[str] | None = None,
        field_values: dict[str, Any] | None = None,
        field_patterns: dict[str, str] | None = None,
        custom_validator: str | None = None,
        session_id: str | None = None,
    ) -> ExpectedResult:
        """Add an expected result for validation.

        Args:
            tool_name: Tool name this expectation applies to.
            required_fields: Fields that must exist in the result.
            field_values: Exact field value matches required.
            field_patterns: Regex patterns for field values.
            custom_validator: Name of custom validator function.
            session_id: Session to add to (defaults to active session).

        Returns:
            The created ExpectedResult.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        expected = ExpectedResult(
            tool_name=tool_name,
            required_fields=required_fields or [],
            field_values=field_values or {},
            field_patterns=field_patterns or {},
            custom_validator=custom_validator,
        )

        session.expected_results.append(expected)
        return expected

    def set_expected_trajectory(
        self,
        required_tools: list[str] | None = None,
        optional_tools: list[str] | None = None,
        forbidden_tools: list[str] | None = None,
        expected_order: list[str] | None = None,
        session_id: str | None = None,
    ) -> ExpectedToolSequence:
        """Set the expected tool sequence for trajectory analysis.

        Args:
            required_tools: Tools that MUST be called.
            optional_tools: Tools that MAY be called.
            forbidden_tools: Tools that should NOT be called.
            expected_order: Expected call order (if order matters).
            session_id: Session to update (defaults to active session).

        Returns:
            The created ExpectedToolSequence.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        sequence = ExpectedToolSequence(
            required_tools=required_tools or [],
            optional_tools=optional_tools or [],
            forbidden_tools=forbidden_tools or [],
            expected_order=expected_order,
        )

        session.expected_sequence = sequence
        return sequence

    def set_trajectory_spec(
        self,
        goal_description: str,
        optimal_trajectory: list[str] | None = None,
        acceptable_trajectories: list[list[str]] | None = None,
        max_steps: int = 100,
        required_checkpoints: list[str] | None = None,
        session_id: str | None = None,
    ) -> TrajectorySpec:
        """Set the trajectory specification for evaluation.

        Args:
            goal_description: Description of the goal to achieve.
            optimal_trajectory: Optimal tool sequence.
            acceptable_trajectories: Alternative valid paths.
            max_steps: Maximum acceptable steps.
            required_checkpoints: Tools that must be called.
            session_id: Session to update (defaults to active session).

        Returns:
            The created TrajectorySpec.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        spec = TrajectorySpec(
            goal_description=goal_description,
            optimal_trajectory=optimal_trajectory or [],
            acceptable_trajectories=acceptable_trajectories or [],
            max_steps=max_steps,
            required_checkpoints=required_checkpoints or [],
        )

        session.trajectory_spec = spec
        return spec

    def set_parameter_specs(
        self,
        tool_name: str,
        parameters: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> list[ParameterSpec]:
        """Set parameter specifications for a tool.

        Args:
            tool_name: Tool name these specs apply to.
            parameters: List of parameter specification dicts.
            session_id: Session to update (defaults to active session).

        Returns:
            The created ParameterSpec list.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No active session")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        specs = [
            ParameterSpec(
                name=p["name"],
                required=p.get("required", True),
                expected_type=p.get("expected_type", "string"),
                pattern=p.get("pattern"),
                min_value=p.get("min_value"),
                max_value=p.get("max_value"),
                allowed_values=p.get("allowed_values"),
            )
            for p in parameters
        ]

        session.parameter_specs[tool_name] = specs
        return specs

    def get_session(self, session_id: str) -> EvaluationSession | None:
        """Get a session by ID.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            The session, or None if not found.
        """
        return self._sessions.get(session_id)

    def list_sessions(self, status: SessionStatus | None = None) -> list[EvaluationSession]:
        """List all sessions, optionally filtered by status.

        Args:
            status: Filter by session status.

        Returns:
            List of matching sessions.
        """
        sessions = list(self._sessions.values())
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        return sessions

    def generate_report(
        self,
        session_id: str | None = None,
        validator: Any = None,
    ) -> EvaluationReport:
        """Generate an evaluation report for a session.

        Args:
            session_id: Session ID (defaults to active session).
            validator: Optional ResultValidator for validation.

        Returns:
            The evaluation report.
        """
        sid = session_id or self._active_session_id
        if not sid:
            raise ValueError("No session specified")

        session = self._sessions.get(sid)
        if not session:
            raise ValueError(f"Session '{sid}' not found")

        # Run validation if validator provided
        validation_results: list[ValidationResult] = []
        if validator and session.expected_results:
            for call in session.tool_calls:
                # Find matching expected result
                expected = next(
                    (er for er in session.expected_results if er.tool_name == call.tool_name),
                    None,
                )
                if expected:
                    result = validator.validate(call.result, expected)
                    validation_results.append(result)

        return EvaluationReport.from_session(session, validation_results)

    def get_current_session_status(self) -> dict[str, Any]:
        """Get status of the current session.

        Returns:
            Dict with session status information.
        """
        if not self._active_session_id:
            return {"active": False, "message": "No active evaluation session"}

        session = self._sessions.get(self._active_session_id)
        if not session:
            return {"active": False, "message": "Active session not found"}

        return {
            "active": True,
            "session_id": session.session_id,
            "name": session.name,
            "task_definition": session.task_definition,
            "status": session.status.value,
            "tool_count": session.tool_count(),
            "success_count": session.success_count(),
            "error_count": session.error_count(),
            "duration_seconds": session.duration_seconds(),
        }

    def clear_sessions(self) -> None:
        """Clear all sessions (useful for testing)."""
        self._sessions.clear()
        self._active_session_id = None
