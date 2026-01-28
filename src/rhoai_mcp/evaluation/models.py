"""Data models for the evaluation harness.

This module defines the core data structures used for tracking
tool calls, evaluation sessions, and validation results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Status of an evaluation session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    """Record of a single tool invocation.

    Captures all relevant information about a tool call for
    later analysis and metrics calculation.
    """

    tool_name: str
    """Name of the tool that was called."""

    arguments: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned by the tool (may be None on error)."""

    duration_ms: float
    """Execution time in milliseconds."""

    success: bool
    """True if the tool completed without exception."""

    error: str | None = None
    """Error message if the tool raised an exception."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the tool call occurred."""

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this tool call."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self._serialize_result(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    def _serialize_result(self) -> Any:
        """Serialize result, handling non-JSON-serializable types."""
        if self.result is None:
            return None
        if isinstance(self.result, dict | list | str | int | float | bool):
            return self.result
        return str(self.result)


@dataclass
class ExpectedResult:
    """Expected result specification for validation.

    Defines what a tool result should look like for validation
    purposes. Supports multiple validation strategies.
    """

    tool_name: str
    """Name of the tool this expectation applies to."""

    required_fields: list[str] = field(default_factory=list)
    """Fields that must exist in the result."""

    field_values: dict[str, Any] = field(default_factory=dict)
    """Exact field value matches required."""

    field_patterns: dict[str, str] = field(default_factory=dict)
    """Regex patterns for field values."""

    custom_validator: str | None = None
    """Name of custom validator function to use."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "required_fields": self.required_fields,
            "field_values": self.field_values,
            "field_patterns": self.field_patterns,
            "custom_validator": self.custom_validator,
        }


@dataclass
class ValidationResult:
    """Result of validating a tool call against expectations.

    Contains detailed information about what passed and failed
    during validation.
    """

    passed: bool
    """True if all validations passed."""

    expected: ExpectedResult | None
    """The expected result that was validated against."""

    actual_result: Any
    """The actual result that was validated."""

    failures: list[str] = field(default_factory=list)
    """List of validation failure messages."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "expected": self.expected.to_dict() if self.expected else None,
            "actual_result": self.actual_result,
            "failures": self.failures,
        }


@dataclass
class ExpectedToolSequence:
    """Define expected tool sequence for a task.

    Used for trajectory analysis to determine if the agent
    followed an optimal path to the goal.
    """

    required_tools: list[str] = field(default_factory=list)
    """Tools that MUST be called."""

    optional_tools: list[str] = field(default_factory=list)
    """Tools that MAY be called."""

    forbidden_tools: list[str] = field(default_factory=list)
    """Tools that should NOT be called."""

    expected_order: list[str] | None = None
    """Expected call order (if order matters)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "required_tools": self.required_tools,
            "optional_tools": self.optional_tools,
            "forbidden_tools": self.forbidden_tools,
            "expected_order": self.expected_order,
        }


@dataclass
class TrajectorySpec:
    """Define expected trajectory for a task.

    Provides comprehensive trajectory expectations for
    multi-step trajectory evaluation.
    """

    goal_description: str
    """Description of the goal to achieve."""

    optimal_trajectory: list[str] = field(default_factory=list)
    """Optimal tool sequence to achieve the goal."""

    acceptable_trajectories: list[list[str]] = field(default_factory=list)
    """Alternative valid paths to the goal."""

    max_steps: int = 100
    """Maximum acceptable steps to complete the task."""

    required_checkpoints: list[str] = field(default_factory=list)
    """Tools that must be called at some point."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal_description": self.goal_description,
            "optimal_trajectory": self.optimal_trajectory,
            "acceptable_trajectories": self.acceptable_trajectories,
            "max_steps": self.max_steps,
            "required_checkpoints": self.required_checkpoints,
        }


@dataclass
class ParameterSpec:
    """Expected parameter specification for precision analysis.

    Defines what a tool parameter should look like for
    parameter precision evaluation.
    """

    name: str
    """Parameter name."""

    required: bool = True
    """Whether the parameter is required."""

    expected_type: str = "string"
    """Expected type: 'string', 'int', 'float', 'bool', 'list', 'dict'."""

    pattern: str | None = None
    """Regex pattern for string validation."""

    min_value: float | None = None
    """Minimum value for numeric types."""

    max_value: float | None = None
    """Maximum value for numeric types."""

    allowed_values: list[Any] | None = None
    """Enum of allowed values."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "required": self.required,
            "expected_type": self.expected_type,
            "pattern": self.pattern,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
        }


@dataclass
class EvaluationSession:
    """An evaluation session tracking tool calls and metrics.

    Represents a complete evaluation session from start to end,
    including all tool calls, expectations, and computed metrics.
    """

    session_id: str
    """Unique identifier for this session."""

    name: str
    """Human-readable session name."""

    task_definition: str
    """Description of the task being evaluated."""

    expected_outcome: str
    """Description of the expected outcome."""

    status: SessionStatus = SessionStatus.ACTIVE
    """Current session status."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """List of all tool calls in this session."""

    expected_results: list[ExpectedResult] = field(default_factory=list)
    """Expected results for validation."""

    expected_sequence: ExpectedToolSequence | None = None
    """Expected tool sequence for trajectory analysis."""

    trajectory_spec: TrajectorySpec | None = None
    """Trajectory specification for evaluation."""

    parameter_specs: dict[str, list[ParameterSpec]] = field(default_factory=dict)
    """Parameter specifications by tool name."""

    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the session started."""

    end_time: datetime | None = None
    """When the session ended (if completed)."""

    task_completed: bool = False
    """Whether the task was successfully completed."""

    notes: str = ""
    """Additional notes about the session."""

    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def tool_count(self) -> int:
        """Count total tool calls in this session."""
        return len(self.tool_calls)

    def success_count(self) -> int:
        """Count successful tool calls."""
        return sum(1 for tc in self.tool_calls if tc.success)

    def error_count(self) -> int:
        """Count failed tool calls."""
        return sum(1 for tc in self.tool_calls if not tc.success)

    def get_tool_sequence(self) -> list[str]:
        """Get the sequence of tool names called."""
        return [tc.tool_name for tc in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "task_definition": self.task_definition,
            "expected_outcome": self.expected_outcome,
            "status": self.status.value,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "expected_results": [er.to_dict() for er in self.expected_results],
            "expected_sequence": (
                self.expected_sequence.to_dict() if self.expected_sequence else None
            ),
            "trajectory_spec": (self.trajectory_spec.to_dict() if self.trajectory_spec else None),
            "parameter_specs": {
                tool: [ps.to_dict() for ps in specs] for tool, specs in self.parameter_specs.items()
            },
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "task_completed": self.task_completed,
            "notes": self.notes,
            "summary": {
                "duration_seconds": self.duration_seconds(),
                "tool_count": self.tool_count(),
                "success_count": self.success_count(),
                "error_count": self.error_count(),
            },
        }


@dataclass
class EvaluationReport:
    """Summary report for an evaluation session.

    Contains computed metrics and analysis for a completed
    evaluation session.
    """

    session_id: str
    """Session this report is for."""

    session_name: str
    """Human-readable session name."""

    task_definition: str
    """Description of the task that was evaluated."""

    task_completed: bool
    """Whether the task was successfully completed."""

    duration_seconds: float
    """Total session duration."""

    tool_count: int
    """Total number of tool calls."""

    success_count: int
    """Number of successful tool calls."""

    error_count: int
    """Number of failed tool calls."""

    success_rate: float
    """Percentage of successful tool calls."""

    tool_sequence: list[str]
    """Sequence of tools called."""

    validation_results: list[ValidationResult] = field(default_factory=list)
    """Results of validating tool calls against expectations."""

    # Metrics summaries (populated after metrics calculation)
    stability_score: float | None = None
    performance_score: float | None = None
    tool_selection_score: float | None = None
    parameter_precision_score: float | None = None
    trajectory_score: float | None = None
    composite_score: float | None = None
    grade: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "task_definition": self.task_definition,
            "task_completed": self.task_completed,
            "duration_seconds": self.duration_seconds,
            "tool_count": self.tool_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "tool_sequence": self.tool_sequence,
            "validation_results": [vr.to_dict() for vr in self.validation_results],
            "scores": {
                "stability": self.stability_score,
                "performance": self.performance_score,
                "tool_selection": self.tool_selection_score,
                "parameter_precision": self.parameter_precision_score,
                "trajectory": self.trajectory_score,
                "composite": self.composite_score,
                "grade": self.grade,
            },
        }

    @classmethod
    def from_session(
        cls,
        session: EvaluationSession,
        validation_results: list[ValidationResult] | None = None,
    ) -> EvaluationReport:
        """Create a report from a session."""
        tool_count = session.tool_count()
        success_count = session.success_count()
        success_rate = (success_count / tool_count * 100) if tool_count > 0 else 0.0

        return cls(
            session_id=session.session_id,
            session_name=session.name,
            task_definition=session.task_definition,
            task_completed=session.task_completed,
            duration_seconds=session.duration_seconds(),
            tool_count=tool_count,
            success_count=success_count,
            error_count=session.error_count(),
            success_rate=success_rate,
            tool_sequence=session.get_tool_sequence(),
            validation_results=validation_results or [],
        )
