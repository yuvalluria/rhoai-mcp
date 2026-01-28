"""Pytest configuration and fixtures for evaluation tests."""

from datetime import datetime, timezone

import pytest

from rhoai_mcp.evaluation.models import (
    EvaluationSession,
    ExpectedResult,
    ExpectedToolSequence,
    ParameterSpec,
    SessionStatus,
    ToolCall,
    TrajectorySpec,
)
from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager
from rhoai_mcp.evaluation.validation import create_default_validator


@pytest.fixture
def session_manager() -> EvaluationSessionManager:
    """Create a fresh session manager for testing."""
    return EvaluationSessionManager()


@pytest.fixture
def validator():
    """Create a validator with built-in validators registered."""
    return create_default_validator()


@pytest.fixture
def sample_tool_calls() -> list[ToolCall]:
    """Create a list of sample tool calls for testing."""
    return [
        ToolCall(
            tool_name="list_projects",
            arguments={},
            result={"items": ["project-a", "project-b"], "count": 2},
            duration_ms=150.0,
            success=True,
        ),
        ToolCall(
            tool_name="get_project_details",
            arguments={"name": "project-a"},
            result={"name": "project-a", "status": "Active"},
            duration_ms=200.0,
            success=True,
        ),
        ToolCall(
            tool_name="create_workbench",
            arguments={"project": "project-a", "name": "wb-1"},
            result={"name": "wb-1", "status": "Created"},
            duration_ms=500.0,
            success=True,
        ),
        ToolCall(
            tool_name="get_workbench_status",
            arguments={"project": "project-a", "name": "wb-1"},
            result={"name": "wb-1", "status": "Running"},
            duration_ms=100.0,
            success=True,
        ),
    ]


@pytest.fixture
def sample_tool_calls_with_errors() -> list[ToolCall]:
    """Create tool calls including some errors."""
    return [
        ToolCall(
            tool_name="list_projects",
            arguments={},
            result={"items": []},
            duration_ms=100.0,
            success=True,
        ),
        ToolCall(
            tool_name="get_project_details",
            arguments={"name": "nonexistent"},
            result=None,
            duration_ms=50.0,
            success=False,
            error="Project not found: nonexistent",
        ),
        ToolCall(
            tool_name="get_project_details",
            arguments={"name": "nonexistent"},
            result=None,
            duration_ms=55.0,
            success=False,
            error="Project not found: nonexistent",
        ),
        ToolCall(
            tool_name="list_workbenches",
            arguments={"project": "test"},
            result={"items": []},
            duration_ms=120.0,
            success=True,
        ),
    ]


@pytest.fixture
def sample_session(sample_tool_calls) -> EvaluationSession:
    """Create a sample evaluation session."""
    session = EvaluationSession(
        session_id="test-session-123",
        name="Test Session",
        task_definition="Create a workbench in project-a",
        expected_outcome="Workbench is created and running",
        status=SessionStatus.ACTIVE,
        tool_calls=sample_tool_calls,
        start_time=datetime.now(timezone.utc),
    )
    return session


@pytest.fixture
def sample_expected_sequence() -> ExpectedToolSequence:
    """Create a sample expected tool sequence."""
    return ExpectedToolSequence(
        required_tools=["list_projects", "create_workbench"],
        optional_tools=["get_project_details", "get_workbench_status"],
        forbidden_tools=["delete_project", "delete_workbench"],
        expected_order=["list_projects", "get_project_details", "create_workbench"],
    )


@pytest.fixture
def sample_trajectory_spec() -> TrajectorySpec:
    """Create a sample trajectory specification."""
    return TrajectorySpec(
        goal_description="Create a running workbench",
        optimal_trajectory=[
            "list_projects",
            "get_project_details",
            "create_workbench",
            "get_workbench_status",
        ],
        acceptable_trajectories=[
            ["list_projects", "create_workbench", "get_workbench_status"],
        ],
        max_steps=10,
        required_checkpoints=["create_workbench", "get_workbench_status"],
    )


@pytest.fixture
def sample_parameter_specs() -> list[ParameterSpec]:
    """Create sample parameter specifications."""
    return [
        ParameterSpec(
            name="project",
            required=True,
            expected_type="string",
            pattern=r"^[a-z][a-z0-9-]*$",
        ),
        ParameterSpec(
            name="name",
            required=True,
            expected_type="string",
        ),
        ParameterSpec(
            name="replicas",
            required=False,
            expected_type="int",
            min_value=1,
            max_value=10,
        ),
    ]


@pytest.fixture
def sample_expected_result() -> ExpectedResult:
    """Create a sample expected result."""
    return ExpectedResult(
        tool_name="get_project_details",
        required_fields=["name", "status"],
        field_values={"status": "Active"},
        field_patterns={"name": r"^project-.*$"},
    )
