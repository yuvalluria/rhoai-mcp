"""Evaluation harness for RHOAI MCP server.

This package provides tools for tracking and evaluating agent performance
when using the server's MCP tools. It captures metrics across six evaluation
dimensions: stability, performance, tool selection, success rate, parameter
precision, and trajectory analysis.

Usage:
    Set RHOAI_MCP_ENABLE_EVALUATION=true to enable the evaluation harness.
    Then use the eval_* MCP tools to manage evaluation sessions.

Example:
    1. Call eval_start_session() to begin tracking
    2. Execute your task using domain tools
    3. Call eval_end_session() to finalize
    4. Call eval_get_composite_score() for detailed metrics
"""

from rhoai_mcp.evaluation.metrics import (
    ParameterPrecisionMetrics,
    PerformanceMetrics,
    StabilityMetrics,
    SuccessErrorMetrics,
    ToolSelectionMetrics,
    TrajectoryMetrics,
)
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
from rhoai_mcp.evaluation.scoring import (
    CompositeEvaluationScore,
    calculate_composite_score,
    calculate_score_from_session,
)
from rhoai_mcp.evaluation.session_manager import EvaluationSessionManager
from rhoai_mcp.evaluation.validation import ResultValidator, create_default_validator

__all__ = [
    # Models
    "EvaluationReport",
    "EvaluationSession",
    "ExpectedResult",
    "ExpectedToolSequence",
    "ParameterSpec",
    "SessionStatus",
    "ToolCall",
    "TrajectorySpec",
    "ValidationResult",
    # Metrics
    "ParameterPrecisionMetrics",
    "PerformanceMetrics",
    "StabilityMetrics",
    "SuccessErrorMetrics",
    "ToolSelectionMetrics",
    "TrajectoryMetrics",
    # Scoring
    "CompositeEvaluationScore",
    "calculate_composite_score",
    "calculate_score_from_session",
    # Session management
    "EvaluationSessionManager",
    # Validation
    "ResultValidator",
    "create_default_validator",
]
