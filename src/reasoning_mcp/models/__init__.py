"""
Data models and type system for the reasoning-mcp server.

This package contains all Pydantic models and enumerations used throughout
the reasoning-mcp system, including:

- Core domain enumerations (methods, categories, thought types, session states)
- Thought nodes and reasoning graphs
- Session configuration and state management
- Pipeline DSL and execution models
- Tool input/output schemas

All models use Pydantic v2 for validation, serialization, and type safety.
"""

from __future__ import annotations

from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    PipelineStageType,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.pipeline import (
    Accumulator,
    Condition,
    ConditionalPipeline,
    ErrorHandler,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    Pipeline,
    PipelineResult,
    PipelineTrace,
    SequencePipeline,
    StageMetrics,
    StageTrace,
    SwitchPipeline,
    Transform,
)
from reasoning_mcp.models.session import Session, SessionConfig, SessionMetrics
from reasoning_mcp.models.thought import ThoughtEdge, ThoughtGraph, ThoughtNode
from reasoning_mcp.models.tools import (
    BranchOutput,
    ComparisonResult,
    ComposeOutput,
    EvaluationReport,
    MergeOutput,
    MethodInfo,
    Recommendation,
    ReasonHints,
    ReasonOutput,
    SessionState,
    SuggestionOutput,
    ThoughtOutput,
    ValidationOutput,
)

__all__ = [
    # Core enumerations
    "MethodIdentifier",
    "MethodCategory",
    "ThoughtType",
    "SessionStatus",
    "PipelineStageType",
    # Thought models
    "ThoughtNode",
    "ThoughtEdge",
    "ThoughtGraph",
    # Session models
    "Session",
    "SessionConfig",
    "SessionMetrics",
    # Pipeline models - Helper types
    "Transform",
    "Condition",
    "MergeStrategy",
    "Accumulator",
    "ErrorHandler",
    # Pipeline models - Stage types
    "MethodStage",
    "SequencePipeline",
    "ParallelPipeline",
    "ConditionalPipeline",
    "LoopPipeline",
    "SwitchPipeline",
    "Pipeline",
    # Pipeline models - Result/Trace types
    "StageMetrics",
    "StageTrace",
    "PipelineTrace",
    "PipelineResult",
    # Tool I/O models
    "ReasonHints",
    "ReasonOutput",
    "ThoughtOutput",
    "SuggestionOutput",
    "ValidationOutput",
    "ComposeOutput",
    "SessionState",
    "BranchOutput",
    "MergeOutput",
    "MethodInfo",
    "Recommendation",
    "ComparisonResult",
    "EvaluationReport",
]
