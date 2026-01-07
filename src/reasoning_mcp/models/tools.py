"""Tool I/O models for MCP tool interfaces.

This module defines all input and output models for the MCP tools exposed by
reasoning-mcp. These models provide the interface layer between external clients
and the internal reasoning engine, handling tool invocations and results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, SessionStatus, ThoughtType
from reasoning_mcp.models.pipeline import PipelineTrace
from reasoning_mcp.models.thought import ThoughtNode


# ============================================================================
# Input Models (Mutable - For Tool Invocations)
# ============================================================================


class ReasonHints(BaseModel):
    """Hints to guide the reasoning method selection.

    ReasonHints allows clients to provide guidance to the reasoning engine about
    which methods to prefer or avoid, based on problem domain, complexity, or
    other contextual factors. This enables adaptive method selection while still
    allowing user preferences.

    Examples:
        Create hints for a code problem:
        >>> hints = ReasonHints(
        ...     domain="code",
        ...     complexity="moderate",
        ...     prefer_methods=[MethodIdentifier.CODE_REASONING],
        ...     avoid_methods=[MethodIdentifier.ETHICAL_REASONING]
        ... )

        Create hints for complex ethical analysis:
        >>> ethical_hints = ReasonHints(
        ...     domain="ethical",
        ...     complexity="high",
        ...     prefer_methods=[
        ...         MethodIdentifier.ETHICAL_REASONING,
        ...         MethodIdentifier.DIALECTIC,
        ...         MethodIdentifier.SOCRATIC
        ...     ],
        ...     custom_hints={"stakeholders": ["users", "developers", "society"]}
        ... )

        Create hints for mathematical reasoning:
        >>> math_hints = ReasonHints(
        ...     domain="math",
        ...     complexity="high",
        ...     prefer_methods=[MethodIdentifier.MATHEMATICAL_REASONING],
        ...     custom_hints={"proof_required": True, "symbolic_manipulation": True}
        ... )
    """

    model_config = ConfigDict(frozen=False)

    domain: str | None = Field(
        default=None,
        description="Problem domain (e.g., 'code', 'math', 'ethical', 'creative')",
    )
    complexity: str | None = Field(
        default=None,
        description="Estimated complexity level (e.g., 'low', 'moderate', 'high', 'expert')",
    )
    prefer_methods: list[MethodIdentifier] = Field(
        default_factory=list,
        description="Reasoning methods to prefer for this problem",
    )
    avoid_methods: list[MethodIdentifier] = Field(
        default_factory=list,
        description="Reasoning methods to avoid for this problem",
    )
    custom_hints: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom hints and context for method selection",
    )


# ============================================================================
# Output Models (Frozen - Immutable Results)
# ============================================================================


class ReasonOutput(BaseModel):
    """Output from the `reason` tool.

    ReasonOutput provides the result of a single reasoning step, including the
    generated thought, the method used, and suggestions for next steps. This is
    the primary output interface for incremental reasoning.

    Examples:
        Create output from a reasoning step:
        >>> from uuid import uuid4
        >>> output = ReasonOutput(
        ...     session_id=str(uuid4()),
        ...     thought=ThoughtNode(
        ...         id=str(uuid4()),
        ...         type=ThoughtType.INITIAL,
        ...         method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...         content="Let's break this problem down step by step.",
        ...         confidence=0.85
        ...     ),
        ...     method_used=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     suggestions=[
        ...         "Continue with deeper analysis",
        ...         "Consider alternative approaches",
        ...         "Validate assumptions"
        ...     ],
        ...     metadata={"tokens_used": 150, "inference_time_ms": 420}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(
        description="UUID of the reasoning session",
    )
    thought: ThoughtNode = Field(
        description="The generated thought from this reasoning step",
    )
    method_used: MethodIdentifier = Field(
        description="Reasoning method that was used to generate this thought",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggested next steps or actions for continuing the reasoning",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the reasoning step (e.g., performance metrics)",
    )


class ThoughtOutput(BaseModel):
    """Simplified thought output for clients.

    ThoughtOutput provides a streamlined view of a thought, exposing only the
    essential information needed by client applications. This reduces payload
    size and simplifies client logic for displaying reasoning progress.

    Examples:
        Create simplified thought output:
        >>> thought_out = ThoughtOutput(
        ...     id="thought-123",
        ...     content="This is the key insight: the problem can be decomposed.",
        ...     thought_type=ThoughtType.SYNTHESIS,
        ...     confidence=0.92,
        ...     step_number=5
        ... )

        Create output without optional fields:
        >>> minimal_out = ThoughtOutput(
        ...     id="thought-456",
        ...     content="Initial observation: the data shows a clear pattern.",
        ...     thought_type=ThoughtType.OBSERVATION
        ... )
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(
        description="Unique identifier for this thought",
    )
    content: str = Field(
        description="The main content of the thought",
    )
    thought_type: ThoughtType = Field(
        description="Type of thought (e.g., initial, continuation, synthesis)",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for this thought (0.0-1.0)",
    )
    step_number: int | None = Field(
        default=None,
        ge=0,
        description="Sequential step number in the reasoning process",
    )


class SuggestionOutput(BaseModel):
    """Suggestions for next reasoning steps.

    SuggestionOutput provides actionable recommendations for how to continue
    the reasoning process, including suggested methods and contextual information.

    Examples:
        Create suggestions with methods:
        >>> suggestions = SuggestionOutput(
        ...     suggestions=[
        ...         "Analyze ethical implications",
        ...         "Consider stakeholder impacts",
        ...         "Evaluate alternative solutions"
        ...     ],
        ...     recommended_methods=[
        ...         MethodIdentifier.ETHICAL_REASONING,
        ...         MethodIdentifier.DIALECTIC
        ...     ],
        ...     context="High complexity ethical decision with multiple stakeholders"
        ... )

        Create simple suggestions:
        >>> simple_suggestions = SuggestionOutput(
        ...     suggestions=[
        ...         "Continue breaking down the problem",
        ...         "Verify intermediate results"
        ...     ]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    suggestions: list[str] = Field(
        description="List of suggested next steps or actions",
    )
    recommended_methods: list[MethodIdentifier] = Field(
        default_factory=list,
        description="Reasoning methods recommended for the next steps",
    )
    context: str | None = Field(
        default=None,
        description="Contextual information explaining the suggestions",
    )


class ValidationOutput(BaseModel):
    """Validation result for thoughts or sessions.

    ValidationOutput provides the result of validating thoughts, sessions, or
    other reasoning artifacts, including any errors or warnings found.

    Examples:
        Create successful validation:
        >>> valid = ValidationOutput(
        ...     valid=True,
        ...     errors=[],
        ...     warnings=["Confidence score is relatively low (0.45)"]
        ... )

        Create failed validation:
        >>> invalid = ValidationOutput(
        ...     valid=False,
        ...     errors=[
        ...         "Thought content is empty",
        ...         "Parent thought ID references non-existent thought"
        ...     ],
        ...     warnings=["Depth exceeds recommended maximum"]
        ... )

        Create validation with no issues:
        >>> perfect = ValidationOutput(valid=True)
    """

    model_config = ConfigDict(frozen=True)

    valid: bool = Field(
        description="Whether the validation passed (True) or failed (False)",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="List of validation errors that caused failure",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="List of non-critical warnings or suggestions",
    )


class ComposeOutput(BaseModel):
    """Output from the `compose` tool (pipeline execution).

    ComposeOutput provides the complete result of executing a reasoning pipeline,
    including final thoughts, execution trace, and any errors encountered.

    Examples:
        Create successful pipeline output:
        >>> from datetime import datetime
        >>> compose_out = ComposeOutput(
        ...     session_id="session-123",
        ...     pipeline_id="pipeline-abc",
        ...     success=True,
        ...     final_thoughts=[
        ...         ThoughtNode(
        ...             id="final-1",
        ...             type=ThoughtType.CONCLUSION,
        ...             method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...             content="Final conclusion based on analysis",
        ...             confidence=0.95
        ...         )
        ...     ],
        ...     trace=PipelineTrace(
        ...         pipeline_id="pipeline-abc",
        ...         session_id="session-123",
        ...         started_at=datetime.now(),
        ...         completed_at=datetime.now(),
        ...         status="completed"
        ...     )
        ... )

        Create failed pipeline output:
        >>> failed_out = ComposeOutput(
        ...     session_id="session-456",
        ...     pipeline_id="pipeline-def",
        ...     success=False,
        ...     final_thoughts=[],
        ...     error="Pipeline execution failed at stage 'ethical_analysis': timeout exceeded"
        ... )
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(
        description="UUID of the session this pipeline executed in",
    )
    pipeline_id: str = Field(
        description="UUID of the pipeline that was executed",
    )
    success: bool = Field(
        description="Whether the pipeline executed successfully to completion",
    )
    final_thoughts: list[ThoughtNode] = Field(
        default_factory=list,
        description="Final thoughts produced by the pipeline",
    )
    trace: PipelineTrace | None = Field(
        default=None,
        description="Complete execution trace for debugging and analysis",
    )
    error: str | None = Field(
        default=None,
        description="Error message if pipeline execution failed",
    )


class SessionState(BaseModel):
    """Current state of a reasoning session.

    SessionState provides a snapshot of a session's current status, including
    thought counts, active methods, and timing information. This enables clients
    to monitor session progress and state.

    Examples:
        Create state for active session:
        >>> state = SessionState(
        ...     session_id="session-123",
        ...     status=SessionStatus.ACTIVE,
        ...     thought_count=42,
        ...     branch_count=3,
        ...     current_method=MethodIdentifier.TREE_OF_THOUGHTS,
        ...     started_at=datetime.now(),
        ...     updated_at=datetime.now()
        ... )

        Create state for completed session:
        >>> completed = SessionState(
        ...     session_id="session-456",
        ...     status=SessionStatus.COMPLETED,
        ...     thought_count=67,
        ...     branch_count=5,
        ...     current_method=None,
        ...     started_at=datetime(2025, 1, 1, 10, 0, 0),
        ...     updated_at=datetime(2025, 1, 1, 10, 15, 30)
        ... )
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(
        description="UUID of the session",
    )
    status: SessionStatus = Field(
        description="Current session status (e.g., created, active, completed)",
    )
    thought_count: int = Field(
        ge=0,
        description="Total number of thoughts in the session",
    )
    branch_count: int = Field(
        ge=0,
        description="Number of active branches in the session",
    )
    current_method: MethodIdentifier | None = Field(
        default=None,
        description="Currently active reasoning method, if any",
    )
    started_at: datetime | None = Field(
        default=None,
        description="Timestamp when the session was started",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last update to the session",
    )


class BranchOutput(BaseModel):
    """Output from session_branch tool.

    BranchOutput provides the result of creating a new branch in a reasoning
    session, allowing exploration of alternative reasoning paths.

    Examples:
        Create successful branch output:
        >>> branch = BranchOutput(
        ...     branch_id="branch-abc",
        ...     parent_thought_id="thought-123",
        ...     session_id="session-456",
        ...     success=True
        ... )

        Create failed branch output:
        >>> failed_branch = BranchOutput(
        ...     branch_id="",
        ...     parent_thought_id="thought-999",
        ...     session_id="session-456",
        ...     success=False
        ... )
    """

    model_config = ConfigDict(frozen=True)

    branch_id: str = Field(
        description="UUID of the newly created branch",
    )
    parent_thought_id: str = Field(
        description="ID of the thought where the branch originates",
    )
    session_id: str = Field(
        description="UUID of the session containing this branch",
    )
    success: bool = Field(
        description="Whether the branch was successfully created",
    )


class MergeOutput(BaseModel):
    """Output from session_merge tool.

    MergeOutput provides the result of merging multiple branches back together,
    combining insights from different reasoning paths into a unified thought.

    Examples:
        Create successful merge output:
        >>> merge = MergeOutput(
        ...     merged_thought_id="thought-merged-123",
        ...     source_branch_ids=["branch-abc", "branch-def", "branch-ghi"],
        ...     session_id="session-456",
        ...     success=True
        ... )

        Create failed merge output:
        >>> failed_merge = MergeOutput(
        ...     merged_thought_id="",
        ...     source_branch_ids=["branch-abc", "branch-def"],
        ...     session_id="session-456",
        ...     success=False
        ... )
    """

    model_config = ConfigDict(frozen=True)

    merged_thought_id: str = Field(
        description="ID of the thought created by merging the branches",
    )
    source_branch_ids: list[str] = Field(
        description="IDs of the branches that were merged",
    )
    session_id: str = Field(
        description="UUID of the session containing the merged branches",
    )
    success: bool = Field(
        description="Whether the merge was successful",
    )


class MethodInfo(BaseModel):
    """Information about a reasoning method.

    MethodInfo provides comprehensive metadata about a reasoning method, including
    its capabilities, configuration parameters, and use cases. This helps clients
    understand and select appropriate methods.

    Examples:
        Create method info:
        >>> info = MethodInfo(
        ...     id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     name="Chain of Thought",
        ...     description="Classic step-by-step reasoning with intermediate steps shown",
        ...     category=MethodCategory.CORE,
        ...     parameters={
        ...         "max_steps": {"type": "int", "default": 10, "min": 1, "max": 50},
        ...         "show_reasoning": {"type": "bool", "default": True}
        ...     },
        ...     tags=["sequential", "transparent", "general-purpose"]
        ... )

        Create specialized method info:
        >>> ethical_info = MethodInfo(
        ...     id=MethodIdentifier.ETHICAL_REASONING,
        ...     name="Ethical Reasoning",
        ...     description="Structured ethical analysis with principles and stakeholder consideration",
        ...     category=MethodCategory.HIGH_VALUE,
        ...     parameters={
        ...         "frameworks": {"type": "list", "default": ["utilitarian", "deontological"]},
        ...         "stakeholder_analysis": {"type": "bool", "default": True}
        ...     },
        ...     tags=["ethical", "stakeholders", "principles", "structured"]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    id: MethodIdentifier = Field(
        description="Unique identifier for this reasoning method",
    )
    name: str = Field(
        description="Human-readable name of the method",
    )
    description: str = Field(
        description="Detailed description of what this method does and when to use it",
    )
    category: MethodCategory = Field(
        description="Category this method belongs to (e.g., core, high_value, specialized)",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific configuration parameters and their schemas",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Searchable tags describing method characteristics and use cases",
    )


class Recommendation(BaseModel):
    """Method recommendation result.

    Recommendation provides a scored suggestion for which reasoning method to use,
    along with the rationale and confidence in the recommendation.

    Examples:
        Create high-confidence recommendation:
        >>> rec = Recommendation(
        ...     method_id=MethodIdentifier.ETHICAL_REASONING,
        ...     score=0.95,
        ...     reason="Problem involves ethical dilemmas and stakeholder analysis",
        ...     confidence=0.92
        ... )

        Create moderate recommendation:
        >>> moderate_rec = Recommendation(
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     score=0.65,
        ...     reason="General-purpose method suitable for most problems",
        ...     confidence=0.70
        ... )
    """

    model_config = ConfigDict(frozen=True)

    method_id: MethodIdentifier = Field(
        description="Recommended reasoning method identifier",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Recommendation score from 0.0 (not recommended) to 1.0 (highly recommended)",
    )
    reason: str = Field(
        description="Explanation of why this method is recommended",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this recommendation (0.0-1.0)",
    )


class ComparisonResult(BaseModel):
    """Result of comparing reasoning methods.

    ComparisonResult provides a comparative analysis of multiple reasoning methods,
    including their relative scores, the top-performing method, and detailed analysis.

    Examples:
        Create comparison result:
        >>> comparison = ComparisonResult(
        ...     methods=[
        ...         MethodIdentifier.CHAIN_OF_THOUGHT,
        ...         MethodIdentifier.TREE_OF_THOUGHTS,
        ...         MethodIdentifier.SELF_CONSISTENCY
        ...     ],
        ...     winner=MethodIdentifier.TREE_OF_THOUGHTS,
        ...     scores={
        ...         "chain_of_thought": 0.75,
        ...         "tree_of_thoughts": 0.92,
        ...         "self_consistency": 0.88
        ...     },
        ...     analysis="Tree of Thoughts is most suitable for this exploratory problem "
        ...              "requiring multiple reasoning paths. It scored highest on "
        ...              "exploration depth (0.95) and path diversity (0.90)."
        ... )

        Create tie comparison:
        >>> tie = ComparisonResult(
        ...     methods=[
        ...         MethodIdentifier.ETHICAL_REASONING,
        ...         MethodIdentifier.DIALECTIC
        ...     ],
        ...     winner=None,
        ...     scores={
        ...         "ethical_reasoning": 0.85,
        ...         "dialectic": 0.85
        ...     },
        ...     analysis="Both methods scored equally for this problem. "
        ...              "Consider using them sequentially or in parallel."
        ... )
    """

    model_config = ConfigDict(frozen=True)

    methods: list[MethodIdentifier] = Field(
        description="List of methods that were compared",
    )
    winner: MethodIdentifier | None = Field(
        default=None,
        description="Method with the highest score, or None if there's a tie",
    )
    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Map of method IDs to their comparative scores",
    )
    analysis: str = Field(
        description="Detailed analysis explaining the comparison results",
    )


class EvaluationReport(BaseModel):
    """Evaluation report for a reasoning session.

    EvaluationReport provides comprehensive quality metrics and analysis for a
    completed reasoning session, including coherence, depth, coverage scores,
    and actionable recommendations.

    Examples:
        Create high-quality evaluation:
        >>> report = EvaluationReport(
        ...     session_id="session-123",
        ...     overall_score=0.89,
        ...     coherence_score=0.92,
        ...     depth_score=0.85,
        ...     coverage_score=0.90,
        ...     insights=[
        ...         "Strong logical flow throughout the reasoning process",
        ...         "Excellent exploration of alternative perspectives",
        ...         "Comprehensive stakeholder analysis"
        ...     ],
        ...     recommendations=[
        ...         "Consider adding more quantitative analysis",
        ...         "Could benefit from additional validation steps"
        ...     ]
        ... )

        Create low-quality evaluation:
        >>> weak_report = EvaluationReport(
        ...     session_id="session-456",
        ...     overall_score=0.45,
        ...     coherence_score=0.40,
        ...     depth_score=0.50,
        ...     coverage_score=0.45,
        ...     insights=[
        ...         "Reasoning path lacks coherent structure",
        ...         "Insufficient depth in critical areas"
        ...     ],
        ...     recommendations=[
        ...         "Use more structured reasoning methods",
        ...         "Increase depth of analysis for key decision points",
        ...         "Consider using self-reflection to identify gaps"
        ...     ]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(
        description="UUID of the session being evaluated",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score for the reasoning session (0.0-1.0)",
    )
    coherence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for logical coherence and consistency (0.0-1.0)",
    )
    depth_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for depth and thoroughness of reasoning (0.0-1.0)",
    )
    coverage_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for breadth and completeness of coverage (0.0-1.0)",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights and observations about the reasoning quality",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improving future reasoning sessions",
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Input models
    "ReasonHints",
    # Output models
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
