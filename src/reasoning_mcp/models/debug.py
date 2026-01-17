"""Debug and tracing models for reasoning-mcp.

This module defines enumerations and models used for debugging, tracing, and
monitoring reasoning processes, including span status tracking and execution
lifecycle states.
"""

from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TraceLevel(StrEnum):
    """Trace verbosity levels for debugging and monitoring reasoning execution.

    Controls the amount of detail captured during reasoning execution, from
    minimal summary information to verbose step-by-step details.
    """

    MINIMAL = "minimal"
    """Minimal tracing - only capture final results and critical errors."""

    STANDARD = "standard"
    """Standard tracing - capture key decisions and intermediate results."""

    DETAILED = "detailed"
    """Detailed tracing - capture all major steps and method transitions."""

    VERBOSE = "verbose"
    """Verbose tracing - capture complete execution details including internals."""


class SpanStatus(Enum):
    """Status of a reasoning span or execution unit.

    Represents the lifecycle state of a span during execution tracing.
    Spans are execution units that track the start, progress, and completion
    of reasoning operations.
    """

    RUNNING = "running"
    """The span is currently executing. Initial state after span creation."""

    COMPLETED = "completed"
    """The span has finished successfully. Final state for successful execution."""

    FAILED = "failed"
    """The span encountered an error and failed. Final state for failed execution."""

    CANCELLED = "cancelled"
    """The span was cancelled before completion. Final state for cancelled execution."""


class TraceStepType(Enum):
    """Type of trace step in a reasoning execution flow.

    Categorizes different types of events and operations that can be traced
    during reasoning execution. Used to provide semantic meaning to trace
    events for analysis, debugging, and visualization.
    """

    METHOD_START = "method_start"
    """Marks the beginning of a reasoning method execution."""

    METHOD_END = "method_end"
    """Marks the completion of a reasoning method execution."""

    THOUGHT = "thought"
    """Represents a reasoning thought or intermediate cognitive step."""

    DECISION = "decision"
    """Captures a decision point or branch in the reasoning process."""

    INPUT = "input"
    """Records input data being processed by a reasoning step."""

    OUTPUT = "output"
    """Records output data produced by a reasoning step."""

    ERROR = "error"
    """Indicates an error occurred during reasoning execution."""

    CHECKPOINT = "checkpoint"
    """Marks a checkpoint or savepoint in the reasoning process."""


class TraceSpan(BaseModel):
    """A trace span representing a unit of work in the reasoning process.

    Trace spans are hierarchical execution units that track timing, status, and
    metadata for reasoning operations. Spans can have parent-child relationships
    to represent nested operations.

    Attributes:
        span_id: Unique identifier for this span
        parent_id: ID of the parent span, or None if this is a root span
        name: Human-readable name describing the operation being traced
        start_time: Timestamp when the span started execution
        end_time: Timestamp when the span completed, or None if still running
        status: Current execution status of the span
        attributes: Additional metadata and context about the span operation
    """

    span_id: str = Field(..., description="Unique identifier for this span")
    parent_id: str | None = Field(
        None, description="ID of the parent span, or None if this is a root span"
    )
    name: str = Field(..., description="Human-readable name for the span")
    start_time: datetime = Field(..., description="Timestamp when span started")
    end_time: datetime | None = Field(
        None, description="Timestamp when span ended, or None if still running"
    )
    status: SpanStatus = Field(..., description="Current execution status")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the span"
    )


class TraceStep(BaseModel):
    """A trace step representing a discrete action or event within a trace span.

    Trace steps capture individual operations, events, or state changes that occur
    during the execution of a reasoning process. Each step is associated with a
    parent span and records detailed information about what happened at a specific
    point in time.

    Attributes:
        step_id: Unique identifier for this step
        span_id: ID of the parent span containing this step
        step_type: Type/category of the step (e.g., thought, decision, input, output)
        timestamp: When this step occurred
        content: The actual content or description of the step
        metadata: Additional contextual information about the step
    """

    step_id: str = Field(..., description="Unique identifier for this step")
    span_id: str = Field(..., description="ID of the parent span containing this step")
    step_type: TraceStepType = Field(..., description="Type or category of the step")
    timestamp: datetime = Field(..., description="Timestamp when this step occurred")
    content: str = Field(..., description="The content or description of the step")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional contextual information"
    )


class TraceError(BaseModel):
    """An error that occurred during reasoning execution tracing.

    Captures detailed information about errors encountered during reasoning
    operations, including error type, location (span), and recovery information.
    This model helps track and diagnose failures in reasoning processes.

    Attributes:
        error_id: Unique identifier for this error instance
        span_id: ID of the span where the error occurred
        error_type: Classification or type of the error (e.g., "ValidationError", "TimeoutError")
        message: Human-readable error message describing what went wrong
        stack_trace: Full stack trace of the error for debugging
        recoverable: Whether the error can be recovered from or requires termination
        recovery_action: Description of the action taken to recover, if applicable
    """

    error_id: str = Field(..., description="Unique identifier for this error instance")
    span_id: str = Field(..., description="ID of the span where the error occurred")
    error_type: str = Field(..., description="Classification or type of the error")
    message: str = Field(..., description="Human-readable error message describing what went wrong")
    stack_trace: str = Field(..., description="Full stack trace of the error for debugging")
    recoverable: bool = Field(
        ..., description="Whether the error can be recovered from or requires termination"
    )
    recovery_action: str | None = Field(
        None, description="Description of the action taken to recover, if applicable"
    )


class TraceDecision(BaseModel):
    """A decision point in the reasoning process with associated metadata.

    TraceDecision captures decision-making events during reasoning execution,
    including the question posed, available options, the chosen option, and
    the reasoning behind the choice. This is useful for debugging, analysis,
    and understanding the reasoning path taken.

    Attributes:
        decision_id: Unique identifier for this decision
        span_id: ID of the span this decision belongs to
        question: The question or decision point being addressed
        options: List of available options to choose from
        chosen: The option that was chosen
        reasoning: Explanation of why this option was chosen
        confidence: Confidence level in the decision (0.0 to 1.0)
    """

    decision_id: str = Field(..., description="Unique identifier for this decision")
    span_id: str = Field(..., description="ID of the span this decision belongs to")
    question: str = Field(..., description="The question or decision point being addressed")
    options: list[str] = Field(..., description="List of available options to choose from")
    chosen: str = Field(..., description="The option that was chosen")
    reasoning: str = Field(..., description="Explanation of why this option was chosen")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level in the decision (0.0 to 1.0)",
    )


class Trace(BaseModel):
    """Top-level trace container for complete reasoning execution traces.

    The Trace model is the main container that aggregates all trace data from a
    reasoning execution session. It contains the root span, all child spans, steps,
    decisions, and errors that occurred during execution. This provides a complete
    hierarchical view of the reasoning process for analysis, debugging, and
    visualization.

    The trace structure is hierarchical:
    - One root_span that represents the entire reasoning execution
    - Multiple child spans that represent sub-operations
    - Steps distributed across spans tracking individual operations
    - Decisions recording choice points during reasoning
    - Errors capturing any failures that occurred

    Attributes:
        trace_id: Unique identifier for this trace
        session_id: ID of the session that generated this trace
        root_span: The root span representing the entire execution
        spans: List of all spans (including root) in the trace
        steps: List of all steps across all spans
        decisions: List of all decisions made during execution
        errors: List of all errors that occurred during execution
    """

    trace_id: str = Field(..., description="Unique identifier for this trace")
    session_id: str = Field(..., description="ID of the session that generated this trace")
    root_span: TraceSpan = Field(..., description="The root span representing the entire execution")
    spans: list[TraceSpan] = Field(
        default_factory=list, description="List of all spans in the trace"
    )
    steps: list[TraceStep] = Field(
        default_factory=list, description="List of all steps across all spans"
    )
    decisions: list[TraceDecision] = Field(
        default_factory=list, description="List of all decisions made during execution"
    )
    errors: list[TraceError] = Field(
        default_factory=list, description="List of all errors that occurred"
    )
