"""Trace collection for reasoning execution.

This module provides the TraceCollector class for collecting and managing trace
data during reasoning execution, including spans, steps, decisions, and errors.
"""

import traceback
from collections import deque
from datetime import datetime
from typing import Any
from uuid import uuid4

# Default maximum sizes for trace collections to prevent memory leaks
DEFAULT_MAX_SPANS = 1000
DEFAULT_MAX_STEPS = 5000
DEFAULT_MAX_DECISIONS = 1000
DEFAULT_MAX_ERRORS = 500

from reasoning_mcp.models.debug import (
    SpanStatus,
    Trace,
    TraceDecision,
    TraceError,
    TraceSpan,
    TraceStep,
    TraceStepType,
)


class TraceCollector:
    """Collects trace data during reasoning execution.

    The TraceCollector manages hierarchical trace spans, steps, decisions, and errors
    during the execution of reasoning methods. It provides a fluent interface for
    creating and managing trace data, automatically tracking parent-child relationships
    between spans.

    Attributes:
        trace_id: Unique identifier for this trace collection
        session_id: Session identifier associated with this trace
    """

    def __init__(
        self,
        session_id: str | None = None,
        max_spans: int = DEFAULT_MAX_SPANS,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_decisions: int = DEFAULT_MAX_DECISIONS,
        max_errors: int = DEFAULT_MAX_ERRORS,
    ) -> None:
        """Initialize a new trace collector.

        Args:
            session_id: Optional session ID to associate with this trace.
                If not provided, a new UUID will be generated.
            max_spans: Maximum number of spans to retain (oldest evicted first)
            max_steps: Maximum number of steps to retain (oldest evicted first)
            max_decisions: Maximum number of decisions to retain (oldest evicted first)
            max_errors: Maximum number of errors to retain (oldest evicted first)
        """
        self.trace_id: str = str(uuid4())
        self.session_id: str = session_id or str(uuid4())
        self._max_spans = max_spans
        self._spans: dict[str, TraceSpan] = {}
        self._span_order: deque[str] = deque(maxlen=max_spans)  # Track span insertion order
        self._steps: deque[TraceStep] = deque(maxlen=max_steps)
        self._decisions: deque[TraceDecision] = deque(maxlen=max_decisions)
        self._errors: deque[TraceError] = deque(maxlen=max_errors)
        self._current_span_id: str | None = None
        self._root_span_id: str | None = None

    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Start a new span and return its ID.

        Creates a new trace span with the given name and optional parent. If no parent
        is specified and a span is currently active, the new span becomes a child of
        the current span. The first span created becomes the root span.

        Args:
            name: Human-readable name describing the operation being traced
            parent_id: Optional ID of the parent span. If not provided, uses the
                current active span as parent.
            attributes: Optional dictionary of metadata about the span operation

        Returns:
            The unique ID of the newly created span
        """
        span_id = str(uuid4())
        span = TraceSpan(
            span_id=span_id,
            parent_id=parent_id or self._current_span_id,
            name=name,
            start_time=datetime.now(),
            end_time=None,
            status=SpanStatus.RUNNING,
            attributes=attributes or {},
        )
        # Evict oldest span if we're at max capacity
        if len(self._span_order) >= self._max_spans and self._span_order:
            oldest_span_id = self._span_order[0]  # Will be auto-removed by deque when we append
            if oldest_span_id in self._spans:
                del self._spans[oldest_span_id]
        self._spans[span_id] = span
        self._span_order.append(span_id)
        if self._root_span_id is None:
            self._root_span_id = span_id
        self._current_span_id = span_id
        return span_id

    def end_span(self, span_id: str, status: SpanStatus = SpanStatus.COMPLETED) -> None:
        """End a span with the given status.

        Marks a span as completed by setting its end time and status. If the span
        being ended is the current active span, the current span is updated to its
        parent.

        Args:
            span_id: ID of the span to end
            status: Final status of the span. Defaults to COMPLETED.

        Raises:
            ValueError: If the span_id does not exist in the trace
        """
        if span_id not in self._spans:
            raise ValueError(f"Unknown span ID: {span_id}")
        span = self._spans[span_id]
        # Update span with end_time and status
        self._spans[span_id] = TraceSpan(
            span_id=span.span_id,
            parent_id=span.parent_id,
            name=span.name,
            start_time=span.start_time,
            end_time=datetime.now(),
            status=status,
            attributes=span.attributes,
        )
        # Update current span to parent
        if self._current_span_id == span_id:
            self._current_span_id = span.parent_id

    def add_step(
        self,
        step_type: TraceStepType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a step to the current span.

        Creates a new trace step associated with the currently active span. Steps
        represent discrete operations, events, or state changes during reasoning.

        Args:
            step_type: Type/category of the step (e.g., THOUGHT, DECISION, INPUT)
            content: The actual content or description of the step
            metadata: Optional dictionary of additional contextual information

        Returns:
            The unique ID of the newly created step

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        if self._current_span_id is None:
            raise ValueError("No active span - call start_span first")
        step_id = str(uuid4())
        step = TraceStep(
            step_id=step_id,
            span_id=self._current_span_id,
            step_type=step_type,
            timestamp=datetime.now(),
            content=content,
            metadata=metadata or {},
        )
        self._steps.append(step)
        return step_id

    def add_decision(
        self,
        question: str,
        options: list[str],
        chosen: str,
        reasoning: str,
        confidence: float,
    ) -> str:
        """Add a decision point to the current span.

        Records a decision made during reasoning execution, including the question
        posed, available options, chosen option, and reasoning behind the choice.

        Args:
            question: The question or decision point being addressed
            options: List of available options to choose from
            chosen: The option that was chosen
            reasoning: Explanation of why this option was chosen
            confidence: Confidence level in the decision (0.0 to 1.0)

        Returns:
            The unique ID of the newly created decision

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        if self._current_span_id is None:
            raise ValueError("No active span - call start_span first")
        decision_id = str(uuid4())
        decision = TraceDecision(
            decision_id=decision_id,
            span_id=self._current_span_id,
            question=question,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
            confidence=confidence,
        )
        self._decisions.append(decision)
        return decision_id

    def add_error(
        self,
        error_type: str,
        message: str,
        stack_trace: str,
        recoverable: bool,
        recovery_action: str | None = None,
    ) -> str:
        """Add an error to the current span.

        Records an error that occurred during reasoning execution, including
        error type, message, stack trace, and recovery information.

        Args:
            error_type: Classification or type of the error (e.g., "ValidationError")
            message: Human-readable error message describing what went wrong
            stack_trace: Full stack trace of the error for debugging
            recoverable: Whether the error can be recovered from
            recovery_action: Optional description of the recovery action taken

        Returns:
            The unique ID of the newly created error

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        if self._current_span_id is None:
            raise ValueError("No active span - call start_span first")
        error_id = str(uuid4())
        error = TraceError(
            error_id=error_id,
            span_id=self._current_span_id,
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            recoverable=recoverable,
            recovery_action=recovery_action,
        )
        self._errors.append(error)
        return error_id

    def record_decision(
        self,
        question: str,
        options: list[str],
        chosen: str,
        reasoning: str,
        confidence: float,
    ) -> str:
        """Record a decision point in the current span.

        Alias for add_decision that matches the plan spec.

        Args:
            question: The question or decision point being addressed
            options: List of available options to choose from
            chosen: The option that was chosen
            reasoning: Explanation of why this option was chosen
            confidence: Confidence level in the decision (0.0 to 1.0)

        Returns:
            The unique ID of the newly created decision

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        return self.add_decision(question, options, chosen, reasoning, confidence)

    def record_error(
        self,
        error: Exception,
        recoverable: bool = False,
        recovery_action: str | None = None,
    ) -> str:
        """Record an error from an exception in the current span.

        Args:
            error: The exception that was raised
            recoverable: Whether the error can be recovered from
            recovery_action: Optional description of the recovery action taken

        Returns:
            The unique ID of the newly created error

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        return self.add_error(
            error_type=type(error).__name__,
            message=str(error),
            stack_trace=traceback.format_exc(),
            recoverable=recoverable,
            recovery_action=recovery_action,
        )

    def checkpoint(self, label: str, data: dict[str, Any]) -> str:
        """Record a checkpoint for debugging purposes.

        Args:
            label: Human-readable label for the checkpoint
            data: Dictionary of checkpoint data to record

        Returns:
            The unique ID of the newly created checkpoint step

        Raises:
            ValueError: If no span is currently active (call start_span first)
        """
        return self.add_step(
            step_type=TraceStepType.CHECKPOINT,
            content=label,
            metadata=data,
        )

    def get_span_tree(self) -> dict[str, Any]:
        """Return hierarchical span structure for visualization.

        Builds a tree representation of all spans showing their parent-child
        relationships, status, duration, and attributes.

        Returns:
            A dictionary representing the hierarchical span tree starting from
            the root span. Returns empty dict if no spans exist.
        """
        if self._root_span_id is None:
            return {}

        span_map = {span_id: span for span_id, span in self._spans.items()}
        children: dict[str | None, list[str]] = {}

        for span_id, span in self._spans.items():
            parent = span.parent_id
            if parent not in children:
                children[parent] = []
            children[parent].append(span_id)

        def build_tree(span_id: str) -> dict[str, Any]:
            span = span_map[span_id]
            duration = None
            if span.end_time and span.start_time:
                duration = (span.end_time - span.start_time).total_seconds()

            return {
                "span_id": span_id,
                "name": span.name,
                "status": span.status.value,
                "duration_seconds": duration,
                "attributes": span.attributes,
                "children": [build_tree(child_id) for child_id in children.get(span_id, [])],
            }

        return build_tree(self._root_span_id)

    def get_trace(self) -> Trace:
        """Get the complete trace with all collected data.

        Assembles and returns a Trace object containing the root span and all
        collected spans, steps, decisions, and errors.

        Returns:
            A Trace object with all collected trace data

        Raises:
            ValueError: If no root span has been created (call start_span first)
        """
        if self._root_span_id is None:
            raise ValueError("No root span - call start_span first")
        root_span = self._spans[self._root_span_id]
        return Trace(
            trace_id=self.trace_id,
            session_id=self.session_id,
            root_span=root_span,
            spans=list(self._spans.values()),
            steps=list(self._steps),
            decisions=list(self._decisions),
            errors=list(self._errors),
        )
