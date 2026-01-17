"""Tests for the TraceCollector class."""

import pytest

from reasoning_mcp.debug.collector import TraceCollector
from reasoning_mcp.models.debug import SpanStatus, TraceStepType


class TestTraceCollectorInit:
    """Test TraceCollector initialization."""

    def test_creates_trace_id(self) -> None:
        """Test that a trace ID is created."""
        collector = TraceCollector()
        assert collector.trace_id is not None
        assert isinstance(collector.trace_id, str)

    def test_creates_session_id(self) -> None:
        """Test that a session ID is created."""
        collector = TraceCollector()
        assert collector.session_id is not None

    def test_uses_provided_session_id(self) -> None:
        """Test that provided session ID is used."""
        collector = TraceCollector(session_id="test-session")
        assert collector.session_id == "test-session"

    def test_initializes_empty_collections(self) -> None:
        """Test that collections are initialized empty."""
        collector = TraceCollector()
        assert collector._spans == {}
        assert len(collector._steps) == 0  # Uses deque with maxlen
        assert len(collector._decisions) == 0  # Uses deque with maxlen
        assert len(collector._errors) == 0  # Uses deque with maxlen
        assert collector._current_span_id is None


class TestStartSpan:
    """Test the start_span method."""

    def test_returns_span_id(self) -> None:
        """Test that start_span returns a span ID."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        assert span_id is not None
        assert isinstance(span_id, str)

    def test_sets_current_span(self) -> None:
        """Test that start_span sets the current span."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        assert collector._current_span_id == span_id

    def test_sets_root_span(self) -> None:
        """Test that the first span becomes the root span."""
        collector = TraceCollector()
        span_id = collector.start_span("root")
        assert collector._root_span_id == span_id

    def test_creates_span_with_running_status(self) -> None:
        """Test that new spans have RUNNING status."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        assert collector._spans[span_id].status == SpanStatus.RUNNING

    def test_child_span_has_parent(self) -> None:
        """Test that child spans reference their parent."""
        collector = TraceCollector()
        parent_id = collector.start_span("parent")
        child_id = collector.start_span("child")
        assert collector._spans[child_id].parent_id == parent_id

    def test_accepts_attributes(self) -> None:
        """Test that attributes are stored in the span."""
        collector = TraceCollector()
        span_id = collector.start_span("test", attributes={"key": "value"})
        assert collector._spans[span_id].attributes == {"key": "value"}


class TestEndSpan:
    """Test the end_span method."""

    def test_sets_end_time(self) -> None:
        """Test that end_span sets the end time."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        assert collector._spans[span_id].end_time is not None

    def test_sets_completed_status_by_default(self) -> None:
        """Test that end_span sets COMPLETED status by default."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.end_span(span_id)
        assert collector._spans[span_id].status == SpanStatus.COMPLETED

    def test_sets_custom_status(self) -> None:
        """Test that end_span can set a custom status."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.end_span(span_id, SpanStatus.FAILED)
        assert collector._spans[span_id].status == SpanStatus.FAILED

    def test_updates_current_span_to_parent(self) -> None:
        """Test that ending a span updates current to its parent."""
        collector = TraceCollector()
        parent_id = collector.start_span("parent")
        child_id = collector.start_span("child")
        collector.end_span(child_id)
        assert collector._current_span_id == parent_id

    def test_raises_for_unknown_span(self) -> None:
        """Test that ending an unknown span raises ValueError."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.end_span("unknown-id")


class TestAddStep:
    """Test the add_step method."""

    def test_returns_step_id(self) -> None:
        """Test that add_step returns a step ID."""
        collector = TraceCollector()
        collector.start_span("test")
        step_id = collector.add_step(TraceStepType.THOUGHT, "thinking")
        assert step_id is not None
        assert isinstance(step_id, str)

    def test_adds_step_to_collection(self) -> None:
        """Test that add_step adds the step to the collection."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_step(TraceStepType.THOUGHT, "thinking")
        assert len(collector._steps) == 1

    def test_associates_step_with_current_span(self) -> None:
        """Test that steps are associated with the current span."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.add_step(TraceStepType.THOUGHT, "thinking")
        assert collector._steps[0].span_id == span_id

    def test_sets_step_type(self) -> None:
        """Test that add_step sets the step type."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_step(TraceStepType.DECISION, "decided")
        assert collector._steps[0].step_type == TraceStepType.DECISION

    def test_sets_content(self) -> None:
        """Test that add_step sets the content."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_step(TraceStepType.THOUGHT, "my thought")
        assert collector._steps[0].content == "my thought"

    def test_accepts_metadata(self) -> None:
        """Test that add_step accepts metadata."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_step(TraceStepType.THOUGHT, "thinking", metadata={"key": "value"})
        assert collector._steps[0].metadata == {"key": "value"}

    def test_raises_without_active_span(self) -> None:
        """Test that add_step raises ValueError without an active span."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.add_step(TraceStepType.THOUGHT, "thinking")


class TestAddDecision:
    """Test the add_decision method."""

    def test_returns_decision_id(self) -> None:
        """Test that add_decision returns a decision ID."""
        collector = TraceCollector()
        collector.start_span("test")
        decision_id = collector.add_decision(
            question="Choose?",
            options=["A", "B"],
            chosen="A",
            reasoning="Because A",
            confidence=0.9,
        )
        assert decision_id is not None
        assert isinstance(decision_id, str)

    def test_adds_decision_to_collection(self) -> None:
        """Test that add_decision adds to the collection."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_decision(
            question="Choose?",
            options=["A", "B"],
            chosen="A",
            reasoning="Because A",
            confidence=0.9,
        )
        assert len(collector._decisions) == 1

    def test_associates_decision_with_current_span(self) -> None:
        """Test that decisions are associated with the current span."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.add_decision(
            question="Choose?",
            options=["A", "B"],
            chosen="A",
            reasoning="Because A",
            confidence=0.9,
        )
        assert collector._decisions[0].span_id == span_id

    def test_sets_decision_attributes(self) -> None:
        """Test that all decision attributes are set correctly."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_decision(
            question="What to do?",
            options=["Option1", "Option2"],
            chosen="Option1",
            reasoning="It's better",
            confidence=0.85,
        )
        decision = collector._decisions[0]
        assert decision.question == "What to do?"
        assert decision.options == ["Option1", "Option2"]
        assert decision.chosen == "Option1"
        assert decision.reasoning == "It's better"
        assert decision.confidence == 0.85

    def test_raises_without_active_span(self) -> None:
        """Test that add_decision raises ValueError without an active span."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.add_decision(
                question="Choose?",
                options=["A", "B"],
                chosen="A",
                reasoning="Because A",
                confidence=0.9,
            )


class TestAddError:
    """Test the add_error method."""

    def test_returns_error_id(self) -> None:
        """Test that add_error returns an error ID."""
        collector = TraceCollector()
        collector.start_span("test")
        error_id = collector.add_error(
            error_type="ValueError",
            message="Something went wrong",
            stack_trace="line 1\nline 2",
            recoverable=True,
        )
        assert error_id is not None
        assert isinstance(error_id, str)

    def test_adds_error_to_collection(self) -> None:
        """Test that add_error adds to the collection."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_error(
            error_type="ValueError",
            message="Something went wrong",
            stack_trace="line 1\nline 2",
            recoverable=True,
        )
        assert len(collector._errors) == 1

    def test_associates_error_with_current_span(self) -> None:
        """Test that errors are associated with the current span."""
        collector = TraceCollector()
        span_id = collector.start_span("test")
        collector.add_error(
            error_type="ValueError",
            message="Something went wrong",
            stack_trace="line 1\nline 2",
            recoverable=True,
        )
        assert collector._errors[0].span_id == span_id

    def test_sets_error_attributes(self) -> None:
        """Test that all error attributes are set correctly."""
        collector = TraceCollector()
        collector.start_span("test")
        collector.add_error(
            error_type="TypeError",
            message="Type mismatch",
            stack_trace="traceback...",
            recoverable=False,
            recovery_action="Restart",
        )
        error = collector._errors[0]
        assert error.error_type == "TypeError"
        assert error.message == "Type mismatch"
        assert error.stack_trace == "traceback..."
        assert error.recoverable is False
        assert error.recovery_action == "Restart"

    def test_raises_without_active_span(self) -> None:
        """Test that add_error raises ValueError without an active span."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.add_error(
                error_type="ValueError",
                message="Something went wrong",
                stack_trace="line 1\nline 2",
                recoverable=True,
            )


class TestGetTrace:
    """Test the get_trace method."""

    def test_returns_trace_object(self) -> None:
        """Test that get_trace returns a Trace object."""
        collector = TraceCollector()
        collector.start_span("root")
        trace = collector.get_trace()
        assert trace is not None
        assert trace.trace_id == collector.trace_id
        assert trace.session_id == collector.session_id

    def test_includes_root_span(self) -> None:
        """Test that the trace includes the root span."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        trace = collector.get_trace()
        assert trace.root_span.span_id == root_id
        assert trace.root_span.name == "root"

    def test_includes_all_spans(self) -> None:
        """Test that the trace includes all spans."""
        collector = TraceCollector()
        collector.start_span("root")
        collector.start_span("child1")
        collector.start_span("child2")
        trace = collector.get_trace()
        assert len(trace.spans) == 3

    def test_includes_all_steps(self) -> None:
        """Test that the trace includes all steps."""
        collector = TraceCollector()
        collector.start_span("root")
        collector.add_step(TraceStepType.THOUGHT, "thought1")
        collector.add_step(TraceStepType.THOUGHT, "thought2")
        trace = collector.get_trace()
        assert len(trace.steps) == 2

    def test_includes_all_decisions(self) -> None:
        """Test that the trace includes all decisions."""
        collector = TraceCollector()
        collector.start_span("root")
        collector.add_decision("Q1?", ["A", "B"], "A", "Because", 0.9)
        collector.add_decision("Q2?", ["C", "D"], "C", "Because", 0.8)
        trace = collector.get_trace()
        assert len(trace.decisions) == 2

    def test_includes_all_errors(self) -> None:
        """Test that the trace includes all errors."""
        collector = TraceCollector()
        collector.start_span("root")
        collector.add_error("Error1", "msg1", "trace1", True)
        collector.add_error("Error2", "msg2", "trace2", False)
        trace = collector.get_trace()
        assert len(trace.errors) == 2

    def test_raises_without_root_span(self) -> None:
        """Test that get_trace raises ValueError without a root span."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.get_trace()


class TestComplexTraceScenarios:
    """Test complex trace collection scenarios."""

    def test_nested_spans_maintain_hierarchy(self) -> None:
        """Test that nested spans maintain correct parent-child relationships."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        child1_id = collector.start_span("child1")
        grandchild_id = collector.start_span("grandchild")

        assert collector._spans[root_id].parent_id is None
        assert collector._spans[child1_id].parent_id == root_id
        assert collector._spans[grandchild_id].parent_id == child1_id

    def test_ending_child_span_restores_parent_context(self) -> None:
        """Test that ending a child span restores the parent as current."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        child_id = collector.start_span("child")
        collector.end_span(child_id)

        # Current should now be root
        assert collector._current_span_id == root_id

        # Adding a step should go to root
        collector.add_step(TraceStepType.THOUGHT, "after child")
        assert collector._steps[0].span_id == root_id

    def test_multiple_sibling_spans(self) -> None:
        """Test creating multiple sibling spans under the same parent."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        child1_id = collector.start_span("child1")
        collector.end_span(child1_id)
        child2_id = collector.start_span("child2")

        assert collector._spans[child1_id].parent_id == root_id
        assert collector._spans[child2_id].parent_id == root_id

    def test_explicit_parent_override(self) -> None:
        """Test that explicit parent_id parameter overrides current span."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        child1_id = collector.start_span("child1")
        # Create child2 as sibling of child1, not child of child1
        child2_id = collector.start_span("child2", parent_id=root_id)

        assert collector._spans[child2_id].parent_id == root_id
        assert collector._spans[child2_id].parent_id != child1_id

    def test_complete_trace_with_all_elements(self) -> None:
        """Test a complete trace with spans, steps, decisions, and errors."""
        collector = TraceCollector(session_id="test-session")

        # Create span hierarchy
        root_id = collector.start_span("reasoning", attributes={"method": "cot"})
        collector.add_step(TraceStepType.INPUT, "User query")

        child_id = collector.start_span("analysis")
        collector.add_step(TraceStepType.THOUGHT, "Analyzing problem")
        collector.add_decision(
            question="Which approach?",
            options=["Sequential", "Parallel"],
            chosen="Sequential",
            reasoning="Simpler for this case",
            confidence=0.85,
        )
        collector.add_step(TraceStepType.OUTPUT, "Analysis complete")
        collector.end_span(child_id)

        # Add error to root span
        collector.add_error(
            error_type="ValidationError",
            message="Invalid input format",
            stack_trace="File line.py, line 42",
            recoverable=True,
            recovery_action="Used default format",
        )

        collector.add_step(TraceStepType.OUTPUT, "Final result")
        collector.end_span(root_id)

        # Get and verify trace
        trace = collector.get_trace()
        assert trace.session_id == "test-session"
        assert len(trace.spans) == 2
        assert len(trace.steps) == 4
        assert len(trace.decisions) == 1
        assert len(trace.errors) == 1
        assert trace.root_span.span_id == root_id
        assert trace.root_span.status == SpanStatus.COMPLETED


class TestRecordDecision:
    """Test the record_decision method."""

    def test_records_decision(self) -> None:
        """Test that record_decision records a decision."""
        collector = TraceCollector()
        collector.start_span("test")
        decision_id = collector.record_decision(
            question="Which path?",
            options=["A", "B", "C"],
            chosen="B",
            reasoning="B is optimal",
            confidence=0.9,
        )
        assert decision_id is not None
        assert len(collector._decisions) == 1

    def test_raises_without_span(self) -> None:
        """Test that record_decision raises without an active span."""
        collector = TraceCollector()
        with pytest.raises(ValueError):
            collector.record_decision("q", ["a"], "a", "r", 0.5)


class TestRecordError:
    """Test the record_error method."""

    def test_records_exception(self) -> None:
        """Test that record_error records an exception."""
        collector = TraceCollector()
        collector.start_span("test")
        try:
            raise ValueError("test error")
        except ValueError as e:
            error_id = collector.record_error(e)
        assert error_id is not None
        assert len(collector._errors) == 1
        assert collector._errors[0].error_type == "ValueError"

    def test_includes_recovery_info(self) -> None:
        """Test that record_error includes recovery info."""
        collector = TraceCollector()
        collector.start_span("test")
        try:
            raise RuntimeError("oops")
        except RuntimeError as e:
            collector.record_error(e, recoverable=True, recovery_action="retry")
        assert collector._errors[0].recoverable is True
        assert collector._errors[0].recovery_action == "retry"


class TestCheckpoint:
    """Test the checkpoint method."""

    def test_creates_checkpoint_step(self) -> None:
        """Test that checkpoint creates a checkpoint step."""
        collector = TraceCollector()
        collector.start_span("test")
        step_id = collector.checkpoint("midpoint", {"progress": 50})
        assert step_id is not None
        assert len(collector._steps) == 1
        assert collector._steps[0].step_type == TraceStepType.CHECKPOINT
        assert collector._steps[0].content == "midpoint"


class TestGetSpanTree:
    """Test the get_span_tree method."""

    def test_returns_tree_structure(self) -> None:
        """Test that get_span_tree returns tree structure."""
        collector = TraceCollector()
        root_id = collector.start_span("root")
        child_id = collector.start_span("child")
        collector.end_span(child_id)
        collector.end_span(root_id)

        tree = collector.get_span_tree()
        assert tree["name"] == "root"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["name"] == "child"

    def test_returns_empty_without_spans(self) -> None:
        """Test that get_span_tree returns empty dict without spans."""
        collector = TraceCollector()
        assert collector.get_span_tree() == {}
