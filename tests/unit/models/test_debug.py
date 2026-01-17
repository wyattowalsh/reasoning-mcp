"""
Tests for debug models in reasoning_mcp.models.debug.

This module provides test coverage for debug-related enumerations and models:
- TraceLevel (4 values)
- SpanStatus (4 values)
- TraceSpan (Pydantic model)
"""

from datetime import datetime
from enum import Enum, StrEnum

import pytest

from reasoning_mcp.models.debug import (
    SpanStatus,
    Trace,
    TraceDecision,
    TraceError,
    TraceLevel,
    TraceSpan,
    TraceStep,
    TraceStepType,
)

# ============================================================================
# TraceLevel Tests
# ============================================================================


class TestTraceLevel:
    """Test suite for TraceLevel enum (4 values)."""

    EXPECTED_TRACE_LEVELS = {
        "MINIMAL",
        "STANDARD",
        "DETAILED",
        "VERBOSE",
    }

    EXPECTED_COUNT = 4

    def test_is_strenum(self):
        """Test that TraceLevel is a StrEnum."""
        assert issubclass(TraceLevel, StrEnum)
        assert issubclass(TraceLevel, str)

    def test_all_expected_values_exist(self):
        """Test that all 4 expected trace levels exist."""
        actual_names = {member.name for member in TraceLevel}
        assert actual_names == self.EXPECTED_TRACE_LEVELS

    def test_value_count(self):
        """Test that exactly 4 trace levels are defined."""
        assert len(TraceLevel) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in TraceLevel:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in TraceLevel]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        assert str(TraceLevel.MINIMAL) == "minimal"
        assert str(TraceLevel.STANDARD) == "standard"
        assert str(TraceLevel.DETAILED) == "detailed"
        assert str(TraceLevel.VERBOSE) == "verbose"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "minimal" in TraceLevel._value2member_map_
        assert "standard" in TraceLevel._value2member_map_
        assert "detailed" in TraceLevel._value2member_map_
        assert "verbose" in TraceLevel._value2member_map_

        # Invalid values
        assert "invalid_level" not in TraceLevel._value2member_map_
        assert "" not in TraceLevel._value2member_map_
        assert "MINIMAL" not in TraceLevel._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert TraceLevel("minimal") == TraceLevel.MINIMAL
        assert TraceLevel("standard") == TraceLevel.STANDARD
        assert TraceLevel("detailed") == TraceLevel.DETAILED
        assert TraceLevel("verbose") == TraceLevel.VERBOSE

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            TraceLevel("invalid_level")

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(TraceLevel)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, TraceLevel) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in TraceLevel:
            # All values should be lowercase
            assert member.value.islower()
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value

    def test_trace_level_hierarchy(self):
        """Test that all trace levels exist in expected hierarchy."""
        # Minimal level
        assert hasattr(TraceLevel, "MINIMAL")
        # Standard level
        assert hasattr(TraceLevel, "STANDARD")
        # Detailed level
        assert hasattr(TraceLevel, "DETAILED")
        # Verbose level
        assert hasattr(TraceLevel, "VERBOSE")

    def test_trace_level_comparison(self):
        """Test that trace levels can be compared within same type."""
        # Identity comparison
        assert TraceLevel.MINIMAL == TraceLevel.MINIMAL
        assert TraceLevel.VERBOSE == TraceLevel.VERBOSE

        # Inequality
        assert TraceLevel.MINIMAL != TraceLevel.STANDARD
        assert TraceLevel.DETAILED != TraceLevel.VERBOSE

    def test_enum_import(self):
        """Test that TraceLevel can be imported correctly."""
        from reasoning_mcp.models.debug import TraceLevel as ImportedTraceLevel

        assert ImportedTraceLevel is TraceLevel
        assert hasattr(ImportedTraceLevel, "MINIMAL")
        assert hasattr(ImportedTraceLevel, "STANDARD")
        assert hasattr(ImportedTraceLevel, "DETAILED")
        assert hasattr(ImportedTraceLevel, "VERBOSE")


# ============================================================================
# SpanStatus Tests
# ============================================================================


class TestSpanStatus:
    """Test suite for SpanStatus enum (4 values)."""

    EXPECTED_STATUSES = {"RUNNING", "COMPLETED", "FAILED", "CANCELLED"}
    EXPECTED_COUNT = 4

    def test_is_enum(self):
        """Test that SpanStatus is an Enum."""
        assert issubclass(SpanStatus, Enum)

    def test_all_expected_values_exist(self):
        """Test that all 4 expected statuses exist."""
        actual_names = {member.name for member in SpanStatus}
        assert actual_names == self.EXPECTED_STATUSES

    def test_value_count(self):
        """Test that exactly 4 statuses are defined."""
        assert len(SpanStatus) == self.EXPECTED_COUNT

    def test_string_values(self):
        """Test string values of enum members."""
        assert SpanStatus.RUNNING.value == "running"
        assert SpanStatus.COMPLETED.value == "completed"
        assert SpanStatus.FAILED.value == "failed"
        assert SpanStatus.CANCELLED.value == "cancelled"

    def test_enum_import(self):
        """Test that SpanStatus can be imported correctly."""
        from reasoning_mcp.models.debug import SpanStatus as ImportedSpanStatus

        assert ImportedSpanStatus is SpanStatus
        assert len(list(ImportedSpanStatus)) == 4


# ============================================================================
# TraceSpan Tests
# ============================================================================


class TestTraceSpan:
    """Test suite for TraceSpan Pydantic model."""

    def test_create_with_required_fields(self):
        """Test creating TraceSpan with required fields."""
        now = datetime.now()
        span = TraceSpan(
            span_id="span-123",
            name="test_span",
            start_time=now,
            status=SpanStatus.RUNNING,
        )
        assert span.span_id == "span-123"
        assert span.name == "test_span"
        assert span.start_time == now
        assert span.status == SpanStatus.RUNNING
        assert span.parent_id is None
        assert span.end_time is None
        assert span.attributes == {}

    def test_create_with_all_fields(self):
        """Test creating TraceSpan with all fields."""
        start = datetime.now()
        end = datetime.now()
        span = TraceSpan(
            span_id="span-456",
            parent_id="span-123",
            name="child_span",
            start_time=start,
            end_time=end,
            status=SpanStatus.COMPLETED,
            attributes={"key": "value", "count": 42},
        )
        assert span.span_id == "span-456"
        assert span.parent_id == "span-123"
        assert span.name == "child_span"
        assert span.start_time == start
        assert span.end_time == end
        assert span.status == SpanStatus.COMPLETED
        assert span.attributes == {"key": "value", "count": 42}

    def test_attributes_default_to_empty_dict(self):
        """Test that attributes defaults to empty dict."""
        span = TraceSpan(
            span_id="span-789",
            name="test",
            start_time=datetime.now(),
            status=SpanStatus.RUNNING,
        )
        assert span.attributes == {}
        assert isinstance(span.attributes, dict)

    def test_model_serialization(self):
        """Test that TraceSpan can be serialized to dict."""
        now = datetime.now()
        span = TraceSpan(
            span_id="span-abc",
            name="serialize_test",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )
        data = span.model_dump()
        assert data["span_id"] == "span-abc"
        assert data["name"] == "serialize_test"
        assert data["status"] == SpanStatus.COMPLETED


# ============================================================================
# TraceStep Tests
# ============================================================================


class TestTraceStep:
    """Test suite for TraceStep Pydantic model."""

    def test_create_with_required_fields(self):
        """Test creating TraceStep with required fields."""
        now = datetime.now()
        step = TraceStep(
            step_id="step-123",
            span_id="span-456",
            step_type=TraceStepType.THOUGHT,
            timestamp=now,
            content="Processing the input data",
        )
        assert step.step_id == "step-123"
        assert step.span_id == "span-456"
        assert step.step_type == TraceStepType.THOUGHT
        assert step.timestamp == now
        assert step.content == "Processing the input data"
        assert step.metadata == {}

    def test_create_with_all_fields(self):
        """Test creating TraceStep with all fields."""
        now = datetime.now()
        step = TraceStep(
            step_id="step-789",
            span_id="span-101",
            step_type=TraceStepType.DECISION,
            timestamp=now,
            content="Choosing method A over method B",
            metadata={"confidence": 0.95, "alternatives": ["method_b", "method_c"]},
        )
        assert step.step_id == "step-789"
        assert step.span_id == "span-101"
        assert step.step_type == TraceStepType.DECISION
        assert step.timestamp == now
        assert step.content == "Choosing method A over method B"
        assert step.metadata == {"confidence": 0.95, "alternatives": ["method_b", "method_c"]}

    def test_metadata_default_to_empty_dict(self):
        """Test that metadata defaults to empty dict."""
        step = TraceStep(
            step_id="step-def",
            span_id="span-ghi",
            step_type=TraceStepType.INPUT,
            timestamp=datetime.now(),
            content="User input received",
        )
        assert step.metadata == {}
        assert isinstance(step.metadata, dict)

    def test_all_step_types(self):
        """Test creating steps with all TraceStepType values."""
        now = datetime.now()
        step_types = [
            TraceStepType.METHOD_START,
            TraceStepType.METHOD_END,
            TraceStepType.THOUGHT,
            TraceStepType.DECISION,
            TraceStepType.INPUT,
            TraceStepType.OUTPUT,
            TraceStepType.ERROR,
            TraceStepType.CHECKPOINT,
        ]

        for idx, step_type in enumerate(step_types):
            step = TraceStep(
                step_id=f"step-{idx}",
                span_id="span-001",
                step_type=step_type,
                timestamp=now,
                content=f"Step of type {step_type.value}",
            )
            assert step.step_type == step_type

    def test_model_serialization(self):
        """Test that TraceStep can be serialized to dict."""
        now = datetime.now()
        step = TraceStep(
            step_id="step-serialize",
            span_id="span-serialize",
            step_type=TraceStepType.OUTPUT,
            timestamp=now,
            content="Final output generated",
        )
        data = step.model_dump()
        assert data["step_id"] == "step-serialize"
        assert data["span_id"] == "span-serialize"
        assert data["step_type"] == TraceStepType.OUTPUT
        assert data["content"] == "Final output generated"


# ============================================================================
# TraceError Tests
# ============================================================================


class TestTraceError:
    """Test suite for TraceError Pydantic model."""

    def test_create_with_required_fields(self):
        """Test creating TraceError with required fields."""
        error = TraceError(
            error_id="error-123",
            span_id="span-456",
            error_type="ValueError",
            message="Invalid input provided",
            stack_trace="Traceback (most recent call last):\n  File...",
            recoverable=True,
        )
        assert error.error_id == "error-123"
        assert error.span_id == "span-456"
        assert error.error_type == "ValueError"
        assert error.message == "Invalid input provided"
        assert error.stack_trace.startswith("Traceback")
        assert error.recoverable is True
        assert error.recovery_action is None

    def test_create_with_recovery_action(self):
        """Test creating TraceError with recovery action."""
        error = TraceError(
            error_id="error-789",
            span_id="span-101",
            error_type="TimeoutError",
            message="Operation timed out",
            stack_trace="Traceback...",
            recoverable=True,
            recovery_action="Retried with exponential backoff",
        )
        assert error.recoverable is True
        assert error.recovery_action == "Retried with exponential backoff"

    def test_non_recoverable_error(self):
        """Test creating a non-recoverable error."""
        error = TraceError(
            error_id="error-fatal",
            span_id="span-critical",
            error_type="SystemError",
            message="Critical system failure",
            stack_trace="Traceback...",
            recoverable=False,
        )
        assert error.recoverable is False
        assert error.recovery_action is None

    def test_model_serialization(self):
        """Test that TraceError can be serialized to dict."""
        error = TraceError(
            error_id="error-serialize",
            span_id="span-serialize",
            error_type="RuntimeError",
            message="Runtime error occurred",
            stack_trace="Stack trace details",
            recoverable=True,
            recovery_action="Applied fallback logic",
        )
        data = error.model_dump()
        assert data["error_id"] == "error-serialize"
        assert data["span_id"] == "span-serialize"
        assert data["error_type"] == "RuntimeError"
        assert data["message"] == "Runtime error occurred"
        assert data["recoverable"] is True
        assert data["recovery_action"] == "Applied fallback logic"

    def test_non_recoverable_with_recovery_action_description(self):
        """Test non-recoverable error can still have recovery_action for documentation."""
        # This might represent a recovery action that was attempted but failed
        error = TraceError(
            error_id="error-failed-recovery",
            span_id="span-failed",
            error_type="CriticalError",
            message="Critical failure occurred",
            stack_trace="Traceback...",
            recoverable=False,
            recovery_action="Attempted recovery but failed; terminating process",
        )
        assert error.recoverable is False
        assert error.recovery_action == "Attempted recovery but failed; terminating process"


# ============================================================================
# TraceDecision Tests
# ============================================================================


class TestTraceDecision:
    """Test suite for TraceDecision Pydantic model."""

    def test_create_with_required_fields(self):
        """Test creating TraceDecision with all required fields."""
        decision = TraceDecision(
            decision_id="decision-001",
            span_id="span-123",
            question="Should we proceed with option A or option B?",
            options=["option A", "option B"],
            chosen="option A",
            reasoning="Option A has better performance characteristics",
            confidence=0.85,
        )
        assert decision.decision_id == "decision-001"
        assert decision.span_id == "span-123"
        assert decision.question == "Should we proceed with option A or option B?"
        assert decision.options == ["option A", "option B"]
        assert decision.chosen == "option A"
        assert decision.reasoning == "Option A has better performance characteristics"
        assert decision.confidence == 0.85

    def test_confidence_within_valid_range(self):
        """Test that confidence values within 0.0 to 1.0 are accepted."""
        # Test minimum value
        decision1 = TraceDecision(
            decision_id="decision-min",
            span_id="span-001",
            question="Test question?",
            options=["yes", "no"],
            chosen="yes",
            reasoning="Test reasoning",
            confidence=0.0,
        )
        assert decision1.confidence == 0.0

        # Test maximum value
        decision2 = TraceDecision(
            decision_id="decision-max",
            span_id="span-002",
            question="Test question?",
            options=["yes", "no"],
            chosen="yes",
            reasoning="Test reasoning",
            confidence=1.0,
        )
        assert decision2.confidence == 1.0

        # Test mid-range value
        decision3 = TraceDecision(
            decision_id="decision-mid",
            span_id="span-003",
            question="Test question?",
            options=["yes", "no"],
            chosen="yes",
            reasoning="Test reasoning",
            confidence=0.5,
        )
        assert decision3.confidence == 0.5

    def test_confidence_below_minimum_fails(self):
        """Test that confidence values below 0.0 raise validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TraceDecision(
                decision_id="decision-invalid",
                span_id="span-001",
                question="Test question?",
                options=["yes", "no"],
                chosen="yes",
                reasoning="Test reasoning",
                confidence=-0.1,
            )

    def test_confidence_above_maximum_fails(self):
        """Test that confidence values above 1.0 raise validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            TraceDecision(
                decision_id="decision-invalid",
                span_id="span-001",
                question="Test question?",
                options=["yes", "no"],
                chosen="yes",
                reasoning="Test reasoning",
                confidence=1.1,
            )

    def test_multiple_options(self):
        """Test creating decisions with various numbers of options."""
        # Two options
        decision1 = TraceDecision(
            decision_id="decision-two",
            span_id="span-001",
            question="Binary choice?",
            options=["option1", "option2"],
            chosen="option1",
            reasoning="Best choice",
            confidence=0.9,
        )
        assert len(decision1.options) == 2

        # Multiple options
        decision2 = TraceDecision(
            decision_id="decision-multi",
            span_id="span-002",
            question="Multi-choice?",
            options=["A", "B", "C", "D", "E"],
            chosen="C",
            reasoning="C is optimal",
            confidence=0.75,
        )
        assert len(decision2.options) == 5
        assert "C" in decision2.options

    def test_chosen_option_validation(self):
        """Test that chosen option can be any string (not validated against options list)."""
        # Chosen option exists in options
        decision1 = TraceDecision(
            decision_id="decision-valid",
            span_id="span-001",
            question="Which method?",
            options=["method_a", "method_b", "method_c"],
            chosen="method_b",
            reasoning="Method B is fastest",
            confidence=0.88,
        )
        assert decision1.chosen == "method_b"
        assert decision1.chosen in decision1.options

        # Note: Pydantic model allows chosen to be any string
        # Validation logic would be in business logic, not the model itself

    def test_model_serialization(self):
        """Test that TraceDecision can be serialized to dict."""
        decision = TraceDecision(
            decision_id="decision-serialize",
            span_id="span-serialize",
            question="Which algorithm to use?",
            options=["quicksort", "mergesort", "heapsort"],
            chosen="mergesort",
            reasoning="Best average-case performance for this dataset",
            confidence=0.92,
        )
        data = decision.model_dump()
        assert data["decision_id"] == "decision-serialize"
        assert data["span_id"] == "span-serialize"
        assert data["question"] == "Which algorithm to use?"
        assert data["options"] == ["quicksort", "mergesort", "heapsort"]
        assert data["chosen"] == "mergesort"
        assert data["reasoning"] == "Best average-case performance for this dataset"
        assert data["confidence"] == 0.92

    def test_empty_options_list(self):
        """Test creating decision with empty options list."""
        decision = TraceDecision(
            decision_id="decision-empty-options",
            span_id="span-001",
            question="Question with no options?",
            options=[],
            chosen="default",
            reasoning="No options available, using default",
            confidence=0.5,
        )
        assert decision.options == []
        assert len(decision.options) == 0

    def test_complex_question_and_reasoning(self):
        """Test decisions with complex multi-line questions and reasoning."""
        complex_question = """Given the following constraints:
        1. Limited memory availability
        2. Need for real-time processing
        3. High accuracy requirements
        Which approach should we take?"""

        complex_reasoning = """Selected streaming approach because:
        - It handles memory constraints well
        - Provides incremental results for real-time processing
        - Can maintain high accuracy with proper buffering"""

        decision = TraceDecision(
            decision_id="decision-complex",
            span_id="span-complex",
            question=complex_question,
            options=["batch processing", "streaming", "hybrid"],
            chosen="streaming",
            reasoning=complex_reasoning,
            confidence=0.87,
        )
        assert decision.question == complex_question
        assert decision.reasoning == complex_reasoning
        assert "\n" in decision.question
        assert "\n" in decision.reasoning

    def test_decision_association_with_span(self):
        """Test that TraceDecision correctly references a span_id."""
        span_id = "parent-span-abc"
        decision1 = TraceDecision(
            decision_id="decision-1",
            span_id=span_id,
            question="First decision?",
            options=["yes", "no"],
            chosen="yes",
            reasoning="Reason 1",
            confidence=0.8,
        )
        decision2 = TraceDecision(
            decision_id="decision-2",
            span_id=span_id,
            question="Second decision?",
            options=["continue", "stop"],
            chosen="continue",
            reasoning="Reason 2",
            confidence=0.9,
        )
        # Both decisions reference the same parent span
        assert decision1.span_id == decision2.span_id == span_id

    def test_low_confidence_decision(self):
        """Test creating a decision with low confidence."""
        decision = TraceDecision(
            decision_id="decision-low-conf",
            span_id="span-001",
            question="Uncertain choice?",
            options=["option1", "option2", "option3"],
            chosen="option1",
            reasoning="All options are similarly viable; choosing first",
            confidence=0.33,
        )
        assert decision.confidence < 0.5
        assert decision.confidence == 0.33

    def test_high_confidence_decision(self):
        """Test creating a decision with high confidence."""
        decision = TraceDecision(
            decision_id="decision-high-conf",
            span_id="span-001",
            question="Obvious choice?",
            options=["clearly_best", "suboptimal"],
            chosen="clearly_best",
            reasoning="This option is objectively superior based on all metrics",
            confidence=0.99,
        )
        assert decision.confidence > 0.9
        assert decision.confidence == 0.99


# ============================================================================
# Trace Tests
# ============================================================================


class TestTrace:
    """Test suite for Trace Pydantic model."""

    def test_create_with_required_fields_only(self):
        """Test creating Trace with only required fields (minimal trace)."""
        root_span = TraceSpan(
            span_id="root-001",
            name="Root Execution",
            start_time=datetime.now(),
            status=SpanStatus.COMPLETED,
        )
        trace = Trace(
            trace_id="trace-001",
            session_id="session-abc",
            root_span=root_span,
        )
        assert trace.trace_id == "trace-001"
        assert trace.session_id == "session-abc"
        assert trace.root_span == root_span
        assert trace.spans == []
        assert trace.steps == []
        assert trace.decisions == []
        assert trace.errors == []

    def test_create_with_all_fields(self):
        """Test creating Trace with all fields populated."""
        now = datetime.now()

        # Create root span
        root_span = TraceSpan(
            span_id="root-001",
            name="Root Execution",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        # Create child spans
        child_span = TraceSpan(
            span_id="child-001",
            parent_id="root-001",
            name="Child Task",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        # Create steps
        step = TraceStep(
            step_id="step-001",
            span_id="child-001",
            step_type=TraceStepType.THOUGHT,
            timestamp=now,
            content="Processing data",
        )

        # Create decision
        decision = TraceDecision(
            decision_id="decision-001",
            span_id="child-001",
            question="Which method to use?",
            options=["method_a", "method_b"],
            chosen="method_a",
            reasoning="Method A is faster",
            confidence=0.85,
        )

        # Create error
        error = TraceError(
            error_id="error-001",
            span_id="child-001",
            error_type="ValueError",
            message="Invalid input",
            stack_trace="Traceback...",
            recoverable=True,
            recovery_action="Used default value",
        )

        # Create complete trace
        trace = Trace(
            trace_id="trace-complete",
            session_id="session-xyz",
            root_span=root_span,
            spans=[root_span, child_span],
            steps=[step],
            decisions=[decision],
            errors=[error],
        )

        assert trace.trace_id == "trace-complete"
        assert trace.session_id == "session-xyz"
        assert trace.root_span == root_span
        assert len(trace.spans) == 2
        assert len(trace.steps) == 1
        assert len(trace.decisions) == 1
        assert len(trace.errors) == 1

    def test_default_empty_lists(self):
        """Test that list fields default to empty lists."""
        root_span = TraceSpan(
            span_id="root-002",
            name="Test Root",
            start_time=datetime.now(),
            status=SpanStatus.RUNNING,
        )
        trace = Trace(
            trace_id="trace-002",
            session_id="session-test",
            root_span=root_span,
        )

        # All list fields should be empty but not None
        assert isinstance(trace.spans, list)
        assert isinstance(trace.steps, list)
        assert isinstance(trace.decisions, list)
        assert isinstance(trace.errors, list)
        assert len(trace.spans) == 0
        assert len(trace.steps) == 0
        assert len(trace.decisions) == 0
        assert len(trace.errors) == 0

    def test_hierarchical_structure(self):
        """Test that Trace maintains hierarchical span relationships."""
        now = datetime.now()

        # Create a 3-level hierarchy: root -> child -> grandchild
        root = TraceSpan(
            span_id="root",
            name="Root",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )
        child = TraceSpan(
            span_id="child",
            parent_id="root",
            name="Child",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )
        grandchild = TraceSpan(
            span_id="grandchild",
            parent_id="child",
            name="Grandchild",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        trace = Trace(
            trace_id="trace-hierarchical",
            session_id="session-001",
            root_span=root,
            spans=[root, child, grandchild],
        )

        # Verify hierarchy through parent_id references
        assert trace.spans[0].parent_id is None  # root has no parent
        assert trace.spans[1].parent_id == "root"
        assert trace.spans[2].parent_id == "child"

    def test_multiple_steps_across_spans(self):
        """Test trace with multiple steps distributed across different spans."""
        now = datetime.now()

        root_span = TraceSpan(
            span_id="root",
            name="Root",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )
        child_span = TraceSpan(
            span_id="child",
            parent_id="root",
            name="Child",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        # Create steps in different spans
        step1 = TraceStep(
            step_id="step-1",
            span_id="root",
            step_type=TraceStepType.INPUT,
            timestamp=now,
            content="Root input",
        )
        step2 = TraceStep(
            step_id="step-2",
            span_id="child",
            step_type=TraceStepType.THOUGHT,
            timestamp=now,
            content="Child thought",
        )
        step3 = TraceStep(
            step_id="step-3",
            span_id="child",
            step_type=TraceStepType.OUTPUT,
            timestamp=now,
            content="Child output",
        )

        trace = Trace(
            trace_id="trace-multi-step",
            session_id="session-002",
            root_span=root_span,
            spans=[root_span, child_span],
            steps=[step1, step2, step3],
        )

        assert len(trace.steps) == 3
        # Verify steps reference their respective spans
        assert trace.steps[0].span_id == "root"
        assert trace.steps[1].span_id == "child"
        assert trace.steps[2].span_id == "child"

    def test_multiple_decisions(self):
        """Test trace with multiple decision points."""
        now = datetime.now()

        root_span = TraceSpan(
            span_id="root",
            name="Root",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        decision1 = TraceDecision(
            decision_id="dec-1",
            span_id="root",
            question="First choice?",
            options=["a", "b"],
            chosen="a",
            reasoning="A is better",
            confidence=0.9,
        )
        decision2 = TraceDecision(
            decision_id="dec-2",
            span_id="root",
            question="Second choice?",
            options=["x", "y", "z"],
            chosen="y",
            reasoning="Y is optimal",
            confidence=0.75,
        )

        trace = Trace(
            trace_id="trace-multi-decision",
            session_id="session-003",
            root_span=root_span,
            decisions=[decision1, decision2],
        )

        assert len(trace.decisions) == 2
        assert trace.decisions[0].chosen == "a"
        assert trace.decisions[1].chosen == "y"

    def test_multiple_errors(self):
        """Test trace with multiple errors recorded."""
        now = datetime.now()

        root_span = TraceSpan(
            span_id="root",
            name="Root",
            start_time=now,
            status=SpanStatus.FAILED,
        )

        error1 = TraceError(
            error_id="err-1",
            span_id="root",
            error_type="ValueError",
            message="Invalid value",
            stack_trace="Trace 1",
            recoverable=True,
            recovery_action="Used default",
        )
        error2 = TraceError(
            error_id="err-2",
            span_id="root",
            error_type="TimeoutError",
            message="Operation timed out",
            stack_trace="Trace 2",
            recoverable=False,
        )

        trace = Trace(
            trace_id="trace-multi-error",
            session_id="session-004",
            root_span=root_span,
            errors=[error1, error2],
        )

        assert len(trace.errors) == 2
        assert trace.errors[0].recoverable is True
        assert trace.errors[1].recoverable is False

    def test_model_serialization(self):
        """Test that Trace can be serialized to dict."""
        now = datetime.now()

        root_span = TraceSpan(
            span_id="root",
            name="Root",
            start_time=now,
            status=SpanStatus.COMPLETED,
        )

        trace = Trace(
            trace_id="trace-serialize",
            session_id="session-serialize",
            root_span=root_span,
        )

        data = trace.model_dump()
        assert data["trace_id"] == "trace-serialize"
        assert data["session_id"] == "session-serialize"
        assert "root_span" in data
        assert isinstance(data["spans"], list)
        assert isinstance(data["steps"], list)
        assert isinstance(data["decisions"], list)
        assert isinstance(data["errors"], list)

    def test_complete_trace_lifecycle(self):
        """Test a complete trace representing a full reasoning lifecycle."""
        start_time = datetime.now()

        # Root span for entire execution
        root_span = TraceSpan(
            span_id="execution-001",
            name="Complete Reasoning Execution",
            start_time=start_time,
            status=SpanStatus.COMPLETED,
        )

        # Child spans for different phases
        analysis_span = TraceSpan(
            span_id="analysis-001",
            parent_id="execution-001",
            name="Analysis Phase",
            start_time=start_time,
            status=SpanStatus.COMPLETED,
        )
        decision_span = TraceSpan(
            span_id="decision-001",
            parent_id="execution-001",
            name="Decision Phase",
            start_time=start_time,
            status=SpanStatus.COMPLETED,
        )

        # Steps across phases
        input_step = TraceStep(
            step_id="step-input",
            span_id="analysis-001",
            step_type=TraceStepType.INPUT,
            timestamp=start_time,
            content="Received user query",
        )
        thought_step = TraceStep(
            step_id="step-thought",
            span_id="analysis-001",
            step_type=TraceStepType.THOUGHT,
            timestamp=start_time,
            content="Analyzing query complexity",
        )
        output_step = TraceStep(
            step_id="step-output",
            span_id="decision-001",
            step_type=TraceStepType.OUTPUT,
            timestamp=start_time,
            content="Generated response",
        )

        # Decision made during execution
        method_decision = TraceDecision(
            decision_id="method-choice",
            span_id="decision-001",
            question="Which reasoning method to apply?",
            options=["chain_of_thought", "tree_of_thought", "react"],
            chosen="chain_of_thought",
            reasoning="Query complexity suggests sequential reasoning",
            confidence=0.88,
        )

        # Create complete trace
        trace = Trace(
            trace_id="complete-lifecycle",
            session_id="session-lifecycle",
            root_span=root_span,
            spans=[root_span, analysis_span, decision_span],
            steps=[input_step, thought_step, output_step],
            decisions=[method_decision],
            errors=[],
        )

        # Verify complete structure
        assert trace.trace_id == "complete-lifecycle"
        assert len(trace.spans) == 3
        assert len(trace.steps) == 3
        assert len(trace.decisions) == 1
        assert len(trace.errors) == 0
        assert trace.root_span.name == "Complete Reasoning Execution"
