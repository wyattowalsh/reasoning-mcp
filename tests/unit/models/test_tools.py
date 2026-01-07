"""
Comprehensive tests for Tool I/O models in reasoning_mcp.models.tools.

This module provides complete test coverage for all tool I/O models:
- Input Models (mutable): ReasonHints
- Output Models (frozen): ReasonOutput, ThoughtOutput, SuggestionOutput,
  ValidationOutput, ComposeOutput, SessionState, BranchOutput, MergeOutput,
  MethodInfo, Recommendation, ComparisonResult, EvaluationReport

Each model is tested for:
1. Creation with minimal and full parameters
2. Default values
3. Mutability/Immutability (frozen vs mutable)
4. Field validation (types, ranges, constraints)
5. JSON serialization/deserialization
6. Schema generation
"""

from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    SessionStatus,
    ThoughtType,
)
from reasoning_mcp.models.pipeline import PipelineTrace
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import (
    BranchOutput,
    ComparisonResult,
    ComposeOutput,
    EvaluationReport,
    MergeOutput,
    MethodInfo,
    ReasonHints,
    ReasonOutput,
    Recommendation,
    SessionState,
    SuggestionOutput,
    ThoughtOutput,
    ValidationOutput,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_thought_node() -> ThoughtNode:
    """Provide a sample ThoughtNode for testing."""
    return ThoughtNode(
        type=ThoughtType.INITIAL,
        method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        content="Let's break this problem down step by step.",
        confidence=0.85,
    )


@pytest.fixture
def sample_pipeline_trace() -> PipelineTrace:
    """Provide a sample PipelineTrace for testing."""
    return PipelineTrace(
        pipeline_id="pipeline-123",
        session_id="session-456",
        started_at=datetime.now(),
        completed_at=datetime.now(),
        status="completed",
    )


# ============================================================================
# TestReasonHints - Input Model (Mutable)
# ============================================================================


class TestReasonHints:
    """Test suite for ReasonHints model (mutable input)."""

    def test_create_reason_hints_minimal(self):
        """Test creating ReasonHints with no parameters (all defaults)."""
        hints = ReasonHints()

        assert hints.domain is None
        assert hints.complexity is None
        assert hints.prefer_methods == []
        assert hints.avoid_methods == []
        assert hints.custom_hints == {}

    def test_create_reason_hints_full(self):
        """Test creating ReasonHints with all parameters."""
        hints = ReasonHints(
            domain="code",
            complexity="high",
            prefer_methods=[MethodIdentifier.CODE_REASONING, MethodIdentifier.CHAIN_OF_THOUGHT],
            avoid_methods=[MethodIdentifier.ETHICAL_REASONING],
            custom_hints={"language": "python", "proof_required": True},
        )

        assert hints.domain == "code"
        assert hints.complexity == "high"
        assert len(hints.prefer_methods) == 2
        assert MethodIdentifier.CODE_REASONING in hints.prefer_methods
        assert MethodIdentifier.ETHICAL_REASONING in hints.avoid_methods
        assert hints.custom_hints["language"] == "python"

    def test_reason_hints_is_mutable(self):
        """Test that ReasonHints is mutable (not frozen)."""
        hints = ReasonHints()

        # Should be able to modify fields
        hints.domain = "math"
        assert hints.domain == "math"

        hints.complexity = "moderate"
        assert hints.complexity == "moderate"

        hints.prefer_methods = [MethodIdentifier.MATHEMATICAL_REASONING]
        assert MethodIdentifier.MATHEMATICAL_REASONING in hints.prefer_methods

    def test_reason_hints_default_values(self):
        """Test that ReasonHints has correct default values."""
        hints = ReasonHints()

        assert hints.domain is None
        assert hints.complexity is None
        assert hints.prefer_methods == []
        assert hints.avoid_methods == []
        assert hints.custom_hints == {}

    def test_reason_hints_validation(self):
        """Test ReasonHints field validation."""
        # Valid: method identifiers
        hints = ReasonHints(
            prefer_methods=[MethodIdentifier.TREE_OF_THOUGHTS],
            avoid_methods=[MethodIdentifier.REACT],
        )
        assert len(hints.prefer_methods) == 1
        assert len(hints.avoid_methods) == 1

        # Invalid: wrong enum type
        with pytest.raises(ValidationError):
            ReasonHints(prefer_methods=["invalid_method"])

    def test_reason_hints_serialization(self):
        """Test ReasonHints serialization to dict and JSON."""
        hints = ReasonHints(
            domain="ethical",
            complexity="high",
            prefer_methods=[MethodIdentifier.ETHICAL_REASONING],
            custom_hints={"stakeholders": ["users", "developers"]},
        )

        # To dict
        data = hints.model_dump()
        assert data["domain"] == "ethical"
        assert data["complexity"] == "high"
        assert len(data["prefer_methods"]) == 1

        # From dict
        restored = ReasonHints(**data)
        assert restored.domain == hints.domain
        assert restored.custom_hints == hints.custom_hints

    def test_reason_hints_is_pydantic_basemodel(self):
        """Test that ReasonHints is a Pydantic BaseModel."""
        assert issubclass(ReasonHints, BaseModel)


# ============================================================================
# TestReasonOutput - Output Model (Frozen)
# ============================================================================


class TestReasonOutput:
    """Test suite for ReasonOutput model (frozen output)."""

    def test_create_reason_output_minimal(self, sample_thought_node: ThoughtNode):
        """Test creating ReasonOutput with minimal required parameters."""
        output = ReasonOutput(
            session_id="session-123",
            thought=sample_thought_node,
            method_used=MethodIdentifier.CHAIN_OF_THOUGHT,
        )

        assert output.session_id == "session-123"
        assert output.thought == sample_thought_node
        assert output.method_used == MethodIdentifier.CHAIN_OF_THOUGHT
        assert output.suggestions == []
        assert output.metadata == {}

    def test_create_reason_output_full(self, sample_thought_node: ThoughtNode):
        """Test creating ReasonOutput with all parameters."""
        output = ReasonOutput(
            session_id="session-456",
            thought=sample_thought_node,
            method_used=MethodIdentifier.TREE_OF_THOUGHTS,
            suggestions=["Continue exploration", "Prune low-confidence branches"],
            metadata={"tokens_used": 250, "inference_time_ms": 540},
        )

        assert output.session_id == "session-456"
        assert output.method_used == MethodIdentifier.TREE_OF_THOUGHTS
        assert len(output.suggestions) == 2
        assert output.metadata["tokens_used"] == 250

    def test_reason_output_is_frozen(self, sample_thought_node: ThoughtNode):
        """Test that ReasonOutput is frozen (immutable)."""
        output = ReasonOutput(
            session_id="session-123",
            thought=sample_thought_node,
            method_used=MethodIdentifier.CHAIN_OF_THOUGHT,
        )

        with pytest.raises(ValidationError):
            output.session_id = "modified"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            output.method_used = MethodIdentifier.REACT  # type: ignore[misc]

    def test_reason_output_validation(self):
        """Test ReasonOutput field validation."""
        # Valid creation
        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SEQUENTIAL_THINKING,
            content="Test thought",
        )
        output = ReasonOutput(
            session_id="test-session",
            thought=thought,
            method_used=MethodIdentifier.SEQUENTIAL_THINKING,
        )
        assert output.session_id == "test-session"

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            ReasonOutput()  # type: ignore[call-arg]

    def test_reason_output_serialization(self, sample_thought_node: ThoughtNode):
        """Test ReasonOutput serialization to dict and JSON."""
        output = ReasonOutput(
            session_id="serialize-test",
            thought=sample_thought_node,
            method_used=MethodIdentifier.SELF_CONSISTENCY,
            suggestions=["Validate results"],
        )

        # To dict
        data = output.model_dump()
        assert data["session_id"] == "serialize-test"
        assert "thought" in data
        assert len(data["suggestions"]) == 1

        # From dict
        restored = ReasonOutput(**data)
        assert restored.session_id == output.session_id
        assert restored.method_used == output.method_used


# ============================================================================
# TestThoughtOutput - Output Model (Frozen)
# ============================================================================


class TestThoughtOutput:
    """Test suite for ThoughtOutput model (frozen output)."""

    def test_create_thought_output_minimal(self):
        """Test creating ThoughtOutput with minimal required parameters."""
        output = ThoughtOutput(
            id="thought-123",
            content="This is a key insight.",
            thought_type=ThoughtType.SYNTHESIS,
        )

        assert output.id == "thought-123"
        assert output.content == "This is a key insight."
        assert output.thought_type == ThoughtType.SYNTHESIS
        assert output.confidence is None
        assert output.step_number is None

    def test_create_thought_output_full(self):
        """Test creating ThoughtOutput with all parameters."""
        output = ThoughtOutput(
            id="thought-456",
            content="Complete analysis with all details.",
            thought_type=ThoughtType.CONCLUSION,
            confidence=0.95,
            step_number=10,
        )

        assert output.id == "thought-456"
        assert output.confidence == 0.95
        assert output.step_number == 10
        assert output.thought_type == ThoughtType.CONCLUSION

    def test_thought_output_is_frozen(self):
        """Test that ThoughtOutput is frozen (immutable)."""
        output = ThoughtOutput(
            id="thought-123",
            content="Test content",
            thought_type=ThoughtType.INITIAL,
        )

        with pytest.raises(ValidationError):
            output.content = "modified"  # type: ignore[misc]

    def test_thought_output_confidence_validation(self):
        """Test ThoughtOutput confidence field validation (0.0-1.0)."""
        # Valid: boundary values
        output_0 = ThoughtOutput(
            id="test-1",
            content="Test",
            thought_type=ThoughtType.INITIAL,
            confidence=0.0,
        )
        assert output_0.confidence == 0.0

        output_1 = ThoughtOutput(
            id="test-2",
            content="Test",
            thought_type=ThoughtType.INITIAL,
            confidence=1.0,
        )
        assert output_1.confidence == 1.0

        # Invalid: below 0
        with pytest.raises(ValidationError):
            ThoughtOutput(
                id="test-3",
                content="Test",
                thought_type=ThoughtType.INITIAL,
                confidence=-0.1,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            ThoughtOutput(
                id="test-4",
                content="Test",
                thought_type=ThoughtType.INITIAL,
                confidence=1.1,
            )

    def test_thought_output_step_number_validation(self):
        """Test ThoughtOutput step_number field validation (>= 0)."""
        # Valid: non-negative
        output = ThoughtOutput(
            id="test",
            content="Test",
            thought_type=ThoughtType.CONTINUATION,
            step_number=0,
        )
        assert output.step_number == 0

        output = ThoughtOutput(
            id="test",
            content="Test",
            thought_type=ThoughtType.CONTINUATION,
            step_number=100,
        )
        assert output.step_number == 100

        # Invalid: negative
        with pytest.raises(ValidationError):
            ThoughtOutput(
                id="test",
                content="Test",
                thought_type=ThoughtType.CONTINUATION,
                step_number=-1,
            )

    def test_thought_output_serialization(self):
        """Test ThoughtOutput serialization to dict and JSON."""
        output = ThoughtOutput(
            id="serialize-test",
            content="Test content",
            thought_type=ThoughtType.HYPOTHESIS,
            confidence=0.75,
            step_number=5,
        )

        # To dict
        data = output.model_dump()
        assert data["id"] == "serialize-test"
        assert data["confidence"] == 0.75

        # From dict
        restored = ThoughtOutput(**data)
        assert restored.id == output.id
        assert restored.thought_type == output.thought_type


# ============================================================================
# TestSuggestionOutput - Output Model (Frozen)
# ============================================================================


class TestSuggestionOutput:
    """Test suite for SuggestionOutput model (frozen output)."""

    def test_create_suggestion_output_minimal(self):
        """Test creating SuggestionOutput with minimal required parameters."""
        output = SuggestionOutput(
            suggestions=["Continue analysis", "Verify assumptions"],
        )

        assert len(output.suggestions) == 2
        assert output.recommended_methods == []
        assert output.context is None

    def test_create_suggestion_output_full(self):
        """Test creating SuggestionOutput with all parameters."""
        output = SuggestionOutput(
            suggestions=[
                "Analyze ethical implications",
                "Consider stakeholder impacts",
            ],
            recommended_methods=[
                MethodIdentifier.ETHICAL_REASONING,
                MethodIdentifier.DIALECTIC,
            ],
            context="High complexity ethical decision with multiple stakeholders",
        )

        assert len(output.suggestions) == 2
        assert len(output.recommended_methods) == 2
        assert "ethical decision" in output.context

    def test_suggestion_output_is_frozen(self):
        """Test that SuggestionOutput is frozen (immutable)."""
        output = SuggestionOutput(
            suggestions=["Test suggestion"],
        )

        with pytest.raises(ValidationError):
            output.suggestions = ["modified"]  # type: ignore[misc]

    def test_suggestion_output_validation(self):
        """Test SuggestionOutput field validation."""
        # Valid: suggestions required
        output = SuggestionOutput(
            suggestions=["Valid suggestion"],
        )
        assert len(output.suggestions) == 1

        # Invalid: missing required field
        with pytest.raises(ValidationError):
            SuggestionOutput()  # type: ignore[call-arg]

    def test_suggestion_output_serialization(self):
        """Test SuggestionOutput serialization to dict and JSON."""
        output = SuggestionOutput(
            suggestions=["Break down the problem", "Use systematic approach"],
            recommended_methods=[MethodIdentifier.CHAIN_OF_THOUGHT],
            context="Complex reasoning task",
        )

        # To dict
        data = output.model_dump()
        assert len(data["suggestions"]) == 2
        assert data["context"] == "Complex reasoning task"

        # From dict
        restored = SuggestionOutput(**data)
        assert restored.suggestions == output.suggestions


# ============================================================================
# TestValidationOutput - Output Model (Frozen)
# ============================================================================


class TestValidationOutput:
    """Test suite for ValidationOutput model (frozen output)."""

    def test_create_validation_output_valid(self):
        """Test creating ValidationOutput for valid state."""
        output = ValidationOutput(valid=True)

        assert output.valid is True
        assert output.errors == []
        assert output.warnings == []

    def test_create_validation_output_invalid(self):
        """Test creating ValidationOutput for invalid state."""
        output = ValidationOutput(
            valid=False,
            errors=["Thought content is empty", "Parent ID references non-existent thought"],
            warnings=["Depth exceeds recommended maximum"],
        )

        assert output.valid is False
        assert len(output.errors) == 2
        assert len(output.warnings) == 1

    def test_create_validation_output_with_warnings_only(self):
        """Test creating ValidationOutput with warnings but still valid."""
        output = ValidationOutput(
            valid=True,
            warnings=["Confidence score is relatively low (0.45)"],
        )

        assert output.valid is True
        assert len(output.errors) == 0
        assert len(output.warnings) == 1

    def test_validation_output_is_frozen(self):
        """Test that ValidationOutput is frozen (immutable)."""
        output = ValidationOutput(valid=True)

        with pytest.raises(ValidationError):
            output.valid = False  # type: ignore[misc]

    def test_validation_output_validation(self):
        """Test ValidationOutput field validation."""
        # Valid creation
        output = ValidationOutput(valid=True)
        assert output.valid is True

        # Invalid: missing required field
        with pytest.raises(ValidationError):
            ValidationOutput()  # type: ignore[call-arg]

    def test_validation_output_serialization(self):
        """Test ValidationOutput serialization to dict and JSON."""
        output = ValidationOutput(
            valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )

        # To dict
        data = output.model_dump()
        assert data["valid"] is False
        assert len(data["errors"]) == 2

        # From dict
        restored = ValidationOutput(**data)
        assert restored.valid == output.valid
        assert restored.errors == output.errors


# ============================================================================
# TestComposeOutput - Output Model (Frozen)
# ============================================================================


class TestComposeOutput:
    """Test suite for ComposeOutput model (frozen output)."""

    def test_create_compose_output_success(self, sample_thought_node: ThoughtNode):
        """Test creating successful ComposeOutput."""
        output = ComposeOutput(
            session_id="session-123",
            pipeline_id="pipeline-abc",
            success=True,
            final_thoughts=[sample_thought_node],
        )

        assert output.session_id == "session-123"
        assert output.pipeline_id == "pipeline-abc"
        assert output.success is True
        assert len(output.final_thoughts) == 1
        assert output.trace is None
        assert output.error is None

    def test_create_compose_output_with_trace(
        self, sample_thought_node: ThoughtNode, sample_pipeline_trace: PipelineTrace
    ):
        """Test creating ComposeOutput with execution trace."""
        output = ComposeOutput(
            session_id="session-456",
            pipeline_id="pipeline-def",
            success=True,
            final_thoughts=[sample_thought_node],
            trace=sample_pipeline_trace,
        )

        assert output.trace is not None
        assert output.trace.pipeline_id == "pipeline-123"

    def test_create_compose_output_failure(self):
        """Test creating failed ComposeOutput."""
        output = ComposeOutput(
            session_id="session-789",
            pipeline_id="pipeline-ghi",
            success=False,
            error="Pipeline execution failed at stage 'ethical_analysis': timeout exceeded",
        )

        assert output.success is False
        assert output.error is not None
        assert "timeout exceeded" in output.error
        assert output.final_thoughts == []

    def test_compose_output_is_frozen(self):
        """Test that ComposeOutput is frozen (immutable)."""
        output = ComposeOutput(
            session_id="session-123",
            pipeline_id="pipeline-abc",
            success=True,
        )

        with pytest.raises(ValidationError):
            output.success = False  # type: ignore[misc]

    def test_compose_output_validation(self):
        """Test ComposeOutput field validation."""
        # Valid: minimal required fields
        output = ComposeOutput(
            session_id="test",
            pipeline_id="test-pipeline",
            success=True,
        )
        assert output.success is True

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            ComposeOutput()  # type: ignore[call-arg]

    def test_compose_output_serialization(self, sample_thought_node: ThoughtNode):
        """Test ComposeOutput serialization to dict and JSON."""
        output = ComposeOutput(
            session_id="serialize-test",
            pipeline_id="pipeline-test",
            success=True,
            final_thoughts=[sample_thought_node],
        )

        # To dict
        data = output.model_dump()
        assert data["session_id"] == "serialize-test"
        assert data["success"] is True

        # From dict
        restored = ComposeOutput(**data)
        assert restored.session_id == output.session_id


# ============================================================================
# TestSessionState - Output Model (Frozen)
# ============================================================================


class TestSessionState:
    """Test suite for SessionState model (frozen output)."""

    def test_create_session_state_minimal(self):
        """Test creating SessionState with minimal required parameters."""
        state = SessionState(
            session_id="session-123",
            status=SessionStatus.ACTIVE,
            thought_count=42,
            branch_count=3,
        )

        assert state.session_id == "session-123"
        assert state.status == SessionStatus.ACTIVE
        assert state.thought_count == 42
        assert state.branch_count == 3
        assert state.current_method is None
        assert state.started_at is None
        assert state.updated_at is None

    def test_create_session_state_full(self):
        """Test creating SessionState with all parameters."""
        started = datetime(2025, 1, 1, 10, 0, 0)
        updated = datetime(2025, 1, 1, 10, 15, 30)

        state = SessionState(
            session_id="session-456",
            status=SessionStatus.COMPLETED,
            thought_count=67,
            branch_count=5,
            current_method=MethodIdentifier.TREE_OF_THOUGHTS,
            started_at=started,
            updated_at=updated,
        )

        assert state.current_method == MethodIdentifier.TREE_OF_THOUGHTS
        assert state.started_at == started
        assert state.updated_at == updated

    def test_session_state_is_frozen(self):
        """Test that SessionState is frozen (immutable)."""
        state = SessionState(
            session_id="session-123",
            status=SessionStatus.ACTIVE,
            thought_count=10,
            branch_count=1,
        )

        with pytest.raises(ValidationError):
            state.status = SessionStatus.COMPLETED  # type: ignore[misc]

    def test_session_state_thought_count_validation(self):
        """Test SessionState thought_count field validation (>= 0)."""
        # Valid: non-negative
        state = SessionState(
            session_id="test",
            status=SessionStatus.ACTIVE,
            thought_count=0,
            branch_count=0,
        )
        assert state.thought_count == 0

        # Invalid: negative
        with pytest.raises(ValidationError):
            SessionState(
                session_id="test",
                status=SessionStatus.ACTIVE,
                thought_count=-1,
                branch_count=0,
            )

    def test_session_state_branch_count_validation(self):
        """Test SessionState branch_count field validation (>= 0)."""
        # Valid: non-negative
        state = SessionState(
            session_id="test",
            status=SessionStatus.ACTIVE,
            thought_count=10,
            branch_count=0,
        )
        assert state.branch_count == 0

        # Invalid: negative
        with pytest.raises(ValidationError):
            SessionState(
                session_id="test",
                status=SessionStatus.ACTIVE,
                thought_count=10,
                branch_count=-1,
            )

    def test_session_state_datetime_handling(self):
        """Test SessionState datetime field handling."""
        now = datetime.now()
        state = SessionState(
            session_id="test",
            status=SessionStatus.ACTIVE,
            thought_count=5,
            branch_count=2,
            started_at=now,
            updated_at=now,
        )

        assert isinstance(state.started_at, datetime)
        assert isinstance(state.updated_at, datetime)

    def test_session_state_serialization(self):
        """Test SessionState serialization to dict and JSON."""
        state = SessionState(
            session_id="serialize-test",
            status=SessionStatus.PAUSED,
            thought_count=25,
            branch_count=4,
            current_method=MethodIdentifier.SELF_CONSISTENCY,
        )

        # To dict
        data = state.model_dump()
        assert data["session_id"] == "serialize-test"
        assert data["status"] == SessionStatus.PAUSED

        # From dict
        restored = SessionState(**data)
        assert restored.session_id == state.session_id


# ============================================================================
# TestBranchOutput - Output Model (Frozen)
# ============================================================================


class TestBranchOutput:
    """Test suite for BranchOutput model (frozen output)."""

    def test_create_branch_output_success(self):
        """Test creating successful BranchOutput."""
        output = BranchOutput(
            branch_id="branch-abc",
            parent_thought_id="thought-123",
            session_id="session-456",
            success=True,
        )

        assert output.branch_id == "branch-abc"
        assert output.parent_thought_id == "thought-123"
        assert output.session_id == "session-456"
        assert output.success is True

    def test_create_branch_output_failure(self):
        """Test creating failed BranchOutput."""
        output = BranchOutput(
            branch_id="",
            parent_thought_id="thought-999",
            session_id="session-456",
            success=False,
        )

        assert output.branch_id == ""
        assert output.success is False

    def test_branch_output_is_frozen(self):
        """Test that BranchOutput is frozen (immutable)."""
        output = BranchOutput(
            branch_id="branch-123",
            parent_thought_id="thought-456",
            session_id="session-789",
            success=True,
        )

        with pytest.raises(ValidationError):
            output.success = False  # type: ignore[misc]

    def test_branch_output_validation(self):
        """Test BranchOutput field validation."""
        # Valid creation
        output = BranchOutput(
            branch_id="test",
            parent_thought_id="parent",
            session_id="session",
            success=True,
        )
        assert output.success is True

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            BranchOutput()  # type: ignore[call-arg]

    def test_branch_output_serialization(self):
        """Test BranchOutput serialization to dict and JSON."""
        output = BranchOutput(
            branch_id="serialize-test",
            parent_thought_id="parent-123",
            session_id="session-456",
            success=True,
        )

        # To dict
        data = output.model_dump()
        assert data["branch_id"] == "serialize-test"
        assert data["success"] is True

        # From dict
        restored = BranchOutput(**data)
        assert restored.branch_id == output.branch_id


# ============================================================================
# TestMergeOutput - Output Model (Frozen)
# ============================================================================


class TestMergeOutput:
    """Test suite for MergeOutput model (frozen output)."""

    def test_create_merge_output_success(self):
        """Test creating successful MergeOutput."""
        output = MergeOutput(
            merged_thought_id="thought-merged-123",
            source_branch_ids=["branch-abc", "branch-def", "branch-ghi"],
            session_id="session-456",
            success=True,
        )

        assert output.merged_thought_id == "thought-merged-123"
        assert len(output.source_branch_ids) == 3
        assert "branch-abc" in output.source_branch_ids
        assert output.success is True

    def test_create_merge_output_failure(self):
        """Test creating failed MergeOutput."""
        output = MergeOutput(
            merged_thought_id="",
            source_branch_ids=["branch-abc", "branch-def"],
            session_id="session-456",
            success=False,
        )

        assert output.merged_thought_id == ""
        assert output.success is False

    def test_merge_output_is_frozen(self):
        """Test that MergeOutput is frozen (immutable)."""
        output = MergeOutput(
            merged_thought_id="merged",
            source_branch_ids=["branch-1"],
            session_id="session",
            success=True,
        )

        with pytest.raises(ValidationError):
            output.success = False  # type: ignore[misc]

    def test_merge_output_validation(self):
        """Test MergeOutput field validation."""
        # Valid creation
        output = MergeOutput(
            merged_thought_id="test",
            source_branch_ids=["b1", "b2"],
            session_id="session",
            success=True,
        )
        assert len(output.source_branch_ids) == 2

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            MergeOutput()  # type: ignore[call-arg]

    def test_merge_output_serialization(self):
        """Test MergeOutput serialization to dict and JSON."""
        output = MergeOutput(
            merged_thought_id="serialize-test",
            source_branch_ids=["branch-1", "branch-2"],
            session_id="session-test",
            success=True,
        )

        # To dict
        data = output.model_dump()
        assert data["merged_thought_id"] == "serialize-test"
        assert len(data["source_branch_ids"]) == 2

        # From dict
        restored = MergeOutput(**data)
        assert restored.merged_thought_id == output.merged_thought_id


# ============================================================================
# TestMethodInfo - Output Model (Frozen)
# ============================================================================


class TestMethodInfo:
    """Test suite for MethodInfo model (frozen output)."""

    def test_create_method_info_minimal(self):
        """Test creating MethodInfo with minimal required parameters."""
        info = MethodInfo(
            id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Classic step-by-step reasoning",
            category=MethodCategory.CORE,
        )

        assert info.id == MethodIdentifier.CHAIN_OF_THOUGHT
        assert info.name == "Chain of Thought"
        assert info.category == MethodCategory.CORE
        assert info.parameters == {}
        assert info.tags == []

    def test_create_method_info_full(self):
        """Test creating MethodInfo with all parameters."""
        info = MethodInfo(
            id=MethodIdentifier.ETHICAL_REASONING,
            name="Ethical Reasoning",
            description="Structured ethical analysis with principles",
            category=MethodCategory.HIGH_VALUE,
            parameters={
                "frameworks": {"type": "list", "default": ["utilitarian", "deontological"]},
                "stakeholder_analysis": {"type": "bool", "default": True},
            },
            tags=["ethical", "stakeholders", "principles", "structured"],
        )

        assert info.id == MethodIdentifier.ETHICAL_REASONING
        assert info.category == MethodCategory.HIGH_VALUE
        assert len(info.parameters) == 2
        assert len(info.tags) == 4
        assert "ethical" in info.tags

    def test_method_info_is_frozen(self):
        """Test that MethodInfo is frozen (immutable)."""
        info = MethodInfo(
            id=MethodIdentifier.REACT,
            name="ReAct",
            description="Test",
            category=MethodCategory.CORE,
        )

        with pytest.raises(ValidationError):
            info.name = "modified"  # type: ignore[misc]

    def test_method_info_validation(self):
        """Test MethodInfo field validation."""
        # Valid creation
        info = MethodInfo(
            id=MethodIdentifier.TREE_OF_THOUGHTS,
            name="Tree of Thoughts",
            description="Test",
            category=MethodCategory.CORE,
        )
        assert info.id == MethodIdentifier.TREE_OF_THOUGHTS

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            MethodInfo()  # type: ignore[call-arg]

    def test_method_info_serialization(self):
        """Test MethodInfo serialization to dict and JSON."""
        info = MethodInfo(
            id=MethodIdentifier.CODE_REASONING,
            name="Code Reasoning",
            description="Specialized code analysis",
            category=MethodCategory.HIGH_VALUE,
            parameters={"language": {"type": "str", "default": "python"}},
            tags=["code", "analysis"],
        )

        # To dict
        data = info.model_dump()
        assert data["name"] == "Code Reasoning"
        assert len(data["tags"]) == 2

        # From dict
        restored = MethodInfo(**data)
        assert restored.id == info.id


# ============================================================================
# TestRecommendation - Output Model (Frozen)
# ============================================================================


class TestRecommendation:
    """Test suite for Recommendation model (frozen output)."""

    def test_create_recommendation_minimal(self):
        """Test creating Recommendation with minimal required parameters."""
        rec = Recommendation(
            method_id=MethodIdentifier.ETHICAL_REASONING,
            score=0.95,
            reason="Problem involves ethical dilemmas",
            confidence=0.92,
        )

        assert rec.method_id == MethodIdentifier.ETHICAL_REASONING
        assert rec.score == 0.95
        assert rec.confidence == 0.92
        assert "ethical dilemmas" in rec.reason

    def test_recommendation_is_frozen(self):
        """Test that Recommendation is frozen (immutable)."""
        rec = Recommendation(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            score=0.7,
            reason="Test",
            confidence=0.8,
        )

        with pytest.raises(ValidationError):
            rec.score = 0.5  # type: ignore[misc]

    def test_recommendation_score_validation(self):
        """Test Recommendation score field validation (0.0-1.0)."""
        # Valid: boundary values
        rec_0 = Recommendation(
            method_id=MethodIdentifier.REACT,
            score=0.0,
            reason="Test",
            confidence=0.5,
        )
        assert rec_0.score == 0.0

        rec_1 = Recommendation(
            method_id=MethodIdentifier.REACT,
            score=1.0,
            reason="Test",
            confidence=0.5,
        )
        assert rec_1.score == 1.0

        # Invalid: below 0
        with pytest.raises(ValidationError):
            Recommendation(
                method_id=MethodIdentifier.REACT,
                score=-0.1,
                reason="Test",
                confidence=0.5,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            Recommendation(
                method_id=MethodIdentifier.REACT,
                score=1.1,
                reason="Test",
                confidence=0.5,
            )

    def test_recommendation_confidence_validation(self):
        """Test Recommendation confidence field validation (0.0-1.0)."""
        # Valid: boundary values
        rec_0 = Recommendation(
            method_id=MethodIdentifier.REACT,
            score=0.5,
            reason="Test",
            confidence=0.0,
        )
        assert rec_0.confidence == 0.0

        rec_1 = Recommendation(
            method_id=MethodIdentifier.REACT,
            score=0.5,
            reason="Test",
            confidence=1.0,
        )
        assert rec_1.confidence == 1.0

        # Invalid: below 0
        with pytest.raises(ValidationError):
            Recommendation(
                method_id=MethodIdentifier.REACT,
                score=0.5,
                reason="Test",
                confidence=-0.1,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            Recommendation(
                method_id=MethodIdentifier.REACT,
                score=0.5,
                reason="Test",
                confidence=1.1,
            )

    def test_recommendation_serialization(self):
        """Test Recommendation serialization to dict and JSON."""
        rec = Recommendation(
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            score=0.88,
            reason="Multiple reasoning paths for validation",
            confidence=0.85,
        )

        # To dict
        data = rec.model_dump()
        assert data["score"] == 0.88
        assert data["confidence"] == 0.85

        # From dict
        restored = Recommendation(**data)
        assert restored.method_id == rec.method_id


# ============================================================================
# TestComparisonResult - Output Model (Frozen)
# ============================================================================


class TestComparisonResult:
    """Test suite for ComparisonResult model (frozen output)."""

    def test_create_comparison_result_with_winner(self):
        """Test creating ComparisonResult with a winner."""
        result = ComparisonResult(
            methods=[
                MethodIdentifier.CHAIN_OF_THOUGHT,
                MethodIdentifier.TREE_OF_THOUGHTS,
                MethodIdentifier.SELF_CONSISTENCY,
            ],
            winner=MethodIdentifier.TREE_OF_THOUGHTS,
            scores={
                "chain_of_thought": 0.75,
                "tree_of_thoughts": 0.92,
                "self_consistency": 0.88,
            },
            analysis="Tree of Thoughts is most suitable for this exploratory problem",
        )

        assert len(result.methods) == 3
        assert result.winner == MethodIdentifier.TREE_OF_THOUGHTS
        assert result.scores["tree_of_thoughts"] == 0.92

    def test_create_comparison_result_no_winner(self):
        """Test creating ComparisonResult with no winner (tie)."""
        result = ComparisonResult(
            methods=[MethodIdentifier.ETHICAL_REASONING, MethodIdentifier.DIALECTIC],
            winner=None,
            scores={"ethical_reasoning": 0.85, "dialectic": 0.85},
            analysis="Both methods scored equally",
        )

        assert result.winner is None
        assert result.scores["ethical_reasoning"] == result.scores["dialectic"]

    def test_comparison_result_is_frozen(self):
        """Test that ComparisonResult is frozen (immutable)."""
        result = ComparisonResult(
            methods=[MethodIdentifier.REACT],
            analysis="Test",
        )

        with pytest.raises(ValidationError):
            result.winner = MethodIdentifier.CHAIN_OF_THOUGHT  # type: ignore[misc]

    def test_comparison_result_validation(self):
        """Test ComparisonResult field validation."""
        # Valid creation
        result = ComparisonResult(
            methods=[MethodIdentifier.SEQUENTIAL_THINKING],
            analysis="Test analysis",
        )
        assert len(result.methods) == 1

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            ComparisonResult()  # type: ignore[call-arg]

    def test_comparison_result_serialization(self):
        """Test ComparisonResult serialization to dict and JSON."""
        result = ComparisonResult(
            methods=[MethodIdentifier.CHAIN_OF_THOUGHT, MethodIdentifier.REACT],
            winner=MethodIdentifier.CHAIN_OF_THOUGHT,
            scores={"chain_of_thought": 0.9, "react": 0.7},
            analysis="Chain of Thought performed better",
        )

        # To dict
        data = result.model_dump()
        assert len(data["methods"]) == 2
        assert data["winner"] == MethodIdentifier.CHAIN_OF_THOUGHT

        # From dict
        restored = ComparisonResult(**data)
        assert restored.winner == result.winner


# ============================================================================
# TestEvaluationReport - Output Model (Frozen)
# ============================================================================


class TestEvaluationReport:
    """Test suite for EvaluationReport model (frozen output)."""

    def test_create_evaluation_report_minimal(self):
        """Test creating EvaluationReport with minimal required parameters."""
        report = EvaluationReport(
            session_id="session-123",
            overall_score=0.89,
            coherence_score=0.92,
            depth_score=0.85,
            coverage_score=0.90,
        )

        assert report.session_id == "session-123"
        assert report.overall_score == 0.89
        assert report.coherence_score == 0.92
        assert report.depth_score == 0.85
        assert report.coverage_score == 0.90
        assert report.insights == []
        assert report.recommendations == []

    def test_create_evaluation_report_full(self):
        """Test creating EvaluationReport with all parameters."""
        report = EvaluationReport(
            session_id="session-456",
            overall_score=0.45,
            coherence_score=0.40,
            depth_score=0.50,
            coverage_score=0.45,
            insights=[
                "Reasoning path lacks coherent structure",
                "Insufficient depth in critical areas",
            ],
            recommendations=[
                "Use more structured reasoning methods",
                "Increase depth of analysis for key decision points",
            ],
        )

        assert len(report.insights) == 2
        assert len(report.recommendations) == 2

    def test_evaluation_report_is_frozen(self):
        """Test that EvaluationReport is frozen (immutable)."""
        report = EvaluationReport(
            session_id="session-123",
            overall_score=0.5,
            coherence_score=0.5,
            depth_score=0.5,
            coverage_score=0.5,
        )

        with pytest.raises(ValidationError):
            report.overall_score = 0.9  # type: ignore[misc]

    def test_evaluation_report_overall_score_validation(self):
        """Test EvaluationReport overall_score validation (0.0-1.0)."""
        # Valid: boundary values
        report_0 = EvaluationReport(
            session_id="test",
            overall_score=0.0,
            coherence_score=0.0,
            depth_score=0.0,
            coverage_score=0.0,
        )
        assert report_0.overall_score == 0.0

        report_1 = EvaluationReport(
            session_id="test",
            overall_score=1.0,
            coherence_score=1.0,
            depth_score=1.0,
            coverage_score=1.0,
        )
        assert report_1.overall_score == 1.0

        # Invalid: below 0
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=-0.1,
                coherence_score=0.5,
                depth_score=0.5,
                coverage_score=0.5,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=1.1,
                coherence_score=0.5,
                depth_score=0.5,
                coverage_score=0.5,
            )

    def test_evaluation_report_coherence_score_validation(self):
        """Test EvaluationReport coherence_score validation (0.0-1.0)."""
        # Valid
        report = EvaluationReport(
            session_id="test",
            overall_score=0.5,
            coherence_score=0.5,
            depth_score=0.5,
            coverage_score=0.5,
        )
        assert report.coherence_score == 0.5

        # Invalid: below 0
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=-0.1,
                depth_score=0.5,
                coverage_score=0.5,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=1.1,
                depth_score=0.5,
                coverage_score=0.5,
            )

    def test_evaluation_report_depth_score_validation(self):
        """Test EvaluationReport depth_score validation (0.0-1.0)."""
        # Valid
        report = EvaluationReport(
            session_id="test",
            overall_score=0.5,
            coherence_score=0.5,
            depth_score=0.5,
            coverage_score=0.5,
        )
        assert report.depth_score == 0.5

        # Invalid: below 0
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=0.5,
                depth_score=-0.1,
                coverage_score=0.5,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=0.5,
                depth_score=1.1,
                coverage_score=0.5,
            )

    def test_evaluation_report_coverage_score_validation(self):
        """Test EvaluationReport coverage_score validation (0.0-1.0)."""
        # Valid
        report = EvaluationReport(
            session_id="test",
            overall_score=0.5,
            coherence_score=0.5,
            depth_score=0.5,
            coverage_score=0.5,
        )
        assert report.coverage_score == 0.5

        # Invalid: below 0
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=0.5,
                depth_score=0.5,
                coverage_score=-0.1,
            )

        # Invalid: above 1
        with pytest.raises(ValidationError):
            EvaluationReport(
                session_id="test",
                overall_score=0.5,
                coherence_score=0.5,
                depth_score=0.5,
                coverage_score=1.1,
            )

    def test_evaluation_report_serialization(self):
        """Test EvaluationReport serialization to dict and JSON."""
        report = EvaluationReport(
            session_id="serialize-test",
            overall_score=0.75,
            coherence_score=0.80,
            depth_score=0.70,
            coverage_score=0.75,
            insights=["Good logical flow"],
            recommendations=["Consider edge cases"],
        )

        # To dict
        data = report.model_dump()
        assert data["session_id"] == "serialize-test"
        assert data["overall_score"] == 0.75

        # From dict
        restored = EvaluationReport(**data)
        assert restored.session_id == report.session_id


# ============================================================================
# Integration Tests
# ============================================================================


class TestToolsIntegration:
    """Integration tests for tool I/O models."""

    def test_reason_output_with_hints_and_suggestions(self, sample_thought_node: ThoughtNode):
        """Test integration between ReasonHints, ReasonOutput, and SuggestionOutput."""
        # Create hints for input
        hints = ReasonHints(
            domain="code",
            complexity="high",
            prefer_methods=[MethodIdentifier.CODE_REASONING],
        )

        # Create output from reasoning
        output = ReasonOutput(
            session_id="session-123",
            thought=sample_thought_node,
            method_used=MethodIdentifier.CODE_REASONING,
            suggestions=["Continue analysis", "Add tests"],
        )

        # Verify integration
        assert hints.domain == "code"
        assert output.method_used in hints.prefer_methods
        assert len(output.suggestions) == 2

    def test_session_state_with_method_info(self):
        """Test integration between SessionState and MethodInfo."""
        # Create session state
        state = SessionState(
            session_id="session-123",
            status=SessionStatus.ACTIVE,
            thought_count=42,
            branch_count=3,
            current_method=MethodIdentifier.TREE_OF_THOUGHTS,
        )

        # Create method info for current method
        info = MethodInfo(
            id=MethodIdentifier.TREE_OF_THOUGHTS,
            name="Tree of Thoughts",
            description="Explore multiple reasoning paths",
            category=MethodCategory.CORE,
        )

        # Verify integration
        assert state.current_method == info.id

    def test_compose_output_with_validation(
        self, sample_thought_node: ThoughtNode, sample_pipeline_trace: PipelineTrace
    ):
        """Test integration between ComposeOutput and ValidationOutput."""
        # Create compose output
        compose = ComposeOutput(
            session_id="session-123",
            pipeline_id="pipeline-abc",
            success=True,
            final_thoughts=[sample_thought_node],
            trace=sample_pipeline_trace,
        )

        # Create validation for the output
        validation = ValidationOutput(
            valid=True,
            warnings=["Pipeline took longer than expected"],
        )

        # Verify integration
        assert compose.success is True
        assert validation.valid is True

    def test_comparison_result_with_recommendations(self):
        """Test integration between ComparisonResult and Recommendation."""
        # Create recommendations
        rec1 = Recommendation(
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            score=0.92,
            reason="Best for exploratory problems",
            confidence=0.90,
        )
        rec2 = Recommendation(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            score=0.75,
            reason="Good general-purpose method",
            confidence=0.85,
        )

        # Create comparison result
        comparison = ComparisonResult(
            methods=[rec1.method_id, rec2.method_id],
            winner=rec1.method_id,
            scores={
                "tree_of_thoughts": rec1.score,
                "chain_of_thought": rec2.score,
            },
            analysis="Tree of Thoughts scored highest",
        )

        # Verify integration
        assert comparison.winner == rec1.method_id
        assert comparison.scores["tree_of_thoughts"] > comparison.scores["chain_of_thought"]

    def test_evaluation_report_with_session_state(self):
        """Test integration between EvaluationReport and SessionState."""
        # Create session state
        state = SessionState(
            session_id="session-123",
            status=SessionStatus.COMPLETED,
            thought_count=67,
            branch_count=5,
        )

        # Create evaluation report for the session
        report = EvaluationReport(
            session_id=state.session_id,
            overall_score=0.89,
            coherence_score=0.92,
            depth_score=0.85,
            coverage_score=0.90,
            insights=["Strong logical flow", "Excellent exploration"],
        )

        # Verify integration
        assert report.session_id == state.session_id
        assert state.status == SessionStatus.COMPLETED
