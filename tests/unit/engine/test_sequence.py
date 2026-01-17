"""Tests for the SequenceExecutor.

This module tests sequential pipeline execution, including stage chaining,
error handling, and data transformation.
"""

from unittest.mock import Mock, patch

import pytest

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.engine.sequence import SequenceExecutor
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import MethodStage, SequencePipeline, Transform
from reasoning_mcp.models.session import Session
from reasoning_mcp.registry import MethodRegistry

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def session() -> Session:
    """Provide a started session for testing."""
    return Session().start()


@pytest.fixture
def registry() -> MethodRegistry:
    """Provide a mock registry for testing."""
    return Mock(spec=MethodRegistry)


@pytest.fixture
def context(session: Session, registry: MethodRegistry) -> ExecutionContext:
    """Provide an ExecutionContext for testing."""
    return ExecutionContext(
        session=session,
        registry=registry,
        input_data={"input": "test query"},
        variables={},
        thought_ids=[],
    )


@pytest.fixture
def simple_sequence() -> SequencePipeline:
    """Provide a simple two-stage sequence pipeline."""
    return SequencePipeline(
        name="test_sequence",
        stages=[
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="stage_1",
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="stage_2",
            ),
        ],
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSequenceExecutorInit:
    """Test SequenceExecutor initialization."""

    def test_basic_init(self, simple_sequence: SequencePipeline):
        """Test basic initialization."""
        executor = SequenceExecutor(pipeline=simple_sequence)

        assert executor.pipeline is simple_sequence
        assert executor.stop_on_error is True
        assert executor.collect_all_outputs is False

    def test_init_with_options(self, simple_sequence: SequencePipeline):
        """Test initialization with custom options."""
        executor = SequenceExecutor(
            pipeline=simple_sequence,
            stop_on_error=False,
            collect_all_outputs=True,
        )

        assert executor.stop_on_error is False
        assert executor.collect_all_outputs is True


# ============================================================================
# Execution Tests
# ============================================================================


class TestSequenceExecutorExecution:
    """Test SequenceExecutor execution behavior."""

    async def test_execute_empty_sequence(self, context: ExecutionContext):
        """Test executing an empty sequence."""
        pipeline = SequencePipeline(name="empty", stages=[])
        executor = SequenceExecutor(pipeline=pipeline)

        result = await executor.execute(context)

        assert result.success is True
        assert result.stage_type == PipelineStageType.SEQUENCE

    async def test_execute_single_stage(self, context: ExecutionContext):
        """Test executing a single-stage sequence."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single_stage",
        )
        pipeline = SequencePipeline(name="single", stages=[stage])
        executor = SequenceExecutor(pipeline=pipeline)

        # Mock execute_stage to return success
        mock_result = StageResult(
            stage_id=stage.id,
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=["thought-1"],
            output_data={"content": "test output"},
        )

        with patch.object(executor, "execute_stage", return_value=mock_result):
            result = await executor.execute(context)

        assert result.success is True
        assert result.stage_type == PipelineStageType.SEQUENCE

    async def test_execute_multi_stage_sequence(
        self, simple_sequence: SequencePipeline, context: ExecutionContext
    ):
        """Test executing a multi-stage sequence."""
        executor = SequenceExecutor(pipeline=simple_sequence)

        # Mock execute_stage to return success for both stages
        mock_results = [
            StageResult(
                stage_id=f"stage-{i}",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[f"thought-{i}"],
                output_data={"content": f"output-{i}"},
            )
            for i in range(2)
        ]

        call_count = 0

        async def mock_execute_stage(stage, ctx):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result

        with patch.object(executor, "execute_stage", side_effect=mock_execute_stage):
            result = await executor.execute(context)

        assert result.success is True
        assert result.metadata["stages_executed"] == 2

    async def test_execute_stop_on_error_true(self, context: ExecutionContext):
        """Test that execution stops on first error when stop_on_error=True."""
        pipeline = SequencePipeline(
            name="failing",
            stages=[
                MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s1"),
                MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="s2"),
                MethodStage(method_id=MethodIdentifier.DIALECTIC, name="s3"),
            ],
        )
        executor = SequenceExecutor(pipeline=pipeline, stop_on_error=True)

        # First stage succeeds, second fails
        mock_results = [
            StageResult(
                stage_id="s1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=["t1"],
            ),
            StageResult(
                stage_id="s2",
                stage_type=PipelineStageType.METHOD,
                success=False,
                error="Stage failed",
            ),
        ]

        call_count = 0

        async def mock_execute_stage(stage, ctx):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result

        with patch.object(executor, "execute_stage", side_effect=mock_execute_stage):
            result = await executor.execute(context)

        assert result.success is False
        assert call_count == 2  # Should stop after second stage fails

    async def test_execute_stop_on_error_false(self, context: ExecutionContext):
        """Test that execution continues on error when stop_on_error=False."""
        pipeline = SequencePipeline(
            name="continuing",
            stages=[
                MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s1"),
                MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="s2"),
                MethodStage(method_id=MethodIdentifier.DIALECTIC, name="s3"),
            ],
        )
        executor = SequenceExecutor(pipeline=pipeline, stop_on_error=False)

        # Second stage fails, but we continue
        mock_results = [
            StageResult(
                stage_id="s1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=["t1"],
            ),
            StageResult(
                stage_id="s2",
                stage_type=PipelineStageType.METHOD,
                success=False,
                error="Stage failed",
            ),
            StageResult(
                stage_id="s3",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=["t3"],
            ),
        ]

        call_count = 0

        async def mock_execute_stage(stage, ctx):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result

        with patch.object(executor, "execute_stage", side_effect=mock_execute_stage):
            result = await executor.execute(context)

        # All stages were attempted
        assert call_count == 3
        assert result.metadata["total_errors"] == 1


# ============================================================================
# Transform Tests
# ============================================================================


class TestSequenceExecutorTransform:
    """Test SequenceExecutor transformation functionality."""

    def test_apply_transform_simple(self, simple_sequence: SequencePipeline):
        """Test simple template substitution transform."""
        executor = SequenceExecutor(pipeline=simple_sequence)
        transform = Transform(
            name="extract",
            expression="{content}",
            input_fields=["content"],
            output_field="summary",
        )
        data = {"content": "Test content", "other": "value"}

        result = executor.apply_transform(transform, data)

        assert result["summary"] == "Test content"
        assert result["content"] == "Test content"  # Original preserved
        assert result["other"] == "value"

    def test_apply_transform_multiple_fields(self, simple_sequence: SequencePipeline):
        """Test transform with multiple input fields."""
        executor = SequenceExecutor(pipeline=simple_sequence)
        transform = Transform(
            name="combine",
            expression="{title}: {content}",
            input_fields=["title", "content"],
            output_field="combined",
        )
        data = {"title": "Test", "content": "Content here"}

        result = executor.apply_transform(transform, data)

        assert result["combined"] == "Test: Content here"

    def test_apply_transform_missing_field(self, simple_sequence: SequencePipeline):
        """Test transform gracefully handles missing fields."""
        executor = SequenceExecutor(pipeline=simple_sequence)
        transform = Transform(
            name="extract",
            expression="{missing}",
            input_fields=["missing"],
            output_field="result",
        )
        data = {"content": "value"}

        result = executor.apply_transform(transform, data)

        # Should not crash, expression unchanged without substitution
        assert "result" in result
        assert result["result"] == "{missing}"


# ============================================================================
# Validation Tests
# ============================================================================


class TestSequenceExecutorValidation:
    """Test SequenceExecutor validation."""

    async def test_validate_valid_sequence(self, simple_sequence: SequencePipeline):
        """Test validating a valid sequence pipeline."""
        executor = SequenceExecutor(pipeline=simple_sequence)

        errors = await executor.validate(simple_sequence)

        assert errors == []

    async def test_validate_wrong_type(self):
        """Test validation fails for wrong pipeline type."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single",
        )
        pipeline = SequencePipeline(name="test", stages=[stage])
        executor = SequenceExecutor(pipeline=pipeline)

        errors = await executor.validate(stage)  # Wrong type

        assert len(errors) == 1
        assert "Expected SequencePipeline" in errors[0]

    async def test_validate_empty_sequence(self):
        """Test validation fails for empty sequence."""
        pipeline = SequencePipeline(name="empty", stages=[])
        executor = SequenceExecutor(pipeline=pipeline)

        errors = await executor.validate(pipeline)

        assert len(errors) == 1
        assert "at least one stage" in errors[0]
