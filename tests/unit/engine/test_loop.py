"""Tests for the LoopExecutor.

This module tests iterative loop execution, including condition checking,
accumulation, and termination behavior.
"""

from unittest.mock import Mock, patch

import pytest

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.engine.loop import LoopExecutor
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import Accumulator, Condition, LoopPipeline, MethodStage
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
        variables={"iteration": 0, "quality_score": 0.5},
        thought_ids=[],
    )


@pytest.fixture
def loop_pipeline() -> LoopPipeline:
    """Provide a loop pipeline for testing."""
    return LoopPipeline(
        name="test_loop",
        body=MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="reflect",
        ),
        condition=Condition(
            name="quality_check",
            expression="quality_score < 0.9",
            operator="<",
            threshold=0.9,
            field="quality_score",
        ),
        max_iterations=5,
        accumulator=Accumulator(
            name="results",
            operation="append",
        ),
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestLoopExecutorInit:
    """Test LoopExecutor initialization."""

    def test_basic_init(self, loop_pipeline: LoopPipeline):
        """Test basic initialization."""
        executor = LoopExecutor(pipeline=loop_pipeline)

        assert executor.pipeline is loop_pipeline


# ============================================================================
# Execution Tests
# ============================================================================


class TestLoopExecutorExecution:
    """Test LoopExecutor execution behavior."""

    async def test_execute_single_iteration(
        self, loop_pipeline: LoopPipeline, context: ExecutionContext
    ):
        """Test executing a single iteration loop."""
        # Set quality high to exit after first iteration
        context.variables["quality_score"] = 0.5
        executor = LoopExecutor(pipeline=loop_pipeline)

        iteration_count = 0

        async def mock_execute_stage(stage, ctx):
            nonlocal iteration_count
            iteration_count += 1
            # Simulate quality improvement that exceeds threshold
            return StageResult(
                stage_id="body",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[f"thought-{iteration_count}"],
                output_data={"content": f"iteration-{iteration_count}", "quality_score": 0.95},
            )

        with patch.object(executor, "execute_stage", side_effect=mock_execute_stage):
            result = await executor.execute(context)

        assert result.success is True
        assert result.stage_type == PipelineStageType.LOOP


# ============================================================================
# Validation Tests
# ============================================================================


class TestLoopExecutorValidation:
    """Test LoopExecutor validation."""

    async def test_validate_valid_loop(self, loop_pipeline: LoopPipeline):
        """Test validating a valid loop pipeline."""
        executor = LoopExecutor(pipeline=loop_pipeline)

        errors = await executor.validate(loop_pipeline)

        assert errors == []

    async def test_validate_wrong_type(self, loop_pipeline: LoopPipeline):
        """Test validation fails for wrong pipeline type."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single",
        )
        executor = LoopExecutor(pipeline=loop_pipeline)

        errors = await executor.validate(stage)  # Wrong type

        assert len(errors) == 1
        assert "Expected LoopPipeline" in errors[0]
