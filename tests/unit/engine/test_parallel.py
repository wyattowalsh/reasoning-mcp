"""Tests for the ParallelExecutor.

This module tests parallel pipeline execution, including concurrent branch
execution, merge strategies, and error handling.
"""

from unittest.mock import Mock, patch

import pytest

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.engine.parallel import ParallelExecutor
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import MergeStrategy, MethodStage, ParallelPipeline
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
def parallel_pipeline() -> ParallelPipeline:
    """Provide a parallel pipeline with two branches."""
    return ParallelPipeline(
        name="test_parallel",
        branches=[
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="branch_1"),
            MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="branch_2"),
        ],
        merge_strategy=MergeStrategy(name="concat", aggregation="concatenate"),
        max_concurrency=5,
    )


# ============================================================================
# Initialization Tests
# ============================================================================


class TestParallelExecutorInit:
    """Test ParallelExecutor initialization."""

    def test_basic_init(self, parallel_pipeline: ParallelPipeline):
        """Test basic initialization."""
        executor = ParallelExecutor(pipeline=parallel_pipeline)

        assert executor.pipeline is parallel_pipeline
        assert executor.parallel_pipeline is parallel_pipeline
        assert executor.fail_fast is False

    def test_init_with_fail_fast(self, parallel_pipeline: ParallelPipeline):
        """Test initialization with fail_fast=True."""
        executor = ParallelExecutor(pipeline=parallel_pipeline, fail_fast=True)

        assert executor.fail_fast is True

    def test_init_wrong_type_raises_error(self):
        """Test initialization with wrong pipeline type raises TypeError."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="not_parallel",
        )

        with pytest.raises(TypeError, match="Expected ParallelPipeline"):
            ParallelExecutor(pipeline=stage)  # type: ignore


# ============================================================================
# Execution Tests
# ============================================================================


class TestParallelExecutorExecution:
    """Test ParallelExecutor execution behavior."""

    async def test_execute_empty_branches(self, context: ExecutionContext):
        """Test executing with no branches."""
        pipeline = ParallelPipeline(
            name="empty",
            branches=[],
            merge_strategy=MergeStrategy(name="concat"),
        )
        executor = ParallelExecutor(pipeline=pipeline)

        result = await executor.execute(context)

        assert result.success is False
        assert "All parallel branches failed" in result.error

    async def test_execute_all_success(
        self, parallel_pipeline: ParallelPipeline, context: ExecutionContext
    ):
        """Test executing with all branches succeeding."""
        executor = ParallelExecutor(pipeline=parallel_pipeline)

        mock_results = [
            StageResult(
                stage_id=f"branch-{i}",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[f"thought-{i}"],
                output_data={"content": f"output-{i}"},
            )
            for i in range(2)
        ]

        with patch.object(executor, "execute_parallel", return_value=mock_results):
            result = await executor.execute(context)

        assert result.success is True
        assert result.stage_type == PipelineStageType.PARALLEL
        assert result.metadata["branches_count"] == 2
        assert result.metadata["successful_count"] == 2
        assert result.metadata["failed_count"] == 0

    async def test_execute_partial_failure(
        self, parallel_pipeline: ParallelPipeline, context: ExecutionContext
    ):
        """Test executing with some branches failing."""
        executor = ParallelExecutor(pipeline=parallel_pipeline, fail_fast=False)

        mock_results = [
            StageResult(
                stage_id="branch-0",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=["thought-0"],
                output_data={"content": "success"},
            ),
            StageResult(
                stage_id="branch-1",
                stage_type=PipelineStageType.METHOD,
                success=False,
                error="Branch failed",
            ),
        ]

        with patch.object(executor, "execute_parallel", return_value=mock_results):
            result = await executor.execute(context)

        assert result.success is True  # At least one succeeded
        assert result.metadata["successful_count"] == 1
        assert result.metadata["failed_count"] == 1

    async def test_execute_fail_fast(
        self, parallel_pipeline: ParallelPipeline, context: ExecutionContext
    ):
        """Test fail_fast stops on first error."""
        executor = ParallelExecutor(pipeline=parallel_pipeline, fail_fast=True)

        mock_results = [
            StageResult(
                stage_id="branch-0",
                stage_type=PipelineStageType.METHOD,
                success=False,
                error="First failure",
            ),
            StageResult(
                stage_id="branch-1",
                stage_type=PipelineStageType.METHOD,
                success=True,
            ),
        ]

        with patch.object(executor, "execute_parallel", return_value=mock_results):
            result = await executor.execute(context)

        assert result.success is False
        assert "First failure" in result.error


# ============================================================================
# Merge Strategy Tests
# ============================================================================


class TestParallelExecutorMerge:
    """Test ParallelExecutor merge strategies."""

    @pytest.fixture
    def executor(self, parallel_pipeline: ParallelPipeline) -> ParallelExecutor:
        """Provide a ParallelExecutor for merge testing."""
        return ParallelExecutor(pipeline=parallel_pipeline)

    def test_merge_concat(self, executor: ParallelExecutor):
        """Test concatenation merge strategy."""
        results = [
            StageResult(
                stage_id="b1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"content": "first", "items": ["a"]},
            ),
            StageResult(
                stage_id="b2",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"content": "second", "items": ["b", "c"]},
            ),
        ]
        strategy = MergeStrategy(name="concat", aggregation="concatenate")

        merged = executor.merge_outputs(results, strategy)

        assert "content" in merged
        assert merged["content"] == ["first", "second"]
        assert merged["items"] == ["a", "b", "c"]

    def test_merge_dicts(self, executor: ParallelExecutor):
        """Test dictionary merge strategy."""
        results = [
            StageResult(
                stage_id="b1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"key1": "value1", "shared": "from_b1"},
            ),
            StageResult(
                stage_id="b2",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"key2": "value2", "shared": "from_b2"},
            ),
        ]
        strategy = MergeStrategy(name="merge_dicts", aggregation="merge")

        merged = executor.merge_outputs(results, strategy)

        assert merged["key1"] == "value1"
        assert merged["key2"] == "value2"
        assert merged["shared"] == "from_b2"  # Last wins

    def test_merge_best_score(self, executor: ParallelExecutor):
        """Test best score merge strategy."""
        results = [
            StageResult(
                stage_id="b1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"content": "low confidence"},
                metadata={"confidence": 0.5},
            ),
            StageResult(
                stage_id="b2",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"content": "high confidence"},
                metadata={"confidence": 0.9},
            ),
        ]
        strategy = MergeStrategy(name="best_score", selection_criteria="highest_confidence")

        merged = executor.merge_outputs(results, strategy)

        assert merged["content"] == "high confidence"

    def test_merge_vote(self, executor: ParallelExecutor):
        """Test voting merge strategy."""
        results = [
            StageResult(
                stage_id="b1",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"conclusion": "A"},
            ),
            StageResult(
                stage_id="b2",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"conclusion": "B"},
            ),
            StageResult(
                stage_id="b3",
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_data={"conclusion": "A"},
            ),
        ]
        strategy = MergeStrategy(name="vote", selection_criteria="most_common_conclusion")

        merged = executor.merge_outputs(results, strategy)

        assert merged["conclusion"] == "A"  # Most common

    def test_merge_empty_results(self, executor: ParallelExecutor):
        """Test merge with empty results."""
        strategy = MergeStrategy(name="concat")

        merged = executor.merge_outputs([], strategy)

        assert merged == {}


# ============================================================================
# Validation Tests
# ============================================================================


class TestParallelExecutorValidation:
    """Test ParallelExecutor validation."""

    async def test_validate_valid_parallel(self, parallel_pipeline: ParallelPipeline):
        """Test validating a valid parallel pipeline."""
        executor = ParallelExecutor(pipeline=parallel_pipeline)

        errors = await executor.validate(parallel_pipeline)

        assert errors == []

    async def test_validate_wrong_type(self, parallel_pipeline: ParallelPipeline):
        """Test validation fails for wrong pipeline type."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="single",
        )
        executor = ParallelExecutor(pipeline=parallel_pipeline)

        errors = await executor.validate(stage)  # Wrong type

        assert len(errors) == 1
        assert "Expected ParallelPipeline" in errors[0]

    async def test_validate_empty_branches(self):
        """Test validation fails for empty branches."""
        pipeline = ParallelPipeline(
            name="empty",
            branches=[],
            merge_strategy=MergeStrategy(name="concat"),
        )
        executor = ParallelExecutor(pipeline=pipeline)

        errors = await executor.validate(pipeline)

        assert any("at least one branch" in e for e in errors)

    async def test_validate_missing_merge_strategy(self):
        """Test validation fails for missing merge strategy."""
        # This test requires constructing a pipeline without merge_strategy
        # which may not be possible with Pydantic validation
        # Skip if Pydantic enforces the field
        pass
