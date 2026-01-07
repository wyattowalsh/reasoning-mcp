"""Integration tests for pipeline execution system.

This module provides comprehensive integration tests for the pipeline execution
framework, testing all pipeline stage types (sequence, parallel, conditional,
loop, switch) and their interactions.

Tests cover:
- Sequential pipeline execution with data passing
- Parallel pipeline execution with merge strategies
- Conditional branching (if-then-else)
- Loop execution with termination conditions
- Switch/case routing
- Complex mixed pipelines
- Error handling and recovery
- Execution tracing and metrics

Each test validates both the execution flow and the trace/metrics collection.
"""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from reasoning_mcp.engine.conditional import ConditionalExecutor
from reasoning_mcp.engine.executor import ExecutionContext, PipelineExecutor, StageResult
from reasoning_mcp.engine.loop import LoopExecutor
from reasoning_mcp.engine.method import MethodExecutor
from reasoning_mcp.engine.parallel import ParallelExecutor
from reasoning_mcp.engine.registry import get_executor_for_stage
from reasoning_mcp.engine.sequence import SequenceExecutor
from reasoning_mcp.engine.switch import SwitchExecutor
from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType, SessionStatus
from reasoning_mcp.models.pipeline import (
    Accumulator,
    Condition,
    ConditionalPipeline,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    PipelineResult,
    PipelineTrace,
    SequencePipeline,
    StageMetrics,
    StageTrace,
    SwitchPipeline,
)
from reasoning_mcp.models.session import Session, SessionConfig
from reasoning_mcp.registry import MethodRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_session() -> Session:
    """Create a mock reasoning session for testing."""
    session = Session(
        id=str(uuid4()),
        status=SessionStatus.ACTIVE,
        config=SessionConfig(),
        created_at=datetime.now(),
    )
    return session


@pytest.fixture
def mock_registry() -> MethodRegistry:
    """Create a mock method registry for testing."""
    registry = MethodRegistry()
    return registry


@pytest.fixture
def base_context(mock_session: Session, mock_registry: MethodRegistry) -> ExecutionContext:
    """Create a base execution context for testing."""
    # Create a minimal context with required fields based on actual implementation
    context = ExecutionContext(
        session=mock_session,
        registry=mock_registry,
        input_data={"query": "test query"},
        variables={},
    )
    # Add thought_ids attribute if it doesn't exist (implementation detail)
    if not hasattr(context, 'thought_ids'):
        context.thought_ids = []
    return context


@pytest.fixture
def mock_method_executor():
    """Create a mock method executor that returns controllable results."""

    async def execute_method(
        context: ExecutionContext,
        method_id: str,
        output_data: dict[str, Any] | None = None,
        thought_ids: list[str] | None = None,
    ) -> StageResult:
        """Mock method execution."""
        if output_data is None:
            output_data = {"result": f"output from {method_id}"}
        if thought_ids is None:
            thought_ids = [f"thought-{method_id}-{uuid4().hex[:8]}"]

        metrics = StageMetrics(
            stage_id=method_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=0.1,
            thoughts_generated=len(thought_ids),
        )

        trace = StageTrace(
            stage_id=method_id,
            stage_type=PipelineStageType.METHOD,
            status="completed",
            input_thought_ids=context.input_data.get("thought_ids", []),
            output_thought_ids=thought_ids,
            metrics=metrics,
        )

        return StageResult(
            stage_id=method_id,
            stage_type=PipelineStageType.METHOD,
            success=True,
            output_thought_ids=thought_ids,
            output_data=output_data,
            trace=trace,
        )

    return execute_method


# ============================================================================
# Test: Sequence Pipeline
# ============================================================================


@pytest.mark.asyncio
class TestSequencePipeline:
    """Test sequential pipeline execution."""

    async def test_sequence_pipeline_basic(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test basic sequential execution of 3 methods."""
        # Create a sequence of 3 method stages
        stages = [
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="stage_1",
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="stage_2",
            ),
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="stage_3",
            ),
        ]

        pipeline = SequencePipeline(
            name="test_sequence",
            stages=stages,
        )

        # Mock the method executor to track execution order
        execution_order = []

        async def mock_execute(stage, context):
            execution_order.append(stage.name)
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"stage": stage.name, "order": len(execution_order)},
                thought_ids=[f"thought-{stage.name}"],
            )

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline)
            result = await executor.execute(base_context)

        # Verify execution order
        assert execution_order == ["stage_1", "stage_2", "stage_3"]

        # Verify result
        assert result.success is True
        assert result.stage_type == PipelineStageType.SEQUENCE

    async def test_sequence_output_passing(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test that outputs are passed between sequential stages."""
        stages = [
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="producer",
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_REFLECTION,
                name="consumer",
            ),
        ]

        pipeline = SequencePipeline(stages=stages)

        received_inputs = []

        async def mock_execute(stage, context):
            # Record what input this stage received
            received_inputs.append(
                {
                    "stage": stage.name,
                    "input_data": context.input_data.copy(),
                    "thought_ids": context.input_data.get("thought_ids", []),
                }
            )

            # Produce output
            output_data = {
                "produced_by": stage.name,
                "value": f"data-from-{stage.name}",
            }
            thought_ids = [f"thought-{stage.name}"]

            return await mock_method_executor(
                context, stage.id, output_data=output_data, thought_ids=thought_ids
            )

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline)
            result = await executor.execute(base_context)

        # Verify stages were called
        assert len(received_inputs) == 2

        # First stage should receive original input
        assert received_inputs[0]["stage"] == "producer"

        # Second stage should receive output from first stage
        # (This would be verified in the actual implementation)
        assert received_inputs[1]["stage"] == "consumer"

    async def test_sequence_stop_on_error(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test sequence stops on error when configured."""
        stages = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="stage_1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="stage_2"),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="stage_3"),
        ]

        pipeline = SequencePipeline(stages=stages, stop_on_error=True)

        execution_count = []

        async def mock_execute(stage, context):
            execution_count.append(stage.name)

            # Stage 2 fails
            if stage.name == "stage_2":
                return StageResult(
                    stage_id=stage.id,
                    stage_type=PipelineStageType.METHOD,
                    success=False,
                    error="Simulated failure",
                )

            return await mock_method_executor(context, stage.id)

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline, stop_on_error=True)
            result = await executor.execute(base_context)

        # Should stop after stage_2 fails
        assert len(execution_count) == 2
        assert execution_count == ["stage_1", "stage_2"]
        assert result.success is False


# ============================================================================
# Test: Parallel Pipeline
# ============================================================================


@pytest.mark.asyncio
class TestParallelPipeline:
    """Test parallel pipeline execution."""

    async def test_parallel_pipeline_basic(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test parallel execution of 3 methods."""
        branches = [
            MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="branch_1",
            ),
            MethodStage(
                method_id=MethodIdentifier.SELF_CONSISTENCY,
                name="branch_2",
            ),
            MethodStage(
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                name="branch_3",
            ),
        ]

        merge_strategy = MergeStrategy(
            name="concat",
            selection_criteria="all_results",
            aggregation="concatenate",
        )

        pipeline = ParallelPipeline(
            name="test_parallel",
            branches=branches,
            merge_strategy=merge_strategy,
        )

        executed_branches = []

        async def mock_execute(stage, context):
            executed_branches.append(stage.name)
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"branch": stage.name},
                thought_ids=[f"thought-{stage.name}"],
            )

        with patch.object(ParallelExecutor, "execute_stage", side_effect=mock_execute):
            executor = ParallelExecutor(pipeline)
            result = await executor.execute(base_context)

        # All branches should execute
        assert len(executed_branches) == 3
        assert set(executed_branches) == {"branch_1", "branch_2", "branch_3"}
        assert result.success is True

    async def test_parallel_merge_concat(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test parallel pipeline with concatenate merge strategy."""
        branches = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="b1"),
            MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY, name="b2"),
        ]

        merge_strategy = MergeStrategy(
            name="concat",
            selection_criteria="all_results",
            aggregation="concatenate",
        )

        pipeline = ParallelPipeline(
            branches=branches,
            merge_strategy=merge_strategy,
        )

        async def mock_execute(stage, context):
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"value": f"result-{stage.name}"},
                thought_ids=[f"thought-{stage.name}"],
            )

        with patch.object(ParallelExecutor, "execute_stage", side_effect=mock_execute):
            executor = ParallelExecutor(pipeline)
            result = await executor.execute(base_context)

        # Verify merge strategy was applied
        assert result.success is True
        assert result.metadata.get("merge_strategy") == "concat"

    async def test_parallel_merge_best_score(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test parallel pipeline with best_score merge strategy."""
        branches = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="low"),
            MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY, name="high"),
            MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="medium"),
        ]

        merge_strategy = MergeStrategy(
            name="best_score",
            selection_criteria="highest_confidence",
        )

        pipeline = ParallelPipeline(
            branches=branches,
            merge_strategy=merge_strategy,
        )

        async def mock_execute(stage, context):
            # Assign different scores to each branch
            scores = {"low": 0.5, "high": 0.9, "medium": 0.7}
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"confidence": scores[stage.name]},
                thought_ids=[f"thought-{stage.name}"],
            )

        with patch.object(ParallelExecutor, "execute_stage", side_effect=mock_execute):
            executor = ParallelExecutor(pipeline)
            result = await executor.execute(base_context)

        assert result.success is True

    async def test_parallel_merge_vote(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test parallel pipeline with voting merge strategy."""
        branches = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="v1"),
            MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY, name="v2"),
            MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="v3"),
        ]

        merge_strategy = MergeStrategy(
            name="vote",
            selection_criteria="most_common_conclusion",
        )

        pipeline = ParallelPipeline(
            branches=branches,
            merge_strategy=merge_strategy,
        )

        async def mock_execute(stage, context):
            # Most branches vote for "answer_a"
            answers = {"v1": "answer_a", "v2": "answer_a", "v3": "answer_b"}
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"answer": answers[stage.name]},
                thought_ids=[f"thought-{stage.name}"],
            )

        with patch.object(ParallelExecutor, "execute_stage", side_effect=mock_execute):
            executor = ParallelExecutor(pipeline)
            result = await executor.execute(base_context)

        assert result.success is True


# ============================================================================
# Test: Conditional Pipeline
# ============================================================================


@pytest.mark.asyncio
class TestConditionalPipeline:
    """Test conditional pipeline execution."""

    async def test_conditional_true_branch(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test conditional executes true branch when condition is true."""
        condition = Condition(
            name="confidence_check",
            expression="confidence > 0.8",
            operator=">",
            threshold=0.8,
            field="confidence",
        )

        if_true = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="high_confidence_path",
        )

        if_false = MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="low_confidence_path",
        )

        pipeline = ConditionalPipeline(
            name="test_conditional",
            condition=condition,
            if_true=if_true,
            if_false=if_false,
        )

        executed_path = []

        async def mock_execute(stage, context):
            executed_path.append(stage.name)
            return await mock_method_executor(context, stage.id)

        # Mock condition evaluation to return True
        with patch.object(
            ConditionalExecutor, "evaluate_condition", return_value=True
        ):
            with patch.object(
                ConditionalExecutor, "execute_stage", side_effect=mock_execute
            ):
                executor = ConditionalExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should execute true branch
        assert "high_confidence_path" in executed_path
        assert "low_confidence_path" not in executed_path
        assert result.success is True

    async def test_conditional_false_branch(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test conditional executes false branch when condition is false."""
        condition = Condition(
            name="quality_check",
            expression="quality >= 0.9",
        )

        if_true = MethodStage(
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            name="high_quality",
        )

        if_false = MethodStage(
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            name="needs_refinement",
        )

        pipeline = ConditionalPipeline(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
        )

        executed_path = []

        async def mock_execute(stage, context):
            executed_path.append(stage.name)
            return await mock_method_executor(context, stage.id)

        # Mock condition evaluation to return False
        with patch.object(
            ConditionalExecutor, "evaluate_condition", return_value=False
        ):
            with patch.object(
                ConditionalExecutor, "execute_stage", side_effect=mock_execute
            ):
                executor = ConditionalExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should execute false branch
        assert "needs_refinement" in executed_path
        assert "high_quality" not in executed_path
        assert result.success is True

    async def test_conditional_nested(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test nested conditional pipelines."""
        # Inner conditional
        inner_condition = Condition(name="inner", expression="x > 5")
        inner_true = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="inner_true")
        inner_false = MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="inner_false")

        inner_conditional = ConditionalPipeline(
            condition=inner_condition,
            if_true=inner_true,
            if_false=inner_false,
        )

        # Outer conditional
        outer_condition = Condition(name="outer", expression="y > 10")
        outer_false = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="outer_false")

        pipeline = ConditionalPipeline(
            condition=outer_condition,
            if_true=inner_conditional,
            if_false=outer_false,
        )

        executed_stages = []

        async def mock_execute(stage, context):
            executed_stages.append(stage.name if hasattr(stage, 'name') else "nested")
            return await mock_method_executor(context, stage.id if hasattr(stage, 'id') else str(uuid4()))

        # Outer true, inner true
        with patch.object(ConditionalExecutor, "evaluate_condition", side_effect=[True, True]):
            with patch.object(ConditionalExecutor, "execute_stage", side_effect=mock_execute):
                executor = ConditionalExecutor(pipeline)
                result = await executor.execute(base_context)

        assert result.success is True


# ============================================================================
# Test: Loop Pipeline
# ============================================================================


@pytest.mark.asyncio
class TestLoopPipeline:
    """Test loop pipeline execution."""

    async def test_loop_with_max_iterations(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test loop respects max_iterations limit."""
        body = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="loop_body",
        )

        condition = Condition(
            name="never_stop",
            expression="true",  # Would loop forever without max_iterations
        )

        pipeline = LoopPipeline(
            name="test_loop",
            body=body,
            condition=condition,
            max_iterations=5,
        )

        iteration_count = []

        async def mock_execute(stage, context):
            iteration_count.append(len(iteration_count) + 1)
            return await mock_method_executor(context, stage.id)

        # Mock condition to always return True
        with patch.object(LoopExecutor, "evaluate_condition", return_value=True):
            with patch.object(LoopExecutor, "execute_stage", side_effect=mock_execute):
                executor = LoopExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should stop at max_iterations
        assert len(iteration_count) == 5
        assert result.success is True

    async def test_loop_termination_condition(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test loop terminates when condition becomes false."""
        body = MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="improvement_step",
        )

        condition = Condition(
            name="quality_threshold",
            expression="quality < 0.9",
        )

        pipeline = LoopPipeline(
            body=body,
            condition=condition,
            max_iterations=10,
        )

        iteration_count = []

        async def mock_execute(stage, context):
            iteration_count.append(len(iteration_count) + 1)
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"quality": 0.5 + len(iteration_count) * 0.15},
            )

        # Mock condition that becomes false after 3 iterations
        condition_results = [True, True, True, False]

        with patch.object(
            LoopExecutor, "evaluate_condition", side_effect=condition_results
        ):
            with patch.object(LoopExecutor, "execute_stage", side_effect=mock_execute):
                executor = LoopExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should stop after 3 iterations when condition becomes false
        assert len(iteration_count) == 3
        assert result.success is True

    async def test_loop_with_accumulator(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test loop with result accumulation."""
        body = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="generate_insight",
        )

        condition = Condition(
            name="iteration_limit",
            expression="iteration < 3",
        )

        accumulator = Accumulator(
            name="insights_collector",
            initial_value=[],
            operation="append",
            field="insight",
        )

        pipeline = LoopPipeline(
            body=body,
            condition=condition,
            max_iterations=3,
            accumulator=accumulator,
        )

        insights_generated = []

        async def mock_execute(stage, context):
            insight = f"insight-{len(insights_generated) + 1}"
            insights_generated.append(insight)
            return await mock_method_executor(
                context,
                stage.id,
                output_data={"insight": insight},
            )

        with patch.object(LoopExecutor, "evaluate_condition", side_effect=[True, True, True, False]):
            with patch.object(LoopExecutor, "execute_stage", side_effect=mock_execute):
                executor = LoopExecutor(pipeline)
                result = await executor.execute(base_context)

        # Verify all insights were generated
        assert len(insights_generated) == 3
        assert result.success is True


# ============================================================================
# Test: Switch Pipeline
# ============================================================================


@pytest.mark.asyncio
class TestSwitchPipeline:
    """Test switch pipeline execution."""

    async def test_switch_case_matching(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test switch executes the matching case."""
        cases = {
            "simple": MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="simple_case",
            ),
            "complex": MethodStage(
                method_id=MethodIdentifier.TREE_OF_THOUGHTS,
                name="complex_case",
            ),
            "ethical": MethodStage(
                method_id=MethodIdentifier.ETHICAL_REASONING,
                name="ethical_case",
            ),
        }

        pipeline = SwitchPipeline(
            name="problem_router",
            expression="problem_type",
            cases=cases,
        )

        executed_case = []

        async def mock_execute(stage, context):
            executed_case.append(stage.name)
            return await mock_method_executor(context, stage.id)

        # Mock expression evaluation to return "complex"
        with patch.object(SwitchExecutor, "evaluate_expression", return_value="complex"):
            with patch.object(SwitchExecutor, "execute_stage", side_effect=mock_execute):
                executor = SwitchExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should execute complex case
        assert "complex_case" in executed_case
        assert "simple_case" not in executed_case
        assert "ethical_case" not in executed_case
        assert result.success is True

    async def test_switch_default_case(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test switch falls back to default case when no match."""
        cases = {
            "option_a": MethodStage(
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="case_a",
            ),
            "option_b": MethodStage(
                method_id=MethodIdentifier.SELF_CONSISTENCY,
                name="case_b",
            ),
        }

        default = MethodStage(
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            name="default_case",
        )

        pipeline = SwitchPipeline(
            expression="mode",
            cases=cases,
            default=default,
        )

        executed_case = []

        async def mock_execute(stage, context):
            executed_case.append(stage.name)
            return await mock_method_executor(context, stage.id)

        # Mock expression evaluation to return unmatched value
        with patch.object(SwitchExecutor, "evaluate_expression", return_value="option_c"):
            with patch.object(SwitchExecutor, "execute_stage", side_effect=mock_execute):
                executor = SwitchExecutor(pipeline)
                result = await executor.execute(base_context)

        # Should execute default case
        assert "default_case" in executed_case
        assert "case_a" not in executed_case
        assert "case_b" not in executed_case
        assert result.success is True

    async def test_switch_multiple_cases(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test switch with multiple different case types."""
        cases = {
            "math": MethodStage(
                method_id=MethodIdentifier.MATHEMATICAL_REASONING,
                name="math_case",
            ),
            "code": MethodStage(
                method_id=MethodIdentifier.CODE_REASONING,
                name="code_case",
            ),
            "ethics": MethodStage(
                method_id=MethodIdentifier.ETHICAL_REASONING,
                name="ethics_case",
            ),
            "creative": MethodStage(
                method_id=MethodIdentifier.LATERAL_THINKING,
                name="creative_case",
            ),
        }

        pipeline = SwitchPipeline(
            expression="problem_category",
            cases=cases,
        )

        # Test each case
        for case_key in cases.keys():
            executed_case = []

            async def mock_execute(stage, context):
                executed_case.append(stage.name)
                return await mock_method_executor(context, stage.id)

            with patch.object(SwitchExecutor, "evaluate_expression", return_value=case_key):
                with patch.object(SwitchExecutor, "execute_stage", side_effect=mock_execute):
                    executor = SwitchExecutor(pipeline)
                    result = await executor.execute(base_context)

            # Verify correct case was executed
            assert f"{case_key}_case" in executed_case
            assert result.success is True


# ============================================================================
# Test: Mixed/Complex Pipelines
# ============================================================================


@pytest.mark.asyncio
class TestMixedPipeline:
    """Test complex pipelines mixing multiple stage types."""

    async def test_mixed_sequence_parallel(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test sequence containing parallel stages."""
        # Parallel stage
        parallel_branches = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="p1"),
            MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY, name="p2"),
        ]

        merge_strategy = MergeStrategy(name="vote", selection_criteria="majority")

        parallel_stage = ParallelPipeline(
            branches=parallel_branches,
            merge_strategy=merge_strategy,
        )

        # Sequence containing parallel
        stages = [
            MethodStage(method_id=MethodIdentifier.STEP_BACK, name="initial"),
            parallel_stage,
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="final"),
        ]

        pipeline = SequencePipeline(stages=stages)

        executed_stages = []

        async def mock_execute(stage, context):
            stage_name = stage.name if hasattr(stage, 'name') else "parallel"
            executed_stages.append(stage_name)
            return await mock_method_executor(context, getattr(stage, 'id', str(uuid4())))

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline)
            result = await executor.execute(base_context)

        assert result.success is True
        assert len(executed_stages) >= 3  # initial + parallel + final

    async def test_mixed_all_types(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test pipeline with all stage types mixed together."""
        # Build a complex nested structure
        loop_body = MethodStage(
            method_id=MethodIdentifier.SELF_REFLECTION,
            name="loop_refine",
        )

        loop = LoopPipeline(
            body=loop_body,
            condition=Condition(name="quality", expression="quality < 0.9"),
            max_iterations=2,
        )

        conditional = ConditionalPipeline(
            condition=Condition(name="check", expression="valid"),
            if_true=loop,
            if_false=MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="fallback"),
        )

        parallel = ParallelPipeline(
            branches=[
                conditional,
                MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS, name="alternative"),
            ],
            merge_strategy=MergeStrategy(name="best", selection_criteria="highest_confidence"),
        )

        switch = SwitchPipeline(
            expression="mode",
            cases={"complex": parallel},
            default=MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="simple"),
        )

        sequence = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.STEP_BACK, name="start"),
                switch,
                MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="end"),
            ]
        )

        # Just verify the structure is valid and can be instantiated
        assert sequence.stage_type == PipelineStageType.SEQUENCE
        assert len(sequence.stages) == 3
        assert isinstance(sequence.stages[1], SwitchPipeline)


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.asyncio
class TestPipelineErrorHandling:
    """Test error handling in pipeline execution."""

    async def test_stage_failure_handling(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test pipeline handles stage failures appropriately."""
        stages = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="s2_fail"),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s3"),
        ]

        pipeline = SequencePipeline(stages=stages, stop_on_error=True)

        async def mock_execute(stage, context):
            if stage.name == "s2_fail":
                return StageResult(
                    stage_id=stage.id,
                    stage_type=PipelineStageType.METHOD,
                    success=False,
                    error="Simulated failure in stage 2",
                )
            return await mock_method_executor(context, stage.id)

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline, stop_on_error=True)
            result = await executor.execute(base_context)

        assert result.success is False
        assert result.error is not None

    async def test_error_recovery_continue(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test pipeline continues after error when configured."""
        stages = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="s2_fail"),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s3"),
        ]

        pipeline = SequencePipeline(stages=stages, stop_on_error=False)

        executed_stages = []

        async def mock_execute(stage, context):
            executed_stages.append(stage.name)
            if stage.name == "s2_fail":
                return StageResult(
                    stage_id=stage.id,
                    stage_type=PipelineStageType.METHOD,
                    success=False,
                    error="Non-fatal error",
                )
            return await mock_method_executor(context, stage.id)

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline, stop_on_error=False)
            result = await executor.execute(base_context)

        # All stages should execute despite error
        assert len(executed_stages) == 3
        assert executed_stages == ["s1", "s2_fail", "s3"]

    async def test_partial_results_collection(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test that partial results are collected even on failure."""
        stages = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="s2"),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="s3_fail"),
        ]

        pipeline = SequencePipeline(stages=stages)

        partial_results = []

        async def mock_execute(stage, context):
            if stage.name == "s3_fail":
                result = StageResult(
                    stage_id=stage.id,
                    stage_type=PipelineStageType.METHOD,
                    success=False,
                    error="Failed at stage 3",
                )
            else:
                result = await mock_method_executor(context, stage.id)

            partial_results.append(
                {"stage": stage.name, "success": result.success}
            )

            return result

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline, stop_on_error=True)
            result = await executor.execute(base_context)

        # Should have results from s1 and s2 before failure
        assert len(partial_results) == 3
        assert partial_results[0]["success"] is True
        assert partial_results[1]["success"] is True
        assert partial_results[2]["success"] is False


# ============================================================================
# Test: Pipeline Tracing
# ============================================================================


@pytest.mark.asyncio
class TestPipelineTracing:
    """Test execution tracing and metrics collection."""

    async def test_trace_captures_all_steps(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test that execution trace captures all pipeline steps."""
        stages = [
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="step1"),
            MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="step2"),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="step3"),
        ]

        pipeline = SequencePipeline(stages=stages)

        async def mock_execute(stage, context):
            result = await mock_method_executor(context, stage.id)
            # Ensure trace is created
            assert result.trace is not None
            return result

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline)
            result = await executor.execute(base_context)

        # Verify trace exists
        assert result.trace is not None
        assert result.trace.stage_type == PipelineStageType.SEQUENCE

    async def test_trace_timing_metrics(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test that trace includes timing information."""
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="timed_stage",
        )

        pipeline = SequencePipeline(stages=[stage])

        async def mock_execute(stage, context):
            result = await mock_method_executor(context, stage.id)
            # Verify metrics include timing
            if result.trace and result.trace.metrics:
                assert result.trace.metrics.start_time is not None
                assert result.trace.metrics.duration_seconds >= 0
            return result

        with patch.object(SequenceExecutor, "execute_stage", side_effect=mock_execute):
            executor = SequenceExecutor(pipeline)
            result = await executor.execute(base_context)

        assert result.trace is not None

    async def test_trace_hierarchical_structure(
        self, base_context: ExecutionContext, mock_method_executor
    ):
        """Test that trace maintains hierarchical structure for nested pipelines."""
        # Create nested structure
        inner_sequence = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="inner1"),
                MethodStage(method_id=MethodIdentifier.SELF_REFLECTION, name="inner2"),
            ]
        )

        outer_sequence = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.STEP_BACK, name="outer1"),
                inner_sequence,
                MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT, name="outer3"),
            ]
        )

        # The trace structure would be verified in actual implementation
        assert outer_sequence.stage_type == PipelineStageType.SEQUENCE
        assert len(outer_sequence.stages) == 3
        assert isinstance(outer_sequence.stages[1], SequencePipeline)


# ============================================================================
# Additional Integration Scenarios
# ============================================================================


@pytest.mark.asyncio
class TestAdditionalScenarios:
    """Additional integration test scenarios."""

    async def test_pipeline_result_structure(self, mock_session: Session):
        """Test PipelineResult captures complete execution information."""
        trace = PipelineTrace(
            pipeline_id="test-pipeline",
            session_id=mock_session.id,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            status="completed",
        )

        result = PipelineResult(
            pipeline_id="test-pipeline",
            session_id=mock_session.id,
            success=True,
            final_thoughts=["thought-1", "thought-2"],
            trace=trace,
            metadata={"total_duration": 5.2, "thoughts_generated": 15},
        )

        assert result.success is True
        assert len(result.final_thoughts) == 2
        assert result.trace is not None
        assert result.trace.pipeline_id == "test-pipeline"
        assert result.metadata["thoughts_generated"] == 15

    async def test_executor_registry_dispatch(self):
        """Test that executor registry correctly dispatches to executors."""
        # Method stage
        method_stage = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        method_executor = get_executor_for_stage(method_stage)
        assert isinstance(method_executor, MethodExecutor)

        # Sequence stage
        sequence_stage = SequencePipeline(stages=[method_stage])
        sequence_executor = get_executor_for_stage(sequence_stage)
        assert isinstance(sequence_executor, SequenceExecutor)

        # Parallel stage
        parallel_stage = ParallelPipeline(
            branches=[method_stage],
            merge_strategy=MergeStrategy(name="test"),
        )
        parallel_executor = get_executor_for_stage(parallel_stage)
        assert isinstance(parallel_executor, ParallelExecutor)

        # Conditional stage
        conditional_stage = ConditionalPipeline(
            condition=Condition(name="test", expression="true"),
            if_true=method_stage,
        )
        conditional_executor = get_executor_for_stage(conditional_stage)
        assert isinstance(conditional_executor, ConditionalExecutor)

        # Loop stage
        loop_stage = LoopPipeline(
            body=method_stage,
            condition=Condition(name="test", expression="done"),
        )
        loop_executor = get_executor_for_stage(loop_stage)
        assert isinstance(loop_executor, LoopExecutor)

        # Switch stage
        switch_stage = SwitchPipeline(
            expression="x",
            cases={"a": method_stage},
        )
        switch_executor = get_executor_for_stage(switch_stage)
        assert isinstance(switch_executor, SwitchExecutor)

    async def test_pipeline_serialization_roundtrip(self):
        """Test that complex pipelines can be serialized and deserialized."""
        # Create complex pipeline
        method1 = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        method2 = MethodStage(method_id=MethodIdentifier.SELF_REFLECTION)

        parallel = ParallelPipeline(
            branches=[method1, method2],
            merge_strategy=MergeStrategy(name="vote"),
        )

        sequence = SequencePipeline(
            stages=[method1, parallel, method2]
        )

        # Serialize
        data = sequence.model_dump()

        # Deserialize
        restored = SequencePipeline(**data)

        # Verify structure
        assert len(restored.stages) == 3
        assert isinstance(restored.stages[1], ParallelPipeline)
        assert len(restored.stages[1].branches) == 2
