"""
Comprehensive tests for Pipeline DSL models in reasoning_mcp.models.pipeline.

This module provides complete test coverage for all pipeline models:
- Helper Models (frozen): Transform, Condition, MergeStrategy, Accumulator, ErrorHandler
- Pipeline Stage Models (mutable): MethodStage, SequencePipeline, ParallelPipeline,
  ConditionalPipeline, LoopPipeline, SwitchPipeline
- Result/Trace Models (frozen): StageMetrics, StageTrace, PipelineTrace, PipelineResult

Each model is tested for:
1. Creation with minimal and full parameters
2. Immutability (frozen models)
3. Field validation
4. Serialization/deserialization
5. Discriminated union behavior (for Pipeline)
"""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType
from reasoning_mcp.models.pipeline import (
    Accumulator,
    Condition,
    ConditionalPipeline,
    ErrorHandler,
    LoopPipeline,
    MergeStrategy,
    MethodStage,
    ParallelPipeline,
    Pipeline,
    PipelineResult,
    PipelineTrace,
    SequencePipeline,
    StageMetrics,
    StageTrace,
    SwitchPipeline,
    Transform,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_transform() -> Transform:
    """Create a sample Transform for testing."""
    return Transform(name="summarize", expression="{content} summarized")


@pytest.fixture
def sample_condition() -> Condition:
    """Create a sample Condition for testing."""
    return Condition(name="confidence_check", expression="confidence > 0.8")


@pytest.fixture
def sample_merge_strategy() -> MergeStrategy:
    """Create a sample MergeStrategy for testing."""
    return MergeStrategy(name="vote")


@pytest.fixture
def sample_accumulator() -> Accumulator:
    """Create a sample Accumulator for testing."""
    return Accumulator(name="collector")


@pytest.fixture
def sample_error_handler() -> ErrorHandler:
    """Create a sample ErrorHandler for testing."""
    return ErrorHandler(strategy="retry")


@pytest.fixture
def sample_method_stage() -> MethodStage:
    """Create a sample MethodStage for testing."""
    return MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)


@pytest.fixture
def sample_sequence_pipeline(sample_method_stage: MethodStage) -> SequencePipeline:
    """Create a sample SequencePipeline for testing."""
    return SequencePipeline(stages=[sample_method_stage])


# ============================================================================
# TestTransform
# ============================================================================


class TestTransform:
    """Test suite for Transform model."""

    def test_create_transform_minimal(self):
        """Test creating Transform with minimal required parameters."""
        transform = Transform(name="test", expression="test_expr")
        assert transform.name == "test"
        assert transform.expression == "test_expr"
        assert transform.input_fields == []
        assert transform.output_field == "transformed_content"
        assert transform.metadata == {}

    def test_create_transform_full(self):
        """Test creating Transform with all parameters."""
        transform = Transform(
            name="extract_summary",
            expression="{result.summary}",
            input_fields=["result", "content"],
            output_field="summary",
            metadata={"key": "value"},
        )
        assert transform.name == "extract_summary"
        assert transform.expression == "{result.summary}"
        assert transform.input_fields == ["result", "content"]
        assert transform.output_field == "summary"
        assert transform.metadata == {"key": "value"}

    def test_transform_is_frozen(self):
        """Test that Transform instances are immutable."""
        transform = Transform(name="test", expression="expr")
        with pytest.raises(ValidationError):
            transform.name = "modified"  # type: ignore[misc]

    def test_transform_validation(self):
        """Test Transform field validation."""
        # Valid creation - name and expression required
        transform = Transform(name="valid", expression="valid_expression")
        assert transform.name == "valid"

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            Transform()  # type: ignore[call-arg]

    def test_transform_serialization(self):
        """Test Transform serialization to dict and JSON."""
        transform = Transform(
            name="serialize_test",
            expression="{x}",
            input_fields=["x"],
        )

        # To dict
        data = transform.model_dump()
        assert data["name"] == "serialize_test"
        assert data["expression"] == "{x}"
        assert data["input_fields"] == ["x"]

        # From dict
        restored = Transform(**data)
        assert restored == transform

    def test_transform_default_values(self):
        """Test Transform default field values."""
        transform = Transform(name="test", expression="expr")
        assert transform.input_fields == []
        assert transform.output_field == "transformed_content"
        assert transform.metadata == {}


# ============================================================================
# TestCondition
# ============================================================================


class TestCondition:
    """Test suite for Condition model."""

    def test_create_condition_minimal(self):
        """Test creating Condition with minimal required parameters."""
        condition = Condition(name="test", expression="test_expr")
        assert condition.name == "test"
        assert condition.expression == "test_expr"
        assert condition.operator == "=="
        assert condition.threshold is None
        assert condition.field is None
        assert condition.metadata == {}

    def test_create_condition_full(self):
        """Test creating Condition with all parameters."""
        condition = Condition(
            name="complexity_check",
            expression="complexity >= 5",
            operator=">=",
            threshold=5.0,
            field="complexity",
            metadata={"description": "check complexity"},
        )
        assert condition.name == "complexity_check"
        assert condition.expression == "complexity >= 5"
        assert condition.operator == ">="
        assert condition.threshold == 5.0
        assert condition.field == "complexity"

    def test_condition_is_frozen(self):
        """Test that Condition instances are immutable."""
        condition = Condition(name="test", expression="expr")
        with pytest.raises(ValidationError):
            condition.name = "modified"  # type: ignore[misc]

    def test_condition_validation(self):
        """Test Condition field validation."""
        # Valid creation
        condition = Condition(name="valid", expression="x > 0")
        assert condition.name == "valid"

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            Condition()  # type: ignore[call-arg]

    def test_condition_serialization(self):
        """Test Condition serialization to dict and JSON."""
        condition = Condition(
            name="test_cond",
            expression="score > 0.5",
            threshold=0.5,
        )

        # To dict
        data = condition.model_dump()
        assert data["name"] == "test_cond"
        assert data["expression"] == "score > 0.5"

        # From dict
        restored = Condition(**data)
        assert restored == condition

    def test_condition_complex_expressions(self):
        """Test Condition with complex expressions."""
        condition = Condition(
            name="complex",
            expression="(a and b) or (c and not d)",
        )
        assert "and" in condition.expression
        assert "or" in condition.expression


# ============================================================================
# TestMergeStrategy
# ============================================================================


class TestMergeStrategy:
    """Test suite for MergeStrategy model."""

    def test_create_merge_strategy_minimal(self):
        """Test creating MergeStrategy with minimal required parameters."""
        strategy = MergeStrategy(name="test")
        assert strategy.name == "test"
        assert strategy.selection_criteria == "highest_confidence"
        assert strategy.weights == {}
        assert strategy.aggregation is None
        assert strategy.metadata == {}

    def test_create_merge_strategy_with_weights(self):
        """Test creating MergeStrategy with weights."""
        weights = {"method_a": 0.6, "method_b": 0.4}
        strategy = MergeStrategy(
            name="weighted_merge",
            selection_criteria="weighted_average",
            weights=weights,
            aggregation="weighted",
        )
        assert strategy.weights == weights
        assert strategy.aggregation == "weighted"

    def test_merge_strategy_is_frozen(self):
        """Test that MergeStrategy instances are immutable."""
        strategy = MergeStrategy(name="test")
        with pytest.raises(ValidationError):
            strategy.name = "modified"  # type: ignore[misc]

    def test_merge_strategy_serialization(self):
        """Test MergeStrategy serialization to dict and JSON."""
        strategy = MergeStrategy(
            name="vote",
            selection_criteria="majority",
        )

        # To dict
        data = strategy.model_dump()
        assert data["name"] == "vote"

        # From dict
        restored = MergeStrategy(**data)
        assert restored == strategy

    def test_merge_strategy_with_complex_weights(self):
        """Test MergeStrategy with complex weight configurations."""
        weights = {
            "sequential": 0.3,
            "tree": 0.4,
            "self_consistency": 0.3,
        }
        strategy = MergeStrategy(
            name="ensemble",
            selection_criteria="weighted",
            weights=weights,
        )
        assert sum(strategy.weights.values()) == pytest.approx(1.0)


# ============================================================================
# TestAccumulator
# ============================================================================


class TestAccumulator:
    """Test suite for Accumulator model."""

    def test_create_accumulator_minimal(self):
        """Test creating Accumulator with minimal required parameters."""
        accumulator = Accumulator(name="test")
        assert accumulator.name == "test"
        assert accumulator.initial_value is None
        assert accumulator.operation == "append"
        assert accumulator.field == "content"
        assert accumulator.metadata == {}

    def test_create_accumulator_full(self):
        """Test creating Accumulator with all parameters."""
        accumulator = Accumulator(
            name="collector",
            initial_value=[],
            operation="concat",
            field="results",
            metadata={"type": "list"},
        )
        assert accumulator.initial_value == []
        assert accumulator.operation == "concat"
        assert accumulator.field == "results"

    def test_accumulator_is_frozen(self):
        """Test that Accumulator instances are immutable."""
        accumulator = Accumulator(name="test")
        with pytest.raises(ValidationError):
            accumulator.operation = "merge"  # type: ignore[misc]

    def test_accumulator_serialization(self):
        """Test Accumulator serialization to dict and JSON."""
        accumulator = Accumulator(
            name="test_acc",
            initial_value=0,
            operation="sum",
        )

        # To dict
        data = accumulator.model_dump()
        assert data["initial_value"] == 0
        assert data["operation"] == "sum"

        # From dict
        restored = Accumulator(**data)
        assert restored == accumulator

    def test_accumulator_operations(self):
        """Test different Accumulator operation types."""
        operations = ["append", "sum", "merge", "max", "min"]
        for op in operations:
            accumulator = Accumulator(name=f"test_{op}", operation=op)
            assert accumulator.operation == op


# ============================================================================
# TestErrorHandler
# ============================================================================


class TestErrorHandler:
    """Test suite for ErrorHandler model."""

    def test_create_error_handler_minimal(self):
        """Test creating ErrorHandler with minimal required parameters."""
        handler = ErrorHandler(strategy="retry")
        assert handler.strategy == "retry"
        assert handler.max_retries == 3
        assert handler.fallback_method is None
        assert handler.on_failure == "raise"
        assert handler.metadata == {}

    def test_create_error_handler_full(self):
        """Test creating ErrorHandler with all parameters."""
        handler = ErrorHandler(
            strategy="fallback",
            max_retries=5,
            fallback_method=MethodIdentifier.CHAIN_OF_THOUGHT,
            on_failure="skip",
            metadata={"reason": "test"},
        )
        assert handler.strategy == "fallback"
        assert handler.max_retries == 5
        assert handler.fallback_method == MethodIdentifier.CHAIN_OF_THOUGHT
        assert handler.on_failure == "skip"

    def test_error_handler_is_frozen(self):
        """Test that ErrorHandler instances are immutable."""
        handler = ErrorHandler(strategy="retry")
        with pytest.raises(ValidationError):
            handler.strategy = "skip"  # type: ignore[misc]

    def test_max_retries_validation(self):
        """Test ErrorHandler max_retries field validation."""
        # Valid range: 0-10
        handler = ErrorHandler(strategy="retry", max_retries=0)
        assert handler.max_retries == 0

        handler = ErrorHandler(strategy="retry", max_retries=10)
        assert handler.max_retries == 10

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            ErrorHandler(strategy="retry", max_retries=11)

        # Invalid: below minimum (negative)
        with pytest.raises(ValidationError):
            ErrorHandler(strategy="retry", max_retries=-1)

    def test_error_handler_serialization(self):
        """Test ErrorHandler serialization to dict and JSON."""
        handler = ErrorHandler(
            strategy="fallback",
            max_retries=2,
        )

        # To dict
        data = handler.model_dump()
        assert data["strategy"] == "fallback"
        assert data["max_retries"] == 2

        # From dict
        restored = ErrorHandler(**data)
        assert restored == handler

    def test_error_handler_strategies(self):
        """Test different ErrorHandler strategy types."""
        strategies = ["retry", "skip", "fallback", "raise"]
        for strategy in strategies:
            handler = ErrorHandler(strategy=strategy)
            assert handler.strategy == strategy


# ============================================================================
# TestMethodStage
# ============================================================================


class TestMethodStage:
    """Test suite for MethodStage model."""

    def test_create_method_stage_minimal(self):
        """Test creating MethodStage with minimal required parameters."""
        stage = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        assert stage.stage_type == PipelineStageType.METHOD
        assert stage.method_id == MethodIdentifier.SEQUENTIAL_THINKING
        assert stage.id is not None  # Auto-generated UUID
        assert stage.name is None
        assert stage.max_thoughts == 10
        assert stage.timeout_seconds == 60.0
        assert stage.transforms == []
        assert stage.error_handler is None

    def test_create_method_stage_full(self):
        """Test creating MethodStage with all parameters."""
        transform = Transform(name="prep", expression="{input}")
        handler = ErrorHandler(strategy="retry", max_retries=2)

        stage = MethodStage(
            method_id=MethodIdentifier.TREE_OF_THOUGHTS,
            name="stage_1",
            description="Test stage",
            max_thoughts=20,
            timeout_seconds=120.0,
            transforms=[transform],
            error_handler=handler,
            metadata={"key": "value"},
        )
        assert stage.method_id == MethodIdentifier.TREE_OF_THOUGHTS
        assert stage.name == "stage_1"
        assert stage.max_thoughts == 20
        assert stage.timeout_seconds == 120.0
        assert len(stage.transforms) == 1
        assert stage.error_handler == handler

    def test_method_stage_has_correct_type(self):
        """Test that MethodStage has correct type discriminator."""
        stage = MethodStage(method_id=MethodIdentifier.REACT)
        assert stage.stage_type == PipelineStageType.METHOD

    def test_method_stage_with_transforms(self):
        """Test MethodStage with transformations."""
        input_tf = Transform(name="input", expression="{question}")
        output_tf = Transform(name="output", expression="{answer}")

        stage = MethodStage(
            method_id=MethodIdentifier.SELF_CONSISTENCY,
            transforms=[input_tf, output_tf],
        )
        assert len(stage.transforms) == 2
        assert stage.transforms[0] == input_tf
        assert stage.transforms[1] == output_tf

    def test_method_stage_with_error_handler(self):
        """Test MethodStage with error handler."""
        handler = ErrorHandler(strategy="skip")
        stage = MethodStage(
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            error_handler=handler,
        )
        assert stage.error_handler == handler
        assert stage.error_handler.strategy == "skip"

    def test_method_stage_validation(self):
        """Test MethodStage field validation."""
        # Valid: max_thoughts within range
        stage = MethodStage(
            method_id=MethodIdentifier.ETHICAL_REASONING,
            max_thoughts=50,
        )
        assert stage.max_thoughts == 50

        # Invalid: max_thoughts too high
        with pytest.raises(ValidationError):
            MethodStage(
                method_id=MethodIdentifier.ETHICAL_REASONING,
                max_thoughts=101,
            )

    def test_method_stage_serialization(self):
        """Test MethodStage serialization to dict and JSON."""
        stage = MethodStage(
            method_id=MethodIdentifier.CODE_REASONING,
            name="code_stage",
        )

        # To dict
        data = stage.model_dump()
        assert data["stage_type"] == PipelineStageType.METHOD
        assert data["name"] == "code_stage"

        # From dict
        restored = MethodStage(**data)
        assert restored.method_id == stage.method_id
        assert restored.name == stage.name

    def test_method_stage_with_all_method_identifiers(self):
        """Test MethodStage works with all MethodIdentifier values."""
        for method in MethodIdentifier:
            stage = MethodStage(method_id=method)
            assert stage.method_id == method
            assert stage.stage_type == PipelineStageType.METHOD


# ============================================================================
# TestSequencePipeline
# ============================================================================


class TestSequencePipeline:
    """Test suite for SequencePipeline model."""

    def test_create_sequence_pipeline_empty(self):
        """Test creating SequencePipeline with no stages."""
        pipeline = SequencePipeline()
        assert pipeline.stage_type == PipelineStageType.SEQUENCE
        assert pipeline.stages == []
        assert pipeline.id is not None  # Auto-generated

    def test_create_sequence_pipeline_with_stages(self):
        """Test creating SequencePipeline with stages."""
        stage1 = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        stage2 = MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)

        pipeline = SequencePipeline(
            name="seq_1",
            stages=[stage1, stage2],
        )
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0] == stage1
        assert pipeline.stages[1] == stage2

    def test_sequence_pipeline_has_correct_type(self):
        """Test that SequencePipeline has correct type discriminator."""
        pipeline = SequencePipeline()
        assert pipeline.stage_type == PipelineStageType.SEQUENCE

    def test_nested_sequence_pipelines(self):
        """Test nesting SequencePipeline within SequencePipeline."""
        inner = SequencePipeline(
            stages=[MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)]
        )
        outer = SequencePipeline(stages=[inner])

        assert len(outer.stages) == 1
        assert isinstance(outer.stages[0], SequencePipeline)

    def test_sequence_pipeline_serialization(self):
        """Test SequencePipeline serialization to dict and JSON."""
        stage = MethodStage(method_id=MethodIdentifier.REACT)
        pipeline = SequencePipeline(name="test_seq", stages=[stage])

        # To dict
        data = pipeline.model_dump()
        assert data["stage_type"] == PipelineStageType.SEQUENCE
        assert data["name"] == "test_seq"
        assert len(data["stages"]) == 1

        # From dict
        restored = SequencePipeline(**data)
        assert restored.name == pipeline.name
        assert len(restored.stages) == len(pipeline.stages)

    def test_sequence_pipeline_with_mixed_stages(self):
        """Test SequencePipeline with different stage types."""
        method_stage = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS)
        parallel_stage = ParallelPipeline(
            branches=[MethodStage(method_id=MethodIdentifier.DIALECTIC)],
            merge_strategy=MergeStrategy(name="test"),
        )

        pipeline = SequencePipeline(stages=[method_stage, parallel_stage])
        assert len(pipeline.stages) == 2
        assert isinstance(pipeline.stages[0], MethodStage)
        assert isinstance(pipeline.stages[1], ParallelPipeline)


# ============================================================================
# TestParallelPipeline
# ============================================================================


class TestParallelPipeline:
    """Test suite for ParallelPipeline model."""

    def test_create_parallel_pipeline_minimal(self):
        """Test creating ParallelPipeline with minimal required parameters."""
        branch = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        merge = MergeStrategy(name="default")
        pipeline = ParallelPipeline(branches=[branch], merge_strategy=merge)

        assert pipeline.stage_type == PipelineStageType.PARALLEL
        assert len(pipeline.branches) == 1
        assert pipeline.merge_strategy.name == "default"

    def test_create_parallel_pipeline_with_branches(self):
        """Test creating ParallelPipeline with multiple branches."""
        branches = [
            MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
            MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY),
        ]
        merge = MergeStrategy(name="vote", selection_criteria="majority")

        pipeline = ParallelPipeline(
            name="parallel_1",
            branches=branches,
            merge_strategy=merge,
        )
        assert len(pipeline.branches) == 3
        assert pipeline.merge_strategy.selection_criteria == "majority"

    def test_parallel_pipeline_has_correct_type(self):
        """Test that ParallelPipeline has correct type discriminator."""
        branch = MethodStage(method_id=MethodIdentifier.REACT)
        merge = MergeStrategy(name="test")
        pipeline = ParallelPipeline(branches=[branch], merge_strategy=merge)
        assert pipeline.stage_type == PipelineStageType.PARALLEL

    def test_parallel_pipeline_validation(self):
        """Test ParallelPipeline validation requires branches and merge."""
        # Valid: required fields provided
        branch = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        merge = MergeStrategy(name="test")
        pipeline = ParallelPipeline(branches=[branch], merge_strategy=merge)
        assert len(pipeline.branches) >= 1

    def test_parallel_pipeline_serialization(self):
        """Test ParallelPipeline serialization to dict and JSON."""
        branches = [
            MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
            MethodStage(method_id=MethodIdentifier.DIALECTIC),
        ]
        merge = MergeStrategy(name="combine")
        pipeline = ParallelPipeline(
            name="test_parallel",
            branches=branches,
            merge_strategy=merge,
        )

        # To dict
        data = pipeline.model_dump()
        assert data["stage_type"] == PipelineStageType.PARALLEL
        assert len(data["branches"]) == 2

        # From dict
        restored = ParallelPipeline(**data)
        assert len(restored.branches) == len(pipeline.branches)

    def test_parallel_pipeline_with_custom_merge(self):
        """Test ParallelPipeline with custom merge strategy."""
        branches = [
            MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
        ]
        merge = MergeStrategy(
            name="weighted",
            selection_criteria="best",
            weights={"branch_0": 0.7, "branch_1": 0.3},
        )
        pipeline = ParallelPipeline(branches=branches, merge_strategy=merge)

        assert pipeline.merge_strategy.weights is not None
        assert pipeline.merge_strategy.weights["branch_0"] == 0.7


# ============================================================================
# TestConditionalPipeline
# ============================================================================


class TestConditionalPipeline:
    """Test suite for ConditionalPipeline model."""

    def test_create_conditional_pipeline_minimal(self):
        """Test creating ConditionalPipeline with minimal required parameters."""
        condition = Condition(name="test", expression="true")
        if_true = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)

        pipeline = ConditionalPipeline(
            condition=condition,
            if_true=if_true,
        )
        assert pipeline.stage_type == PipelineStageType.CONDITIONAL
        assert pipeline.condition == condition
        assert pipeline.if_true == if_true
        assert pipeline.if_false is None

    def test_create_conditional_pipeline_with_else(self):
        """Test creating ConditionalPipeline with else branch."""
        condition = Condition(name="check", expression="score > 0.5")
        if_true = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS)
        if_false = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)

        pipeline = ConditionalPipeline(
            name="cond_1",
            condition=condition,
            if_true=if_true,
            if_false=if_false,
        )
        assert pipeline.if_false == if_false

    def test_conditional_pipeline_has_correct_type(self):
        """Test that ConditionalPipeline has correct type discriminator."""
        condition = Condition(name="test", expression="x")
        if_true = MethodStage(method_id=MethodIdentifier.REACT)
        pipeline = ConditionalPipeline(
            condition=condition,
            if_true=if_true,
        )
        assert pipeline.stage_type == PipelineStageType.CONDITIONAL

    def test_conditional_pipeline_serialization(self):
        """Test ConditionalPipeline serialization to dict and JSON."""
        condition = Condition(name="test", expression="x > 0")
        if_true = MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)
        pipeline = ConditionalPipeline(
            name="test_cond",
            condition=condition,
            if_true=if_true,
        )

        # To dict
        data = pipeline.model_dump()
        assert data["stage_type"] == PipelineStageType.CONDITIONAL
        assert data["name"] == "test_cond"

        # From dict
        restored = ConditionalPipeline(**data)
        assert restored.name == pipeline.name

    def test_conditional_pipeline_nested(self):
        """Test nested ConditionalPipeline."""
        inner_condition = Condition(name="inner", expression="y > 0")
        inner_if_true = MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING)
        inner_pipeline = ConditionalPipeline(
            condition=inner_condition,
            if_true=inner_if_true,
        )

        outer_condition = Condition(name="outer", expression="x > 0")
        outer_pipeline = ConditionalPipeline(
            condition=outer_condition,
            if_true=inner_pipeline,
        )

        assert isinstance(outer_pipeline.if_true, ConditionalPipeline)

    def test_conditional_pipeline_with_sequence(self):
        """Test ConditionalPipeline with SequencePipeline branches."""
        condition = Condition(name="test", expression="valid")
        if_true = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
                MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY),
            ]
        )
        pipeline = ConditionalPipeline(
            condition=condition,
            if_true=if_true,
        )
        assert isinstance(pipeline.if_true, SequencePipeline)


# ============================================================================
# TestLoopPipeline
# ============================================================================


class TestLoopPipeline:
    """Test suite for LoopPipeline model."""

    def test_create_loop_pipeline_minimal(self):
        """Test creating LoopPipeline with minimal required parameters."""
        body = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        until = Condition(name="done", expression="complete")

        pipeline = LoopPipeline(body=body, condition=until)
        assert pipeline.stage_type == PipelineStageType.LOOP
        assert pipeline.body == body
        assert pipeline.condition == until
        assert pipeline.max_iterations == 10
        assert pipeline.accumulator is None

    def test_create_loop_pipeline_with_accumulator(self):
        """Test creating LoopPipeline with accumulator."""
        body = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        until = Condition(name="threshold", expression="score > 0.9")
        accumulator = Accumulator(name="acc", initial_value=[], operation="append")

        pipeline = LoopPipeline(
            name="loop_1",
            body=body,
            condition=until,
            max_iterations=5,
            accumulator=accumulator,
        )
        assert pipeline.max_iterations == 5
        assert pipeline.accumulator == accumulator

    def test_loop_pipeline_has_correct_type(self):
        """Test that LoopPipeline has correct type discriminator."""
        body = MethodStage(method_id=MethodIdentifier.REACT)
        until = Condition(name="test", expression="done")
        pipeline = LoopPipeline(body=body, condition=until)
        assert pipeline.stage_type == PipelineStageType.LOOP

    def test_loop_pipeline_max_iterations_validation(self):
        """Test LoopPipeline max_iterations field validation."""
        body = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS)
        until = Condition(name="test", expression="done")

        # Valid range: 1-100
        pipeline = LoopPipeline(
            body=body,
            condition=until,
            max_iterations=1,
        )
        assert pipeline.max_iterations == 1

        pipeline = LoopPipeline(
            body=body,
            condition=until,
            max_iterations=100,
        )
        assert pipeline.max_iterations == 100

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            LoopPipeline(body=body, condition=until, max_iterations=0)

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            LoopPipeline(body=body, condition=until, max_iterations=101)

    def test_loop_pipeline_serialization(self):
        """Test LoopPipeline serialization to dict and JSON."""
        body = MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)
        until = Condition(name="done", expression="finished")
        pipeline = LoopPipeline(name="test_loop", body=body, condition=until)

        # To dict
        data = pipeline.model_dump()
        assert data["stage_type"] == PipelineStageType.LOOP
        assert data["name"] == "test_loop"

        # From dict
        restored = LoopPipeline(**data)
        assert restored.name == pipeline.name

    def test_loop_pipeline_with_sequence_body(self):
        """Test LoopPipeline with SequencePipeline as body."""
        body = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
                MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
            ]
        )
        until = Condition(name="convergence", expression="converged")
        pipeline = LoopPipeline(body=body, condition=until)

        assert isinstance(pipeline.body, SequencePipeline)


# ============================================================================
# TestSwitchPipeline
# ============================================================================


class TestSwitchPipeline:
    """Test suite for SwitchPipeline model."""

    def test_create_switch_pipeline_minimal(self):
        """Test creating SwitchPipeline with minimal required parameters."""
        cases = {
            "case_a": MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            "case_b": MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
        }
        pipeline = SwitchPipeline(expression="mode", cases=cases)

        assert pipeline.stage_type == PipelineStageType.SWITCH
        assert pipeline.expression == "mode"
        assert len(pipeline.cases) == 2
        assert pipeline.default is None

    def test_create_switch_pipeline_with_default(self):
        """Test creating SwitchPipeline with default case."""
        cases = {
            "fast": MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            "thorough": MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
        }
        default = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)

        pipeline = SwitchPipeline(
            name="switch_1",
            expression="analysis_mode",
            cases=cases,
            default=default,
        )
        assert pipeline.default == default

    def test_switch_pipeline_has_correct_type(self):
        """Test that SwitchPipeline has correct type discriminator."""
        cases = {"a": MethodStage(method_id=MethodIdentifier.REACT)}
        pipeline = SwitchPipeline(expression="x", cases=cases)
        assert pipeline.stage_type == PipelineStageType.SWITCH

    def test_switch_pipeline_serialization(self):
        """Test SwitchPipeline serialization to dict and JSON."""
        cases = {
            "option1": MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
        }
        pipeline = SwitchPipeline(
            name="test_switch",
            expression="option",
            cases=cases,
        )

        # To dict
        data = pipeline.model_dump()
        assert data["stage_type"] == PipelineStageType.SWITCH
        assert data["name"] == "test_switch"
        assert "option1" in data["cases"]

        # From dict
        restored = SwitchPipeline(**data)
        assert restored.name == pipeline.name

    def test_switch_pipeline_with_sequence_cases(self):
        """Test SwitchPipeline with SequencePipeline cases."""
        cases = {
            "simple": MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            "complex": SequencePipeline(
                stages=[
                    MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
                    MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY),
                ]
            ),
        }
        pipeline = SwitchPipeline(expression="complexity", cases=cases)

        assert isinstance(pipeline.cases["simple"], MethodStage)
        assert isinstance(pipeline.cases["complex"], SequencePipeline)

    def test_switch_pipeline_empty_cases(self):
        """Test SwitchPipeline with empty cases dict."""
        # Should be valid - cases can be empty if there's a default
        default = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        pipeline = SwitchPipeline(
            expression="x",
            cases={},
            default=default,
        )
        assert len(pipeline.cases) == 0
        assert pipeline.default is not None


# ============================================================================
# TestPipelineUnion
# ============================================================================


class TestPipelineUnion:
    """Test suite for Pipeline discriminated union."""

    def test_pipeline_union_method_stage(self):
        """Test Pipeline union accepts MethodStage."""
        stage: Pipeline = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        assert isinstance(stage, MethodStage)
        assert stage.stage_type == PipelineStageType.METHOD

    def test_pipeline_union_sequence(self):
        """Test Pipeline union accepts SequencePipeline."""
        pipeline: Pipeline = SequencePipeline()
        assert isinstance(pipeline, SequencePipeline)
        assert pipeline.stage_type == PipelineStageType.SEQUENCE

    def test_pipeline_union_parallel(self):
        """Test Pipeline union accepts ParallelPipeline."""
        branch = MethodStage(method_id=MethodIdentifier.REACT)
        merge = MergeStrategy(name="test")
        pipeline: Pipeline = ParallelPipeline(branches=[branch], merge_strategy=merge)
        assert isinstance(pipeline, ParallelPipeline)
        assert pipeline.stage_type == PipelineStageType.PARALLEL

    def test_pipeline_union_conditional(self):
        """Test Pipeline union accepts ConditionalPipeline."""
        condition = Condition(name="test", expression="true")
        if_true = MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        pipeline: Pipeline = ConditionalPipeline(
            condition=condition,
            if_true=if_true,
        )
        assert isinstance(pipeline, ConditionalPipeline)
        assert pipeline.stage_type == PipelineStageType.CONDITIONAL

    def test_pipeline_union_loop(self):
        """Test Pipeline union accepts LoopPipeline."""
        body = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS)
        until = Condition(name="test", expression="done")
        pipeline: Pipeline = LoopPipeline(body=body, condition=until)
        assert isinstance(pipeline, LoopPipeline)
        assert pipeline.stage_type == PipelineStageType.LOOP

    def test_pipeline_union_switch(self):
        """Test Pipeline union accepts SwitchPipeline."""
        cases = {"a": MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)}
        pipeline: Pipeline = SwitchPipeline(expression="x", cases=cases)
        assert isinstance(pipeline, SwitchPipeline)
        assert pipeline.stage_type == PipelineStageType.SWITCH

    def test_pipeline_discriminator_works(self):
        """Test that Pipeline discriminator correctly identifies types."""
        merge = MergeStrategy(name="test")
        pipelines: list[Pipeline] = [
            MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            SequencePipeline(),
            ParallelPipeline(
                branches=[MethodStage(method_id=MethodIdentifier.REACT)],
                merge_strategy=merge,
            ),
        ]

        assert pipelines[0].stage_type == PipelineStageType.METHOD
        assert pipelines[1].stage_type == PipelineStageType.SEQUENCE
        assert pipelines[2].stage_type == PipelineStageType.PARALLEL

    def test_pipeline_nested_serialization(self):
        """Test serialization of deeply nested pipeline structures."""
        # Create a complex nested structure
        inner = SequencePipeline(
            stages=[
                MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
                MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
            ]
        )
        merge = MergeStrategy(name="combine")
        parallel = ParallelPipeline(branches=[inner], merge_strategy=merge)
        outer = SequencePipeline(stages=[parallel])

        # Serialize to dict
        data = outer.model_dump()
        assert data["stage_type"] == PipelineStageType.SEQUENCE
        assert data["stages"][0]["stage_type"] == PipelineStageType.PARALLEL

        # Deserialize
        restored = SequencePipeline(**data)
        assert isinstance(restored.stages[0], ParallelPipeline)


# ============================================================================
# TestStageMetrics
# ============================================================================


class TestStageMetrics:
    """Test suite for StageMetrics model."""

    def test_create_stage_metrics(self):
        """Test creating StageMetrics."""
        from datetime import datetime

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 5)
        metrics = StageMetrics(
            stage_id="stage_1",
            start_time=start,
            end_time=end,
            duration_seconds=5.0,
            thoughts_generated=10,
            errors_count=0,
            retries_count=1,
        )
        assert metrics.stage_id == "stage_1"
        assert metrics.start_time == start
        assert metrics.end_time == end
        assert metrics.duration_seconds == 5.0
        assert metrics.thoughts_generated == 10
        assert metrics.errors_count == 0
        assert metrics.retries_count == 1

    def test_stage_metrics_is_frozen(self):
        """Test that StageMetrics instances are immutable."""
        from datetime import datetime

        metrics = StageMetrics(
            stage_id="test",
            start_time=datetime.now(),
        )
        with pytest.raises(ValidationError):
            metrics.thoughts_generated = 50  # type: ignore[misc]

    def test_stage_metrics_validation(self):
        """Test StageMetrics field validation."""
        from datetime import datetime

        # Valid: non-negative counts
        metrics = StageMetrics(
            stage_id="test",
            start_time=datetime.now(),
            thoughts_generated=0,
            errors_count=0,
            retries_count=0,
        )
        assert metrics.thoughts_generated >= 0
        assert metrics.retries_count >= 0

        # Invalid: negative duration
        with pytest.raises(ValidationError):
            StageMetrics(
                stage_id="test",
                start_time=datetime.now(),
                duration_seconds=-1.0,
            )

    def test_stage_metrics_serialization(self):
        """Test StageMetrics serialization to dict and JSON."""
        from datetime import datetime

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 5)
        metrics = StageMetrics(
            stage_id="test",
            start_time=start,
            end_time=end,
            duration_seconds=5.0,
            thoughts_generated=10,
            retries_count=2,
        )

        # To dict
        data = metrics.model_dump()
        assert data["stage_id"] == "test"
        assert data["duration_seconds"] == 5.0
        assert data["thoughts_generated"] == 10

        # From dict
        restored = StageMetrics(**data)
        assert restored.stage_id == metrics.stage_id


# ============================================================================
# TestStageTrace
# ============================================================================


class TestStageTrace:
    """Test suite for StageTrace model."""

    def test_create_stage_trace_minimal(self):
        """Test creating StageTrace with minimal required parameters."""
        trace = StageTrace(
            stage_id="stage_1",
            stage_type=PipelineStageType.METHOD,
            status="completed",
        )
        assert trace.stage_id == "stage_1"
        assert trace.stage_type == PipelineStageType.METHOD
        assert trace.status == "completed"
        assert trace.input_thought_ids == []
        assert trace.output_thought_ids == []
        assert trace.error is None
        assert trace.children == []

    def test_create_stage_trace_with_children(self):
        """Test creating StageTrace with child traces."""
        from datetime import datetime

        child = StageTrace(
            stage_id="child_1",
            stage_type=PipelineStageType.METHOD,
            status="completed",
            input_thought_ids=["thought-1"],
            output_thought_ids=["thought-2"],
        )
        parent = StageTrace(
            stage_id="parent",
            stage_type=PipelineStageType.SEQUENCE,
            status="completed",
            input_thought_ids=["thought-1"],
            output_thought_ids=["thought-2"],
            children=[child],
        )
        assert len(parent.children) == 1
        assert parent.children[0] == child

    def test_stage_trace_is_frozen(self):
        """Test that StageTrace instances are immutable."""
        trace = StageTrace(
            stage_id="test",
            stage_type=PipelineStageType.METHOD,
            status="completed",
        )
        with pytest.raises(ValidationError):
            trace.status = "failed"  # type: ignore[misc]

    def test_stage_trace_serialization(self):
        """Test StageTrace serialization to dict and JSON."""
        trace = StageTrace(
            stage_id="trace_1",
            stage_type=PipelineStageType.PARALLEL,
            status="completed",
            input_thought_ids=["thought-1"],
            output_thought_ids=["thought-2", "thought-3"],
        )

        # To dict
        data = trace.model_dump()
        assert data["stage_id"] == "trace_1"
        assert data["stage_type"] == PipelineStageType.PARALLEL
        assert data["status"] == "completed"

        # From dict
        restored = StageTrace(**data)
        assert restored.stage_id == trace.stage_id

    def test_stage_trace_with_error(self):
        """Test StageTrace with error information."""
        trace = StageTrace(
            stage_id="failed_stage",
            stage_type=PipelineStageType.METHOD,
            status="failed",
            error="Timeout exceeded",
        )
        assert trace.status == "failed"
        assert trace.error == "Timeout exceeded"


# ============================================================================
# TestPipelineTrace
# ============================================================================


class TestPipelineTrace:
    """Test suite for PipelineTrace model."""

    def test_create_pipeline_trace_minimal(self):
        """Test creating PipelineTrace with minimal required parameters."""
        from datetime import datetime

        trace = PipelineTrace(
            pipeline_id="pipeline_1",
            session_id="session_1",
            started_at=datetime.now(),
        )
        assert trace.pipeline_id == "pipeline_1"
        assert trace.session_id == "session_1"
        assert trace.root_trace is None

    def test_create_pipeline_trace_with_root(self):
        """Test creating PipelineTrace with root trace."""
        from datetime import datetime

        root = StageTrace(
            stage_id="root",
            stage_type=PipelineStageType.SEQUENCE,
            status="completed",
        )
        trace = PipelineTrace(
            pipeline_id="pipeline_1",
            session_id="session_1",
            started_at=datetime.now(),
            root_trace=root,
        )
        assert trace.root_trace == root

    def test_pipeline_trace_is_frozen(self):
        """Test that PipelineTrace instances are immutable."""
        from datetime import datetime

        trace = PipelineTrace(
            pipeline_id="test",
            session_id="session_1",
            started_at=datetime.now(),
        )
        with pytest.raises(ValidationError):
            trace.pipeline_id = "modified"  # type: ignore[misc]

    def test_pipeline_trace_serialization(self):
        """Test PipelineTrace serialization to dict and JSON."""
        from datetime import datetime

        root = StageTrace(
            stage_id="root",
            stage_type=PipelineStageType.METHOD,
            status="completed",
        )
        trace = PipelineTrace(
            pipeline_id="trace_test",
            session_id="session_1",
            started_at=datetime.now(),
            root_trace=root,
        )

        # To dict
        data = trace.model_dump()
        assert data["pipeline_id"] == "trace_test"
        assert data["root_trace"] is not None

        # From dict
        restored = PipelineTrace(**data)
        assert restored.pipeline_id == trace.pipeline_id

    def test_pipeline_trace_with_nested_stages(self):
        """Test PipelineTrace with nested stage hierarchy."""
        from datetime import datetime

        child1 = StageTrace(
            stage_id="child1",
            stage_type=PipelineStageType.METHOD,
            status="completed",
            input_thought_ids=["t1"],
            output_thought_ids=["t2"],
        )
        child2 = StageTrace(
            stage_id="child2",
            stage_type=PipelineStageType.METHOD,
            status="completed",
            input_thought_ids=["t2"],
            output_thought_ids=["t3"],
        )
        root = StageTrace(
            stage_id="root",
            stage_type=PipelineStageType.SEQUENCE,
            status="completed",
            input_thought_ids=["t1"],
            output_thought_ids=["t3"],
            children=[child1, child2],
        )
        trace = PipelineTrace(
            pipeline_id="nested",
            session_id="session_1",
            started_at=datetime.now(),
            root_trace=root,
        )

        assert len(trace.root_trace.children) == 2  # type: ignore[union-attr]


# ============================================================================
# TestPipelineResult
# ============================================================================


class TestPipelineResult:
    """Test suite for PipelineResult model."""

    def test_create_pipeline_result_success(self):
        """Test creating successful PipelineResult."""
        result = PipelineResult(
            pipeline_id="result_1",
            session_id="session_1",
            success=True,
            final_thoughts=["thought-1", "thought-2"],
        )
        assert result.pipeline_id == "result_1"
        assert result.session_id == "session_1"
        assert result.success is True
        assert result.final_thoughts == ["thought-1", "thought-2"]
        assert result.error is None

    def test_create_pipeline_result_failure(self):
        """Test creating failed PipelineResult."""
        result = PipelineResult(
            pipeline_id="failed_1",
            session_id="session_1",
            success=False,
            error="Stage timeout exceeded",
        )
        assert result.success is False
        assert result.error == "Stage timeout exceeded"

    def test_pipeline_result_with_trace(self):
        """Test PipelineResult with execution trace."""
        from datetime import datetime

        trace = PipelineTrace(
            pipeline_id="traced",
            session_id="session_1",
            started_at=datetime.now(),
        )
        result = PipelineResult(
            pipeline_id="traced",
            session_id="session_1",
            success=True,
            trace=trace,
        )
        assert result.trace == trace

    def test_pipeline_result_serialization(self):
        """Test PipelineResult serialization to dict and JSON."""
        result = PipelineResult(
            pipeline_id="serialize_test",
            session_id="session_1",
            success=True,
            final_thoughts=["thought-1"],
            metadata={"key": "value"},
        )

        # To dict
        data = result.model_dump()
        assert data["pipeline_id"] == "serialize_test"
        assert data["success"] is True
        assert data["metadata"] == {"key": "value"}

        # From dict
        restored = PipelineResult(**data)
        assert restored.pipeline_id == result.pipeline_id
        assert restored.success == result.success

    def test_pipeline_result_roundtrip(self):
        """Test PipelineResult JSON roundtrip."""
        from datetime import datetime

        stage_trace = StageTrace(
            stage_id="stage1",
            stage_type=PipelineStageType.METHOD,
            status="completed",
        )
        trace = PipelineTrace(
            pipeline_id="roundtrip",
            session_id="session_1",
            started_at=datetime.now(),
            root_trace=stage_trace,
        )
        original = PipelineResult(
            pipeline_id="roundtrip",
            session_id="session_1",
            success=True,
            final_thoughts=["thought-final"],
            trace=trace,
            metadata={"result": "value"},
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize from JSON
        data = json.loads(json_str)
        restored = PipelineResult(**data)

        assert restored.pipeline_id == original.pipeline_id
        assert restored.success == original.success
        assert restored.final_thoughts == original.final_thoughts

    def test_pipeline_result_is_frozen(self):
        """Test that PipelineResult instances are immutable."""
        result = PipelineResult(
            pipeline_id="test",
            session_id="session_1",
            success=True,
        )
        with pytest.raises(ValidationError):
            result.success = False  # type: ignore[misc]


# ============================================================================
# Integration Tests
# ============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline models."""

    def test_complex_pipeline_serialization(self):
        """Test serialization of a complex multi-level pipeline."""
        # Build a complex pipeline
        method1 = MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING)
        method2 = MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS)

        merge = MergeStrategy(name="vote", selection_criteria="majority")
        parallel = ParallelPipeline(
            branches=[method1, method2],
            merge_strategy=merge,
        )

        condition = Condition(name="check", expression="valid")
        conditional = ConditionalPipeline(
            condition=condition,
            if_true=parallel,
            if_false=method1,
        )

        sequence = SequencePipeline(stages=[conditional, method2])

        # Serialize
        data = sequence.model_dump()
        assert data["stage_type"] == PipelineStageType.SEQUENCE

        # Deserialize
        restored = SequencePipeline(**data)
        assert len(restored.stages) == 2

    def test_pipeline_with_all_stage_types(self):
        """Test pipeline containing all stage types."""
        merge = MergeStrategy(name="test")
        stages: list[Pipeline] = [
            MethodStage(method_id=MethodIdentifier.SEQUENTIAL_THINKING),
            SequencePipeline(),
            ParallelPipeline(
                branches=[MethodStage(method_id=MethodIdentifier.REACT)],
                merge_strategy=merge,
            ),
            ConditionalPipeline(
                condition=Condition(name="test", expression="true"),
                if_true=MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
            ),
            LoopPipeline(
                body=MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
                condition=Condition(name="test", expression="done"),
            ),
            SwitchPipeline(
                expression="x",
                cases={"a": MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)},
            ),
        ]

        pipeline = SequencePipeline(stages=stages)
        assert len(pipeline.stages) == 6

        # Verify each type
        assert isinstance(pipeline.stages[0], MethodStage)
        assert isinstance(pipeline.stages[1], SequencePipeline)
        assert isinstance(pipeline.stages[2], ParallelPipeline)
        assert isinstance(pipeline.stages[3], ConditionalPipeline)
        assert isinstance(pipeline.stages[4], LoopPipeline)
        assert isinstance(pipeline.stages[5], SwitchPipeline)
