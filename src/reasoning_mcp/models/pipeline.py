"""Pipeline DSL models for orchestrating reasoning methods.

This module defines the domain-specific language (DSL) for creating complex
reasoning workflows. Pipelines can combine multiple reasoning methods through
sequential, parallel, conditional, loop, and switch control flows.

The pipeline DSL enables:
- Sequential execution of multiple reasoning stages
- Parallel execution with result merging strategies
- Conditional branching based on thought properties
- Iterative loops with accumulation
- Switch/case routing based on expressions
- Data transformations between stages
- Error handling and retry strategies
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from reasoning_mcp.models.core import MethodIdentifier, PipelineStageType


# ============================================================================
# Helper Models (Frozen - Immutable Configuration)
# ============================================================================


class Transform(BaseModel):
    """Defines a transformation to apply to thoughts or intermediate results.

    Transforms enable data manipulation between pipeline stages, such as
    extracting specific fields, formatting content, or combining multiple
    values into new representations.

    Examples:
        Create a summarization transform:
        >>> transform = Transform(
        ...     name="summarize",
        ...     expression="{content} summarized in one sentence",
        ...     input_fields=["content"],
        ...     output_field="summary"
        ... )

        Create a confidence aggregation transform:
        >>> agg_transform = Transform(
        ...     name="avg_confidence",
        ...     expression="average({confidence})",
        ...     input_fields=["confidence"],
        ...     output_field="avg_confidence",
        ...     metadata={"aggregation": "mean"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="Transform name for identification and logging",
    )
    expression: str = Field(
        description="Transform expression or template (e.g., '{content} summarized')",
    )
    input_fields: list[str] = Field(
        default_factory=list,
        description="Fields to use as input to the transformation",
    )
    output_field: str = Field(
        default="transformed_content",
        description="Field to store the transformation result",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transform configuration (e.g., aggregation functions)",
    )


class Condition(BaseModel):
    """Defines a condition for branching or looping decisions.

    Conditions evaluate thought properties or intermediate results to control
    pipeline execution flow. Common use cases include confidence thresholds,
    iteration limits, and validation checks.

    Examples:
        Create a confidence threshold condition:
        >>> condition = Condition(
        ...     name="high_confidence",
        ...     expression="confidence > 0.8",
        ...     operator=">",
        ...     threshold=0.8,
        ...     field="confidence"
        ... )

        Create a validation condition:
        >>> valid_condition = Condition(
        ...     name="is_valid",
        ...     expression="is_valid == True",
        ...     operator="==",
        ...     field="is_valid"
        ... )

        Create a custom expression condition:
        >>> custom_condition = Condition(
        ...     name="quality_check",
        ...     expression="quality_score > 0.7 and confidence > 0.6",
        ...     metadata={"description": "Combined quality and confidence check"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="Condition name for identification and logging",
    )
    expression: str = Field(
        description="Condition expression (e.g., 'confidence > 0.8', 'is_valid == True')",
    )
    operator: str = Field(
        default="==",
        description="Comparison operator for simple conditions (e.g., '>', '<', '==', '!=', '>=', '<=')",
    )
    threshold: float | None = Field(
        default=None,
        description="Optional threshold value for numeric comparisons",
    )
    field: str | None = Field(
        default=None,
        description="Field to evaluate in simple conditions",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional condition configuration",
    )


class MergeStrategy(BaseModel):
    """Defines how to merge results from parallel pipeline branches.

    Merge strategies control how outputs from concurrent reasoning paths are
    combined into a single result. Common strategies include voting, selecting
    the best result, or aggregating multiple outputs.

    Examples:
        Create a voting merge strategy:
        >>> vote_merge = MergeStrategy(
        ...     name="vote",
        ...     selection_criteria="most_common_conclusion",
        ...     metadata={"min_agreement": 0.6}
        ... )

        Create a best-result merge strategy:
        >>> best_merge = MergeStrategy(
        ...     name="best",
        ...     selection_criteria="highest_confidence",
        ...     weights={"branch_a": 1.0, "branch_b": 0.8}
        ... )

        Create an aggregation merge strategy:
        >>> agg_merge = MergeStrategy(
        ...     name="combine",
        ...     selection_criteria="all_results",
        ...     aggregation="concatenate",
        ...     metadata={"separator": "\\n---\\n"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="Strategy name (e.g., 'vote', 'best', 'combine', 'average')",
    )
    selection_criteria: str = Field(
        default="highest_confidence",
        description="How to select or combine results (e.g., 'highest_confidence', 'most_common', 'all_results')",
    )
    weights: dict[str, float] = Field(
        default_factory=dict,
        description="Optional weights for different branches in weighted merging",
    )
    aggregation: str | None = Field(
        default=None,
        description="Aggregation function if combining multiple results (e.g., 'concatenate', 'average', 'max')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional merge configuration",
    )


class Accumulator(BaseModel):
    """Defines accumulation strategy for loop iterations.

    Accumulators collect and combine results across loop iterations, enabling
    patterns like building up evidence, tracking state changes, or aggregating
    metrics over multiple reasoning steps.

    Examples:
        Create a list accumulator:
        >>> list_acc = Accumulator(
        ...     name="evidence_collector",
        ...     initial_value=[],
        ...     operation="append",
        ...     field="content"
        ... )

        Create a numeric accumulator:
        >>> sum_acc = Accumulator(
        ...     name="confidence_sum",
        ...     initial_value=0.0,
        ...     operation="sum",
        ...     field="confidence",
        ...     metadata={"final_operation": "average"}
        ... )

        Create a dictionary accumulator:
        >>> dict_acc = Accumulator(
        ...     name="metrics_tracker",
        ...     initial_value={},
        ...     operation="merge",
        ...     field="metadata",
        ...     metadata={"merge_strategy": "update"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description="Accumulator name for identification",
    )
    initial_value: Any = Field(
        default=None,
        description="Starting value for the accumulator (e.g., [], 0, {})",
    )
    operation: str = Field(
        default="append",
        description="Accumulation operation (e.g., 'append', 'sum', 'merge', 'max', 'min')",
    )
    field: str = Field(
        default="content",
        description="Field to accumulate from each iteration",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional accumulator configuration",
    )


class ErrorHandler(BaseModel):
    """Defines error handling strategy for pipeline stages.

    Error handlers control how failures are managed during pipeline execution,
    including retry logic, fallback methods, and failure policies.

    Examples:
        Create a retry handler:
        >>> retry_handler = ErrorHandler(
        ...     strategy="retry",
        ...     max_retries=3,
        ...     on_failure="raise",
        ...     metadata={"backoff": "exponential"}
        ... )

        Create a fallback handler:
        >>> fallback_handler = ErrorHandler(
        ...     strategy="fallback",
        ...     fallback_method=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     on_failure="skip",
        ...     metadata={"reason": "Use simpler method on failure"}
        ... )

        Create a skip handler:
        >>> skip_handler = ErrorHandler(
        ...     strategy="skip",
        ...     max_retries=0,
        ...     on_failure="continue",
        ...     metadata={"log_error": True}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(
        description="Handler strategy (e.g., 'retry', 'skip', 'fallback', 'raise')",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts before giving up",
    )
    fallback_method: MethodIdentifier | None = Field(
        default=None,
        description="Alternative reasoning method to use if primary fails",
    )
    on_failure: str = Field(
        default="raise",
        description="Action on final failure (e.g., 'raise', 'skip', 'continue')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error handling configuration",
    )


# ============================================================================
# Pipeline Stage Models (Mutable - For Building Workflows)
# ============================================================================


class MethodStage(BaseModel):
    """A single reasoning method execution stage.

    MethodStage represents the atomic unit of pipeline execution - invoking
    a single reasoning method with specific configuration. This is the basic
    building block for more complex pipeline compositions.

    Examples:
        Create a simple method stage:
        >>> stage = MethodStage(
        ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...     name="initial_analysis",
        ...     max_thoughts=15
        ... )

        Create a stage with transformations:
        >>> stage_with_transform = MethodStage(
        ...     method_id=MethodIdentifier.SELF_CONSISTENCY,
        ...     name="multi_path_reasoning",
        ...     max_thoughts=20,
        ...     transforms=[
        ...         Transform(
        ...             name="extract_conclusion",
        ...             expression="{conclusion}",
        ...             input_fields=["content"],
        ...             output_field="final_answer"
        ...         )
        ...     ],
        ...     timeout_seconds=120.0
        ... )

        Create a stage with error handling:
        >>> stage_with_handler = MethodStage(
        ...     method_id=MethodIdentifier.TREE_OF_THOUGHTS,
        ...     name="exploratory_search",
        ...     max_thoughts=30,
        ...     error_handler=ErrorHandler(
        ...         strategy="fallback",
        ...         fallback_method=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...         max_retries=2
        ...     )
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this stage (UUID)",
    )
    stage_type: Literal[PipelineStageType.METHOD] = Field(
        default=PipelineStageType.METHOD,
        description="Stage type discriminator for union types",
    )
    method_id: MethodIdentifier = Field(
        description="Reasoning method to execute in this stage",
    )
    name: str | None = Field(
        default=None,
        description="Optional human-readable stage name",
    )
    description: str | None = Field(
        default=None,
        description="Optional stage description explaining its purpose",
    )
    max_thoughts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of thoughts this stage can generate",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Maximum time allowed for stage execution in seconds",
    )
    transforms: list[Transform] = Field(
        default_factory=list,
        description="Transformations to apply to stage outputs",
    )
    error_handler: ErrorHandler | None = Field(
        default=None,
        description="Error handling strategy for this stage",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional stage configuration and context",
    )


class SequencePipeline(BaseModel):
    """Sequential execution of multiple pipeline stages.

    SequencePipeline executes a list of stages in order, passing the output
    of each stage as input to the next. This is the most common pipeline
    pattern for multi-step reasoning workflows.

    Examples:
        Create a simple sequence:
        >>> pipeline = SequencePipeline(
        ...     name="analysis_workflow",
        ...     stages=[
        ...         MethodStage(
        ...             method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...             name="initial_analysis"
        ...         ),
        ...         MethodStage(
        ...             method_id=MethodIdentifier.SELF_REFLECTION,
        ...             name="critique"
        ...         ),
        ...         MethodStage(
        ...             method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...             name="refinement"
        ...         )
        ...     ]
        ... )

        Create a sequence with error stopping:
        >>> strict_pipeline = SequencePipeline(
        ...     name="strict_validation",
        ...     stages=[
        ...         MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
        ...         MethodStage(method_id=MethodIdentifier.CODE_REASONING)
        ...     ],
        ...     stop_on_error=True,
        ...     metadata={"require_all_stages": True}
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline (UUID)",
    )
    stage_type: Literal[PipelineStageType.SEQUENCE] = Field(
        default=PipelineStageType.SEQUENCE,
        description="Stage type discriminator for union types",
    )
    name: str | None = Field(
        default=None,
        description="Optional pipeline name",
    )
    stages: list[Pipeline] = Field(
        default_factory=list,
        description="Ordered list of stages to execute sequentially",
    )
    stop_on_error: bool = Field(
        default=True,
        description="Whether to halt execution if any stage fails",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline configuration",
    )


class ParallelPipeline(BaseModel):
    """Parallel execution of multiple pipeline branches with result merging.

    ParallelPipeline executes multiple reasoning paths concurrently and then
    merges their results using a specified strategy. This enables exploring
    multiple approaches simultaneously and combining their insights.

    Examples:
        Create a parallel exploration pipeline:
        >>> parallel = ParallelPipeline(
        ...     name="multi_perspective_analysis",
        ...     branches=[
        ...         MethodStage(
        ...             method_id=MethodIdentifier.ETHICAL_REASONING,
        ...             name="ethical_view"
        ...         ),
        ...         MethodStage(
        ...             method_id=MethodIdentifier.CAUSAL_REASONING,
        ...             name="causal_view"
        ...         ),
        ...         MethodStage(
        ...             method_id=MethodIdentifier.COUNTERFACTUAL,
        ...             name="alternative_view"
        ...         )
        ...     ],
        ...     merge_strategy=MergeStrategy(
        ...         name="combine_perspectives",
        ...         selection_criteria="all_results",
        ...         aggregation="synthesize"
        ...     ),
        ...     max_concurrency=3
        ... )

        Create a voting-based parallel pipeline:
        >>> voting = ParallelPipeline(
        ...     name="consensus_builder",
        ...     branches=[
        ...         MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT),
        ...         MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
        ...         MethodStage(method_id=MethodIdentifier.SELF_CONSISTENCY)
        ...     ],
        ...     merge_strategy=MergeStrategy(
        ...         name="vote",
        ...         selection_criteria="most_common_conclusion"
        ...     ),
        ...     timeout_seconds=180.0
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline (UUID)",
    )
    stage_type: Literal[PipelineStageType.PARALLEL] = Field(
        default=PipelineStageType.PARALLEL,
        description="Stage type discriminator for union types",
    )
    name: str | None = Field(
        default=None,
        description="Optional pipeline name",
    )
    branches: list[Pipeline] = Field(
        default_factory=list,
        description="List of pipeline branches to execute in parallel",
    )
    merge_strategy: MergeStrategy = Field(
        description="Strategy for merging results from all branches",
    )
    max_concurrency: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of branches to run concurrently",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Total timeout for all parallel branches",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline configuration",
    )


class ConditionalPipeline(BaseModel):
    """Conditional execution based on thought properties or results.

    ConditionalPipeline evaluates a condition and executes different pipeline
    branches based on the result. This enables adaptive reasoning workflows
    that respond to intermediate results.

    Examples:
        Create a confidence-based conditional:
        >>> conditional = ConditionalPipeline(
        ...     name="adaptive_refinement",
        ...     condition=Condition(
        ...         name="high_confidence",
        ...         expression="confidence > 0.8",
        ...         operator=">",
        ...         threshold=0.8,
        ...         field="confidence"
        ...     ),
        ...     if_true=MethodStage(
        ...         method_id=MethodIdentifier.SELF_REFLECTION,
        ...         name="light_review"
        ...     ),
        ...     if_false=SequencePipeline(
        ...         name="deep_refinement",
        ...         stages=[
        ...             MethodStage(method_id=MethodIdentifier.SOCRATIC),
        ...             MethodStage(method_id=MethodIdentifier.DIALECTIC),
        ...             MethodStage(method_id=MethodIdentifier.SELF_REFLECTION)
        ...         ]
        ...     )
        ... )

        Create a validation conditional:
        >>> validation = ConditionalPipeline(
        ...     name="error_check",
        ...     condition=Condition(
        ...         name="is_valid",
        ...         expression="is_valid == True",
        ...         field="is_valid"
        ...     ),
        ...     if_true=MethodStage(
        ...         method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...         name="continue_reasoning"
        ...     ),
        ...     if_false=None,  # Skip if invalid
        ...     metadata={"skip_on_invalid": True}
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline (UUID)",
    )
    stage_type: Literal[PipelineStageType.CONDITIONAL] = Field(
        default=PipelineStageType.CONDITIONAL,
        description="Stage type discriminator for union types",
    )
    name: str | None = Field(
        default=None,
        description="Optional pipeline name",
    )
    condition: Condition = Field(
        description="Condition to evaluate for branching decision",
    )
    if_true: Pipeline = Field(
        description="Pipeline to execute if condition evaluates to true",
    )
    if_false: Pipeline | None = Field(
        default=None,
        description="Optional pipeline to execute if condition evaluates to false",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline configuration",
    )


class LoopPipeline(BaseModel):
    """Iterative loop execution until a condition is met.

    LoopPipeline repeatedly executes a pipeline body while a condition holds,
    optionally accumulating results across iterations. This enables iterative
    refinement, exploration, and convergence patterns.

    Examples:
        Create a refinement loop:
        >>> refinement_loop = LoopPipeline(
        ...     name="iterative_improvement",
        ...     body=SequencePipeline(
        ...         stages=[
        ...             MethodStage(method_id=MethodIdentifier.SELF_REFLECTION),
        ...             MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        ...         ]
        ...     ),
        ...     condition=Condition(
        ...         name="quality_threshold",
        ...         expression="quality_score > 0.9",
        ...         operator=">",
        ...         threshold=0.9,
        ...         field="quality_score"
        ...     ),
        ...     max_iterations=5,
        ...     accumulator=Accumulator(
        ...         name="improvement_history",
        ...         initial_value=[],
        ...         operation="append",
        ...         field="content"
        ...     )
        ... )

        Create a convergence loop:
        >>> convergence_loop = LoopPipeline(
        ...     name="consensus_search",
        ...     body=MethodStage(
        ...         method_id=MethodIdentifier.SELF_CONSISTENCY
        ...     ),
        ...     condition=Condition(
        ...         name="high_agreement",
        ...         expression="agreement > 0.8",
        ...         field="agreement"
        ...     ),
        ...     max_iterations=10,
        ...     metadata={"convergence_metric": "agreement"}
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline (UUID)",
    )
    stage_type: Literal[PipelineStageType.LOOP] = Field(
        default=PipelineStageType.LOOP,
        description="Stage type discriminator for union types",
    )
    name: str | None = Field(
        default=None,
        description="Optional pipeline name",
    )
    body: Pipeline = Field(
        description="Pipeline to execute in each loop iteration",
    )
    condition: Condition = Field(
        description="Loop continuation condition (loop continues while true)",
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of loop iterations to prevent infinite loops",
    )
    accumulator: Accumulator | None = Field(
        default=None,
        description="Optional accumulator for collecting results across iterations",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline configuration",
    )


class SwitchPipeline(BaseModel):
    """Switch/case routing based on expression evaluation.

    SwitchPipeline evaluates an expression and routes execution to one of
    multiple cases based on the result. This enables multi-way branching
    for method selection or workflow routing.

    Examples:
        Create a problem-type router:
        >>> router = SwitchPipeline(
        ...     name="problem_type_router",
        ...     expression="problem_type",
        ...     cases={
        ...         "ethical": MethodStage(
        ...             method_id=MethodIdentifier.ETHICAL_REASONING
        ...         ),
        ...         "mathematical": MethodStage(
        ...             method_id=MethodIdentifier.MATHEMATICAL_REASONING
        ...         ),
        ...         "code": MethodStage(
        ...             method_id=MethodIdentifier.CODE_REASONING
        ...         ),
        ...         "creative": MethodStage(
        ...             method_id=MethodIdentifier.LATERAL_THINKING
        ...         )
        ...     },
        ...     default=MethodStage(
        ...         method_id=MethodIdentifier.CHAIN_OF_THOUGHT
        ...     ),
        ...     metadata={"routing_field": "problem_type"}
        ... )

        Create a complexity-based router:
        >>> complexity_router = SwitchPipeline(
        ...     name="complexity_router",
        ...     expression="complexity_level",
        ...     cases={
        ...         "simple": MethodStage(
        ...             method_id=MethodIdentifier.CHAIN_OF_THOUGHT
        ...         ),
        ...         "moderate": SequencePipeline(
        ...             stages=[
        ...                 MethodStage(method_id=MethodIdentifier.STEP_BACK),
        ...                 MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        ...             ]
        ...         ),
        ...         "complex": ParallelPipeline(
        ...             branches=[
        ...                 MethodStage(method_id=MethodIdentifier.TREE_OF_THOUGHTS),
        ...                 MethodStage(method_id=MethodIdentifier.DECOMPOSED_PROMPTING)
        ...             ],
        ...             merge_strategy=MergeStrategy(
        ...                 name="best",
        ...                 selection_criteria="highest_confidence"
        ...             )
        ...         )
        ...     },
        ...     default=None
        ... )
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this pipeline (UUID)",
    )
    stage_type: Literal[PipelineStageType.SWITCH] = Field(
        default=PipelineStageType.SWITCH,
        description="Stage type discriminator for union types",
    )
    name: str | None = Field(
        default=None,
        description="Optional pipeline name",
    )
    expression: str = Field(
        description="Expression or field to evaluate for case selection",
    )
    cases: dict[str, Pipeline] = Field(
        description="Mapping of case values to their respective pipelines",
    )
    default: Pipeline | None = Field(
        default=None,
        description="Optional default pipeline if no case matches",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline configuration",
    )


# ============================================================================
# Pipeline Union Type
# ============================================================================

Pipeline = Annotated[
    Union[
        MethodStage,
        SequencePipeline,
        ParallelPipeline,
        ConditionalPipeline,
        LoopPipeline,
        SwitchPipeline,
    ],
    Field(discriminator="stage_type"),
]
"""Discriminated union of all pipeline stage types.

The Pipeline type enables composing complex reasoning workflows from multiple
stage types. The discriminator field 'stage_type' allows Pydantic to correctly
deserialize the appropriate concrete type.

Examples:
    Type hints for pipeline functions:
    >>> def execute_pipeline(pipeline: Pipeline) -> PipelineResult:
    ...     match pipeline.stage_type:
    ...         case PipelineStageType.METHOD:
    ...             return execute_method(pipeline)
    ...         case PipelineStageType.SEQUENCE:
    ...             return execute_sequence(pipeline)
    ...         # ... handle other types
"""

# Rebuild models for forward references
SequencePipeline.model_rebuild()
ParallelPipeline.model_rebuild()
ConditionalPipeline.model_rebuild()
LoopPipeline.model_rebuild()
SwitchPipeline.model_rebuild()


# ============================================================================
# Result and Trace Models (Frozen - Immutable Execution Records)
# ============================================================================


class StageMetrics(BaseModel):
    """Execution metrics for a single pipeline stage.

    StageMetrics captures performance and outcome data for individual stage
    executions, enabling analysis, optimization, and debugging of pipelines.

    Examples:
        Create metrics for a completed stage:
        >>> from datetime import datetime, timedelta
        >>> start = datetime.now()
        >>> end = start + timedelta(seconds=5.5)
        >>> metrics = StageMetrics(
        ...     stage_id="stage-123",
        ...     start_time=start,
        ...     end_time=end,
        ...     duration_seconds=5.5,
        ...     thoughts_generated=12,
        ...     errors_count=0,
        ...     retries_count=0,
        ...     metadata={"avg_confidence": 0.85}
        ... )

        Create metrics for a failed stage:
        >>> failed_metrics = StageMetrics(
        ...     stage_id="stage-456",
        ...     start_time=start,
        ...     end_time=end,
        ...     duration_seconds=2.1,
        ...     thoughts_generated=3,
        ...     errors_count=1,
        ...     retries_count=2,
        ...     metadata={"error_type": "timeout"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    stage_id: str = Field(
        description="ID of the stage these metrics apply to",
    )
    start_time: datetime = Field(
        description="Timestamp when stage execution started",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Timestamp when stage execution ended (None if still running)",
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total execution time in seconds",
    )
    thoughts_generated: int = Field(
        default=0,
        ge=0,
        description="Number of thoughts generated during stage execution",
    )
    errors_count: int = Field(
        default=0,
        ge=0,
        description="Number of errors encountered during execution",
    )
    retries_count: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts made",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metrics and performance data",
    )


class StageTrace(BaseModel):
    """Execution trace for a single pipeline stage.

    StageTrace provides a hierarchical view of pipeline execution, tracking
    inputs, outputs, status, and nested stage executions. This enables
    comprehensive debugging and understanding of pipeline behavior.

    Examples:
        Create a trace for a successful stage:
        >>> trace = StageTrace(
        ...     stage_id="stage-123",
        ...     stage_type=PipelineStageType.METHOD,
        ...     status="completed",
        ...     input_thought_ids=["thought-1", "thought-2"],
        ...     output_thought_ids=["thought-3", "thought-4", "thought-5"],
        ...     metrics=StageMetrics(
        ...         stage_id="stage-123",
        ...         start_time=datetime.now(),
        ...         duration_seconds=3.5,
        ...         thoughts_generated=3
        ...     )
        ... )

        Create a trace for a failed stage:
        >>> failed_trace = StageTrace(
        ...     stage_id="stage-456",
        ...     stage_type=PipelineStageType.CONDITIONAL,
        ...     status="failed",
        ...     input_thought_ids=["thought-1"],
        ...     output_thought_ids=[],
        ...     error="Condition evaluation failed: missing field 'confidence'",
        ...     metadata={"condition": "confidence > 0.8"}
        ... )

        Create a trace with children:
        >>> parent_trace = StageTrace(
        ...     stage_id="sequence-789",
        ...     stage_type=PipelineStageType.SEQUENCE,
        ...     status="completed",
        ...     input_thought_ids=["thought-1"],
        ...     output_thought_ids=["thought-10"],
        ...     children=[
        ...         StageTrace(
        ...             stage_id="child-1",
        ...             stage_type=PipelineStageType.METHOD,
        ...             status="completed",
        ...             input_thought_ids=["thought-1"],
        ...             output_thought_ids=["thought-5"]
        ...         ),
        ...         StageTrace(
        ...             stage_id="child-2",
        ...             stage_type=PipelineStageType.METHOD,
        ...             status="completed",
        ...             input_thought_ids=["thought-5"],
        ...             output_thought_ids=["thought-10"]
        ...         )
        ...     ]
        ... )
    """

    model_config = ConfigDict(frozen=True)

    stage_id: str = Field(
        description="ID of the stage this trace represents",
    )
    stage_type: PipelineStageType = Field(
        description="Type of pipeline stage",
    )
    status: str = Field(
        description="Execution status (e.g., 'pending', 'running', 'completed', 'failed')",
    )
    input_thought_ids: list[str] = Field(
        default_factory=list,
        description="IDs of thoughts provided as input to this stage",
    )
    output_thought_ids: list[str] = Field(
        default_factory=list,
        description="IDs of thoughts produced by this stage",
    )
    metrics: StageMetrics | None = Field(
        default=None,
        description="Performance metrics for this stage execution",
    )
    error: str | None = Field(
        default=None,
        description="Error message if stage execution failed",
    )
    children: list[StageTrace] = Field(
        default_factory=list,
        description="Traces of child stages (for composite stages like sequences)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional trace data and context",
    )


class PipelineTrace(BaseModel):
    """Complete execution trace for an entire pipeline.

    PipelineTrace captures the full execution history of a pipeline run,
    including all stage executions, timing, and outcomes. This provides
    a comprehensive audit trail for pipeline executions.

    Examples:
        Create a trace for a running pipeline:
        >>> trace = PipelineTrace(
        ...     pipeline_id="pipeline-abc",
        ...     session_id="session-123",
        ...     started_at=datetime.now(),
        ...     status="running",
        ...     root_trace=StageTrace(
        ...         stage_id="root",
        ...         stage_type=PipelineStageType.SEQUENCE,
        ...         status="running",
        ...         input_thought_ids=["initial-thought"]
        ...     )
        ... )

        Create a trace for a completed pipeline:
        >>> completed_trace = PipelineTrace(
        ...     pipeline_id="pipeline-abc",
        ...     session_id="session-123",
        ...     started_at=datetime.now(),
        ...     completed_at=datetime.now(),
        ...     status="completed",
        ...     root_trace=StageTrace(
        ...         stage_id="root",
        ...         stage_type=PipelineStageType.SEQUENCE,
        ...         status="completed",
        ...         input_thought_ids=["initial-thought"],
        ...         output_thought_ids=["final-thought"],
        ...         children=[...]
        ...     ),
        ...     metadata={"total_thoughts": 45, "total_duration": 15.7}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    pipeline_id: str = Field(
        description="ID of the pipeline being executed",
    )
    session_id: str = Field(
        description="Session ID for this pipeline execution",
    )
    started_at: datetime = Field(
        description="Timestamp when pipeline execution began",
    )
    completed_at: datetime | None = Field(
        default=None,
        description="Timestamp when pipeline execution completed (None if still running)",
    )
    status: str = Field(
        default="pending",
        description="Overall pipeline execution status",
    )
    root_trace: StageTrace | None = Field(
        default=None,
        description="Root stage trace containing the full execution tree",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional trace metadata and summary statistics",
    )


class PipelineResult(BaseModel):
    """Final result of pipeline execution.

    PipelineResult summarizes the outcome of a complete pipeline run,
    including success status, final outputs, execution trace, and any
    errors encountered.

    Examples:
        Create a successful result:
        >>> result = PipelineResult(
        ...     pipeline_id="pipeline-abc",
        ...     session_id="session-123",
        ...     success=True,
        ...     final_thoughts=["thought-final-1", "thought-final-2"],
        ...     trace=PipelineTrace(
        ...         pipeline_id="pipeline-abc",
        ...         session_id="session-123",
        ...         started_at=datetime.now(),
        ...         completed_at=datetime.now(),
        ...         status="completed"
        ...     ),
        ...     metadata={"execution_time": 12.3, "thoughts_generated": 42}
        ... )

        Create a failed result:
        >>> failed_result = PipelineResult(
        ...     pipeline_id="pipeline-def",
        ...     session_id="session-456",
        ...     success=False,
        ...     final_thoughts=[],
        ...     error="Stage 'ethical_analysis' failed: timeout exceeded",
        ...     trace=PipelineTrace(
        ...         pipeline_id="pipeline-def",
        ...         session_id="session-456",
        ...         started_at=datetime.now(),
        ...         status="failed"
        ...     ),
        ...     metadata={"failed_stage": "ethical_analysis", "error_type": "timeout"}
        ... )
    """

    model_config = ConfigDict(frozen=True)

    pipeline_id: str = Field(
        description="ID of the executed pipeline",
    )
    session_id: str = Field(
        description="Session ID for this execution",
    )
    success: bool = Field(
        description="Whether the pipeline executed successfully",
    )
    final_thoughts: list[str] = Field(
        default_factory=list,
        description="IDs of final output thoughts from the pipeline",
    )
    trace: PipelineTrace | None = Field(
        default=None,
        description="Complete execution trace for debugging and analysis",
    )
    error: str | None = Field(
        default=None,
        description="Error message if pipeline execution failed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata and summary information",
    )
