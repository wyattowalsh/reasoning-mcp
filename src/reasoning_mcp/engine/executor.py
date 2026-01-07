"""Pipeline execution engine for reasoning-mcp.

This module defines the core execution framework for pipeline stages, including:
- Abstract base class for all pipeline executors
- Execution context and result models
- Executor registry for looking up executors by stage type

The executor framework enables pluggable execution strategies for different
pipeline stage types (method, sequence, parallel, conditional, loop, switch).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import Pipeline, PipelineTrace, StageMetrics, StageTrace
from reasoning_mcp.models.session import Session
from reasoning_mcp.registry import MethodRegistry


# ============================================================================
# Execution Context and Results
# ============================================================================


@dataclass
class ExecutionContext:
    """Context passed to pipeline stage executors during execution.

    ExecutionContext encapsulates all the state and dependencies needed for
    executing a pipeline stage, including the session, registry, input data,
    variables, and execution trace.

    Attributes:
        session: Current reasoning session containing the thought graph
        registry: Method registry for looking up reasoning methods
        input_data: Input data from the previous pipeline stage
        variables: Pipeline-level variables for sharing state across stages
        trace: Execution trace for recording stage executions
        thought_ids: IDs of thoughts to pass to the next stage
        metadata: Additional execution metadata

    Examples:
        Create an execution context:
        >>> from datetime import datetime
        >>> session = Session().start()
        >>> registry = MethodRegistry()
        >>> trace = PipelineTrace(
        ...     pipeline_id="pipeline-123",
        ...     session_id=session.id,
        ...     started_at=datetime.now(),
        ...     status="running"
        ... )
        >>> context = ExecutionContext(
        ...     session=session,
        ...     registry=registry,
        ...     input_data={"query": "What is reasoning?"},
        ...     variables={"max_depth": 5},
        ...     trace=trace
        ... )
        >>> assert context.session.id == session.id
        >>> assert context.input_data["query"] == "What is reasoning?"
        >>> assert context.variables["max_depth"] == 5
    """

    session: Session
    registry: MethodRegistry
    input_data: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    trace: PipelineTrace | None = None
    thought_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_update(
        self,
        *,
        session: Session | None = None,
        registry: "MethodRegistry | None" = None,
        input_data: dict[str, Any] | None = None,
        variables: dict[str, Any] | None = None,
        trace: PipelineTrace | None = None,
        thought_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ExecutionContext":
        """Create a copy of this context with updated values.

        Only specified parameters are updated; others retain their original values.

        Args:
            session: New session (optional)
            registry: New registry (optional)
            input_data: New input data (optional)
            variables: New variables (optional)
            trace: New trace (optional)
            thought_ids: New thought IDs (optional)
            metadata: New metadata (optional)

        Returns:
            New ExecutionContext with updated values

        Examples:
            >>> new_context = context.with_update(
            ...     variables={"iteration": 1},
            ...     thought_ids=["thought-1", "thought-2"]
            ... )
        """
        return ExecutionContext(
            session=session if session is not None else self.session,
            registry=registry if registry is not None else self.registry,
            input_data=input_data if input_data is not None else self.input_data,
            variables=variables if variables is not None else self.variables,
            trace=trace if trace is not None else self.trace,
            thought_ids=thought_ids if thought_ids is not None else self.thought_ids,
            metadata=metadata if metadata is not None else self.metadata,
        )


@dataclass
class StageResult:
    """Result from executing a pipeline stage.

    StageResult captures the outcome of a stage execution, including success
    status, output data, performance metrics, error information, and whether
    the pipeline should continue executing.

    Attributes:
        stage_id: ID of the stage that produced this result
        stage_type: Type of the pipeline stage
        success: Whether the stage executed successfully
        output_thought_ids: IDs of thoughts generated by this stage
        output_data: Output data produced by the stage
        trace: Execution trace for this stage
        error: Error message if the stage failed (None if successful)
        metadata: Additional result metadata
        should_continue: Whether pipeline execution should continue to next stage

    Examples:
        Create a successful result:
        >>> result = StageResult(
        ...     stage_id="stage-123",
        ...     stage_type=PipelineStageType.METHOD,
        ...     success=True,
        ...     output_thought_ids=["t1", "t2", "t3"],
        ...     output_data={"thoughts": ["t1", "t2", "t3"]},
        ... )
        >>> assert result.success is True
        >>> assert result.should_continue is True
        >>> assert result.error is None

        Create a failed result:
        >>> failed_result = StageResult(
        ...     stage_id="stage-123",
        ...     stage_type=PipelineStageType.METHOD,
        ...     success=False,
        ...     error="Method not found: invalid_method",
        ... )
        >>> assert failed_result.success is False
        >>> assert failed_result.error is not None
    """

    stage_id: str
    stage_type: PipelineStageType
    success: bool
    output_thought_ids: list[str] = field(default_factory=list)
    output_data: dict[str, Any] = field(default_factory=dict)
    trace: "PipelineTrace | None" = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True


# ============================================================================
# Pipeline Executor Base Class
# ============================================================================


class PipelineExecutor(ABC):
    """Abstract base class for all pipeline stage executors.

    PipelineExecutor defines the interface that all stage executors must
    implement. Each executor is responsible for executing a specific type
    of pipeline stage (method, sequence, parallel, conditional, loop, switch).

    Executors are registered in the ExecutorRegistry and looked up by stage
    type during pipeline execution.

    Examples:
        Implement a custom executor:
        >>> class MyMethodExecutor(PipelineExecutor):
        ...     async def execute(self, context: ExecutionContext) -> StageResult:
        ...         # Execute the stage logic
        ...         return StageResult(success=True, output={})
        ...
        ...     async def validate(self, stage: Pipeline) -> list[str]:
        ...         # Validate the stage configuration
        ...         errors = []
        ...         if not stage.name:
        ...             errors.append("Stage name is required")
        ...         return errors

        Register the executor:
        >>> registry = ExecutorRegistry()
        >>> registry.register(PipelineStageType.METHOD, MyMethodExecutor())
    """

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the pipeline stage with the given context.

        This method contains the core execution logic for the stage. It receives
        an ExecutionContext containing all necessary state and dependencies,
        and returns a StageResult with the execution outcome.

        Args:
            context: Execution context containing session, registry, and input data

        Returns:
            StageResult with execution outcome, output data, and metrics

        Raises:
            Exception: Implementation-specific exceptions may be raised

        Examples:
            >>> executor = MyMethodExecutor()
            >>> context = ExecutionContext(
            ...     session=Session().start(),
            ...     registry=MethodRegistry(),
            ...     input_data={"query": "Test"}
            ... )
            >>> result = await executor.execute(context)
            >>> assert isinstance(result, StageResult)
        """
        ...

    @abstractmethod
    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the pipeline stage configuration.

        This method checks the stage configuration for errors and returns
        a list of validation error messages. An empty list indicates the
        stage is valid.

        Validation should check for:
        - Required fields are present
        - Field values are within valid ranges
        - Referenced methods/stages exist
        - Configuration is internally consistent

        Args:
            stage: Pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)

        Examples:
            >>> executor = MyMethodExecutor()
            >>> from reasoning_mcp.models.pipeline import MethodStage
            >>> from reasoning_mcp.models.core import MethodIdentifier
            >>> stage = MethodStage(
            ...     method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            ...     name="test_stage"
            ... )
            >>> errors = await executor.validate(stage)
            >>> assert isinstance(errors, list)
            >>> if errors:
            ...     print(f"Validation failed: {errors}")
        """
        ...

    def create_metrics(
        self,
        stage_id: str,
        start_time: datetime,
        end_time: datetime | None = None,
        thoughts_generated: int = 0,
        errors_count: int = 0,
        retries_count: int = 0,
        **extra_metadata: Any,
    ) -> StageMetrics:
        """Create execution metrics for a pipeline stage.

        Helper method to construct StageMetrics with calculated duration.

        Args:
            stage_id: ID of the stage
            start_time: When execution started
            end_time: When execution ended (default: now)
            thoughts_generated: Number of thoughts generated
            errors_count: Number of errors encountered
            retries_count: Number of retry attempts
            **extra_metadata: Additional metadata to include

        Returns:
            StageMetrics instance with calculated duration

        Examples:
            >>> metrics = executor.create_metrics(
            ...     stage_id="stage-123",
            ...     start_time=start_time,
            ...     end_time=end_time,
            ...     thoughts_generated=5
            ... )
        """
        if end_time is None:
            end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        return StageMetrics(
            stage_id=stage_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            thoughts_generated=thoughts_generated,
            errors_count=errors_count,
            retries_count=retries_count,
            metadata=extra_metadata,
        )

    def create_trace(
        self,
        stage_id: str,
        stage_type: PipelineStageType,
        status: str,
        input_thought_ids: list[str],
        output_thought_ids: list[str],
        metrics: StageMetrics | None = None,
        error: str | None = None,
        children: list["StageTrace"] | None = None,
        **extra_metadata: Any,
    ) -> StageTrace:
        """Create an execution trace for a pipeline stage.

        Helper method to construct StageTrace with all relevant execution data.

        Args:
            stage_id: ID of the stage
            stage_type: Type of the pipeline stage
            status: Execution status (e.g., 'completed', 'failed')
            input_thought_ids: IDs of input thoughts
            output_thought_ids: IDs of output thoughts
            metrics: Performance metrics (optional)
            error: Error message if failed (optional)
            children: Child stage traces (optional)
            **extra_metadata: Additional metadata to include

        Returns:
            StageTrace instance with execution information

        Examples:
            >>> trace = executor.create_trace(
            ...     stage_id="stage-123",
            ...     stage_type=PipelineStageType.METHOD,
            ...     status="completed",
            ...     input_thought_ids=["t1"],
            ...     output_thought_ids=["t2", "t3"],
            ...     metrics=metrics
            ... )
        """
        return StageTrace(
            stage_id=stage_id,
            stage_type=stage_type,
            status=status,
            input_thought_ids=input_thought_ids,
            output_thought_ids=output_thought_ids,
            metrics=metrics,
            error=error,
            children=children or [],
            metadata=extra_metadata,
        )


# ============================================================================
# Executor Registry
# ============================================================================


class ExecutorRegistry:
    """Registry for looking up pipeline executors by stage type.

    ExecutorRegistry maintains a mapping from PipelineStageType to executor
    instances. This enables the pipeline engine to dynamically dispatch to
    the appropriate executor based on the stage type.

    Examples:
        Create and use a registry:
        >>> registry = ExecutorRegistry()
        >>> executor = MyMethodExecutor()
        >>> registry.register(PipelineStageType.METHOD, executor)
        >>> assert registry.is_registered(PipelineStageType.METHOD) is True
        >>> retrieved = registry.get(PipelineStageType.METHOD)
        >>> assert retrieved is executor

        Attempt to get unregistered executor:
        >>> unknown = registry.get(PipelineStageType.SEQUENCE)
        >>> assert unknown is None

        List all registered executors:
        >>> stage_types = registry.list_registered()
        >>> assert PipelineStageType.METHOD in stage_types
    """

    def __init__(self) -> None:
        """Initialize an empty executor registry."""
        self._executors: dict[PipelineStageType, PipelineExecutor] = {}

    def register(
        self,
        stage_type: PipelineStageType,
        executor: PipelineExecutor,
        *,
        replace: bool = False,
    ) -> None:
        """Register an executor for a specific stage type.

        Args:
            stage_type: Pipeline stage type this executor handles
            executor: Executor instance to register
            replace: If True, replace existing executor; if False, raise on duplicate

        Raises:
            ValueError: If stage_type already registered and replace=False
            TypeError: If executor doesn't inherit from PipelineExecutor

        Examples:
            >>> registry = ExecutorRegistry()
            >>> executor = MyMethodExecutor()
            >>> registry.register(PipelineStageType.METHOD, executor)

            Replace an existing executor:
            >>> new_executor = MyMethodExecutor()
            >>> registry.register(
            ...     PipelineStageType.METHOD,
            ...     new_executor,
            ...     replace=True
            ... )

            Attempt duplicate registration (raises ValueError):
            >>> try:
            ...     registry.register(PipelineStageType.METHOD, executor)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
        """
        if not isinstance(executor, PipelineExecutor):
            raise TypeError(
                f"Executor must inherit from PipelineExecutor, got {type(executor)}"
            )

        if stage_type in self._executors and not replace:
            raise ValueError(
                f"Executor already registered for stage type '{stage_type}'"
            )

        self._executors[stage_type] = executor

    def unregister(self, stage_type: PipelineStageType) -> bool:
        """Unregister an executor for a specific stage type.

        Args:
            stage_type: Pipeline stage type to unregister

        Returns:
            True if executor was removed, False if not found

        Examples:
            >>> registry = ExecutorRegistry()
            >>> executor = MyMethodExecutor()
            >>> registry.register(PipelineStageType.METHOD, executor)
            >>> result = registry.unregister(PipelineStageType.METHOD)
            >>> assert result is True
            >>> assert registry.is_registered(PipelineStageType.METHOD) is False
            >>>
            >>> result = registry.unregister(PipelineStageType.METHOD)
            >>> assert result is False  # Already removed
        """
        if stage_type not in self._executors:
            return False

        del self._executors[stage_type]
        return True

    def get(self, stage_type: PipelineStageType) -> PipelineExecutor | None:
        """Get the executor for a specific stage type.

        Args:
            stage_type: Pipeline stage type to look up

        Returns:
            Executor instance if registered, None otherwise

        Examples:
            >>> registry = ExecutorRegistry()
            >>> executor = MyMethodExecutor()
            >>> registry.register(PipelineStageType.METHOD, executor)
            >>> retrieved = registry.get(PipelineStageType.METHOD)
            >>> assert retrieved is executor
            >>>
            >>> unknown = registry.get(PipelineStageType.SEQUENCE)
            >>> assert unknown is None
        """
        return self._executors.get(stage_type)

    def is_registered(self, stage_type: PipelineStageType) -> bool:
        """Check if an executor is registered for a stage type.

        Args:
            stage_type: Pipeline stage type to check

        Returns:
            True if registered, False otherwise

        Examples:
            >>> registry = ExecutorRegistry()
            >>> assert registry.is_registered(PipelineStageType.METHOD) is False
            >>> registry.register(PipelineStageType.METHOD, MyMethodExecutor())
            >>> assert registry.is_registered(PipelineStageType.METHOD) is True
        """
        return stage_type in self._executors

    def list_registered(self) -> list[PipelineStageType]:
        """Get a list of all registered stage types.

        Returns:
            List of registered PipelineStageType values

        Examples:
            >>> registry = ExecutorRegistry()
            >>> registry.register(PipelineStageType.METHOD, MyMethodExecutor())
            >>> registry.register(PipelineStageType.SEQUENCE, MyMethodExecutor())
            >>> stage_types = registry.list_registered()
            >>> assert PipelineStageType.METHOD in stage_types
            >>> assert PipelineStageType.SEQUENCE in stage_types
            >>> assert len(stage_types) == 2
        """
        return list(self._executors.keys())

    @property
    def executor_count(self) -> int:
        """Get the number of registered executors.

        Returns:
            Number of registered executors

        Examples:
            >>> registry = ExecutorRegistry()
            >>> assert registry.executor_count == 0
            >>> registry.register(PipelineStageType.METHOD, MyMethodExecutor())
            >>> assert registry.executor_count == 1
        """
        return len(self._executors)
