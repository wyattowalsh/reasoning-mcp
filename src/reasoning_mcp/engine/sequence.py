"""Sequential pipeline executor.

This module implements the SequenceExecutor, which runs pipeline stages in
linear sequence, passing the output of each stage as input to the next.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from reasoning_mcp.engine.executor import (
    ExecutionContext,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import MethodStage, Pipeline, SequencePipeline, Transform


class SequenceExecutor(PipelineExecutor):
    """Executor for sequential pipeline stages.

    SequenceExecutor runs pipeline stages one after another in order, passing
    the output of each stage as input to the next. It supports:
    - Sequential stage execution
    - Data transformations between stages
    - Error handling with stop-on-error or continue modes
    - Output collection from all stages

    Configuration:
        - stop_on_error: Whether to halt on first error (default: True)
        - collect_all_outputs: Whether to collect outputs from all stages (default: False)

    Examples:
        Create and execute a sequence:
        >>> executor = SequenceExecutor(
        ...     pipeline=sequence_pipeline,
        ...     stop_on_error=True
        ... )
        >>> context = ExecutionContext(
        ...     session_id="session-123",
        ...     pipeline_id="pipeline-abc"
        ... )
        >>> result = await executor.execute(context)
    """

    def __init__(
        self,
        pipeline: SequencePipeline,
        stop_on_error: bool = True,
        collect_all_outputs: bool = False,
    ):
        """Initialize the sequence executor.

        Args:
            pipeline: The sequence pipeline to execute
            stop_on_error: Whether to stop execution on first error
            collect_all_outputs: Whether to collect outputs from all stages
        """
        self.pipeline = pipeline
        self.stop_on_error = stop_on_error
        self.collect_all_outputs = collect_all_outputs

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute all stages in the sequence.

        Runs each stage in order, passing outputs to the next stage.
        Applies transforms between stages if configured.

        Args:
            context: Execution context with session and state

        Returns:
            StageResult with final output or error

        Raises:
            Exception: If a stage fails and stop_on_error is True
        """
        start_time = datetime.now()
        stage_traces = []
        all_outputs = []
        current_thought_ids = context.thought_ids.copy()

        # Track overall execution
        total_thoughts = 0
        total_errors = 0

        # Execute each stage in sequence
        for i, stage in enumerate(self.pipeline.stages):
            # Update context with current thought IDs
            stage_context = context.with_update(thought_ids=current_thought_ids)

            try:
                # Execute this stage
                stage_result = await self.execute_stage(stage, stage_context)

                # Track metrics
                total_thoughts += len(stage_result.output_thought_ids)
                if not stage_result.success:
                    total_errors += 1

                # Add trace
                if stage_result.trace:
                    stage_traces.append(stage_result.trace)

                # Collect output if configured
                if self.collect_all_outputs:
                    all_outputs.append(stage_result.output_data)

                # Check for errors
                if not stage_result.success:
                    if self.stop_on_error:
                        # Create error result
                        return self._create_error_result(
                            stage_id=self.pipeline.id,
                            error=stage_result.error or "Stage execution failed",
                            start_time=start_time,
                            total_errors=total_errors,
                            stage_traces=stage_traces,
                        )
                    else:
                        # Continue to next stage on error
                        continue

                # Update current thought IDs for next stage
                current_thought_ids = stage_result.output_thought_ids

                # Apply transforms if this is a MethodStage and it has transforms
                if isinstance(stage, MethodStage) and stage.transforms:
                    for transform in stage.transforms:
                        stage_result.output_data = self.apply_transform(
                            transform, stage_result.output_data
                        )

                # Update context variables with output data
                context.variables.update(stage_result.output_data)

            except Exception as e:
                total_errors += 1

                # Create error trace
                error_trace = self.create_trace(
                    stage_id=getattr(stage, "id", f"stage-{i}"),
                    stage_type=getattr(
                        stage, "stage_type", PipelineStageType.METHOD
                    ),
                    status="failed",
                    input_thought_ids=current_thought_ids,
                    output_thought_ids=[],
                    error=str(e),
                )
                stage_traces.append(error_trace)

                if self.stop_on_error:
                    return self._create_error_result(
                        stage_id=self.pipeline.id,
                        error=str(e),
                        start_time=start_time,
                        total_errors=total_errors,
                        stage_traces=stage_traces,
                    )

        # All stages completed successfully
        end_time = datetime.now()

        # Prepare final output
        final_output = all_outputs if self.collect_all_outputs else context.variables

        # Create metrics
        metrics = self.create_metrics(
            stage_id=self.pipeline.id,
            start_time=start_time,
            end_time=end_time,
            thoughts_generated=total_thoughts,
            errors_count=total_errors,
        )

        # Create trace
        trace = self.create_trace(
            stage_id=self.pipeline.id,
            stage_type=PipelineStageType.SEQUENCE,
            status="completed",
            input_thought_ids=context.thought_ids,
            output_thought_ids=current_thought_ids,
            metrics=metrics,
            children=stage_traces,
        )

        return StageResult(
            stage_id=self.pipeline.id,
            stage_type=PipelineStageType.SEQUENCE,
            success=True,
            output_thought_ids=current_thought_ids,
            output_data=final_output,
            trace=trace,
            metadata={
                "stages_executed": len(self.pipeline.stages),
                "total_errors": total_errors,
            },
        )

    async def execute_stage(
        self, stage: MethodStage, context: ExecutionContext
    ) -> StageResult:
        """Execute a single stage in the sequence.

        This method delegates to the appropriate executor based on stage type.
        Currently supports MethodStage execution.

        Args:
            stage: The stage to execute
            context: Execution context

        Returns:
            StageResult from stage execution

        Note:
            For MethodStage, this is a placeholder that would call the actual
            reasoning method. In a complete implementation, this would:
            1. Get the method from the registry
            2. Create a session if needed
            3. Call method.execute()
            4. Convert the result to StageResult
        """
        start_time = datetime.now()

        try:
            # This is a simplified implementation
            # In the full system, this would:
            # 1. Get method from registry based on stage.method_id
            # 2. Create or get session
            # 3. Execute method
            # 4. Convert ThoughtNode to StageResult

            # For now, create a mock result
            # TODO: Integrate with method registry and session manager
            end_time = datetime.now()

            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                thoughts_generated=1,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="completed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],  # Would be populated by method execution
                metrics=metrics,
            )

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[],  # Would be populated
                output_data={"status": "executed"},  # Would contain method output
                trace=trace,
            )

        except Exception as e:
            # Handle stage execution error
            end_time = datetime.now()

            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                errors_count=1,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="failed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],
                metrics=metrics,
                error=str(e),
            )

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=False,
                error=str(e),
                trace=trace,
            )

    def apply_transform(self, transform: Transform, data: dict[str, Any]) -> dict[str, Any]:
        """Apply a transformation to stage output data.

        Transforms enable data manipulation between pipeline stages, such as
        extracting specific fields, formatting content, or combining values.

        Args:
            transform: The transformation to apply
            data: Input data to transform

        Returns:
            Transformed data dictionary

        Examples:
            Extract a field:
            >>> transform = Transform(
            ...     name="extract",
            ...     expression="{content}",
            ...     input_fields=["content"],
            ...     output_field="summary"
            ... )
            >>> data = {"content": "Analysis complete", "confidence": 0.9}
            >>> result = executor.apply_transform(transform, data)
            >>> result["summary"]
            'Analysis complete'
        """
        # Create a copy to avoid mutating input
        result = data.copy()

        try:
            # Extract input values
            input_values = {}
            for field in transform.input_fields:
                if field in data:
                    input_values[field] = data[field]

            # Simple template substitution
            # In a full implementation, this would support:
            # - JMESPath expressions
            # - Python expressions (safely evaluated)
            # - Custom transform functions
            expression = transform.expression
            for field, value in input_values.items():
                expression = expression.replace(f"{{{field}}}", str(value))

            # Store result in output field
            result[transform.output_field] = expression

        except Exception as e:
            # Log transform error but don't fail the pipeline
            # Store error information in metadata
            result[f"{transform.output_field}_error"] = str(e)

        return result

    def _create_error_result(
        self,
        stage_id: str,
        error: str,
        start_time: datetime,
        total_errors: int,
        stage_traces: list,
    ) -> StageResult:
        """Create a StageResult for a failed sequence execution.

        Args:
            stage_id: ID of the sequence pipeline
            error: Error message
            start_time: When execution started
            total_errors: Total number of errors encountered
            stage_traces: Traces from executed stages

        Returns:
            StageResult indicating failure
        """
        end_time = datetime.now()

        metrics = self.create_metrics(
            stage_id=stage_id,
            start_time=start_time,
            end_time=end_time,
            errors_count=total_errors,
        )

        trace = self.create_trace(
            stage_id=stage_id,
            stage_type=PipelineStageType.SEQUENCE,
            status="failed",
            input_thought_ids=[],
            output_thought_ids=[],
            metrics=metrics,
            error=error,
            children=stage_traces,
        )

        return StageResult(
            stage_id=stage_id,
            stage_type=PipelineStageType.SEQUENCE,
            success=False,
            error=error,
            trace=trace,
            metadata={
                "total_errors": total_errors,
                "stages_completed": len(stage_traces),
            },
        )

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the sequence pipeline configuration.

        Checks that the pipeline has at least one stage and all stages
        are properly configured.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(stage, SequencePipeline):
            errors.append(f"Expected SequencePipeline, got {type(stage).__name__}")
            return errors

        if not stage.stages:
            errors.append("Sequence pipeline must have at least one stage")

        # Validate each sub-stage
        for i, sub_stage in enumerate(stage.stages):
            if not hasattr(sub_stage, "id") or not sub_stage.id:
                errors.append(f"Stage {i} is missing an ID")

        return errors
