"""Method executor for single reasoning method execution.

This module implements the MethodExecutor which executes a single reasoning
method stage within a pipeline.
"""

from __future__ import annotations

from datetime import datetime

from reasoning_mcp.engine.executor import ExecutionContext, PipelineExecutor, StageResult
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import MethodStage, Pipeline


class MethodExecutor(PipelineExecutor):
    """Executor for single method stages.

    MethodExecutor executes a single reasoning method and returns its results.
    This is the basic building block for pipeline execution.
    """

    def __init__(self, pipeline: MethodStage):
        """Initialize the method executor.

        Args:
            pipeline: MethodStage configuration to execute
        """
        if not isinstance(pipeline, MethodStage):
            raise TypeError(f"Expected MethodStage, got {type(pipeline)}")
        self.pipeline = pipeline
        self.method_stage = pipeline

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the reasoning method.

        Args:
            context: Execution context with session and state

        Returns:
            StageResult with method outputs

        Note:
            This is a placeholder implementation. The actual method execution
            would integrate with the reasoning method registry and session manager.
        """
        start_time = datetime.now()
        stage = self.method_stage

        try:
            # Placeholder: In real implementation, this would:
            # 1. Get the method from registry
            # 2. Execute it with the session
            # 3. Collect the generated thoughts
            # For now, return a minimal successful result

            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=stage.id,
                start_time=start_time,
                end_time=end_time,
                thoughts_generated=0,
            )

            trace = self.create_trace(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                status="completed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],
                metrics=metrics,
            )

            return StageResult(
                stage_id=stage.id,
                stage_type=PipelineStageType.METHOD,
                success=True,
                output_thought_ids=[],
                output_data={},
                trace=trace,
            )

        except Exception as e:
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

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the method stage configuration.

        Checks that the stage is a valid MethodStage with a method_id.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(stage, MethodStage):
            errors.append(f"Expected MethodStage, got {type(stage).__name__}")
            return errors

        if not stage.method_id:
            errors.append("MethodStage must have a method_id")

        return errors
