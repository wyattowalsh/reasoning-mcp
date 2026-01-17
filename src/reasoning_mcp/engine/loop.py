"""Loop executor for iterative pipeline execution.

This module implements the LoopExecutor which repeatedly executes a pipeline
body until a termination condition is met, with support for result accumulation
and iteration limits.
"""

from __future__ import annotations

import operator
import re
from datetime import datetime
from typing import Any

import structlog

from reasoning_mcp.engine.executor import (
    ExecutionContext,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import Accumulator, Condition, LoopPipeline, Pipeline
from reasoning_mcp.telemetry.instrumentation import traced_executor

logger = structlog.get_logger(__name__)

# Safe operator mapping for condition evaluation
_SAFE_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


class LoopExecutor(PipelineExecutor):
    """Executor for loop pipeline stages.

    Executes a pipeline body repeatedly until a termination condition is met.
    Supports multiple termination conditions, result accumulation strategies,
    and iteration limits.

    The loop can terminate based on:
    - Maximum iterations reached
    - Condition evaluation (until_condition or while_condition)
    - Stage result indicating should_continue=False
    - Execution error (if not configured to continue on error)

    Example:
        >>> from reasoning_mcp.models.pipeline import LoopPipeline, Condition, Accumulator
        >>> loop = LoopPipeline(
        ...     name="refinement_loop",
        ...     body=MethodStage(method_id=MethodIdentifier.SELF_REFLECTION),
        ...     condition=Condition(
        ...         name="quality_threshold",
        ...         expression="quality_score > 0.9",
        ...         field="quality_score",
        ...         operator=">",
        ...         threshold=0.9
        ...     ),
        ...     max_iterations=5,
        ...     accumulator=Accumulator(
        ...         name="improvements",
        ...         operation="append",
        ...         field="content"
        ...     )
        ... )
        >>> executor = LoopExecutor(loop)
        >>> context = ExecutionContext(
        ...     session_id="session-123",
        ...     pipeline_id="pipeline-456"
        ... )
        >>> result = await executor.execute(context)
    """

    def __init__(self, pipeline: LoopPipeline):
        """Initialize the loop executor.

        Args:
            pipeline: Loop pipeline configuration
        """
        self.pipeline = pipeline
        self._body_executor: PipelineExecutor | None = None

    @traced_executor("loop.execute")
    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the loop pipeline.

        Repeatedly executes the body stage until a termination condition is met,
        accumulating results across iterations.

        Args:
            context: Execution context with session and state

        Returns:
            StageResult containing accumulated outputs and execution trace
        """
        start_time = datetime.now()
        iteration = 0
        accumulated_results: dict[str, Any] = {}
        all_output_thoughts: list[str] = []
        iteration_traces: list[Any] = []
        total_thoughts = 0
        errors_count = 0

        # Initialize accumulator if present
        if self.pipeline.accumulator:
            accumulated_results[self.pipeline.accumulator.name] = (
                self.pipeline.accumulator.initial_value
                if self.pipeline.accumulator.initial_value is not None
                else []
            )

        # Track the last result for should_continue checks
        last_result: StageResult | None = None

        # Main loop
        while iteration < self.pipeline.max_iterations:
            # Check condition BEFORE executing body (while semantics)
            if self.pipeline.condition:
                eval_vars = {
                    **context.variables,
                    "iteration": iteration + 1,  # Next iteration number
                    "accumulated": accumulated_results,
                }
                if last_result:
                    eval_vars.update(last_result.output_data)

                try:
                    condition_met = self.evaluate_condition(
                        self.pipeline.condition,
                        eval_vars,
                    )
                    if not condition_met:
                        break
                except (ValueError, TypeError, KeyError) as e:
                    # If condition evaluation fails, stop the loop
                    logger.warning(
                        "loop_condition_evaluation_failed",
                        error=str(e),
                        iteration=iteration,
                    )
                    break

            iteration += 1

            # Update context with current iteration state
            iter_context = context.with_update(
                variables={
                    **context.variables,
                    "iteration": iteration,
                    "accumulated": accumulated_results,
                },
                metadata={
                    **context.metadata,
                    "loop_iteration": iteration,
                },
            )

            # Execute body stage
            try:
                # Execute the body stage (can be mocked in tests)
                iter_result = await self.execute_stage(self.pipeline.body, iter_context)
                last_result = iter_result

                # Track results
                if iter_result.trace:
                    iteration_traces.append(iter_result.trace)
                all_output_thoughts.extend(iter_result.output_thought_ids)
                total_thoughts += len(iter_result.output_thought_ids)

                # Handle execution failure
                if not iter_result.success:
                    errors_count += 1
                    # If stage failed and we should stop, break
                    if not self.pipeline.metadata.get("continue_on_error", False):
                        break

                # Accumulate results
                if self.pipeline.accumulator:
                    accumulated_results = self.accumulate(
                        self.pipeline.accumulator,
                        iter_result.output_data,
                        accumulated_results,
                    )

                # Update context variables with iteration results
                context.variables.update(iter_result.output_data)
                # Update input_data for the next iteration
                context.input_data.update(iter_result.output_data)
                if "content" in iter_result.output_data:
                    context.input_data["input"] = iter_result.output_data["content"]
                context.input_data["thought_ids"] = iter_result.output_thought_ids

                # Check should_continue for result-based termination (not condition)
                if not iter_result.should_continue:
                    break

            except Exception as e:
                errors_count += 1
                # Create error trace for this iteration
                error_trace = self.create_trace(
                    stage_id=f"{self.pipeline.id}-iter-{iteration}",
                    stage_type=PipelineStageType.LOOP,
                    status="failed",
                    input_thought_ids=context.thought_ids,
                    output_thought_ids=[],
                    error=str(e),
                    iteration=iteration,
                )
                iteration_traces.append(error_trace)

                # Stop on error unless configured to continue
                if not self.pipeline.metadata.get("continue_on_error", False):
                    break

        # Create final metrics and trace
        end_time = datetime.now()
        metrics = self.create_metrics(
            stage_id=self.pipeline.id,
            start_time=start_time,
            end_time=end_time,
            thoughts_generated=total_thoughts,
            errors_count=errors_count,
            iterations_completed=iteration,
            max_iterations=self.pipeline.max_iterations,
        )

        trace = self.create_trace(
            stage_id=self.pipeline.id,
            stage_type=PipelineStageType.LOOP,
            status="completed" if errors_count == 0 else "completed_with_errors",
            input_thought_ids=context.thought_ids,
            output_thought_ids=all_output_thoughts,
            metrics=metrics,
            children=iteration_traces,
            iterations_completed=iteration,
            accumulated_results=accumulated_results,
        )

        return StageResult(
            stage_id=self.pipeline.id,
            stage_type=PipelineStageType.LOOP,
            success=errors_count == 0,
            output_thought_ids=all_output_thoughts,
            output_data={
                "accumulated": accumulated_results,
                "iterations_completed": iteration,
                **accumulated_results,
            },
            trace=trace,
            metadata={
                "iterations_completed": iteration,
                "max_iterations": self.pipeline.max_iterations,
                "accumulated_results": accumulated_results,
            },
        )

    def should_continue(
        self,
        iteration: int,
        result: StageResult,
        context: ExecutionContext,
    ) -> bool:
        """Determine if loop should continue.

        Checks multiple termination conditions:
        1. Stage result's should_continue flag
        2. Maximum iterations limit
        3. Loop condition evaluation

        Args:
            iteration: Current iteration number (1-based)
            result: Result from the current iteration
            context: Current execution context

        Returns:
            True if loop should continue, False to terminate
        """
        # Check if stage explicitly signaled to stop
        if not result.should_continue:
            return False

        # Check max iterations (already handled in execute, but defensive check)
        if iteration >= self.pipeline.max_iterations:
            return False

        # Evaluate loop condition if present
        if self.pipeline.condition:
            # Merge result data into context variables for evaluation
            eval_vars = {
                **context.variables,
                **result.output_data,
                "iteration": iteration,
            }

            try:
                condition_met = self.evaluate_condition(
                    self.pipeline.condition,
                    eval_vars,
                )

                # Loop continues while condition is true
                # (opposite of "until" semantics where we stop when condition is true)
                return condition_met

            except (ValueError, TypeError, KeyError) as e:
                # If condition evaluation fails, stop the loop
                logger.warning(
                    "loop_should_continue_evaluation_failed",
                    error=str(e),
                    condition=str(self.pipeline.condition),
                )
                return False

        # No condition specified, continue until max iterations
        return True

    def accumulate(
        self,
        accumulator: Accumulator,
        iteration_result: dict[str, Any],
        accumulated: dict[str, Any],
    ) -> dict[str, Any]:
        """Accumulate iteration results.

        Applies the accumulator's operation to combine the current iteration's
        results with previously accumulated data.

        Supported operations:
        - append: Append value to a list
        - merge: Merge dictionaries
        - replace: Replace with latest value
        - sum: Sum numeric values
        - max: Keep maximum value
        - min: Keep minimum value
        - custom: Use custom accumulation function from metadata

        Args:
            accumulator: Accumulator configuration
            iteration_result: Data from current iteration
            accumulated: Previously accumulated data

        Returns:
            Updated accumulated data
        """
        # Get the value to accumulate from iteration result
        field_value = iteration_result.get(accumulator.field)
        if field_value is None:
            # Nothing to accumulate, return unchanged
            return accumulated

        # Get current accumulated value
        acc_value = accumulated.get(accumulator.name)

        # Apply accumulation operation
        operation = accumulator.operation.lower()

        if operation == "append":
            # Append to list
            if not isinstance(acc_value, list):
                acc_value = []
            acc_value.append(field_value)

        elif operation == "merge":
            # Merge dictionaries
            if not isinstance(acc_value, dict):
                acc_value = {}
            if isinstance(field_value, dict):
                acc_value.update(field_value)

        elif operation == "replace":
            # Replace with latest
            acc_value = field_value

        elif operation == "sum":
            # Sum numeric values
            if acc_value is None:
                acc_value = 0
            acc_value += field_value

        elif operation == "max":
            # Keep maximum
            if acc_value is None:
                acc_value = field_value
            else:
                acc_value = max(acc_value, field_value)

        elif operation == "min":
            # Keep minimum
            if acc_value is None:
                acc_value = field_value
            else:
                acc_value = min(acc_value, field_value)

        elif operation == "custom":
            # Custom accumulation from metadata
            custom_fn = accumulator.metadata.get("accumulator_fn")
            if custom_fn and callable(custom_fn):
                acc_value = custom_fn(acc_value, field_value)
            else:
                # Fallback to append if no custom function provided
                if not isinstance(acc_value, list):
                    acc_value = []
                acc_value.append(field_value)

        # Update accumulated results
        return {
            **accumulated,
            accumulator.name: acc_value,
        }

    def evaluate_condition(
        self,
        condition: Condition,
        variables: dict[str, Any],
    ) -> bool:
        """Evaluate a condition against current variables.

        Uses a safe expression parser that only supports:
        - Simple comparisons (==, !=, >, <, >=, <=)
        - Boolean logic (and, or, not)
        - Existence checks ("field exists")
        - Variable truthiness checks

        Args:
            condition: Condition to evaluate
            variables: Current variable values

        Returns:
            True if condition is met, False otherwise
        """
        # If field is specified, use simple operator comparison
        if condition.field and condition.field in variables:
            field_value = variables[condition.field]

            # Handle threshold-based comparisons
            if condition.threshold is not None:
                op_str = condition.operator
                threshold = condition.threshold
                op_func = _SAFE_OPERATORS.get(op_str)
                if op_func:
                    return bool(op_func(field_value, threshold))

            # Handle boolean comparisons
            if condition.operator == "==":
                return bool(field_value)
            elif condition.operator == "!=":
                return not bool(field_value)

        # Fall back to safe expression evaluation
        return self._safe_evaluate_expression(condition.expression, variables)

    def _safe_evaluate_expression(
        self,
        expression: str,
        variables: dict[str, Any],
    ) -> bool:
        """Safely evaluate a condition expression without using eval().

        Supports:
        - Simple comparisons: "x > 5", "name == 'test'"
        - Boolean logic: "x > 5 and y < 10", "a or b"
        - Negation: "not done"
        - Existence checks: "field exists"
        - Variable truthiness: "is_complete"

        Args:
            expression: Expression string to evaluate
            variables: Available variables for evaluation

        Returns:
            Boolean result of the expression
        """
        expression = expression.strip()

        # Handle existence checks
        if " exists" in expression:
            var_name = expression.replace(" exists", "").strip()
            return var_name in variables

        # Handle logical operators (split on 'and' first, then 'or')
        if " and " in expression:
            parts = expression.split(" and ")
            return all(self._safe_evaluate_expression(part.strip(), variables) for part in parts)

        if " or " in expression:
            parts = expression.split(" or ")
            return any(self._safe_evaluate_expression(part.strip(), variables) for part in parts)

        # Handle negation
        if expression.startswith("not "):
            inner_expr = expression[4:].strip()
            return not self._safe_evaluate_expression(inner_expr, variables)

        # Handle simple comparison expressions
        return self._evaluate_simple_comparison(expression, variables)

    def _evaluate_simple_comparison(
        self,
        expression: str,
        variables: dict[str, Any],
    ) -> bool:
        """Evaluate a simple comparison expression.

        Args:
            expression: Simple comparison like "x > 5" or "name == 'test'"
            variables: Available variables

        Returns:
            Boolean result
        """
        # Pattern: variable operator value
        pattern = r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)"
        match = re.match(pattern, expression.strip())

        if not match:
            # Try to evaluate as direct variable (truthy check)
            var_name = expression.strip()
            if var_name in variables:
                return bool(variables[var_name])
            return False

        var_name, op_str, value_str = match.groups()

        # Check if variable exists
        if var_name not in variables:
            return False

        var_value = variables[var_name]
        value_str = value_str.strip()

        # Parse the comparison value safely
        compare_value = self._parse_literal(value_str)

        # Apply operator
        op_func = _SAFE_OPERATORS.get(op_str)
        if op_func:
            try:
                return bool(op_func(var_value, compare_value))
            except (TypeError, ValueError):
                return False
        return False

    def _parse_literal(self, value_str: str) -> Any:
        """Parse a literal value from string (no eval).

        Args:
            value_str: String representation of a value

        Returns:
            Parsed value (bool, None, int, float, or string)
        """
        value_str = value_str.strip()

        # Boolean literals
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # None literal
        if value_str.lower() == "none":
            return None

        # String literals (quoted)
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Try numeric conversion
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Return as-is (string)
        return value_str

    async def execute_stage(
        self,
        stage: Pipeline,
        context: ExecutionContext,
    ) -> StageResult:
        """Execute a single pipeline stage.

        This method is used internally and can be mocked for testing.

        Args:
            stage: The pipeline stage to execute
            context: Execution context

        Returns:
            StageResult from stage execution
        """
        # Import here to avoid circular dependency
        from reasoning_mcp.engine.registry import get_executor_for_stage

        executor = get_executor_for_stage(stage)
        return await executor.execute(context)

    def _create_body_executor(self) -> PipelineExecutor:
        """Create executor for the loop body.

        Returns:
            Executor for the body pipeline stage

        Raises:
            NotImplementedError: Body executor creation not yet implemented
        """
        # This would be implemented by a PipelineExecutorFactory
        # For now, raise NotImplementedError as a placeholder
        raise NotImplementedError(
            "Body executor creation requires PipelineExecutorFactory. "
            "This will be implemented when the full executor infrastructure is in place."
        )

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the loop pipeline configuration.

        Checks that the loop has a body and valid iteration limits.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """

        errors = []

        if not isinstance(stage, LoopPipeline):
            errors.append(f"Expected LoopPipeline, got {type(stage).__name__}")
            return errors

        if not stage.body:
            errors.append("Loop pipeline must have a body stage")

        if stage.max_iterations < 1:
            errors.append("Loop max_iterations must be at least 1")

        # Validate body stage if present
        if stage.body and (not hasattr(stage.body, "id") or not stage.body.id):
            errors.append("Loop body stage is missing an ID")

        return errors
