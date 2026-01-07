"""Conditional executor for pipeline branching logic.

This module implements the ConditionalExecutor, which evaluates conditions and
executes different pipeline branches based on the evaluation result. It supports
variable comparisons, existence checks, logical operators, and safe Python
expression evaluation.
"""

from __future__ import annotations

import operator
import re
from typing import Any

from reasoning_mcp.engine.executor import ExecutionContext, StageResult
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import Condition, ConditionalPipeline, Pipeline


class ConditionalExecutor:
    """Executor for conditional pipeline branching.

    ConditionalExecutor evaluates conditions and executes different pipeline
    branches based on the result. It supports:
    - Simple comparisons (==, !=, >, <, >=, <=)
    - Existence checks (variable exists)
    - Logical operators (and, or, not)
    - Safe Python expression evaluation
    - Nested conditional pipelines (elif-like chaining)

    Examples:
        Create and use a conditional executor:
        >>> from reasoning_mcp.models.pipeline import Condition, MethodStage
        >>> from reasoning_mcp.models.core import MethodIdentifier
        >>>
        >>> executor = ConditionalExecutor()
        >>> pipeline = ConditionalPipeline(
        ...     name="quality_check",
        ...     condition=Condition(
        ...         name="high_quality",
        ...         expression="quality_score > 0.8",
        ...         operator=">",
        ...         threshold=0.8,
        ...         field="quality_score"
        ...     ),
        ...     if_true=MethodStage(
        ...         method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
        ...         name="continue_reasoning"
        ...     ),
        ...     if_false=MethodStage(
        ...         method_id=MethodIdentifier.SELF_REFLECTION,
        ...         name="improve_quality"
        ...     )
        ... )
    """

    # Mapping of operator strings to operator functions
    _OPERATORS = {
        "==": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
    }

    def __init__(
        self,
        pipeline: ConditionalPipeline | None = None,
        default_branch: str = "else",
    ) -> None:
        """Initialize the conditional executor.

        Args:
            pipeline: ConditionalPipeline to execute
            default_branch: Which branch to take if condition evaluation fails
                          ("else" to take if_false branch, "then" to take if_true)
        """
        self.pipeline = pipeline
        self.default_branch = default_branch

    async def execute(
        self,
        context: ExecutionContext,
        pipeline: ConditionalPipeline | None = None,
    ) -> StageResult:
        """Execute a conditional pipeline by evaluating condition and running selected branch.

        This method:
        1. Evaluates the condition using current context variables
        2. Selects the appropriate branch (if_true or if_false)
        3. Executes the selected branch pipeline
        4. Returns the result from the executed branch

        Args:
            pipeline: The ConditionalPipeline to execute
            context: Current execution context with variables and graph

        Returns:
            StageResult from the executed branch

        Examples:
            >>> import asyncio
            >>> executor = ConditionalExecutor(pipeline)
            >>> # ... set up context ...
            >>> result = asyncio.run(executor.execute(context))
        """
        # Use passed pipeline or fall back to self.pipeline
        active_pipeline = pipeline if pipeline is not None else self.pipeline
        if active_pipeline is None:
            return StageResult(
                stage_id="unknown",
                stage_type=PipelineStageType.CONDITIONAL,
                success=False,
                output_thought_ids=[],
                error="No pipeline provided",
            )

        try:
            # Evaluate the condition
            condition_met = self.evaluate_condition(active_pipeline.condition, context)

            # Select the appropriate branch
            if condition_met:
                selected_branch = active_pipeline.if_true
            else:
                selected_branch = active_pipeline.if_false

            # If selected branch is None, return empty success result
            if selected_branch is None:
                return StageResult(
                    stage_id=active_pipeline.id,
                    stage_type=PipelineStageType.CONDITIONAL,
                    success=True,
                    output_thought_ids=[],
                    metadata={
                        "condition_met": condition_met,
                        "branch_taken": "if_true" if condition_met else "if_false",
                        "branch_skipped": True,
                    },
                )

            # Execute the selected branch
            result = await self.execute_branch(selected_branch, context)

            # Add metadata about which branch was taken
            metadata = dict(result.metadata) if result.metadata else {}
            metadata.update(
                {
                    "condition_met": condition_met,
                    "branch_taken": "if_true" if condition_met else "if_false",
                }
            )

            return StageResult(
                stage_id=active_pipeline.id,
                stage_type=PipelineStageType.CONDITIONAL,
                success=result.success,
                output_thought_ids=result.output_thought_ids,
                error=result.error,
                metadata=metadata,
            )

        except Exception as e:
            error_msg = f"Conditional execution failed: {str(e)}"
            return StageResult(
                stage_id=active_pipeline.id,
                stage_type=PipelineStageType.CONDITIONAL,
                success=False,
                output_thought_ids=[],
                error=error_msg,
                metadata={"condition": active_pipeline.condition.expression},
            )

    def evaluate_condition(
        self,
        condition: Condition,
        context: ExecutionContext,
    ) -> bool:
        """Evaluate a condition against the current execution context.

        This method supports multiple evaluation strategies:
        1. Simple field comparison (if field and operator are specified)
        2. Expression parsing for basic comparisons
        3. Logical operators (and, or, not)
        4. Existence checks (variable exists)

        Args:
            condition: The Condition to evaluate
            context: Current execution context with variables

        Returns:
            True if condition is met, False otherwise

        Raises:
            ValueError: If condition expression is malformed

        Examples:
            >>> condition = Condition(
            ...     name="threshold",
            ...     expression="confidence > 0.8",
            ...     operator=">",
            ...     threshold=0.8,
            ...     field="confidence"
            ... )
            >>> context = ExecutionContext(
            ...     session_id="test",
            ...     graph=ThoughtGraph(),
            ...     variables={"confidence": 0.9}
            ... )
            >>> executor = ConditionalExecutor()
            >>> executor.evaluate_condition(condition, context)
            True
        """
        variables = context.variables

        # Strategy 1: Simple field-based comparison
        if condition.field and condition.operator in self._OPERATORS:
            if condition.field not in variables:
                return False

            field_value = variables[condition.field]
            compare_value = condition.threshold if condition.threshold is not None else True

            op_func = self._OPERATORS[condition.operator]
            return bool(op_func(field_value, compare_value))

        # Strategy 2: Parse and evaluate expression
        expression = condition.expression.strip()

        # Handle existence checks
        if " exists" in expression:
            var_name = expression.replace(" exists", "").strip()
            return var_name in variables

        # Handle logical operators
        if " and " in expression:
            parts = expression.split(" and ")
            return all(
                self._evaluate_simple_expression(part.strip(), variables)
                for part in parts
            )

        if " or " in expression:
            parts = expression.split(" or ")
            return any(
                self._evaluate_simple_expression(part.strip(), variables)
                for part in parts
            )

        if expression.startswith("not "):
            inner_expr = expression[4:].strip()
            return not self._evaluate_simple_expression(inner_expr, variables)

        # Strategy 3: Evaluate as simple expression
        return self._evaluate_simple_expression(expression, variables)

    def _evaluate_simple_expression(
        self,
        expression: str,
        variables: dict[str, Any],
    ) -> bool:
        """Evaluate a simple comparison expression.

        Parses expressions like:
        - "confidence > 0.8"
        - "is_valid == True"
        - "count != 0"

        Args:
            expression: Simple comparison expression
            variables: Available variables for evaluation

        Returns:
            Boolean result of expression evaluation

        Raises:
            ValueError: If expression cannot be parsed
        """
        # Pattern: variable operator value
        pattern = r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)"
        match = re.match(pattern, expression)

        if not match:
            # Try to evaluate as direct variable (truthy check)
            var_name = expression.strip()
            if var_name in variables:
                return bool(variables[var_name])
            return False

        var_name, op, value_str = match.groups()

        # Check if variable exists
        if var_name not in variables:
            return False

        var_value = variables[var_name]
        value_str = value_str.strip()

        # Parse the comparison value
        try:
            # Try to convert to appropriate type
            if value_str.lower() == "true":
                compare_value: Any = True
            elif value_str.lower() == "false":
                compare_value = False
            elif value_str.lower() == "none":
                compare_value = None
            elif value_str.startswith('"') or value_str.startswith("'"):
                # String literal
                compare_value = value_str.strip('"\'')
            else:
                # Try numeric conversion
                try:
                    if "." in value_str:
                        compare_value = float(value_str)
                    else:
                        compare_value = int(value_str)
                except ValueError:
                    # Keep as string
                    compare_value = value_str

        except Exception:
            compare_value = value_str

        # Apply operator
        if op not in self._OPERATORS:
            raise ValueError(f"Unsupported operator: {op}")

        op_func = self._OPERATORS[op]
        return bool(op_func(var_value, compare_value))

    async def execute_stage(
        self,
        stage: Pipeline,
        context: ExecutionContext,
    ) -> StageResult:
        """Execute a single pipeline stage.

        This method is used internally and can be mocked for testing.

        Args:
            stage: The pipeline stage to execute
            context: Current execution context

        Returns:
            Result from executing the stage
        """
        # Import here to avoid circular dependency
        from reasoning_mcp.engine.registry import get_executor_for_stage

        executor = get_executor_for_stage(stage)
        return await executor.execute(context)

    async def execute_branch(
        self,
        stage: Pipeline,
        context: ExecutionContext,
    ) -> StageResult:
        """Execute a pipeline branch.

        Delegates to execute_stage which dispatches to the appropriate executor
        based on the stage type.

        Args:
            stage: The pipeline stage to execute
            context: Current execution context

        Returns:
            Result from executing the branch
        """
        return await self.execute_stage(stage, context)
