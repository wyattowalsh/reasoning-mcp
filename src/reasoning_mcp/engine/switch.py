"""Switch executor for multi-way branching based on expression evaluation.

This module implements the SwitchExecutor, which evaluates an expression and routes
execution to one of multiple case branches based on the result. Similar to switch/case
statements in programming languages, this enables clean multi-way branching in pipelines.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from reasoning_mcp.engine.executor import (
    ExecutionContext,
    PipelineExecutor,
    StageResult,
)
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import Pipeline, SwitchPipeline


class SwitchExecutor(PipelineExecutor):
    """Executor for switch/case pipeline stages.

    SwitchExecutor evaluates an expression to produce a value, then matches that
    value against defined cases to determine which pipeline branch to execute.
    Supports exact matching, pattern matching (regex), range matching, and type
    matching strategies.

    Example:
        >>> switch = SwitchPipeline(
        ...     expression="problem_type",
        ...     cases={
        ...         "ethical": MethodStage(method_id=MethodIdentifier.ETHICAL_REASONING),
        ...         "mathematical": MethodStage(method_id=MethodIdentifier.MATHEMATICAL_REASONING),
        ...         "code": MethodStage(method_id=MethodIdentifier.CODE_REASONING),
        ...     },
        ...     default=MethodStage(method_id=MethodIdentifier.CHAIN_OF_THOUGHT)
        ... )
        >>> executor = SwitchExecutor(switch)
        >>> result = await executor.execute(context)
    """

    def __init__(
        self,
        pipeline: SwitchPipeline,
        case_sensitive: bool = True,
        allow_fallthrough: bool = False,
    ):
        """Initialize the switch executor.

        Args:
            pipeline: SwitchPipeline configuration to execute
            case_sensitive: Whether string matching is case-sensitive (default: True)
            allow_fallthrough: Whether to continue to next case after match (default: False)
        """
        self.pipeline = pipeline
        self.case_sensitive = case_sensitive
        self.allow_fallthrough = allow_fallthrough

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute the switch stage.

        Evaluates the switch expression, finds a matching case, and executes
        the corresponding pipeline. Falls back to default if no match is found.

        Args:
            context: Execution context with variables and state

        Returns:
            StageResult from the executed case branch

        Raises:
            ValueError: If expression evaluation fails or no match/default found
        """
        start_time = datetime.now()
        stage_id = self.pipeline.id

        try:
            # Evaluate the switch expression to get the value to match
            switch_value = self.evaluate_expression(
                self.pipeline.expression,
                context,
            )

            # Find the matching case
            matching_case = self.find_matching_case(
                switch_value,
                self.pipeline.cases,
            )

            # Execute the matching case or default
            if matching_case is not None:
                result = await self.execute_case(
                    matching_case,
                    context,
                    case_key=self._get_case_key(switch_value),
                )
            elif self.pipeline.default is not None:
                result = await self.execute_case(
                    self.pipeline.default,
                    context,
                    case_key="default",
                )
            else:
                # No match and no default
                raise ValueError(
                    f"No matching case for value '{switch_value}' "
                    f"and no default provided in switch '{self.pipeline.name or stage_id}'"
                )

            # Create metrics and trace
            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=stage_id,
                start_time=start_time,
                end_time=end_time,
                thoughts_generated=len(result.output_thought_ids),
                switch_value=str(switch_value),
            )

            trace = self.create_trace(
                stage_id=stage_id,
                stage_type=PipelineStageType.SWITCH,
                status="completed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=result.output_thought_ids,
                metrics=metrics,
                children=[result.trace] if result.trace else [],
                switch_expression=self.pipeline.expression,
                switch_value=str(switch_value),
            )

            return StageResult(
                stage_id=stage_id,
                stage_type=PipelineStageType.SWITCH,
                success=True,
                output_thought_ids=result.output_thought_ids,
                output_data=result.output_data,
                trace=trace,
                metadata={
                    "switch_value": switch_value,
                    "executed_case": result.metadata.get("case_key", "unknown"),
                },
            )

        except Exception as e:
            # Handle execution errors
            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=stage_id,
                start_time=start_time,
                end_time=end_time,
                errors_count=1,
            )

            trace = self.create_trace(
                stage_id=stage_id,
                stage_type=PipelineStageType.SWITCH,
                status="failed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=[],
                metrics=metrics,
                error=str(e),
            )

            return StageResult(
                stage_id=stage_id,
                stage_type=PipelineStageType.SWITCH,
                success=False,
                trace=trace,
                error=str(e),
            )

    def evaluate_expression(self, expression: str, context: ExecutionContext) -> Any:
        """Evaluate the switch expression to get a value.

        Supports:
        - Simple variable lookup: "problem_type"
        - Nested access: "metadata.category"
        - Literal values: "'literal_string'" or "42"

        Args:
            expression: Expression to evaluate
            context: Execution context with variables

        Returns:
            Evaluated value

        Raises:
            ValueError: If expression cannot be evaluated
        """
        # Handle literal strings (quoted)
        if expression.startswith("'") and expression.endswith("'"):
            return expression[1:-1]
        if expression.startswith('"') and expression.endswith('"'):
            return expression[1:-1]

        # Handle numeric literals
        try:
            if "." in expression:
                return float(expression)
            return int(expression)
        except ValueError:
            pass  # Not a numeric literal

        # Handle variable lookup with dot notation
        parts = expression.split(".")
        value: Any = context.variables

        for part in parts:
            if isinstance(value, dict):
                if part not in value:
                    raise ValueError(
                        f"Variable '{part}' not found in context "
                        f"(expression: '{expression}')"
                    )
                value = value[part]
            else:
                raise ValueError(
                    f"Cannot access '{part}' on non-dict value "
                    f"(expression: '{expression}')"
                )

        return value

    def find_matching_case(
        self,
        value: Any,
        cases: dict[str, Pipeline],
    ) -> Pipeline | None:
        """Find the case that matches the given value.

        Matching strategies (in order of precedence):
        1. Exact match (with case sensitivity control)
        2. Regex pattern match (if case key starts with 'regex:')
        3. Range match for numbers (if case key is 'range:min-max')
        4. Type match (if case key is 'type:typename')

        Args:
            value: Value to match against cases
            cases: Dictionary of case keys to pipelines

        Returns:
            Matching Pipeline or None if no match found
        """
        value_str = str(value)
        if not self.case_sensitive:
            value_str = value_str.lower()

        for case_key, case_pipeline in cases.items():
            # Exact match
            compare_key = case_key if self.case_sensitive else case_key.lower()
            if value_str == compare_key:
                return case_pipeline

            # Regex pattern match
            if case_key.startswith("regex:"):
                pattern = case_key[6:]  # Remove 'regex:' prefix
                try:
                    if re.match(pattern, value_str):
                        return case_pipeline
                except re.error:
                    pass  # Invalid regex, skip

            # Range match for numbers
            if case_key.startswith("range:"):
                range_spec = case_key[6:]  # Remove 'range:' prefix
                if self._matches_range(value, range_spec):
                    return case_pipeline

            # Type match
            if case_key.startswith("type:"):
                type_name = case_key[5:]  # Remove 'type:' prefix
                if self._matches_type(value, type_name):
                    return case_pipeline

        return None

    def _matches_range(self, value: Any, range_spec: str) -> bool:
        """Check if value matches a numeric range.

        Range formats:
        - "min-max": value >= min and value <= max
        - "min-": value >= min
        - "-max": value <= max

        Args:
            value: Value to check
            range_spec: Range specification string

        Returns:
            True if value is in range, False otherwise
        """
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False

        parts = range_spec.split("-", 1)
        if len(parts) != 2:
            return False

        min_str, max_str = parts

        # Check minimum
        if min_str:
            try:
                min_val = float(min_str)
                if num_value < min_val:
                    return False
            except ValueError:
                return False

        # Check maximum
        if max_str:
            try:
                max_val = float(max_str)
                if num_value > max_val:
                    return False
            except ValueError:
                return False

        return True

    def _matches_type(self, value: Any, type_name: str) -> bool:
        """Check if value matches a type name.

        Supported type names:
        - str, string
        - int, integer
        - float, number
        - bool, boolean
        - list, array
        - dict, object

        Args:
            value: Value to check
            type_name: Type name to match

        Returns:
            True if value matches type, False otherwise
        """
        type_map = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": float,
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        type_class = type_map.get(type_name.lower())
        if type_class is None:
            return False

        return isinstance(value, type_class)

    def _get_case_key(self, value: Any) -> str:
        """Get a readable case key for the matched value.

        Args:
            value: The value that was matched

        Returns:
            String representation of the case key
        """
        return str(value)

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

    async def execute_case(
        self,
        case_stage: Pipeline,
        context: ExecutionContext,
        case_key: str = "unknown",
    ) -> StageResult:
        """Execute a case branch pipeline.

        Args:
            case_stage: Pipeline stage to execute
            context: Execution context
            case_key: Key of the case being executed (for metadata)

        Returns:
            StageResult from the case execution

        Raises:
            NotImplementedError: If case_stage type doesn't have an executor
        """
        result = await self.execute_stage(case_stage, context)

        # Add case metadata
        result.metadata["case_key"] = case_key

        return result

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the switch pipeline configuration.

        Checks that the switch has an expression and at least one case.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(stage, SwitchPipeline):
            errors.append(f"Expected SwitchPipeline, got {type(stage).__name__}")
            return errors

        if not stage.expression:
            errors.append("Switch pipeline must have an expression")

        if not stage.cases:
            errors.append("Switch pipeline must have at least one case")

        # Validate each case
        for case_key, case_stage in stage.cases.items():
            if not hasattr(case_stage, "id") or not case_stage.id:
                errors.append(f"Case '{case_key}' stage is missing an ID")

        return errors
