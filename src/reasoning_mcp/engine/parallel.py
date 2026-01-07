"""Parallel executor for concurrent pipeline execution.

This module implements the ParallelExecutor which runs multiple pipeline stages
concurrently using asyncio.gather and merges their results using configurable
merge strategies.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime
from typing import Any

from reasoning_mcp.engine.executor import ExecutionContext, PipelineExecutor, StageResult
from reasoning_mcp.models.core import PipelineStageType
from reasoning_mcp.models.pipeline import MergeStrategy, ParallelPipeline, Pipeline


class ParallelExecutor(PipelineExecutor):
    """Executor for parallel pipeline execution with result merging.

    ParallelExecutor runs multiple pipeline branches concurrently and merges
    their outputs according to the specified merge strategy. It supports
    concurrency limits and various merge strategies including voting, best
    selection, concatenation, and dictionary merging.

    Examples:
        Create a parallel executor:
        >>> parallel = ParallelPipeline(
        ...     name="multi_path",
        ...     branches=[stage1, stage2, stage3],
        ...     merge_strategy=MergeStrategy(
        ...         name="vote",
        ...         selection_criteria="most_common_conclusion"
        ...     ),
        ...     max_concurrency=3
        ... )
        >>> executor = ParallelExecutor(parallel)
        >>> result = await executor.execute(context)

        Different merge strategies:
        >>> # Best score strategy
        >>> merge = MergeStrategy(name="best_score", selection_criteria="highest_confidence")
        >>> # Concatenation strategy
        >>> merge = MergeStrategy(name="concat", aggregation="concatenate")
        >>> # Dictionary merge strategy
        >>> merge = MergeStrategy(name="merge_dicts", aggregation="merge")
    """

    def __init__(self, pipeline: ParallelPipeline, fail_fast: bool = False):
        """Initialize the parallel executor.

        Args:
            pipeline: ParallelPipeline configuration to execute
            fail_fast: If True, stop execution on first error (default: False)
        """
        if not isinstance(pipeline, ParallelPipeline):
            raise TypeError(f"Expected ParallelPipeline, got {type(pipeline)}")
        self.pipeline = pipeline
        self.parallel_pipeline = pipeline
        self.fail_fast = fail_fast

    async def execute(self, context: ExecutionContext) -> StageResult:
        """Execute all branches in parallel and merge results.

        Args:
            context: Execution context with session and state

        Returns:
            StageResult with merged outputs from all branches

        Raises:
            Exception: If fail_fast is True and any branch fails
        """
        start_time = datetime.now()
        pipeline = self.parallel_pipeline

        try:
            # Execute branches in parallel with concurrency limit
            results = await self.execute_parallel(
                stages=pipeline.branches,
                context=context,
            )

            # Check for failures
            failed = [r for r in results if not r.success]
            if failed and self.fail_fast:
                error_msg = f"Parallel execution failed: {failed[0].error}"
                return self._create_error_result(
                    pipeline.id,
                    context.thought_ids,
                    error_msg,
                    start_time,
                )

            # Merge successful results
            successful = [r for r in results if r.success]
            if not successful:
                return self._create_error_result(
                    pipeline.id,
                    context.thought_ids,
                    "All parallel branches failed",
                    start_time,
                )

            # Apply merge strategy
            merged_data = self.merge_outputs(successful, pipeline.merge_strategy)

            # Collect all output thought IDs
            all_thought_ids = []
            for result in successful:
                all_thought_ids.extend(result.output_thought_ids)

            # Create metrics and trace
            end_time = datetime.now()
            metrics = self.create_metrics(
                stage_id=pipeline.id,
                start_time=start_time,
                end_time=end_time,
                thoughts_generated=len(all_thought_ids),
                errors_count=len(failed),
                branches_executed=len(results),
                successful_branches=len(successful),
            )

            trace = self.create_trace(
                stage_id=pipeline.id,
                stage_type=PipelineStageType.PARALLEL,
                status="completed",
                input_thought_ids=context.thought_ids,
                output_thought_ids=all_thought_ids,
                metrics=metrics,
                children=[r.trace for r in results if r.trace],
                merge_strategy=pipeline.merge_strategy.name,
            )

            return StageResult(
                stage_id=pipeline.id,
                stage_type=PipelineStageType.PARALLEL,
                success=True,
                output_thought_ids=all_thought_ids,
                output_data=merged_data,
                trace=trace,
                metadata={
                    "merge_strategy": pipeline.merge_strategy.name,
                    "branches_count": len(pipeline.branches),
                    "successful_count": len(successful),
                    "failed_count": len(failed),
                },
            )

        except Exception as e:
            return self._create_error_result(
                pipeline.id,
                context.thought_ids,
                str(e),
                start_time,
            )

    async def execute_parallel(
        self,
        stages: list[Pipeline],
        context: ExecutionContext,
    ) -> list[StageResult]:
        """Execute stages in parallel with concurrency control.

        Args:
            stages: List of pipeline stages to execute
            context: Execution context to use for all stages

        Returns:
            List of StageResults, one per stage

        Examples:
            >>> results = await executor.execute_parallel(
            ...     stages=[stage1, stage2, stage3],
            ...     context=context
            ... )
            >>> successful = [r for r in results if r.success]
        """
        max_concurrency = self.parallel_pipeline.max_concurrency
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_limit(stage: Pipeline) -> StageResult:
            """Execute a single stage with semaphore control."""
            async with semaphore:
                return await self.execute_stage(stage, context)

        # Execute all stages concurrently with concurrency limit
        tasks = [execute_with_limit(stage) for stage in stages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    StageResult(
                        stage_id=stages[i].id,
                        stage_type=PipelineStageType.METHOD,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

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

    def merge_outputs(
        self,
        results: list[StageResult],
        strategy: MergeStrategy,
    ) -> dict[str, Any]:
        """Merge outputs from multiple results using the specified strategy.

        Args:
            results: List of StageResults to merge
            strategy: MergeStrategy defining how to combine results

        Returns:
            Merged output data dictionary

        Merge Strategies:
            - concat: Concatenate all outputs into a list
            - merge_dicts: Merge output dictionaries
            - best_score: Select output with best score
            - vote: Use voting to select best answer

        Examples:
            >>> # Concatenation
            >>> merged = executor.merge_outputs(
            ...     results,
            ...     MergeStrategy(name="concat", aggregation="concatenate")
            ... )
            >>> # Best score
            >>> merged = executor.merge_outputs(
            ...     results,
            ...     MergeStrategy(name="best_score", selection_criteria="highest_confidence")
            ... )
        """
        if not results:
            return {}

        strategy_name = strategy.name.lower()

        # Concatenation strategy - combine all outputs
        if strategy_name == "concat" or strategy.aggregation == "concatenate":
            return self._merge_concat(results)

        # Dictionary merge strategy
        elif strategy_name == "merge_dicts" or strategy.aggregation == "merge":
            return self._merge_dicts(results)

        # Best score strategy - select highest scoring result
        elif strategy_name == "best_score" or "best" in strategy_name:
            return self._merge_best_score(results, strategy)

        # Voting strategy - majority wins
        elif strategy_name == "vote" or "vote" in strategy.selection_criteria.lower():
            return self._merge_vote(results, strategy)

        # Default: return first successful result
        else:
            return results[0].output_data if results else {}

    def _merge_concat(self, results: list[StageResult]) -> dict[str, Any]:
        """Concatenate all result outputs into lists.

        Args:
            results: Results to concatenate

        Returns:
            Dictionary with concatenated values
        """
        merged: dict[str, list[Any]] = {}

        for result in results:
            for key, value in result.output_data.items():
                if key not in merged:
                    merged[key] = []
                if isinstance(value, list):
                    merged[key].extend(value)
                else:
                    merged[key].append(value)

        return {k: v for k, v in merged.items()}

    def _merge_dicts(self, results: list[StageResult]) -> dict[str, Any]:
        """Merge dictionaries from all results.

        Args:
            results: Results to merge

        Returns:
            Merged dictionary (later values override earlier ones)
        """
        merged: dict[str, Any] = {}

        for result in results:
            merged.update(result.output_data)

        return merged

    def _merge_best_score(
        self,
        results: list[StageResult],
        strategy: MergeStrategy,
    ) -> dict[str, Any]:
        """Select result with best score.

        Args:
            results: Results to evaluate
            strategy: Merge strategy with selection criteria

        Returns:
            Output data from best-scoring result
        """
        # Determine scoring key from selection criteria
        score_key = "confidence"
        if "quality" in strategy.selection_criteria.lower():
            score_key = "quality_score"
        elif "score" in strategy.selection_criteria.lower():
            score_key = "score"

        # Find result with highest score
        best_result = max(
            results,
            key=lambda r: r.metadata.get(score_key, 0.0),
            default=results[0],
        )

        return best_result.output_data

    def _merge_vote(
        self,
        results: list[StageResult],
        strategy: MergeStrategy,
    ) -> dict[str, Any]:
        """Use voting to select most common output.

        Args:
            results: Results to vote on
            strategy: Merge strategy with voting configuration

        Returns:
            Most commonly occurring output data
        """
        # For voting, we'll look at the "conclusion" or "answer" field
        vote_key = "conclusion"
        if "answer" in strategy.selection_criteria.lower():
            vote_key = "answer"

        # Count occurrences of each value
        votes = Counter()
        result_map: dict[str, dict[str, Any]] = {}

        for result in results:
            value = result.output_data.get(vote_key)
            if value is not None:
                # Convert to string for counting
                vote_str = str(value)
                votes[vote_str] += 1
                result_map[vote_str] = result.output_data

        # Return most common result
        if votes:
            most_common = votes.most_common(1)[0][0]
            return result_map[most_common]

        # Fallback to first result
        return results[0].output_data if results else {}

    def _create_error_result(
        self,
        stage_id: str,
        input_ids: list[str],
        error: str,
        start_time: datetime,
    ) -> StageResult:
        """Create an error result for failed execution.

        Args:
            stage_id: ID of the stage
            input_ids: Input thought IDs
            error: Error message
            start_time: When execution started

        Returns:
            StageResult representing the error
        """
        end_time = datetime.now()
        metrics = self.create_metrics(
            stage_id=stage_id,
            start_time=start_time,
            end_time=end_time,
            errors_count=1,
        )

        trace = self.create_trace(
            stage_id=stage_id,
            stage_type=PipelineStageType.PARALLEL,
            status="failed",
            input_thought_ids=input_ids,
            output_thought_ids=[],
            metrics=metrics,
            error=error,
        )

        return StageResult(
            stage_id=stage_id,
            stage_type=PipelineStageType.PARALLEL,
            success=False,
            error=error,
            trace=trace,
        )

    async def validate(self, stage: Pipeline) -> list[str]:
        """Validate the parallel pipeline configuration.

        Checks that the pipeline has at least one branch and all branches
        are properly configured.

        Args:
            stage: The pipeline stage to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not isinstance(stage, ParallelPipeline):
            errors.append(f"Expected ParallelPipeline, got {type(stage).__name__}")
            return errors

        if not stage.branches:
            errors.append("Parallel pipeline must have at least one branch")

        # Validate each branch
        for i, branch in enumerate(stage.branches):
            if not hasattr(branch, "id") or not branch.id:
                errors.append(f"Branch {i} is missing an ID")

        # Validate merge strategy
        if not stage.merge_strategy:
            errors.append("Parallel pipeline requires a merge strategy")

        return errors
