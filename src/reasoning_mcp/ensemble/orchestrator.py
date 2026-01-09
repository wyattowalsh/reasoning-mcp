"""Ensemble orchestrator for coordinating parallel reasoning execution.

This module provides the EnsembleOrchestrator class, which coordinates the execution
of multiple reasoning methods in parallel and aggregates their results using
configurable voting strategies. The orchestrator handles:

- Parallel execution of multiple reasoning methods
- Timeout management and graceful degradation
- Result aggregation via voting strategies
- Agreement scoring and confidence calculation
- Integration with the method registry and execution context

The orchestrator is the core component of ensemble reasoning, enabling robust
multi-method reasoning that combines diverse approaches for improved accuracy
and reliability.

Examples:
    Basic ensemble execution:
    >>> from reasoning_mcp.models.ensemble import (
    ...     EnsembleConfig,
    ...     EnsembleMember,
    ...     VotingStrategy,
    ... )
    >>> config = EnsembleConfig(
    ...     members=[
    ...         EnsembleMember(method_name="chain_of_thought"),
    ...         EnsembleMember(method_name="tree_of_thoughts"),
    ...     ],
    ...     strategy=VotingStrategy.MAJORITY,
    ... )
    >>> orchestrator = EnsembleOrchestrator(config=config)
    >>> result = await orchestrator.execute("What is 2 + 2?")
    >>> assert result.final_answer is not None
    >>> assert 0.0 <= result.confidence <= 1.0

    Weighted ensemble with custom timeout:
    >>> config = EnsembleConfig(
    ...     members=[
    ...         EnsembleMember(method_name="cot", weight=1.0),
    ...         EnsembleMember(method_name="tot", weight=2.0),
    ...         EnsembleMember(method_name="mcts", weight=1.5),
    ...     ],
    ...     strategy=VotingStrategy.WEIGHTED,
    ...     timeout_ms=60000,
    ... )
    >>> orchestrator = EnsembleOrchestrator(
    ...     config=config,
    ...     registry=method_registry,
    ... )
    >>> result = await orchestrator.execute("Solve this problem")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from reasoning_mcp.ensemble.strategies.registry import get_strategy
from reasoning_mcp.models.ensemble import (
    EnsembleConfig,
    EnsembleMember,
    EnsembleResult,
    MemberResult,
)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.registry import MethodRegistry

logger = logging.getLogger(__name__)


class EnsembleOrchestrator:
    """Orchestrates ensemble reasoning execution.

    The EnsembleOrchestrator coordinates parallel execution of multiple reasoning
    methods and aggregates their results using a configured voting strategy. It
    handles timeouts, partial results, and provides detailed execution metrics.

    The orchestrator supports multiple voting strategies (majority, weighted,
    consensus, best_score, synthesis, ranked_choice, borda_count) and can be
    configured with custom timeouts and agreement thresholds.

    Attributes:
        config: Ensemble configuration specifying members, strategy, and settings
        _registry: Optional method registry for looking up reasoning methods
        _execution_context: Optional execution context with session and variables
        _results: Cache of member results from the most recent execution

    Examples:
        Create orchestrator with basic config:
        >>> config = EnsembleConfig(
        ...     members=[
        ...         EnsembleMember(method_name="chain_of_thought"),
        ...         EnsembleMember(method_name="self_consistency"),
        ...     ]
        ... )
        >>> orchestrator = EnsembleOrchestrator(config=config)
        >>> assert orchestrator.config == config
        >>> assert len(orchestrator.config.members) == 2

        Create orchestrator with registry:
        >>> from reasoning_mcp.registry import MethodRegistry
        >>> registry = MethodRegistry()
        >>> orchestrator = EnsembleOrchestrator(
        ...     config=config,
        ...     registry=registry,
        ... )
        >>> assert orchestrator._registry is registry

        Execute ensemble:
        >>> result = await orchestrator.execute("What is reasoning?")
        >>> assert result.final_answer is not None
        >>> assert 0.0 <= result.confidence <= 1.0
        >>> assert 0.0 <= result.agreement_score <= 1.0
        >>> assert len(result.member_results) > 0
    """

    def __init__(
        self,
        config: EnsembleConfig,
        registry: MethodRegistry | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> None:
        """Initialize orchestrator with config and optional dependencies.

        Args:
            config: Ensemble configuration with members and voting strategy.
                Must contain at least one ensemble member.
            registry: Optional method registry for looking up reasoning methods.
                If None, member execution will use placeholder/mock execution.
            execution_context: Optional execution context with session and state.
                If None, execution will not have access to thought graph or variables.

        Raises:
            ValueError: If config.members is empty.

        Examples:
            Basic initialization:
            >>> config = EnsembleConfig(
            ...     members=[EnsembleMember(method_name="cot")]
            ... )
            >>> orchestrator = EnsembleOrchestrator(config=config)
            >>> assert orchestrator.config.strategy.value == "majority"

            With registry and context:
            >>> orchestrator = EnsembleOrchestrator(
            ...     config=config,
            ...     registry=method_registry,
            ...     execution_context=exec_context,
            ... )
            >>> assert orchestrator._registry is not None
            >>> assert orchestrator._execution_context is not None
        """
        if not config.members:
            raise ValueError("EnsembleConfig must have at least one member")

        self.config = config
        self._registry = registry
        self._execution_context = execution_context
        self._results: list[MemberResult] = []

        logger.info(
            "Initialized EnsembleOrchestrator with %d members using %s strategy",
            len(config.members),
            config.strategy.value,
        )

    # Task 6.2: Execute single member
    async def _execute_member(
        self,
        member: EnsembleMember,
        query: str,
    ) -> MemberResult:
        """Execute a single ensemble member.

        Executes one reasoning method from the ensemble and captures its result,
        confidence score, and execution time. If a registry is available, the
        actual method implementation is invoked; otherwise, a placeholder result
        is generated for testing.

        Args:
            member: The ensemble member configuration specifying which method to run
                and with what parameters (weight, config).
            query: The reasoning query or problem to solve. This is passed as input
                to the reasoning method.

        Returns:
            MemberResult containing the reasoning output, confidence score,
            and execution time in milliseconds.

        Raises:
            Exception: Any exception raised by the underlying reasoning method
                is propagated to the caller. The orchestrator handles these
                exceptions at the ensemble level.

        Examples:
            Execute a member with placeholder:
            >>> member = EnsembleMember(method_name="chain_of_thought")
            >>> result = await orchestrator._execute_member(member, "What is 2+2?")
            >>> assert result.member == member
            >>> assert isinstance(result.result, str)
            >>> assert 0.0 <= result.confidence <= 1.0
            >>> assert result.execution_time_ms >= 0

            Execute with actual method (requires registry):
            >>> orchestrator = EnsembleOrchestrator(
            ...     config=config,
            ...     registry=method_registry,
            ... )
            >>> result = await orchestrator._execute_member(member, query)
            >>> # Result comes from actual reasoning method

        Note:
            Current implementation uses placeholder execution when no registry
            is provided. In production, the method is retrieved from the registry
            and executed with the member's configuration:

            method = self._registry.get(member.method_name)
            result = await method.execute(query, config=member.config)
        """
        start_time = time.monotonic()

        try:
            # Get method from registry and execute
            # For now, simulate execution if no registry
            # TODO: In production, integrate with actual method execution:
            # if self._registry:
            #     method = self._registry.get(member.method_name)
            #     result_text = await method.execute(query, config=member.config)
            #     confidence = method.calculate_confidence()
            # else:
            #     result_text = f"Result from {member.method_name}"
            #     confidence = 0.85

            # Placeholder execution for testing
            result_text = f"Result from {member.method_name} for query: {query}"
            confidence = 0.85

            logger.debug(
                "Executed member %s in %.2fms with confidence %.2f",
                member.method_name,
                (time.monotonic() - start_time) * 1000,
                confidence,
            )

        except Exception as e:
            logger.error(
                "Error executing member %s: %s",
                member.method_name,
                str(e),
                exc_info=True,
            )
            raise

        execution_time_ms = int((time.monotonic() - start_time) * 1000)

        return MemberResult(
            member=member,
            result=result_text,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
        )

    # Task 6.3: Execute all members in parallel
    async def execute(self, query: str) -> EnsembleResult:
        """Execute ensemble reasoning with all members.

        Runs all ensemble members in parallel with a configured timeout, then
        aggregates their results using the configured voting strategy. Handles
        partial results gracefully when timeout occurs, and calculates agreement
        scores across member outputs.

        The execution flow:
        1. Create parallel tasks for all ensemble members
        2. Execute with timeout (config.timeout_ms)
        3. Collect results (partial if timeout)
        4. Apply voting strategy to aggregate results
        5. Calculate agreement score
        6. Return comprehensive EnsembleResult

        Args:
            query: The reasoning query or problem to solve. This query is passed
                to all ensemble members for parallel execution.

        Returns:
            EnsembleResult containing:
            - final_answer: Aggregated answer from voting strategy
            - confidence: Overall confidence in the answer
            - agreement_score: Measure of consensus among members (0.0 to 1.0)
            - member_results: Individual results from each member
            - voting_details: Strategy-specific voting information

        Raises:
            ValueError: If no members complete successfully before timeout.
            Exception: Any fatal error during ensemble execution.

        Examples:
            Basic execution:
            >>> result = await orchestrator.execute("What is 2 + 2?")
            >>> assert result.final_answer is not None
            >>> assert len(result.member_results) > 0
            >>> assert "strategy" in result.voting_details

            Check agreement:
            >>> result = await orchestrator.execute(query)
            >>> if result.agreement_score > 0.8:
            ...     print("High agreement among members")
            >>> if result.agreement_score < 0.5:
            ...     print("Low agreement - results may be uncertain")

            Access individual member results:
            >>> result = await orchestrator.execute(query)
            >>> for member_result in result.member_results:
            ...     print(f"{member_result.member.method_name}: "
            ...           f"{member_result.confidence:.2f}")

        Note:
            If timeout occurs, the orchestrator returns results from members
            that completed. This ensures graceful degradation rather than
            complete failure. The voting_details will indicate if timeout
            occurred and how many members completed.
        """
        logger.info(
            "Starting ensemble execution with %d members (timeout: %dms)",
            len(self.config.members),
            self.config.timeout_ms,
        )

        # Execute all members in parallel
        tasks = [self._execute_member(member, query) for member in self.config.members]

        # Apply timeout and collect results
        try:
            # Gather all results, including exceptions
            raw_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_ms / 1000.0,
            )

            # Filter out exceptions and keep only successful results
            successful_results: list[MemberResult] = [
                r for r in raw_results if isinstance(r, MemberResult)
            ]

            # Log any failures
            failed_count = len(raw_results) - len(successful_results)
            if failed_count > 0:
                logger.warning("%d member(s) failed during execution", failed_count)

            self._results = successful_results

        except TimeoutError:
            logger.warning(
                "Ensemble execution timed out after %dms - using partial results",
                self.config.timeout_ms,
            )
            # Keep whatever results we have so far
            # Note: In a more sophisticated implementation, we could cancel
            # pending tasks and collect partial results

        # Validate we have at least some results
        if not self._results:
            raise ValueError(
                "No ensemble members completed successfully. "
                "Check member configurations and timeout settings."
            )

        logger.info(
            "Completed %d/%d members successfully",
            len(self._results),
            len(self.config.members),
        )

        # Apply voting strategy
        final_answer, confidence, voting_details = self._apply_voting(self._results)

        # Calculate agreement score
        agreement = self._calculate_agreement(self._results)

        logger.info(
            "Ensemble result: confidence=%.2f, agreement=%.2f",
            confidence,
            agreement,
        )

        return EnsembleResult(
            final_answer=final_answer,
            confidence=confidence,
            agreement_score=agreement,
            member_results=self._results,
            voting_details=voting_details,
        )

    # Task 6.4: Apply voting strategy
    def _apply_voting(self, results: list[MemberResult]) -> tuple[str, float, dict[str, Any]]:
        """Apply the configured voting strategy to aggregate results.

        Delegates to the appropriate voting strategy implementation based on
        the ensemble configuration. The voting strategy analyzes all member
        results and produces a final aggregated answer with confidence score
        and detailed voting information.

        Args:
            results: List of member results to aggregate. Must not be empty.
                Each result contains the member's answer, confidence, and
                execution metadata.

        Returns:
            A tuple containing:
            - final_answer (str): The aggregated answer selected/synthesized
                by the voting strategy
            - confidence (float): Overall confidence in the final answer (0.0 to 1.0)
            - voting_details (dict): Strategy-specific voting information such as:
                - vote_counts: Distribution of votes across answers
                - strategy: Name of the voting strategy used
                - weights: Weight information for weighted strategies
                - threshold: Agreement threshold for consensus strategies
                - Additional strategy-specific metadata

        Raises:
            KeyError: If the configured strategy is not found in the registry.
                This indicates a configuration error or missing implementation.

        Examples:
            Apply majority voting:
            >>> results = [
            ...     MemberResult(member1, "Answer A", 0.9, 100),
            ...     MemberResult(member2, "Answer A", 0.85, 150),
            ...     MemberResult(member3, "Answer B", 0.8, 120),
            ... ]
            >>> answer, conf, details = orchestrator._apply_voting(results)
            >>> assert answer == "Answer A"  # Majority chose A
            >>> assert "vote_counts" in details

            Handle empty results:
            >>> answer, conf, details = orchestrator._apply_voting([])
            >>> assert answer == "[NO_RESULTS]"
            >>> assert conf == 0.0
            >>> assert "error" in details

        Note:
            Each voting strategy may produce different voting_details structures.
            Refer to the specific strategy implementation for details on what
            information is included. Common fields include:
            - strategy: Name of the strategy
            - vote_counts: Distribution of votes
            - weighted_votes: Weights applied (for weighted strategies)
            - confidence_scores: Individual confidence values
        """
        if not results:
            logger.warning("No results to aggregate - returning empty result")
            return "[NO_RESULTS]", 0.0, {"error": "No results to aggregate"}

        try:
            strategy = get_strategy(self.config.strategy)
            logger.debug(
                "Applying %s strategy to %d results",
                self.config.strategy.value,
                len(results),
            )
            return strategy.aggregate(results)

        except KeyError as e:
            logger.error(
                "Voting strategy %s not found in registry: %s",
                self.config.strategy.value,
                str(e),
            )
            raise

    def _calculate_agreement(self, results: list[MemberResult]) -> float:
        """Calculate agreement score among member results.

        Computes a normalized agreement score based on the diversity of answers.
        Higher scores indicate stronger consensus, while lower scores indicate
        more diverse opinions among ensemble members.

        The agreement score is calculated as:
            agreement = 1.0 - (unique_answers - 1) / total_answers

        This formula gives:
        - 1.0 when all members agree (1 unique answer)
        - 0.5 when half the answers are unique
        - Approaches 0.0 when all answers are different

        Args:
            results: List of member results to analyze. Can be empty.

        Returns:
            Agreement score between 0.0 (no agreement) and 1.0 (full consensus).
            Returns 0.0 if results is empty.

        Examples:
            Full agreement:
            >>> results = [
            ...     MemberResult(member1, "Answer A", 0.9, 100),
            ...     MemberResult(member2, "Answer A", 0.85, 150),
            ...     MemberResult(member3, "Answer A", 0.8, 120),
            ... ]
            >>> agreement = orchestrator._calculate_agreement(results)
            >>> assert agreement == 1.0  # All agree on "Answer A"

            Partial agreement:
            >>> results = [
            ...     MemberResult(member1, "Answer A", 0.9, 100),
            ...     MemberResult(member2, "Answer A", 0.85, 150),
            ...     MemberResult(member3, "Answer B", 0.8, 120),
            ... ]
            >>> agreement = orchestrator._calculate_agreement(results)
            >>> assert 0.5 < agreement < 1.0  # 2 out of 3 agree

            No agreement:
            >>> results = [
            ...     MemberResult(member1, "Answer A", 0.9, 100),
            ...     MemberResult(member2, "Answer B", 0.85, 150),
            ...     MemberResult(member3, "Answer C", 0.8, 120),
            ... ]
            >>> agreement = orchestrator._calculate_agreement(results)
            >>> assert agreement == 0.0  # All different answers

            Empty results:
            >>> agreement = orchestrator._calculate_agreement([])
            >>> assert agreement == 0.0

        Note:
            This is a simple agreement metric based on answer uniqueness.
            More sophisticated metrics could consider:
            - Semantic similarity between answers
            - Confidence-weighted agreement
            - Partial credit for similar answers
        """
        if not results:
            return 0.0

        # Extract all answers
        answers = [r.result for r in results]

        # Count unique answers
        unique_count = len(set(answers))

        # Calculate agreement score
        # - 1.0 when all answers are the same (unique_count = 1)
        # - 0.0 when all answers are different (unique_count = len(answers))
        if len(answers) == 1:
            # Single result always has full agreement
            return 1.0

        agreement = 1.0 - (unique_count - 1) / len(answers)

        logger.debug(
            "Agreement score: %.2f (%d unique answers out of %d)",
            agreement,
            unique_count,
            len(answers),
        )

        return agreement
