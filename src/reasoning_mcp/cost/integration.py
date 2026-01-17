"""Cost integration utilities.

This module provides integration utilities for cost-aware execution,
including context managers and executors that combine cost calculation,
budget enforcement, and session tracking.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from reasoning_mcp.cost.calculator import CostCalculator
    from reasoning_mcp.cost.enforcer import BudgetEnforcer
    from reasoning_mcp.cost.tracker import SessionCostTracker
    from reasoning_mcp.models.cost import CostBreakdown, CostEstimate


class CostAwareExecutor:
    """Executor that integrates cost calculation, budget enforcement, and tracking.

    CostAwareExecutor combines the functionality of CostCalculator, BudgetEnforcer,
    and SessionCostTracker to provide a unified interface for cost-aware reasoning
    execution.

    Attributes:
        calculator: Cost calculator for estimating and computing costs
        enforcer: Optional budget enforcer for preventing budget overruns
        tracker: Optional session cost tracker for recording costs

    Examples:
        Create executor with just calculator:
        >>> calculator = CostCalculator(registry, estimator)
        >>> executor = CostAwareExecutor(calculator)

        Create executor with all components:
        >>> executor = CostAwareExecutor(
        ...     calculator=calculator,
        ...     enforcer=enforcer,
        ...     tracker=tracker,
        ... )
    """

    def __init__(
        self,
        calculator: CostCalculator | None = None,
        enforcer: BudgetEnforcer | None = None,
        tracker: SessionCostTracker | None = None,
    ) -> None:
        """Initialize cost-aware executor.

        Args:
            calculator: Cost calculator for estimating and computing costs
            enforcer: Optional budget enforcer for preventing budget overruns
            tracker: Optional session cost tracker for recording costs
        """
        self.calculator = calculator
        self.enforcer = enforcer
        self.tracker = tracker
        self._current_estimate: CostEstimate | None = None

    def pre_execute(
        self,
        method: str,
        input_text: str,
        model_id: str,
        check_budget: bool = True,
    ) -> CostEstimate | None:
        """Called before reasoning execution.

        Estimates costs and optionally checks budget constraints.

        Args:
            method: Reasoning method to use
            input_text: Input text/prompt
            model_id: Model to use
            check_budget: Whether to check budget constraints

        Returns:
            Cost estimate if calculator available, None otherwise

        Raises:
            BudgetExceededError: If check_budget is True and budget would be exceeded
        """
        estimate = None

        if self.calculator:
            try:
                estimate = self.calculator.estimate_cost(method, input_text, model_id)
                self._current_estimate = estimate
            except ValueError:
                pass  # No pricing available

        if check_budget and self.enforcer and estimate:
            self.enforcer.check_budget(
                estimated_cost=estimate.estimated_cost_usd,
                estimated_tokens=estimate.estimated_total_tokens,
                raise_on_exceed=True,
            )

        return estimate

    def post_execute(
        self,
        breakdown: CostBreakdown,
        operation_id: str | None = None,
    ) -> None:
        """Called after reasoning execution completes.

        Records actual costs to enforcer and tracker.

        Args:
            breakdown: Actual cost breakdown from the operation
            operation_id: Optional identifier for the operation
        """
        if self.enforcer:
            self.enforcer.record_breakdown(breakdown)

        if self.tracker:
            self.tracker.add_cost(breakdown, operation_id)

        self._current_estimate = None

    @contextmanager
    def execution_context(
        self,
        method: str,
        input_text: str,
        model_id: str,
        check_budget: bool = True,
        operation_id: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for cost-aware execution.

        Handles pre and post execution hooks automatically.
        The context dict contains the estimate and can be updated with the breakdown.

        Args:
            method: Reasoning method to use
            input_text: Input text/prompt
            model_id: Model to use
            check_budget: Whether to check budget constraints
            operation_id: Optional operation identifier

        Yields:
            Dictionary with 'estimate' key, to be updated with 'breakdown' after execution
        """
        ctx: dict[str, Any] = {
            "estimate": None,
            "breakdown": None,
            "method": method,
            "model_id": model_id,
        }

        # Pre-execution
        estimate = self.pre_execute(method, input_text, model_id, check_budget)
        ctx["estimate"] = estimate

        try:
            yield ctx
        finally:
            # Post-execution (if breakdown provided)
            if ctx.get("breakdown"):
                self.post_execute(ctx["breakdown"], operation_id)


@contextmanager
def cost_context(
    calculator: CostCalculator | None = None,
    enforcer: BudgetEnforcer | None = None,
    tracker: SessionCostTracker | None = None,
) -> Generator[CostAwareExecutor, None, None]:
    """Context manager for cost-aware execution.

    This context manager creates a CostAwareExecutor with the provided
    calculator, enforcer, and tracker, and yields it for use within the context.

    Args:
        calculator: Cost calculator for estimating and computing costs
        enforcer: Optional budget enforcer for preventing budget overruns
        tracker: Optional session cost tracker for recording costs

    Yields:
        CostAwareExecutor configured with the provided components

    Examples:
        Basic usage:
        >>> calculator = CostCalculator(registry, estimator)
        >>> with cost_context(calculator) as executor:
        ...     # Use executor for cost-aware reasoning
        ...     pass

        With budget enforcement:
        >>> with cost_context(calculator, enforcer=enforcer) as executor:
        ...     # Budget will be enforced
        ...     pass

        With full tracking:
        >>> with cost_context(calculator, enforcer, tracker) as executor:
        ...     # Costs will be tracked and budgets enforced
        ...     pass
    """
    executor = CostAwareExecutor(calculator, enforcer, tracker)
    yield executor
