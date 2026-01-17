"""Budget enforcement for reasoning operations.

This module provides functionality to enforce budget constraints
during reasoning operations, preventing cost overruns.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.models.cost import Budget, BudgetStatus, CostBreakdown


class BudgetExceededError(Exception):
    """Raised when a budget limit has been exceeded.

    Attributes:
        budget_type: Type of budget exceeded (cost, tokens, requests)
        limit: The budget limit
        current: The current usage
        message: Human-readable error message
    """

    def __init__(
        self,
        budget_type: str,
        limit: Decimal | int,
        current: Decimal | int,
        message: str | None = None,
    ) -> None:
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        self.message = message or f"Budget exceeded: {budget_type} limit {limit}, current {current}"
        super().__init__(self.message)


class BudgetEnforcer:
    """Enforces budget constraints for reasoning operations.

    Tracks usage and enforces limits on cost, tokens, and requests.
    Can be used to prevent operations that would exceed budget.

    Examples:
        >>> from reasoning_mcp.cost.enforcer import BudgetEnforcer
        >>> from reasoning_mcp.models.cost import Budget
        >>> from decimal import Decimal
        >>> budget = Budget(max_cost_usd=Decimal("1.00"))
        >>> enforcer = BudgetEnforcer(budget)
        >>> enforcer.check_budget(Decimal("0.50"))  # Returns True
        >>> enforcer.record_usage(cost_usd=Decimal("0.50"), tokens=1000)
    """

    def __init__(self, budget: Budget) -> None:
        """Initialize the enforcer with budget constraints.

        Args:
            budget: Budget constraints to enforce
        """
        self._budget = budget
        self._spent_usd = Decimal("0")
        self._tokens_used = 0
        self._requests_made = 0

    @property
    def budget(self) -> Budget:
        """Get the current budget constraints."""
        return self._budget

    @property
    def spent_usd(self) -> Decimal:
        """Get total amount spent in USD."""
        return self._spent_usd

    @property
    def tokens_used(self) -> int:
        """Get total tokens consumed."""
        return self._tokens_used

    @property
    def requests_made(self) -> int:
        """Get total requests made."""
        return self._requests_made

    def check_budget(
        self,
        estimated_cost: Decimal | None = None,
        estimated_tokens: int | None = None,
        raise_on_exceed: bool = False,
    ) -> bool:
        """Check if an operation would exceed budget limits.

        Args:
            estimated_cost: Estimated cost of the operation in USD
            estimated_tokens: Estimated tokens the operation will use
            raise_on_exceed: If True, raise BudgetExceededError instead of returning False

        Returns:
            True if the operation would be within budget, False otherwise

        Raises:
            BudgetExceededError: If raise_on_exceed is True and budget would be exceeded
        """
        # Check cost budget
        if self._budget.max_cost_usd is not None and estimated_cost is not None:
            projected = self._spent_usd + estimated_cost
            if projected > self._budget.max_cost_usd:
                if raise_on_exceed:
                    raise BudgetExceededError(
                        "cost",
                        self._budget.max_cost_usd,
                        projected,
                        f"Operation would exceed cost budget: ${projected} > ${self._budget.max_cost_usd}",
                    )
                return False

        # Check token budget
        if self._budget.max_tokens is not None and estimated_tokens is not None:
            projected_tokens: int = self._tokens_used + estimated_tokens
            if projected_tokens > self._budget.max_tokens:
                if raise_on_exceed:
                    raise BudgetExceededError(
                        "tokens",
                        self._budget.max_tokens,
                        projected_tokens,
                        f"Operation would exceed token budget: {projected_tokens} > {self._budget.max_tokens}",
                    )
                return False

        # Check request budget
        if self._budget.max_requests is not None:
            projected_requests: int = self._requests_made + 1
            if projected_requests > self._budget.max_requests:
                if raise_on_exceed:
                    raise BudgetExceededError(
                        "requests",
                        self._budget.max_requests,
                        projected_requests,
                        f"Operation would exceed request budget: {projected_requests} > {self._budget.max_requests}",
                    )
                return False

        return True

    def record_usage(
        self,
        cost_usd: Decimal | None = None,
        tokens: int | None = None,
        increment_requests: bool = True,
    ) -> None:
        """Record resource usage after an operation.

        Args:
            cost_usd: Cost of the operation in USD
            tokens: Number of tokens consumed
            increment_requests: Whether to increment the request counter
        """
        if cost_usd is not None:
            self._spent_usd += cost_usd
        if tokens is not None:
            self._tokens_used += tokens
        if increment_requests:
            self._requests_made += 1

    def record_breakdown(self, breakdown: CostBreakdown) -> None:
        """Record usage from a cost breakdown.

        Args:
            breakdown: Cost breakdown from a completed operation
        """
        self.record_usage(
            cost_usd=breakdown.total_cost,
            tokens=breakdown.tokens.total_tokens if breakdown.tokens else None,
            increment_requests=True,
        )

    def get_status(self) -> BudgetStatus:
        """Get the current budget status.

        Returns:
            BudgetStatus with current usage and remaining budget
        """
        from reasoning_mcp.models.cost import BudgetStatus

        remaining_usd = None
        remaining_tokens = None
        remaining_requests = None
        is_exceeded = False

        if self._budget.max_cost_usd is not None:
            remaining_usd = self._budget.max_cost_usd - self._spent_usd
            if remaining_usd < 0:
                is_exceeded = True

        if self._budget.max_tokens is not None:
            remaining_tokens = self._budget.max_tokens - self._tokens_used
            if remaining_tokens < 0:
                is_exceeded = True

        if self._budget.max_requests is not None:
            remaining_requests = self._budget.max_requests - self._requests_made
            if remaining_requests < 0:
                is_exceeded = True

        return BudgetStatus(
            budget=self._budget,
            spent_usd=self._spent_usd,
            tokens_used=self._tokens_used,
            requests_made=self._requests_made,
            remaining_usd=remaining_usd,
            remaining_tokens=remaining_tokens,
            remaining_requests=remaining_requests,
            is_exceeded=is_exceeded,
        )

    def reset(self) -> None:
        """Reset all usage counters to zero."""
        self._spent_usd = Decimal("0")
        self._tokens_used = 0
        self._requests_made = 0
