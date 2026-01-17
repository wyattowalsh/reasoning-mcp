"""Cost estimation and budget management tools for reasoning-mcp.

This module provides MCP tools for:
- Estimating computational costs before execution
- Checking budget status
- Comparing method costs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from reasoning_mcp.cost.calculator import CostCalculator
    from reasoning_mcp.cost.enforcer import BudgetEnforcer


class EstimateCostInput(BaseModel):
    """Input for cost estimation tool.

    EstimateCostInput allows clients to request cost estimates for reasoning
    operations before executing them, enabling better resource planning and
    budget management.

    Examples:
        Estimate cost for a specific method:
        >>> input = EstimateCostInput(
        ...     method="chain_of_thought",
        ...     input_text="What is the meaning of life?",
        ... )

        Estimate with specific model:
        >>> input = EstimateCostInput(
        ...     method="tree_of_thoughts",
        ...     input_text="Solve this complex optimization problem",
        ...     model_id="claude-opus-4-5-20251101"
        ... )

        Estimate for auto-selected method:
        >>> input = EstimateCostInput(
        ...     method="auto",
        ...     input_text="Analyze this ethical dilemma",
        ...     model_id="claude-sonnet-4-5-20250929"
        ... )
    """

    method: str = Field(description="Reasoning method to estimate cost for")
    input_text: str = Field(description="Input text/prompt for the operation")
    model_id: str | None = Field(
        default=None,
        description="Model to use (defaults to configured model)",
    )


class EstimateCostOutput(BaseModel):
    """Output from cost estimation tool."""

    method: str = Field(description="Reasoning method")
    model_id: str = Field(description="Model used for estimate")
    estimated_input_tokens: int = Field(description="Estimated input tokens")
    estimated_output_tokens: int = Field(description="Estimated output tokens")
    estimated_total_tokens: int = Field(description="Estimated total tokens")
    estimated_cost_usd: str = Field(description="Estimated cost in USD")
    confidence: float = Field(description="Confidence level (0-1)")


class GetBudgetStatusInput(BaseModel):
    """Input for budget status tool."""

    session_id: str | None = Field(
        default=None,
        description="Optional session ID to get status for",
    )


class GetBudgetStatusOutput(BaseModel):
    """Output from budget status tool."""

    has_budget: bool = Field(description="Whether a budget is configured")
    max_cost_usd: str | None = Field(description="Maximum budget in USD if set")
    max_tokens: int | None = Field(description="Maximum tokens if set")
    max_requests: int | None = Field(description="Maximum requests if set")
    spent_usd: str = Field(description="Amount spent in USD")
    tokens_used: int = Field(description="Tokens consumed")
    requests_made: int = Field(description="Requests made")
    remaining_usd: str | None = Field(description="Remaining budget in USD")
    remaining_tokens: int | None = Field(description="Remaining token budget")
    remaining_requests: int | None = Field(description="Remaining request budget")
    utilization_percent: float | None = Field(description="Budget utilization percentage")
    is_exceeded: bool = Field(description="Whether budget is exceeded")


class CompareCostsInput(BaseModel):
    """Input for cost comparison tool."""

    input_text: str = Field(description="Input text/prompt")
    model_id: str | None = Field(default=None, description="Model to use")
    methods: list[str] | None = Field(
        default=None,
        description="Methods to compare (defaults to all)",
    )
    top_n: int = Field(
        default=5,
        description="Number of results to return",
    )


class MethodCostComparison(BaseModel):
    """Cost comparison for a single method."""

    method: str = Field(description="Reasoning method")
    estimated_cost_usd: str = Field(description="Estimated cost in USD")
    estimated_tokens: int = Field(description="Estimated total tokens")
    rank: int = Field(description="Rank (1 = cheapest)")


class CompareCostsOutput(BaseModel):
    """Output from cost comparison tool."""

    model_id: str = Field(description="Model used for comparison")
    comparisons: list[MethodCostComparison] = Field(description="Cost comparisons")
    cheapest_method: str = Field(description="Most cost-effective method")


def estimate_cost_handler(
    input_data: EstimateCostInput,
    calculator: CostCalculator,
    default_model: str = "claude-3-5-sonnet-latest",
) -> EstimateCostOutput:
    """Handle cost estimation request.

    Args:
        input_data: Cost estimation input
        calculator: Cost calculator instance
        default_model: Default model if none specified

    Returns:
        Cost estimation output
    """
    model_id = input_data.model_id or default_model

    estimate = calculator.estimate_cost(
        method=input_data.method,
        input_text=input_data.input_text,
        model_id=model_id,
    )

    return EstimateCostOutput(
        method=estimate.method,
        model_id=estimate.model_id,
        estimated_input_tokens=estimate.estimated_input_tokens,
        estimated_output_tokens=estimate.estimated_output_tokens,
        estimated_total_tokens=estimate.estimated_total_tokens,
        estimated_cost_usd=str(estimate.estimated_cost_usd),
        confidence=estimate.confidence,
    )


def get_budget_status_handler(
    input_data: GetBudgetStatusInput,
    enforcer: BudgetEnforcer | None,
) -> GetBudgetStatusOutput:
    """Handle budget status request.

    Args:
        input_data: Budget status input
        enforcer: Budget enforcer instance (may be None)

    Returns:
        Budget status output
    """
    if enforcer is None:
        return GetBudgetStatusOutput(
            has_budget=False,
            max_cost_usd=None,
            max_tokens=None,
            max_requests=None,
            spent_usd="0",
            tokens_used=0,
            requests_made=0,
            remaining_usd=None,
            remaining_tokens=None,
            remaining_requests=None,
            utilization_percent=None,
            is_exceeded=False,
        )

    status = enforcer.get_status()
    budget = status.budget

    return GetBudgetStatusOutput(
        has_budget=True,
        max_cost_usd=str(budget.max_cost_usd) if budget.max_cost_usd else None,
        max_tokens=budget.max_tokens,
        max_requests=budget.max_requests,
        spent_usd=str(status.spent_usd),
        tokens_used=status.tokens_used,
        requests_made=status.requests_made,
        remaining_usd=str(status.remaining_usd) if status.remaining_usd else None,
        remaining_tokens=status.remaining_tokens,
        remaining_requests=status.remaining_requests,
        utilization_percent=status.utilization_percent,
        is_exceeded=status.is_exceeded,
    )


def compare_costs_handler(
    input_data: CompareCostsInput,
    calculator: CostCalculator,
    default_model: str = "claude-3-5-sonnet-latest",
) -> CompareCostsOutput:
    """Handle cost comparison request.

    Args:
        input_data: Cost comparison input
        calculator: Cost calculator instance
        default_model: Default model if none specified

    Returns:
        Cost comparison output
    """
    model_id = input_data.model_id or default_model

    estimates = calculator.compare_methods(
        input_text=input_data.input_text,
        model_id=model_id,
        methods=input_data.methods,
    )

    # Take top N
    top_estimates = estimates[: input_data.top_n]

    comparisons = [
        MethodCostComparison(
            method=est.method,
            estimated_cost_usd=str(est.estimated_cost_usd),
            estimated_tokens=est.estimated_total_tokens,
            rank=i + 1,
        )
        for i, est in enumerate(top_estimates)
    ]

    return CompareCostsOutput(
        model_id=model_id,
        comparisons=comparisons,
        cheapest_method=comparisons[0].method if comparisons else "unknown",
    )
