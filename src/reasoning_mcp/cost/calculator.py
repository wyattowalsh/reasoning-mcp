"""Cost calculator for reasoning operations.

This module provides functionality to calculate and estimate costs
for reasoning operations based on model pricing and token usage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from reasoning_mcp.cost.estimator import METHOD_TOKEN_MULTIPLIERS

if TYPE_CHECKING:
    from reasoning_mcp.cost.pricing import PricingRegistry
    from reasoning_mcp.models.cost import (
        CostBreakdown,
        CostEstimate,
        TokenCount,
    )


class CostCalculator:
    """Calculator for reasoning operation costs.

    Provides methods to estimate costs before execution and calculate
    actual costs after execution based on token usage.

    Examples:
        >>> from reasoning_mcp.cost.calculator import CostCalculator
        >>> from reasoning_mcp.cost.pricing import PricingRegistry
        >>> registry = PricingRegistry()
        >>> calc = CostCalculator(registry)
        >>> estimate = calc.estimate_cost("chain_of_thought", "Hello", "claude-3-5-sonnet")
    """

    def __init__(self, pricing_registry: PricingRegistry) -> None:
        """Initialize the calculator with a pricing registry.

        Args:
            pricing_registry: Registry containing model pricing information
        """
        self._registry = pricing_registry

    def estimate_cost(
        self,
        method: str,
        input_text: str,
        model_id: str,
        estimated_output_multiplier: float = 1.5,
    ) -> CostEstimate:
        """Estimate the cost of a reasoning operation before execution.

        Args:
            method: The reasoning method to use
            input_text: The input text/prompt
            model_id: The model identifier
            estimated_output_multiplier: Multiplier for estimating output tokens

        Returns:
            CostEstimate with predicted costs

        Raises:
            ValueError: If model pricing is not found
        """
        from reasoning_mcp.models.cost import CostBreakdown, CostEstimate, TokenCount

        pricing = self._registry.get(model_id)
        if pricing is None:
            raise ValueError(f"No pricing found for model: {model_id}")

        # Estimate input tokens (rough: ~4 chars per token)
        base_input_tokens = max(1, len(input_text) // 4)

        # Get method multiplier
        method_multiplier = METHOD_TOKEN_MULTIPLIERS.get(
            method, METHOD_TOKEN_MULTIPLIERS["default"]
        )

        # Apply method multiplier to get adjusted token counts
        estimated_input = int(base_input_tokens * method_multiplier)
        estimated_output = int(base_input_tokens * estimated_output_multiplier * method_multiplier)
        estimated_total = estimated_input + estimated_output

        # Calculate costs
        input_cost = (Decimal(estimated_input) / 1000) * pricing.input_price_per_1k
        output_cost = (Decimal(estimated_output) / 1000) * pricing.output_price_per_1k
        total_cost = input_cost + output_cost

        breakdown = CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            tokens=TokenCount(input_tokens=estimated_input, output_tokens=estimated_output),
            model_id=model_id,
            method=method,
        )

        return CostEstimate(
            method=method,
            model_id=model_id,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            estimated_total_tokens=estimated_total,
            estimated_cost_usd=total_cost,
            confidence=0.7,  # Lower confidence for estimates
            breakdown=breakdown,
        )

    def calculate_cost(
        self,
        tokens: TokenCount,
        model_id: str,
        method: str | None = None,
    ) -> CostBreakdown:
        """Calculate the actual cost based on token usage.

        Args:
            tokens: Actual token counts from the operation
            model_id: The model used
            method: Optional reasoning method used

        Returns:
            CostBreakdown with actual costs

        Raises:
            ValueError: If model pricing is not found
        """
        from reasoning_mcp.models.cost import CostBreakdown

        pricing = self._registry.get(model_id)
        if pricing is None:
            raise ValueError(f"No pricing found for model: {model_id}")

        input_cost = (Decimal(tokens.input_tokens) / 1000) * pricing.input_price_per_1k
        output_cost = (Decimal(tokens.output_tokens) / 1000) * pricing.output_price_per_1k
        total_cost = input_cost + output_cost

        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            tokens=tokens,
            model_id=model_id,
            method=method,
        )

    def compare_methods(
        self,
        input_text: str,
        model_id: str,
        methods: list[str] | None = None,
    ) -> list[CostEstimate]:
        """Compare estimated costs across multiple reasoning methods.

        Args:
            input_text: The input text/prompt
            model_id: The model to use
            methods: List of methods to compare (defaults to all known methods)

        Returns:
            List of CostEstimate objects sorted by cost (cheapest first)
        """
        if methods is None:
            methods = list(METHOD_TOKEN_MULTIPLIERS.keys())
            methods = [m for m in methods if m != "default"]

        estimates = []
        for method in methods:
            try:
                estimate = self.estimate_cost(method, input_text, model_id)
                estimates.append(estimate)
            except ValueError:
                continue  # Skip if pricing not found

        return sorted(estimates, key=lambda e: e.estimated_cost_usd)

    def get_cheapest_method(
        self,
        input_text: str,
        model_id: str,
        methods: list[str] | None = None,
    ) -> CostEstimate | None:
        """Get the cheapest method for a given input.

        Args:
            input_text: The input text/prompt
            model_id: The model to use
            methods: List of methods to consider

        Returns:
            CostEstimate for the cheapest method, or None if no estimates available
        """
        estimates = self.compare_methods(input_text, model_id, methods)
        return estimates[0] if estimates else None
