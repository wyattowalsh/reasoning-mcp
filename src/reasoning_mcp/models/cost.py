"""Cost tracking models for reasoning operations.

This module defines models for tracking resource usage and costs associated
with reasoning operations, including token counts and computed metrics.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class TokenCount(BaseModel):
    """Token usage tracking for reasoning operations.

    Tracks input and output token counts and provides computed total tokens.
    Useful for monitoring API usage, calculating costs, and analyzing
    reasoning efficiency.

    Examples:
        Create a token count:
        >>> tokens = TokenCount(input_tokens=100, output_tokens=50)
        >>> print(f"Total: {tokens.total_tokens}")
        Total: 150

        Serialize to dict:
        >>> data = tokens.model_dump()
        >>> assert data["total_tokens"] == 150
    """

    input_tokens: int = Field(
        ge=0,
        description="Number of tokens in the input/prompt",
    )
    output_tokens: int = Field(
        ge=0,
        description="Number of tokens in the output/completion",
    )

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Compute the total token count.

        Returns:
            Sum of input_tokens and output_tokens

        Examples:
            >>> tokens = TokenCount(input_tokens=100, output_tokens=50)
            >>> assert tokens.total_tokens == 150
        """
        return self.input_tokens + self.output_tokens


class PricingTier(BaseModel):
    """Pricing tier for a model."""

    name: str
    input_price_per_1k: Decimal
    output_price_per_1k: Decimal
    context_window: int


class ModelPricing(BaseModel):
    """Pricing information for a specific model."""

    model_id: str = Field(description="Unique identifier for the model")
    input_price_per_1k: Decimal = Field(description="Price per 1000 input tokens in USD")
    output_price_per_1k: Decimal = Field(description="Price per 1000 output tokens in USD")
    context_window: int = Field(description="Maximum context window size in tokens")


class CostUnit(str, Enum):
    """Unit of measurement for costs."""

    USD = "usd"
    TOKENS = "tokens"
    REQUESTS = "requests"


class CostBreakdown(BaseModel):
    """Detailed breakdown of costs for a reasoning operation."""

    input_cost: Decimal = Field(
        default=Decimal("0"),
        description="Cost attributed to input tokens",
    )
    output_cost: Decimal = Field(
        default=Decimal("0"),
        description="Cost attributed to output tokens",
    )
    total_cost: Decimal = Field(
        default=Decimal("0"),
        description="Total cost in USD",
    )
    tokens: TokenCount | None = Field(
        default=None,
        description="Token counts if available",
    )
    model_id: str | None = Field(
        default=None,
        description="Model used for the operation",
    )
    method: str | None = Field(
        default=None,
        description="Reasoning method used",
    )


class Budget(BaseModel):
    """Budget constraints for reasoning operations."""

    max_cost_usd: Decimal | None = None
    max_tokens: int | None = None
    max_requests: int | None = None
    period: Literal["request", "session", "daily"] = "request"


class BudgetStatus(BaseModel):
    """Current status of budget consumption."""

    budget: Budget = Field(description="The budget constraints")
    spent_usd: Decimal = Field(
        default=Decimal("0"),
        description="Amount spent in USD",
    )
    tokens_used: int = Field(
        default=0,
        description="Number of tokens consumed",
    )
    requests_made: int = Field(
        default=0,
        description="Number of requests made",
    )
    remaining_usd: Decimal | None = Field(
        default=None,
        description="Remaining budget in USD if applicable",
    )
    remaining_tokens: int | None = Field(
        default=None,
        description="Remaining token budget if applicable",
    )
    remaining_requests: int | None = Field(
        default=None,
        description="Remaining request budget if applicable",
    )
    is_exceeded: bool = Field(
        default=False,
        description="Whether any budget limit has been exceeded",
    )

    @computed_field
    @property
    def utilization_percent(self) -> float | None:
        """Calculate budget utilization percentage based on cost."""
        if self.budget.max_cost_usd is None or self.budget.max_cost_usd == 0:
            return None
        return float(self.spent_usd / self.budget.max_cost_usd * 100)


class CostEstimate(BaseModel):
    """Estimated cost for a reasoning operation before execution."""

    method: str = Field(description="Reasoning method to be used")
    model_id: str = Field(description="Model to be used")
    estimated_input_tokens: int = Field(
        ge=0,
        description="Estimated input tokens",
    )
    estimated_output_tokens: int = Field(
        ge=0,
        description="Estimated output tokens",
    )
    estimated_total_tokens: int = Field(
        ge=0,
        description="Estimated total tokens",
    )
    estimated_cost_usd: Decimal = Field(
        description="Estimated cost in USD",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Confidence level in the estimate (0-1)",
    )
    breakdown: CostBreakdown | None = Field(
        default=None,
        description="Detailed cost breakdown if available",
    )
