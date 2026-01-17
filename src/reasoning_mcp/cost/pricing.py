"""Pricing registry for model cost tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.models.cost import ModelPricing


class PricingRegistry:
    """Registry for model pricing information."""

    def __init__(self) -> None:
        self._pricing: dict[str, ModelPricing] = {}

    def register(self, pricing: ModelPricing) -> None:
        """Register pricing for a model."""
        self._pricing[pricing.model_id] = pricing

    def get(self, model_id: str) -> ModelPricing | None:
        """Get pricing for a model."""
        return self._pricing.get(model_id)
