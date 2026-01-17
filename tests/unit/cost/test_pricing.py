"""Tests for pricing registry module."""

from decimal import Decimal

from reasoning_mcp.cost.pricing import PricingRegistry
from reasoning_mcp.models.cost import ModelPricing


def test_pricing_registry():
    """Test basic PricingRegistry functionality."""
    # Create a pricing registry
    registry = PricingRegistry()

    # Create a model pricing instance
    pricing = ModelPricing(
        model_id="gpt-4",
        input_price_per_1k=Decimal("0.03"),
        output_price_per_1k=Decimal("0.06"),
        context_window=8192,
    )

    # Register the pricing
    registry.register(pricing)

    # Retrieve the pricing
    retrieved = registry.get("gpt-4")
    assert retrieved is not None
    assert retrieved.model_id == "gpt-4"
    assert retrieved.input_price_per_1k == Decimal("0.03")
    assert retrieved.output_price_per_1k == Decimal("0.06")
    assert retrieved.context_window == 8192

    # Test non-existent model
    not_found = registry.get("non-existent-model")
    assert not_found is None


def test_pricing_registry_multiple_models():
    """Test registry with multiple models."""
    registry = PricingRegistry()

    # Register multiple models
    gpt4_pricing = ModelPricing(
        model_id="gpt-4",
        input_price_per_1k=Decimal("0.03"),
        output_price_per_1k=Decimal("0.06"),
        context_window=8192,
    )

    claude_pricing = ModelPricing(
        model_id="claude-3-opus",
        input_price_per_1k=Decimal("0.015"),
        output_price_per_1k=Decimal("0.075"),
        context_window=200000,
    )

    registry.register(gpt4_pricing)
    registry.register(claude_pricing)

    # Verify both can be retrieved
    assert registry.get("gpt-4") is not None
    assert registry.get("claude-3-opus") is not None
    assert registry.get("gpt-4").model_id == "gpt-4"
    assert registry.get("claude-3-opus").model_id == "claude-3-opus"


def test_pricing_registry_update():
    """Test updating pricing for existing model."""
    registry = PricingRegistry()

    # Register initial pricing
    initial = ModelPricing(
        model_id="gpt-4",
        input_price_per_1k=Decimal("0.03"),
        output_price_per_1k=Decimal("0.06"),
        context_window=8192,
    )
    registry.register(initial)

    # Update with new pricing
    updated = ModelPricing(
        model_id="gpt-4",
        input_price_per_1k=Decimal("0.025"),
        output_price_per_1k=Decimal("0.05"),
        context_window=8192,
    )
    registry.register(updated)

    # Verify the pricing was updated
    retrieved = registry.get("gpt-4")
    assert retrieved is not None
    assert retrieved.input_price_per_1k == Decimal("0.025")
    assert retrieved.output_price_per_1k == Decimal("0.05")
