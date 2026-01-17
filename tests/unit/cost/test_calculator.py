"""Tests for cost calculator module."""

from decimal import Decimal

import pytest

from reasoning_mcp.cost.calculator import CostCalculator
from reasoning_mcp.cost.pricing import PricingRegistry
from reasoning_mcp.models.cost import ModelPricing, TokenCount


@pytest.fixture
def pricing_registry():
    """Create a pricing registry with test pricing."""
    registry = PricingRegistry()
    registry.register(
        ModelPricing(
            model_id="test-model",
            input_price_per_1k=Decimal("0.003"),
            output_price_per_1k=Decimal("0.015"),
            context_window=128000,
        )
    )
    registry.register(
        ModelPricing(
            model_id="expensive-model",
            input_price_per_1k=Decimal("0.015"),
            output_price_per_1k=Decimal("0.075"),
            context_window=200000,
        )
    )
    return registry


@pytest.fixture
def calculator(pricing_registry):
    """Create a cost calculator with test pricing."""
    return CostCalculator(pricing_registry)


def test_calculator_init(pricing_registry):
    """Test calculator initialization."""
    calc = CostCalculator(pricing_registry)
    assert calc._registry is pricing_registry


def test_estimate_cost_basic(calculator):
    """Test basic cost estimation."""
    estimate = calculator.estimate_cost(
        method="chain_of_thought",
        input_text="What is 2 + 2?",
        model_id="test-model",
    )

    assert estimate.method == "chain_of_thought"
    assert estimate.model_id == "test-model"
    assert estimate.estimated_input_tokens > 0
    assert estimate.estimated_output_tokens > 0
    assert (
        estimate.estimated_total_tokens
        == estimate.estimated_input_tokens + estimate.estimated_output_tokens
    )
    assert estimate.estimated_cost_usd > Decimal("0")
    assert 0 <= estimate.confidence <= 1


def test_estimate_cost_different_methods(calculator):
    """Test that different methods have different costs."""
    input_text = "This is a test input for cost estimation."

    cot_estimate = calculator.estimate_cost("chain_of_thought", input_text, "test-model")
    mcts_estimate = calculator.estimate_cost("mcts", input_text, "test-model")

    # MCTS should be more expensive (higher multiplier)
    assert mcts_estimate.estimated_cost_usd > cot_estimate.estimated_cost_usd


def test_estimate_cost_unknown_model(calculator):
    """Test that unknown model raises ValueError."""
    with pytest.raises(ValueError, match="No pricing found"):
        calculator.estimate_cost("chain_of_thought", "test", "unknown-model")


def test_calculate_cost_basic(calculator):
    """Test actual cost calculation."""
    tokens = TokenCount(input_tokens=1000, output_tokens=500)
    breakdown = calculator.calculate_cost(tokens, "test-model", "chain_of_thought")

    assert breakdown.input_cost == Decimal("0.003")  # 1000 * 0.003 / 1000
    assert breakdown.output_cost == Decimal("0.0075")  # 500 * 0.015 / 1000
    assert breakdown.total_cost == Decimal("0.0105")
    assert breakdown.tokens == tokens
    assert breakdown.model_id == "test-model"
    assert breakdown.method == "chain_of_thought"


def test_calculate_cost_unknown_model(calculator):
    """Test that unknown model raises ValueError."""
    tokens = TokenCount(input_tokens=100, output_tokens=50)
    with pytest.raises(ValueError, match="No pricing found"):
        calculator.calculate_cost(tokens, "unknown-model")


def test_compare_methods(calculator):
    """Test method cost comparison."""
    comparisons = calculator.compare_methods(
        input_text="Test input for comparison",
        model_id="test-model",
    )

    assert len(comparisons) > 0
    # Should be sorted by cost (cheapest first)
    for i in range(len(comparisons) - 1):
        assert comparisons[i].estimated_cost_usd <= comparisons[i + 1].estimated_cost_usd


def test_compare_methods_specific(calculator):
    """Test comparing specific methods."""
    comparisons = calculator.compare_methods(
        input_text="Test input",
        model_id="test-model",
        methods=["chain_of_thought", "mcts"],
    )

    assert len(comparisons) == 2
    methods = {c.method for c in comparisons}
    assert "chain_of_thought" in methods
    assert "mcts" in methods


def test_get_cheapest_method(calculator):
    """Test getting the cheapest method."""
    cheapest = calculator.get_cheapest_method(
        input_text="Test input",
        model_id="test-model",
    )

    assert cheapest is not None
    # sequential has the lowest multiplier (1.5)
    assert cheapest.method == "sequential"


def test_get_cheapest_method_specific(calculator):
    """Test getting cheapest from specific methods."""
    cheapest = calculator.get_cheapest_method(
        input_text="Test input",
        model_id="test-model",
        methods=["chain_of_thought", "mcts"],
    )

    assert cheapest is not None
    assert cheapest.method == "chain_of_thought"  # Lower multiplier than mcts
