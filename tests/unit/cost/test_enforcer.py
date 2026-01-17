"""Tests for budget enforcer module."""

from decimal import Decimal

import pytest

from reasoning_mcp.cost.enforcer import BudgetEnforcer, BudgetExceededError
from reasoning_mcp.models.cost import Budget, CostBreakdown, TokenCount


@pytest.fixture
def budget():
    """Create a test budget."""
    return Budget(
        max_cost_usd=Decimal("1.00"),
        max_tokens=10000,
        max_requests=100,
    )


@pytest.fixture
def enforcer(budget):
    """Create a budget enforcer."""
    return BudgetEnforcer(budget)


def test_enforcer_init(budget):
    """Test enforcer initialization."""
    enforcer = BudgetEnforcer(budget)
    assert enforcer.budget == budget
    assert enforcer.spent_usd == Decimal("0")
    assert enforcer.tokens_used == 0
    assert enforcer.requests_made == 0


def test_check_budget_within_limits(enforcer):
    """Test budget check when within limits."""
    assert enforcer.check_budget(
        estimated_cost=Decimal("0.50"),
        estimated_tokens=5000,
    )


def test_check_budget_exceeds_cost(enforcer):
    """Test budget check when cost exceeds limit."""
    assert not enforcer.check_budget(
        estimated_cost=Decimal("1.50"),
        estimated_tokens=5000,
    )


def test_check_budget_exceeds_tokens(enforcer):
    """Test budget check when tokens exceed limit."""
    assert not enforcer.check_budget(
        estimated_cost=Decimal("0.50"),
        estimated_tokens=15000,
    )


def test_check_budget_raises_on_exceed(enforcer):
    """Test that check_budget can raise on exceed."""
    with pytest.raises(BudgetExceededError) as exc_info:
        enforcer.check_budget(
            estimated_cost=Decimal("1.50"),
            raise_on_exceed=True,
        )
    assert exc_info.value.budget_type == "cost"


def test_record_usage(enforcer):
    """Test recording usage."""
    enforcer.record_usage(cost_usd=Decimal("0.25"), tokens=1000)

    assert enforcer.spent_usd == Decimal("0.25")
    assert enforcer.tokens_used == 1000
    assert enforcer.requests_made == 1


def test_record_usage_accumulates(enforcer):
    """Test that usage accumulates."""
    enforcer.record_usage(cost_usd=Decimal("0.25"), tokens=1000)
    enforcer.record_usage(cost_usd=Decimal("0.25"), tokens=1000)

    assert enforcer.spent_usd == Decimal("0.50")
    assert enforcer.tokens_used == 2000
    assert enforcer.requests_made == 2


def test_record_breakdown(enforcer):
    """Test recording from a breakdown."""
    breakdown = CostBreakdown(
        input_cost=Decimal("0.05"),
        output_cost=Decimal("0.10"),
        total_cost=Decimal("0.15"),
        tokens=TokenCount(input_tokens=500, output_tokens=300),
    )

    enforcer.record_breakdown(breakdown)

    assert enforcer.spent_usd == Decimal("0.15")
    assert enforcer.tokens_used == 800
    assert enforcer.requests_made == 1


def test_get_status(enforcer):
    """Test getting budget status."""
    enforcer.record_usage(cost_usd=Decimal("0.50"), tokens=5000)

    status = enforcer.get_status()

    assert status.budget == enforcer.budget
    assert status.spent_usd == Decimal("0.50")
    assert status.tokens_used == 5000
    assert status.requests_made == 1
    assert status.remaining_usd == Decimal("0.50")
    assert status.remaining_tokens == 5000
    assert status.remaining_requests == 99
    assert not status.is_exceeded
    assert status.utilization_percent == 50.0


def test_get_status_exceeded(enforcer):
    """Test status when budget is exceeded."""
    enforcer.record_usage(cost_usd=Decimal("1.50"), tokens=15000)

    status = enforcer.get_status()

    assert status.is_exceeded
    assert status.remaining_usd < 0


def test_reset(enforcer):
    """Test resetting the enforcer."""
    enforcer.record_usage(cost_usd=Decimal("0.50"), tokens=5000)
    enforcer.reset()

    assert enforcer.spent_usd == Decimal("0")
    assert enforcer.tokens_used == 0
    assert enforcer.requests_made == 0


def test_budget_exceeded_error():
    """Test BudgetExceededError."""
    error = BudgetExceededError("cost", Decimal("1.00"), Decimal("1.50"))
    assert error.budget_type == "cost"
    assert error.limit == Decimal("1.00")
    assert error.current == Decimal("1.50")
    assert "cost" in str(error)


def test_check_budget_partial_limits():
    """Test budget with only some limits set."""
    budget = Budget(max_cost_usd=Decimal("1.00"))
    enforcer = BudgetEnforcer(budget)

    # Should pass - no token limit
    assert enforcer.check_budget(estimated_tokens=1000000)

    # Should fail - exceeds cost
    assert not enforcer.check_budget(estimated_cost=Decimal("2.00"))
