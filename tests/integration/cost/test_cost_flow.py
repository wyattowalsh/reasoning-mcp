"""Integration tests for complete cost tracking flow."""

from decimal import Decimal

import pytest

from reasoning_mcp.cost import (
    AlertManager,
    BudgetEnforcer,
    BudgetExceededError,
    CostAlertType,
    CostAwareExecutor,
    CostCalculator,
    PricingRegistry,
    SessionCostTracker,
    cost_context,
)
from reasoning_mcp.models.cost import (
    Budget,
    ModelPricing,
    TokenCount,
)


@pytest.fixture
def pricing_registry():
    """Create a pricing registry with realistic pricing."""
    registry = PricingRegistry()

    # Claude 3.5 Sonnet pricing
    registry.register(
        ModelPricing(
            model_id="claude-3-5-sonnet-latest",
            input_price_per_1k=Decimal("0.003"),
            output_price_per_1k=Decimal("0.015"),
            context_window=200000,
        )
    )

    # Claude 3.5 Haiku pricing
    registry.register(
        ModelPricing(
            model_id="claude-3-5-haiku-latest",
            input_price_per_1k=Decimal("0.0008"),
            output_price_per_1k=Decimal("0.004"),
            context_window=200000,
        )
    )

    return registry


@pytest.fixture
def budget():
    """Create a test budget."""
    return Budget(
        max_cost_usd=Decimal("0.10"),
        max_tokens=50000,
        max_requests=10,
    )


@pytest.fixture
def full_setup(pricing_registry, budget):
    """Create a complete cost tracking setup."""
    calculator = CostCalculator(pricing_registry)
    enforcer = BudgetEnforcer(budget)
    tracker = SessionCostTracker("test-session")
    alerts = AlertManager("test-session")

    return {
        "calculator": calculator,
        "enforcer": enforcer,
        "tracker": tracker,
        "alerts": alerts,
        "registry": pricing_registry,
        "budget": budget,
    }


class TestCompleteCostFlow:
    """Test complete cost tracking workflows."""

    def test_estimate_then_execute(self, full_setup):
        """Test estimating cost, then executing and recording actual cost."""
        calc = full_setup["calculator"]
        enforcer = full_setup["enforcer"]
        tracker = full_setup["tracker"]

        # Step 1: Estimate cost
        estimate = calc.estimate_cost(
            method="chain_of_thought",
            input_text="What is the capital of France?",
            model_id="claude-3-5-sonnet-latest",
        )

        assert estimate.estimated_cost_usd > 0
        assert estimate.method == "chain_of_thought"

        # Step 2: Check budget before execution
        can_execute = enforcer.check_budget(
            estimated_cost=estimate.estimated_cost_usd,
            estimated_tokens=estimate.estimated_total_tokens,
        )
        assert can_execute

        # Step 3: Simulate execution and record actual cost
        actual_tokens = TokenCount(input_tokens=50, output_tokens=100)
        breakdown = calc.calculate_cost(
            tokens=actual_tokens,
            model_id="claude-3-5-sonnet-latest",
            method="chain_of_thought",
        )

        # Step 4: Record to enforcer and tracker
        enforcer.record_breakdown(breakdown)
        tracker.add_cost(breakdown, operation_id="op-1")

        # Verify state
        assert enforcer.spent_usd == breakdown.total_cost
        assert enforcer.tokens_used == 150
        assert tracker.get_total_cost() == breakdown.total_cost

    def test_budget_exceeded_flow(self, full_setup):
        """Test that budget exceeded is properly detected and handled."""
        enforcer = full_setup["enforcer"]

        # Exceed the cost budget
        enforcer.record_usage(cost_usd=Decimal("0.11"), tokens=1000)

        status = enforcer.get_status()
        assert status.is_exceeded
        assert status.remaining_usd < 0

        # Future operations should be blocked
        with pytest.raises(BudgetExceededError):
            enforcer.check_budget(
                estimated_cost=Decimal("0.01"),
                raise_on_exceed=True,
            )

    def test_alert_integration(self, full_setup):
        """Test that alerts are triggered at correct thresholds."""
        enforcer = full_setup["enforcer"]
        alerts = full_setup["alerts"]

        triggered_alerts = []

        def on_alert(alert):
            triggered_alerts.append(alert)

        alerts.on_alert(on_alert)

        # Spend 55% of budget
        enforcer.record_usage(cost_usd=Decimal("0.055"), tokens=1000)
        status = enforcer.get_status()
        new_alerts = alerts.check_budget_status(status)

        assert len(new_alerts) == 1
        assert new_alerts[0].alert_type == CostAlertType.BUDGET_50_PERCENT

        # Spend to 85% of budget
        enforcer.record_usage(cost_usd=Decimal("0.030"), tokens=1000)
        status = enforcer.get_status()
        new_alerts = alerts.check_budget_status(status)

        assert len(new_alerts) == 1
        assert new_alerts[0].alert_type == CostAlertType.BUDGET_80_PERCENT

        # Total alerts
        assert len(triggered_alerts) == 2

    def test_session_tracking(self, full_setup):
        """Test session-level cost tracking and summary."""
        calc = full_setup["calculator"]
        tracker = full_setup["tracker"]

        # Record multiple operations
        methods = ["chain_of_thought", "mcts", "chain_of_thought"]
        for i, method in enumerate(methods):
            tokens = TokenCount(input_tokens=100 + i * 50, output_tokens=200 + i * 100)
            breakdown = calc.calculate_cost(
                tokens=tokens,
                model_id="claude-3-5-sonnet-latest",
                method=method,
            )
            tracker.add_cost(breakdown, operation_id=f"op-{i}")

        # Check summary
        summary = tracker.get_summary()

        assert summary.total_operations == 3
        assert summary.methods_used == {"chain_of_thought": 2, "mcts": 1}
        assert summary.models_used == {"claude-3-5-sonnet-latest": 3}
        assert summary.total_cost_usd > 0

    def test_cost_aware_executor(self, full_setup):
        """Test CostAwareExecutor integration."""
        calc = full_setup["calculator"]
        enforcer = full_setup["enforcer"]
        tracker = full_setup["tracker"]

        executor = CostAwareExecutor(
            calculator=calc,
            enforcer=enforcer,
            tracker=tracker,
        )

        # Pre-execute
        estimate = executor.pre_execute(
            method="chain_of_thought",
            input_text="Test input",
            model_id="claude-3-5-sonnet-latest",
            check_budget=True,
        )

        assert estimate is not None
        assert estimate.method == "chain_of_thought"

        # Simulate execution and post-execute
        actual_tokens = TokenCount(input_tokens=50, output_tokens=100)
        breakdown = calc.calculate_cost(
            tokens=actual_tokens,
            model_id="claude-3-5-sonnet-latest",
            method="chain_of_thought",
        )

        executor.post_execute(breakdown, operation_id="test-op")

        # Verify tracking
        assert enforcer.spent_usd == breakdown.total_cost
        assert len(tracker.entries) == 1

    def test_cost_context_manager(self, full_setup):
        """Test cost_context context manager."""
        calc = full_setup["calculator"]
        enforcer = full_setup["enforcer"]
        tracker = full_setup["tracker"]

        with cost_context(calc, enforcer, tracker) as executor:
            assert executor.calculator is calc
            assert executor.enforcer is enforcer
            assert executor.tracker is tracker

    def test_method_comparison(self, full_setup):
        """Test comparing costs across methods."""
        calc = full_setup["calculator"]

        comparisons = calc.compare_methods(
            input_text="Analyze this complex problem",
            model_id="claude-3-5-sonnet-latest",
            methods=["chain_of_thought", "mcts", "sequential"],
        )

        assert len(comparisons) == 3

        # Should be sorted by cost
        for i in range(len(comparisons) - 1):
            assert comparisons[i].estimated_cost_usd <= comparisons[i + 1].estimated_cost_usd

        # Sequential should be cheapest (lowest multiplier)
        cheapest = calc.get_cheapest_method(
            input_text="Test",
            model_id="claude-3-5-sonnet-latest",
            methods=["chain_of_thought", "mcts", "sequential"],
        )
        assert cheapest.method == "sequential"

    def test_model_comparison(self, pricing_registry):
        """Test comparing costs across models."""
        calc = CostCalculator(pricing_registry)

        sonnet_estimate = calc.estimate_cost(
            method="chain_of_thought",
            input_text="Test input for comparison",
            model_id="claude-3-5-sonnet-latest",
        )

        haiku_estimate = calc.estimate_cost(
            method="chain_of_thought",
            input_text="Test input for comparison",
            model_id="claude-3-5-haiku-latest",
        )

        # Haiku should be cheaper
        assert haiku_estimate.estimated_cost_usd < sonnet_estimate.estimated_cost_usd


class TestBudgetEnforcementIntegration:
    """Test budget enforcement in realistic scenarios."""

    def test_token_budget_enforcement(self, pricing_registry):
        """Test enforcement based on token limits."""
        budget = Budget(max_tokens=1000)
        enforcer = BudgetEnforcer(budget)

        # Should allow
        assert enforcer.check_budget(estimated_tokens=500)

        # Record usage
        enforcer.record_usage(tokens=600)

        # Should block next operation
        assert not enforcer.check_budget(estimated_tokens=500)

    def test_request_budget_enforcement(self, pricing_registry):
        """Test enforcement based on request limits."""
        budget = Budget(max_requests=3)
        enforcer = BudgetEnforcer(budget)

        for _i in range(3):
            enforcer.record_usage(increment_requests=True)

        # Fourth request should fail
        assert not enforcer.check_budget()

    def test_combined_budget_enforcement(self, pricing_registry):
        """Test enforcement with multiple constraints."""
        budget = Budget(
            max_cost_usd=Decimal("0.05"),
            max_tokens=1000,
            max_requests=5,
        )
        enforcer = BudgetEnforcer(budget)

        # Record usage that doesn't exceed any single limit
        enforcer.record_usage(cost_usd=Decimal("0.02"), tokens=400)
        enforcer.record_usage(cost_usd=Decimal("0.02"), tokens=400)

        # Token budget is close but not exceeded
        assert enforcer.check_budget(estimated_cost=Decimal("0.005"), estimated_tokens=100)

        # But this would exceed tokens
        assert not enforcer.check_budget(estimated_tokens=300)
