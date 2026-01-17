"""Tests for cost integration module."""

from unittest.mock import Mock

import pytest

from reasoning_mcp.cost.integration import CostAwareExecutor, cost_context


def test_cost_aware_executor_init():
    """Test CostAwareExecutor initialization."""
    # Create mock dependencies
    calculator = Mock()
    enforcer = Mock()
    tracker = Mock()

    # Test with all components
    executor = CostAwareExecutor(calculator, enforcer, tracker)
    assert executor.calculator is calculator
    assert executor.enforcer is enforcer
    assert executor.tracker is tracker

    # Test with only calculator (required)
    executor_minimal = CostAwareExecutor(calculator)
    assert executor_minimal.calculator is calculator
    assert executor_minimal.enforcer is None
    assert executor_minimal.tracker is None

    # Test with calculator and enforcer
    executor_with_enforcer = CostAwareExecutor(calculator, enforcer=enforcer)
    assert executor_with_enforcer.calculator is calculator
    assert executor_with_enforcer.enforcer is enforcer
    assert executor_with_enforcer.tracker is None

    # Test with calculator and tracker
    executor_with_tracker = CostAwareExecutor(calculator, tracker=tracker)
    assert executor_with_tracker.calculator is calculator
    assert executor_with_tracker.enforcer is None
    assert executor_with_tracker.tracker is tracker


def test_cost_context():
    """Test cost_context context manager."""
    # Create mock dependencies
    calculator = Mock()
    enforcer = Mock()
    tracker = Mock()

    # Test basic usage with all components
    with cost_context(calculator, enforcer, tracker) as executor:
        assert isinstance(executor, CostAwareExecutor)
        assert executor.calculator is calculator
        assert executor.enforcer is enforcer
        assert executor.tracker is tracker

    # Test with only calculator
    with cost_context(calculator) as executor:
        assert isinstance(executor, CostAwareExecutor)
        assert executor.calculator is calculator
        assert executor.enforcer is None
        assert executor.tracker is None

    # Test with calculator and enforcer
    with cost_context(calculator, enforcer=enforcer) as executor:
        assert isinstance(executor, CostAwareExecutor)
        assert executor.calculator is calculator
        assert executor.enforcer is enforcer
        assert executor.tracker is None

    # Test with calculator and tracker
    with cost_context(calculator, tracker=tracker) as executor:
        assert isinstance(executor, CostAwareExecutor)
        assert executor.calculator is calculator
        assert executor.enforcer is None
        assert executor.tracker is tracker


def test_cost_context_cleanup():
    """Test that cost_context properly manages resources."""
    calculator = Mock()

    # Verify the executor is accessible inside the context
    with cost_context(calculator) as executor:
        assert executor is not None
        # Store a reference to verify it was the same object
        executor_ref = executor

    # Verify we got the expected executor
    assert executor_ref.calculator is calculator


def test_cost_context_with_exception():
    """Test that cost_context handles exceptions properly."""
    calculator = Mock()

    # Test that exceptions propagate correctly
    with pytest.raises(ValueError, match="test error"), cost_context(calculator) as executor:
        assert executor is not None
        raise ValueError("test error")


def test_cost_aware_executor_attributes():
    """Test that CostAwareExecutor properly stores attributes."""
    calculator = Mock()
    calculator.name = "test_calculator"
    enforcer = Mock()
    enforcer.budget = "test_budget"
    tracker = Mock()
    tracker.session_id = "test_session"

    executor = CostAwareExecutor(calculator, enforcer, tracker)

    # Verify attributes are accessible
    assert executor.calculator.name == "test_calculator"
    assert executor.enforcer.budget == "test_budget"
    assert executor.tracker.session_id == "test_session"


def test_cost_context_multiple_contexts():
    """Test that multiple cost contexts can be created independently."""
    calc1 = Mock()
    calc1.id = 1
    calc2 = Mock()
    calc2.id = 2

    # Create two contexts with different calculators
    with cost_context(calc1) as executor1, cost_context(calc2) as executor2:
        # Verify they're independent
        assert executor1.calculator.id == 1
        assert executor2.calculator.id == 2
        assert executor1.calculator is not executor2.calculator
