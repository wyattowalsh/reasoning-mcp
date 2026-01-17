"""Tests for session cost tracker module."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from reasoning_mcp.cost.tracker import SessionCostEntry, SessionCostSummary, SessionCostTracker
from reasoning_mcp.models.cost import CostBreakdown, TokenCount


@pytest.fixture
def tracker():
    """Create a session cost tracker."""
    return SessionCostTracker("test-session")


def test_tracker_init():
    """Test tracker initialization."""
    tracker = SessionCostTracker("my-session")
    assert tracker.session_id == "my-session"
    assert tracker.entries == []


def test_add_entry(tracker):
    """Test adding a cost entry."""
    entry = tracker.add_entry(
        method="chain_of_thought",
        model_id="test-model",
        input_tokens=100,
        output_tokens=200,
        cost_usd=Decimal("0.01"),
    )

    assert entry.method == "chain_of_thought"
    assert entry.model_id == "test-model"
    assert entry.input_tokens == 100
    assert entry.output_tokens == 200
    assert entry.total_tokens == 300
    assert entry.cost_usd == Decimal("0.01")
    assert entry.timestamp is not None


def test_add_entry_with_timestamp(tracker):
    """Test adding entry with custom timestamp."""
    ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    entry = tracker.add_entry(
        method="chain_of_thought",
        model_id="test-model",
        input_tokens=100,
        output_tokens=200,
        cost_usd=Decimal("0.01"),
        timestamp=ts,
    )

    assert entry.timestamp == ts


def test_add_cost_from_breakdown(tracker):
    """Test adding cost from a breakdown."""
    breakdown = CostBreakdown(
        input_cost=Decimal("0.003"),
        output_cost=Decimal("0.0075"),
        total_cost=Decimal("0.0105"),
        tokens=TokenCount(input_tokens=1000, output_tokens=500),
        method="chain_of_thought",
        model_id="test-model",
    )

    entry = tracker.add_cost(breakdown, operation_id="op-123")

    assert entry.method == "chain_of_thought"
    assert entry.model_id == "test-model"
    assert entry.input_tokens == 1000
    assert entry.output_tokens == 500
    assert entry.cost_usd == Decimal("0.0105")
    assert entry.operation_id == "op-123"


def test_get_summary_empty(tracker):
    """Test summary of empty tracker."""
    summary = tracker.get_summary()

    assert summary.session_id == "test-session"
    assert summary.total_cost_usd == Decimal("0")
    assert summary.total_operations == 0


def test_get_summary_with_entries(tracker):
    """Test summary with entries."""
    tracker.add_entry("chain_of_thought", "model-a", 100, 200, Decimal("0.01"))
    tracker.add_entry("mcts", "model-a", 200, 400, Decimal("0.02"))
    tracker.add_entry("chain_of_thought", "model-b", 150, 300, Decimal("0.015"))

    summary = tracker.get_summary()

    assert summary.total_cost_usd == Decimal("0.045")
    assert summary.total_input_tokens == 450
    assert summary.total_output_tokens == 900
    assert summary.total_tokens == 1350
    assert summary.total_operations == 3
    assert summary.methods_used == {"chain_of_thought": 2, "mcts": 1}
    assert summary.models_used == {"model-a": 2, "model-b": 1}
    assert summary.cost_by_method["chain_of_thought"] == Decimal("0.025")
    assert summary.cost_by_method["mcts"] == Decimal("0.02")


def test_get_total_cost(tracker):
    """Test getting total cost."""
    tracker.add_entry("cot", "model", 100, 100, Decimal("0.01"))
    tracker.add_entry("cot", "model", 100, 100, Decimal("0.02"))

    assert tracker.get_total_cost() == Decimal("0.03")


def test_get_total_tokens(tracker):
    """Test getting total tokens."""
    tracker.add_entry("cot", "model", 100, 200, Decimal("0.01"))
    tracker.add_entry("cot", "model", 150, 250, Decimal("0.02"))

    assert tracker.get_total_tokens() == 700


def test_reset(tracker):
    """Test resetting the tracker."""
    tracker.add_entry("cot", "model", 100, 200, Decimal("0.01"))
    tracker.reset()

    assert tracker.entries == []
    assert tracker.get_total_cost() == Decimal("0")


def test_entries_copy(tracker):
    """Test that entries property returns a copy."""
    tracker.add_entry("cot", "model", 100, 200, Decimal("0.01"))

    entries = tracker.entries
    entries.clear()

    # Original should still have the entry
    assert len(tracker.entries) == 1


def test_session_cost_entry_model():
    """Test SessionCostEntry model."""
    entry = SessionCostEntry(
        method="chain_of_thought",
        model_id="test-model",
        input_tokens=100,
        output_tokens=200,
        total_tokens=300,
        cost_usd=Decimal("0.01"),
    )

    assert entry.timestamp is not None
    assert entry.operation_id is None


def test_session_cost_summary_model():
    """Test SessionCostSummary model."""
    summary = SessionCostSummary(
        session_id="test",
        total_cost_usd=Decimal("0.05"),
        total_operations=3,
    )

    assert summary.session_id == "test"
    assert summary.methods_used == {}
    assert summary.started_at is None
