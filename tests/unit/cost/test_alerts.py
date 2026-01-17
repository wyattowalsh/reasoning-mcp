"""Tests for cost alerts module."""

from reasoning_mcp.cost.alerts import CostAlertType


def test_alert_type_enum():
    """Verify all CostAlertType enum values."""
    # Test that all expected enum members exist
    assert hasattr(CostAlertType, "THRESHOLD_WARNING")
    assert hasattr(CostAlertType, "BUDGET_50_PERCENT")
    assert hasattr(CostAlertType, "BUDGET_80_PERCENT")
    assert hasattr(CostAlertType, "BUDGET_EXCEEDED")
    assert hasattr(CostAlertType, "RATE_LIMIT_WARNING")

    # Test enum values
    assert CostAlertType.THRESHOLD_WARNING.value == "threshold_warning"
    assert CostAlertType.BUDGET_50_PERCENT.value == "budget_50_percent"
    assert CostAlertType.BUDGET_80_PERCENT.value == "budget_80_percent"
    assert CostAlertType.BUDGET_EXCEEDED.value == "budget_exceeded"
    assert CostAlertType.RATE_LIMIT_WARNING.value == "rate_limit_warning"

    # Test that we have exactly 5 members
    assert len(CostAlertType) == 5

    # Test enum member identity
    assert CostAlertType.THRESHOLD_WARNING == CostAlertType.THRESHOLD_WARNING
    assert CostAlertType.THRESHOLD_WARNING != CostAlertType.BUDGET_EXCEEDED

    # Test that enum values are accessible
    all_values = [alert_type.value for alert_type in CostAlertType]
    assert "threshold_warning" in all_values
    assert "budget_50_percent" in all_values
    assert "budget_80_percent" in all_values
    assert "budget_exceeded" in all_values
    assert "rate_limit_warning" in all_values
