"""Cost alerting system for budget management.

This module provides types and managers for cost-related alerts,
including threshold warnings and budget exceeded notifications.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.models.cost import BudgetStatus


class CostAlertType(Enum):
    """Types of cost alerts."""

    THRESHOLD_WARNING = "threshold_warning"
    BUDGET_50_PERCENT = "budget_50_percent"
    BUDGET_80_PERCENT = "budget_80_percent"
    BUDGET_EXCEEDED = "budget_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"


class CostAlert(BaseModel):
    """A cost alert notification."""

    alert_type: CostAlertType = Field(description="Type of alert")
    message: str = Field(description="Human-readable alert message")
    current_value: Decimal = Field(description="Current value that triggered the alert")
    threshold_value: Decimal = Field(description="Threshold that was crossed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the alert was generated",
    )
    session_id: str | None = Field(default=None, description="Optional session identifier")


# Type alias for alert callback
AlertCallback = Callable[[CostAlert], None]


class AlertManager:
    """Manages cost alerts and notifications.

    Tracks thresholds and triggers alerts when they are crossed.
    Supports callback registration for custom alert handling.

    Examples:
        >>> from reasoning_mcp.cost.alerts import AlertManager, CostAlertType
        >>> manager = AlertManager()
        >>> manager.add_threshold(CostAlertType.BUDGET_50_PERCENT, Decimal("0.50"))
        >>> manager.on_alert(lambda alert: print(f"Alert: {alert.message}"))
        >>> manager.check_thresholds(Decimal("0.60"))  # Triggers 50% alert
    """

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize the alert manager.

        Args:
            session_id: Optional session identifier for alert context
        """
        self._session_id = session_id
        self._thresholds: dict[CostAlertType, Decimal] = {}
        self._triggered: set[CostAlertType] = set()
        self._callbacks: list[AlertCallback] = []
        self._alerts: list[CostAlert] = []

        # Set default thresholds
        self._thresholds[CostAlertType.BUDGET_50_PERCENT] = Decimal("0.50")
        self._thresholds[CostAlertType.BUDGET_80_PERCENT] = Decimal("0.80")
        self._thresholds[CostAlertType.BUDGET_EXCEEDED] = Decimal("1.00")

    @property
    def alerts(self) -> list[CostAlert]:
        """Get all generated alerts."""
        return self._alerts.copy()

    def add_threshold(self, alert_type: CostAlertType, threshold: Decimal) -> None:
        """Add or update a threshold for an alert type.

        Args:
            alert_type: Type of alert
            threshold: Threshold value (as a fraction, e.g., 0.80 for 80%)
        """
        self._thresholds[alert_type] = threshold

    def on_alert(self, callback: AlertCallback) -> None:
        """Register a callback for alerts.

        Args:
            callback: Function to call when an alert is triggered
        """
        self._callbacks.append(callback)

    def check_thresholds(
        self,
        utilization: Decimal,
        budget_limit: Decimal | None = None,
    ) -> list[CostAlert]:
        """Check if any thresholds have been crossed.

        Args:
            utilization: Current utilization as a fraction (0-1+)
            budget_limit: Optional budget limit for context in alerts

        Returns:
            List of newly triggered alerts
        """
        new_alerts: list[CostAlert] = []

        for alert_type, threshold in sorted(
            self._thresholds.items(),
            key=lambda x: x[1],
        ):
            if alert_type in self._triggered:
                continue

            if utilization >= threshold:
                alert = self._create_alert(
                    alert_type,
                    utilization,
                    threshold,
                    budget_limit,
                )
                new_alerts.append(alert)
                self._triggered.add(alert_type)
                self._alerts.append(alert)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        # Don't let callback errors stop processing, but log for debugging
                        logger.warning(
                            "cost_alert_callback_failed",
                            callback=getattr(callback, "__name__", str(callback)),
                            error=str(e),
                        )

        return new_alerts

    def check_budget_status(self, status: BudgetStatus) -> list[CostAlert]:
        """Check alerts based on budget status.

        Args:
            status: Current budget status

        Returns:
            List of newly triggered alerts
        """
        if status.utilization_percent is None:
            return []

        utilization = Decimal(str(status.utilization_percent)) / 100
        return self.check_thresholds(
            utilization,
            status.budget.max_cost_usd,
        )

    def _create_alert(
        self,
        alert_type: CostAlertType,
        current: Decimal,
        threshold: Decimal,
        budget_limit: Decimal | None,
    ) -> CostAlert:
        """Create an alert instance.

        Args:
            alert_type: Type of alert
            current: Current value
            threshold: Threshold value
            budget_limit: Optional budget limit for context

        Returns:
            CostAlert instance
        """
        messages = {
            CostAlertType.THRESHOLD_WARNING: "Cost threshold warning",
            CostAlertType.BUDGET_50_PERCENT: "Budget is 50% consumed",
            CostAlertType.BUDGET_80_PERCENT: "Budget is 80% consumed - approaching limit",
            CostAlertType.BUDGET_EXCEEDED: "Budget has been exceeded",
            CostAlertType.RATE_LIMIT_WARNING: "Approaching rate limit",
        }

        percent = int(current * 100)
        message = f"{messages.get(alert_type, 'Cost alert')}: {percent}% of budget used"

        if budget_limit is not None:
            message += f" (limit: ${budget_limit})"

        return CostAlert(
            alert_type=alert_type,
            message=message,
            current_value=current,
            threshold_value=threshold,
            session_id=self._session_id,
        )

    def reset(self) -> None:
        """Reset triggered alerts (allows re-triggering)."""
        self._triggered.clear()
        self._alerts.clear()
