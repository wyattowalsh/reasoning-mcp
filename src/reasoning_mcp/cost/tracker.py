"""Session cost tracking for reasoning operations.

This module provides functionality to track costs across
reasoning sessions and provide summaries and analytics.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from reasoning_mcp.models.cost import CostBreakdown


class SessionCostEntry(BaseModel):
    """A single cost entry in a session.

    Records the cost details for one reasoning operation.
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the operation occurred",
    )
    method: str = Field(description="Reasoning method used")
    model_id: str = Field(description="Model used")
    input_tokens: int = Field(ge=0, description="Input tokens consumed")
    output_tokens: int = Field(ge=0, description="Output tokens generated")
    total_tokens: int = Field(ge=0, description="Total tokens")
    cost_usd: Decimal = Field(description="Cost in USD")
    operation_id: str | None = Field(
        default=None,
        description="Optional identifier for the operation",
    )


class SessionCostSummary(BaseModel):
    """Summary of costs for a session."""

    session_id: str = Field(description="Session identifier")
    total_cost_usd: Decimal = Field(
        default=Decimal("0"),
        description="Total cost for the session",
    )
    total_input_tokens: int = Field(
        default=0,
        description="Total input tokens consumed",
    )
    total_output_tokens: int = Field(
        default=0,
        description="Total output tokens generated",
    )
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed",
    )
    total_operations: int = Field(
        default=0,
        description="Number of operations in the session",
    )
    methods_used: dict[str, int] = Field(
        default_factory=dict,
        description="Count of operations by method",
    )
    models_used: dict[str, int] = Field(
        default_factory=dict,
        description="Count of operations by model",
    )
    cost_by_method: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Total cost breakdown by method",
    )
    cost_by_model: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Total cost breakdown by model",
    )
    started_at: datetime | None = Field(
        default=None,
        description="When the session started",
    )
    last_operation_at: datetime | None = Field(
        default=None,
        description="When the last operation occurred",
    )


class SessionCostTracker:
    """Tracks costs across a reasoning session.

    Maintains a history of cost entries and provides summary statistics.

    Examples:
        >>> from reasoning_mcp.cost.tracker import SessionCostTracker
        >>> tracker = SessionCostTracker("session-123")
        >>> tracker.add_entry(
        ...     method="chain_of_thought",
        ...     model_id="claude-3-5-sonnet",
        ...     input_tokens=100,
        ...     output_tokens=200,
        ...     cost_usd=Decimal("0.001"),
        ... )
        >>> summary = tracker.get_summary()
    """

    def __init__(self, session_id: str) -> None:
        """Initialize the tracker for a session.

        Args:
            session_id: Unique identifier for the session
        """
        self._session_id = session_id
        self._entries: list[SessionCostEntry] = []

    @property
    def session_id(self) -> str:
        """Get the session identifier."""
        return self._session_id

    @property
    def entries(self) -> list[SessionCostEntry]:
        """Get all cost entries."""
        return self._entries.copy()

    def add_entry(
        self,
        method: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
        operation_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> SessionCostEntry:
        """Add a cost entry to the session.

        Args:
            method: Reasoning method used
            model_id: Model used
            input_tokens: Input tokens consumed
            output_tokens: Output tokens generated
            cost_usd: Cost in USD
            operation_id: Optional operation identifier
            timestamp: Optional timestamp (defaults to now)

        Returns:
            The created SessionCostEntry
        """
        entry = SessionCostEntry(
            timestamp=timestamp or datetime.now(UTC),
            method=method,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost_usd,
            operation_id=operation_id,
        )
        self._entries.append(entry)
        return entry

    def add_cost(
        self, breakdown: CostBreakdown, operation_id: str | None = None
    ) -> SessionCostEntry:
        """Add a cost entry from a breakdown.

        Args:
            breakdown: Cost breakdown from a completed operation
            operation_id: Optional operation identifier

        Returns:
            The created SessionCostEntry
        """
        tokens = breakdown.tokens
        return self.add_entry(
            method=breakdown.method or "unknown",
            model_id=breakdown.model_id or "unknown",
            input_tokens=tokens.input_tokens if tokens else 0,
            output_tokens=tokens.output_tokens if tokens else 0,
            cost_usd=breakdown.total_cost,
            operation_id=operation_id,
        )

    def get_summary(self) -> SessionCostSummary:
        """Get a summary of the session costs.

        Returns:
            SessionCostSummary with aggregated statistics
        """
        if not self._entries:
            return SessionCostSummary(session_id=self._session_id)

        total_cost = Decimal("0")
        total_input = 0
        total_output = 0
        total_tokens = 0
        methods_used: dict[str, int] = {}
        models_used: dict[str, int] = {}
        cost_by_method: dict[str, Decimal] = {}
        cost_by_model: dict[str, Decimal] = {}

        for entry in self._entries:
            total_cost += entry.cost_usd
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            total_tokens += entry.total_tokens

            methods_used[entry.method] = methods_used.get(entry.method, 0) + 1
            models_used[entry.model_id] = models_used.get(entry.model_id, 0) + 1

            cost_by_method[entry.method] = (
                cost_by_method.get(entry.method, Decimal("0")) + entry.cost_usd
            )
            cost_by_model[entry.model_id] = (
                cost_by_model.get(entry.model_id, Decimal("0")) + entry.cost_usd
            )

        return SessionCostSummary(
            session_id=self._session_id,
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_tokens,
            total_operations=len(self._entries),
            methods_used=methods_used,
            models_used=models_used,
            cost_by_method=cost_by_method,
            cost_by_model=cost_by_model,
            started_at=self._entries[0].timestamp if self._entries else None,
            last_operation_at=self._entries[-1].timestamp if self._entries else None,
        )

    def get_total_cost(self) -> Decimal:
        """Get the total cost for the session.

        Returns:
            Total cost in USD
        """
        return sum((e.cost_usd for e in self._entries), Decimal("0"))

    def get_total_tokens(self) -> int:
        """Get the total tokens consumed in the session.

        Returns:
            Total token count
        """
        return sum(e.total_tokens for e in self._entries)

    def reset(self) -> None:
        """Clear all cost entries."""
        self._entries.clear()
