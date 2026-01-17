"""Base protocol and metadata for reasoning routers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reasoning_mcp.models.core import RouterIdentifier


@dataclass(frozen=True)
class RouterMetadata:
    """Metadata describing a reasoning router.

    This dataclass stores information about a router's capabilities,
    features, and characteristics for use by the registry and selector.

    Attributes:
        identifier: Unique identifier for the router
        name: Human-readable name
        description: Brief description of the router
        tags: Tags for discovery and filtering
        complexity: Complexity rating (1-10, higher = more complex)
        supports_budget_control: Whether the router supports token budget allocation
        supports_multi_model: Whether the router supports multi-model routing
        best_for: List of task types this router excels at
        not_recommended_for: List of task types to avoid
    """

    identifier: RouterIdentifier
    name: str
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
    complexity: int = 5
    supports_budget_control: bool = False
    supports_multi_model: bool = False
    best_for: tuple[str, ...] = field(default_factory=tuple)
    not_recommended_for: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not 1 <= self.complexity <= 10:
            raise ValueError(f"complexity must be 1-10, got {self.complexity}")


@runtime_checkable
class RouterBase(Protocol):
    """Protocol defining the interface for all reasoning routers.

    All routers must implement this protocol to be registered
    with the RouterRegistry. Routers dynamically select and allocate
    reasoning methods based on query analysis.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for this router (matches RouterIdentifier enum)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the router."""
        ...

    @property
    def description(self) -> str:
        """Brief description of what the router does."""
        ...

    async def initialize(self) -> None:
        """Initialize the router (load resources, models, etc.)."""
        ...

    async def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Select the best reasoning method for the given query.

        Args:
            query: The input query or problem to analyze
            context: Optional additional context for routing decision

        Returns:
            The method identifier (string) to use for this query
        """
        ...

    async def allocate_budget(
        self,
        query: str,
        budget: int,
    ) -> dict[str, int]:
        """Allocate token budget across reasoning methods.

        This is an optional capability for routers that support budget control.
        If not supported, should raise NotImplementedError.

        Args:
            query: The input query or problem to analyze
            budget: Total token budget available

        Returns:
            Dictionary mapping method identifiers to allocated token counts

        Raises:
            NotImplementedError: If budget control is not supported
        """
        ...

    async def health_check(self) -> bool:
        """Check if the router is healthy and ready to operate.

        Returns:
            True if healthy, False otherwise
        """
        ...
