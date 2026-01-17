"""Base protocol and metadata for ensemblers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reasoning_mcp.models.core import EnsemblerIdentifier


@dataclass(frozen=True)
class EnsemblerMetadata:
    """Metadata describing an ensembler.

    This dataclass stores information about an ensembler's capabilities,
    requirements, and characteristics for use by the registry and selector.

    Attributes:
        identifier: Unique identifier for the ensembler
        name: Human-readable name
        description: Brief description of the ensembler
        tags: Tags for discovery and filtering
        complexity: Complexity rating (1-10, higher = more complex)
        min_models: Minimum number of models needed for ensemble
        max_models: Maximum number of models supported (0 = unlimited)
        supports_weighted_voting: Whether the ensembler supports weighted voting
        supports_dynamic_selection: Whether the ensembler can dynamically select models
        best_for: List of use cases this ensembler excels at
        not_recommended_for: List of use cases to avoid
    """

    identifier: EnsemblerIdentifier
    name: str
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
    complexity: int = 5
    min_models: int = 2
    max_models: int = 0  # 0 = unlimited
    supports_weighted_voting: bool = False
    supports_dynamic_selection: bool = False
    best_for: tuple[str, ...] = field(default_factory=tuple)
    not_recommended_for: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not 1 <= self.complexity <= 10:
            raise ValueError(f"complexity must be 1-10, got {self.complexity}")
        if self.min_models < 1:
            raise ValueError(f"min_models must be >= 1, got {self.min_models}")
        if self.max_models < 0:
            raise ValueError(f"max_models must be >= 0, got {self.max_models}")
        if self.max_models > 0 and self.max_models < self.min_models:
            raise ValueError(
                f"max_models ({self.max_models}) must be >= min_models ({self.min_models})"
            )


@runtime_checkable
class EnsemblerBase(Protocol):
    """Protocol defining the interface for all ensemblers.

    All ensemblers must implement this protocol to be registered
    with the EnsemblerRegistry. Ensemblers can be native implementations
    or plugin-provided.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for this ensembler (matches EnsemblerIdentifier enum)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the ensembler."""
        ...

    @property
    def description(self) -> str:
        """Brief description of what the ensembler does."""
        ...

    async def initialize(self) -> None:
        """Initialize the ensembler (load resources, configure models, etc.)."""
        ...

    async def ensemble(
        self,
        query: str,
        solutions: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Combine multiple solutions into a single best answer.

        Args:
            query: The original query that generated the solutions
            solutions: List of solution strings from different models/methods
            context: Optional additional context (e.g., solution metadata, scores)

        Returns:
            A string representing the best or combined solution
        """
        ...

    async def select_models(
        self,
        query: str,
        available_models: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Dynamically select which models to use for the ensemble.

        Args:
            query: The query to be processed
            available_models: List of available model identifiers
            context: Optional additional context for selection

        Returns:
            List of selected model identifiers to use in the ensemble
        """
        ...

    async def health_check(self) -> bool:
        """Check if the ensembler is healthy and ready to execute.

        Returns:
            True if healthy, False otherwise
        """
        ...
