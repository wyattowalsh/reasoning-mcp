"""Base protocol and metadata for reasoning methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from reasoning_mcp.models.core import MethodCategory, MethodIdentifier

if TYPE_CHECKING:
    from reasoning_mcp.models import Session, ThoughtNode


@dataclass(frozen=True)
class MethodMetadata:
    """Metadata describing a reasoning method.

    This dataclass stores information about a method's capabilities,
    requirements, and characteristics for use by the registry and selector.

    Attributes:
        identifier: Unique identifier for the method
        name: Human-readable name
        description: Brief description of the method
        category: Category the method belongs to
        tags: Tags for discovery and filtering
        complexity: Complexity rating (1-10, higher = more complex)
        supports_branching: Whether the method supports branching
        supports_revision: Whether the method supports thought revision
        requires_context: Whether context is required for execution
        min_thoughts: Minimum thoughts typically needed
        max_thoughts: Maximum thoughts typically generated (0 = unlimited)
        avg_tokens_per_thought: Average token usage per thought
        best_for: List of problem types this method excels at
        not_recommended_for: List of problem types to avoid
    """
    identifier: MethodIdentifier
    name: str
    description: str
    category: MethodCategory
    tags: frozenset[str] = field(default_factory=frozenset)
    complexity: int = 5
    supports_branching: bool = False
    supports_revision: bool = False
    requires_context: bool = False
    min_thoughts: int = 1
    max_thoughts: int = 0  # 0 = unlimited
    avg_tokens_per_thought: int = 500
    best_for: tuple[str, ...] = field(default_factory=tuple)
    not_recommended_for: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not 1 <= self.complexity <= 10:
            raise ValueError(f"complexity must be 1-10, got {self.complexity}")
        if self.min_thoughts < 1:
            raise ValueError(f"min_thoughts must be >= 1, got {self.min_thoughts}")
        if self.max_thoughts < 0:
            raise ValueError(f"max_thoughts must be >= 0, got {self.max_thoughts}")
        if self.max_thoughts > 0 and self.max_thoughts < self.min_thoughts:
            raise ValueError(
                f"max_thoughts ({self.max_thoughts}) must be >= min_thoughts ({self.min_thoughts})"
            )


@runtime_checkable
class ReasoningMethod(Protocol):
    """Protocol defining the interface for all reasoning methods.

    All reasoning methods must implement this protocol to be registered
    with the MethodRegistry. Methods can be native implementations or
    plugin-provided.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for this method (matches MethodIdentifier enum)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the method."""
        ...

    @property
    def description(self) -> str:
        """Brief description of what the method does."""
        ...

    @property
    def category(self) -> str:
        """Category this method belongs to (matches MethodCategory enum)."""
        ...

    async def initialize(self) -> None:
        """Initialize the method (load resources, etc.)."""
        ...

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute the reasoning method.

        Args:
            session: The current reasoning session
            input_text: The input to reason about
            context: Optional additional context

        Returns:
            A ThoughtNode representing the reasoning output
        """
        ...

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the reasoning
        """
        ...

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if healthy, False otherwise
        """
        ...
