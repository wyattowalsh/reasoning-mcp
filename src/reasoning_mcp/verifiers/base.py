"""Base protocol and metadata for reasoning verifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from reasoning_mcp.models.core import VerifierIdentifier


@dataclass(frozen=True)
class VerifierMetadata:
    """Metadata describing a reasoning verifier.

    This dataclass stores information about a verifier's capabilities,
    requirements, and characteristics for use by the registry and selector.

    Attributes:
        identifier: Unique identifier for the verifier
        name: Human-readable name
        description: Brief description of the verifier
        tags: Tags for discovery and filtering
        complexity: Complexity rating (1-10, higher = more complex)
        supports_step_level: Whether the verifier scores individual reasoning steps
        supports_outcome_level: Whether the verifier scores final outcomes
        supports_cot_verification: Whether the verifier uses CoT for verification
        best_for: List of problem types this verifier excels at
        not_recommended_for: List of problem types to avoid
    """

    identifier: VerifierIdentifier
    name: str
    description: str
    tags: frozenset[str] = field(default_factory=frozenset)
    complexity: int = 5
    supports_step_level: bool = True
    supports_outcome_level: bool = True
    supports_cot_verification: bool = False
    best_for: tuple[str, ...] = field(default_factory=tuple)
    not_recommended_for: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not 1 <= self.complexity <= 10:
            raise ValueError(f"complexity must be 1-10, got {self.complexity}")
        if not self.supports_step_level and not self.supports_outcome_level:
            raise ValueError("verifier must support at least one of step_level or outcome_level")


@runtime_checkable
class VerifierBase(Protocol):
    """Protocol defining the interface for all reasoning verifiers.

    All verifiers must implement this protocol to be registered with the
    VerifierRegistry. Verifiers can be native implementations or plugin-provided.

    Verifiers serve as Process Reward Models (PRMs) that score reasoning steps
    and final outcomes, providing feedback to guide search and improve quality.
    """

    @property
    def identifier(self) -> str:
        """Unique identifier for this verifier (matches VerifierIdentifier enum)."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the verifier."""
        ...

    @property
    def description(self) -> str:
        """Brief description of what the verifier does."""
        ...

    async def initialize(self) -> None:
        """Initialize the verifier (load models, resources, etc.).

        This method is called once during verifier registration to prepare
        the verifier for execution. For model-based verifiers, this is where
        you would load model weights or connect to external services.

        Raises:
            Exception: If initialization fails
        """
        ...

    async def verify(
        self,
        solution: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """Verify a solution and return a score with reasoning.

        This is the primary verification method that scores a final solution
        or reasoning outcome. The score should be in the range [0.0, 1.0],
        where 1.0 indicates the highest confidence in correctness.

        Args:
            solution: The solution or final answer to verify
            context: Optional additional context for verification, may include:
                - "problem": The original problem statement
                - "reasoning_steps": List of intermediate reasoning steps
                - "method": The reasoning method used
                - "constraints": Any constraints or requirements
                - Any other method-specific context

        Returns:
            A tuple of (score, reasoning) where:
                - score: Float in [0.0, 1.0] indicating verification confidence
                - reasoning: String explaining the verification result

        Raises:
            ValueError: If solution is empty or context is invalid
            Exception: If verification fails
        """
        ...

    async def score_steps(
        self,
        steps: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score individual reasoning steps (process reward).

        This method implements step-level verification, providing a process
        reward signal for each step in a reasoning chain. This is useful for
        guiding search algorithms like MCTS or beam search.

        Args:
            steps: List of reasoning steps to score
            context: Optional additional context, same as verify()

        Returns:
            List of scores (one per step), each in [0.0, 1.0]

        Raises:
            ValueError: If steps is empty or context is invalid
            NotImplementedError: If verifier doesn't support step-level scoring
            Exception: If scoring fails
        """
        ...

    async def health_check(self) -> bool:
        """Check if the verifier is healthy and ready to execute.

        This method should verify that all required resources (models,
        services, etc.) are available and functioning correctly.

        Returns:
            True if healthy and ready, False otherwise
        """
        ...
