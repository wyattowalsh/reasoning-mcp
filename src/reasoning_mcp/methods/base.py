"""Base protocol and metadata for reasoning methods."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session, ThoughtNode
    from reasoning_mcp.models.core import MethodCategory, MethodIdentifier
    from reasoning_mcp.streaming.context import StreamingContext

logger = structlog.get_logger(__name__)

# Sampling temperature constants
DEFAULT_SAMPLING_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1500
CREATIVE_TEMPERATURE = 0.9
PRECISE_TEMPERATURE = 0.3


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

    # Streaming support
    streaming_context: StreamingContext | None

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
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the reasoning method.

        Args:
            session: The current reasoning session
            input_text: The input to reason about
            context: Optional additional context (dict of method-specific params)
            execution_context: Optional ExecutionContext for sampling/tool access
                If provided and execution_context.can_sample is True, methods can
                use execution_context.sample() for LLM calls (FastMCP v2.14+)

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
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context (dict of method-specific params)
            execution_context: Optional ExecutionContext for sampling/tool access

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

    async def emit_thought(self, content: str, confidence: float | None = None) -> None:
        """Emit a thought event if streaming is enabled.

        This is a helper method that emits a ThoughtEvent through the streaming
        context if available. Should be called during reasoning execution to
        provide real-time updates.

        Args:
            content: The thought content to emit
            confidence: Optional confidence score (0.0 to 1.0)
        """
        ...


class ReasoningMethodBase(ReasoningMethod):
    """Base class providing default streaming behavior for reasoning methods.

    This base class provides common functionality for all reasoning methods:
    - Streaming support via streaming_context
    - LLM sampling with fallback via _sample_with_fallback
    - Execution context validation via _require_execution_context

    Subclasses should set _execution_context in their execute() method if they
    need LLM sampling capabilities.
    """

    streaming_context: StreamingContext | None = None
    _execution_context: ExecutionContext | None = None

    async def emit_thought(self, content: str, confidence: float | None = None) -> None:
        """Emit a thought event if streaming is enabled."""
        if self.streaming_context is None:
            return
        method_name = getattr(self, "name", self.__class__.__name__)
        await self.streaming_context.emit_thought(content, method_name, confidence)

    def _require_execution_context(self) -> None:
        """Validate that execution context is available.

        Raises:
            RuntimeError: If execution context is None
        """
        if self._execution_context is None:
            raise RuntimeError(
                f"Execution context required for {self.identifier} "
                "but was not provided"
            )

    async def _sample_with_fallback(
        self,
        user_prompt: str,
        fallback_generator: Callable[[], str],
        system_prompt: str = "",
        temperature: float = DEFAULT_SAMPLING_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float | None = None,
    ) -> str:
        """Sample from LLM with proper error handling and fallback.

        This method provides a standardized way to call LLM sampling with
        proper exception handling and fallback behavior.

        Args:
            user_prompt: The user prompt to send
            fallback_generator: Callable that returns fallback content if sampling fails
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens to generate (default: 1500)
            timeout: Optional timeout in seconds

        Returns:
            The sampled response or fallback content

        Note:
            This method catches expected exceptions (TimeoutError, ConnectionError,
            ValueError) and falls back gracefully. Unexpected exceptions are logged
            and re-raised to avoid masking programming errors.
        """
        if self._execution_context is None:
            logger.debug("no_execution_context", method=self.identifier)
            return fallback_generator()

        if not self._execution_context.can_sample:
            logger.debug("sampling_not_available", method=self.identifier)
            return fallback_generator()

        try:
            coro = self._execution_context.sample(
                user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            # Ensure string return
            return str(result) if not isinstance(result, str) else result

        except TimeoutError as e:
            logger.warning(
                "llm_sampling_timeout",
                method=self.identifier,
                timeout=timeout,
                error=str(e),
            )
            return fallback_generator()
        except (ConnectionError, OSError) as e:
            logger.warning(
                "llm_sampling_connection_error",
                method=self.identifier,
                error=str(e),
            )
            return fallback_generator()
        except ValueError as e:
            logger.warning(
                "llm_sampling_value_error",
                method=self.identifier,
                error=str(e),
            )
            return fallback_generator()
        except Exception as e:
            # Log unexpected exceptions and re-raise to avoid masking bugs
            logger.error(
                "llm_sampling_unexpected_error",
                method=self.identifier,
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise
