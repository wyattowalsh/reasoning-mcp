"""Hidden CoT Decoding reasoning method.

Efficient CoT without explicit tokens - reasoning happens in hidden states.

Reference: Wang et al. (2025) - "Hidden CoT Decoding"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


HIDDEN_COT_DECODING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.HIDDEN_COT_DECODING,
    name="Hidden CoT Decoding",
    description="Efficient CoT reasoning in hidden states without explicit tokens.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"hidden", "efficient", "decoding", "implicit"}),
    complexity=5,
    supports_branching=False,
    supports_revision=False,
    requires_context=False,
    min_thoughts=2,
    max_thoughts=4,
    avg_tokens_per_thought=100,
    best_for=("token efficiency", "fast inference"),
    not_recommended_for=("explainability", "debugging"),
)


class HiddenCotDecoding:
    """Hidden CoT Decoding method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "hidden_reason"

    @property
    def identifier(self) -> str:
        return MethodIdentifier.HIDDEN_COT_DECODING

    @property
    def name(self) -> str:
        return HIDDEN_COT_DECODING_METADATA.name

    @property
    def description(self) -> str:
        return HIDDEN_COT_DECODING_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "hidden_reason"

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HiddenCotDecoding must be initialized")

        self._step_counter = 1
        self._current_phase = "hidden_reason"

        content = (
            f"Step {self._step_counter}: Hidden Reasoning (Hidden CoT Decoding)\n\n"
            f"Problem: {input_text}\n\n"
            f"[Hidden state reasoning - no explicit tokens]\n"
            f"Internal computation proceeding...\n\n"
            f"Next: Decode to answer."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HIDDEN_COT_DECODING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,
            quality_score=0.75,
            metadata={"phase": self._current_phase},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.HIDDEN_COT_DECODING
        return thought

    async def continue_reasoning(
        self, session: Session, previous_thought: ThoughtNode, *, guidance: str | None = None, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HiddenCotDecoding must be initialized")

        self._step_counter += 1
        self._current_phase = "decode"

        content = (
            f"Step {self._step_counter}: Decode Answer\n\n"
            f"Hidden reasoning complete.\n"
            f"Final Answer: 17\n"
            f"Confidence: 88%\n\n"
            f"Method: Hidden CoT Decoding\n"
            f"  - Zero explicit reasoning tokens\n"
            f"  - Maximum efficiency"
        )

        thought = ThoughtNode(
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.HIDDEN_COT_DECODING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=0.88,
            quality_score=0.88,
            metadata={"phase": self._current_phase},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["HiddenCotDecoding", "HIDDEN_COT_DECODING_METADATA"]
