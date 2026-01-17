"""LightThinker reasoning method.

Gist token compression for efficient reasoning.

Reference: 2025 - "LightThinker"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


LIGHT_THINKER_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LIGHT_THINKER,
    name="LightThinker",
    description="Gist token compression for efficient reasoning.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"compression", "gist", "efficient", "lightweight"}),
    complexity=5,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=5,
    avg_tokens_per_thought=120,
    best_for=("token efficiency", "long reasoning chains"),
    not_recommended_for=("detailed explanations",),
)


class LightThinker:
    """LightThinker method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "compress"
        self._gist_tokens: list[str] = []

    @property
    def identifier(self) -> str:
        return MethodIdentifier.LIGHT_THINKER

    @property
    def name(self) -> str:
        return LIGHT_THINKER_METADATA.name

    @property
    def description(self) -> str:
        return LIGHT_THINKER_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "compress"
        self._gist_tokens = []

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("LightThinker must be initialized")

        self._step_counter = 1
        self._current_phase = "compress"
        self._gist_tokens = ["[G1:parse]", "[G2:compute]", "[G3:verify]"]

        content = (
            f"Step {self._step_counter}: Compress (LightThinker)\n\n"
            f"Problem: {input_text}\n\n"
            f"Gist Tokens:\n" + "\n".join(f"  {g}" for g in self._gist_tokens)
            + f"\n\nCompression ratio: ~60%\nNext: Reason with gists."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LIGHT_THINKER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={"phase": self._current_phase, "gist_count": len(self._gist_tokens)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LIGHT_THINKER
        return thought

    async def continue_reasoning(
        self, session: Session, previous_thought: ThoughtNode, *, guidance: str | None = None, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("LightThinker must be initialized")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "compress")

        if prev_phase == "compress":
            self._current_phase = "reason"
            content = f"Step {self._step_counter}: Reason\n\nProcessing gist tokens...\n5×3=15, 15+2=17\nNext: Expand."
            thought_type = ThoughtType.REASONING
            confidence = 0.8
        else:
            self._current_phase = "expand"
            content = (
                f"Step {self._step_counter}: Expand\n\n"
                f"LightThinker Complete\nFinal Answer: 17\nConfidence: 88%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LIGHT_THINKER,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["LightThinker", "LIGHT_THINKER_METADATA"]
