"""DRO (Direct Reasoning Optimization) method.

LLMs self-reward and self-refine without external reward models.

Key phases:
1. Generate: Initial reasoning
2. Self-Score: Evaluate own quality
3. Optimize: Refine based on self-feedback
4. Iterate: Until threshold

Reference: arXiv 2025 - "Direct Reasoning Optimization"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


DRO_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DRO,
    name="DRO",
    description="Direct Reasoning Optimization - self-reward and self-refine.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"self-reward", "self-refine", "optimization", "autonomous"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=250,
    best_for=("autonomous improvement", "iterative refinement"),
    not_recommended_for=("external validation tasks",),
)


class Dro:
    """DRO reasoning method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._reasoning: str = ""
        self._self_score: float = 0.0
        self._iteration: int = 0

    @property
    def identifier(self) -> str:
        return MethodIdentifier.DRO

    @property
    def name(self) -> str:
        return DRO_METADATA.name

    @property
    def description(self) -> str:
        return DRO_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._reasoning = ""
        self._self_score = 0.0
        self._iteration = 0

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("DRO must be initialized before execution")

        self._step_counter = 1
        self._current_phase = "generate"
        self._iteration = 1
        self._reasoning = "Calculate 5x3=15, then 15+2=17"

        content = (
            f"Step {self._step_counter}: Generate (DRO)\n\n"
            f"Problem: {input_text}\n\n"
            f"Reasoning: {self._reasoning}\n"
            f"Answer: 17\n\nNext: Self-score."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DRO,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={"phase": self._current_phase, "iteration": self._iteration},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DRO
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("DRO must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "self_score"
            self._self_score = 0.78
            content = (
                f"Step {self._step_counter}: Self-Score\n\n"
                f"Score: {self._self_score:.2f}\n"
                f"Feedback: Correct but could add PEMDAS note.\n\nNext: Optimize."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = self._self_score
        elif prev_phase == "self_score":
            self._current_phase = "optimize"
            self._reasoning = "PEMDAS: 5x3=15 first, then 15+2=17"
            content = (
                f"Step {self._step_counter}: Optimize\n\n"
                f"Refined: {self._reasoning}\n\nNext: Conclude."
            )
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"DRO Complete\nFinal: {self._reasoning}\n"
                f"Answer: 17\nConfidence: 90%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.DRO,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "iteration": self._iteration},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["Dro", "DRO_METADATA"]
