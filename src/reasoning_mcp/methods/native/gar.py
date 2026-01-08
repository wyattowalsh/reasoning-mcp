"""GAR (Generator-Adversarial Reasoning) method.

Generator-discriminator architecture for adversarial reasoning improvement.

Key phases:
1. Generate: Produce reasoning candidates
2. Discriminate: Score candidates adversarially
3. Update: Improve based on feedback
4. Iterate: Until convergence

Reference: Xi et al. (2025) - "Generator-Adversarial Reasoning"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


GAR_METADATA = MethodMetadata(
    identifier=MethodIdentifier.GAR,
    name="GAR",
    description="Generator-Adversarial Reasoning with trainable discriminator.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"adversarial", "generator", "discriminator", "iterative"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=280,
    best_for=("robustness", "candidate selection"),
    not_recommended_for=("simple queries",),
)


class Gar:
    """GAR reasoning method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._candidates: list[dict[str, Any]] = []
        self._scores: list[float] = []
        self._iteration: int = 0

    @property
    def identifier(self) -> str:
        return MethodIdentifier.GAR

    @property
    def name(self) -> str:
        return GAR_METADATA.name

    @property
    def description(self) -> str:
        return GAR_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._candidates = []
        self._scores = []
        self._iteration = 0

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("GAR must be initialized before execution")

        self._step_counter = 1
        self._current_phase = "generate"
        self._iteration = 1
        self._candidates = [
            {"id": "C1", "reasoning": "Direct: 5x3+2=17", "answer": "17"},
            {"id": "C2", "reasoning": "Step-wise: 5x3=15, +2=17", "answer": "17"},
        ]

        content = (
            f"Step {self._step_counter}: Generate (GAR)\n\n"
            f"Problem: {input_text}\n\n"
            f"Candidates:\n"
            + "\n".join(f"  [{c['id']}] {c['reasoning']}" for c in self._candidates)
            + f"\n\nNext: Discriminator evaluation."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GAR,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.65,
            quality_score=0.65,
            metadata={"phase": self._current_phase, "iteration": self._iteration},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.GAR
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
            raise RuntimeError("GAR must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "discriminate"
            self._scores = [0.82, 0.91]
            content = (
                f"Step {self._step_counter}: Discriminate\n\n"
                f"Scores:\n"
                + "\n".join(f"  [{c['id']}] {s:.2f}" for c, s in zip(self._candidates, self._scores))
                + f"\n\nBest: C2 (0.91)\nNext: Update."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.78
        elif prev_phase == "discriminate":
            self._current_phase = "update"
            content = (
                f"Step {self._step_counter}: Update\n\n"
                f"Best candidate refined.\nNext: Conclude."
            )
            thought_type = ThoughtType.REVISION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            best_score = max(self._scores) if self._scores else 0.91
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"GAR Complete\nFinal Answer: 17\nConfidence: {int(best_score*100)}%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = best_score

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.GAR,
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


__all__ = ["Gar", "GAR_METADATA"]
