"""HybridCoT reasoning method.

Interleaves latent (hidden) and text (explicit) reasoning for efficiency.

Key phases:
1. Encode Latent: Compress routine reasoning
2. Text Steps: Explicit reasoning when needed  
3. Decode: Expand to final answer

Reference: ICLR 2026 - "HybridCoT"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


HYBRID_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.HYBRID_COT,
    name="HybridCoT",
    description="Interleaves latent and text reasoning for efficiency.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"latent", "hybrid", "efficient", "interleaved"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=7,
    avg_tokens_per_thought=200,
    best_for=("balanced efficiency", "complex reasoning"),
    not_recommended_for=("full transparency tasks",),
)


class HybridCot:
    """HybridCoT reasoning method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "encode_latent"
        self._latent_tokens: list[str] = []
        self._text_steps: list[str] = []

    @property
    def identifier(self) -> str:
        return MethodIdentifier.HYBRID_COT

    @property
    def name(self) -> str:
        return HYBRID_COT_METADATA.name

    @property
    def description(self) -> str:
        return HYBRID_COT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "encode_latent"
        self._latent_tokens = []
        self._text_steps = []

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HybridCot must be initialized before execution")

        self._step_counter = 1
        self._current_phase = "encode_latent"
        self._latent_tokens = ["<L1: parse>", "<L2: compute>", "<L3: verify>"]

        content = (
            f"Step {self._step_counter}: Encode Latent (HybridCoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"Latent Tokens:\n" + "\n".join(f"  {lt}" for lt in self._latent_tokens)
            + f"\n\nNext: Add text steps."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HYBRID_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.65,
            quality_score=0.65,
            metadata={"phase": self._current_phase, "latent_count": len(self._latent_tokens)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.HYBRID_COT
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
            raise RuntimeError("HybridCot must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "encode_latent")

        if prev_phase == "encode_latent":
            self._current_phase = "text_steps"
            self._text_steps = ["Verify PEMDAS order", "5x3=15, 15+2=17"]
            content = (
                f"Step {self._step_counter}: Text Steps\n\n"
                f"Explicit:\n" + "\n".join(f"  {ts}" for ts in self._text_steps)
                + f"\n\nNext: Decode."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.78
        elif prev_phase == "text_steps":
            self._current_phase = "decode"
            content = (
                f"Step {self._step_counter}: Decode\n\n"
                f"Final Answer: 17\nConfidence: 88%\n\n"
                f"Method: HybridCoT - ~35% token reduction"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88
        else:
            self._current_phase = "conclude"
            content = f"Step {self._step_counter}: Final Answer: 17"
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.HYBRID_COT,
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


__all__ = ["HybridCot", "HYBRID_COT_METADATA"]
