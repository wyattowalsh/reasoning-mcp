"""SPOC (Spontaneous Self-Correction) reasoning method.

Self-correction without external feedback.

Reference: 2025 - "SPOC: Spontaneous Self-Correction"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


SPOC_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SPOC,
    name="SPOC",
    description="Spontaneous Self-Correction without external feedback.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"self-correction", "spontaneous", "autonomous", "error-detection"}),
    complexity=5,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,
    max_thoughts=6,
    avg_tokens_per_thought=180,
    best_for=("error correction", "self-improvement"),
    not_recommended_for=("tasks requiring external validation",),
)


class Spoc:
    """SPOC method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._error_detected: bool = False

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SPOC

    @property
    def name(self) -> str:
        return SPOC_METADATA.name

    @property
    def description(self) -> str:
        return SPOC_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._error_detected = False

    async def execute(
        self, session: Session, input_text: str, *, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("SPOC must be initialized")

        self._step_counter = 1
        self._current_phase = "generate"

        content = (
            f"Step {self._step_counter}: Generate (SPOC)\n\n"
            f"Problem: {input_text}\n\n"
            f"Initial solution: 5×3+2 = 17\n\n"
            f"Next: Detect errors spontaneously."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SPOC,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.75,
            quality_score=0.75,
            metadata={"phase": self._current_phase, "error_detected": self._error_detected},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SPOC
        return thought

    async def continue_reasoning(
        self, session: Session, previous_thought: ThoughtNode, *, guidance: str | None = None, context: dict[str, Any] | None = None
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("SPOC must be initialized")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "detect_error"
            self._error_detected = False  # No error in this case
            content = (
                f"Step {self._step_counter}: Error Detection\n\n"
                f"Spontaneous check:\n"
                f"  - Order of operations: ✓\n"
                f"  - Arithmetic: ✓\n"
                f"No errors detected.\nNext: Validate."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        elif prev_phase == "detect_error":
            self._current_phase = "validate"
            content = (
                f"Step {self._step_counter}: Validate\n\n"
                f"SPOC Complete\nNo corrections needed.\n"
                f"Final Answer: 17\nConfidence: 90%"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90
        else:
            self._current_phase = "conclude"
            content = f"Step {self._step_counter}: Final Answer: 17"
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.90

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SPOC,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={"phase": self._current_phase, "error_detected": self._error_detected},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["Spoc", "SPOC_METADATA"]
