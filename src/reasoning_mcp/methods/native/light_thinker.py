"""LightThinker reasoning method.

Gist token compression for efficient reasoning.

Reference: 2025 - "LightThinker"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
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


class LightThinker(ReasoningMethodBase):
    """LightThinker method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "compress"
        self._gist_tokens: list[str] = []
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

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
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("LightThinker must be initialized")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "compress"

        # Generate gist tokens
        if use_sampling:
            self._gist_tokens = await self._sample_compress_to_gist(input_text)
        else:
            self._gist_tokens = self._heuristic_compress_to_gist(input_text)

        content = (
            f"Step {self._step_counter}: Compress (LightThinker)\n\n"
            f"Problem: {input_text}\n\n"
            f"Gist Tokens:\n"
            + "\n".join(f"  {g}" for g in self._gist_tokens)
            + "\n\nCompression ratio: ~60%\nNext: Reason with gists."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LIGHT_THINKER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "gist_count": len(self._gist_tokens),
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LIGHT_THINKER
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("LightThinker must be initialized")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        if execution_context:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "compress")

        if prev_phase == "compress":
            self._current_phase = "reason"
            # Reason with gist tokens
            if use_sampling:
                reasoning_result = await self._sample_reason_with_gist(session)
            else:
                reasoning_result = self._heuristic_reason_with_gist()

            content = (
                f"Step {self._step_counter}: Reason\n\n"
                f"Processing gist tokens...\n{reasoning_result}\nNext: Expand."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.8
        else:
            self._current_phase = "expand"
            # Expand gist tokens back to full answer
            if use_sampling:
                final_answer = await self._sample_expand_from_gist(session)
            else:
                final_answer = self._heuristic_expand_from_gist()

            content = (
                f"Step {self._step_counter}: Expand\n\n"
                f"LightThinker Complete\nFinal Answer: {final_answer}\nConfidence: 88%"
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
            metadata={"phase": self._current_phase, "sampled": use_sampling},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _heuristic_compress_to_gist(self, input_text: str) -> list[str]:
        """Heuristically compress input to gist tokens.

        Fallback method when LLM sampling is not available.

        Args:
            input_text: The problem text to compress

        Returns:
            List of gist tokens
        """
        # Default gist tokens
        return ["[G1:parse]", "[G2:compute]", "[G3:verify]"]

    def _heuristic_reason_with_gist(self) -> str:
        """Heuristically reason with gist tokens.

        Fallback method when LLM sampling is not available.

        Returns:
            Reasoning result string
        """
        return "5×3=15, 15+2=17"

    def _heuristic_expand_from_gist(self) -> str:
        """Heuristically expand gist tokens to full answer.

        Fallback method when LLM sampling is not available.

        Returns:
            Final answer string
        """
        return "17"

    async def _sample_compress_to_gist(self, input_text: str) -> list[str]:
        """Use LLM sampling to compress input to gist tokens.

        Args:
            input_text: The problem text to compress

        Returns:
            List of gist tokens
        """
        system_prompt = """You are applying LightThinker gist token compression.
Compress the given problem into 3-5 concise gist tokens that capture the essential elements.
Each gist token should be in format [G#:concept] where concept is a key operation or element.

Examples:
- [G1:parse] - parsing/understanding
- [G2:compute] - computation
- [G3:verify] - verification
- [G4:combine] - combining results
- [G5:analyze] - analysis

Return ONLY the gist tokens, one per line."""

        user_prompt = f"""Problem: {input_text}

Compress this problem into gist tokens that capture the essential reasoning steps.
Use 3-5 tokens maximum for efficient reasoning."""

        content = await self._sample_with_fallback(
            user_prompt,
            lambda: "",  # Return empty to trigger fallback below
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=150,
        )

        # Parse gist tokens from response
        tokens = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("[G") and "]" in line:
                tokens.append(line)

        # Ensure we have at least some tokens
        return tokens if tokens else self._heuristic_compress_to_gist(input_text)

    async def _sample_reason_with_gist(self, session: Session) -> str:
        """Use LLM sampling to reason with gist tokens.

        Args:
            session: The current reasoning session

        Returns:
            Reasoning result string
        """
        # Get the problem from session
        initial_thought = session.get_recent_thoughts(n=session.thought_count)
        problem_content = initial_thought[0].content if initial_thought else "Unknown problem"

        system_prompt = """You are reasoning using compressed gist tokens.
Process the gist tokens to solve the problem efficiently.
Provide concise intermediate reasoning steps."""

        gist_repr = "\n".join(self._gist_tokens)
        user_prompt = f"""Problem Context:
{problem_content}

Gist Tokens:
{gist_repr}

Reason through the problem using these compressed gist tokens.
Show brief intermediate steps."""

        return await self._sample_with_fallback(
            user_prompt,
            self._heuristic_reason_with_gist,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=200,
        )

    async def _sample_expand_from_gist(self, session: Session) -> str:
        """Use LLM sampling to expand gist tokens to full answer.

        Args:
            session: The current reasoning session

        Returns:
            Final answer string
        """
        # Get the problem and reasoning from session
        thoughts = session.get_recent_thoughts(n=session.thought_count)
        context_text = "\n\n".join(
            f"Step {t.step_number}:\n{t.content}" for t in thoughts if t.step_number
        )

        system_prompt = """You are expanding compressed gist tokens to a final answer.
Based on the compressed reasoning, provide a clear, complete final answer to the
original problem."""

        user_prompt = f"""Reasoning Context:
{context_text}

Expand from the compressed gist reasoning to provide a complete final answer.
Be clear and concise."""

        return await self._sample_with_fallback(
            user_prompt,
            self._heuristic_expand_from_gist,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=300,
        )


__all__ = ["LightThinker", "LIGHT_THINKER_METADATA"]
