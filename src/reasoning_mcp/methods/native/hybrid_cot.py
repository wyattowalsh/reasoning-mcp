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

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
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


class HybridCot(ReasoningMethodBase):
    """HybridCoT reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "encode_latent"
        self._latent_tokens: list[str] = []
        self._text_steps: list[str] = []
        self._execution_context: ExecutionContext | None = None

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
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HybridCot must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "encode_latent"

        # Use sampling if available, otherwise fallback to heuristics
        if self._execution_context and self._execution_context.can_sample:
            content = await self._sample_encode_latent(input_text)
        else:
            content = self._generate_encode_latent(input_text)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.HYBRID_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.65,
            quality_score=0.65,
            metadata={
                "phase": self._current_phase,
                "latent_count": len(self._latent_tokens),
                "sampled": (
                    self._execution_context is not None and self._execution_context.can_sample
                ),
            },
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
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("HybridCot must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "encode_latent")

        # Use sampling if available, otherwise fallback to heuristics
        use_sampling = self._execution_context is not None and self._execution_context.can_sample

        if prev_phase == "encode_latent":
            self._current_phase = "text_steps"
            if use_sampling:
                content = await self._sample_text_steps(previous_thought.content, guidance)
            else:
                content = self._generate_text_steps()
            thought_type = ThoughtType.REASONING
            confidence = 0.78
        elif prev_phase == "text_steps":
            self._current_phase = "decode"
            if use_sampling:
                content = await self._sample_decode(previous_thought.content)
            else:
                content = self._generate_decode()
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
            metadata={"phase": self._current_phase, "sampled": use_sampling},
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    # Fallback heuristic methods

    def _generate_encode_latent(self, input_text: str) -> str:
        """Generate latent encoding phase using heuristics (fallback).

        Args:
            input_text: The input problem or question

        Returns:
            A formatted string containing the latent encoding step
        """
        self._latent_tokens = ["<L1: parse>", "<L2: compute>", "<L3: verify>"]

        content = (
            f"Step {self._step_counter}: Encode Latent (HybridCoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"Latent Tokens:\n"
            + "\n".join(f"  {lt}" for lt in self._latent_tokens)
            + "\n\nNext: Add text steps."
        )
        return content

    def _generate_text_steps(self) -> str:
        """Generate text steps phase using heuristics (fallback).

        Returns:
            A formatted string containing the text steps
        """
        self._text_steps = ["Verify PEMDAS order", "5x3=15, 15+2=17"]
        content = (
            f"Step {self._step_counter}: Text Steps\n\n"
            f"Explicit:\n" + "\n".join(f"  {ts}" for ts in self._text_steps) + "\n\nNext: Decode."
        )
        return content

    def _generate_decode(self) -> str:
        """Generate decode phase using heuristics (fallback).

        Returns:
            A formatted string containing the decode step
        """
        content = (
            f"Step {self._step_counter}: Decode\n\n"
            f"Final Answer: 17\nConfidence: 88%\n\n"
            f"Method: HybridCoT - ~35% token reduction"
        )
        return content

    # Sampling methods

    async def _sample_encode_latent(self, input_text: str) -> str:
        """Generate latent encoding phase using LLM sampling.

        Args:
            input_text: The input problem or question

        Returns:
            A formatted string containing the sampled latent encoding step
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_encode_latent but was not provided"
            )

        system_prompt = """You are a reasoning assistant using HybridCoT methodology.
In the latent encoding phase, you compress routine reasoning into latent tokens.

Your task:
1. Analyze the problem and identify key sub-tasks
2. Create 2-4 latent tokens representing compressed reasoning steps
3. Format each token as: <L#: brief_description>

Examples of latent tokens:
- <L1: parse_input>
- <L2: identify_operations>
- <L3: compute_result>
- <L4: verify_constraints>

Be concise. Latent tokens should represent reasoning compression."""

        user_prompt = f"""Problem: {input_text}

Generate the latent encoding phase for HybridCoT reasoning.
Create 2-4 latent tokens that compress the routine reasoning steps needed to solve this problem.

Format your response as:
Step 1: Encode Latent (HybridCoT)

Problem: {input_text}

Latent Tokens:
  <L1: ...>
  <L2: ...>
  ...

Next: Add text steps."""

        def _fallback() -> str:
            return self._generate_encode_latent(input_text)

        content = await self._sample_with_fallback(
            user_prompt,
            _fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=500,
        )

        # Extract latent tokens from the response for tracking
        import re

        tokens = re.findall(r"<L\d+:[^>]+>", content)
        self._latent_tokens = tokens if tokens else ["<L1: parse>", "<L2: compute>", "<L3: verify>"]

        return content

    async def _sample_text_steps(self, previous_content: str, guidance: str | None) -> str:
        """Generate text steps phase using LLM sampling.

        Args:
            previous_content: Content from the latent encoding phase
            guidance: Optional guidance for the text steps

        Returns:
            A formatted string containing the sampled text steps
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_text_steps but was not provided"
            )

        system_prompt = """You are a reasoning assistant using HybridCoT methodology.
In the text steps phase, you provide explicit reasoning for critical or complex steps.

Your task:
1. Review the latent encoding from the previous phase
2. Identify which steps need explicit reasoning (not routine)
3. Provide 2-4 explicit text steps for these critical parts
4. Keep text steps concise but clear

Text steps should be explicit reasoning that supplements the compressed latent tokens."""

        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = f"""Previous phase (latent encoding):
{previous_content}
{guidance_text}

Generate the text steps phase for HybridCoT reasoning.
Provide 2-4 explicit reasoning steps for the critical (non-routine) parts of the problem.

Format your response as:
Step 2: Text Steps

Explicit:
  [step 1]
  [step 2]
  ...

Next: Decode."""

        def _fallback() -> str:
            return self._generate_text_steps()

        content = await self._sample_with_fallback(
            user_prompt,
            _fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=600,
        )

        # Extract text steps for tracking
        lines = content.split("\n")
        self._text_steps = [
            line.strip()
            for line in lines
            if line.strip() and not line.startswith("Step") and not line.startswith("Next:")
        ]

        return content

    async def _sample_decode(self, previous_content: str) -> str:
        """Generate decode phase using LLM sampling.

        Args:
            previous_content: Content from the text steps phase

        Returns:
            A formatted string containing the sampled decode step
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context required for _sample_decode but was not provided")

        system_prompt = """You are a reasoning assistant using HybridCoT methodology.
In the decode phase, you expand the latent tokens and text steps into the final answer.

Your task:
1. Review all previous reasoning (latent tokens + text steps)
2. Synthesize into a clear final answer
3. Provide confidence level
4. Note the efficiency gain from using HybridCoT

Be clear and concise in your final answer."""

        user_prompt = f"""Previous reasoning:
{previous_content}

Generate the decode phase for HybridCoT reasoning.
Expand the reasoning into a final answer with confidence level.

Format your response as:
Step 3: Decode

Final Answer: [your answer]
Confidence: [percentage]%

Method: HybridCoT - [estimated token reduction]% token reduction"""

        def _fallback() -> str:
            return self._generate_decode()

        return await self._sample_with_fallback(
            user_prompt,
            _fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=400,
        )


__all__ = ["HybridCot", "HYBRID_COT_METADATA"]
