"""Simple Test-Time Scaling (s1) reasoning method.

This module implements Simple Test-Time Scaling based on s1 by Muennighoff et al.
(2025), which uses budget-aware inference with "wait" tokens to allow the model
to think longer on harder problems. Simple yet effective approach to test-time
compute scaling.

Key phases:
1. Budget: Assess difficulty and allocate thinking budget
2. Think: Use wait tokens to extend thinking time
3. Solve: Work through the problem with allocated compute
4. Answer: Produce final answer within budget

Reference: Muennighoff et al. (2025) - "s1: Simple Test-Time Scaling"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session

logger = structlog.get_logger(__name__)


SIMPLE_TEST_TIME_SCALING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SIMPLE_TEST_TIME_SCALING,
    name="Simple Test-Time Scaling (s1)",
    description="Budget-aware test-time scaling with wait tokens. Simple approach "
    "to inference-time compute scaling through budget → think → solve → answer phases.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"test-time-scaling", "budget-aware", "wait-tokens", "s1", "simple", "2025"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=("variable difficulty problems", "budget-constrained inference", "adaptive reasoning"),
    not_recommended_for=("time-critical tasks", "uniform difficulty batches"),
)


class SimpleTestTimeScaling(ReasoningMethodBase):
    """Simple Test-Time Scaling (s1) reasoning method implementation."""

    # Budget levels
    BUDGET_LOW = 2
    BUDGET_MEDIUM = 4
    BUDGET_HIGH = 6

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "budget"
        self._thinking_budget: int = self.BUDGET_MEDIUM
        self._wait_tokens_used: int = 0
        self._difficulty: str = "medium"
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.SIMPLE_TEST_TIME_SCALING

    @property
    def name(self) -> str:
        return SIMPLE_TEST_TIME_SCALING_METADATA.name

    @property
    def description(self) -> str:
        return SIMPLE_TEST_TIME_SCALING_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "budget"
        self._thinking_budget = self.BUDGET_MEDIUM
        self._wait_tokens_used = 0
        self._difficulty = "medium"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Simple Test-Time Scaling must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "budget"

        # Assess difficulty using LLM sampling or fallback heuristic
        def difficulty_fallback() -> str:
            """Fallback heuristic for difficulty assessment."""
            if len(input_text) > 200:
                return "high"
            elif len(input_text) > 100:
                return "medium"
            else:
                return "low"

        difficulty_prompt = f"""Assess the difficulty of this problem and classify it as 'low', 'medium', or 'high':

Problem: {input_text}

Consider:
- Complexity of concepts involved
- Number of steps likely needed
- Ambiguity or nuance required

Respond with only one word: low, medium, or high."""

        difficulty_result_raw = await self._sample_with_fallback(
            difficulty_prompt,
            difficulty_fallback,
            system_prompt="You are an expert at assessing problem difficulty. Respond with only: low, medium, or high.",
        )
        self._difficulty = str(difficulty_result_raw).strip().lower()
        if self._difficulty not in ["low", "medium", "high"]:
            self._difficulty = "medium"

        # Set budget based on difficulty
        if self._difficulty == "high":
            self._thinking_budget = self.BUDGET_HIGH
        elif self._difficulty == "medium":
            self._thinking_budget = self.BUDGET_MEDIUM
        else:
            self._thinking_budget = self.BUDGET_LOW

        content = (
            f"Step {self._step_counter}: Budget Assessment (s1)\n\n"
            f"Problem: {input_text}\n\n"
            f"Difficulty Assessment:\n"
            f"  Estimated difficulty: {self._difficulty.upper()}\n"
            f"  Thinking budget: {self._thinking_budget} steps\n\n"
            f"Budget Allocation:\n"
            f"  ├── Think phase: {self._thinking_budget - 1} steps\n"
            f"  └── Answer phase: 1 step\n\n"
            f"Beginning budget-aware reasoning..."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SIMPLE_TEST_TIME_SCALING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "difficulty": self._difficulty,
                "budget": self._thinking_budget,
                "wait_tokens": self._wait_tokens_used,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.SIMPLE_TEST_TIME_SCALING
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
            raise RuntimeError("Simple Test-Time Scaling must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "budget")

        if prev_phase == "budget":
            self._current_phase = "think"
            self._wait_tokens_used = 1

            # Generate thinking step using LLM sampling or fallback
            def think_fallback() -> str:
                """Fallback heuristic for initial thinking step."""
                return (
                    "Processing problem...\n"
                    "Identifying key elements...\n"
                    "Considering approaches..."
                )

            think_prompt = f"""Based on the problem below, perform initial analysis:

Problem: {session.input_text if hasattr(session, "input_text") else "Previous problem"}

Analyze:
1. What are the key elements?
2. What approaches might work?
3. What should we focus on?

Keep your response concise (2-3 sentences)."""

            thinking_result = await self._sample_with_fallback(
                think_prompt,
                think_fallback,
                system_prompt="You are helping with test-time scaling. Provide brief initial analysis.",
            )

            content = (
                f"Step {self._step_counter}: Thinking (Wait Token 1/{self._thinking_budget - 1})\n\n"
                f"<wait>\n"
                f"{thinking_result}\n"
                f"</wait>\n\n"
                f"Progress: Initial analysis complete\n"
                f"Remaining budget: {self._thinking_budget - self._wait_tokens_used - 1} steps"
            )

            thought_type = ThoughtType.REASONING
            confidence = 0.65
        elif prev_phase == "think":
            self._wait_tokens_used += 1
            if self._wait_tokens_used < self._thinking_budget - 1:
                # Continue thinking
                def continue_fallback() -> str:
                    """Fallback heuristic for continued thinking."""
                    return (
                        "Deepening analysis...\n"
                        "Working through implications...\n"
                        "Refining understanding..."
                    )

                continue_prompt = f"""Continue deeper analysis on the problem.

Previous thought: {previous_thought.content}

Now:
1. Deepen the analysis
2. Work through implications
3. Refine understanding

Keep your response concise (2-3 sentences)."""

                continue_result = await self._sample_with_fallback(
                    continue_prompt,
                    continue_fallback,
                    system_prompt="You are helping with test-time scaling. Provide deeper analysis building on previous work.",
                )

                content = (
                    f"Step {self._step_counter}: Thinking (Wait Token {self._wait_tokens_used}/{self._thinking_budget - 1})\n\n"
                    f"<wait>\n"
                    f"{continue_result}\n"
                    f"</wait>\n\n"
                    f"Progress: Analysis {self._wait_tokens_used * 100 // (self._thinking_budget - 1)}% complete\n"
                    f"Remaining budget: {self._thinking_budget - self._wait_tokens_used - 1} steps"
                )

                thought_type = ThoughtType.REASONING
                confidence = 0.65 + (self._wait_tokens_used * 0.05)
            else:
                # Move to solve
                self._current_phase = "solve"

                def solve_fallback() -> str:
                    """Fallback heuristic for solution derivation."""
                    return (
                        "Solution Process:\n"
                        "  1. [Apply insights from thinking phase]\n"
                        "  2. [Execute solution strategy]\n"
                        "  3. [Verify result]"
                    )

                solve_prompt = f"""Thinking budget consumed. Now derive the solution.

Previous thoughts: {previous_thought.content}

Provide a solution process:
1. Apply insights from thinking phase
2. Execute solution strategy
3. Verify result

Keep your response structured and concise."""

                solve_result = await self._sample_with_fallback(
                    solve_prompt,
                    solve_fallback,
                    system_prompt="You are solving a problem after completing the thinking phase. Provide a clear solution process.",
                )

                content = (
                    f"Step {self._step_counter}: Solving\n\n"
                    f"Thinking budget consumed. Deriving solution...\n\n"
                    f"{solve_result}\n\n"
                    f"Ready to produce final answer."
                )

                thought_type = ThoughtType.SYNTHESIS
                confidence = 0.8
        elif prev_phase == "solve":
            self._current_phase = "answer"

            def answer_fallback() -> str:
                """Fallback for final answer generation."""
                confidence_text = "High" if self._difficulty != "high" else "Moderate"
                confidence_pct = 85 if self._difficulty != "high" else 75
                return (
                    f"Final Answer: [Answer derived with allocated compute]\n\n"
                    f"Confidence: {confidence_text} ({confidence_pct}%)"
                )

            answer_prompt = f"""Provide the final answer based on the solution process.

Previous solution: {previous_thought.content}

Provide:
1. The final answer
2. Confidence level and reason

Keep your response concise and clear."""

            answer_result = await self._sample_with_fallback(
                answer_prompt,
                answer_fallback,
                system_prompt="You are providing the final answer after completing the solution process. Be clear and concise.",
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"s1 Test-Time Scaling Complete:\n"
                f"  - Difficulty: {self._difficulty}\n"
                f"  - Budget: {self._thinking_budget} steps\n"
                f"  - Wait tokens used: {self._wait_tokens_used}\n\n"
                f"{answer_result}\n\n"
                f"Reason: Budget-appropriate thinking for {self._difficulty} difficulty problem."
            )

            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85 if self._difficulty != "high" else 0.75
        else:
            self._current_phase = "answer"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Final Answer: [Answer]\n"
                f"Confidence: Moderate (80%)"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.8

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SIMPLE_TEST_TIME_SCALING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "difficulty": self._difficulty,
                "budget": self._thinking_budget,
                "wait_tokens": self._wait_tokens_used,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["SimpleTestTimeScaling", "SIMPLE_TEST_TIME_SCALING_METADATA"]
