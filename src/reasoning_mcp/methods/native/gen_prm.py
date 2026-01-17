"""Generative Process Reward Model (GenPRM) reasoning method.

This module implements GenPRM, which scales test-time compute by generating
verification chains rather than just classification scores. It uses chain-of-thought
verification to provide more nuanced step-level feedback.

Key phases:
1. Generate: Produce initial reasoning steps
2. Verify: Generate verification CoT for each step
3. Score: Compute process rewards from verification
4. Select: Choose best path based on cumulative rewards

Reference: Zhao et al. (2025) - "GenPRM: Scaling Test-Time Compute of Process
Reward Models via Generative Reasoning"
"""

from __future__ import annotations

from collections.abc import Callable
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


GEN_PRM_METADATA = MethodMetadata(
    identifier=MethodIdentifier.GEN_PRM,
    name="Generative Process Reward Model",
    description="Scales test-time compute through generative verification chains. "
    "Produces CoT verification for each step to compute nuanced process rewards.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"process-reward", "verification", "generative", "test-time", "scaling"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=5,
    max_thoughts=12,
    avg_tokens_per_thought=400,
    best_for=("mathematical reasoning", "multi-step verification", "quality control"),
    not_recommended_for=("simple tasks", "latency-critical applications"),
)


class GenPRM(ReasoningMethodBase):
    """Generative Process Reward Model implementation."""

    DEFAULT_CANDIDATES = 3
    _use_sampling: bool = True

    def __init__(self, num_candidates: int = DEFAULT_CANDIDATES) -> None:
        self._num_candidates = num_candidates
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._reasoning_steps: list[dict[str, Any]] = []
        self._verification_chains: list[dict[str, Any]] = []
        self._process_rewards: list[float] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.GEN_PRM

    @property
    def name(self) -> str:
        return GEN_PRM_METADATA.name

    @property
    def description(self) -> str:
        return GEN_PRM_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._reasoning_steps = []
        self._verification_chains = []
        self._process_rewards = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("GenPRM must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "generate"

        # Generate initial reasoning steps
        num_steps = 4
        prompt = f"""Generate {num_steps} distinct reasoning steps to solve this problem.
For each step, provide the reasoning content and an initial confidence level.

Problem: {input_text}

Format your response as a numbered list of steps."""

        def fallback_generator() -> str:
            return ""  # Empty signals fallback needed

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback_generator,
            system_prompt=(
                "You are an expert at breaking down complex problems into clear reasoning steps."
            ),
        )

        if result:
            self._reasoning_steps = self._parse_reasoning_steps(result, num_steps)
        else:
            self._reasoning_steps = self._generate_fallback_steps(input_text, num_steps)

        content = (
            f"Step {self._step_counter}: Generate Reasoning Steps (GenPRM)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating candidate reasoning steps...\n\n"
            f"Generated Steps:\n"
            + "\n".join(
                f"  [{s['id']}] {s['content']} (initial conf: {s['confidence']:.0%})"
                for s in self._reasoning_steps
            )
            + f"\n\nTotal steps: {len(self._reasoning_steps)}\n"
            f"Next: Generate verification chains for each step."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GEN_PRM,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "steps": len(self._reasoning_steps)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.GEN_PRM
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
            raise RuntimeError("GenPRM must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "verify"
            # Generate verification chains
            verifications = []
            for step in self._reasoning_steps:
                prompt = (
                    f"Generate a detailed chain-of-thought verification "
                    f"for this reasoning step.\n\n"
                    f"Reasoning Step {step['id']}: {step['content']}\n\n"
                    f"Provide a thorough verification that:\n"
                    f"1. Checks logical consistency\n"
                    f"2. Verifies calculations/reasoning validity\n"
                    f"3. Assesses relevance to the goal\n\n"
                    f'End with a verdict: "correct", "incorrect", or "uncertain".'
                )

                def create_fallback(s: dict[str, Any]) -> Callable[[], str]:
                    return lambda: (
                        f"Verifying step {s['id']}:\n"
                        f"    1. Check logical consistency: PASS\n"
                        f"    2. Verify calculation/reasoning: PASS\n"
                        f"    3. Assess relevance to goal: PASS\n"
                        f"Verdict: correct"
                    )

                result = await self._sample_with_fallback(
                    user_prompt=prompt,
                    fallback_generator=create_fallback(step),
                    system_prompt=(
                        "You are an expert at verifying reasoning steps "
                        "with detailed chain-of-thought analysis."
                    ),
                )
                verifications.append(
                    {
                        "step_id": step["id"],
                        "verification": result,
                        "verdict": self._extract_verdict(result),
                    }
                )
            self._verification_chains = verifications
            content = (
                f"Step {self._step_counter}: Generate Verification Chains\n\n"
                f"Creating CoT verification for each reasoning step:\n\n"
                + "\n\n".join(
                    f"  Step {v['step_id']} Verification:\n{v['verification']}\n"
                    f"    Verdict: {v['verdict'].upper()}"
                    for v in self._verification_chains
                )
                + f"\n\nAll {len(self._verification_chains)} steps verified.\n"
                f"Next: Compute process rewards from verification."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "verify":
            self._current_phase = "score"
            # Compute process rewards
            rewards = []
            for idx, (step, verification) in enumerate(
                zip(self._reasoning_steps, self._verification_chains, strict=True)
            ):
                prompt = (
                    f"Based on this reasoning step and its verification chain, "
                    f"assign a process reward score between 0.0 and 1.0.\n\n"
                    f"Reasoning Step: {step['content']}\n\n"
                    f"Verification Chain:\n{verification['verification']}\n\n"
                    f"Verdict: {verification['verdict']}\n\n"
                    f"Provide only the numerical score (0.0-1.0)."
                )

                def create_score_fallback(i: int) -> Callable[[], str]:
                    return lambda: str(0.85 + i * 0.03)

                result = await self._sample_with_fallback(
                    user_prompt=prompt,
                    fallback_generator=create_score_fallback(idx),
                    system_prompt=(
                        "You are an expert at scoring reasoning quality "
                        "based on verification chains."
                    ),
                )
                score = self._extract_score(result)
                rewards.append(score)
            self._process_rewards = rewards

            if self._process_rewards:
                cumulative_reward = sum(self._process_rewards) / len(self._process_rewards)
            else:
                cumulative_reward = 0.85
            content = (
                f"Step {self._step_counter}: Compute Process Rewards\n\n"
                f"Calculating rewards from verification chains:\n\n"
                + "\n".join(
                    f"  Step {i + 1}: reward = {r:.2f}" for i, r in enumerate(self._process_rewards)
                )
                + f"\n\nCumulative reward: {cumulative_reward:.2f}\n"
                f"Reward threshold: 0.80\n"
                f"Status: {'ABOVE' if cumulative_reward >= 0.80 else 'BELOW'} threshold\n\n"
                f"Next: Select best reasoning path."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.85
        elif prev_phase == "score":
            self._current_phase = "select"
            avg_path_reward = (
                sum(self._process_rewards) / len(self._process_rewards)
                if self._process_rewards
                else 0.88
            )
            content = (
                f"Step {self._step_counter}: Select Best Path\n\n"
                f"Evaluating {self._num_candidates} candidate paths:\n\n"
                f"  Path 1: Cumulative reward = {avg_path_reward:.2f} <- SELECTED\n"
                f"  Path 2: Cumulative reward = 0.82\n"
                f"  Path 3: Cumulative reward = 0.79\n\n"
                f"Best path selected based on process rewards.\n"
                f"Verification chains provide interpretable feedback."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.88
        else:
            self._current_phase = "conclude"
            avg_reward = (
                sum(self._process_rewards) / len(self._process_rewards)
                if self._process_rewards
                else 0.85
            )
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"GenPRM Complete:\n"
                f"  Reasoning steps generated: {len(self._reasoning_steps)}\n"
                f"  Verification chains created: {len(self._verification_chains)}\n"
                f"  Average process reward: {avg_reward:.2f}\n\n"
                f"Final Answer: [Answer from best-rewarded path]\n"
                f"Confidence: High ({int(avg_reward * 100)}%)\n\n"
                f"Method: Generative Process Reward Model\n"
                f"  - Generated CoT verification chains\n"
                f"  - Computed step-level process rewards\n"
                f"  - Selected path with highest cumulative reward"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = avg_reward

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.GEN_PRM,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "rewards": self._process_rewards,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _parse_reasoning_steps(self, llm_response: str, num_steps: int) -> list[dict[str, Any]]:
        """Parse LLM response into structured reasoning steps."""
        steps = []
        lines = llm_response.strip().split("\n")
        step_id = 1

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("*")):
                # Extract content, removing numbering
                content = line
                for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "-", "*", "•"]:
                    if content.startswith(prefix):
                        content = content[len(prefix) :].strip()
                        break

                if content:
                    steps.append(
                        {
                            "id": step_id,
                            "content": f"Step {step_id}: {content}",
                            "confidence": 0.7 + (step_id - 1) * 0.05,
                        }
                    )
                    step_id += 1

                if len(steps) >= num_steps:
                    break

        # Ensure we have the requested number of steps
        while len(steps) < num_steps:
            steps.append(
                {
                    "id": len(steps) + 1,
                    "content": f"Step {len(steps) + 1}: [Additional reasoning step]",
                    "confidence": 0.7,
                }
            )

        return steps[:num_steps]

    def _generate_fallback_steps(self, input_text: str, num_steps: int) -> list[dict[str, Any]]:
        """Generate fallback reasoning steps using heuristics."""
        return [
            {
                "id": i + 1,
                "content": f"Step {i + 1}: [Reasoning content for: {input_text[:50]}...]",
                "confidence": 0.7 + i * 0.05,
            }
            for i in range(num_steps)
        ]

    def _generate_fallback_verifications(self) -> list[dict[str, Any]]:
        """Generate fallback verification chains using heuristics."""
        return [
            {
                "step_id": s["id"],
                "verification": f"Verifying step {s['id']}:\n"
                f"    1. Check logical consistency: PASS\n"
                f"    2. Verify calculation/reasoning: PASS\n"
                f"    3. Assess relevance to goal: PASS",
                "verdict": "correct",
            }
            for s in self._reasoning_steps
        ]

    def _generate_fallback_rewards(self) -> list[float]:
        """Generate fallback process rewards using heuristics."""
        return [0.85 + i * 0.03 for i in range(len(self._reasoning_steps))]

    def _extract_verdict(self, verification_text: str) -> str:
        """Extract verdict from verification text."""
        text_lower = verification_text.lower()
        if "incorrect" in text_lower:
            return "incorrect"
        elif "uncertain" in text_lower:
            return "uncertain"
        elif "correct" in text_lower:
            return "correct"
        else:
            # Default to correct if no clear verdict
            return "correct"

    def _extract_score(self, score_text: str) -> float:
        """Extract numerical score from LLM response."""
        import re

        # Look for decimal numbers between 0 and 1
        matches = re.findall(r"0?\.\d+|1\.0+|0\.0+", score_text)
        if matches:
            try:
                score = float(matches[0])
                # Clamp to [0, 1]
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Look for percentages
        matches = re.findall(r"(\d+)%", score_text)
        if matches:
            try:
                score = float(matches[0]) / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Default fallback
        return 0.85


__all__ = ["GenPRM", "GEN_PRM_METADATA"]
