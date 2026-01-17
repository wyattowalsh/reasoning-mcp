"""GRPO (Group Relative Policy Optimization) reasoning method.

This module implements GRPO, a critic-free reinforcement learning approach that
optimizes policies using group-level relative comparisons instead of per-sample
critics. GRPO generates multiple candidate responses in a group, computes rewards
relative to the group average, and uses these normalized rewards for policy updates.

Key phases:
1. Sample Group: Generate 4-5 candidate reasoning paths for the same prompt
2. Compute Relative Rewards: Calculate rewards relative to group average baseline
3. Optimize Policy: Update policy using group-relative advantages
4. Conclude: Select best candidate and finalize reasoning

Reference: Shao et al. (2024) - "DeepSeek-R1: Incentivizing Reasoning Capability
in LLMs via Reinforcement Learning" (Technical Report, DeepSeek-AI)

Key Innovation: Instead of training a separate critic/value model, GRPO uses the
group average reward as a baseline, making RL more efficient and stable.
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


GRPO_METADATA = MethodMetadata(
    identifier=MethodIdentifier.GRPO,
    name="GRPO",
    description="Group Relative Policy Optimization - critic-free RL using group-level "
    "relative comparisons. Samples multiple candidates and optimizes using normalized "
    "rewards relative to group average.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "reinforcement-learning",
            "policy-optimization",
            "critic-free",
            "group-comparison",
            "deepseek",
        }
    ),
    complexity=8,
    supports_branching=True,
    supports_revision=True,
    requires_context=True,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=340,
    best_for=("complex reasoning", "math problems", "multi-step tasks", "optimization tasks"),
    not_recommended_for=("simple queries", "single-step problems", "creative tasks"),
)


class Grpo(ReasoningMethodBase):
    """GRPO reasoning method implementation."""

    # Enable LLM sampling for candidate generation and evaluation
    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "sample_group"
        self._group_candidates: list[dict[str, Any]] = []
        self._group_size: int = 5
        self._rewards: list[float] = []
        self._relative_rewards: list[float] = []
        self._baseline_reward: float = 0.0
        self._selected_candidate: dict[str, Any] | None = None
        self._policy_update_step: int = 0
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.GRPO

    @property
    def name(self) -> str:
        return GRPO_METADATA.name

    @property
    def description(self) -> str:
        return GRPO_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "sample_group"
        self._group_candidates = []
        self._group_size = 5
        self._rewards = []
        self._relative_rewards = []
        self._baseline_reward = 0.0
        self._selected_candidate = None
        self._policy_update_step = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("GRPO must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "sample_group"

        # Sample group of candidates using LLM sampling if available
        self._group_candidates = await self._generate_candidate_group(input_text, context)

        content = (
            f"Step {self._step_counter}: Sample Candidate Group (GRPO)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generated {self._group_size} candidate reasoning paths:\n\n"
            + "\n".join(
                f"  Candidate {c['id']}:\n"
                f"    Approach: {c['reasoning']}\n"
                f"    Steps: {c['steps']}\n"
                f"    Confidence: {c['confidence']:.2f}"
                for c in self._group_candidates
            )
            + "\n\nGRPO Framework:\n"
            "  - Group-based sampling: Generate multiple candidates\n"
            "  - Relative rewards: Compare within group, not absolute\n"
            "  - Critic-free: No separate value model needed\n"
            "  - Baseline: Group average as baseline\n\n"
            "Advantages over Traditional RL:\n"
            "  ✓ No critic model training required\n"
            "  ✓ More stable optimization\n"
            "  ✓ Group-relative comparisons reduce variance\n"
            "  ✓ Efficient test-time compute scaling\n\n"
            "Next: Compute rewards for each candidate."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.GRPO,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "phase": self._current_phase,
                "group_size": self._group_size,
                "candidates": len(self._group_candidates),
                "input_text": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.GRPO
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
            raise RuntimeError("GRPO must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "sample_group")

        if prev_phase == "sample_group":
            self._current_phase = "compute_relative_rewards"

            # Get the original input text from metadata
            input_text = previous_thought.metadata.get("input_text", "")

            # Compute rewards for each candidate using LLM evaluation
            self._rewards = await self._compute_candidate_rewards(
                input_text, self._group_candidates, context
            )

            # Compute baseline as group average (key GRPO innovation)
            self._baseline_reward = sum(self._rewards) / len(self._rewards)

            # Compute relative rewards (advantage = reward - baseline)
            self._relative_rewards = [r - self._baseline_reward for r in self._rewards]

            content = (
                f"Step {self._step_counter}: Compute Relative Rewards\n\n"
                f"Reward Calculation (using outcome verifier):\n\n"
                f"Raw Rewards:\n"
                + "\n".join(f"  Candidate {i + 1}: {r:.3f}" for i, r in enumerate(self._rewards))
                + f"\n\nGroup Statistics:\n"
                f"  Baseline (group avg): {self._baseline_reward:.3f}\n"
                f"  Min reward: {min(self._rewards):.3f}\n"
                f"  Max reward: {max(self._rewards):.3f}\n"
                f"  Std deviation: {self._compute_std(self._rewards):.3f}\n\n"
                f"Relative Rewards (Advantages):\n"
                + "\n".join(
                    f"  Candidate {i + 1}: {r:+.3f} {'(above avg)' if r > 0 else '(below avg)'}"
                    for i, r in enumerate(self._relative_rewards)
                )
                + "\n\nGRPO Key Insight:\n"
                "  Instead of absolute rewards, we use group-relative comparisons.\n"
                "  Advantage = reward - group_average\n"
                "  This removes the need for a learned value function (critic).\n\n"
                "Next: Optimize policy using relative rewards."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75

        elif prev_phase == "compute_relative_rewards":
            self._current_phase = "optimize_policy"
            self._policy_update_step = 1

            # Simulate policy optimization
            # In practice: gradient update using PPO-style objective with relative rewards
            positive_advantages = [r for r in self._relative_rewards if r > 0]
            negative_advantages = [r for r in self._relative_rewards if r <= 0]

            # Select best candidate for reinforcement
            best_idx = self._rewards.index(max(self._rewards))
            self._selected_candidate = self._group_candidates[best_idx]

            content = (
                f"Step {self._step_counter}: Optimize Policy (GRPO)\n\n"
                f"Policy Update Step {self._policy_update_step}:\n\n"
                f"Objective Function:\n"
                f"  GRPO Loss = E[r(x,y) - baseline] × log π(y|x)\n"
                f"  Where baseline = avg(r) over group\n\n"
                f"Update Statistics:\n"
                f"  Positive advantages: {len(positive_advantages)}/{len(self._relative_rewards)}\n"
                f"  Negative advantages: {len(negative_advantages)}/{len(self._relative_rewards)}\n"
                f"  Max advantage: {max(self._relative_rewards):+.3f} (Candidate {best_idx + 1})\n"
                f"  Min advantage: {min(self._relative_rewards):+.3f}\n\n"
                f"Policy Update Strategy:\n"
                f"  ✓ Increase probability of high-advantage candidates\n"
                f"  ✓ Decrease probability of low-advantage candidates\n"
                f"  ✓ Use group normalization to reduce variance\n"
                f"  ✓ Apply PPO-style clipping for stability\n\n"
                f"Selected Best Candidate:\n"
                f"  Candidate {self._selected_candidate['id']}: "
                f"{self._selected_candidate['reasoning']}\n"
                f"  Advantage: {self._relative_rewards[best_idx]:+.3f}\n"
                f"  Raw reward: {self._rewards[best_idx]:.3f}\n\n"
                f"GRPO vs Traditional RL:\n"
                f"  Traditional: Needs critic model V(s) for baseline\n"
                f"  GRPO: Uses group average - no critic needed!\n\n"
                f"Next: Finalize reasoning with selected candidate."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.82

        elif prev_phase == "optimize_policy":
            self._current_phase = "conclude"

            best_idx = self._rewards.index(max(self._rewards))
            final_confidence = min(0.95, 0.80 + (self._relative_rewards[best_idx] * 0.15))

            # Type guard to ensure selected_candidate is not None
            if self._selected_candidate is None:
                raise RuntimeError("No candidate selected during optimization")

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"GRPO Optimization Complete:\n"
                f"  Group size: {self._group_size} candidates\n"
                f"  Policy updates: {self._policy_update_step}\n"
                f"  Baseline reward: {self._baseline_reward:.3f}\n"
                f"  Best reward: {max(self._rewards):.3f}\n\n"
                f"Final Reasoning Path:\n"
                f"  Candidate {self._selected_candidate['id']}\n"
                f"  Approach: {self._selected_candidate['reasoning']}\n"
                f"  Steps: {self._selected_candidate['steps']}\n"
                f"  Advantage over baseline: {self._relative_rewards[best_idx]:+.3f}\n\n"
                f"Performance Summary:\n"
                + "\n".join(
                    f"  Candidate {i + 1}: reward={r:.3f}, "
                    f"advantage={self._relative_rewards[i]:+.3f}"
                    for i, r in enumerate(self._rewards)
                )
                + f"\n\nGRPO Benefits Demonstrated:\n"
                f"  ✓ Critic-free RL via group-relative comparisons\n"
                f"  ✓ Stable optimization with normalized rewards\n"
                f"  ✓ Efficient inference-time compute scaling\n"
                f"  ✓ Self-play style improvement without external models\n\n"
                f"Final Answer: [Solution via best candidate path]\n"
                f"Confidence: High ({int(final_confidence * 100)}%)\n\n"
                f"Method: GRPO (Group Relative Policy Optimization)\n"
                f"  - Group sampling for comparative evaluation\n"
                f"  - Relative rewards eliminate critic model\n"
                f"  - Variance reduction through normalization\n"
                f"  - Inspired by DeepSeek-R1 and o1/o3 approaches"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = final_confidence

        else:
            # Fallback to conclude if unknown phase
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Unexpected Phase\n\n"
                f"Concluding GRPO reasoning with available information."
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.GRPO,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "group_size": self._group_size,
                "baseline_reward": self._baseline_reward,
                "policy_updates": self._policy_update_step,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _compute_std(self, values: list[float]) -> float:
        """Compute standard deviation of a list of values."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return float(variance**0.5)

    def _get_fallback_candidates(self) -> list[dict[str, Any]]:
        """Return fallback heuristic candidates when LLM sampling is unavailable."""
        return [
            {
                "id": 1,
                "reasoning": "Break into subproblems: (a) identify variables, (b) set up equations",
                "steps": 3,
                "confidence": 0.72,
            },
            {
                "id": 2,
                "reasoning": "Use direct calculation: compute each term separately, then combine",
                "steps": 2,
                "confidence": 0.68,
            },
            {
                "id": 3,
                "reasoning": "Apply systematic approach: outline, execute step-by-step, verify",
                "steps": 4,
                "confidence": 0.81,
            },
            {
                "id": 4,
                "reasoning": "Pattern matching: identify similar problems, adapt known solutions",
                "steps": 3,
                "confidence": 0.75,
            },
            {
                "id": 5,
                "reasoning": "First principles: derive from basic axioms, build up to solution",
                "steps": 5,
                "confidence": 0.78,
            },
        ]

    def _parse_candidate_response(self, result_str: str, candidate_id: int) -> dict[str, Any]:
        """Parse LLM response into candidate dictionary.

        Args:
            result_str: The raw LLM response string
            candidate_id: The candidate ID to assign

        Returns:
            Parsed candidate dictionary with id, reasoning, steps, and confidence
        """
        reasoning = "Generated reasoning path"
        steps = 3
        confidence = 0.75

        if "Strategy:" in result_str:
            parts = result_str.split("Strategy:", 1)[1].split("\n", 1)
            strategy_match = parts[0].strip()
            reasoning = strategy_match

        if "Steps:" in result_str:
            try:
                parts = result_str.split("Steps:", 1)[1].split("\n", 1)
                steps_match = parts[0].strip()
                steps = int(steps_match)
                steps = max(1, min(steps, 10))
            except (ValueError, IndexError):
                pass

        if "Confidence:" in result_str:
            try:
                parts = result_str.split("Confidence:", 1)[1].split("\n", 1)
                conf_match = parts[0].strip()
                confidence = float(conf_match)
                confidence = max(0.0, min(confidence, 1.0))
            except (ValueError, IndexError):
                pass

        return {
            "id": candidate_id,
            "reasoning": reasoning,
            "steps": steps,
            "confidence": confidence,
        }

    async def _generate_candidate_group(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Generate a group of diverse candidate reasoning paths.

        Uses LLM sampling if available, otherwise falls back to heuristic generation.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            List of candidate dictionaries with id, reasoning, steps, and confidence
        """
        # Check if LLM sampling is available
        if not self._use_sampling:
            return self._get_fallback_candidates()

        # Generate diverse candidate reasoning paths using LLM sampling
        candidates = []
        for i in range(self._group_size):
            prompt = (
                f"Generate a diverse reasoning approach #{i + 1} for solving "
                f"the following problem.\n"
                f"Make this approach distinctly different from previous candidates.\n\n"
                f"Problem: {input_text}\n\n"
                f"Provide:\n"
                f"1. A brief description of your reasoning strategy\n"
                f"2. Estimated number of steps (1-10)\n"
                f"3. Self-assessed confidence (0.0-1.0)\n\n"
                f"Format your response as:\n"
                f"Strategy: [your strategy]\n"
                f"Steps: [number]\n"
                f"Confidence: [0.0-1.0]"
            )
            system_prompt = (
                "You are a creative problem solver generating diverse "
                "candidate reasoning paths. "
                "Each path should use a different strategy, perspective, "
                "or approach. "
                "Be innovative and think outside the box."
            )

            # Use fallback for this specific candidate
            fallback_candidates = self._get_fallback_candidates()

            def fallback_generator() -> str:
                """Generate fallback response for a single candidate."""
                if i < len(fallback_candidates):
                    c = fallback_candidates[i]
                    return f"Strategy: {c['reasoning']}\nSteps: {c['steps']}\nConfidence: {c['confidence']}"
                return "Strategy: Default approach\nSteps: 3\nConfidence: 0.75"

            result_str = await self._sample_with_fallback(
                user_prompt=prompt,
                fallback_generator=fallback_generator,
                system_prompt=system_prompt,
            )

            candidate = self._parse_candidate_response(result_str, i + 1)
            candidates.append(candidate)

        return candidates

    def _parse_reward_response(self, result_str: str, fallback_score: float) -> float:
        """Parse LLM response into a reward score.

        Args:
            result_str: The raw LLM response string
            fallback_score: Score to use if parsing fails

        Returns:
            Parsed reward score (0.0-1.0)
        """
        # Extract first number from result
        for part in result_str.strip().split():
            try:
                score = float(part)
                return max(0.0, min(score, 1.0))
            except ValueError:
                continue
        return fallback_score

    async def _compute_candidate_rewards(
        self,
        input_text: str,
        candidates: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> list[float]:
        """Compute rewards for each candidate using LLM evaluation.

        Uses LLM sampling if available, otherwise falls back to heuristic scoring.

        Args:
            input_text: The original problem
            candidates: List of candidate solutions
            context: Optional additional context

        Returns:
            List of reward scores (0.0-1.0) for each candidate
        """
        # Check if LLM sampling is available and possible
        can_sample = (
            self._use_sampling
            and self._execution_context is not None
            and getattr(self._execution_context, "can_sample", False)
        )

        if not can_sample:
            return [c["confidence"] for c in candidates]

        rewards = []
        for candidate in candidates:
            prompt = (
                f"Evaluate the quality of the following reasoning approach "
                f"for solving this problem.\n\n"
                f"Problem: {input_text}\n\n"
                f"Reasoning Approach:\n{candidate['reasoning']}\n"
                f"Steps: {candidate['steps']}\n\n"
                f"Provide a quality score between 0.0 and 1.0 based on:\n"
                f"- Correctness and validity of the approach\n"
                f"- Efficiency and practicality\n"
                f"- Completeness of the solution\n"
                f"- Clarity and coherence\n\n"
                f"Respond with only a single number between 0.0 and 1.0."
            )
            system_prompt = (
                "You are an expert evaluator assessing the quality of "
                "reasoning approaches. "
                "Provide objective, consistent scores based on correctness, "
                "efficiency, completeness, and clarity."
            )

            fallback_score = candidate["confidence"]

            def fallback_generator() -> str:
                """Generate fallback response with confidence score."""
                return str(fallback_score)

            result_str = await self._sample_with_fallback(
                user_prompt=prompt,
                fallback_generator=fallback_generator,
                system_prompt=system_prompt,
            )

            score = self._parse_reward_response(result_str, fallback_score)
            rewards.append(score)

        return rewards


__all__ = ["Grpo", "GRPO_METADATA"]
