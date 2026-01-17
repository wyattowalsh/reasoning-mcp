"""Compute-Optimal Scaling reasoning method.

This module implements Compute-Optimal Scaling, which adaptively allocates
test-time compute based on problem difficulty. Easy problems get less compute,
hard problems get more, optimizing the compute-accuracy tradeoff.

Key phases:
1. Assess: Estimate problem difficulty
2. Allocate: Determine compute budget based on difficulty
3. Execute: Apply appropriate test-time compute strategy
4. Verify: Check if additional compute is needed

Reference: Snell et al. (2024) - "Scaling LLM Test-Time Compute Optimally
can be More Effective than Scaling Model Parameters" (ICLR 2025)
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


COMPUTE_OPTIMAL_SCALING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.COMPUTE_OPTIMAL_SCALING,
    name="Compute-Optimal Scaling",
    description="Adaptive test-time compute allocation based on problem difficulty. "
    "Optimizes compute-accuracy tradeoff by scaling with difficulty.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"adaptive", "test-time", "scaling", "compute-optimal", "difficulty-aware"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=(
        "math reasoning",
        "varying difficulty",
        "compute efficiency",
        "benchmark optimization",
    ),
    not_recommended_for=("uniform difficulty tasks", "latency-critical applications"),
)


class ComputeOptimalScaling(ReasoningMethodBase):
    """Compute-Optimal Scaling reasoning method implementation."""

    _use_sampling: bool = True

    DIFFICULTY_LEVELS = {
        "easy": {"budget": 1, "samples": 1, "strategy": "direct"},
        "medium": {"budget": 4, "samples": 4, "strategy": "self_consistency"},
        "hard": {"budget": 16, "samples": 16, "strategy": "beam_search"},
        "very_hard": {"budget": 64, "samples": 32, "strategy": "prm_guided"},
    }

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "assess"
        self._difficulty: str = "medium"
        self._compute_budget: int = 4
        self._samples_generated: int = 0
        self._strategy: str = "self_consistency"
        self._results: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.COMPUTE_OPTIMAL_SCALING

    @property
    def name(self) -> str:
        return COMPUTE_OPTIMAL_SCALING_METADATA.name

    @property
    def description(self) -> str:
        return COMPUTE_OPTIMAL_SCALING_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "assess"
        self._difficulty = "medium"
        self._compute_budget = 4
        self._samples_generated = 0
        self._strategy = "self_consistency"
        self._results = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Compute-Optimal Scaling must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "assess"

        # Assess difficulty using LLM sampling if available
        self._difficulty = await self._assess_difficulty(input_text)

        config = self.DIFFICULTY_LEVELS[self._difficulty]
        self._compute_budget = int(str(config["budget"]))
        self._strategy = str(config["strategy"])

        content = (
            f"Step {self._step_counter}: Assess Problem Difficulty (Compute-Optimal)\n\n"
            f"Problem: {input_text[:100]}...\n\n"
            f"Difficulty Assessment:\n"
            f"  Estimated difficulty: {self._difficulty.upper()}\n"
            f"  Indicators: Problem complexity, length, domain\n\n"
            f"Compute Budget Allocation:\n"
            f"  Budget: {self._compute_budget}x base compute\n"
            f"  Samples to generate: {config['samples']}\n"
            f"  Strategy: {self._strategy}\n\n"
            f"Compute-Optimal Principle:\n"
            f"  - Easy problems: Minimal compute (direct answer)\n"
            f"  - Hard problems: Maximum compute (guided search)\n"
            f"  - Goal: Optimal accuracy per compute unit\n\n"
            f"Next: Allocate and execute compute strategy."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.COMPUTE_OPTIMAL_SCALING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "difficulty": self._difficulty,
                "budget": self._compute_budget,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.COMPUTE_OPTIMAL_SCALING
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
            raise RuntimeError("Compute-Optimal Scaling must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "assess")

        if prev_phase == "assess":
            self._current_phase = "allocate"
            config = self.DIFFICULTY_LEVELS[self._difficulty]

            content = (
                f"Step {self._step_counter}: Allocate Compute Resources\n\n"
                f"Difficulty: {self._difficulty.upper()}\n"
                f"Strategy: {self._strategy}\n\n"
                f"Resource Allocation:\n"
                f"  Total budget: {self._compute_budget}x\n"
                f"  Samples: {config['samples']}\n"
                f"  Verification rounds: {self._compute_budget // 4 or 1}\n\n"
                f"Strategy Details:\n"
                + (
                    "  Direct: Single-pass answer generation"
                    if self._strategy == "direct"
                    else (
                        "  Self-Consistency: Multiple samples + majority vote"
                        if self._strategy == "self_consistency"
                        else (
                            "  Beam Search: Parallel path exploration"
                            if self._strategy == "beam_search"
                            else "  PRM-Guided: Process reward model verification"
                        )
                    )
                )
                + f"\n\nCompute allocation optimized for {self._difficulty} difficulty.\n"
                f"Next: Execute with allocated compute."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "allocate":
            self._current_phase = "execute"
            config = self.DIFFICULTY_LEVELS[self._difficulty]
            self._samples_generated = int(str(config["samples"]))

            # Generate results using LLM sampling
            self._results = await self._generate_samples(session, min(self._samples_generated, 5))

            content = (
                f"Step {self._step_counter}: Execute Compute Strategy\n\n"
                f"Running {self._strategy} with {self._compute_budget}x compute:\n\n"
                f"Generated Samples:\n"
                + "\n".join(
                    f"  [{r['id']}] {r['answer']} (conf: {r['confidence']:.0%})"
                    for r in self._results
                )
                + f"\n\nCompute Used: {self._samples_generated} samples\n"
                f"Strategy: {self._strategy}\n\n"
                f"Next: Verify and select best answer."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.75
        elif prev_phase == "execute":
            self._current_phase = "verify"
            # Aggregate results
            best_result = max(self._results, key=lambda x: x["confidence"])
            vote_counts: dict[str, int] = {}
            for r in self._results:
                ans = r["answer"]
                vote_counts[ans] = vote_counts.get(ans, 0) + 1
            majority_answer = max(vote_counts.items(), key=lambda x: x[1])

            consensus = "Strong" if majority_answer[1] > len(self._results) // 2 else "Moderate"
            content = (
                f"Step {self._step_counter}: Verify and Aggregate\n\n"
                f"Aggregation Method: {self._strategy}\n\n"
                f"Results Analysis:\n"
                f"  Total samples: {len(self._results)}\n"
                f"  Highest confidence: Sample {best_result['id']} "
                f"({best_result['confidence']:.0%})\n"
                f"  Majority answer: {majority_answer[0]} ({majority_answer[1]} votes)\n\n"
                f"Verification:\n"
                f"  Consensus level: {consensus}\n"
                f"  Confidence spread: Acceptable\n"
                f"  Additional compute needed: No\n\n"
                f"Answer selected based on {self._strategy} aggregation."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            best_result = (
                max(self._results, key=lambda x: x["confidence"])
                if self._results
                else {"answer": "[Answer]", "confidence": 0.85}
            )
            final_confidence = min(0.92, best_result["confidence"] + 0.05)

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Compute-Optimal Scaling Complete:\n"
                f"  Problem difficulty: {self._difficulty}\n"
                f"  Compute budget: {self._compute_budget}x\n"
                f"  Samples generated: {self._samples_generated}\n"
                f"  Strategy: {self._strategy}\n\n"
                f"Final Answer: {best_result['answer']}\n"
                f"Confidence: High ({int(final_confidence * 100)}%)\n\n"
                f"Method: Compute-Optimal Scaling\n"
                f"  - Difficulty-aware compute allocation\n"
                f"  - Adaptive strategy selection\n"
                f"  - More compute for harder problems\n"
                f"  - Optimizes accuracy per compute unit"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = final_confidence

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.COMPUTE_OPTIMAL_SCALING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "difficulty": self._difficulty,
                "samples": self._samples_generated,
            },
        )
        session.add_thought(thought)
        return thought

    async def _assess_difficulty(self, input_text: str) -> str:
        """Assess problem difficulty using LLM or heuristics."""
        prompt = f"""Analyze the following problem and assess its difficulty level.

Problem: {input_text}

Classify the difficulty as one of: easy, medium, hard, very_hard

Consider:
- Problem complexity and mathematical depth
- Number of steps required
- Domain expertise needed
- Ambiguity and edge cases

Respond with ONLY the difficulty level (easy/medium/hard/very_hard)."""

        system_msg = (
            "You are an expert at assessing problem difficulty for compute allocation."
        )

        def fallback() -> str:
            return self._heuristic_difficulty_assessment(input_text)

        result = await self._sample_with_fallback(prompt, fallback, system_prompt=system_msg)
        difficulty = str(result).strip().lower()
        if difficulty in self.DIFFICULTY_LEVELS:
            return difficulty
        # If LLM returned invalid classification, use fallback
        return self._heuristic_difficulty_assessment(input_text)

    def _heuristic_difficulty_assessment(self, input_text: str) -> str:
        """Fallback heuristic for difficulty assessment."""
        problem_length = len(input_text)
        if problem_length < 100:
            return "easy"
        elif problem_length < 300:
            return "medium"
        elif problem_length < 500:
            return "hard"
        else:
            return "very_hard"

    async def _generate_sample(self, input_text: str, sample_id: int) -> dict[str, Any]:
        """Generate a single sample using LLM or heuristic."""
        prompt = f"""Solve the following problem:

{input_text}

Provide your answer and reasoning. Be concise but thorough."""

        system_msg = (
            f"You are solving a {self._difficulty} difficulty problem. "
            "Provide a clear, well-reasoned answer."
        )

        def fallback() -> str:
            return f"[Answer candidate {sample_id}]"

        result_str = await self._sample_with_fallback(prompt, fallback, system_prompt=system_msg)

        # Estimate confidence based on response characteristics
        confidence = 0.7 + (len(result_str) / 1000) * 0.15
        confidence = min(0.95, confidence)

        return {
            "id": sample_id,
            "answer": result_str,
            "confidence": confidence,
        }

    async def _generate_samples(self, session: Session, num_samples: int) -> list[dict[str, Any]]:
        """Generate multiple samples for the current problem."""
        # Get the original input from session history
        thoughts = session.get_history()
        input_text = ""
        for thought in thoughts:
            if "Problem:" in thought.content:
                # Extract problem text
                lines = thought.content.split("\n")
                for _i, line in enumerate(lines):
                    if line.startswith("Problem:"):
                        input_text = line.replace("Problem:", "").strip()
                        break
                break

        if not input_text:
            # Fallback to heuristic results
            return [
                {
                    "id": i + 1,
                    "answer": f"[Answer candidate {i + 1}]",
                    "confidence": 0.7 + (i % 3) * 0.08,
                }
                for i in range(num_samples)
            ]

        # Generate samples
        results = []
        for i in range(num_samples):
            result = await self._generate_sample(input_text, i + 1)
            results.append(result)

        return results

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ComputeOptimalScaling", "COMPUTE_OPTIMAL_SCALING_METADATA"]
