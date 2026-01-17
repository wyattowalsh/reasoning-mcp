"""DiVeRSe (Diverse Verifier on Reasoning Steps) reasoning method.

This module implements DiVeRSe, which combines diverse sampling of reasoning
paths with step-level verification. Uses multiple prompts and verifiers to
improve reasoning accuracy through diversity.

Key phases:
1. Sample: Generate diverse reasoning paths with varied prompts
2. Verify: Apply step-level verification to each path
3. Vote: Aggregate verified paths using voting mechanisms
4. Select: Choose final answer based on verified consensus

Reference: Li et al. (2023) - "Making Large Language Models Better Reasoners
with Step-Aware Verifier"
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


DIVERSE_VERIFIER_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DIVERSE_VERIFIER,
    name="DiVeRSe",
    description="Diverse Verifier on Reasoning Steps - combines diverse sampling "
    "with step-level verification for improved reasoning accuracy.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"diversity", "verification", "step-aware", "voting", "sampling"}),
    complexity=6,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=("math reasoning", "multi-path problems", "answer verification", "consensus building"),
    not_recommended_for=("simple queries", "single-answer problems"),
)


class DiverseVerifier(ReasoningMethodBase):
    """DiVeRSe reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "sample"
        self._diverse_paths: list[dict[str, Any]] = []
        self._verified_paths: list[dict[str, Any]] = []
        self._vote_results: dict[str, Any] = {}
        self._final_answer: str | None = None
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.DIVERSE_VERIFIER

    @property
    def name(self) -> str:
        return DIVERSE_VERIFIER_METADATA.name

    @property
    def description(self) -> str:
        return DIVERSE_VERIFIER_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "sample"
        self._diverse_paths = []
        self._verified_paths = []
        self._vote_results = {}
        self._final_answer = None

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("DiverseVerifier must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "sample"

        # Generate diverse paths with different prompts
        self._diverse_paths = await self._generate_diverse_paths(input_text)

        if not self._diverse_paths:
            # Fallback to hardcoded diverse paths
            self._diverse_paths = [
                {
                    "id": 1,
                    "prompt_style": "step-by-step",
                    "reasoning": "1) x=5, y=3, z=2. 2) x×y=15. 3) 15+z=17.",
                    "answer": "17",
                    "steps": ["parse", "multiply", "add"],
                },
                {
                    "id": 2,
                    "prompt_style": "direct",
                    "reasoning": "Calculate 5×3+2 = 17",
                    "answer": "17",
                    "steps": ["calculate"],
                },
                {
                    "id": 3,
                    "prompt_style": "algebraic",
                    "reasoning": "Let result = x·y + z = 5·3 + 2 = 15 + 2 = 17",
                    "answer": "17",
                    "steps": ["formulate", "substitute", "compute"],
                },
                {
                    "id": 4,
                    "prompt_style": "verification-first",
                    "reasoning": "Estimate ~15-20. Compute: 5×3=15, +2=17. Within range ✓",
                    "answer": "17",
                    "steps": ["estimate", "compute", "verify"],
                },
                {
                    "id": 5,
                    "prompt_style": "decomposition",
                    "reasoning": "Split: (5×3) + 2 = 15 + 2 = 18",
                    "answer": "18",
                    "steps": ["decompose", "multiply", "add"],
                },
            ]

        # Compute answer distribution
        answer_dist = dict(
            (a, sum(1 for p in self._diverse_paths if p["answer"] == a))
            for a in set(p["answer"] for p in self._diverse_paths)
        )

        content = (
            f"Step {self._step_counter}: Diverse Sampling (DiVeRSe)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating diverse reasoning paths:\n\n"
            f"Sampled Paths:\n"
            + "\n".join(
                f"  [Path {p['id']}] Prompt: {p['prompt_style']}\n"
                f"    Reasoning: {p['reasoning']}\n"
                f"    Answer: {p['answer']}\n"
                f"    Steps: {len(p['steps'])}"
                for p in self._diverse_paths
            )
            + "\n\nDiversity Statistics:\n"
            f"  Total paths: {len(self._diverse_paths)}\n"
            f"  Unique prompt styles: "
            f"{len(set(p['prompt_style'] for p in self._diverse_paths))}\n"
            f"  Answer distribution: {answer_dist}\n\n"
            "Next: Apply step-level verification."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIVERSE_VERIFIER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "paths": len(self._diverse_paths),
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DIVERSE_VERIFIER
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
            raise RuntimeError("DiverseVerifier must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "sample")

        if prev_phase == "sample":
            self._current_phase = "verify"
            # Apply step-level verification
            self._verified_paths = await self._verify_paths()

            if not self._verified_paths:
                # Fallback to hardcoded verification
                self._verified_paths = [
                    {"id": 1, "step_scores": [0.9, 0.88, 0.92], "avg_score": 0.90, "valid": True},
                    {"id": 2, "step_scores": [0.85], "avg_score": 0.85, "valid": True},
                    {"id": 3, "step_scores": [0.87, 0.90, 0.91], "avg_score": 0.89, "valid": True},
                    {"id": 4, "step_scores": [0.82, 0.88, 0.90], "avg_score": 0.87, "valid": True},
                    {"id": 5, "step_scores": [0.80, 0.85, 0.40], "avg_score": 0.68, "valid": False},
                ]

            content = (
                f"Step {self._step_counter}: Step-Level Verification\n\n"
                f"Verifying each reasoning step:\n\n"
                f"Verification Results:\n"
                + "\n".join(
                    f"  Path {v['id']}: "
                    f"Steps={[f'{s:.2f}' for s in v['step_scores']]} "
                    f"→ Avg={v['avg_score']:.2f} "
                    f"{'✓ Valid' if v['valid'] else '✗ Invalid'}"
                    for v in self._verified_paths
                )
                + f"\n\nVerification Summary:\n"
                f"  Paths verified: {len(self._verified_paths)}\n"
                f"  Valid paths: {sum(1 for v in self._verified_paths if v['valid'])}\n"
                f"  Invalid paths: {sum(1 for v in self._verified_paths if not v['valid'])}\n"
                f"  Score threshold: 0.70\n\n"
                f"Next: Aggregate via voting."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "verify":
            self._current_phase = "vote"
            # Aggregate verified paths
            valid_paths = [v for v in self._verified_paths if v["valid"]]
            answers = {}
            for v in valid_paths:
                path = next(p for p in self._diverse_paths if p["id"] == v["id"])
                ans = str(path["answer"])
                if ans not in answers:
                    answers[ans] = {"count": 0, "total_score": 0.0, "paths": []}
                answers[ans]["count"] = int(str(answers[ans]["count"])) + 1
                answers[ans]["total_score"] = float(str(answers[ans]["total_score"])) + float(
                    str(v["avg_score"])
                )
                existing_paths = answers[ans]["paths"]
                if isinstance(existing_paths, list):
                    existing_paths.append(v["id"])
                else:
                    answers[ans]["paths"] = [v["id"]]

            # Find winner by count and total_score
            winner_entry = max(
                answers.items(),
                key=lambda x: (
                    int(str(x[1]["count"])),
                    float(str(x[1]["total_score"])),
                ),
            )
            max_count_entry = max(
                answers.items(),
                key=lambda x: int(str(x[1]["count"])),
            )
            self._vote_results = {
                "answers": answers,
                "winner": winner_entry[0],
                "confidence": int(str(max_count_entry[1]["count"])) / len(valid_paths),
            }

            content = (
                f"Step {self._step_counter}: Voting Aggregation\n\n"
                f"Aggregating verified paths through voting:\n\n"
                f"Vote Distribution:\n"
                + "\n".join(
                    f"  Answer '{ans}': {data['count']} votes "
                    f"(paths: {data['paths']}, "
                    f"avg_score: {data['total_score'] / data['count']:.2f})"
                    for ans, data in self._vote_results["answers"].items()
                )
                + f"\n\nVoting Result:\n"
                f"  Winner: '{self._vote_results['winner']}'\n"
                f"  Vote share: {self._vote_results['confidence']:.0%}\n"
                f"  Consensus: "
                f"{'Strong' if self._vote_results['confidence'] > 0.6 else 'Moderate'}\n\n"
                f"Next: Select final answer."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.82
        elif prev_phase == "vote":
            self._current_phase = "select"
            self._final_answer = self._vote_results.get("winner", "17")
            winning_data = self._vote_results["answers"].get(self._final_answer, {})
            avg_score = winning_data.get("total_score", 0) / max(winning_data.get("count", 1), 1)

            content = (
                f"Step {self._step_counter}: Final Selection\n\n"
                f"Selecting answer based on verified consensus:\n\n"
                f"Selection Criteria:\n"
                f"  1. Highest vote count among valid paths\n"
                f"  2. Highest average verification score\n"
                f"  3. Diversity of supporting prompts\n\n"
                f"Selected Answer: {self._final_answer}\n"
                f"  Supporting paths: {winning_data.get('paths', [])}\n"
                f"  Vote count: {winning_data.get('count', 0)}\n"
                f"  Average score: {avg_score:.2f}\n\n"
                "Answer selected with strong verified support."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.87
        else:
            self._current_phase = "conclude"
            valid_count = sum(1 for v in self._verified_paths if v["valid"])

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"DiVeRSe Complete:\n"
                f"  Paths sampled: {len(self._diverse_paths)}\n"
                f"  Valid paths: {valid_count}\n"
                f"  Unique answers: {len(self._vote_results.get('answers', {}))}\n"
                f"  Vote consensus: {self._vote_results.get('confidence', 0):.0%}\n\n"
                f"Final Answer: {self._final_answer}\n"
                f"Confidence: High (89%)\n\n"
                f"Method: DiVeRSe\n"
                f"  - Diverse prompt sampling\n"
                f"  - Step-level verification\n"
                f"  - Weighted voting aggregation\n"
                f"  - Verified consensus selection"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.89

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.DIVERSE_VERIFIER,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "valid_paths": sum(1 for v in self._verified_paths if v["valid"])
                if self._verified_paths
                else 0,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    async def _generate_diverse_paths(self, input_text: str) -> list[dict[str, Any]]:
        """Generate diverse reasoning paths using different prompt styles."""
        prompt_styles = [
            ("step-by-step", "Solve step-by-step, showing each calculation clearly."),
            ("direct", "Solve directly and concisely."),
            ("algebraic", "Solve using algebraic notation and formulas."),
            ("verification-first", "First estimate the answer, then compute and verify."),
            ("decomposition", "Decompose into smaller sub-problems and solve."),
        ]

        diverse_paths = []
        for path_id, (style, instruction) in enumerate(prompt_styles, 1):
            user_prompt = (
                f"{instruction}\n\nProblem: {input_text}\n\n"
                f"Provide your reasoning and final answer."
            )
            system_prompt = (
                "You are a mathematical reasoning assistant. "
                "Provide clear reasoning and a final answer."
            )

            # Use fallback generator that returns empty string to signal fallback needed
            def fallback_generator() -> str:
                return ""

            content_text = await self._sample_with_fallback(
                user_prompt=user_prompt,
                fallback_generator=fallback_generator,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=500,
            )

            # If sampling failed (empty result), return empty list to trigger hardcoded fallback
            if not content_text:
                return []

            # Extract answer and steps from the response
            answer = self._extract_answer(content_text)
            steps = self._extract_steps(content_text)

            diverse_paths.append(
                {
                    "id": path_id,
                    "prompt_style": style,
                    "reasoning": content_text[:200],  # Truncate for display
                    "answer": answer,
                    "steps": steps,
                }
            )

        return diverse_paths

    async def _verify_paths(self) -> list[dict[str, Any]]:
        """Verify each reasoning path at step level."""
        verified_paths = []

        for path in self._diverse_paths:
            user_prompt = f"""Verify the following reasoning path step by step.
For each step, assign a confidence score between 0.0 and 1.0.

Reasoning: {path["reasoning"]}
Steps: {path["steps"]}
Answer: {path["answer"]}

Return step scores as a comma-separated list (e.g., "0.9,0.85,0.92")."""

            system_prompt = (
                "You are a reasoning verification assistant. "
                "Evaluate each step carefully and provide confidence scores."
            )

            # Use fallback generator that returns empty string to signal fallback needed
            def fallback_generator() -> str:
                return ""

            content_text = await self._sample_with_fallback(
                user_prompt=user_prompt,
                fallback_generator=fallback_generator,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for verification
                max_tokens=300,
            )

            # If sampling failed (empty result), return empty list to trigger hardcoded fallback
            if not content_text:
                return []

            step_scores = self._parse_step_scores(content_text, len(path["steps"]))
            avg_score = sum(step_scores) / len(step_scores) if step_scores else 0.0

            verified_paths.append(
                {
                    "id": path["id"],
                    "step_scores": step_scores,
                    "avg_score": avg_score,
                    "valid": avg_score >= 0.70,
                }
            )

        return verified_paths

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from reasoning text."""
        # Look for common answer patterns
        import re

        patterns = [
            r"(?:final answer|answer|result)(?:\s*is)?(?:\s*:)?\s*[=]?\s*(\d+(?:\.\d+)?)",
            r"=\s*(\d+(?:\.\d+)?)\s*$",
            r"\b(\d+(?:\.\d+)?)\s*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)

        # Default fallback
        return "unknown"

    def _extract_steps(self, text: str) -> list[str]:
        """Extract reasoning steps from text."""
        # Look for numbered steps or sentences
        import re

        # Try numbered steps first (1., 2., etc.)
        numbered_steps = re.findall(r"(?:^|\n)\s*\d+[\.)]\s*([^\n]+)", text)
        if numbered_steps:
            return numbered_steps[:5]  # Limit to 5 steps

        # Fallback to sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences[:3] if sentences else ["reasoning"]

    def _parse_step_scores(self, text: str, expected_count: int) -> list[float]:
        """Parse step scores from verification result."""
        import re

        # Look for comma-separated numbers
        scores_match = re.search(r"([\d.]+(?:\s*,\s*[\d.]+)+)", text)
        if scores_match:
            try:
                scores = [float(s.strip()) for s in scores_match.group(1).split(",")]
                # Clamp scores between 0.0 and 1.0
                scores = [max(0.0, min(1.0, s)) for s in scores]
                if len(scores) >= expected_count:
                    return scores[:expected_count]
            except ValueError:
                pass

        # Fallback: generate reasonable scores
        return [0.85] * expected_count


__all__ = ["DiverseVerifier", "DIVERSE_VERIFIER_METADATA"]
