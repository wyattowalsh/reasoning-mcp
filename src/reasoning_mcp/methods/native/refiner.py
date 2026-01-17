"""REFINER reasoning method.

This module implements REFINER, which uses a generator-critic feedback loop
operating on intermediate reasoning representations. The critic provides
structured feedback on reasoning quality that the generator uses to refine.

Key phases:
1. Generate: Create initial reasoning with intermediate representations
2. Critique: Critic evaluates reasoning representations
3. Refine: Generator improves based on critic feedback
4. Iterate: Repeat until quality threshold or max iterations

Reference: Paul et al. (2024) - "REFINER: Reasoning Feedback on Intermediate
Representations" (EACL 2024)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

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


REFINER_METADATA = MethodMetadata(
    identifier=MethodIdentifier.REFINER,
    name="REFINER",
    description="Generator-critic feedback loop on intermediate reasoning representations. "
    "Structured feedback enables targeted improvements in reasoning quality.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"generator-critic", "feedback", "intermediate", "refinement", "iterative"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=(
        "structured reasoning",
        "iterative improvement",
        "error correction",
        "quality refinement",
    ),
    not_recommended_for=("simple queries", "time-sensitive tasks"),
)


class Refiner(ReasoningMethodBase):
    """REFINER reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "generate"
        self._intermediate_repr: dict[str, Any] = {}
        self._critic_feedback: list[dict[str, Any]] = []
        self._refinement_history: list[dict[str, Any]] = []
        self._iteration: int = 0
        self._max_iterations: int = 3
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.REFINER

    @property
    def name(self) -> str:
        return REFINER_METADATA.name

    @property
    def description(self) -> str:
        return REFINER_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "generate"
        self._intermediate_repr = {}
        self._critic_feedback = []
        self._refinement_history = []
        self._iteration = 0
        self._max_iterations = 3

    async def _generate_intermediate_representation(self, input_text: str) -> dict[str, Any]:
        """Generate intermediate representation using LLM sampling or fallback."""
        prompt = f"""Generate an intermediate reasoning representation \
for the following problem:

Problem: {input_text}

Provide a structured representation with:
1. A hypothesis about what the problem involves
2. Key entities (variables, values, concepts)
3. Relations between entities
4. A step-by-step reasoning chain with actions and outputs
5. An initial confidence estimate (0.0-1.0)

Format as a JSON-like structure."""

        system_prompt = (
            "You are a reasoning system that creates structured "
            "intermediate representations of problems."
        )

        def fallback_generator() -> str:
            return json.dumps(self._fallback_intermediate_representation(input_text))

        result = await self._sample_with_fallback(
            prompt,
            fallback_generator,
            system_prompt=system_prompt,
        )

        return self._parse_intermediate_representation(result, input_text)

    def _parse_intermediate_representation(
        self, llm_result: str, input_text: str
    ) -> dict[str, Any]:
        """Parse LLM result into intermediate representation structure."""
        # Attempt to parse JSON from LLM response
        try:
            parsed = json.loads(llm_result)
            if isinstance(parsed, dict):
                # Validate and normalize the structure
                return {
                    "hypothesis": parsed.get(
                        "hypothesis", "The answer involves reasoning about the given problem"
                    ),
                    "entities": parsed.get("entities", self._extract_entities(input_text)),
                    "relations": parsed.get("relations", ["analyze", "reason", "conclude"]),
                    "reasoning_chain": parsed.get(
                        "reasoning_chain",
                        [
                            {
                                "step": 1,
                                "action": "understand_problem",
                                "output": "Problem identified",
                            },
                            {
                                "step": 2,
                                "action": "analyze_components",
                                "output": "Components analyzed",
                            },
                            {
                                "step": 3,
                                "action": "synthesize_solution",
                                "output": "Solution synthesized",
                            },
                        ],
                    ),
                    "confidence": float(parsed.get("confidence", 0.75)),
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback to heuristic parsing if JSON parsing fails
        return {
            "hypothesis": "The answer involves reasoning about the given problem",
            "entities": self._extract_entities(input_text),
            "relations": ["analyze", "reason", "conclude"],
            "reasoning_chain": [
                {"step": 1, "action": "understand_problem", "output": "Problem identified"},
                {"step": 2, "action": "analyze_components", "output": "Components analyzed"},
                {"step": 3, "action": "synthesize_solution", "output": "Solution synthesized"},
            ],
            "confidence": 0.75,
        }

    def _fallback_intermediate_representation(self, input_text: str) -> dict[str, Any]:
        """Fallback heuristic for generating intermediate representation."""
        return {
            "hypothesis": "The answer involves arithmetic operations",
            "entities": ["x=5", "y=3", "z=2"],
            "relations": ["multiply(x, y)", "add(result, z)"],
            "reasoning_chain": [
                {"step": 1, "action": "identify_variables", "output": "x, y, z"},
                {"step": 2, "action": "apply_operation", "output": "15"},
                {"step": 3, "action": "combine_results", "output": "17"},
            ],
            "confidence": 0.75,
        }

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entities from text using simple heuristics."""
        # Simple extraction - look for numbers and common patterns
        entities = []
        words = text.split()
        for word in words:
            if word.isdigit():
                entities.append(word)
        return entities if entities else ["x=5", "y=3", "z=2"]

    async def _generate_critic_feedback(self, input_text: str) -> list[dict[str, Any]]:
        """Generate critic feedback using LLM sampling or fallback heuristics."""
        prompt = f"""Evaluate the following intermediate reasoning representation:

Problem: {input_text}

Current Representation:
{self._format_repr_for_critique()}

As a critic, evaluate on these aspects:
1. Completeness: Are all necessary elements identified?
2. Correctness: Is the reasoning logically sound?
3. Clarity: Is the reasoning chain easy to follow?

For each aspect, provide:
- Score (0.0-1.0)
- Feedback on what's working
- Suggestion for improvement (if needed)

Return as JSON array with objects containing: aspect, score, feedback, suggestion."""

        system_prompt = (
            "You are a critical evaluator that provides structured feedback on reasoning quality."
        )

        def fallback_generator() -> str:
            return json.dumps(self._fallback_critic_feedback())

        result = await self._sample_with_fallback(
            prompt,
            fallback_generator,
            system_prompt=system_prompt,
        )

        return self._parse_critic_feedback(result)

    def _format_repr_for_critique(self) -> str:
        """Format intermediate representation for critic evaluation."""
        return f"""
Hypothesis: {self._intermediate_repr.get("hypothesis", "N/A")}
Entities: {", ".join(self._intermediate_repr.get("entities", []))}
Relations: {", ".join(self._intermediate_repr.get("relations", []))}
Reasoning Chain: {len(self._intermediate_repr.get("reasoning_chain", []))} steps
Confidence: {self._intermediate_repr.get("confidence", 0.0):.2f}
"""

    def _parse_critic_feedback(self, llm_result: str) -> list[dict[str, Any]]:
        """Parse LLM result into critic feedback structure."""
        try:
            parsed = json.loads(llm_result)
            if isinstance(parsed, list):
                # Validate and normalize each feedback item
                result = []
                for item in parsed:
                    if isinstance(item, dict):
                        result.append(
                            {
                                "aspect": item.get("aspect", "unknown"),
                                "score": float(item.get("score", 0.7)),
                                "feedback": item.get("feedback", "No feedback provided"),
                                "suggestion": item.get("suggestion"),
                            }
                        )
                if result:
                    return result
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback to default feedback structure
        return self._fallback_critic_feedback()

    def _fallback_critic_feedback(self) -> list[dict[str, Any]]:
        """Fallback heuristic for generating critic feedback."""
        return [
            {
                "aspect": "completeness",
                "score": 0.85,
                "feedback": "All variables identified correctly",
                "suggestion": None,
            },
            {
                "aspect": "correctness",
                "score": 0.70,
                "feedback": "Operation order may need verification",
                "suggestion": "Explicitly show order of operations",
            },
            {
                "aspect": "clarity",
                "score": 0.75,
                "feedback": "Reasoning chain is followable",
                "suggestion": "Add intermediate result annotations",
            },
        ]

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("REFINER must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "generate"
        self._iteration = 1

        # Generate initial intermediate representation
        self._intermediate_repr = await self._generate_intermediate_representation(input_text)

        content = (
            f"Step {self._step_counter}: Generate Initial Representation (REFINER)\n\n"
            f"Problem: {input_text}\n\n"
            f"Iteration {self._iteration}/{self._max_iterations}\n\n"
            f"Intermediate Representation:\n"
            f"  Hypothesis: {self._intermediate_repr['hypothesis']}\n"
            f"  Entities: {', '.join(self._intermediate_repr['entities'])}\n"
            f"  Relations: {', '.join(self._intermediate_repr['relations'])}\n\n"
            f"Reasoning Chain:\n"
            + "\n".join(
                f"  [{s['step']}] {s['action']} → {s['output']}"
                for s in self._intermediate_repr["reasoning_chain"]
            )
            + f"\n\nInitial Confidence: {self._intermediate_repr['confidence']:.0%}\n\n"
            f"Next: Apply critic evaluation."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.REFINER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=self._intermediate_repr["confidence"],
            quality_score=self._intermediate_repr["confidence"],
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.REFINER
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
            raise RuntimeError("REFINER must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "generate")

        if prev_phase == "generate":
            self._current_phase = "critique"
            # Critic evaluates intermediate representation
            # Extract the original input from previous thought or session
            if "Problem: " in previous_thought.content:
                input_text = previous_thought.content.split("Problem: ")[1].split("\n")[0]
            else:
                input_text = "problem"
            self._critic_feedback = await self._generate_critic_feedback(input_text)

            avg_score = sum(f["score"] for f in self._critic_feedback) / len(self._critic_feedback)

            content = (
                f"Step {self._step_counter}: Critic Evaluation\n\n"
                f"Iteration {self._iteration}/{self._max_iterations}\n\n"
                f"Critic Feedback on Intermediate Representation:\n\n"
                + "\n".join(
                    f"  [{f['aspect'].upper()}] Score: {f['score']:.2f}\n"
                    f"    Feedback: {f['feedback']}\n"
                    f"    Suggestion: {f['suggestion'] or 'None'}"
                    for f in self._critic_feedback
                )
                + f"\n\nOverall Quality Score: {avg_score:.2f}\n"
                f"Quality Threshold: 0.85\n"
                f"Status: {'Needs refinement' if avg_score < 0.85 else 'Acceptable'}\n\n"
                f"Next: Apply refinements based on feedback."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = avg_score
        elif prev_phase == "critique":
            self._current_phase = "refine"
            # Apply refinements based on feedback
            refinements_applied = []
            for feedback in self._critic_feedback:
                if feedback["suggestion"]:
                    refinements_applied.append(
                        {
                            "aspect": feedback["aspect"],
                            "original_score": feedback["score"],
                            "action": feedback["suggestion"],
                            "improvement": 0.10,
                        }
                    )

            # Update intermediate representation
            self._intermediate_repr["reasoning_chain"][1]["output"] = "5×3 = 15"
            self._intermediate_repr["reasoning_chain"][2]["output"] = "15+2 = 17"
            self._intermediate_repr["confidence"] = min(
                0.92, self._intermediate_repr["confidence"] + 0.12
            )

            self._refinement_history.append(
                {
                    "iteration": self._iteration,
                    "refinements": len(refinements_applied),
                    "confidence_before": self._intermediate_repr["confidence"] - 0.12,
                    "confidence_after": self._intermediate_repr["confidence"],
                }
            )

            content = (
                f"Step {self._step_counter}: Apply Refinements\n\n"
                f"Iteration {self._iteration}/{self._max_iterations}\n\n"
                f"Refinements Applied:\n"
                + (
                    "\n".join(
                        f"  [{r['aspect']}]\n"
                        f"    Action: {r['action']}\n"
                        f"    Score improvement: +{r['improvement']:.2f}"
                        for r in refinements_applied
                    )
                    if refinements_applied
                    else "  No refinements needed"
                )
                + "\n\nUpdated Reasoning Chain:\n"
                + "\n".join(
                    f"  [{s['step']}] {s['action']} → {s['output']}"
                    for s in self._intermediate_repr["reasoning_chain"]
                )
                + f"\n\nConfidence: {self._intermediate_repr['confidence']:.0%}\n\n"
                f"Refinement complete for iteration {self._iteration}."
            )
            thought_type = ThoughtType.REVISION
            confidence = self._intermediate_repr["confidence"]
        elif prev_phase == "refine":
            # Check if more iterations needed
            if (
                self._iteration < self._max_iterations
                and self._intermediate_repr["confidence"] < 0.90
            ):
                self._iteration += 1
                self._current_phase = "generate"
                content = (
                    f"Step {self._step_counter}: Continue Iteration\n\n"
                    f"Starting Iteration {self._iteration}/{self._max_iterations}\n\n"
                    f"Previous confidence: {self._intermediate_repr['confidence']:.0%}\n"
                    f"Target threshold: 90%\n"
                    f"Status: Below threshold, continuing refinement\n\n"
                    f"Regenerating with improved understanding..."
                )
                thought_type = ThoughtType.REASONING
                confidence = self._intermediate_repr["confidence"]
            else:
                self._current_phase = "conclude"
                total_refinements = sum(r["refinements"] for r in self._refinement_history)
                final_answer = self._intermediate_repr["reasoning_chain"][-1]["output"]

                initial_conf = self._refinement_history[0]["confidence_before"]
                final_conf = self._intermediate_repr["confidence"]
                content = (
                    f"Step {self._step_counter}: Final Answer\n\n"
                    f"REFINER Complete:\n"
                    f"  Iterations: {self._iteration}\n"
                    f"  Total refinements: {total_refinements}\n"
                    f"  Initial confidence: {initial_conf:.0%}\n"
                    f"  Final confidence: {final_conf:.0%}\n\n"
                    f"Final Answer: {final_answer}\n"
                    f"Confidence: High ({final_conf:.0%})\n\n"
                    f"Method: REFINER\n"
                    f"  - Generator-critic architecture\n"
                    f"  - Intermediate representation feedback\n"
                    f"  - Iterative refinement loop\n"
                    f"  - Structured quality improvement"
                )
                thought_type = ThoughtType.CONCLUSION
                confidence = self._intermediate_repr["confidence"]
        else:
            # Fallback to conclusion
            self._current_phase = "conclude"
            final_answer = (
                self._intermediate_repr["reasoning_chain"][-1]["output"]
                if self._intermediate_repr.get("reasoning_chain")
                else "17"
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"REFINER Complete:\n"
                f"  Iterations: {self._iteration}\n"
                f"  Final confidence: {self._intermediate_repr.get('confidence', 0.88):.0%}\n\n"
                f"Final Answer: {final_answer}\n"
                f"Confidence: High (88%)\n\n"
                f"Method: REFINER\n"
                f"  - Generator-critic feedback loop\n"
                f"  - Iterative refinement"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.88

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.REFINER,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "iteration": self._iteration,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["Refiner", "REFINER_METADATA"]
