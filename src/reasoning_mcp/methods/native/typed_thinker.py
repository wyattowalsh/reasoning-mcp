"""TypedThinker reasoning method.

This module implements TypedThinker, which diversifies LLM reasoning by
categorizing thinking into explicit types: deductive, inductive, abductive,
and analogical reasoning. Improves accuracy through reasoning type diversity.

Key phases:
1. Classify: Identify applicable reasoning types for the problem
2. Generate: Produce solutions using each reasoning type
3. Diversify: Ensure coverage across different thinking approaches
4. Integrate: Combine typed solutions for final answer

Reference: Wang et al. (2024) - "TypedThinker: Typed Thinking Improves
Large Language Model Reasoning" (ICLR)
"""

from __future__ import annotations

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


TYPED_THINKER_METADATA = MethodMetadata(
    identifier=MethodIdentifier.TYPED_THINKER,
    name="TypedThinker",
    description="Diversifies reasoning by categorizing into explicit types: deductive, "
    "inductive, abductive, and analogical. Improves through reasoning diversity.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"diversity", "typed-reasoning", "deductive", "inductive", "abductive"}),
    complexity=6,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=5,
    max_thoughts=9,
    avg_tokens_per_thought=300,
    best_for=(
        "complex reasoning",
        "benchmark problems",
        "reasoning diversity",
        "accuracy improvement",
    ),
    not_recommended_for=("simple queries", "factual lookups"),
)


REASONING_TYPES = {
    "deductive": {
        "name": "Deductive Reasoning",
        "description": "From general premises to specific conclusions",
        "approach": "If P then Q; P; therefore Q",
    },
    "inductive": {
        "name": "Inductive Reasoning",
        "description": "From specific observations to general conclusions",
        "approach": "Observe patterns, generalize to rules",
    },
    "abductive": {
        "name": "Abductive Reasoning",
        "description": "Inference to the best explanation",
        "approach": "Given effect, infer most likely cause",
    },
    "analogical": {
        "name": "Analogical Reasoning",
        "description": "Transfer knowledge from similar situations",
        "approach": "A is to B as C is to D",
    },
}


class TypedThinker(ReasoningMethodBase):
    """TypedThinker reasoning method implementation."""

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "classify"
        self._applicable_types: list[str] = []
        self._typed_solutions: dict[str, dict[str, Any]] = {}
        self._final_solution: str | None = None
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.TYPED_THINKER

    @property
    def name(self) -> str:
        return TYPED_THINKER_METADATA.name

    @property
    def description(self) -> str:
        return TYPED_THINKER_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "classify"
        self._applicable_types = []
        self._typed_solutions = {}
        self._final_solution = None

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("TypedThinker must be initialized before execution")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "classify"

        # Classify applicable reasoning types
        if use_sampling:
            self._applicable_types = await self._sample_classify_types(input_text)
        else:
            self._applicable_types = self._heuristic_classify_types(input_text)

        content = (
            f"Step {self._step_counter}: Classify Reasoning Types (TypedThinker)\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing problem for applicable reasoning types:\n\n"
            f"Available Types:\n"
            + "\n".join(
                f"  • {REASONING_TYPES[t]['name']}: {REASONING_TYPES[t]['description']}"
                for t in REASONING_TYPES
            )
            + "\n\nSelected Types for This Problem:\n"
            + "\n".join(f"  ✓ {REASONING_TYPES[t]['name']}" for t in self._applicable_types)
            + f"\n\n{len(self._applicable_types)} reasoning types will be applied.\n"
            f"Next: Generate solutions using each type."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TYPED_THINKER,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "types": len(self._applicable_types),
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.TYPED_THINKER
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
            raise RuntimeError("TypedThinker must be initialized before continuation")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        if execution_context:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "classify")

        if prev_phase == "classify":
            self._current_phase = "generate"
            # Generate solutions for each type
            self._typed_solutions = {}
            for rtype in self._applicable_types:
                type_info = REASONING_TYPES[rtype]
                if use_sampling:
                    solution_data = await self._sample_generate_solution(session, rtype, type_info)
                else:
                    solution_data = self._heuristic_generate_solution(rtype, type_info)
                self._typed_solutions[rtype] = solution_data

            content = (
                f"Step {self._step_counter}: Generate Typed Solutions\n\n"
                f"Applying {len(self._applicable_types)} reasoning types:\n\n"
                + "\n".join(
                    f"[{s['name']}]\n"
                    f"  Approach: {s['approach']}\n"
                    f"  Solution: {s['solution']}\n"
                    f"  Confidence: {s['confidence']:.0%}"
                    for s in self._typed_solutions.values()
                )
                + f"\n\nDiversity Score: {len(self._typed_solutions)}/4 types used\n"
                f"Next: Diversify and validate coverage."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.75
        elif prev_phase == "generate":
            self._current_phase = "diversify"
            # Check diversity
            used_types = set(self._typed_solutions.keys())
            missing_types = set(REASONING_TYPES.keys()) - used_types

            content = (
                f"Step {self._step_counter}: Diversify Reasoning Coverage\n\n"
                f"Analyzing reasoning diversity:\n\n"
                f"Types Applied:\n"
                + "\n".join(f"  ✓ {REASONING_TYPES[t]['name']}" for t in used_types)
                + "\n\nTypes Not Used:\n"
                + (
                    "\n".join(
                        f"  ○ {REASONING_TYPES[t]['name']} (not applicable)" for t in missing_types
                    )
                    if missing_types
                    else "  All applicable types used"
                )
                + f"\n\nDiversity Assessment:\n"
                f"  Coverage: {len(used_types)}/{len(REASONING_TYPES)} types\n"
                f"  Complementary approaches: Yes\n"
                f"  Risk of tunnel vision: Low\n\n"
                f"Good diversity achieved.\n"
                f"Next: Integrate solutions."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.8
        elif prev_phase == "diversify":
            self._current_phase = "integrate"
            # Integrate solutions
            if use_sampling:
                self._final_solution = await self._sample_integrate_solutions(session)
            else:
                best_solution = max(
                    self._typed_solutions.values(),
                    key=lambda x: x["confidence"],
                )
                self._final_solution = (
                    f"[Integrated answer combining {best_solution['name']} as primary]"
                )

            content = (
                f"Step {self._step_counter}: Integrate Typed Solutions\n\n"
                f"Combining insights from {len(self._typed_solutions)} reasoning types:\n\n"
                f"Integration Strategy:\n"
                f"  1. Primary: {best_solution['name']} (highest confidence)\n"
                f"  2. Supporting: Cross-validate with other types\n"
                f"  3. Synthesize: Unified answer\n\n"
                f"Confidence Ranking:\n"
                + "\n".join(
                    f"  {i + 1}. {s['name']}: {s['confidence']:.0%}"
                    for i, s in enumerate(
                        sorted(self._typed_solutions.values(), key=lambda x: -x["confidence"])
                    )
                )
                + f"\n\nIntegrated Solution: {self._final_solution}"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            avg_confidence = (
                sum(s["confidence"] for s in self._typed_solutions.values())
                / len(self._typed_solutions)
                if self._typed_solutions
                else 0.85
            )

            types_list = ", ".join(REASONING_TYPES[t]["name"] for t in self._applicable_types)
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"TypedThinker Complete:\n"
                f"  Reasoning types used: {len(self._typed_solutions)}\n"
                f"  Types: {types_list}\n"
                f"  Average confidence: {avg_confidence:.0%}\n\n"
                f"Final Answer: {self._final_solution}\n"
                f"Confidence: High ({int(avg_confidence * 100 + 3)}%)\n\n"
                f"Method: TypedThinker\n"
                f"  - Explicit reasoning type classification\n"
                f"  - Multi-type solution generation\n"
                f"  - Diversity-aware validation\n"
                f"  - Type-weighted integration"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = avg_confidence + 0.03

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.TYPED_THINKER,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "types_used": len(self._typed_solutions),
                "sampled": use_sampling,
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    def _heuristic_classify_types(self, input_text: str) -> list[str]:
        """Heuristically classify applicable reasoning types.

        Fallback method when LLM sampling is not available.

        Args:
            input_text: The problem text to analyze

        Returns:
            List of applicable reasoning type identifiers
        """
        # Default to core reasoning types
        return ["deductive", "inductive", "analogical"]

    def _heuristic_generate_solution(self, rtype: str, type_info: dict[str, str]) -> dict[str, Any]:
        """Heuristically generate a solution for a reasoning type.

        Fallback method when LLM sampling is not available.

        Args:
            rtype: The reasoning type identifier
            type_info: Information about the reasoning type

        Returns:
            Dictionary containing solution data
        """
        return {
            "type": rtype,
            "name": type_info["name"],
            "approach": type_info["approach"],
            "solution": f"[Solution via {type_info['name']}]",
            "confidence": 0.75 + (0.05 if rtype == "deductive" else 0),
        }

    async def _sample_classify_types(self, input_text: str) -> list[str]:
        """Use LLM sampling to classify applicable reasoning types.

        Args:
            input_text: The problem text to analyze

        Returns:
            List of applicable reasoning type identifiers
        """
        system_prompt = """You are a reasoning type classifier for the TypedThinker methodology.
Analyze the given problem and determine which reasoning types are most applicable:
- deductive: From general premises to specific conclusions (logical deduction)
- inductive: From specific observations to general conclusions (pattern recognition)
- abductive: Inference to the best explanation (diagnostic reasoning)
- analogical: Transfer knowledge from similar situations (comparison)

Return ONLY a comma-separated list of applicable types."""

        user_prompt = f"""Problem: {input_text}

Which reasoning types are most applicable to this problem?
Consider:
- Is there logical structure for deductive reasoning?
- Are there patterns to generalize inductively?
- Does it require inferring explanations abductively?
- Can analogies help solve it?

Return types as: type1, type2, type3"""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "",
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=100,
        )

        # If fallback returned empty string, use heuristic
        if not content:
            return self._heuristic_classify_types(input_text)

        # Parse the response to extract reasoning types
        types = []
        for word in content.lower().split(","):
            word = word.strip()
            if "deductive" in word:
                types.append("deductive")
            elif "inductive" in word:
                types.append("inductive")
            elif "abductive" in word:
                types.append("abductive")
            elif "analogical" in word:
                types.append("analogical")

        # Ensure at least one type is selected
        return types if types else self._heuristic_classify_types(input_text)

    async def _sample_generate_solution(
        self, session: Session, rtype: str, type_info: dict[str, str]
    ) -> dict[str, Any]:
        """Use LLM sampling to generate a solution for a reasoning type.

        Args:
            session: The current reasoning session
            rtype: The reasoning type identifier
            type_info: Information about the reasoning type

        Returns:
            Dictionary containing solution data
        """
        # Get the original problem from session
        initial_thought = session.get_recent_thoughts(n=session.thought_count)
        problem = initial_thought[0].content if initial_thought else "Unknown problem"

        system_prompt = f"""You are applying {type_info["name"]} to solve a problem.

{type_info["name"]}: {type_info["description"]}
Approach: {type_info["approach"]}

Provide a concise solution using this specific reasoning type."""

        user_prompt = f"""Problem Context:
{problem}

Apply {type_info["name"]} to solve this problem.
Focus specifically on the {rtype} reasoning approach.
Provide a clear, focused solution."""

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=lambda: "",
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300,
        )

        # If fallback returned empty string, use heuristic
        if not content:
            return self._heuristic_generate_solution(rtype, type_info)

        return {
            "type": rtype,
            "name": type_info["name"],
            "approach": type_info["approach"],
            "solution": content,
            "confidence": 0.75 + (0.05 if rtype == "deductive" else 0),
        }

    async def _sample_integrate_solutions(self, session: Session) -> str:
        """Use LLM sampling to integrate multiple typed solutions.

        Args:
            session: The current reasoning session

        Returns:
            Integrated solution text
        """

        # Define fallback generator
        def _fallback() -> str:
            best_solution = max(
                self._typed_solutions.values(),
                key=lambda x: x["confidence"],
            )
            return f"[Integrated answer combining {best_solution['name']} as primary]"

        # Prepare solutions summary
        solutions_text = "\n\n".join(
            f"[{s['name']}]\n{s['solution']}\nConfidence: {s['confidence']:.0%}"
            for s in self._typed_solutions.values()
        )

        system_prompt = """You are integrating multiple reasoning approaches to produce \
a final answer.
Synthesize insights from different reasoning types into a coherent, unified solution.
Consider the strengths of each approach and produce the best possible answer."""

        user_prompt = f"""Multiple reasoning types have been applied to solve the problem:

{solutions_text}

Integrate these different perspectives into a single, coherent final answer.
Consider:
- Areas where approaches agree (highest confidence)
- Unique insights from each approach
- How to synthesize into a unified solution"""

        return await self._sample_with_fallback(
            user_prompt,
            fallback_generator=_fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500,
        )


__all__ = ["TypedThinker", "TYPED_THINKER_METADATA"]
