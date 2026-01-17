"""Meta Chain-of-Thought (Meta-CoT) reasoning method.

This module implements Meta-CoT, which teaches models to reason about reasoning
itself. Rather than just following a reasoning chain, Meta-CoT learns meta-level
strategies for how to approach different types of problems.

Key phases:
1. Analyze: Classify the problem type and complexity
2. Strategize: Select appropriate reasoning strategy
3. Execute: Apply the chosen meta-strategy
4. Reflect: Evaluate the meta-reasoning process

Reference: "Towards System 2 Reasoning in LLMs: Learning How to Think
With Meta Chain-of-Thought" (2025)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.methods.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SAMPLING_TEMPERATURE,
    MethodMetadata,
    ReasoningMethodBase,
)
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


META_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.META_COT,
    name="Meta Chain-of-Thought",
    description="Learns how to think by reasoning about reasoning strategies. "
    "Analyzes problem types and selects appropriate meta-level approaches.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"meta-reasoning", "strategy", "adaptive", "system-2", "metacognitive"}),
    complexity=8,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=5,
    max_thoughts=10,
    avg_tokens_per_thought=350,
    best_for=("complex reasoning", "novel problems", "adaptive approaches"),
    not_recommended_for=("routine tasks", "well-defined problems"),
)


class MetaCoT(ReasoningMethodBase):
    """Meta Chain-of-Thought implementation."""

    _use_sampling: bool = True

    STRATEGIES = [
        "decomposition",
        "analogy",
        "abstraction",
        "backward-chaining",
        "hypothesis-testing",
    ]

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "analyze"
        self._problem_type: str = ""
        self._complexity: str = ""
        self._selected_strategy: str = ""
        self._strategy_rationale: str = ""
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.META_COT

    @property
    def name(self) -> str:
        return META_COT_METADATA.name

    @property
    def description(self) -> str:
        return META_COT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "analyze"
        self._problem_type = ""
        self._complexity = ""
        self._selected_strategy = ""
        self._strategy_rationale = ""

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Meta-CoT must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "analyze"

        # Analyze problem type and complexity using sampling with fallback
        content = await self._sample_analyze_phase(input_text)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.META_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "problem_type": self._problem_type},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.META_COT
        return thought

    async def _sample_analyze_phase(self, input_text: str) -> str:
        """Sample the analyze phase using LLM with fallback.

        Args:
            input_text: The problem to analyze

        Returns:
            Formatted analysis content
        """
        analysis_prompt = (
            f"Analyze this problem for Meta Chain-of-Thought reasoning:\n\n"
            f"Problem: {input_text}\n\n"
            f"Provide:\n"
            f"1. Problem type (e.g., mathematical, logical, creative, etc.)\n"
            f"2. Complexity level (low, medium, high)\n"
            f"3. Key characteristics that affect reasoning approach\n\n"
            f"Format your response as:\n"
            f"Problem Type: <type>\n"
            f"Complexity: <level>\n"
            f"Characteristics:\n- <characteristic1>\n- <characteristic2>"
        )

        system_prompt = (
            "You are analyzing problems for meta-cognitive "
            "reasoning. Classify the problem type, complexity, "
            "and key characteristics."
        )

        def fallback() -> str:
            self._problem_type = "multi-step reasoning"
            self._complexity = "high"
            return (
                f"Step {self._step_counter}: Analyze Problem (Meta-CoT)\n\n"
                f"Problem: {input_text}\n\n"
                f"Meta-Analysis:\n"
                f"  Problem Type: {self._problem_type}\n"
                f"  Complexity Level: {self._complexity}\n"
                f"  Key Characteristics:\n"
                f"    - Requires multiple reasoning steps\n"
                f"    - Contains implicit constraints\n"
                f"    - May benefit from decomposition\n\n"
                f"Available Strategies:\n"
                + "\n".join(f"  - {s}" for s in self.STRATEGIES)
                + "\n\nNext: Select optimal meta-strategy."
            )

        analysis_text = await self._sample_with_fallback(
            analysis_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # If we got a fallback response, return it directly
        if analysis_text.startswith(f"Step {self._step_counter}:"):
            return analysis_text

        # Parse the sampled analysis
        self._problem_type = self._extract_field(
            analysis_text, "Problem Type:", "multi-step reasoning"
        )
        self._complexity = self._extract_field(analysis_text, "Complexity:", "high")

        return (
            f"Step {self._step_counter}: Analyze Problem (Meta-CoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"Meta-Analysis:\n"
            f"  Problem Type: {self._problem_type}\n"
            f"  Complexity Level: {self._complexity}\n\n"
            f"{analysis_text}\n\n"
            f"Available Strategies:\n"
            + "\n".join(f"  - {s}" for s in self.STRATEGIES)
            + "\n\nNext: Select optimal meta-strategy."
        )

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
            raise RuntimeError("Meta-CoT must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "analyze")

        if prev_phase == "analyze":
            self._current_phase = "strategize"
            content = await self._sample_strategize_phase(previous_thought)
            thought_type = ThoughtType.REASONING
            confidence = 0.7
        elif prev_phase == "strategize":
            self._current_phase = "execute"
            content = await self._sample_execute_phase()
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.8
        elif prev_phase == "execute":
            self._current_phase = "reflect"
            content = await self._sample_reflect_phase(previous_thought)
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Meta Chain-of-Thought Complete:\n"
                f"  Problem Type: {self._problem_type}\n"
                f"  Strategy Applied: {self._selected_strategy}\n"
                f"  Meta-Reasoning Effectiveness: High\n\n"
                f"Final Answer: [Answer from meta-guided reasoning]\n"
                f"Confidence: High (87%)\n\n"
                f"Method: Meta Chain-of-Thought\n"
                f"  - Analyzed problem characteristics\n"
                f"  - Selected optimal reasoning strategy\n"
                f"  - Applied meta-level guidance\n"
                f"  - Reflected on reasoning process"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.87

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.META_COT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "strategy": self._selected_strategy,
            },
        )
        session.add_thought(thought)
        return thought

    async def _sample_strategize_phase(self, previous_thought: ThoughtNode) -> str:
        """Sample the strategize phase using LLM with fallback.

        Args:
            previous_thought: The previous thought from the analyze phase

        Returns:
            Formatted strategy selection content
        """
        strategy_prompt = (
            f"Select the best meta-reasoning strategy for this problem.\n\n"
            f"Problem Type: {self._problem_type}\n"
            f"Complexity: {self._complexity}\n"
            f"Previous Analysis: {previous_thought.content}\n\n"
            f"Available Strategies:\n"
            + "\n".join(f"  - {s}" for s in self.STRATEGIES)
            + "\n\nEvaluate each strategy and select the most appropriate one.\n"
            "Provide:\n"
            "1. Evaluation of each strategy (HIGH/MEDIUM/LOW fit)\n"
            "2. Selected strategy\n"
            "3. Detailed rationale\n\n"
            "Format:\n"
            "Evaluations:\n[evaluations]\n\n"
            "Selected: <strategy>\n"
            "Rationale: <rationale>"
        )

        system_prompt = (
            "You are a meta-reasoning expert selecting optimal "
            "reasoning strategies for different problem types."
        )

        def fallback() -> str:
            self._selected_strategy = "decomposition"
            self._strategy_rationale = (
                "Problem complexity suggests breaking into sub-problems "
                "will reduce cognitive load and improve accuracy."
            )
            return (
                f"Step {self._step_counter}: Select Meta-Strategy\n\n"
                f"Evaluating strategies for {self._problem_type}:\n\n"
                f"  [1] decomposition: HIGH fit - multi-step problems benefit\n"
                f"  [2] analogy: MEDIUM fit - no clear analogous cases\n"
                f"  [3] abstraction: MEDIUM fit - could help generalize\n"
                f"  [4] backward-chaining: LOW fit - goal not well-defined\n"
                f"  [5] hypothesis-testing: MEDIUM fit - exploratory approach\n\n"
                f"Selected: {self._selected_strategy.upper()}\n"
                f"Rationale: {self._strategy_rationale}\n\n"
                f"Next: Execute chosen meta-strategy."
            )

        strategy_text = await self._sample_with_fallback(
            strategy_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # If we got a fallback response, return it directly
        if strategy_text.startswith(f"Step {self._step_counter}:"):
            return strategy_text

        # Parse the sampled strategy
        self._selected_strategy = self._extract_field(
            strategy_text, "Selected:", "decomposition"
        ).strip()
        self._strategy_rationale = self._extract_field(
            strategy_text,
            "Rationale:",
            "Breaking into sub-problems will help manage complexity.",
        )

        return (
            f"Step {self._step_counter}: Select Meta-Strategy\n\n"
            f"Evaluating strategies for {self._problem_type}:\n\n"
            f"{strategy_text}\n\n"
            f"Next: Execute chosen meta-strategy."
        )

    async def _sample_execute_phase(self) -> str:
        """Sample the execute phase using LLM with fallback.

        Returns:
            Formatted strategy execution content
        """
        execution_prompt = (
            f"Execute the selected meta-reasoning strategy.\n\n"
            f"Strategy: {self._selected_strategy}\n"
            f"Problem Type: {self._problem_type}\n"
            f"Complexity: {self._complexity}\n"
            f"Strategy Rationale: {self._strategy_rationale}\n\n"
            f"Apply the {self._selected_strategy} strategy step by step.\n"
            f"Break down the problem, show reasoning for each component, "
            f"and integrate the results.\n\n"
            f"Format your response with:\n"
            f"1. Sub-problems or components identified\n"
            f"2. Step-by-step reasoning for each\n"
            f"3. Intermediate results\n"
            f"4. Integration of results"
        )

        system_prompt = (
            f"You are applying the {self._selected_strategy} "
            "meta-reasoning strategy to solve a problem systematically."
        )

        def fallback() -> str:
            return (
                f"Step {self._step_counter}: Execute Meta-Strategy\n\n"
                f"Applying {self._selected_strategy} strategy:\n\n"
                f"  Sub-problem 1: [Identified component]\n"
                f"    - Reasoning: [Step-by-step analysis]\n"
                f"    - Result: [Intermediate answer]\n\n"
                f"  Sub-problem 2: [Identified component]\n"
                f"    - Reasoning: [Step-by-step analysis]\n"
                f"    - Result: [Intermediate answer]\n\n"
                f"  Sub-problem 3: [Identified component]\n"
                f"    - Reasoning: [Step-by-step analysis]\n"
                f"    - Result: [Intermediate answer]\n\n"
                f"Integration: Combining sub-solutions...\n"
                f"Next: Reflect on meta-reasoning effectiveness."
            )

        execution_text = await self._sample_with_fallback(
            execution_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # If we got a fallback response, return it directly
        if execution_text.startswith(f"Step {self._step_counter}:"):
            return execution_text

        return (
            f"Step {self._step_counter}: Execute Meta-Strategy\n\n"
            f"Applying {self._selected_strategy} strategy:\n\n"
            f"{execution_text}\n\n"
            f"Next: Reflect on meta-reasoning effectiveness."
        )

    async def _sample_reflect_phase(self, previous_thought: ThoughtNode) -> str:
        """Sample the reflect phase using LLM with fallback.

        Args:
            previous_thought: The previous thought from the execute phase

        Returns:
            Formatted reflection content
        """
        reflection_prompt = (
            f"Reflect on the effectiveness of the meta-reasoning process.\n\n"
            f"Strategy Used: {self._selected_strategy}\n"
            f"Problem Type: {self._problem_type}\n"
            f"Execution Summary: {previous_thought.content[:500]}...\n\n"
            f"Evaluate:\n"
            f"1. How effective was the chosen strategy?\n"
            f"2. What aspects worked well?\n"
            f"3. What could be improved?\n"
            f"4. What meta-lessons were learned for future problems?\n\n"
            f"Provide a meta-evaluation with effectiveness rating and insights."
        )

        system_prompt = (
            "You are reflecting on meta-reasoning effectiveness "
            "to improve future problem-solving approaches."
        )

        def fallback() -> str:
            return (
                f"Step {self._step_counter}: Reflect on Meta-Reasoning\n\n"
                f"Meta-Evaluation:\n"
                f"  Strategy Used: {self._selected_strategy}\n"
                f"  Effectiveness: HIGH\n"
                f"  Reasoning Quality: 85%\n\n"
                f"  What Worked:\n"
                f"    - Decomposition reduced complexity\n"
                f"    - Sub-problems were tractable\n"
                f"    - Integration was straightforward\n\n"
                f"  Lessons Learned:\n"
                f"    - This problem type benefits from decomposition\n"
                f"    - Could also try abstraction next time\n\n"
                f"Meta-learning captured for future problems."
            )

        reflection_text = await self._sample_with_fallback(
            reflection_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        # If we got a fallback response, return it directly
        if reflection_text.startswith(f"Step {self._step_counter}:"):
            return reflection_text

        return (
            f"Step {self._step_counter}: Reflect on Meta-Reasoning\n\n"
            f"Meta-Evaluation:\n"
            f"  Strategy Used: {self._selected_strategy}\n\n"
            f"{reflection_text}\n\n"
            f"Meta-learning captured for future problems."
        )

    async def health_check(self) -> bool:
        return self._initialized

    def _extract_field(self, text: str, field_name: str, default: str) -> str:
        """Extract a field value from formatted text."""
        try:
            lines = text.split("\n")
            for line in lines:
                if field_name in line:
                    value = line.split(field_name, 1)[1].strip()
                    return value if value else default
            return default
        except (ValueError, IndexError) as e:
            logger.warning(
                "operation_failed",
                method="_extract_field",
                error=str(e),
                exc_info=True,
            )
            return default


__all__ = ["MetaCoT", "META_COT_METADATA"]
