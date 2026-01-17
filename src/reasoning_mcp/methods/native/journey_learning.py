"""Journey Learning reasoning method.

This module implements Journey Learning, which focuses on learning from the
entire reasoning journey rather than just the final answer. The method
captures insights from exploration, mistakes, and course corrections.

Key phases:
1. Explore: Begin reasoning journey with open exploration
2. Reflect: Capture learnings from the journey so far
3. Adjust: Make corrections based on journey insights
4. Synthesize: Combine journey learnings into final answer

Reference: Microsoft Research (2024) - "Journey Learning" approaches
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


JOURNEY_LEARNING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.JOURNEY_LEARNING,
    name="Journey Learning",
    description="Learns from the reasoning journey, not just the final answer. "
    "Captures exploration insights, mistakes, and corrections for richer understanding.",
    category=MethodCategory.HOLISTIC,
    tags=frozenset({"journey", "exploration", "learning", "reflection", "process"}),
    complexity=6,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=10,
    avg_tokens_per_thought=300,
    best_for=("learning from process", "exploration tasks", "complex problem solving"),
    not_recommended_for=("simple lookups", "time-critical tasks"),
)


class JourneyLearning(ReasoningMethodBase):
    """Journey Learning reasoning method implementation."""

    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "explore"
        self._journey_log: list[dict[str, Any]] = []
        self._insights: list[str] = []
        self._adjustments: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.JOURNEY_LEARNING

    @property
    def name(self) -> str:
        return JOURNEY_LEARNING_METADATA.name

    @property
    def description(self) -> str:
        return JOURNEY_LEARNING_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "explore"
        self._journey_log = []
        self._insights = []
        self._adjustments = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Journey Learning must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "explore"
        self._journey_log.append(
            {
                "step": 1,
                "action": "Begin exploration",
                "observation": "Initial problem understanding",
            }
        )

        content = (
            f"Step {self._step_counter}: Begin Journey (Journey Learning)\n\n"
            f"Problem: {input_text}\n\n"
            f"🗺️ Starting the reasoning journey...\n\n"
            f"Initial Exploration:\n"
            f"  • Mapping the problem space\n"
            f"  • Identifying potential paths\n"
            f"  • Noting initial impressions\n\n"
            f"Journey Log Entry #1:\n"
            f"  Action: Begin exploration\n"
            f"  Observation: Initial problem understanding\n\n"
            f"Next: Continue exploration and reflect on learnings."
        )

        thought = ThoughtNode(
            type=ThoughtType.EXPLORATION,
            method_id=MethodIdentifier.JOURNEY_LEARNING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.5,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "journey_log": self._journey_log},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.JOURNEY_LEARNING
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
            raise RuntimeError("Journey Learning must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "explore")

        if prev_phase == "explore":
            self._current_phase = "reflect"
            self._journey_log.append(
                {
                    "step": 2,
                    "action": "Deeper exploration",
                    "observation": "Discovered unexpected complexity",
                }
            )
            # Extract input_text from session or previous thought
            input_text = ""
            if session.thoughts:
                first_thought = session.thoughts[0]
                # Try to extract problem from first thought content
                for line in first_thought.content.split("\n"):
                    if line.startswith("Problem:"):
                        input_text = line.replace("Problem:", "").strip()
                        break

            self._insights = await self._generate_insights(input_text, self._journey_log)
            content = (
                f"Step {self._step_counter}: Reflect on Journey\n\n"
                f"📝 Reflecting on the journey so far...\n\n"
                f"Journey Insights:\n"
                + "\n".join(f"  💡 {i}" for i in self._insights)
                + "\n\nLearnings from exploration:\n"
                "  • What worked: Open exploration revealed options\n"
                "  • What didn't: Initial assumptions too narrow\n"
                "  • Key discovery: Problem deeper than it appeared\n\n"
                "Next: Adjust approach based on learnings."
            )
            thought_type = ThoughtType.INSIGHT
            confidence = 0.65
        elif prev_phase == "reflect":
            self._current_phase = "adjust"
            # Extract input_text from session
            input_text = ""
            if session.thoughts:
                first_thought = session.thoughts[0]
                for line in first_thought.content.split("\n"):
                    if line.startswith("Problem:"):
                        input_text = line.replace("Problem:", "").strip()
                        break

            self._adjustments = await self._generate_adjustments(input_text, self._insights)
            self._journey_log.append(
                {
                    "step": 3,
                    "action": "Course correction",
                    "observation": "Adjusted based on reflections",
                }
            )
            content = (
                f"Step {self._step_counter}: Adjust Course\n\n"
                f"🔄 Making adjustments based on journey learnings...\n\n"
                f"Course Corrections:\n"
                + "\n".join(f"  → {a}" for a in self._adjustments)
                + "\n\nJourney Progress:\n"
                f"  • Steps taken: {len(self._journey_log)}\n"
                f"  • Insights gathered: {len(self._insights)}\n"
                f"  • Adjustments made: {len(self._adjustments)}\n\n"
                f"Next: Synthesize journey learnings into solution."
            )
            thought_type = ThoughtType.REVISION
            confidence = 0.75
        elif prev_phase == "adjust":
            self._current_phase = "synthesize"
            # Extract input_text from session
            input_text = ""
            if session.thoughts:
                first_thought = session.thoughts[0]
                for line in first_thought.content.split("\n"):
                    if line.startswith("Problem:"):
                        input_text = line.replace("Problem:", "").strip()
                        break

            synthesized = await self._synthesize_solution(
                input_text, self._journey_log, self._insights, self._adjustments
            )

            content = (
                f"Step {self._step_counter}: Synthesize Journey\n\n"
                f"🔮 Synthesizing all journey learnings...\n\n"
                f"Journey Summary:\n"
                + "\n".join(
                    f"  {j['step']}. {j['action']}: {j['observation']}" for j in self._journey_log
                )
                + "\n\nKey Takeaways:\n"
                + "\n".join(f"  • {i}" for i in self._insights[:2])
                + f"\n\nSynthesized Solution:\n{synthesized}"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.8
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"🏁 Journey Learning Complete:\n"
                f"  • Journey steps: {len(self._journey_log)}\n"
                f"  • Insights gained: {len(self._insights)}\n"
                f"  • Adjustments made: {len(self._adjustments)}\n\n"
                f"Final Answer: [Solution enriched by journey experience]\n"
                f"Confidence: High (85%)\n\n"
                f"Journey Value: The process taught us as much as the answer.\n"
                f"  • Learned about problem structure\n"
                f"  • Discovered unexpected connections\n"
                f"  • Built intuition for similar problems"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.JOURNEY_LEARNING,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "journey_log": self._journey_log,
                "insights": self._insights,
            },
        )
        session.add_thought(thought)
        return thought

    def _fallback_insights(self) -> list[str]:
        """Return fallback insights when LLM sampling is unavailable."""
        return [
            "The problem has hidden dependencies",
            "Initial approach may need revision",
            "Multiple valid solution paths exist",
        ]

    async def _generate_insights(
        self, problem: str, journey_log: list[dict[str, Any]]
    ) -> list[str]:
        """Generate insights from the journey using LLM sampling or fallback heuristics."""
        journey_summary = "\n".join(
            f"{j['step']}. {j['action']}: {j['observation']}" for j in journey_log
        )
        prompt = f"""Based on this reasoning journey for the problem:
Problem: {problem}

Journey so far:
{journey_summary}

Provide 3 key insights you've learned from this exploration. Focus on:
- What worked and what didn't
- Unexpected discoveries or complexities
- Hidden patterns or dependencies

Format as a list of concise insights."""

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: "\n".join(self._fallback_insights()),
            system_prompt=(
                "You are analyzing a reasoning journey to extract key insights and learnings."
            ),
        )

        # Parse insights from result
        lines = result.strip().split("\n")
        insights = [
            line.strip().lstrip("•-*").strip()
            for line in lines
            if line.strip() and not line.strip().startswith(("Insight", "Key", "Based"))
        ][:3]

        if insights:
            return insights

        # If parsing yielded nothing, return fallback
        return self._fallback_insights()

    def _fallback_adjustments(self) -> list[str]:
        """Return fallback adjustments when LLM sampling is unavailable."""
        return [
            "Broaden solution search space",
            "Account for dependencies discovered",
            "Consider multiple solution paths",
        ]

    async def _generate_adjustments(self, problem: str, insights: list[str]) -> list[str]:
        """Generate course corrections using LLM sampling or fallback heuristics."""
        insights_text = "\n".join(f"- {i}" for i in insights)
        prompt = f"""Based on these insights from the reasoning journey:

Problem: {problem}

Insights learned:
{insights_text}

Suggest 3 specific course corrections or adjustments to the approach. Focus on:
- How to address discovered complexities
- Ways to broaden or refine the solution space
- Concrete changes to the reasoning strategy

Format as a list of actionable adjustments."""

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: "\n".join(self._fallback_adjustments()),
            system_prompt="You are suggesting course corrections based on journey insights.",
        )

        # Parse adjustments from result
        lines = result.strip().split("\n")
        adjustments = [
            line.strip().lstrip("•-*").strip()
            for line in lines
            if line.strip() and not line.strip().startswith(("Adjust", "Based", "Suggest"))
        ][:3]

        if adjustments:
            return adjustments

        # If parsing yielded nothing, return fallback
        return self._fallback_adjustments()

    async def _synthesize_solution(
        self,
        problem: str,
        journey_log: list[dict[str, Any]],
        insights: list[str],
        adjustments: list[str],
    ) -> str:
        """Synthesize final solution from journey using LLM sampling or fallback."""
        journey_summary = "\n".join(
            f"{j['step']}. {j['action']}: {j['observation']}" for j in journey_log
        )
        insights_text = "\n".join(f"- {i}" for i in insights)
        adjustments_text = "\n".join(f"- {a}" for a in adjustments)

        prompt = f"""Synthesize a final solution based on this complete reasoning journey:

Problem: {problem}

Journey steps:
{journey_summary}

Insights learned:
{insights_text}

Adjustments made:
{adjustments_text}

Provide a concise final answer that incorporates all the learnings from this journey."""

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: "[Solution enriched by journey experience]",
            system_prompt=(
                "You are synthesizing a final solution from a complete reasoning journey."
            ),
        )

        return result.strip()

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["JourneyLearning", "JOURNEY_LEARNING_METADATA"]
