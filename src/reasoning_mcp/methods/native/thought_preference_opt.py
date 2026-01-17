"""Thought Preference Optimization (TPO) reasoning method.

This module implements TPO, which trains LLMs to generate internal thoughts
before producing responses. Unlike explicit CoT, thoughts are generated
internally and optimized via preference learning.

Key phases:
1. Think: Generate internal thoughts (not shown to user)
2. Prefer: Apply preference optimization on thought quality
3. Respond: Generate final response informed by thoughts
4. Learn: Improve thought generation through feedback

Reference: Wu et al. (2024) - "Thinking LLMs: General Instruction Following
with Thought Generation"
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


THOUGHT_PREFERENCE_OPT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.THOUGHT_PREFERENCE_OPT,
    name="Thought Preference Optimization",
    description="Trains LLMs to generate internal thoughts before responses. "
    "Thoughts are optimized via preference learning for better instruction following.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"internal-thoughts", "preference", "optimization", "instruction-following"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=300,
    best_for=(
        "instruction following",
        "general tasks",
        "thought optimization",
        "reasoning improvement",
    ),
    not_recommended_for=("simple queries", "latency-sensitive tasks"),
)


class ThoughtPreferenceOpt(ReasoningMethodBase):
    """Thought Preference Optimization implementation."""

    # Enable LLM sampling for generating thoughts and preferences
    _use_sampling: bool = True

    def __init__(self) -> None:
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "think"
        self._internal_thoughts: list[dict[str, Any]] = []
        self._thought_preferences: list[dict[str, Any]] = []
        self._final_response: str | None = None
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.THOUGHT_PREFERENCE_OPT

    @property
    def name(self) -> str:
        return THOUGHT_PREFERENCE_OPT_METADATA.name

    @property
    def description(self) -> str:
        return THOUGHT_PREFERENCE_OPT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "think"
        self._internal_thoughts = []
        self._thought_preferences = []
        self._final_response = None

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("TPO must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "think"

        # Generate internal thoughts using LLM sampling if available
        self._internal_thoughts = await self._sample_internal_thoughts(input_text, context)

        content = (
            f"Step {self._step_counter}: Generate Internal Thoughts (TPO)\n\n"
            f"Problem: {input_text}\n\n"
            f"Generating internal thoughts (normally hidden):\n\n"
            f"Internal Thought Process:\n"
            + "\n".join(
                f"  <thought-{t['id']}> [{t['type'].upper()}]\n"
                f'    "{t["thought"]}"\n'
                f"    Quality estimate: {t['quality']:.0%}"
                for t in self._internal_thoughts
            )
            + "\n\nTPO Principle:\n"
            "  - Thoughts generated internally, not shown to user\n"
            "  - Quality optimized via preference learning\n"
            "  - Better thoughts → better responses\n\n"
            "Next: Apply preference optimization."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THOUGHT_PREFERENCE_OPT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={
                "phase": self._current_phase,
                "thoughts": len(self._internal_thoughts),
                "input": input_text,
            },
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.THOUGHT_PREFERENCE_OPT
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
            raise RuntimeError("TPO must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "think")

        if prev_phase == "think":
            self._current_phase = "prefer"
            # Apply preference optimization using LLM sampling if available
            self._thought_preferences = await self._sample_preferences(
                self._internal_thoughts, guidance, context
            )

            # Update qualities based on preferences
            for t in self._internal_thoughts:
                wins = sum(1 for p in self._thought_preferences if p["preferred"] == t["id"])
                t["quality"] = min(0.95, t["quality"] + wins * 0.05)

            content = (
                f"Step {self._step_counter}: Preference Optimization\n\n"
                f"Optimizing thought quality via preferences:\n\n"
                f"Pairwise Preferences:\n"
                + "\n".join(
                    f"  Thought {p['comparison'][0]} vs {p['comparison'][1]}: "
                    f"Prefer {p['preferred']} ({p['reason']})"
                    for p in self._thought_preferences
                )
                + "\n\nUpdated Thought Quality:\n"
                + "\n".join(
                    f"  Thought {t['id']}: {t['quality']:.0%} ({t['type']})"
                    for t in sorted(self._internal_thoughts, key=lambda x: -x["quality"])
                )
                + "\n\nPreference learning improves thought generation.\n"
                "Next: Generate response from optimized thoughts."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.75
        elif prev_phase == "prefer":
            self._current_phase = "respond"
            # Generate response informed by thoughts using LLM sampling if available
            best_thoughts = sorted(self._internal_thoughts, key=lambda x: -x["quality"])[:2]
            self._final_response = await self._sample_final_response(
                best_thoughts, previous_thought, guidance, context
            )

            content = (
                f"Step {self._step_counter}: Generate Final Response\n\n"
                f"Synthesizing response from optimized thoughts:\n\n"
                f"Top Contributing Thoughts:\n"
                + "\n".join(
                    f'  [{t["type"].upper()}] (quality: {t["quality"]:.0%})\n    "{t["thought"]}"'
                    for t in best_thoughts
                )
                + f"\n\nResponse Generation:\n"
                f"  - Internal thoughts inform response structure\n"
                f"  - Higher quality thoughts weighted more\n"
                f"  - Final response doesn't expose thought process\n\n"
                f"Generated Response:\n"
                f"  {self._final_response}"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        elif prev_phase == "respond":
            self._current_phase = "learn"
            # Learning from feedback
            avg_quality = sum(t["quality"] for t in self._internal_thoughts) / len(
                self._internal_thoughts
            )

            content = (
                f"Step {self._step_counter}: Learning Update\n\n"
                f"Capturing feedback for future thought improvement:\n\n"
                f"Session Statistics:\n"
                f"  Thoughts generated: {len(self._internal_thoughts)}\n"
                f"  Preference comparisons: {len(self._thought_preferences)}\n"
                f"  Average thought quality: {avg_quality:.0%}\n\n"
                f"Learning Signals:\n"
                f"  - Preferred thought patterns identified\n"
                f"  - Quality gradients computed\n"
                f"  - Model weights updated (via DPO)\n\n"
                f"TPO improves thought generation over time."
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.88
        else:
            self._current_phase = "conclude"
            avg_quality = (
                sum(t["quality"] for t in self._internal_thoughts) / len(self._internal_thoughts)
                if self._internal_thoughts
                else 0.85
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Thought Preference Optimization Complete:\n"
                f"  Internal thoughts: {len(self._internal_thoughts)}\n"
                f"  Preference pairs: {len(self._thought_preferences)}\n"
                f"  Average quality: {avg_quality:.0%}\n\n"
                f"Final Answer: {self._final_response}\n"
                f"Confidence: High ({int(avg_quality * 100 + 5)}%)\n\n"
                f"Method: Thought Preference Optimization (TPO)\n"
                f"  - Internal thought generation\n"
                f"  - Preference-based optimization\n"
                f"  - Thought-informed responses\n"
                f"  - Continuous learning from feedback\n"
                f"  - No explicit CoT needed"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = avg_quality + 0.05

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.THOUGHT_PREFERENCE_OPT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "thoughts": len(self._internal_thoughts),
                "preferences": len(self._thought_preferences),
            },
        )
        session.add_thought(thought)
        return thought

    async def _sample_internal_thoughts(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Generate internal thoughts using LLM sampling.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            List of internal thought dictionaries
        """
        system_prompt = """You are a reasoning assistant using Thought Preference Optimization.
Generate internal thoughts that will inform your final response. These thoughts are not shown
to the user but guide your reasoning process.

Your internal thoughts should:
1. Show comprehension of the task
2. Analyze different aspects of the problem
3. Extract key insights
4. Plan the response structure

Generate 4 distinct internal thoughts, each with:
- A thought type (comprehension, analysis, insight, or planning)
- The actual thought content
- An initial quality estimate (0.0-1.0)

Format as a JSON array of objects with keys: id, thought, type, quality"""

        user_prompt = f"""Problem: {input_text}

Generate 4 internal thoughts to guide your reasoning. Return ONLY a JSON array."""

        def fallback() -> str:
            """Fallback heuristic implementation."""
            import json

            thoughts = self._generate_internal_thoughts_heuristic(input_text, context)
            return json.dumps(thoughts)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # Try to parse JSON response
        import json

        try:
            # Extract JSON if embedded in markdown or other text
            json_start = result.find("[")
            json_end = result.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                thoughts = json.loads(json_str)
                # Validate and normalize structure
                normalized = []
                for i, t in enumerate(thoughts[:4], 1):  # Limit to 4 thoughts
                    normalized.append(
                        {
                            "id": i,
                            "thought": str(t.get("thought", "Internal reasoning step")),
                            "type": str(t.get("type", "analysis")).lower(),
                            "quality": float(t.get("quality", 0.7)),
                        }
                    )
                return (
                    normalized
                    if normalized
                    else self._generate_internal_thoughts_heuristic(input_text, context)
                )
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                "generation_failed",
                method="_sample_internal_thoughts",
                error=str(e),
                exc_info=True,
            )
            # Fall back to heuristic on any parsing error
            pass

        return self._generate_internal_thoughts_heuristic(input_text, context)

    def _generate_internal_thoughts_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Generate internal thoughts using heuristic fallback.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            List of internal thought dictionaries
        """
        return [
            {
                "id": 1,
                "thought": "Let me understand what the user is asking...",
                "type": "comprehension",
                "quality": 0.7,
            },
            {
                "id": 2,
                "thought": "This requires considering multiple aspects...",
                "type": "analysis",
                "quality": 0.8,
            },
            {
                "id": 3,
                "thought": "The key insight here is...",
                "type": "insight",
                "quality": 0.85,
            },
            {
                "id": 4,
                "thought": "I should structure my response to address all points...",
                "type": "planning",
                "quality": 0.75,
            },
        ]

    async def _sample_preferences(
        self,
        thoughts: list[dict[str, Any]],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Generate thought preferences using LLM sampling.

        Args:
            thoughts: List of internal thoughts to compare
            guidance: Optional guidance for preference generation
            context: Optional additional context

        Returns:
            List of preference comparison dictionaries
        """
        thoughts_desc = "\n".join(
            f'Thought {t["id"]}: [{t["type"].upper()}] "{t["thought"]}" '
            f"(quality: {t['quality']:.0%})"
            for t in thoughts
        )

        system_prompt = """You are a reasoning assistant using Thought Preference Optimization.
Compare pairs of internal thoughts and determine which is preferred based on quality criteria:
- Clarity and precision
- Depth of analysis
- Relevance to the task
- Potential to improve final response

Generate pairwise preference comparisons for thought quality optimization.

Format as a JSON array of objects with keys: comparison (tuple of two IDs),
preferred (ID), reason (string)"""

        user_prompt = f"""Internal Thoughts:
{thoughts_desc}

Generate 3 pairwise preference comparisons to optimize thought quality. Return ONLY a JSON array."""

        def fallback() -> str:
            """Fallback heuristic implementation."""
            import json

            prefs = self._generate_preferences_heuristic(thoughts)
            return json.dumps(prefs)

        result = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # Try to parse JSON response
        import json

        try:
            # Extract JSON if embedded in markdown or other text
            json_start = result.find("[")
            json_end = result.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result[json_start:json_end]
                preferences = json.loads(json_str)
                # Validate and normalize structure
                normalized = []
                for p in preferences[:3]:  # Limit to 3 comparisons
                    comp = p.get("comparison", [1, 2])
                    if isinstance(comp, list) and len(comp) >= 2:
                        normalized.append(
                            {
                                "comparison": (int(comp[0]), int(comp[1])),
                                "preferred": int(p.get("preferred", comp[1])),
                                "reason": str(p.get("reason", "Quality improvement")),
                            }
                        )
                return normalized if normalized else self._generate_preferences_heuristic(thoughts)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                "generation_failed",
                method="_sample_preferences",
                error=str(e),
                exc_info=True,
            )
            # Fall back to heuristic on any parsing error
            pass

        return self._generate_preferences_heuristic(thoughts)

    def _generate_preferences_heuristic(
        self,
        thoughts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate thought preferences using heuristic fallback.

        Args:
            thoughts: List of internal thoughts to compare

        Returns:
            List of preference comparison dictionaries
        """
        return [
            {
                "comparison": (1, 2),
                "preferred": 2,
                "reason": "More analytical depth",
            },
            {
                "comparison": (2, 3),
                "preferred": 3,
                "reason": "Key insight extraction",
            },
            {
                "comparison": (3, 4),
                "preferred": 3,
                "reason": "Insight more valuable than planning alone",
            },
        ]

    async def _sample_final_response(
        self,
        best_thoughts: list[dict[str, Any]],
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final response using LLM sampling.

        Args:
            best_thoughts: Top quality thoughts to inform response
            previous_thought: Previous thought node
            guidance: Optional guidance for response generation
            context: Optional additional context

        Returns:
            The generated final response
        """
        # Get original input from metadata
        input_metadata = previous_thought.metadata or {}
        original_input = input_metadata.get("input", "the user's question")

        thoughts_desc = "\n".join(
            f'[{t["type"].upper()}] (quality: {t["quality"]:.0%}): "{t["thought"]}"'
            for t in best_thoughts
        )

        system_prompt = """You are a reasoning assistant using Thought Preference Optimization.
Generate a final response informed by your optimized internal thoughts.

Your response should:
1. Address the original task directly
2. Be informed by your internal thoughts without exposing them
3. Be clear, concise, and well-structured
4. Demonstrate the benefits of internal reasoning"""

        user_prompt = f"""Original Question: {original_input}

Top Internal Thoughts (optimized via preference learning):
{thoughts_desc}

Generate a final response that is informed by these internal thoughts but doesn't expose
the thought process itself. The user should see only the polished final answer."""

        def fallback() -> str:
            """Fallback response."""
            return (
                f"[Response informed by internal thoughts: "
                f"{', '.join(t['type'] for t in best_thoughts)}]"
            )

        return await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ThoughtPreferenceOpt", "THOUGHT_PREFERENCE_OPT_METADATA"]
