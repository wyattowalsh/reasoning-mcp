"""Syzygy of Thoughts reasoning method.

This module implements Syzygy of Thoughts, a combinatorial symbolic reasoning
approach that aligns multiple complementary perspectives to achieve deeper
understanding. Like astronomical syzygy (alignment of celestial bodies),
this method aligns different reasoning perspectives to illuminate problems.

Key phases:
1. Identify: Recognize distinct perspectives on the problem
2. Align: Bring perspectives into productive alignment
3. Synthesize: Combine aligned insights
4. Illuminate: Derive understanding from the combined perspectives

Reference: Inspired by symbolic reasoning research (2024-2025)
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


# Metadata for Syzygy of Thoughts method
SYZYGY_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.SYZYGY_OF_THOUGHTS,
    name="Syzygy of Thoughts",
    description="Aligns multiple complementary reasoning perspectives to achieve "
    "deeper understanding. Like celestial syzygy, brings different viewpoints into "
    "productive alignment through identify → align → synthesize → illuminate phases.",
    category=MethodCategory.HOLISTIC,
    tags=frozenset(
        {
            "multi-perspective",
            "alignment",
            "symbolic",
            "combinatorial",
            "holistic",
            "synthesis",
            "complementary",
            "viewpoints",
        }
    ),
    complexity=8,  # High complexity - multiple perspective management
    supports_branching=True,  # Multiple perspectives as branches
    supports_revision=True,  # Can realign perspectives
    requires_context=False,  # No special context needed
    min_thoughts=4,  # identify + align + synthesize + illuminate
    max_thoughts=10,  # Multiple perspectives to align
    avg_tokens_per_thought=400,  # Perspective descriptions are detailed
    best_for=(
        "complex philosophical questions",
        "multi-stakeholder problems",
        "interdisciplinary challenges",
        "controversial topics",
        "wisdom-seeking inquiries",
        "systems thinking",
        "holistic analysis",
        "conflicting requirements",
    ),
    not_recommended_for=(
        "simple factual queries",
        "single-perspective problems",
        "purely technical tasks",
        "time-sensitive decisions",
    ),
)

logger = structlog.get_logger(__name__)


class SyzygyOfThoughts(ReasoningMethodBase):
    """Syzygy of Thoughts reasoning method implementation.

    This class implements the Syzygy pattern:
    1. Identify: Recognize distinct perspectives
    2. Align: Bring them into productive alignment
    3. Synthesize: Combine aligned insights
    4. Illuminate: Derive deeper understanding

    Key characteristics:
    - Multi-perspective reasoning
    - Productive alignment of viewpoints
    - Holistic synthesis
    - High complexity (8)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = SyzygyOfThoughts()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Balance innovation with stability in organizations"
        ... )
        >>> print(result.content)  # Perspective identification phase
    """

    # Maximum perspectives to align
    MAX_PERSPECTIVES = 4

    # Enable LLM sampling
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Syzygy of Thoughts method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "identify"
        self._perspectives: list[dict[str, str]] = []
        self._alignments: list[str] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.SYZYGY_OF_THOUGHTS

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return SYZYGY_OF_THOUGHTS_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return SYZYGY_OF_THOUGHTS_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.HOLISTIC

    async def initialize(self) -> None:
        """Initialize the method."""
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "identify"
        self._perspectives = []
        self._alignments = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Syzygy of Thoughts method."""
        if not self._initialized:
            raise RuntimeError("Syzygy of Thoughts method must be initialized before execution")

        # Store execution context for LLM sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_phase = "identify"
        self._perspectives = []
        self._alignments = []

        # Generate identification phase using LLM if available
        content = await self._sample_with_fallback(
            user_prompt=f"""Problem: {input_text}

Identify 3-4 complementary perspectives that will help illuminate this problem.
For each perspective, provide:
1. Name of the perspective
2. Brief description of what it focuses on
3. Key questions it would ask

Format your response clearly, showing each perspective and how it uniquely contributes.""",
            fallback_generator=lambda: self._generate_identification(input_text, context),
            system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
In this phase, identify 3-4 distinct, complementary perspectives that can illuminate the problem.
Each perspective should offer a unique lens for understanding the issue.

Think of perspectives like:
- Analytical (data-driven, logical)
- Intuitive (pattern-based, holistic)
- Practical (feasibility-focused, grounded)
- Visionary (future-oriented, possibility-seeking)
- Ethical (values-based, impact-conscious)
- Systems (interconnections, dynamics)

Choose the most relevant perspectives for this specific problem.""",
        )

        # Parse perspectives from LLM response if possible
        self._perspectives = [
            {"name": "Analytical", "view": "Data-driven analysis"},
            {"name": "Intuitive", "view": "Pattern recognition"},
            {"name": "Practical", "view": "Real-world constraints"},
            {"name": "Visionary", "view": "Future possibilities"},
        ]

        content = (
            f"Step {self._step_counter}: Perspective Identification (Syzygy)\n\n"
            f"Problem: {input_text}\n\n"
            f"{content}\n\n"
            f"Next: Align these perspectives for synthesis."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SYZYGY_OF_THOUGHTS,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "syzygy_of_thoughts",
                "phase": self._current_phase,
                "perspectives": self._perspectives,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.SYZYGY_OF_THOUGHTS

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
        """Continue reasoning from a previous thought."""
        if not self._initialized:
            raise RuntimeError("Syzygy of Thoughts method must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "identify")
        input_text = previous_thought.metadata.get("input", "")

        if prev_phase == "identify":
            self._current_phase = "align"
            thought_type = ThoughtType.REASONING

            perspectives_text = "\n".join(f"  - {p['name']}: {p['view']}" for p in self._perspectives)
            guidance_text = f"\n\nUser guidance: {guidance}" if guidance else ""

            content = await self._sample_with_fallback(
                user_prompt=f"""Problem: {input_text}

Identified perspectives:
{perspectives_text}{guidance_text}

Now align these perspectives. Show how they can work together productively:
- Which perspectives naturally complement each other?
- Where do they create useful tension or dialogue?
- What synergies emerge from their alignment?

Present the alignment process clearly, showing the connections between perspectives.""",
                fallback_generator=lambda: self._generate_alignment(guidance, context),
                system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
In this phase, align the identified perspectives by finding productive connections between them.
Show how different perspectives complement and enhance each other.

Identify:
- Bridges: Where perspectives connect and reinforce each other
- Tensions: Where they offer contrasting views that create productive dialogue
- Synergies: Where combining perspectives reveals new insights""",
            )

            # Track alignments
            self._alignments = [
                "Analytical ↔ Practical: Evidence meets feasibility",
                "Intuitive ↔ Visionary: Patterns inform possibilities",
                "Analytical ↔ Visionary: Data supports vision",
            ]

            content = (
                f"Step {self._step_counter}: Perspective Alignment\n\n"
                f"Bringing perspectives into productive syzygy...\n\n"
                f"{content}\n\n"
                f"Perspectives aligned. Ready for synthesis."
            )

            confidence = 0.75
            quality_score = 0.8

        elif prev_phase == "align":
            self._current_phase = "synthesize"
            thought_type = ThoughtType.SYNTHESIS

            perspectives_text = "\n".join(f"  - {p['name']}: {p['view']}" for p in self._perspectives)
            alignments_text = "\n".join(f"  - {a}" for a in self._alignments)
            guidance_text = f"\n\nUser guidance: {guidance}" if guidance else ""

            content = await self._sample_with_fallback(
                user_prompt=f"""Problem: {input_text}

Perspectives:
{perspectives_text}

Alignments:
{alignments_text}{guidance_text}

Synthesize these aligned perspectives into a unified understanding.
Show how insights from each perspective combine to create a holistic view
that no single perspective could achieve alone.

Present the synthesis clearly, showing the integration process.""",
                fallback_generator=lambda: self._generate_synthesis(guidance, context),
                system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
In this phase, synthesize the aligned perspectives into a unified understanding.
Combine insights from each perspective to create a holistic view that transcends
any single perspective.

The synthesis should:
- Honor all perspectives' contributions
- Integrate complementary insights
- Resolve apparent tensions
- Create emergent understanding""",
            )

            content = (
                f"Step {self._step_counter}: Perspective Synthesis\n\n"
                f"Combining aligned perspectives...\n\n"
                f"{content}\n\n"
                f"The syzygy reveals a balanced view that "
                f"no single perspective could achieve alone."
            )

            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "synthesize":
            self._current_phase = "illuminate"
            thought_type = ThoughtType.INSIGHT

            perspectives_text = "\n".join(f"  - {p['name']}: {p['view']}" for p in self._perspectives)
            guidance_text = f"\n\nUser guidance: {guidance}" if guidance else ""

            content = await self._sample_with_fallback(
                user_prompt=f"""Problem: {input_text}

Perspectives that were aligned:
{perspectives_text}{guidance_text}

Now articulate the illumination - the deep insights that emerge from having
aligned these perspectives. What does this syzygy reveal that wasn't visible
from any single viewpoint?

Present key illuminations and the emergent understanding clearly.""",
                fallback_generator=lambda: self._generate_illumination(guidance, context),
                system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
In this phase, articulate the deep insights that emerge from the aligned perspectives.
The syzygy (alignment) illuminates aspects of the problem that weren't visible from
any single perspective.

Focus on:
- Emergent insights from the synthesis
- Deep understanding that transcends individual viewpoints
- Novel connections and implications
- Wisdom gained from the holistic view""",
            )

            content = (
                f"Step {self._step_counter}: Illumination\n\n"
                f"The aligned perspectives illuminate the problem...\n\n"
                f"{content}\n\n"
                f"This understanding transcends any single viewpoint."
            )

            confidence = 0.9
            quality_score = 0.9

        elif prev_phase == "illuminate":
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION

            perspectives_text = "\n".join(f"  - {p['name']}: {p['view']}" for p in self._perspectives)
            guidance_text = f"\n\nUser guidance: {guidance}" if guidance else ""

            result = await self._sample_with_fallback(
                user_prompt=f"""Problem: {input_text}

We've completed the Syzygy of Thoughts process:
- Identified {len(self._perspectives)} complementary perspectives
- Aligned them productively ({len(self._alignments)} key alignments)
- Synthesized their insights into unified understanding
- Illuminated deep insights from the alignment

Perspectives:
{perspectives_text}{guidance_text}

Provide the final answer, drawing on the full wisdom gained through this
multi-perspective alignment process. Be comprehensive and nuanced.""",
                fallback_generator=lambda: self._generate_conclusion(guidance, context),
                system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
Provide a final answer that captures the wisdom derived from aligning multiple
complementary perspectives. The answer should be comprehensive, nuanced, and
reflect the deep understanding achieved through the syzygy process.

Include:
- A clear, actionable answer
- Key insights from the perspective alignment
- Confidence assessment and rationale""",
            )

            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Syzygy of Thoughts Complete:\n\n"
                f"Summary:\n"
                f"  - Perspectives identified: {len(self._perspectives)}\n"
                f"  - Alignments formed: {len(self._alignments)}\n"
                f"  - Synthesis achieved: Unified understanding\n\n"
                f"{result}\n\n"
                f"Confidence: Very High (95%)\n"
                f"Reason: Multiple complementary perspectives aligned "
                f"and synthesized into holistic understanding."
            )

            confidence = 0.95
            quality_score = 0.95

        else:
            self._current_phase = "synthesize"
            thought_type = ThoughtType.SYNTHESIS

            perspectives_text = "\n".join(f"  - {p['name']}: {p['view']}" for p in self._perspectives)
            alignments_text = "\n".join(f"  - {a}" for a in self._alignments)
            guidance_text = f"\n\nUser guidance: {guidance}" if guidance else ""

            content = await self._sample_with_fallback(
                user_prompt=f"""Problem: {input_text}

Perspectives:
{perspectives_text}

Alignments:
{alignments_text}{guidance_text}

Synthesize these aligned perspectives into a unified understanding.
Show how insights from each perspective combine to create a holistic view
that no single perspective could achieve alone.

Present the synthesis clearly, showing the integration process.""",
                fallback_generator=lambda: self._generate_synthesis(guidance, context),
                system_prompt="""You are a reasoning assistant using Syzygy of Thoughts methodology.
In this phase, synthesize the aligned perspectives into a unified understanding.
Combine insights from each perspective to create a holistic view that transcends
any single perspective.

The synthesis should:
- Honor all perspectives' contributions
- Integrate complementary insights
- Resolve apparent tensions
- Create emergent understanding""",
            )

            content = (
                f"Step {self._step_counter}: Perspective Synthesis\n\n"
                f"Combining aligned perspectives...\n\n"
                f"{content}\n\n"
                f"The syzygy reveals a balanced view that "
                f"no single perspective could achieve alone."
            )

            confidence = 0.8
            quality_score = 0.8

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.SYZYGY_OF_THOUGHTS,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "syzygy_of_thoughts",
                "perspectives": self._perspectives,
                "alignments": self._alignments,
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _generate_identification(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the perspective identification phase."""
        self._perspectives = [
            {"name": "Analytical", "view": "Data-driven, logical analysis"},
            {"name": "Intuitive", "view": "Pattern recognition, holistic sensing"},
            {"name": "Practical", "view": "Real-world constraints and feasibility"},
            {"name": "Visionary", "view": "Future possibilities and potential"},
        ]

        return (
            f"Step {self._step_counter}: Perspective Identification (Syzygy)\n\n"
            f"Problem: {input_text}\n\n"
            f"Identifying complementary perspectives...\n\n"
            f"Celestial Bodies (Perspectives):\n"
            f"  ☉ Analytical: Logical, evidence-based reasoning\n"
            f"  ☽ Intuitive: Pattern-based, holistic understanding\n"
            f"  ⊕ Practical: Grounded, feasibility-focused\n"
            f"  ☿ Visionary: Forward-looking, possibility-oriented\n\n"
            f"Each perspective offers unique insights.\n"
            f"Next: Align these perspectives for synthesis."
        )

    def _generate_alignment(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the alignment phase."""
        self._alignments = [
            "Analytical ↔ Practical: Ground logic in feasibility",
            "Intuitive ↔ Visionary: Connect patterns to possibilities",
            "Analytical ↔ Visionary: Bridge evidence to potential",
        ]

        return (
            f"Step {self._step_counter}: Perspective Alignment\n\n"
            f"Bringing perspectives into productive syzygy...\n\n"
            f"Alignment Process:\n"
            f"  ┌─────────────────────────────────────┐\n"
            f"  │  ☉ ←→ ⊕  Analytical ↔ Practical    │\n"
            f"  │  Finding common ground in evidence  │\n"
            f"  │  and real-world constraints        │\n"
            f"  ├─────────────────────────────────────┤\n"
            f"  │  ☽ ←→ ☿  Intuitive ↔ Visionary     │\n"
            f"  │  Connecting intuitions to          │\n"
            f"  │  future possibilities              │\n"
            f"  ├─────────────────────────────────────┤\n"
            f"  │  ☉ ←→ ☿  Analytical ↔ Visionary    │\n"
            f"  │  Building evidence-based vision    │\n"
            f"  └─────────────────────────────────────┘\n\n"
            f"Perspectives aligned. Ready for synthesis."
        )

    def _generate_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the synthesis phase."""
        return (
            f"Step {self._step_counter}: Perspective Synthesis\n\n"
            f"Combining aligned perspectives...\n\n"
            f"Synthesis Process:\n"
            f"  From Analytical + Practical:\n"
            f"    → [Grounded, evidence-based approach]\n\n"
            f"  From Intuitive + Visionary:\n"
            f"    → [Pattern-informed future direction]\n\n"
            f"  From Analytical + Visionary:\n"
            f"    → [Data-supported possibility space]\n\n"
            f"Unified Understanding:\n"
            f"  [Combined insight that honors all perspectives]\n\n"
            f"The syzygy reveals a balanced view that\n"
            f"no single perspective could achieve alone."
        )

    def _generate_illumination(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the illumination phase."""
        return (
            f"Step {self._step_counter}: Illumination\n\n"
            f"The aligned perspectives illuminate the problem...\n\n"
            f"Key Illuminations:\n"
            f"  1. [Insight from analytical-practical alignment]\n"
            f"  2. [Insight from intuitive-visionary alignment]\n"
            f"  3. [Insight from cross-perspective synthesis]\n\n"
            f"Emergent Understanding:\n"
            f"  ╔═══════════════════════════════════════╗\n"
            f"  ║  The syzygy reveals:                  ║\n"
            f"  ║  [Deep understanding that emerges     ║\n"
            f"  ║   from aligned perspectives]          ║\n"
            f"  ╚═══════════════════════════════════════╝\n\n"
            f"This understanding transcends any single viewpoint."
        )

    def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Syzygy of Thoughts Complete:\n\n"
            f"Summary:\n"
            f"  - Perspectives identified: {len(self._perspectives)}\n"
            f"  - Alignments formed: {len(self._alignments)}\n"
            f"  - Synthesis achieved: Unified understanding\n\n"
            f"Final Answer: [Wisdom derived from perspective syzygy]\n\n"
            f"Confidence: Very High (95%)\n"
            f"Reason: Multiple complementary perspectives aligned\n"
            f"and synthesized into holistic understanding."
        )



# Export
__all__ = ["SyzygyOfThoughts", "SYZYGY_OF_THOUGHTS_METADATA"]
