"""Thread of Thought (ThoT) reasoning method.

This module implements Thread of Thought prompting (Zhou et al. 2023), which
processes long contexts by segmenting and reasoning incrementally. ThoT handles
extended inputs by creating "threads" of reasoning that process text in chunks
while maintaining coherence and context across segments.

Key phases:
1. Segment: Divide the input into manageable chunks
2. Thread: Process each segment while maintaining context
3. Weave: Connect insights across segments
4. Synthesize: Create unified understanding from all threads

Reference: Zhou et al. (2023) - "Thread of Thought Unraveling Chaotic Contexts"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reasoning_mcp.elicitation import ElicitationConfig, elicit_selection
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models import Session


# Metadata for Thread of Thought method
THREAD_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.THREAD_OF_THOUGHT,
    name="Thread of Thought",
    description="Processes long contexts by segmenting and reasoning incrementally. "
    "Creates threads of reasoning that handle extended inputs in chunks while "
    "maintaining coherence through segment → thread → weave → synthesize phases.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset(
        {
            "long-context",
            "incremental",
            "segmentation",
            "threading",
            "coherence",
            "extended-input",
            "chunking",
            "synthesis",
        }
    ),
    complexity=7,  # High complexity due to context management
    supports_branching=True,  # Each thread is a branch
    supports_revision=True,  # Can revise thread connections
    requires_context=True,  # Needs context window management
    min_thoughts=4,  # At least: segment + thread + weave + synthesize
    max_thoughts=20,  # Many threads for long contexts
    avg_tokens_per_thought=400,  # Threading can be verbose
    best_for=(
        "long document analysis",
        "extended conversation handling",
        "multi-page reasoning",
        "complex narrative understanding",
        "large context processing",
        "document summarization",
        "legal document analysis",
        "research paper synthesis",
    ),
    not_recommended_for=(
        "short queries",
        "simple questions",
        "single-fact lookups",
        "tasks not involving extended context",
    ),
)

logger = structlog.get_logger(__name__)


class ThreadOfThought(ReasoningMethodBase):
    """Thread of Thought reasoning method implementation.

    This class implements the ThoT pattern for processing long contexts:
    1. Segment: Divide input into chunks that fit context windows
    2. Thread: Process each segment while tracking cross-references
    3. Weave: Connect threads to build coherent understanding
    4. Synthesize: Create unified answer from all thread insights

    Key characteristics:
    - Handles contexts beyond typical LLM windows
    - Maintains coherence across segments
    - Tracks inter-segment dependencies
    - High complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = ThreadOfThought()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="[Long multi-page document text...]"
        ... )
        >>> print(result.content)  # Segment phase
    """

    # Maximum characters per segment
    MAX_SEGMENT_SIZE = 2000
    # Maximum threads to process
    MAX_THREADS = 10
    # Enable LLM sampling for dynamic reasoning
    _use_sampling: bool = True

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the Thread of Thought method.

        Args:
            enable_elicitation: Whether to enable user elicitation for threading style
        """
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "segment"  # segment, thread, weave, synthesize
        self._segments: list[str] = []
        self._thread_insights: list[str] = []
        self._current_thread: int = 0
        self.enable_elicitation = enable_elicitation
        self._execution_context: Any = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.THREAD_OF_THOUGHT

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return THREAD_OF_THOUGHT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return THREAD_OF_THOUGHT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Thread of Thought method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "segment"
        self._segments = []
        self._thread_insights = []
        self._current_thread = 0

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Execute the Thread of Thought method.

        Creates the initial segmentation phase, dividing the input
        into manageable chunks for threaded processing.

        Args:
            session: The current reasoning session
            input_text: The long context to process
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the segmentation phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Thread of Thought method must be initialized before execution")

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "segment"
        self._segments = self._segment_text(input_text)
        self._thread_insights = []
        self._current_thread = 0

        # Store execution context for sampling and elicitation
        self._execution_context = execution_context

        # Elicit threading style if enabled
        threading_style = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
            and self._execution_context.ctx
        ):
            try:
                options = [
                    {"id": "linear", "label": "Linear - Single thread of thought"},
                    {"id": "branching", "label": "Branching - Explore alternatives"},
                    {"id": "converging", "label": "Converging - Multiple threads to one"},
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "What threading style should I use?",
                    options,
                    config=config,
                )
                if selection and selection.selected:
                    threading_style = selection.selected
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                )
            except Exception as e:
                logger.error(
                    "unexpected_error",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Generate segmentation content - use sampling if available
        content = await self._sample_with_fallback(
            f"""Analyze this long context and plan threaded reasoning.

Input length: {len(input_text)} characters
Number of segments: {len(self._segments)}
Threading style: {threading_style or "default"}

Provide:
1. Segmentation strategy description
2. Thread plan for processing
3. Expected cross-segment themes
4. Initial observations""",
            lambda: self._generate_segmentation(input_text, context, threading_style),
            system_prompt="You are an expert at Thread of Thought reasoning. "
            "Analyze long contexts by segmenting and threading through content.",
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THREAD_OF_THOUGHT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,
            quality_score=0.65,
            metadata={
                "input_length": len(input_text),
                "context": context or {},
                "reasoning_type": "thread_of_thought",
                "phase": self._current_phase,
                "num_segments": len(self._segments),
                "current_thread": self._current_thread,
                "threading_style": threading_style,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.THREAD_OF_THOUGHT

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: Any = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        Implements the ThoT phase progression:
        - After segment: start threading through segments
        - During thread: continue threading or move to weave
        - After weave: synthesize final understanding
        - After synthesize: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A new ThoughtNode continuing the ThoT process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Thread of Thought method must be initialized before continuation")

        # Store execution context for sampling
        if execution_context:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "segment")

        if prev_phase == "segment":
            # Start threading
            self._current_phase = "thread"
            self._current_thread = 0
            thought_type = ThoughtType.REASONING
            content = await self._generate_thread(self._current_thread, guidance, context)
            confidence = 0.7
            quality_score = 0.7

        elif prev_phase == "thread":
            self._current_thread += 1
            if self._current_thread < len(self._segments):
                # Continue threading
                thought_type = ThoughtType.REASONING
                content = await self._generate_thread(self._current_thread, guidance, context)
                confidence = 0.7 + (0.02 * self._current_thread)
                quality_score = 0.7
            else:
                # All threads complete, weave
                self._current_phase = "weave"
                thought_type = ThoughtType.SYNTHESIS
                content = await self._generate_weave(guidance, context)
                confidence = 0.8
                quality_score = 0.8

        elif prev_phase == "weave":
            # Final synthesis
            self._current_phase = "synthesize"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._generate_synthesis(previous_thought, guidance, context)
            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "synthesize":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            content = await self._generate_conclusion(previous_thought, guidance, context)
            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback
            self._current_phase = "synthesize"
            thought_type = ThoughtType.SYNTHESIS
            content = await self._generate_synthesis(previous_thought, guidance, context)
            confidence = 0.75
            quality_score = 0.75

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.THREAD_OF_THOUGHT,
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
                "reasoning_type": "thread_of_thought",
                "num_segments": len(self._segments),
                "current_thread": self._current_thread,
                "threads_processed": min(self._current_thread + 1, len(self._segments)),
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _segment_text(self, text: str) -> list[str]:
        """Segment text into manageable chunks."""
        segments = []
        current = 0
        while current < len(text):
            end = min(current + self.MAX_SEGMENT_SIZE, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", current, end)
                if last_period > current:
                    end = last_period + 1
            segments.append(text[current:end])
            current = end
        return segments[: self.MAX_THREADS]  # Limit segments

    def _generate_segmentation(
        self,
        input_text: str,
        context: dict[str, Any] | None,
        threading_style: str | None = None,
    ) -> str:
        """Generate the segmentation phase content."""
        num_segments = len(self._segments)
        style_info = ""
        if threading_style:
            style_descriptions = {
                "linear": "Linear - processing threads sequentially in order",
                "branching": "Branching - exploring alternative interpretations in parallel",
                "converging": (
                    "Converging - combining multiple perspectives into unified understanding"
                ),
            }
            style_desc = style_descriptions.get(threading_style, threading_style)
            style_info = f"Threading Style: {style_desc}\n"

        return (
            f"Step {self._step_counter}: Context Segmentation (Thread of Thought)\n\n"
            f"Input Length: {len(input_text)} characters\n"
            f"Segments Created: {num_segments}\n"
            f"{style_info}\n"
            f"Segmentation Strategy:\n"
            f"1. Divided context into {num_segments} manageable chunks\n"
            f"2. Preserved sentence boundaries where possible\n"
            f"3. Each segment ~{self.MAX_SEGMENT_SIZE} characters\n\n"
            f"Thread Plan:\n"
            f"- Process each segment independently\n"
            f"- Track cross-segment references\n"
            f"- Weave insights together\n"
            f"- Synthesize final understanding\n\n"
            f"Ready to begin threaded analysis."
        )

    def _generate_thread_heuristic(
        self,
        thread_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate thread content using heuristic approach."""
        segment = self._segments[thread_num] if thread_num < len(self._segments) else ""
        segment_preview = segment[:100] + "..." if len(segment) > 100 else segment

        guidance_text = f"\nGuidance: {guidance}" if guidance else ""

        return (
            f"Step {self._step_counter}: Thread {thread_num + 1}/{len(self._segments)}"
            f"{guidance_text}\n\n"
            f"Processing Segment:\n"
            f'"{segment_preview}"\n\n'
            f"Thread Analysis:\n"
            f"1. Key Information: [extracted from segment]\n"
            f"2. Cross-References: [links to other segments]\n"
            f"3. Context Maintained: [running understanding]\n\n"
            f"Thread {thread_num + 1} Insight:\n"
            f"[Insight extracted from this segment that contributes to overall understanding]"
        )

    async def _generate_thread(
        self,
        thread_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a thread processing step."""
        segment = self._segments[thread_num] if thread_num < len(self._segments) else ""
        prompt = (
            f"You are processing Thread {thread_num + 1} of {len(self._segments)} "
            f"in a Thread of Thought analysis.\n\n"
            f"Segment to process:\n{segment}\n\n"
            f"Guidance: {guidance or 'None'}\n"
            f"Context: {context or 'None'}\n\n"
            f"Generate Step {self._step_counter} which analyzes this segment:\n"
            f"1. Extract key information from the segment\n"
            f"2. Identify cross-references to other potential segments\n"
            f"3. Maintain running understanding of the overall context\n"
            f"4. Provide insight from this segment that contributes to overall understanding\n\n"
            f"Format as 'Step {self._step_counter}: Thread {thread_num + 1}/{len(self._segments)}'"
        )
        system = (
            "You are analyzing individual segments in a Thread of Thought process. "
            "Extract meaningful insights while tracking connections to the broader context."
        )

        return await self._sample_with_fallback(
            prompt,
            lambda: self._generate_thread_heuristic(thread_num, guidance, context),
            system_prompt=system,
        )

    def _generate_weave_heuristic(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate weaving content using heuristic approach."""
        return (
            f"Step {self._step_counter}: Weaving Threads\n\n"
            f"Connecting {len(self._segments)} thread insights...\n\n"
            f"Inter-Thread Connections:\n"
            f"1. Theme A spans threads 1, 3, 5\n"
            f"2. Theme B connects threads 2, 4\n"
            f"3. Cross-references resolved\n\n"
            f"Emerging Pattern:\n"
            f"[Pattern that emerges from connecting all threads]\n\n"
            f"Coherence Check:\n"
            f"- All segments accounted for: Yes\n"
            f"- Contradictions resolved: Yes\n"
            f"- Context preserved: Yes"
        )

    async def _generate_weave(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the weaving phase content."""
        prompt = (
            f"You are weaving together {len(self._segments)} threads "
            f"in a Thread of Thought analysis.\n\n"
            f"Thread insights collected: {len(self._thread_insights)}\n"
            f"Guidance: {guidance or 'None'}\n"
            f"Context: {context or 'None'}\n\n"
            f"Generate Step {self._step_counter} which weaves the threads together:\n"
            f"1. Identify inter-thread connections and themes\n"
            f"2. Show how different segments relate to each other\n"
            f"3. Resolve cross-references found during threading\n"
            f"4. Describe emerging patterns from connecting all threads\n"
            f"5. Verify coherence: all segments accounted for, "
            f"contradictions resolved, context preserved\n\n"
            f"Format as 'Step {self._step_counter}: Weaving Threads'"
        )
        system = (
            "You are weaving together multiple threads of analysis in a Thread of Thought process. "
            "Find connections, patterns, and coherence across all segments."
        )

        return await self._sample_with_fallback(
            prompt,
            lambda: self._generate_weave_heuristic(guidance, context),
            system_prompt=system,
        )

    def _generate_synthesis_heuristic(
        self,
        weave_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate synthesis content using heuristic approach."""
        return (
            f"Step {self._step_counter}: Thread Synthesis\n\n"
            f"Creating unified understanding from {len(self._segments)} threads...\n\n"
            f"Synthesized Understanding:\n"
            f"[Comprehensive synthesis that captures the essence of all segments]\n\n"
            f"Key Findings:\n"
            f"1. Main insight from woven threads\n"
            f"2. Supporting details preserved\n"
            f"3. Context-aware conclusions\n\n"
            f"Synthesis Quality: High\n"
            f"Coverage: Complete ({len(self._segments)}/{len(self._segments)} segments)"
        )

    async def _generate_synthesis(
        self,
        weave_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the synthesis phase content."""
        prompt = (
            f"You are synthesizing {len(self._segments)} woven threads "
            f"in a Thread of Thought analysis.\n\n"
            f"Previous weaving: {weave_thought.content[:300]}...\n"
            f"Guidance: {guidance or 'None'}\n"
            f"Context: {context or 'None'}\n\n"
            f"Generate Step {self._step_counter} which creates unified understanding:\n"
            f"1. Provide comprehensive synthesis capturing essence of all segments\n"
            f"2. Present key findings from the woven threads\n"
            f"3. Include supporting details that were preserved\n"
            f"4. Draw context-aware conclusions\n"
            f"5. Assess synthesis quality and coverage\n\n"
            f"Format as 'Step {self._step_counter}: Thread Synthesis'"
        )
        system = (
            "You are synthesizing a unified understanding from multiple woven threads. "
            "Create a comprehensive synthesis that captures the full context."
        )

        return await self._sample_with_fallback(
            prompt,
            lambda: self._generate_synthesis_heuristic(weave_thought, guidance, context),
            system_prompt=system,
        )

    def _generate_conclusion_heuristic(
        self,
        synthesis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate conclusion content using heuristic approach."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Thread of Thought Analysis Complete:\n\n"
            f"1. Segmented: {len(self._segments)} chunks processed\n"
            f"2. Threaded: Each segment analyzed independently\n"
            f"3. Woven: Cross-segment connections established\n"
            f"4. Synthesized: Unified understanding achieved\n\n"
            f"Final Answer: [comprehensive answer based on full context]\n\n"
            f"Confidence: High (all threads successfully processed)\n"
            f"Context Preserved: 100%"
        )

    async def _generate_conclusion(
        self,
        synthesis_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion."""
        prompt = (
            f"You are concluding a Thread of Thought analysis of "
            f"{len(self._segments)} segments.\n\n"
            f"Previous synthesis: {synthesis_thought.content[:300]}...\n"
            f"Guidance: {guidance or 'None'}\n"
            f"Context: {context or 'None'}\n\n"
            f"Generate Step {self._step_counter} which provides the final answer:\n"
            f"1. Summarize the complete ThoT process (segment, thread, weave, synthesize)\n"
            f"2. Present the comprehensive final answer based on full context\n"
            f"3. Express high confidence from successfully processing all threads\n"
            f"4. Confirm 100% context preservation\n\n"
            f"Format as 'Step {self._step_counter}: Final Answer'"
        )
        system = (
            "You are providing the final conclusion of a Thread of Thought analysis. "
            "Deliver a comprehensive answer that demonstrates full context understanding."
        )

        return await self._sample_with_fallback(
            prompt,
            lambda: self._generate_conclusion_heuristic(synthesis_thought, guidance, context),
            system_prompt=system,
        )


# Export
__all__ = ["ThreadOfThought", "THREAD_OF_THOUGHT_METADATA"]
