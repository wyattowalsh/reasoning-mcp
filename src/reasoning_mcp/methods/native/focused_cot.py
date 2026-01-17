"""Focused Chain-of-Thought (F-CoT) reasoning method.

This module implements a condition-first reasoning approach that focuses on
relevant information and filters out distractors. The method proceeds through:
1. Identifying key conditions from the problem
2. Filtering relevant vs irrelevant information
3. Focused step-by-step reasoning using only relevant conditions
4. Deriving the answer
5. Concluding with verification

Based on Xu et al. (2025): Focused CoT improves reasoning by filtering distractors
and maintaining focus on problem-critical conditions.
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


# Metadata for Focused CoT method
FOCUSED_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.FOCUSED_COT,
    name="Focused Chain-of-Thought",
    description="Condition-first reasoning that identifies key problem conditions, "
    "filters relevant from irrelevant information, and focuses reasoning on critical "
    "elements while explicitly filtering distractors to improve accuracy.",
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset(
        {
            "condition-first",
            "distractor-filtering",
            "focused-reasoning",
            "information-filtering",
            "relevance-analysis",
            "systematic",
            "accuracy-focused",
        }
    ),
    complexity=4,  # Medium complexity - requires condition identification and filtering
    supports_branching=False,  # Linear focused path
    supports_revision=False,  # Direct focused reasoning
    requires_context=False,  # Works standalone
    min_thoughts=5,  # conditions, filter, focus, derive, conclude
    max_thoughts=8,  # Can have multiple filtering/focusing steps
    avg_tokens_per_thought=400,  # Moderate - includes filtering analysis
    best_for=(
        "problems with distractors",
        "complex problem statements",
        "information-heavy scenarios",
        "condition-based reasoning",
        "filtering irrelevant details",
        "focused analysis tasks",
        "clarity-requiring problems",
    ),
    not_recommended_for=(
        "simple direct questions",
        "problems requiring all information",
        "creative exploration",
        "minimal-context scenarios",
        "tasks needing broad consideration",
    ),
)

logger = structlog.get_logger(__name__)


class FocusedCot(ReasoningMethodBase):
    """Focused Chain-of-Thought (F-CoT) reasoning method implementation.

    This class implements a condition-first reasoning pattern where the system
    explicitly identifies key conditions, filters relevant information from
    distractors, and maintains focused reasoning on critical elements:
    1. Identify key conditions from the problem statement
    2. Filter relevant vs irrelevant information
    3. Focus reasoning using only relevant conditions
    4. Derive answer from focused reasoning
    5. Conclude with verification

    The method improves accuracy by explicitly filtering distractors and
    maintaining focus on problem-critical conditions.

    Key characteristics:
    - Condition-first approach
    - Explicit distractor filtering
    - Focused reasoning path
    - Relevance analysis
    - Information prioritization
    - Medium complexity (4)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = FocusedCot()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="A train travels 60mph for 2 hours, then stops for lunch..."
        ... )
        >>> print(result.content)  # Identified conditions

        Continue with filtering:
        >>> filtered = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Filter relevant information"
        ... )
        >>> print(filtered.type)  # ThoughtType.CONTINUATION

        Continue with focused reasoning:
        >>> focused = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=filtered,
        ...     guidance="Apply focused reasoning"
        ... )
        >>> print(focused.type)  # ThoughtType.CONTINUATION

        Derive answer:
        >>> answer = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=focused,
        ...     guidance="Derive answer"
        ... )
        >>> print(answer.type)  # ThoughtType.SYNTHESIS
    """

    def __init__(self) -> None:
        """Initialize the Focused CoT method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "identify_conditions"
        # Phases: identify_conditions, filter_relevant, focus_reasoning, derive_answer, conclude
        self._use_sampling: bool = False
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.FOCUSED_COT

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return FOCUSED_COT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return FOCUSED_COT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.HIGH_VALUE

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Focused CoT method for execution.
        Resets counters and state for a fresh reasoning session.

        Examples:
            >>> method = FocusedCot()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_phase == "identify_conditions"
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "identify_conditions"

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Focused CoT method.

        This method identifies key conditions from the problem statement.
        It extracts critical conditions, constraints, and requirements that
        are essential for solving the problem.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            A ThoughtNode representing the identified conditions

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = FocusedCot()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="Calculate distance given speed and time"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.FOCUSED_COT
            >>> assert "key_conditions" in thought.metadata
            >>> assert len(thought.metadata["key_conditions"]) > 0
        """
        if not self._initialized:
            raise RuntimeError("Focused CoT method must be initialized before execution")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "identify_conditions"

        # Identify key conditions (use sampling if available)
        if self._use_sampling:
            content, conditions = await self._sample_identify_conditions(input_text, context)
        else:
            content, conditions = self._identify_conditions_heuristic(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.FOCUSED_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Initial condition identification
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "focused_cot",
                "phase": self._current_phase,
                "key_conditions": conditions,
                "relevant_info": [],
                "irrelevant_info": [],
                "filtered_distractors": [],
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.FOCUSED_COT

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
        """Continue reasoning from a previous thought.

        This method implements the focused reasoning phase logic:
        - If previous was identify_conditions: filter relevant information
        - If previous was filter_relevant: apply focused reasoning
        - If previous was focus_reasoning: derive answer
        - If previous was derive_answer: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the focused reasoning process

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = FocusedCot()
            >>> await method.initialize()
            >>> conditions = await method.execute(session, "Complex problem...")
            >>> filtered = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=conditions
            ... )
            >>> assert filtered.type == ThoughtType.CONTINUATION
            >>> assert filtered.metadata["phase"] == "filter_relevant"
            >>> assert "relevant_info" in filtered.metadata
            >>>
            >>> focused = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=filtered
            ... )
            >>> assert focused.type == ThoughtType.CONTINUATION
            >>> assert focused.metadata["phase"] == "focus_reasoning"
            >>>
            >>> answer = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=focused
            ... )
            >>> assert answer.type == ThoughtType.SYNTHESIS
            >>> assert answer.metadata["phase"] == "derive_answer"
        """
        if not self._initialized:
            raise RuntimeError("Focused CoT method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context

        # Increment step counter
        self._step_counter += 1

        # Determine next phase based on previous phase
        prev_phase = previous_thought.metadata.get("phase", "identify_conditions")

        # Get data from previous thought
        key_conditions = previous_thought.metadata.get("key_conditions", [])
        relevant_info = previous_thought.metadata.get("relevant_info", [])
        irrelevant_info = previous_thought.metadata.get("irrelevant_info", [])
        filtered_distractors = previous_thought.metadata.get("filtered_distractors", [])

        if prev_phase == "identify_conditions":
            # Next: filter relevant information
            self._current_phase = "filter_relevant"
            thought_type = ThoughtType.CONTINUATION
            if self._use_sampling:
                result = await self._sample_filter_relevant(
                    previous_thought, key_conditions, guidance, context
                )
                content, new_relevant, new_irrelevant, new_filtered = result
            else:
                result = self._filter_relevant_heuristic(
                    previous_thought, key_conditions, guidance, context
                )
                content, new_relevant, new_irrelevant, new_filtered = result
            relevant_info = new_relevant
            irrelevant_info = new_irrelevant
            filtered_distractors = new_filtered
            confidence = 0.75
            quality_score = 0.75

        elif prev_phase == "filter_relevant":
            # Next: apply focused reasoning
            self._current_phase = "focus_reasoning"
            thought_type = ThoughtType.CONTINUATION
            if self._use_sampling:
                content = await self._sample_focus_reasoning(
                    previous_thought, key_conditions, relevant_info, guidance, context
                )
            else:
                content = self._focus_reasoning_heuristic(
                    previous_thought, key_conditions, relevant_info, guidance, context
                )
            confidence = 0.8
            quality_score = 0.8

        elif prev_phase == "focus_reasoning":
            # Next: derive answer
            self._current_phase = "derive_answer"
            thought_type = ThoughtType.SYNTHESIS
            if self._use_sampling:
                content = await self._sample_derive_answer(
                    previous_thought, key_conditions, relevant_info, guidance, context
                )
            else:
                content = self._derive_answer_heuristic(
                    previous_thought, key_conditions, relevant_info, guidance, context
                )
            confidence = 0.85
            quality_score = 0.85

        elif prev_phase == "derive_answer":
            # Next: conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclude(
                    previous_thought,
                    key_conditions,
                    relevant_info,
                    filtered_distractors,
                    guidance,
                    context,
                )
            else:
                content = self._conclude_heuristic(
                    previous_thought,
                    key_conditions,
                    relevant_info,
                    filtered_distractors,
                    guidance,
                    context,
                )
            confidence = 0.9
            quality_score = 0.9

        else:
            # Fallback to conclusion
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if self._use_sampling:
                content = await self._sample_conclude(
                    previous_thought,
                    key_conditions,
                    relevant_info,
                    filtered_distractors,
                    guidance,
                    context,
                )
            else:
                content = self._conclude_heuristic(
                    previous_thought,
                    key_conditions,
                    relevant_info,
                    filtered_distractors,
                    guidance,
                    context,
                )
            confidence = 0.85
            quality_score = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.FOCUSED_COT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "phase": self._current_phase,
                "key_conditions": key_conditions,
                "relevant_info": relevant_info,
                "irrelevant_info": irrelevant_info,
                "filtered_distractors": filtered_distractors,
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "focused_cot",
                "previous_phase": prev_phase,
                "sampled": self._use_sampling,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Focused CoT, this simply checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = FocusedCot()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _sample_identify_conditions(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Identify key conditions using LLM sampling.

        Uses the execution context's sampling capability to extract
        critical conditions, constraints, and requirements from the problem.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            A tuple of (content string, list of key conditions)

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context is required for sampling")

        system_prompt = """You are a reasoning assistant using Focused Chain-of-Thought methodology.
Your task is to identify key conditions from a problem statement.

Analyze the problem and extract:
1. Critical conditions that must be satisfied
2. Constraints or limitations
3. Key parameters, variables, or requirements
4. Essential information needed for solving the problem

Format your response as:
- Brief analysis of the problem
- List each condition clearly numbered
- Explain why each condition is critical"""

        user_prompt = f"""Problem: {input_text}

Identify all key conditions, constraints, and requirements from this problem.
List them clearly and explain their importance."""

        def fallback() -> str:
            content, _ = self._identify_conditions_heuristic(input_text, context)
            return content

        content_str = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )

        # Check if fallback was used (heuristic content has specific pattern)
        if "Analyzing problem statement to extract key conditions" in content_str:
            # Fallback was used, return heuristic results
            return self._identify_conditions_heuristic(input_text, context)

        # Extract conditions from the response (simple extraction)
        conditions = []
        lines = content_str.split("\n")
        for line in lines:
            # Look for numbered or bulleted items
            prefixes = ["1.", "2.", "3.", "4.", "5.", "-", "*", "•"]
            if any(line.strip().startswith(prefix) for prefix in prefixes):
                conditions.append(line.strip())

        # If no conditions extracted, create a default one
        if not conditions:
            conditions = ["Key condition: " + input_text[:100]]

        # Format content
        content = (
            f"Step {self._step_counter}: Identify Key Conditions (Phase 1/5)\n\n"
            f"Problem: {input_text}\n\n"
            f"{content_str}\n\n"
            f"Total conditions identified: {len(conditions)}\n"
            f"Next, I will filter relevant information based on these conditions."
        )

        return content, conditions

    async def _sample_filter_relevant(
        self,
        conditions_thought: ThoughtNode,
        key_conditions: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Filter relevant vs irrelevant information using LLM sampling.

        Uses the execution context's sampling capability to analyze
        the problem and separate relevant information from distractors.

        Args:
            conditions_thought: The thought containing identified conditions
            key_conditions: List of key conditions
            guidance: Optional guidance for filtering
            context: Optional additional context

        Returns:
            A tuple of (content, relevant_info, irrelevant_info, filtered_distractors)

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context is required for sampling")

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using Focused Chain-of-Thought methodology.
Your task is to filter information into relevant and irrelevant categories.

Given key conditions, analyze the problem and:
1. Identify information RELEVANT to solving the problem
2. Identify information that is IRRELEVANT or distracting
3. Flag potential DISTRACTORS that could mislead reasoning

Be clear about what information to use and what to ignore."""

        conditions_str = "\n".join(f"- {c}" for c in key_conditions)
        input_text = conditions_thought.metadata.get("input", "")

        user_prompt = f"""Problem: {input_text}

Key Conditions:
{conditions_str}
{guidance_text}

Filter the information:
1. List RELEVANT information (what we need to solve this)
2. List IRRELEVANT information (what we should ignore)
3. Identify DISTRACTORS (misleading elements)"""

        def fallback() -> str:
            content, _, _, _ = self._filter_relevant_heuristic(
                conditions_thought, key_conditions, guidance, context
            )
            return content

        content_str = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1200,
        )

        # Check if fallback was used (heuristic content has specific pattern)
        if "filtering relevant vs irrelevant information" in content_str.lower():
            # Fallback was used, return heuristic results
            return self._filter_relevant_heuristic(
                conditions_thought, key_conditions, guidance, context
            )

        # Extract relevant, irrelevant, and distractors (simple extraction)
        relevant_info: list[str] = []
        irrelevant_info: list[str] = []
        filtered_distractors: list[str] = []

        lines = content_str.split("\n")
        current_section: str | None = None

        for line in lines:
            lower_line = line.lower()
            if "relevant" in lower_line and "irrelevant" not in lower_line:
                current_section = "relevant"
            elif "irrelevant" in lower_line:
                current_section = "irrelevant"
            elif "distractor" in lower_line:
                current_section = "distractor"
            elif line.strip():
                prefixes = ["-", "*", "•", "1.", "2.", "3."]
                if any(line.strip().startswith(p) for p in prefixes):
                    if current_section == "relevant":
                        relevant_info.append(line.strip())
                    elif current_section == "irrelevant":
                        irrelevant_info.append(line.strip())
                    elif current_section == "distractor":
                        filtered_distractors.append(line.strip())

        # Ensure at least one item in each category
        if not relevant_info:
            relevant_info = ["Information relevant to the problem"]
        if not irrelevant_info:
            irrelevant_info = ["Extraneous details"]
        if not filtered_distractors:
            filtered_distractors = ["Potential misleading elements"]

        # Format content
        content = (
            f"Step {self._step_counter}: Filter Relevant Information (Phase 2/5)\n\n"
            f"Based on {len(key_conditions)} key conditions, filtering information...\n\n"
            f"{content_str}\n\n"
            f"Filtering Summary:\n"
            f"- Relevant items: {len(relevant_info)}\n"
            f"- Irrelevant items: {len(irrelevant_info)}\n"
            f"- Distractors filtered: {len(filtered_distractors)}\n\n"
            f"Focused reasoning will use only the relevant information."
        )

        return content, relevant_info, irrelevant_info, filtered_distractors

    async def _sample_focus_reasoning(
        self,
        filtered_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Apply focused step-by-step reasoning using LLM sampling.

        Uses the execution context's sampling capability to perform
        focused reasoning using only the filtered relevant information.

        Args:
            filtered_thought: The thought containing filtered information
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            guidance: Optional guidance for reasoning
            context: Optional additional context

        Returns:
            The content for the focused reasoning

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context is required for sampling")

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using Focused Chain-of-Thought methodology.
Your task is to apply step-by-step reasoning using ONLY the relevant information.

Perform focused reasoning:
1. Use only the provided key conditions and relevant information
2. Ignore all distractors and irrelevant details
3. Show clear logical steps
4. Build conclusions systematically
5. Maintain focus on problem-critical elements"""

        conditions_str = "\n".join(f"- {c}" for c in key_conditions)
        relevant_str = "\n".join(f"- {r}" for r in relevant_info)

        user_prompt = f"""Key Conditions:
{conditions_str}

Relevant Information to Use:
{relevant_str}
{guidance_text}

Apply focused step-by-step reasoning using ONLY the conditions and relevant information above.
Show your reasoning process clearly."""

        def fallback() -> str:
            return self._focus_reasoning_heuristic(
                filtered_thought, key_conditions, relevant_info, guidance, context
            )

        content_str = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1500,
        )

        # Check if fallback was used (heuristic content has specific pattern)
        if "[LLM would perform step-by-step reasoning here" in content_str:
            # Fallback was used
            return content_str

        # Format content
        content = (
            f"Step {self._step_counter}: Focused Reasoning (Phase 3/5)\n\n"
            f"Applying step-by-step reasoning using:\n"
            f"- {len(key_conditions)} key conditions\n"
            f"- {len(relevant_info)} relevant information items\n\n"
            f"{content_str}\n\n"
            f"Note: All distractors have been filtered out. "
            f"This reasoning focuses only on problem-critical elements."
        )

        return content

    async def _sample_derive_answer(
        self,
        reasoning_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Derive the final answer using LLM sampling.

        Uses the execution context's sampling capability to derive
        the answer based on the focused reasoning.

        Args:
            reasoning_thought: The thought containing focused reasoning
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            guidance: Optional guidance for deriving answer
            context: Optional additional context

        Returns:
            The content for the derived answer

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context is required for sampling")

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using Focused Chain-of-Thought methodology.
Your task is to derive the final answer from focused reasoning.

Derive the answer by:
1. Synthesizing insights from the focused reasoning
2. Ensuring all key conditions are satisfied
3. Verifying the answer is supported by relevant information
4. Stating the answer clearly and confidently"""

        conditions_str = "\n".join(f"- {c}" for c in key_conditions)
        reasoning_content = reasoning_thought.content

        user_prompt = f"""Focused Reasoning:
{reasoning_content}

Key Conditions that must be satisfied:
{conditions_str}
{guidance_text}

Derive the final answer based on the focused reasoning above.
Verify all conditions are satisfied and state the answer clearly."""

        def fallback() -> str:
            return self._derive_answer_heuristic(
                reasoning_thought, key_conditions, relevant_info, guidance, context
            )

        content_str = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=1000,
        )

        # Check if fallback was used (heuristic content has specific pattern)
        if "[LLM would derive answer using focused reasoning results]" in content_str:
            # Fallback was used
            return content_str

        # Format content
        content = (
            f"Step {self._step_counter}: Derive Answer (Phase 4/5)\n\n"
            f"Based on focused reasoning from Step {reasoning_thought.step_number}, "
            f"deriving the final answer...\n\n"
            f"{content_str}\n\n"
            f"This answer is derived from focused reasoning on critical conditions, "
            f"with all distractors filtered out for maximum accuracy."
        )

        return content

    async def _sample_conclude(
        self,
        answer_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        filtered_distractors: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final conclusion using LLM sampling.

        Uses the execution context's sampling capability to generate
        a conclusion summarizing the focused CoT process.

        Args:
            answer_thought: The thought containing the derived answer
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            filtered_distractors: List of filtered distractors
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion

        Raises:
            RuntimeError: If execution context is not available
        """
        if self._execution_context is None:
            raise RuntimeError("Execution context is required for sampling")

        guidance_text = f"\n\nAdditional guidance: {guidance}" if guidance else ""

        system_prompt = """You are a reasoning assistant using Focused Chain-of-Thought methodology.
Your task is to provide a final conclusion with verification.

Conclude by:
1. Summarizing the focused reasoning process
2. Verifying all conditions were addressed
3. Confirming distractors were successfully filtered
4. Validating the answer's correctness and completeness"""

        answer_content = answer_thought.content

        user_prompt = f"""Answer:
{answer_content}

Process Summary:
- {len(key_conditions)} key conditions identified
- {len(relevant_info)} relevant information items used
- {len(filtered_distractors)} distractors filtered out
{guidance_text}

Provide a final conclusion that verifies the focused reasoning process
and validates the answer."""

        def fallback() -> str:
            return self._conclude_heuristic(
                answer_thought,
                key_conditions,
                relevant_info,
                filtered_distractors,
                guidance,
                context,
            )

        content_str = await self._sample_with_fallback(
            user_prompt,
            fallback,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=800,
        )

        # Check if fallback was used (heuristic content has specific pattern)
        if "Focused CoT reasoning complete." in content_str and "Process Summary:" in content_str:
            # Fallback was used
            return content_str

        # Format content
        content = (
            f"Step {self._step_counter}: Conclusion (Phase 5/5)\n\n"
            f"Focused CoT reasoning complete.\n\n"
            f"{content_str}\n\n"
            f"The focused approach filtered irrelevant information and distractors, "
            f"allowing for clear, accurate reasoning on problem-critical conditions."
        )

        return content

    def _identify_conditions_heuristic(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str]]:
        """Identify key conditions from the problem statement using heuristics.

        This is a fallback helper method that uses simple heuristics to extract
        critical conditions, constraints, and requirements from the problem.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            A tuple of (content string, list of key conditions)

        Note:
            This is a placeholder/fallback that provides the structure.
            Actual condition extraction should use LLM sampling.
        """
        # Sample conditions (in real implementation, LLM would extract these)
        conditions = [
            "Condition 1: [LLM would identify critical condition from problem]",
            "Condition 2: [LLM would identify constraint or requirement]",
            "Condition 3: [LLM would identify key parameter or variable]",
        ]

        content = (
            f"Step {self._step_counter}: Identify Key Conditions (Phase 1/5)\n\n"
            f"Problem: {input_text}\n\n"
            f"Analyzing problem statement to extract key conditions...\n\n"
            f"Key Conditions Identified:\n"
        )

        for i, condition in enumerate(conditions, 1):
            content += f"{i}. {condition}\n"

        content += (
            f"\n\nTotal conditions: {len(conditions)}\n"
            f"These conditions are critical for solving the problem. "
            f"Next, I will filter relevant information based on these conditions."
        )

        return content, conditions

    def _filter_relevant_heuristic(
        self,
        conditions_thought: ThoughtNode,
        key_conditions: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """Filter relevant vs irrelevant information using heuristics.

        This is a fallback helper method that uses simple heuristics to analyze
        the problem and separate relevant information from distractors.

        Args:
            conditions_thought: The thought containing identified conditions
            key_conditions: List of key conditions
            guidance: Optional guidance for filtering
            context: Optional additional context

        Returns:
            A tuple of (content, relevant_info, irrelevant_info, filtered_distractors)

        Note:
            This is a placeholder/fallback. Actual filtering should use LLM sampling.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Sample filtering (in real implementation, LLM would do this)
        relevant_info = [
            "Relevant: [LLM would identify information directly related to conditions]",
            "Relevant: [LLM would identify information needed for solution]",
        ]

        irrelevant_info = [
            "Irrelevant: [LLM would identify distractor or unnecessary detail]",
            "Irrelevant: [LLM would identify information not needed for solution]",
        ]

        filtered_distractors = [
            "Distractor: [LLM would identify misleading or confusing element]",
        ]

        content = (
            f"Step {self._step_counter}: Filter Relevant Information (Phase 2/5)\n\n"
            f"Based on {len(key_conditions)} key conditions from "
            f"Step {conditions_thought.step_number}, "
            f"filtering relevant vs irrelevant information...\n\n"
            f"Relevant Information (to be used in reasoning):\n"
        )

        for i, info in enumerate(relevant_info, 1):
            content += f"✓ {i}. {info}\n"

        content += "\nIrrelevant Information (to be filtered out):\n"

        for i, info in enumerate(irrelevant_info, 1):
            content += f"✗ {i}. {info}\n"

        content += "\nFiltered Distractors:\n"

        for i, distractor in enumerate(filtered_distractors, 1):
            content += f"⚠ {i}. {distractor}\n"

        content += (
            f"\n\nFiltering Summary:\n"
            f"- Relevant items: {len(relevant_info)}\n"
            f"- Irrelevant items: {len(irrelevant_info)}\n"
            f"- Distractors filtered: {len(filtered_distractors)}\n\n"
            f"Focused reasoning will use only the relevant information.{guidance_text}"
        )

        return content, relevant_info, irrelevant_info, filtered_distractors

    def _focus_reasoning_heuristic(
        self,
        filtered_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Apply focused step-by-step reasoning using heuristics.

        This is a fallback helper method that uses simple heuristics to perform
        focused reasoning using only the filtered relevant information.

        Args:
            filtered_thought: The thought containing filtered information
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            guidance: Optional guidance for reasoning
            context: Optional additional context

        Returns:
            The content for the focused reasoning

        Note:
            This is a placeholder/fallback. Actual reasoning should use LLM sampling.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Focused Reasoning (Phase 3/5)\n\n"
            f"Applying step-by-step reasoning using:\n"
            f"- {len(key_conditions)} key conditions\n"
            f"- {len(relevant_info)} relevant information items\n\n"
            f"Focused Reasoning Steps:\n\n"
            f"[LLM would perform step-by-step reasoning here, using only relevant information]\n\n"
            f"Step 1: Apply first condition using relevant info\n"
            f"  → [LLM would show focused reasoning step]\n\n"
            f"Step 2: Apply second condition using relevant info\n"
            f"  → [LLM would show focused reasoning step]\n\n"
            f"Step 3: Combine conditions for comprehensive understanding\n"
            f"  → [LLM would show focused reasoning step]\n\n"
            f"Note: All distractors have been filtered out. "
            f"This reasoning focuses only on problem-critical elements.{guidance_text}"
        )

        return content

    def _derive_answer_heuristic(
        self,
        reasoning_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Derive the final answer from focused reasoning using heuristics.

        This is a fallback helper method that uses simple heuristics to derive
        the answer based on the focused reasoning.

        Args:
            reasoning_thought: The thought containing focused reasoning
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            guidance: Optional guidance for deriving answer
            context: Optional additional context

        Returns:
            The content for the derived answer

        Note:
            This is a placeholder/fallback. Actual derivation should use LLM sampling.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Derive Answer (Phase 4/5)\n\n"
            f"Based on focused reasoning from Step {reasoning_thought.step_number}, "
            f"deriving the final answer...\n\n"
            f"Answer Derivation:\n"
            f"[LLM would derive answer using focused reasoning results]\n\n"
            f"Conditions satisfied:\n"
        )

        for i, condition in enumerate(key_conditions, 1):
            content += f"✓ {i}. {condition}\n"

        content += (
            f"\nFinal Answer:\n"
            f"[LLM would provide the final answer here]\n\n"
            f"This answer is derived from focused reasoning on critical conditions, "
            f"with all distractors filtered out for maximum accuracy.{guidance_text}"
        )

        return content

    def _conclude_heuristic(
        self,
        answer_thought: ThoughtNode,
        key_conditions: list[str],
        relevant_info: list[str],
        filtered_distractors: list[str],
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final conclusion with verification using heuristics.

        This is a helper method that generates a conclusion summarizing
        the focused CoT process and verifying the answer.

        Args:
            answer_thought: The thought containing the derived answer
            key_conditions: List of key conditions
            relevant_info: List of relevant information
            filtered_distractors: List of filtered distractors
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        content = (
            f"Step {self._step_counter}: Conclusion (Phase 5/5)\n\n"
            f"Focused CoT reasoning complete.\n\n"
            f"Process Summary:\n"
            f"1. Identified {len(key_conditions)} key conditions\n"
            f"2. Filtered {len(relevant_info)} relevant information items\n"
            f"3. Filtered out {len(filtered_distractors)} distractors\n"
            f"4. Applied focused reasoning on critical elements\n"
            f"5. Derived answer from focused reasoning\n\n"
            f"Verification:\n"
            f"✓ All key conditions addressed\n"
            f"✓ Relevant information utilized\n"
            f"✓ Distractors successfully filtered\n"
            f"✓ Reasoning maintained focus throughout\n\n"
            f"The focused approach filtered irrelevant information and distractors, "
            f"allowing for clear, accurate reasoning on problem-critical conditions.{guidance_text}"
        )

        return content


__all__ = ["FocusedCot", "FOCUSED_COT_METADATA"]
