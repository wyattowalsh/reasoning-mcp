"""Test-Time Scaling reasoning method.

This module implements Test-Time Scaling (OpenAI o1/o3, DeepSeek-R1 style), which
scales inference-time compute by allowing the model to "think longer" through
extended reasoning chains, search, and verification before producing outputs.

Key phases:
1. Analyze: Understand the problem and assess difficulty
2. Expand: Generate extended thinking with search and exploration
3. Verify: Check reasoning paths for correctness
4. Synthesize: Combine insights into final answer

Reference: DeepSeek-R1 (2025), OpenAI o1/o3 (2024-2025) - Test-Time Compute Scaling
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


# Metadata for Test-Time Scaling method
TEST_TIME_SCALING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.TEST_TIME_SCALING,
    name="Test-Time Scaling",
    description="Scales inference-time compute through extended thinking, search, and "
    "verification. Allows the model to 'think longer' on difficult problems by "
    "exploring multiple paths and self-correcting before finalizing answers.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "test-time-compute",
            "extended-thinking",
            "search",
            "verification",
            "scaling",
            "o1-style",
            "deepseek-r1",
            "inference-compute",
        }
    ),
    complexity=9,  # High complexity - sophisticated scaling mechanism
    supports_branching=True,  # Explores multiple paths
    supports_revision=True,  # Self-correction during thinking
    requires_context=False,  # No special context needed
    min_thoughts=6,  # Extended thinking requires more steps
    max_thoughts=20,  # Can scale to many thoughts
    avg_tokens_per_thought=450,  # Detailed reasoning per step
    best_for=(
        "complex reasoning problems",
        "mathematical proofs",
        "difficult puzzles",
        "multi-step planning",
        "problems requiring deep analysis",
        "competition-level problems",
        "hard coding challenges",
        "research-level questions",
    ),
    not_recommended_for=(
        "simple queries",
        "quick lookups",
        "time-sensitive responses",
        "low-complexity tasks",
    ),
)

logger = structlog.get_logger(__name__)


class TestTimeScaling(ReasoningMethodBase):
    """Test-Time Scaling reasoning method implementation.

    This class implements inference-time compute scaling:
    1. Analyze: Assess problem difficulty and plan thinking budget
    2. Expand: Generate extended chain-of-thought with exploration
    3. Verify: Check reasoning paths and correct errors
    4. Synthesize: Combine verified insights into answer

    Key characteristics:
    - Extended thinking time
    - Multiple exploration paths
    - Self-verification and correction
    - Compute budget awareness
    - High complexity (9)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = TestTimeScaling()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Prove that there are infinitely many primes"
        ... )
        >>> print(result.content)  # Extended analysis phase
    """

    # Default thinking budget (number of exploration steps)
    DEFAULT_THINKING_BUDGET = 10
    # Maximum verification rounds
    MAX_VERIFICATIONS = 3

    def __init__(self) -> None:
        """Initialize the Test-Time Scaling method."""
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "analyze"
        self._thinking_budget: int = self.DEFAULT_THINKING_BUDGET
        self._exploration_count = 0
        self._verification_count = 0
        self._explored_paths: list[str] = []
        self._use_sampling: bool = True
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier."""
        return MethodIdentifier.TEST_TIME_SCALING

    @property
    def name(self) -> str:
        """Get the human-readable method name."""
        return TEST_TIME_SCALING_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description."""
        return TEST_TIME_SCALING_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category."""
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        Prepares the Test-Time Scaling method for execution.
        Resets all state for a fresh reasoning session.
        """
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "analyze"
        self._thinking_budget = self.DEFAULT_THINKING_BUDGET
        self._exploration_count = 0
        self._verification_count = 0
        self._explored_paths = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Test-Time Scaling method.

        Creates the initial analysis phase, assessing problem difficulty.

        Args:
            session: The current reasoning session
            input_text: The problem to solve
            context: Optional additional context (may include thinking_budget)
            execution_context: Optional execution context for LLM sampling

        Returns:
            A ThoughtNode representing the analysis phase

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Test-Time Scaling method must be initialized before execution")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_phase = "analyze"
        self._exploration_count = 0
        self._verification_count = 0
        self._explored_paths = []

        # Allow custom thinking budget
        if context and "thinking_budget" in context:
            self._thinking_budget = min(context["thinking_budget"], 20)
        else:
            self._thinking_budget = self.DEFAULT_THINKING_BUDGET

        # Generate analysis content
        if use_sampling:
            content = await self._sample_analysis(input_text, context)
        else:
            content = self._generate_analysis(input_text, context)

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.TEST_TIME_SCALING,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial confidence is moderate
            quality_score=0.7,
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "test_time_scaling",
                "phase": self._current_phase,
                "thinking_budget": self._thinking_budget,
                "exploration_count": self._exploration_count,
            },
        )

        session.add_thought(thought)
        session.current_method = MethodIdentifier.TEST_TIME_SCALING

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

        Implements the Test-Time Scaling phase progression:
        - After analyze: start exploration
        - During expand: continue exploring or verify
        - After verify: synthesize or continue verifying
        - After synthesize: conclude

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context
            execution_context: Optional execution context for LLM sampling

        Returns:
            A new ThoughtNode continuing the Test-Time Scaling process

        Raises:
            RuntimeError: If the method has not been initialized
        """
        if not self._initialized:
            raise RuntimeError("Test-Time Scaling method must be initialized before continuation")

        # Configure sampling if execution_context provides it
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        if execution_context:
            self._execution_context = execution_context

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "analyze")

        if prev_phase == "analyze":
            # Start exploration phase
            self._current_phase = "expand"
            self._exploration_count = 1
            thought_type = ThoughtType.REASONING
            if use_sampling:
                content = await self._sample_exploration(self._exploration_count, guidance, context)
            else:
                content = self._generate_exploration(self._exploration_count, guidance, context)
            confidence = 0.65
            quality_score = 0.7

        elif prev_phase == "expand":
            self._exploration_count += 1
            if self._exploration_count < self._thinking_budget:
                # Check if we should verify now
                should_verify = self._exploration_count >= (self._thinking_budget // 2)
                if should_verify:
                    self._current_phase = "verify"
                    self._verification_count = 1
                    thought_type = ThoughtType.VERIFICATION
                    if use_sampling:
                        content = await self._sample_verification(
                            self._verification_count, guidance, context
                        )
                    else:
                        content = self._generate_verification(
                            self._verification_count, guidance, context
                        )
                    confidence = 0.75
                    quality_score = 0.8
                else:
                    # Continue exploration
                    thought_type = ThoughtType.REASONING
                    if use_sampling:
                        content = await self._sample_exploration(
                            self._exploration_count, guidance, context
                        )
                    else:
                        content = self._generate_exploration(
                            self._exploration_count, guidance, context
                        )
                    confidence = 0.65 + (self._exploration_count * 0.02)
                    quality_score = 0.7 + (self._exploration_count * 0.02)
            else:
                # Max exploration, move to verify
                self._current_phase = "verify"
                self._verification_count = 1
                thought_type = ThoughtType.VERIFICATION
                if use_sampling:
                    content = await self._sample_verification(
                        self._verification_count, guidance, context
                    )
                else:
                    content = self._generate_verification(
                        self._verification_count, guidance, context
                    )
                confidence = 0.75
                quality_score = 0.8

        elif prev_phase == "verify":
            self._verification_count += 1
            if self._verification_count <= self.MAX_VERIFICATIONS:
                # Continue verification or synthesize
                needs_more = self._verification_count < 2
                if needs_more:
                    thought_type = ThoughtType.VERIFICATION
                    if use_sampling:
                        content = await self._sample_verification(
                            self._verification_count, guidance, context
                        )
                    else:
                        content = self._generate_verification(
                            self._verification_count, guidance, context
                        )
                    confidence = 0.8
                    quality_score = 0.85
                else:
                    # Move to synthesis
                    self._current_phase = "synthesize"
                    thought_type = ThoughtType.SYNTHESIS
                    if use_sampling:
                        content = await self._sample_synthesis(guidance, context)
                    else:
                        content = self._generate_synthesis(guidance, context)
                    confidence = 0.85
                    quality_score = 0.9
            else:
                # Max verifications, synthesize
                self._current_phase = "synthesize"
                thought_type = ThoughtType.SYNTHESIS
                if use_sampling:
                    content = await self._sample_synthesis(guidance, context)
                else:
                    content = self._generate_synthesis(guidance, context)
                confidence = 0.85
                quality_score = 0.9

        elif prev_phase == "synthesize":
            # Conclude
            self._current_phase = "conclude"
            thought_type = ThoughtType.CONCLUSION
            if use_sampling:
                content = await self._sample_conclusion(guidance, context)
            else:
                content = self._generate_conclusion(guidance, context)
            confidence = 0.95
            quality_score = 0.95

        else:
            # Fallback
            self._current_phase = "synthesize"
            thought_type = ThoughtType.SYNTHESIS
            if use_sampling:
                content = await self._sample_synthesis(guidance, context)
            else:
                content = self._generate_synthesis(guidance, context)
            confidence = 0.8
            quality_score = 0.85

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.TEST_TIME_SCALING,
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
                "reasoning_type": "test_time_scaling",
                "thinking_budget": self._thinking_budget,
                "exploration_count": self._exploration_count,
                "verification_count": self._verification_count,
                "previous_phase": prev_phase,
            },
        )

        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute."""
        return self._initialized

    def _generate_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the analysis phase content."""
        return (
            f"Step {self._step_counter}: Problem Analysis (Test-Time Scaling)\n\n"
            f"Problem: {input_text}\n\n"
            f"Difficulty Assessment:\n"
            f"  [Analyzing problem complexity...]\n"
            f"  - Estimated complexity: [high/medium/low]\n"
            f"  - Key challenges: [identified challenges]\n"
            f"  - Required knowledge: [domain requirements]\n\n"
            f"Thinking Budget Allocation:\n"
            f"  - Total budget: {self._thinking_budget} exploration steps\n"
            f"  - Exploration phase: ~{self._thinking_budget // 2} steps\n"
            f"  - Verification phase: ~{self.MAX_VERIFICATIONS} rounds\n\n"
            f"Strategy:\n"
            f"  1. Explore multiple reasoning paths\n"
            f"  2. Self-verify and correct errors\n"
            f"  3. Synthesize best approach\n\n"
            f"Beginning extended thinking process..."
        )

    def _generate_exploration(
        self,
        exploration_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an exploration step."""
        path_id = f"Path-{exploration_num}"
        self._explored_paths.append(path_id)

        return (
            f"Step {self._step_counter}: Extended Thinking #{exploration_num}\n\n"
            f"<|thinking|>\n"
            f"Exploration {path_id}:\n"
            f"  Current approach: [reasoning approach for this path]\n\n"
            f"  Working through:\n"
            f"  - [Step A: initial consideration]\n"
            f"  - [Step B: developing the idea]\n"
            f"  - [Step C: testing implications]\n\n"
            f"  Wait, let me check this...\n"
            f"  [Self-correction if needed]\n\n"
            f"  Intermediate conclusion: [what we've learned]\n"
            f"</|thinking|>\n\n"
            f"Progress: {exploration_num}/{self._thinking_budget} explorations\n"
            f"Active paths: {len(self._explored_paths)}"
        )

    def _generate_verification(
        self,
        verification_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a verification step."""
        return (
            f"Step {self._step_counter}: Verification Round #{verification_num}\n\n"
            f"Checking explored reasoning paths...\n\n"
            f"Verification Checklist:\n"
            f"  [ ] Logical consistency across paths\n"
            f"  [ ] No computational errors\n"
            f"  [ ] Assumptions are valid\n"
            f"  [ ] Edge cases considered\n\n"
            f"Path Analysis:\n"
            + "\n".join(
                f"  - {path}: [valid/needs correction/rejected]" for path in self._explored_paths
            )
            + f"\n\n"
            f"Corrections Applied:\n"
            f"  [Any self-corrections made during verification]\n\n"
            f"Confidence after verification: {0.7 + (verification_num * 0.1):.1%}"
        )

    def _generate_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the synthesis phase."""
        return (
            f"Step {self._step_counter}: Synthesis of Extended Thinking\n\n"
            f"Combining insights from {len(self._explored_paths)} explored paths...\n\n"
            f"Key Insights:\n"
            f"  1. [Most important discovery from exploration]\n"
            f"  2. [Second key insight]\n"
            f"  3. [Third key insight]\n\n"
            f"Best Reasoning Path:\n"
            f"  Selected: [best path from exploration]\n"
            f"  Reason: [why this path is most promising]\n\n"
            f"Synthesized Solution:\n"
            f"  [Combining the best elements from all paths]\n"
            f"  [Final reasoning chain]\n\n"
            f"Ready to formulate final answer."
        )

    def _generate_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion."""
        return (
            f"Step {self._step_counter}: Final Answer\n\n"
            f"Test-Time Scaling Analysis Complete:\n\n"
            f"Compute Summary:\n"
            f"  - Thinking steps: {self._exploration_count}\n"
            f"  - Verification rounds: {self._verification_count}\n"
            f"  - Paths explored: {len(self._explored_paths)}\n"
            f"  - Total reasoning depth: {self._step_counter} steps\n\n"
            f"Final Answer: [Answer derived through extended thinking]\n\n"
            f"Confidence: Very High (95%+)\n"
            f"Reason: Extended thinking with multiple exploration paths,\n"
            f"self-verification, and error correction before synthesis."
        )

    async def _sample_analysis(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the analysis phase using LLM sampling.

        Args:
            input_text: The problem to analyze
            context: Optional additional context

        Returns:
            The content for the analysis thought
        """
        system_prompt = """You are a Test-Time Scaling reasoning assistant.
Use o1/o3-style extended thinking to analyze the problem and plan your compute budget:
1. Assess problem difficulty and complexity
2. Identify key challenges and required knowledge
3. Allocate thinking budget for exploration and verification
4. Outline your extended thinking strategy

Be thorough and strategic in your analysis."""

        user_prompt = f"""Problem to solve using Test-Time Scaling:

{input_text}

Thinking Budget: {self._thinking_budget} exploration steps
Verification Rounds: up to {self.MAX_VERIFICATIONS}

Perform a detailed analysis of this problem and plan your extended thinking approach."""

        if context:
            user_prompt += f"\n\nAdditional Context: {context}"

        return await self._sample_with_fallback(
            user_prompt,
            lambda: self._generate_analysis(input_text, context),
            system_prompt=system_prompt,
            temperature=0.4,  # Moderate temperature for analysis
            max_tokens=800,
        )

    async def _sample_exploration(
        self,
        exploration_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate an exploration step using LLM sampling.

        Args:
            exploration_num: The current exploration number
            guidance: Optional guidance for this exploration
            context: Optional additional context

        Returns:
            The content for the exploration thought
        """
        system_prompt = """You are performing extended thinking during Test-Time Scaling.
Generate a detailed exploration of a reasoning path:
1. Consider one specific approach to the problem
2. Work through the reasoning step-by-step
3. Self-correct if you notice errors or issues
4. Document your intermediate conclusions

Think deeply and carefully. You have time to explore."""

        user_prompt = f"""Exploration #{exploration_num} of {self._thinking_budget}

Previously explored paths: {len(self._explored_paths)}
Current phase: Extended Thinking

Continue your deep exploration of the problem."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        if context:
            user_prompt += f"\n\nContext: {context}"

        return await self._sample_with_fallback(
            user_prompt,
            lambda: self._generate_exploration(exploration_num, guidance, context),
            system_prompt=system_prompt,
            temperature=0.6,  # Higher temperature for exploration
            max_tokens=1000,
        )

    async def _sample_verification(
        self,
        verification_num: int,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate a verification step using LLM sampling.

        Args:
            verification_num: The current verification number
            guidance: Optional guidance for verification
            context: Optional additional context

        Returns:
            The content for the verification thought
        """
        system_prompt = """You are performing verification during Test-Time Scaling.
Carefully check your reasoning:
1. Review logical consistency across all explored paths
2. Verify calculations and steps for correctness
3. Check assumptions and edge cases
4. Apply corrections where needed

Be rigorous and thorough in your verification."""

        user_prompt = f"""Verification Round #{verification_num}

Paths explored: {len(self._explored_paths)}
Total explorations: {self._exploration_count}

Check your reasoning paths for correctness, consistency, and completeness."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        if context:
            user_prompt += f"\n\nContext: {context}"

        return await self._sample_with_fallback(
            user_prompt,
            lambda: self._generate_verification(verification_num, guidance, context),
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for verification
            max_tokens=900,
        )

    async def _sample_synthesis(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the synthesis phase using LLM sampling.

        Args:
            guidance: Optional guidance for synthesis
            context: Optional additional context

        Returns:
            The content for the synthesis thought
        """
        system_prompt = """You are synthesizing your extended thinking during Test-Time Scaling.
Combine insights from all exploration:
1. Identify the most important discoveries
2. Select the best reasoning path
3. Synthesize a coherent solution
4. Integrate verified insights

Create a unified, well-reasoned solution."""

        user_prompt = f"""Synthesis Phase

Explorations completed: {self._exploration_count}
Verifications performed: {self._verification_count}
Paths explored: {len(self._explored_paths)}

Synthesize your extended thinking into a coherent solution."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        if context:
            user_prompt += f"\n\nContext: {context}"

        return await self._sample_with_fallback(
            user_prompt,
            lambda: self._generate_synthesis(guidance, context),
            system_prompt=system_prompt,
            temperature=0.4,  # Moderate temperature for synthesis
            max_tokens=1000,
        )

    async def _sample_conclusion(
        self,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the final conclusion using LLM sampling.

        Args:
            guidance: Optional guidance for conclusion
            context: Optional additional context

        Returns:
            The content for the conclusion thought
        """
        system_prompt = """You are concluding your Test-Time Scaling reasoning.
Provide the final answer:
1. State your solution clearly
2. Summarize the compute used
3. Express your confidence level
4. Explain why extended thinking led to this answer

Be clear, confident, and complete."""

        user_prompt = f"""Final Answer Phase

Total reasoning steps: {self._step_counter}
Explorations: {self._exploration_count}
Verifications: {self._verification_count}
Paths explored: {len(self._explored_paths)}

Provide your final, well-reasoned answer."""

        if guidance:
            user_prompt += f"\n\nGuidance: {guidance}"

        if context:
            user_prompt += f"\n\nContext: {context}"

        return await self._sample_with_fallback(
            user_prompt,
            lambda: self._generate_conclusion(guidance, context),
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for final answer
            max_tokens=800,
        )


# Export
__all__ = ["TestTimeScaling", "TEST_TIME_SCALING_METADATA"]
