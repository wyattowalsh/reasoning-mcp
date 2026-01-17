"""Diagram of Thought reasoning method.

This module implements DAG-based reasoning with three distinct roles:
PROPOSER (generates reasoning propositions), CRITIC (evaluates and challenges),
and SUMMARIZER (synthesizes valid propositions). The method navigates a directed
acyclic graph of propositions, building reliable conclusions through iterative
proposal-critique-synthesis cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

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


# Metadata for Diagram of Thought method
DIAGRAM_OF_THOUGHT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.DIAGRAM_OF_THOUGHT,
    name="Diagram of Thought",
    description="DAG-based reasoning with proposer-critic-summarizer roles. "
    "Generates propositions, evaluates them critically, and synthesizes "
    "valid propositions into coherent conclusions through directed acyclic graph navigation.",
    category=MethodCategory.ADVANCED,
    tags=frozenset(
        {
            "dag",
            "graph-based",
            "multi-role",
            "proposer-critic",
            "synthesis",
            "iterative",
            "validation",
            "structured",
        }
    ),
    complexity=7,  # Advanced - requires DAG management and role coordination
    supports_branching=True,  # DAG supports multiple branches
    supports_revision=True,  # Critic can revise propositions
    requires_context=False,  # No special context needed
    min_thoughts=4,  # At least: initial + propose + critique + summarize
    max_thoughts=30,  # Multiple proposal-critique-synthesis cycles
    avg_tokens_per_thought=450,  # Moderate - structured but detailed
    best_for=(
        "complex problem solving",
        "multi-faceted analysis",
        "argument construction",
        "hypothesis evaluation",
        "systematic reasoning",
        "collaborative thinking",
        "structured exploration",
    ),
    not_recommended_for=(
        "simple factual queries",
        "linear reasoning tasks",
        "time-critical decisions",
        "single-perspective problems",
    ),
)

logger = structlog.get_logger(__name__)


class DiagramOfThought(ReasoningMethodBase):
    """Diagram of Thought reasoning method implementation.

    This class implements a DAG-based reasoning pattern with three distinct roles:
    - PROPOSER: Generates new reasoning propositions
    - CRITIC: Evaluates propositions, identifies flaws, challenges assumptions
    - SUMMARIZER: Synthesizes valid propositions into coherent conclusions

    The method navigates through phases:
    1. Propose: Generate reasoning propositions
    2. Critique: Evaluate and validate propositions
    3. Summarize: Synthesize valid propositions into insights

    The DAG structure tracks:
    - All propositions generated
    - Critique relationships between nodes
    - Valid vs. rejected propositions
    - Synthesis paths through the graph

    Key characteristics:
    - Multi-role reasoning (proposer/critic/summarizer)
    - DAG-based proposition tracking
    - Iterative validation cycles
    - Synthesis-driven conclusions
    - High complexity (7)

    Examples:
        Initialize and execute:
        >>> from reasoning_mcp.models import Session
        >>> method = DiagramOfThought()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Should AI development be regulated?"
        ... )
        >>> print(result.content)  # Initial problem setup

        Continue with proposer:
        >>> proposal = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=result,
        ...     guidance="Generate propositions"
        ... )
        >>> print(proposal.type)  # ThoughtType.HYPOTHESIS (proposer phase)

        Continue with critic:
        >>> critique = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=proposal,
        ...     guidance="Evaluate propositions"
        ... )
        >>> print(critique.type)  # ThoughtType.VERIFICATION (critic phase)

        Continue with summarizer:
        >>> summary = await method.continue_reasoning(
        ...     session=session,
        ...     previous_thought=critique,
        ...     guidance="Synthesize valid propositions"
        ... )
        >>> print(summary.type)  # ThoughtType.SYNTHESIS (summarizer phase)
    """

    # Maximum propositions per cycle
    MAX_PROPOSITIONS = 10
    # Maximum critique rounds to prevent infinite loops
    MAX_CRITIQUE_ROUNDS = 2
    # Enable sampling for LLM-driven reasoning
    _use_sampling: bool = True

    def __init__(self) -> None:
        """Initialize the Diagram of Thought method."""
        self._initialized = False
        self._step_counter = 0
        self._current_role: str = "proposer"  # proposer, critic, summarizer
        self._critique_round = 0
        self._proposition_count = 0
        self._valid_propositions: list[str] = []
        self._rejected_propositions: list[str] = []
        self._proposition_dag: dict[str, list[str]] = {}  # node_id -> child_ids
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return MethodIdentifier.DIAGRAM_OF_THOUGHT

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return DIAGRAM_OF_THOUGHT_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return DIAGRAM_OF_THOUGHT_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Diagram of Thought method for execution.
        Resets counters, role state, and DAG structure for a fresh reasoning session.

        Examples:
            >>> method = DiagramOfThought()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
            >>> assert method._current_role == "proposer"
            >>> assert len(method._valid_propositions) == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._current_role = "proposer"
        self._critique_round = 0
        self._proposition_count = 0
        self._valid_propositions = []
        self._rejected_propositions = []
        self._proposition_dag = {}

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the Diagram of Thought method.

        This method creates the initial problem setup that will be explored
        through the proposer-critic-summarizer cycle. It establishes the DAG
        root and prepares for proposition generation.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context (can include max_propositions)

        Returns:
            A ThoughtNode representing the initial problem setup

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = DiagramOfThought()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session=session,
            ...     input_text="How can we solve climate change?"
            ... )
            >>> assert thought.type == ThoughtType.INITIAL
            >>> assert thought.step_number == 1
            >>> assert thought.method_id == MethodIdentifier.DIAGRAM_OF_THOUGHT
            >>> assert "role" in thought.metadata
            >>> assert thought.metadata["role"] == "proposer"
        """
        if not self._initialized:
            raise RuntimeError("Diagram of Thought method must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        # Reset for new execution
        self._step_counter = 1
        self._current_role = "proposer"
        self._critique_round = 0
        self._proposition_count = 0
        self._valid_propositions = []
        self._rejected_propositions = []
        self._proposition_dag = {}

        # Extract max propositions from context if provided
        max_propositions = self.MAX_PROPOSITIONS
        if context and "max_propositions" in context:
            max_propositions = max(1, min(context["max_propositions"], 20))

        # Generate initial problem setup
        content = await self._generate_initial_setup(input_text, context)

        # Generate ID first and add to DAG before creating thought
        thought_id = str(uuid4())
        self._proposition_dag[thought_id] = []

        thought = ThoughtNode(
            id=thought_id,
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.DIAGRAM_OF_THOUGHT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.7,  # Moderate initial confidence
            quality_score=0.6,  # Will improve through cycles
            metadata={
                "input": input_text,
                "context": context or {},
                "reasoning_type": "diagram_of_thought",
                "role": self._current_role,
                "critique_round": self._critique_round,
                "proposition_count": self._proposition_count,
                "max_propositions": max_propositions,
                "valid_propositions": self._valid_propositions.copy(),
                "rejected_propositions": self._rejected_propositions.copy(),
                "dag_nodes": list(self._proposition_dag.keys()),
            },
        )

        # Add to session
        session.add_thought(thought)
        session.current_method = MethodIdentifier.DIAGRAM_OF_THOUGHT

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

        This method implements the proposer-critic-summarizer cycle logic:
        - Proposer: Generate new propositions (HYPOTHESIS)
        - Critic: Evaluate propositions (VERIFICATION)
        - Summarizer: Synthesize valid propositions (SYNTHESIS)

        The method automatically transitions between roles based on the
        current phase and progress through the DAG.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for the next step
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the DAG-based reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = DiagramOfThought()
            >>> await method.initialize()
            >>> initial = await method.execute(session, "Analyze AI ethics")
            >>>
            >>> # Proposer generates propositions
            >>> proposal = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=initial
            ... )
            >>> assert proposal.type == ThoughtType.HYPOTHESIS
            >>> assert proposal.metadata["role"] == "proposer"
            >>>
            >>> # Critic evaluates propositions
            >>> critique = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=proposal
            ... )
            >>> assert critique.type == ThoughtType.VERIFICATION
            >>> assert critique.metadata["role"] == "critic"
            >>>
            >>> # Summarizer synthesizes valid propositions
            >>> summary = await method.continue_reasoning(
            ...     session=session,
            ...     previous_thought=critique
            ... )
            >>> assert summary.type == ThoughtType.SYNTHESIS
            >>> assert summary.metadata["role"] == "summarizer"
        """
        if not self._initialized:
            raise RuntimeError("Diagram of Thought method must be initialized before continuation")

        # Increment step counter
        self._step_counter += 1

        # Get max propositions from previous thought's metadata
        max_propositions = previous_thought.metadata.get("max_propositions", self.MAX_PROPOSITIONS)

        # Determine next role based on current role
        prev_role = previous_thought.metadata.get("role", "proposer")

        if prev_role == "proposer":
            # After proposer: critic evaluates
            self._current_role = "critic"
            thought_type = ThoughtType.VERIFICATION
            content = await self._generate_critique(previous_thought, guidance, context)

            # Critic identifies valid/rejected propositions
            # In real implementation, this would be LLM-driven
            self._update_propositions_from_critique(previous_thought)

            quality_score = 0.7
            confidence = 0.75

        elif prev_role == "critic":
            # After critic: decide whether to summarize or propose again
            if (
                self._critique_round >= self.MAX_CRITIQUE_ROUNDS
                or len(self._valid_propositions) >= 3
            ):
                # Enough valid propositions: summarize
                self._current_role = "summarizer"
                thought_type = ThoughtType.SYNTHESIS
                content = await self._generate_summary(previous_thought, guidance, context)
                quality_score = min(0.8 + (len(self._valid_propositions) * 0.02), 0.95)
                confidence = min(0.8 + (len(self._valid_propositions) * 0.02), 0.95)
            else:
                # Need more propositions: back to proposer
                self._current_role = "proposer"
                self._critique_round += 1
                thought_type = ThoughtType.HYPOTHESIS
                content = await self._generate_propositions(previous_thought, guidance, context)
                quality_score = 0.65
                confidence = 0.7

        elif prev_role == "summarizer":
            # After summarizer: check if we should conclude or continue exploring
            if (
                len(self._valid_propositions) >= 5
                or self._critique_round >= self.MAX_CRITIQUE_ROUNDS
            ):
                # Sufficient synthesis: conclude
                self._current_role = "summarizer"
                thought_type = ThoughtType.CONCLUSION
                content = await self._generate_final_synthesis(previous_thought, guidance, context)
                quality_score = 0.9
                confidence = 0.9
            else:
                # Continue exploring: new propositions
                self._current_role = "proposer"
                thought_type = ThoughtType.HYPOTHESIS
                content = await self._generate_propositions(previous_thought, guidance, context)
                quality_score = 0.65
                confidence = 0.7

        else:
            # Fallback to proposer
            self._current_role = "proposer"
            thought_type = ThoughtType.HYPOTHESIS
            content = await self._generate_propositions(previous_thought, guidance, context)
            quality_score = 0.65
            confidence = 0.7

        # Generate ID first and add to DAG before creating thought
        thought_id = str(uuid4())
        if previous_thought.id in self._proposition_dag:
            self._proposition_dag[previous_thought.id].append(thought_id)
        self._proposition_dag[thought_id] = []

        thought = ThoughtNode(
            id=thought_id,
            type=thought_type,
            method_id=MethodIdentifier.DIAGRAM_OF_THOUGHT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=quality_score,
            metadata={
                "role": self._current_role,
                "critique_round": self._critique_round,
                "proposition_count": self._proposition_count,
                "max_propositions": max_propositions,
                "valid_propositions": self._valid_propositions.copy(),
                "rejected_propositions": self._rejected_propositions.copy(),
                "dag_nodes": list(self._proposition_dag.keys()),
                "guidance": guidance or "",
                "context": context or {},
                "reasoning_type": "diagram_of_thought",
                "previous_role": prev_role,
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Diagram of Thought, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = DiagramOfThought()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    async def _generate_initial_setup(
        self,
        input_text: str,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate the initial problem setup.

        This is a helper method that would typically call an LLM to generate
        the initial problem analysis and setup for DAG exploration.

        Args:
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            The content for the initial setup

        Note:
            In a full implementation, this would use an LLM to generate
            the actual setup. This is a placeholder that provides
            the structure.
        """
        user_prompt = (
            f"Problem to analyze: {input_text}\n\n"
            f"Additional context: {context or 'None'}\n\n"
            f"Provide an initial problem setup that identifies the key aspects to explore "
            f"using a DAG-based approach."
        )

        system_prompt = (
            "You are the PROPOSER role in a Diagram of Thought reasoning system. "
            "Your task is to set up the problem for DAG-based exploration with "
            "three roles:\n"
            "1. PROPOSER: Generate reasoning propositions\n"
            "2. CRITIC: Evaluate and validate propositions\n"
            "3. SUMMARIZER: Synthesize valid propositions into insights\n\n"
            "Analyze the problem and prepare it for systematic "
            "proposition-based reasoning."
        )

        def fallback() -> str:
            return (
                "I will analyze this problem using a DAG-based reasoning approach with "
                "three roles:\n"
                "1. PROPOSER: Generate reasoning propositions\n"
                "2. CRITIC: Evaluate and validate propositions\n"
                "3. SUMMARIZER: Synthesize valid propositions into insights\n\n"
                "Starting DAG exploration. The proposer will now generate initial "
                "propositions for evaluation."
            )

        result_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return (
            f"Step {self._step_counter}: Initial Problem Setup (Diagram of Thought)\n\n"
            f"Problem: {input_text}\n\n"
            f"Role: PROPOSER\n\n"
            f"{result_text}\n\n"
            f"DAG Status:\n"
            f"- Nodes: 1 (root)\n"
            f"- Valid Propositions: 0\n"
            f"- Critique Round: 0"
        )

    async def _generate_propositions(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate new reasoning propositions.

        This is a helper method that would typically call an LLM in the PROPOSER
        role to generate new propositions for the DAG.

        Args:
            previous_thought: The previous thought in the chain
            guidance: Optional guidance for proposition generation
            context: Optional additional context

        Returns:
            The content for the propositions

        Note:
            In a full implementation, this would use an LLM to generate
            actual propositions. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        # Simulate proposition generation
        num_new = min(3, self.MAX_PROPOSITIONS - self._proposition_count)
        self._proposition_count += num_new

        user_prompt = (
            f"Previous thought (Step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Current DAG state:\n"
            f"- Total Propositions: {self._proposition_count - num_new}\n"
            f"- Valid: {len(self._valid_propositions)}\n"
            f"- Rejected: {len(self._rejected_propositions)}\n"
            f"- Critique Round: {self._critique_round}\n\n"
            f"Generate {num_new} new propositions to explore the problem "
            f"from different angles."
            f"{guidance_text}"
        )

        system_prompt = (
            "You are the PROPOSER role in a Diagram of Thought reasoning system. "
            "Your task is to generate clear, testable propositions that advance "
            "understanding. "
            "Each proposition should explore a specific aspect or angle of the "
            "problem. "
            "Format your response as numbered propositions (P1, P2, P3, etc.)."
        )

        def fallback() -> str:
            return (
                f"Based on the current DAG state, I will generate {num_new} new propositions "
                f"for evaluation.\n\n"
                f"Propositions:\n"
                f"[LLM would generate specific propositions here]\n"
                f"P{self._proposition_count - num_new + 1}: [Proposition about aspect 1]\n"
                f"P{self._proposition_count - num_new + 2}: [Proposition about aspect 2]\n"
                f"P{self._proposition_count - num_new + 3}: [Proposition about aspect 3]"
            )

        result_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return (
            f"Step {self._step_counter}: Proposition Generation "
            f"(Critique Round {self._critique_round})\n\n"
            f"Role: PROPOSER\n\n"
            f"{result_text}\n\n"
            f"DAG Status:\n"
            f"- Total Propositions: {self._proposition_count}\n"
            f"- Valid: {len(self._valid_propositions)}\n"
            f"- Rejected: {len(self._rejected_propositions)}\n"
            f"- Critique Round: {self._critique_round}{guidance_text}"
        )

    async def _generate_critique(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate critique of propositions.

        This is a helper method that would typically call an LLM in the CRITIC
        role to evaluate propositions and identify valid vs. rejected ones.

        Args:
            previous_thought: The propositions to critique
            guidance: Optional guidance for the critique
            context: Optional additional context

        Returns:
            The content for the critique

        Note:
            In a full implementation, this would use an LLM to generate
            the actual critique. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = (
            f"Propositions to critique (Step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Current DAG state:\n"
            f"- Valid: {len(self._valid_propositions)}\n"
            f"- Rejected: {len(self._rejected_propositions)}\n"
            f"- Critique Round: {self._critique_round}\n\n"
            f"Evaluate each proposition rigorously. Identify which are valid and "
            f"which have flaws. "
            f"Challenge assumptions and suggest improvements."
            f"{guidance_text}"
        )

        system_prompt = (
            "You are the CRITIC role in a Diagram of Thought reasoning system. "
            "Your task is to rigorously evaluate propositions, identifying:\n"
            "1. Valid propositions (sound reasoning, supported claims)\n"
            "2. Rejected propositions (flaws, unsupported claims, logical errors)\n"
            "3. Challenged assumptions that need questioning\n"
            "4. Recommendations for improvement\n\n"
            "Be thorough and constructively critical."
        )

        def fallback() -> str:
            return (
                f"Evaluating propositions from Step {previous_thought.step_number}...\n\n"
                f"Analysis:\n"
                f"[LLM would provide detailed critique here]\n\n"
                f"Valid Propositions:\n"
                f"[List propositions that passed critique]\n\n"
                f"Rejected Propositions:\n"
                f"[List propositions with flaws, with reasons]\n\n"
                f"Challenges:\n"
                f"[Identify assumptions to question]\n\n"
                f"Recommendations:\n"
                f"[Suggest improvements or new directions]"
            )

        result_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return (
            f"Step {self._step_counter}: Proposition Critique "
            f"(Critique Round {self._critique_round})\n\n"
            f"Role: CRITIC\n\n"
            f"{result_text}\n\n"
            f"DAG Status:\n"
            f"- Valid: {len(self._valid_propositions)}\n"
            f"- Rejected: {len(self._rejected_propositions)}\n"
            f"- Critique Round: {self._critique_round}{guidance_text}"
        )

    async def _generate_summary(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate synthesis of valid propositions.

        This is a helper method that would typically call an LLM in the SUMMARIZER
        role to synthesize valid propositions into coherent insights.

        Args:
            previous_thought: The critique to synthesize from
            guidance: Optional guidance for the synthesis
            context: Optional additional context

        Returns:
            The content for the synthesis

        Note:
            In a full implementation, this would use an LLM to generate
            the actual synthesis. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = (
            f"Critique to synthesize from (Step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Valid propositions to integrate: {len(self._valid_propositions)}\n"
            f"DAG state:\n"
            f"- DAG Nodes: {len(self._proposition_dag)}\n"
            f"- Critique Rounds: {self._critique_round}\n\n"
            f"Synthesize the valid propositions into a coherent understanding, "
            f"showing connections "
            f"and weaving them into a unified narrative."
            f"{guidance_text}"
        )

        system_prompt = (
            "You are the SUMMARIZER role in a Diagram of Thought reasoning system. "
            "Your task is to synthesize valid propositions into coherent insights:\n"
            "1. Extract key insights from valid propositions\n"
            "2. Show relationships and connections in the DAG\n"
            "3. Weave propositions into a unified understanding\n"
            "4. Build a coherent narrative\n\n"
            "Create an integrated synthesis that brings together validated reasoning."
        )

        def fallback() -> str:
            return (
                f"Synthesizing {len(self._valid_propositions)} valid propositions "
                f"from the DAG...\n\n"
                f"Synthesis:\n"
                f"[LLM would provide integrated synthesis here]\n\n"
                f"Key Insights:\n"
                f"[Extract main insights from valid propositions]\n\n"
                f"Connections:\n"
                f"[Show relationships between propositions in DAG]\n\n"
                f"Coherent Narrative:\n"
                f"[Weave propositions into unified understanding]"
            )

        result_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return (
            f"Step {self._step_counter}: Proposition Synthesis "
            f"(Critique Round {self._critique_round})\n\n"
            f"Role: SUMMARIZER\n\n"
            f"{result_text}\n\n"
            f"DAG Status:\n"
            f"- Valid Propositions: {len(self._valid_propositions)}\n"
            f"- DAG Nodes: {len(self._proposition_dag)}\n"
            f"- Critique Rounds: {self._critique_round}{guidance_text}"
        )

    async def _generate_final_synthesis(
        self,
        previous_thought: ThoughtNode,
        guidance: str | None,
        context: dict[str, Any] | None,
    ) -> str:
        """Generate final synthesis and conclusion.

        This is a helper method that would typically call an LLM in the SUMMARIZER
        role to create the final conclusion from all valid propositions.

        Args:
            previous_thought: The previous synthesis
            guidance: Optional guidance for the final synthesis
            context: Optional additional context

        Returns:
            The content for the final synthesis

        Note:
            In a full implementation, this would use an LLM to generate
            the actual final synthesis. This is a placeholder that provides
            the structure.
        """
        guidance_text = f"\n\nGuidance: {guidance}" if guidance else ""

        user_prompt = (
            f"Previous synthesis (Step {previous_thought.step_number}):\n"
            f"{previous_thought.content}\n\n"
            f"Total valid propositions: {len(self._valid_propositions)}\n"
            f"Complete DAG exploration:\n"
            f"- Total Nodes: {len(self._proposition_dag)}\n"
            f"- Critique Rounds: {self._critique_round + 1}\n\n"
            f"Provide a comprehensive final synthesis integrating all valid "
            f"propositions. "
            f"Include the definitive conclusion, reasoning path through the DAG, "
            f"and confidence assessment."
            f"{guidance_text}"
        )

        system_prompt = (
            "You are the SUMMARIZER role in a Diagram of Thought reasoning system. "
            "Your task is to create the final comprehensive synthesis:\n"
            "1. Integrate all valid propositions into complete analysis\n"
            "2. Provide a definitive conclusion based on DAG exploration\n"
            "3. Trace the key reasoning path through the DAG\n"
            "4. Assess confidence based on proposition validation\n\n"
            "This is the final output - make it thorough and conclusive."
        )

        def fallback() -> str:
            return (
                f"Final synthesis integrating all {len(self._valid_propositions)} "
                f"valid propositions from {self._critique_round + 1} critique round(s).\n\n"
                f"Complete Analysis:\n"
                f"[LLM would provide comprehensive synthesis here]\n\n"
                f"Final Answer:\n"
                f"[Definitive conclusion based on DAG exploration]\n\n"
                f"Reasoning Path:\n"
                f"[Trace key propositions through DAG that led to conclusion]\n\n"
                f"Confidence Assessment:\n"
                f"[Evaluate confidence based on proposition validation]"
            )

        result_text = await self._sample_with_fallback(
            user_prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
            temperature=DEFAULT_SAMPLING_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

        return (
            f"Step {self._step_counter}: Final Synthesis (Conclusion)\n\n"
            f"Role: SUMMARIZER\n\n"
            f"{result_text}\n\n"
            f"Final DAG Status:\n"
            f"- Total Nodes: {len(self._proposition_dag)}\n"
            f"- Valid Propositions: {len(self._valid_propositions)}\n"
            f"- Critique Rounds: {self._critique_round + 1}{guidance_text}"
        )

    def _update_propositions_from_critique(
        self,
        proposer_thought: ThoughtNode,
    ) -> None:
        """Update valid and rejected propositions based on critique.

        This helper simulates the critic's evaluation of propositions.
        In a full implementation, this would parse LLM output to extract
        which propositions were validated or rejected.

        Args:
            proposer_thought: The thought containing propositions to evaluate

        Note:
            This is a simulation. Real implementation would use LLM output
            to determine valid/rejected propositions.
        """
        # Simulate: ~70% of propositions are valid
        # In real implementation, this would be LLM-driven
        recent_props = (
            self._proposition_count
            - len(self._valid_propositions)
            - len(self._rejected_propositions)
        )

        for i in range(recent_props):
            prop_id = f"P{len(self._valid_propositions) + len(self._rejected_propositions) + 1}"
            # Simulate validation (70% success rate)
            if (i % 3) != 2:  # Simple deterministic simulation
                self._valid_propositions.append(prop_id)
            else:
                self._rejected_propositions.append(prop_id)
