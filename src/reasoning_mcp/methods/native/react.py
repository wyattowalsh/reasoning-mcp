"""ReAct (Reasoning and Acting) method implementation.

ReAct interleaves reasoning steps with actions and observations, following a
Reason → Act → Observe cycle until a conclusion is reached. This method is
particularly effective for problems requiring external tool interactions or
multi-step exploration.

References:
    Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023).
    ReAct: Synergizing Reasoning and Acting in Language Models.
    arXiv preprint arXiv:2210.03629.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for the ReAct method
REACT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.REACT,
    name="ReAct (Reasoning and Acting)",
    description=(
        "Interleaves reasoning, action, and observation in a cyclic pattern. "
        "Each cycle begins with reasoning about the current state, proposes an "
        "action to take, and then observes the results before continuing. "
        "Effective for multi-step problems requiring external interactions."
    ),
    category=MethodCategory.CORE,
    tags=frozenset(
        {
            "iterative",
            "action-oriented",
            "tool-use",
            "observation",
            "multi-step",
            "interactive",
        }
    ),
    complexity=5,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=3,  # At least one Reason → Act → Observe cycle
    max_thoughts=0,  # Unlimited - continues until conclusion
    avg_tokens_per_thought=400,
    best_for=(
        "Problems requiring external tool interactions",
        "Multi-step exploration and information gathering",
        "Situations needing iterative refinement based on feedback",
        "Tasks with observable state changes",
        "Information retrieval and synthesis problems",
        "Debugging and diagnostic reasoning",
    ),
    not_recommended_for=(
        "Simple one-shot reasoning tasks",
        "Pure mathematical or logical proofs",
        "Problems with no external actions possible",
        "Real-time critical decision making",
    ),
)


class ReActMethod(ReasoningMethodBase):
    """ReAct (Reasoning and Acting) reasoning method.

    ReAct follows a structured cycle of:
    1. Reasoning - Analyze the current state and determine what to do next
    2. Action - Specify an action to take or information to gather
    3. Observation - Record the results or observations from the action

    This cycle continues until the reasoning determines a conclusion has been reached.
    The method is particularly powerful for problems that benefit from iterative
    exploration and learning from intermediate results.

    Examples:
        Basic usage:
        >>> method = ReActMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="How can I find the population of Tokyo?"
        ... )
        >>> assert result.type == ThoughtType.CONCLUSION

        With action context:
        >>> context = {
        ...     "available_tools": ["search", "calculate", "lookup"],
        ...     "max_cycles": 5
        ... }
        >>> result = await method.execute(
        ...     session=session,
        ...     input_text="Calculate the area of a circle with radius 5",
        ...     context=context
        ... )
    """

    def __init__(self) -> None:
        """Initialize the ReAct method."""
        self._is_initialized = False
        self._max_cycles = 10  # Default maximum reasoning cycles
        self._use_sampling = False  # Set per-execution based on context
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        """Unique identifier for this method."""
        return str(MethodIdentifier.REACT)

    @property
    def name(self) -> str:
        """Human-readable name for the method."""
        return REACT_METADATA.name

    @property
    def description(self) -> str:
        """Brief description of what the method does."""
        return REACT_METADATA.description

    @property
    def category(self) -> str:
        """Category this method belongs to."""
        return str(REACT_METADATA.category)

    async def initialize(self) -> None:
        """Initialize the method (load resources, etc.)."""
        # No special initialization needed for ReAct
        self._is_initialized = True

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute the ReAct reasoning method.

        This method generates a series of Reason → Act → Observe cycles until
        a conclusion is reached. Each cycle:
        1. Reasons about the current state and what action to take
        2. Proposes an action (simulated for now, or via LLM sampling)
        3. Records observations from the action result
        4. Evaluates if conclusion is reached

        When execution_context with sampling capability is provided, this method
        uses LLM sampling to generate intelligent reasoning content. Otherwise,
        it falls back to placeholder content generation.

        Args:
            session: The current reasoning session
            input_text: The input problem or question to reason about
            context: Optional context including:
                - max_cycles: Maximum number of R→A→O cycles (default: 10)
                - available_tools: List of tools that could be used (simulated)
                - initial_observations: Initial facts or observations to start with
            execution_context: Optional ExecutionContext for LLM sampling (v2.14+)

        Returns:
            A ThoughtNode representing the final conclusion

        Raises:
            ValueError: If session is not active
        """
        # Check if LLM sampling is available
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context
        if not session.is_active:
            raise ValueError("Session must be active to execute reasoning")

        # Extract context parameters
        ctx = context or {}
        max_cycles = ctx.get("max_cycles", self._max_cycles)
        available_tools = ctx.get("available_tools", ["search", "lookup", "calculate"])
        initial_obs = ctx.get("initial_observations", [])

        # Create initial thought analyzing the problem
        initial_thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.REACT,
            content=f"Problem Analysis: {input_text}\n\nI will use the ReAct method to solve this problem through iterative reasoning, action, and observation cycles.",
            confidence=0.7,
            depth=0,
            step_number=1,
            metadata={
                "cycle": 0,
                "phase": "initial",
                "available_tools": available_tools,
            },
        )
        session.add_thought(initial_thought)

        # Track the reasoning chain
        current_parent_id = initial_thought.id
        step_number = 2
        cycle = 1
        conclusion_reached = False

        # Add initial observations if provided
        if initial_obs:
            obs_content = "Initial Observations:\n" + "\n".join(f"- {obs}" for obs in initial_obs)
            obs_thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.OBSERVATION,
                method_id=MethodIdentifier.REACT,
                content=obs_content,
                parent_id=current_parent_id,
                confidence=0.9,
                depth=1,
                step_number=step_number,
                metadata={"cycle": 0, "phase": "initial_observation"},
            )
            session.add_thought(obs_thought)
            current_parent_id = obs_thought.id
            step_number += 1

        # Main ReAct loop: Reason → Act → Observe
        while cycle <= max_cycles and not conclusion_reached:
            # Phase 1: Reasoning (async for LLM sampling)
            reasoning_thought = await self._create_reasoning_thought(
                input_text=input_text,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
                session=session,
            )
            session.add_thought(reasoning_thought)
            current_parent_id = reasoning_thought.id
            step_number += 1

            # Check if reasoning determined we have enough information
            if self._should_conclude(reasoning_thought, cycle, max_cycles):
                conclusion_reached = True
                break

            # Phase 2: Action (async for LLM sampling)
            action_thought = await self._create_action_thought(
                reasoning_thought=reasoning_thought,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
                available_tools=available_tools,
            )
            session.add_thought(action_thought)
            current_parent_id = action_thought.id
            step_number += 1

            # Phase 3: Observation (async for LLM sampling)
            observation_thought = await self._create_observation_thought(
                action_thought=action_thought,
                cycle=cycle,
                step_number=step_number,
                parent_id=current_parent_id,
            )
            session.add_thought(observation_thought)
            current_parent_id = observation_thought.id
            step_number += 1

            cycle += 1

        # Create final conclusion (async for LLM sampling)
        conclusion = await self._create_conclusion(
            input_text=input_text,
            session=session,
            parent_id=current_parent_id,
            step_number=step_number,
            total_cycles=cycle - 1,
        )
        session.add_thought(conclusion)

        return conclusion

    async def _create_reasoning_thought(
        self,
        input_text: str,
        cycle: int,
        step_number: int,
        parent_id: str,
        session: Session,
    ) -> ThoughtNode:
        """Create a reasoning thought analyzing the current state.

        When sampling is available, uses LLM to generate intelligent reasoning.
        Otherwise falls back to placeholder content.

        Args:
            input_text: The original problem
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID
            session: Current session

        Returns:
            A ThoughtNode with REASONING type
        """
        # Try LLM sampling if available
        if self._use_sampling and self._execution_context:
            content, confidence = await self._sample_reasoning(
                input_text=input_text,
                cycle=cycle,
                session=session,
            )
        else:
            # Fallback: Generate placeholder reasoning based on cycle
            content, confidence = self._generate_placeholder_reasoning(
                input_text=input_text,
                cycle=cycle,
                session=session,
            )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.REASONING,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=confidence,
            depth=session.current_depth + 1,
            step_number=step_number,
            metadata={
                "cycle": cycle,
                "phase": "reasoning",
                "sampled": self._use_sampling,
            },
        )

    async def _sample_reasoning(
        self,
        input_text: str,
        cycle: int,
        session: Session,
    ) -> tuple[str, float]:
        """Sample reasoning content from LLM.

        Args:
            input_text: The original problem
            cycle: Current cycle number
            session: Current session for context

        Returns:
            Tuple of (content, confidence)
        """
        # Build context from previous thoughts
        recent_thoughts = session.get_recent_thoughts(n=5)
        context_str = ""
        if recent_thoughts:
            context_str = "\n\nPrevious reasoning:\n"
            for t in recent_thoughts:
                context_str += f"- [{t.type.value}]: {t.content[:200]}...\n"

        prompt = f"""You are performing ReAct (Reasoning and Acting) reasoning.

Problem: {input_text}

Cycle: {cycle}
{context_str}
Generate a reasoning thought for cycle {cycle}. Analyze the current state of the problem and determine what information is needed or what action should be taken next.

Respond with ONLY the reasoning thought content, no prefixes or labels."""

        content = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: self._generate_placeholder_reasoning(
                input_text, cycle, session
            )[0],
            system_prompt="You are a reasoning assistant using the ReAct method. Generate clear, analytical reasoning thoughts.",
            temperature=0.7,
            max_tokens=500,
        )
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        return f"Thought {cycle}: {content}", 0.85

    def _generate_placeholder_reasoning(
        self,
        input_text: str,
        cycle: int,
        session: Session,
    ) -> tuple[str, float]:
        """Generate placeholder reasoning content when sampling unavailable.

        Args:
            input_text: The original problem
            cycle: Current cycle number
            session: Current session

        Returns:
            Tuple of (content, confidence)
        """
        if cycle == 1:
            content = (
                f"Thought {cycle}: To solve '{input_text}', I need to break this down. "
                "Let me start by identifying what information I need and what actions "
                "would help me gather that information."
            )
            confidence = 0.7
        elif cycle == 2:
            content = (
                f"Thought {cycle}: Based on my previous observation, I should now "
                "refine my approach. I'll consider what additional information "
                "would help me reach a conclusion."
            )
            confidence = 0.75
        elif cycle >= 3:
            # Check if we have enough information
            recent_thoughts = session.get_recent_thoughts(n=5)
            observation_count = sum(1 for t in recent_thoughts if t.type == ThoughtType.OBSERVATION)

            if observation_count >= 2:
                content = (
                    f"Thought {cycle}: I've gathered sufficient information through "
                    f"{observation_count} observations. I can now synthesize these "
                    "findings to form a conclusion."
                )
                confidence = 0.9
            else:
                content = (
                    f"Thought {cycle}: I need to gather more specific information. "
                    "Let me identify the most critical missing piece."
                )
                confidence = 0.75
        else:
            content = f"Thought {cycle}: Continuing analysis..."
            confidence = 0.7

        return content, confidence

    async def _create_action_thought(
        self,
        reasoning_thought: ThoughtNode,
        cycle: int,
        step_number: int,
        parent_id: str,
        available_tools: list[str],
    ) -> ThoughtNode:
        """Create an action thought specifying what action to take.

        When sampling is available, uses LLM to generate intelligent action selection.
        Otherwise falls back to placeholder content.

        Args:
            reasoning_thought: The reasoning thought that led to this action
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID
            available_tools: List of available tools/actions

        Returns:
            A ThoughtNode with ACTION type
        """
        # Try LLM sampling if available
        if self._use_sampling and self._execution_context:
            content, tool = await self._sample_action(
                reasoning_thought=reasoning_thought,
                cycle=cycle,
                available_tools=available_tools,
            )
        else:
            # Fallback: Generate placeholder action based on cycle
            content, tool = self._generate_placeholder_action(
                cycle=cycle,
                available_tools=available_tools,
            )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.ACTION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=0.8,
            depth=reasoning_thought.depth + 1,
            step_number=step_number,
            metadata={
                "cycle": cycle,
                "phase": "action",
                "tool_used": tool,
                "sampled": self._use_sampling,
            },
        )

    async def _sample_action(
        self,
        reasoning_thought: ThoughtNode,
        cycle: int,
        available_tools: list[str],
    ) -> tuple[str, str]:
        """Sample action content from LLM.

        Args:
            reasoning_thought: The reasoning that led to this action
            cycle: Current cycle number
            available_tools: List of available tools

        Returns:
            Tuple of (content, tool_used)
        """
        tools_str = ", ".join(available_tools) if available_tools else "search, lookup, calculate"

        prompt = f"""You are performing ReAct (Reasoning and Acting) reasoning.

Previous reasoning: {reasoning_thought.content}

Cycle: {cycle}
Available tools: {tools_str}

Based on the reasoning above, determine what action to take next. Choose one of the available tools and explain what you will do with it.

Respond in the format:
TOOL: <tool_name>
ACTION: <description of what you will do>"""

        def _parse_action() -> tuple[str, str]:
            """Generate fallback action."""
            content, tool = self._generate_placeholder_action(cycle, available_tools)
            return content

        response = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=_parse_action,
            system_prompt="You are a reasoning assistant using the ReAct method. Select appropriate actions to gather information.",
            temperature=0.7,
            max_tokens=300,
        )
        # Ensure response is a string
        if not isinstance(response, str):
            response = str(response)

        # Parse response
        lines = response.strip().split("\n")
        tool = available_tools[0] if available_tools else "search"
        action_desc = response

        for line in lines:
            if line.upper().startswith("TOOL:"):
                tool = line.split(":", 1)[1].strip()
            elif line.upper().startswith("ACTION:"):
                action_desc = line.split(":", 1)[1].strip()

        content = f"Action {cycle}: {action_desc}"
        return content, tool

    def _generate_placeholder_action(
        self,
        cycle: int,
        available_tools: list[str],
    ) -> tuple[str, str]:
        """Generate placeholder action content when sampling unavailable.

        Args:
            cycle: Current cycle number
            available_tools: List of available tools

        Returns:
            Tuple of (content, tool_used)
        """
        if cycle == 1:
            tool = available_tools[0] if available_tools else "search"
            content = (
                f"Action {cycle}: I will use '{tool}' to gather initial information. "
                "This should help me understand the problem space better."
            )
        elif cycle == 2:
            tool = available_tools[1] if len(available_tools) > 1 else "lookup"
            content = (
                f"Action {cycle}: I will use '{tool}' to get more specific details. "
                "This will help me verify and refine my understanding."
            )
        else:
            tool = available_tools[-1] if available_tools else "calculate"
            content = (
                f"Action {cycle}: I will use '{tool}' to validate and finalize "
                "my findings from the previous observations."
            )

        return content, tool

    async def _create_observation_thought(
        self,
        action_thought: ThoughtNode,
        cycle: int,
        step_number: int,
        parent_id: str,
    ) -> ThoughtNode:
        """Create an observation thought recording action results.

        When sampling is available, uses LLM to generate intelligent observations.
        Otherwise falls back to placeholder content.

        Args:
            action_thought: The action thought that was executed
            cycle: Current cycle number
            step_number: Current step number
            parent_id: Parent thought ID

        Returns:
            A ThoughtNode with OBSERVATION type
        """
        tool_used = action_thought.metadata.get("tool_used", "unknown")

        # Try LLM sampling if available
        if self._use_sampling and self._execution_context:
            content, confidence = await self._sample_observation(
                action_thought=action_thought,
                cycle=cycle,
                tool_used=tool_used,
            )
        else:
            # Fallback: Generate placeholder observation based on cycle
            content, confidence = self._generate_placeholder_observation(
                cycle=cycle,
                tool_used=tool_used,
            )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.OBSERVATION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=confidence,
            depth=action_thought.depth + 1,
            step_number=step_number,
            metadata={
                "cycle": cycle,
                "phase": "observation",
                "sampled": self._use_sampling,
            },
        )

    async def _sample_observation(
        self,
        action_thought: ThoughtNode,
        cycle: int,
        tool_used: str,
    ) -> tuple[str, float]:
        """Sample observation content from LLM.

        Args:
            action_thought: The action that was executed
            cycle: Current cycle number
            tool_used: The tool that was used

        Returns:
            Tuple of (content, confidence)
        """
        prompt = f"""You are performing ReAct (Reasoning and Acting) reasoning.

Action taken: {action_thought.content}
Tool used: {tool_used}
Cycle: {cycle}

Simulate the observation/result from executing this action. What information was gathered? What was learned?

Respond with ONLY the observation content, no prefixes or labels."""

        content = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=lambda: self._generate_placeholder_observation(cycle, tool_used)[0],
            system_prompt="You are a reasoning assistant using the ReAct method. Generate realistic observations from actions.",
            temperature=0.7,
            max_tokens=400,
        )
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        return f"Observation {cycle}: {content}", 0.85

    def _generate_placeholder_observation(
        self,
        cycle: int,
        tool_used: str,
    ) -> tuple[str, float]:
        """Generate placeholder observation content when sampling unavailable.

        Args:
            cycle: Current cycle number
            tool_used: The tool that was used

        Returns:
            Tuple of (content, confidence)
        """
        if cycle == 1:
            content = (
                f"Observation {cycle}: The '{tool_used}' action provided useful "
                "initial context. I now have a better understanding of the problem "
                "domain and can identify what additional information is needed."
            )
            confidence = 0.75
        elif cycle == 2:
            content = (
                f"Observation {cycle}: The '{tool_used}' action revealed important "
                "details. This information builds on my previous findings and helps "
                "narrow down the solution space."
            )
            confidence = 0.85
        else:
            content = (
                f"Observation {cycle}: The '{tool_used}' action confirmed my "
                "hypothesis and provided the final pieces of information needed. "
                "I now have sufficient data to form a conclusion."
            )
            confidence = 0.9

        return content, confidence

    def _should_conclude(
        self,
        reasoning_thought: ThoughtNode,
        cycle: int,
        max_cycles: int,
    ) -> bool:
        """Determine if we should conclude the reasoning process.

        Args:
            reasoning_thought: The latest reasoning thought
            cycle: Current cycle number
            max_cycles: Maximum allowed cycles

        Returns:
            True if we should conclude, False to continue
        """
        # Conclude if we've reached max cycles
        if cycle >= max_cycles:
            return True

        # Conclude if reasoning confidence is high and we're past cycle 2
        if cycle >= 3 and reasoning_thought.confidence >= 0.85:
            return True

        # Check if reasoning content suggests conclusion readiness
        content_lower = reasoning_thought.content.lower()
        conclusion_indicators = [
            "sufficient information",
            "can now synthesize",
            "form a conclusion",
            "ready to conclude",
        ]

        return any(indicator in content_lower for indicator in conclusion_indicators)

    async def _create_conclusion(
        self,
        input_text: str,
        session: Session,
        parent_id: str,
        step_number: int,
        total_cycles: int,
    ) -> ThoughtNode:
        """Create the final conclusion thought.

        When sampling is available, uses LLM to generate intelligent conclusion.
        Otherwise falls back to placeholder content.

        Args:
            input_text: The original problem
            session: Current session
            parent_id: Parent thought ID
            step_number: Current step number
            total_cycles: Total number of cycles completed

        Returns:
            A ThoughtNode with CONCLUSION type
        """
        # Gather insights from the reasoning chain
        recent_thoughts = session.get_recent_thoughts(n=10)
        observations = [t for t in recent_thoughts if t.type == ThoughtType.OBSERVATION]
        actions = [t for t in recent_thoughts if t.type == ThoughtType.ACTION]

        # Try LLM sampling if available
        if self._use_sampling and self._execution_context:
            content = await self._sample_conclusion(
                input_text=input_text,
                session=session,
                total_cycles=total_cycles,
                observations=observations,
                actions=actions,
            )
        else:
            # Fallback: Generate placeholder conclusion
            content = self._generate_placeholder_conclusion(
                input_text=input_text,
                total_cycles=total_cycles,
                num_actions=len(actions),
                num_observations=len(observations),
            )

        # Calculate conclusion confidence based on observations
        if observations:
            avg_obs_confidence = sum(t.confidence for t in observations) / len(observations)
            conclusion_confidence = min(0.95, avg_obs_confidence + 0.05)
        else:
            conclusion_confidence = 0.8

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=MethodIdentifier.REACT,
            content=content,
            parent_id=parent_id,
            confidence=conclusion_confidence,
            quality_score=0.85,
            depth=session.current_depth + 1,
            step_number=step_number,
            metadata={
                "total_cycles": total_cycles,
                "total_actions": len(actions),
                "total_observations": len(observations),
                "phase": "conclusion",
                "sampled": self._use_sampling,
            },
        )

    async def _sample_conclusion(
        self,
        input_text: str,
        session: Session,
        total_cycles: int,
        observations: list[ThoughtNode],
        actions: list[ThoughtNode],
    ) -> str:
        """Sample conclusion content from LLM.

        Args:
            input_text: The original problem
            session: Current session
            total_cycles: Total number of cycles completed
            observations: List of observation thoughts
            actions: List of action thoughts

        Returns:
            Conclusion content string
        """
        # Build context from observations
        obs_summary = "\n".join(f"- {o.content[:200]}..." for o in observations[-5:])

        prompt = f"""You are performing ReAct (Reasoning and Acting) reasoning.

Problem: {input_text}

Completed {total_cycles} reasoning cycles with {len(actions)} actions and {len(observations)} observations.

Recent observations:
{obs_summary}

Based on all the reasoning, actions, and observations, provide a final conclusion that synthesizes the findings and answers the original problem.

Respond with ONLY the conclusion content, starting with "Conclusion:"."""

        def _format_conclusion() -> str:
            """Generate fallback conclusion."""
            return self._generate_placeholder_conclusion(
                input_text=input_text,
                total_cycles=total_cycles,
                num_actions=len(actions),
                num_observations=len(observations),
            )

        content = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=_format_conclusion,
            system_prompt="You are a reasoning assistant using the ReAct method. Provide clear, well-reasoned conclusions.",
            temperature=0.7,
            max_tokens=600,
        )
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        # Ensure it starts with Conclusion:
        if not content.startswith("Conclusion:"):
            content = f"Conclusion: {content}"
        return content

    def _generate_placeholder_conclusion(
        self,
        input_text: str,
        total_cycles: int,
        num_actions: int,
        num_observations: int,
    ) -> str:
        """Generate placeholder conclusion content when sampling unavailable.

        Args:
            input_text: The original problem
            total_cycles: Total number of cycles completed
            num_actions: Number of actions taken
            num_observations: Number of observations made

        Returns:
            Conclusion content string
        """
        return (
            f"Conclusion: After {total_cycles} cycles of reasoning, action, and observation, "
            f"I have reached a conclusion for: '{input_text}'\n\n"
            f"Through {num_actions} actions and {num_observations} observations, "
            "I have gathered and synthesized the necessary information. "
            "The iterative ReAct process allowed me to progressively refine my "
            "understanding and validate findings at each step.\n\n"
            "The solution emerges from the systematic exploration of the problem space, "
            "where each action informed the next reasoning step, creating a robust "
            "chain of evidence-based reasoning."
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
        """Continue reasoning from a previous thought.

        This allows branching or extending the ReAct reasoning chain
        with additional cycles if needed.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context
            execution_context: Optional ExecutionContext for LLM sampling (v2.14+)

        Returns:
            A new ThoughtNode continuing the reasoning
        """
        # Update sampling capability from execution_context
        self._use_sampling = execution_context is not None and execution_context.can_sample
        self._execution_context = execution_context
        # Determine what phase to continue from
        phase = previous_thought.metadata.get("phase", "unknown")
        cycle = previous_thought.metadata.get("cycle", 0)

        if phase == "conclusion":
            # If continuing from conclusion, start a new branch
            return ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.BRANCH,
                method_id=MethodIdentifier.REACT,
                content=f"Branching: {guidance or 'Exploring alternative approach based on previous conclusion.'}",
                parent_id=previous_thought.id,
                branch_id=f"branch_{str(uuid4())[:8]}",
                confidence=0.7,
                depth=previous_thought.depth + 1,
                step_number=previous_thought.step_number + 1,
                metadata={"cycle": cycle + 1, "phase": "branch_reasoning"},
            )

        # Otherwise, continue the cycle
        next_step = previous_thought.step_number + 1

        if phase == "reasoning":
            # After reasoning, create action
            return await self._create_action_thought(
                reasoning_thought=previous_thought,
                cycle=cycle,
                step_number=next_step,
                parent_id=previous_thought.id,
                available_tools=context.get("available_tools", []) if context else [],
            )
        elif phase == "action":
            # After action, create observation
            return await self._create_observation_thought(
                action_thought=previous_thought,
                cycle=cycle,
                step_number=next_step,
                parent_id=previous_thought.id,
            )
        else:
            # Default: create new reasoning thought
            return await self._create_reasoning_thought(
                input_text=guidance or "Continue reasoning",
                cycle=cycle + 1,
                step_number=next_step,
                parent_id=previous_thought.id,
                session=session,
            )
